"""The Docker image must predict exactly the mask that predict.py predicts outside it.

Skipped unless SPLEENSEG_DOCKER_IMAGE names a built image: CI builds one and sets it, and on a
laptop `docker build -t spleenseg .` then `SPLEENSEG_DOCKER_IMAGE=spleenseg uv run pytest
tests/test_docker.py`. A synthetic scan and random weights are enough: the test checks the
packaging (files, libraries, user, mounts), not the model's quality. No data leaves the machine.
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from test_predict import _write_case
from test_train import TINY

from spleenseg.model import UNet3D
from spleenseg.predict import main

IMAGE = os.environ.get("SPLEENSEG_DOCKER_IMAGE")


@pytest.mark.skipif(not IMAGE, reason="set SPLEENSEG_DOCKER_IMAGE to a built image")
def test_the_image_predicts_the_same_mask_as_the_code(tmp_path: Path) -> None:
    assert IMAGE is not None
    scan = _write_case(tmp_path / "data", "a")["imagesTr"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = replace(TINY, amp=False)
    (run_dir / "summary.json").write_text(json.dumps({"config": asdict(cfg)}))
    torch.manual_seed(0)
    torch.save(UNet3D(cfg.channels).state_dict(), run_dir / "best.pt")

    local_out = tmp_path / "local"
    main(["--run-dir", str(run_dir), "--image", str(scan), "--out-dir", str(local_out)])

    docker_out = tmp_path / "docker"
    docker_out.mkdir()
    # On Linux a mounted folder belongs to a user number; the image's user (1000) may not be
    # the one running the tests, so run the container as the current user. Docker Desktop on
    # Windows does not need it.
    user: list[str] = []
    if sys.platform != "win32":
        user = ["--user", f"{os.getuid()}:{os.getgid()}"]
    mounts = [f"{scan.parent}:/work/in:ro", f"{run_dir}:/work/model:ro", f"{docker_out}:/work/out"]
    subprocess.run(
        ["docker", "run", "--rm", *user, *(arg for m in mounts for arg in ("-v", m)), IMAGE]
        + ["--run-dir", "/work/model", "--image", f"/work/in/{scan.name}"]
        + ["--out-dir", "/work/out"],
        check=True,
    )

    local = nib.nifti1.load(local_out / scan.name)
    in_docker = nib.nifti1.load(docker_out / scan.name)
    assert np.allclose(local.affine, in_docker.affine)
    assert np.array_equal(np.asanyarray(local.dataobj), np.asanyarray(in_docker.dataobj))
    print("spleen voxels in both masks:", int(np.asanyarray(local.dataobj).sum()))
