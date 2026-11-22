"""Smoke test of the benchmark on a tiny run and a synthetic scan (CPU backends in CI)."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from test_train import TINY

from spleenseg.benchmark import main
from spleenseg.export import export
from spleenseg.model import UNet3D


def test_benchmark_times_every_backend_and_records_the_setup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = replace(TINY, amp=False)
    (run_dir / "summary.json").write_text(json.dumps({"config": asdict(cfg), "git_commit": "x"}))
    torch.save(UNet3D(cfg.channels).state_dict(), run_dir / "best.pt")
    export(run_dir, tmp_path / "model.onnx")
    image = tmp_path / "scan.nii.gz"
    scan = np.full((30, 30, 8), 50.0, dtype=np.float32)  # soft tissue, no air to crop
    nib.save(nib.Nifti1Image(scan, np.diag([0.8, 0.8, 5.0, 1.0])), str(image))

    out = tmp_path / "bench.json"
    args = ["--run-dir", str(run_dir), "--onnx", str(tmp_path / "model.onnx")]
    main([*args, "--image", str(image), "--repeats", "1", "--out", str(out), "--power-note", "ci"])

    record = json.loads(out.read_text(encoding="utf-8"))
    assert {"pytorch-cpu", "onnxruntime-cpu"} <= set(record["results"])
    for timings in record["results"].values():
        for kind in ("window", "volume"):
            t = timings[kind]
            assert 0 < t["min_s"] <= t["median_s"] <= t["max_s"] and t["repeats"] == 1
    assert record["power_note"] == "ci" and record["image"] == "scan.nii.gz"
