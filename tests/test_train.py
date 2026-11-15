"""Smoke test: two training steps on tiny synthetic CTs, on CPU (as in CI)."""

import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from spleenseg.config import TrainConfig
from spleenseg.model import UNet3D
from spleenseg.train import train

TINY = TrainConfig(
    seed=0,
    patch_size=(16, 16, 16),
    samples_per_case=2,
    cases_per_batch=1,
    learning_rate=1e-3,
    weight_decay=0.0,
    max_steps=2,
    val_every_steps=1,
    num_workers=0,
    amp=True,  # must switch itself off on CPU
    model="smoke",
    architecture="unet3d",
    channels=(4, 8),
)


def _write_case(dataset_dir: Path, case: str) -> None:
    image = np.full((40, 40, 10), -1000.0, dtype=np.float32)  # air
    image[8:32, 8:32, :] = 50.0  # body
    label = np.zeros((40, 40, 10), dtype=np.uint8)
    label[12:20, 12:20, 3:7] = 1  # spleen
    image[label == 1] = 100.0
    affine = np.diag([0.75, 0.75, 5.0, 1.0])
    for folder, array in (("imagesTr", image), ("labelsTr", label)):
        (dataset_dir / folder).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array, affine), str(dataset_dir / folder / f"{case}.nii.gz"))


def test_two_training_steps_on_cpu(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Task09_Spleen"
    _write_case(dataset_dir, "spleen_1")
    _write_case(dataset_dir, "spleen_2")
    split = tmp_path / "split.json"
    # spleen_99 has no file at all: if the code ever read the test set, this test would crash.
    split.write_text(
        json.dumps({"train": ["spleen_1"], "val": ["spleen_2"], "test": ["spleen_99"]})
    )

    run_dir = tmp_path / "run"
    summary = train(TINY, dataset_dir, split, run_dir, torch.device("cpu"))

    assert summary["steps"] == 2
    assert summary["amp"] is False
    with (run_dir / "log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [1, 2]
    assert all(math.isfinite(float(r["train_loss"])) for r in rows)
    # The saved checkpoint loads back into a fresh model of the same shape.
    UNet3D(TINY.channels).load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
