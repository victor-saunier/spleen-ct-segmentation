"""End-to-end test of the evaluation script on synthetic labels and predictions."""

import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from spleenseg.evaluate import main

AFFINE = np.diag([0.8, 0.8, 5.0, 1.0])


def _save(path: Path, mask: np.ndarray, affine: np.ndarray = AFFINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def _cube(x_start: int) -> np.ndarray:
    mask = np.zeros((12, 12, 8), dtype=np.uint8)
    mask[x_start : x_start + 4, 3:7, 2:6] = 1
    return mask


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Path]:
    dataset, preds = tmp_path / "Task09_Spleen", tmp_path / "preds"
    for case in ("spleen_1", "spleen_2", "spleen_3"):
        _save(dataset / "labelsTr" / f"{case}.nii.gz", _cube(3))
    _save(preds / "spleen_1.nii.gz", _cube(3))  # perfect
    _save(preds / "spleen_2.nii.gz", _cube(5))  # shifted by 2 voxels: half overlap
    _save(preds / "spleen_3.nii.gz", np.zeros((12, 12, 8)))  # the model found nothing
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"val": ["spleen_1", "spleen_2", "spleen_3"], "test": []}))
    return {"dataset": dataset, "preds": preds, "split": split, "out": tmp_path / "out"}


def _run(paths: dict[str, Path]) -> None:
    main(
        [
            "--pred-dir", str(paths["preds"]),
            "--dataset-dir", str(paths["dataset"]),
            "--split-file", str(paths["split"]),
            "--out-dir", str(paths["out"]),
        ]
    )  # fmt: skip


def test_per_case_table(setup: dict[str, Path]) -> None:
    _run(setup)
    with (setup["out"] / "per_case.csv").open(encoding="utf-8") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    assert float(rows["spleen_1"]["dice"]) == 1.0
    assert float(rows["spleen_2"]["dice"]) == 0.5
    assert float(rows["spleen_3"]["dice"]) == 0.0
    assert math.isnan(float(rows["spleen_3"]["hd95_mm"]))


def test_summary_counts_the_undefined_hd95(setup: dict[str, Path]) -> None:
    _run(setup)
    summary = json.loads((setup["out"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "val" and summary["n_cases"] == 3
    assert summary["dice"]["mean"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    assert summary["hd95_mm"]["n_defined"] == 2
    assert summary["hd95_mm"]["n_undefined"] == 1
    assert summary["git_commit"]


def test_a_prediction_on_another_grid_is_refused(setup: dict[str, Path]) -> None:
    _save(setup["preds"] / "spleen_1.nii.gz", _cube(3), affine=np.diag([1.5, 1.5, 2.0, 1.0]))
    with pytest.raises(ValueError, match="shape or position"):
        _run(setup)
