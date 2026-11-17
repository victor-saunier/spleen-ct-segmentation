"""Tests for predict.py: masks must come back on the scan's original grid.

The synthetic scan is stored the way real scans can be, so that every preprocessing step really
changes the grid: axes swapped and flipped relative to RAS, 0.8 x 0.8 x 5 mm voxels, odd sizes,
air around the body. On a scan already in RAS at 1.5 x 1.5 x 2 mm with no air, the round trip
would pass while undoing nothing.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from test_train import TINY

from spleenseg.evaluate import evaluate_case
from spleenseg.model import UNet3D
from spleenseg.predict import main, save_on_original_grid
from spleenseg.transforms import preprocessing

# Voxel axis 0 runs along physical y, axis 1 along -x: not RAS, so Orientationd must work.
AFFINE = np.array(
    [[0.0, -0.8, 0.0, 30.0], [0.8, 0.0, 0.0, -20.0], [0.0, 0.0, 5.0, 40.0], [0.0, 0.0, 0.0, 1.0]]
)
SHAPE = (37, 41, 11)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "Task09_Spleen"
VAL_CASES = json.loads((ROOT / "splits" / "split.json").read_text(encoding="utf-8"))["val"]
needs_data = pytest.mark.skipif(
    not (DATASET_DIR / "dataset.json").exists(), reason="dataset not downloaded (as in CI)"
)
# A true validation label sent through preprocessing and back measured Dice 0.978 to 0.987 on
# the 7 validation cases: the cost of the 1.5 x 1.5 x 2 mm grid alone, with no model involved.
ROUND_TRIP_MIN_DICE = 0.97


def _write_case(dataset_dir: Path, case: str) -> dict[str, Path]:
    image = np.full(SHAPE, -1000.0, dtype=np.float32)  # air
    image[6:31, 5:36, :] = 50.0  # body
    label = np.zeros(SHAPE, dtype=np.uint8)
    label[8:18, 22:32, 1:5] = 1  # off-centre on every axis: a mask flipped by mistake misses it
    image[label == 1] = 120.0
    paths = {}
    for folder, array in (("imagesTr", image), ("labelsTr", label)):
        (dataset_dir / folder).mkdir(parents=True, exist_ok=True)
        paths[folder] = dataset_dir / folder / f"{case}.nii.gz"
        nib.save(nib.Nifti1Image(array, AFFINE), str(paths[folder]))
    return paths


def _round_trip(image: Path, label: Path, out: Path) -> None:
    """Send the true label through prediction's grid and back: a "perfect model"."""
    pipeline = preprocessing(keys=("image",))  # exactly what predict.py uses
    data = pipeline({"image": str(image)})
    perfect = preprocessing()({"image": str(image), "label": str(label)})["label"]
    save_on_original_grid(perfect.as_tensor(), data, pipeline, image, out)


def test_true_label_survives_the_round_trip(tmp_path: Path) -> None:
    paths = _write_case(tmp_path / "data", "spleen_1")
    out = tmp_path / "spleen_1.nii.gz"
    _round_trip(paths["imagesTr"], paths["labelsTr"], out)

    saved = nib.Nifti1Image.from_filename(out)
    assert saved.shape == SHAPE
    assert np.allclose(saved.affine, AFFINE)
    assert saved.get_data_dtype() == np.uint8
    assert set(np.unique(np.asanyarray(saved.dataobj)).tolist()) == {0, 1}
    # evaluate.py accepts it (same grid) and the overlap is high, not perfect: the 1.5 mm
    # grid blurs the border.
    assert evaluate_case(out, paths["labelsTr"])["dice"] > 0.8


def test_a_mask_on_the_wrong_grid_is_refused(tmp_path: Path) -> None:
    paths = _write_case(tmp_path / "data", "spleen_1")
    other = tmp_path / "other.nii.gz"  # a scan with another shape, e.g. the wrong file
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), AFFINE), str(other))
    pipeline = preprocessing(keys=("image",))
    data = pipeline({"image": str(paths["imagesTr"])})
    out = tmp_path / "out.nii.gz"
    with pytest.raises(RuntimeError, match="grid"):
        save_on_original_grid(torch.zeros_like(data["image"]), data, pipeline, other, out)
    assert not out.exists()


def test_predict_writes_one_mask_per_validation_case(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Task09_Spleen"
    _write_case(dataset_dir, "spleen_2")
    split = tmp_path / "split.json"
    # spleen_99 has no file at all: if the code ever read the test set, this test would crash.
    split.write_text(json.dumps({"train": [], "val": ["spleen_2"], "test": ["spleen_99"]}))
    # A run folder as train.py leaves it: untrained weights are enough to test the wiring.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = replace(TINY, amp=False)
    (run_dir / "summary.json").write_text(json.dumps({"config": asdict(cfg)}))
    torch.save(UNet3D(cfg.channels).state_dict(), run_dir / "best.pt")

    out_dir = tmp_path / "preds"
    args = ["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir)]
    main([*args, "--split-file", str(split), "--out-dir", str(out_dir)])

    assert [p.name for p in out_dir.iterdir()] == ["spleen_2.nii.gz"]
    evaluate_case(out_dir / "spleen_2.nii.gz", dataset_dir / "labelsTr" / "spleen_2.nii.gz")


@needs_data
@pytest.mark.parametrize("case", VAL_CASES)
def test_real_validation_label_survives_the_round_trip(case: str, tmp_path: Path) -> None:
    image = DATASET_DIR / "imagesTr" / f"{case}.nii.gz"
    label = DATASET_DIR / "labelsTr" / f"{case}.nii.gz"
    out = tmp_path / f"{case}.nii.gz"
    _round_trip(image, label, out)
    scores = evaluate_case(out, label)
    print(f"{case}: round-trip Dice {scores['dice']:.4f}")
    assert scores["dice"] > ROUND_TRIP_MIN_DICE
