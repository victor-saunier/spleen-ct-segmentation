"""Tests for figure.py: the cases are picked by rule, and the figure is drawn from saved files."""

from pathlib import Path

import nibabel as nib
import numpy as np

from spleenseg.figure import main, pick_cases


def test_the_median_and_the_lowest_dice_are_picked() -> None:
    scores = {"a": 0.95, "b": 0.90, "c": 0.97, "d": 0.93, "e": 0.96}
    assert pick_cases(scores) == ("a", "b")
    assert pick_cases({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}) == ("c", "d")  # lower middle


def test_the_figure_is_drawn_from_saved_masks(tmp_path: Path) -> None:
    affine = np.diag([0.8, 0.8, 5.0, 1.0])
    label = np.zeros((40, 40, 6), dtype=np.uint8)
    label[10:25, 12:30, 1:5] = 1
    for folder in ("imagesTr", "labelsTr", "pred"):
        (tmp_path / folder).mkdir()
    # a and c are perfect, so the median panel shows a case with no disagreement at all.
    for case, shift in (("a", 0), ("b", 3), ("c", 0)):
        pred = np.roll(label, shift, axis=0)
        scan = np.where(label == 1, 120.0, 40.0).astype(np.float32)
        nib.save(nib.Nifti1Image(scan, affine), str(tmp_path / "imagesTr" / f"{case}.nii.gz"))
        nib.save(nib.Nifti1Image(label, affine), str(tmp_path / "labelsTr" / f"{case}.nii.gz"))
        nib.save(nib.Nifti1Image(pred, affine), str(tmp_path / "pred" / f"{case}.nii.gz"))
    (tmp_path / "per_case.csv").write_text("case,dice,hd95_mm\na,1.0,0.0\nb,0.8,2.4\nc,1.0,0.0\n")

    out = tmp_path / "figure.png"
    main(
        ["--pred-dir", str(tmp_path / "pred"), "--eval-dir", str(tmp_path)]
        + ["--dataset-dir", str(tmp_path), "--out", str(out)]
    )
    assert out.stat().st_size > 0
