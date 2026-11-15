"""Evaluate predicted masks against the original labels: per-case table and summary.

Usage:
    uv run python -m spleenseg.evaluate --pred-dir outputs/predictions/<model>
    uv run python -m spleenseg.evaluate --pred-dir outputs/predictions/<model> --split test

Predictions are NIfTI files named <case>.nii.gz with the label's shape and position (same
affine), i.e. back at the original resolution. The default split is validation; the test split
is for the single final evaluation only.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from spleenseg.metrics import bootstrap_ci, dice, hd95

DEFAULT_DATASET_DIR = Path("data/Task09_Spleen")
DEFAULT_SPLIT_FILE = Path("splits/split.json")


def _git_commit() -> str:
    """Commit of the code that produced the numbers, marked "+dirty" if files were edited."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
        except OSError:  # git not installed, e.g. inside a Docker image
            return ""

    commit = git("rev-parse", "HEAD") or "unknown"
    return commit + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def _summary(values: list[float]) -> dict[str, object]:
    defined = [v for v in values if not math.isnan(v)]
    result: dict[str, object] = {
        "n_defined": len(defined),
        "n_undefined": len(values) - len(defined),
    }
    if len(defined) >= 2:
        mean, low, high = bootstrap_ci(defined)
        result |= {"mean": mean, "std": float(np.std(defined, ddof=1)), "ci95": [low, high]}
    return result


def evaluate_case(pred_path: Path, label_path: Path) -> dict[str, float]:
    pred_img = nib.Nifti1Image.from_filename(pred_path)
    label_img = nib.Nifti1Image.from_filename(label_path)
    # Comparing masks on different grids would give meaningless numbers: refuse instead.
    if pred_img.shape != label_img.shape or not np.allclose(pred_img.affine, label_img.affine):
        raise ValueError(f"{pred_path.name}: shape or position differs from the label.")
    pred = np.asanyarray(pred_img.dataobj) > 0
    truth = np.asanyarray(label_img.dataobj) > 0
    spacing = tuple(float(z) for z in label_img.header.get_zooms()[:3])
    return {"dice": dice(pred, truth), "hd95_mm": hd95(pred, truth, spacing)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/eval/<pred-dir>-<split>")
    args = parser.parse_args(argv)

    cases = json.loads(args.split_file.read_text(encoding="utf-8"))[args.split]
    rows = []
    for case in cases:
        scores = evaluate_case(
            args.pred_dir / f"{case}.nii.gz", args.dataset_dir / "labelsTr" / f"{case}.nii.gz"
        )
        rows.append({"case": case, **scores})
        print(f"{case}: Dice {scores['dice']:.4f} | HD95 {scores['hd95_mm']:.2f} mm")

    summary = {
        "split": args.split,
        "n_cases": len(rows),
        "dice": _summary([r["dice"] for r in rows]),
        "hd95_mm": _summary([r["hd95_mm"] for r in rows]),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
    }
    out_dir = args.out_dir or Path("outputs/eval") / f"{args.pred_dir.name}-{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice", "hd95_mm"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
