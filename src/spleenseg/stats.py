"""Statistics of the TRAINING cases, used to choose the preprocessing parameters.

Usage (after the download):
    uv run python -m spleenseg.stats

Only the training cases of splits/split.json are read: choosing preprocessing from validation
or test cases would leak information about them into the method.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_DATASET_DIR = Path("data/Task09_Spleen")
DEFAULT_SPLIT_FILE = Path("splits/split.json")


@dataclass
class CaseStats:
    spacing: tuple[float, ...]  # voxel size in mm along each array axis
    spleen_fraction: float  # share of the volume's voxels that are spleen
    spleen_extent_mm: tuple[float, ...]  # size of the spleen's bounding box in mm
    spleen_hu: np.ndarray  # intensities (Hounsfield units) of every spleen voxel


def case_stats(image: np.ndarray, label: np.ndarray, spacing: tuple[float, ...]) -> CaseStats:
    """Spacing, spleen size and spleen intensities of one case."""
    spleen = label > 0
    coords = np.argwhere(spleen)
    extent_voxels = coords.max(axis=0) - coords.min(axis=0) + 1
    return CaseStats(
        spacing=spacing,
        spleen_fraction=float(spleen.mean()),
        spleen_extent_mm=tuple(float(n * s) for n, s in zip(extent_voxels, spacing, strict=True)),
        spleen_hu=image[spleen],
    )


def _min_med_max(values: np.ndarray) -> str:
    return f"min {values.min():.2f} | median {np.median(values):.2f} | max {values.max():.2f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    args = parser.parse_args(argv)

    train_ids = json.loads(args.split_file.read_text(encoding="utf-8"))["train"]
    stats = []
    for cid in train_ids:
        image = nib.Nifti1Image.from_filename(args.dataset_dir / "imagesTr" / f"{cid}.nii.gz")
        label = nib.Nifti1Image.from_filename(args.dataset_dir / "labelsTr" / f"{cid}.nii.gz")
        spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
        # get_fdata applies the file's intensity scaling, so values are true Hounsfield units.
        image_hu = image.get_fdata(dtype=np.float32)
        stats.append(case_stats(image_hu, np.asanyarray(label.dataobj), spacing))

    spacings = np.array([s.spacing for s in stats])
    extents = np.array([s.spleen_extent_mm for s in stats])
    fractions = np.array([s.spleen_fraction for s in stats]) * 100
    hu = np.concatenate([s.spleen_hu for s in stats])

    print(f"Training cases: {len(stats)}")
    for axis, name in enumerate("xyz"):
        print(f"Spacing {name} (mm):         {_min_med_max(spacings[:, axis])}")
    for axis, name in enumerate("xyz"):
        print(f"Spleen extent {name} (mm):   {_min_med_max(extents[:, axis])}")
    print(f"Spleen share of volume (%): {_min_med_max(fractions)}")
    percentiles = [0.5, 5, 50, 95, 99.5]
    values = np.percentile(hu, percentiles)
    print(
        "Spleen HU percentiles:      "
        + " | ".join(f"p{p}: {v:.0f}" for p, v in zip(percentiles, values, strict=True))
    )


if __name__ == "__main__":
    main()
