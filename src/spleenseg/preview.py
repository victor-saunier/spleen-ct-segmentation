"""Save a picture of one TRAINING case, before and after preprocessing.

Usage (after the download):
    uv run python -m spleenseg.preview

The PNG goes to outputs/ (git-ignored). It shows CT data under CC BY-SA 4.0: any shared copy
must credit the Medical Segmentation Decathlon and keep that license.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only, never open a window

import matplotlib.pyplot as plt  # noqa: E402 (must come after choosing the backend)
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from spleenseg.transforms import HU_WINDOW, SPACING_MM, preprocessing  # noqa: E402

DEFAULT_DATASET_DIR = Path("data/Task09_Spleen")
DEFAULT_SPLIT_FILE = Path("splits/split.json")
DEFAULT_OUT_DIR = Path("outputs")


def largest_slice(mask: np.ndarray) -> int:
    """Index of the axial slice (last axis) that holds the most spleen voxels."""
    return int(mask.reshape(-1, mask.shape[-1]).sum(axis=0).argmax())


def _show(
    ax: Axes,
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float],
    value_range: tuple[float, float],
    title: str,
) -> None:
    # extent in mm, so both panels have true proportions despite different voxel sizes.
    extent = (0.0, image.shape[0] * spacing[0], 0.0, image.shape[1] * spacing[1])
    ax.imshow(
        image.T,
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=value_range[0],
        vmax=value_range[1],
    )
    ax.contour(mask.T, levels=[0.5], colors="red", origin="lower", extent=extent)
    ax.invert_xaxis()  # radiological convention: the patient's left appears on the image's right
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mm")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case", help="a training case ID; default: the first one")
    args = parser.parse_args(argv)

    train_ids = json.loads(args.split_file.read_text(encoding="utf-8"))["train"]
    case = args.case or train_ids[0]
    if case not in train_ids:
        raise SystemExit(f"{case} is not a training case: previews never show validation or test.")
    paths = {
        "image": str(args.dataset_dir / "imagesTr" / f"{case}.nii.gz"),
        "label": str(args.dataset_dir / "labelsTr" / f"{case}.nii.gz"),
    }

    # Original: reoriented to RAS like the pipeline, but still in raw Hounsfield units.
    raw_image = nib.as_closest_canonical(nib.Nifti1Image.from_filename(paths["image"]))
    raw_mask = nib.as_closest_canonical(nib.Nifti1Image.from_filename(paths["label"]))
    raw_hu = raw_image.get_fdata(dtype=np.float32)
    raw_label = np.asanyarray(raw_mask.dataobj)
    raw_spacing = tuple(float(z) for z in raw_image.header.get_zooms()[:2])

    processed = preprocessing()(paths)
    image = processed["image"][0].numpy()
    label = processed["label"][0].numpy()

    raw_z, z = largest_slice(raw_label), largest_slice(label)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _show(
        axes[0],
        raw_hu[:, :, raw_z],
        raw_label[:, :, raw_z],
        (raw_spacing[0], raw_spacing[1]),
        (-1000.0, 1000.0),
        f"Original, -1000 to 1000 HU\n{raw_spacing[0]:.2f} mm pixels",
    )
    _show(
        axes[1],
        image[:, :, z],
        label[:, :, z],
        (SPACING_MM[0], SPACING_MM[1]),
        (0.0, 1.0),
        f"Preprocessed, window {HU_WINDOW[0]:.0f} to {HU_WINDOW[1]:.0f} HU\n"
        f"{SPACING_MM[0]} mm pixels, cropped to the body",
    )
    fig.suptitle(f"Training case {case}: axial slice with the most spleen (red outline)")
    fig.text(
        0.5,
        0.01,
        "Data: Medical Segmentation Decathlon, Task09 Spleen (CC BY-SA 4.0)",
        ha="center",
        fontsize=8,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"preview_{case}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
