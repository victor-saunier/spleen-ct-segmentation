"""Draw the README figure: the median and the worst case of a saved evaluation.

Usage (after predict and evaluate):
    uv run python -m spleenseg.figure --pred-dir outputs/predictions/test/<run>
        --eval-dir outputs/eval/<run>-test

The cases are picked by code from per_case.csv, not by hand: the one with the median Dice and
the one with the lowest. Each panel shows the axial slice where prediction and label disagree
most, so the figure shows the errors rather than the best-looking slice. Nothing is evaluated
here: masks and Dice scores are read from files saved earlier.
The figure shows CT data under CC BY-SA 4.0: any copy must credit the Medical Segmentation
Decathlon and keep that licence.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only, never open a window

import matplotlib.pyplot as plt  # noqa: E402 (must come after choosing the backend)
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from spleenseg.data import DEFAULT_DATASET_DIR  # noqa: E402
from spleenseg.preview import largest_slice  # noqa: E402

DISPLAY_HU = (-150.0, 250.0)  # soft-tissue display window, for viewing only
MARGIN_MM = 40.0
LABEL_COLOUR, PRED_COLOUR = "#0072B2", "#E69F00"  # blue and orange: readable with colour blindness


def pick_cases(per_case: dict[str, float]) -> tuple[str, str]:
    """The case with the median Dice (the lower middle one if the count is even) and the worst."""
    ranked = sorted(per_case, key=per_case.__getitem__)
    return ranked[(len(ranked) - 1) // 2], ranked[0]


def _load_ras(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.Nifti1Image.from_filename(str(path)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", type=Path, required=True, help="saved masks, <case>.nii.gz")
    parser.add_argument("--eval-dir", type=Path, required=True, help="has per_case.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/test-cases.png"))
    args = parser.parse_args(argv)

    with (args.eval_dir / "per_case.csv").open(encoding="utf-8") as f:
        per_case = {row["case"]: float(row["dice"]) for row in csv.DictReader(f)}
    median, worst = pick_cases(per_case)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, case, kind in zip(axes, (median, worst), ("Median", "Lowest"), strict=True):
        image = _load_ras(args.dataset_dir / "imagesTr" / f"{case}.nii.gz")
        label = np.asanyarray(_load_ras(args.dataset_dir / "labelsTr" / f"{case}.nii.gz").dataobj)
        pred = np.asanyarray(_load_ras(args.pred_dir / f"{case}.nii.gz").dataobj)
        errors = (label != pred).sum(axis=(0, 1))
        z = int(errors.argmax()) if errors.any() else largest_slice(label)  # perfect: most spleen
        sx, sy = (float(s) for s in image.header.get_zooms()[:2])

        # Crop around spleen and prediction on that slice, with a margin, in mm.
        xs, ys = np.nonzero((label[:, :, z] > 0) | (pred[:, :, z] > 0))
        x0, x1 = xs.min() * sx - MARGIN_MM, (xs.max() + 1) * sx + MARGIN_MM
        y0, y1 = ys.min() * sy - MARGIN_MM, (ys.max() + 1) * sy + MARGIN_MM
        extent = (0.0, label.shape[0] * sx, 0.0, label.shape[1] * sy)

        hu = image.dataobj[:, :, z]
        ax.imshow(
            hu.T, cmap="gray", origin="lower", extent=extent, vmin=DISPLAY_HU[0], vmax=DISPLAY_HU[1]
        )
        for mask, colour in ((label, LABEL_COLOUR), (pred, PRED_COLOUR)):
            ax.contour(
                mask[:, :, z].T,
                levels=[0.5],
                colors=colour,
                origin="lower",
                extent=extent,
                linewidths=1.6,
            )
        ax.set_xlim(x1, x0)  # reversed: radiological convention, patient's left on the right
        ax.set_ylim(y0, y1)
        ax.set_axis_off()
        ax.set_title(f"{kind} Dice of the test set: {case}, Dice {per_case[case]:.3f}", fontsize=10)

    legend = [
        Line2D([], [], color=LABEL_COLOUR, label="Reference label"),
        Line2D([], [], color=PRED_COLOUR, label="Model prediction (baseline U-Net)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
    fig.suptitle("Axial slice with the most disagreement between prediction and label")
    fig.text(
        0.5,
        0.08,
        "CT and labels: Medical Segmentation Decathlon, Task09 Spleen, "
        "CC BY-SA 4.0. Cropped, windowed and outlined; this figure is CC BY-SA 4.0.",
        ha="center",
        fontsize=8,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"Saved {args.out} ({median} median, {worst} lowest)")


if __name__ == "__main__":
    main()
