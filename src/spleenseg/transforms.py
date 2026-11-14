"""MONAI transform pipelines: deterministic preprocessing and training augmentations.

The values below come from the training-set statistics (`uv run python -m spleenseg.stats`).
docs/decisions/0002-preprocessing.md explains each choice.
"""

from collections.abc import Sequence

from monai import transforms as mt

KEYS = ("image", "label")

# 1.5 x 1.5 x 2.0 mm: a 96^3 patch then covers 144 x 144 x 192 mm, more than the largest
# training spleen (117 x 132 x 143 mm), so a single patch can contain a whole spleen.
SPACING_MM = (1.5, 1.5, 2.0)
# Spleen intensities in the training cases: -37 to 176 HU (0.5th to 99.5th percentile),
# rounded outward to 10 HU.
HU_WINDOW = (-40.0, 180.0)


def preprocessing(
    spacing: Sequence[float] = SPACING_MM, hu_window: tuple[float, float] = HU_WINDOW
) -> mt.Compose:
    """Deterministic steps applied to every case: train, validation and test."""
    return mt.Compose(
        [
            mt.LoadImaged(keys=KEYS),
            mt.EnsureChannelFirstd(keys=KEYS),
            # labels=None: read the axis convention from the image metadata (MONAI's new default).
            mt.Orientationd(keys=KEYS, axcodes="RAS", labels=None),
            # Smooth interpolation for the image; nearest neighbour keeps the mask strictly 0/1.
            mt.Spacingd(keys=KEYS, pixdim=spacing, mode=("bilinear", "nearest")),
            mt.ScaleIntensityRanged(
                keys="image",
                a_min=hu_window[0],
                a_max=hu_window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Crop to the box around everything brighter than the window's lower bound, which
            # removes the air around the patient.
            mt.CropForegroundd(keys=KEYS, source_key="image"),
        ]
    )
