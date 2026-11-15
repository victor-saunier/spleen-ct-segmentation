"""Metric tests on synthetic masks whose answers can be worked out by hand."""

import math

import numpy as np
import pytest
import torch
from monai.metrics import compute_hausdorff_distance
from scipy import ndimage

from spleenseg.metrics import bootstrap_ci, dice, hd95


def cube(z_start: int, x_start: int = 2, size: int = 4) -> np.ndarray:
    mask = np.zeros((10, 10, 12), dtype=np.uint8)
    mask[x_start : x_start + size, 2 : 2 + size, z_start : z_start + size] = 1
    return mask


def test_identical_masks() -> None:
    assert dice(cube(2), cube(2)) == 1.0
    assert hd95(cube(2), cube(2), spacing=(1.0, 1.0, 5.0)) == 0.0


def test_masks_without_overlap_have_dice_zero() -> None:
    assert dice(cube(0), cube(6)) == 0.0


def test_half_overlap() -> None:
    # Shifted by 2 voxels along x: 2 x 4 x 4 = 32 shared voxels out of 64 + 64.
    assert dice(cube(2), cube(2, x_start=4)) == 0.5


def test_hd95_uses_the_voxel_size() -> None:
    # Shifted by 2 slices: the top and bottom faces are 2 slices away from the truth's.
    assert hd95(cube(4), cube(2), spacing=(1.0, 1.0, 1.0)) == 2.0
    assert hd95(cube(4), cube(2), spacing=(1.0, 1.0, 5.0)) == 10.0  # 2 slices of 5 mm


def test_empty_prediction() -> None:
    empty = np.zeros((10, 10, 12), dtype=np.uint8)
    assert dice(empty, cube(2)) == 0.0
    assert math.isnan(hd95(empty, cube(2), spacing=(1.0, 1.0, 1.0)))


def _blob(seed: int) -> np.ndarray:
    """A random smooth 3D shape, kept away from the array edges."""
    noise = np.random.default_rng(seed).normal(size=(24, 24, 16))
    mask = np.zeros_like(noise, dtype=bool)
    mask[3:-3, 3:-3, 3:-3] = ndimage.gaussian_filter(noise, 2.0)[3:-3, 3:-3, 3:-3] > 0
    return mask


# MONAI 1.6.1 calls one of its own deprecated options inside this function.
@pytest.mark.filterwarnings("ignore:.*always_return_as_numpy.*:FutureWarning")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hd95_matches_monai(seed: int) -> None:
    pred, truth, spacing = _blob(seed), _blob(seed + 10), (0.8, 0.8, 5.0)
    # MONAI expects (batch, channel, x, y, z); a single binary channel needs include_background.
    expected = compute_hausdorff_distance(
        torch.from_numpy(pred)[None, None],
        torch.from_numpy(truth)[None, None],
        include_background=True,
        percentile=95,
        spacing=spacing,
    ).item()
    assert hd95(pred, truth, spacing) == pytest.approx(expected, rel=1e-6)


def test_bootstrap_of_identical_values_collapses_to_that_value() -> None:
    assert bootstrap_ci([0.9] * 7) == pytest.approx((0.9, 0.9, 0.9))


def test_bootstrap_interval_contains_the_mean_and_is_reproducible() -> None:
    values = [0.95, 0.93, 0.96, 0.70, 0.94, 0.91, 0.97]
    mean, low, high = bootstrap_ci(values, seed=0)
    assert low < mean < high
    assert mean == pytest.approx(float(np.mean(values)))
    assert bootstrap_ci(values, seed=0) == (mean, low, high)


def test_bootstrap_interval_widens_with_spread() -> None:
    _, low_a, high_a = bootstrap_ci([0.90, 0.91, 0.92, 0.93])
    _, low_b, high_b = bootstrap_ci([0.60, 0.80, 0.95, 0.99])
    assert high_b - low_b > high_a - low_a


def test_bootstrap_refuses_missing_values() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([0.9, math.nan])
