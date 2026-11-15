"""Segmentation metrics: Dice overlap and 95th-percentile Hausdorff distance (HD95).

Definitions and the empty-prediction policy: docs/decisions/0003-evaluation.md.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """2 * |pred AND truth| / (|pred| + |truth|): 1 is a perfect overlap, 0 is none."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = int(pred.sum()) + int(truth.sum())
    if total == 0:
        return 1.0  # both empty: they agree perfectly
    return 2 * int(np.logical_and(pred, truth).sum()) / total


def _border(mask: np.ndarray) -> np.ndarray:
    """Voxels of the mask that touch the background."""
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, truth: np.ndarray, spacing: Sequence[float]) -> float:
    """Larger of the two directed 95th-percentile border distances, in millimetres.

    Returns NaN when either mask is empty: the distance is then undefined, and callers count
    such cases instead of hiding them behind an invented penalty value.
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    if not pred.any() or not truth.any():
        return math.nan
    pred_border, truth_border = _border(pred), _border(truth)
    # For every voxel: distance in mm to the nearest border voxel of the other mask.
    to_truth = ndimage.distance_transform_edt(~truth_border, sampling=spacing)
    to_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    return float(
        max(np.percentile(to_truth[pred_border], 95), np.percentile(to_pred[truth_border], 95))
    )


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of `values` and a percentile-bootstrap confidence interval of that mean.

    Redraws the cases with replacement `n_resamples` times and keeps the middle `confidence`
    share of the resampled means. Applied to per-case differences between two models evaluated on
    the same cases, it gives a paired comparison.
    shortcut: with very few cases (7 here) the percentile bootstrap gives too narrow an interval;
    upgrade to BCa or to more cases if a decision depends on it.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or np.isnan(data).any():
        raise ValueError("Need a non-empty 1D list of numbers without NaN.")
    rng = np.random.default_rng(seed)
    resampled_means = rng.choice(data, size=(n_resamples, data.size), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(resampled_means, [tail, 100 - tail])
    return float(data.mean()), float(low), float(high)
