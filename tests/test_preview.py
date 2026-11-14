import numpy as np

from spleenseg.preview import largest_slice


def test_largest_slice_finds_the_slice_with_most_spleen() -> None:
    mask = np.zeros((4, 4, 5), dtype=np.uint8)
    mask[0, 0, 1] = 1
    mask[:2, :2, 3] = 1  # 4 voxels on slice 3, more than anywhere else
    assert largest_slice(mask) == 3
