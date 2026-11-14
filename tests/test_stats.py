import numpy as np

from spleenseg.stats import case_stats


def test_case_stats_on_a_synthetic_cube() -> None:
    image = np.full((10, 10, 10), -1000.0, dtype=np.float32)  # air everywhere
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[2:4, 2:5, 2:6] = 1  # a 2 x 3 x 4 voxel "spleen"
    image[label == 1] = 100.0  # soft-tissue intensity inside it

    stats = case_stats(image, label, spacing=(1.0, 1.0, 2.0))

    assert stats.spleen_fraction == 24 / 1000
    assert stats.spleen_extent_mm == (2.0, 3.0, 8.0)  # z voxels are 2 mm thick
    assert stats.spleen_hu.shape == (24,)
    assert np.all(stats.spleen_hu == 100.0)
