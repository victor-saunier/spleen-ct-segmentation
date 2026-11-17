"""Tests for the transform pipelines, on a small synthetic CT (no real data needed)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from spleenseg.transforms import preprocessing, train_transforms

# Synthetic case: 40 x 40 x 10 voxels of 0.75 x 0.75 x 5 mm, like a thick-slice scan.
SOURCE_SPACING = (0.75, 0.75, 5.0)
SPLEEN_MM3 = (8 * 0.75) * (8 * 0.75) * (4 * 5.0)  # the 8 x 8 x 4 voxel cube below


@pytest.fixture
def case(tmp_path: Path) -> dict[str, str]:
    image = np.full((40, 40, 10), -1000.0, dtype=np.float32)  # air
    image[8:32, 8:32, :] = 50.0  # the "body": soft tissue
    label = np.zeros((40, 40, 10), dtype=np.uint8)
    label[12:20, 12:20, 3:7] = 1  # the "spleen"
    image[label == 1] = 100.0

    affine = np.diag([*SOURCE_SPACING, 1.0])
    paths = {"image": str(tmp_path / "image.nii.gz"), "label": str(tmp_path / "label.nii.gz")}
    nib.save(nib.Nifti1Image(image, affine), paths["image"])
    nib.save(nib.Nifti1Image(label, affine), paths["label"])
    return paths


def test_preprocessing_resamples_to_the_target_spacing(case: dict[str, str]) -> None:
    out = preprocessing(spacing=(1.5, 1.5, 2.0))(case)
    assert tuple(float(s) for s in out["image"].pixdim) == (1.5, 1.5, 2.0)
    assert out["image"].shape == out["label"].shape


def test_preprocessing_windows_intensities_into_0_1(case: dict[str, str]) -> None:
    image = preprocessing()(case)["image"]
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_preprocessing_keeps_the_mask_binary_and_its_volume(case: dict[str, str]) -> None:
    label = preprocessing(spacing=(1.5, 1.5, 2.0))(case)["label"]
    assert set(torch.unique(label).tolist()) == {0.0, 1.0}
    # Nearest-neighbour resampling may move the border by a voxel, not change the organ's size.
    volume_mm3 = float(label.sum()) * 1.5 * 1.5 * 2.0
    assert volume_mm3 == pytest.approx(SPLEEN_MM3, rel=0.25)


def test_preprocessing_the_image_alone_gives_the_same_image(case: dict[str, str]) -> None:
    # Prediction reads no label: it must still see exactly what training and selection saw.
    alone = preprocessing(keys=("image",))({"image": case["image"]})["image"]
    assert torch.equal(alone, preprocessing()(case)["image"])


def test_preprocessing_crops_the_air_around_the_body(case: dict[str, str]) -> None:
    image = preprocessing(spacing=(1.5, 1.5, 2.0))(case)["image"]
    # Full width would be 40 * 0.75 / 1.5 = 20 voxels; the body spans only 24 * 0.75 / 1.5 = 12.
    assert image.shape[1:3] == (12, 12)


def test_training_gives_patches_of_the_requested_size(case: dict[str, str]) -> None:
    # The synthetic case is only 12 x 12 x 25 voxels after preprocessing: padding must kick in.
    patches = train_transforms(patch_size=(16, 16, 16), num_samples=3)(case)
    assert len(patches) == 3
    for patch in patches:
        assert patch["image"].shape == (1, 16, 16, 16)
        assert patch["label"].shape == (1, 16, 16, 16)


def test_training_keeps_the_mask_binary_after_augmentation(case: dict[str, str]) -> None:
    pipeline = train_transforms(patch_size=(16, 16, 16), num_samples=8)
    pipeline.set_random_state(seed=0)
    for patch in pipeline(case):
        assert set(torch.unique(patch["label"]).tolist()) <= {0.0, 1.0}


def test_training_is_reproducible_with_a_seed(case: dict[str, str]) -> None:
    def run() -> list[torch.Tensor]:
        pipeline = train_transforms(patch_size=(16, 16, 16), num_samples=4)
        pipeline.set_random_state(seed=123)
        return [patch["image"] for patch in pipeline(case)]

    for first, second in zip(run(), run(), strict=True):
        assert torch.equal(first, second)
