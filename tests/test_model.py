import torch

from spleenseg.model import UNet3D


def test_output_has_two_class_scores_per_voxel() -> None:
    model = UNet3D(channels=(4, 8, 16))
    out = model(torch.zeros(2, 1, 16, 16, 16))  # 2 patches of 16^3, 1 channel
    assert out.shape == (2, 2, 16, 16, 16)


def test_gradients_reach_every_weight() -> None:
    model = UNet3D(channels=(4, 8, 16))
    model(torch.randn(1, 1, 16, 16, 16)).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_default_size() -> None:
    n_weights = sum(p.numel() for p in UNet3D().parameters())
    assert 1_000_000 < n_weights < 3_000_000  # "small": a few million adjustable numbers
