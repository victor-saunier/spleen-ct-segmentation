from dataclasses import replace
from pathlib import Path

import torch

from spleenseg.config import DEFAULT_CONFIG, load_config
from spleenseg.model import UNet3D
from spleenseg.train import build_model

ROOT = Path(__file__).resolve().parents[1]


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


def test_the_second_model_is_built_from_the_config() -> None:
    cfg = load_config(ROOT / DEFAULT_CONFIG, "monai_unet")
    n_weights = sum(p.numel() for p in build_model(cfg).parameters())
    assert 4_000_000 < n_weights < 6_000_000  # 4.8 M measured in the T05b pilot
    # Same output contract as the baseline, on a tiny version: 2 scores per voxel.
    model = build_model(replace(cfg, channels=(4, 8, 16)))
    out = model(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 2, 16, 16, 16)
    out.mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
