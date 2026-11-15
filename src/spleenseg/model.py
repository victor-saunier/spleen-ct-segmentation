"""A small 3D U-Net written from scratch: the baseline model.

Input (batch, 1, X, Y, Z) -> output (batch, 2, X, Y, Z): one score per class (background,
spleen) for every voxel. X, Y and Z must be divisible by 2 ** (len(channels) - 1), because each
level halves the size and the decoder must find a skip connection of the same size.
"""

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two 3x3x3 convolutions, each followed by instance normalisation and a LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # Instance norm normalises each patch on its own: batch statistics are unreliable
            # with only a few patches per step.
            nn.InstanceNorm3d(out_channels, affine=True),
            # LeakyReLU keeps a small slope below zero, so no unit can get stuck at zero output.
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UNet3D(nn.Module):
    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64, 128),
        in_channels: int = 1,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        # Encoder: one block per level; between levels, max pooling halves X, Y and Z.
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoder.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool3d(2)

        # Decoder, from the deepest level up: a transposed convolution doubles the size, then a
        # block merges it with the encoder's features of the same level (the skip connection).
        deeper, shallower = channels[:0:-1], channels[-2::-1]  # (128, 64, 32), (64, 32, 16)
        self.upsample = nn.ModuleList(
            nn.ConvTranspose3d(d, s, kernel_size=2, stride=2)
            for d, s in zip(deeper, shallower, strict=True)
        )
        self.decoder = nn.ModuleList(ConvBlock(2 * s, s) for s in shallower)
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)  # one score per class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for level, block in enumerate(self.encoder):
            x = block(x)
            if level < len(self.encoder) - 1:  # the deepest level is not pooled
                skips.append(x)
                x = self.pool(x)
        for upsample, block, skip in zip(self.upsample, self.decoder, reversed(skips), strict=True):
            x = block(torch.cat([upsample(x), skip], dim=1))  # stack along the channel axis
        return self.head(x)
