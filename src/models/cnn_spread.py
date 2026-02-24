"""
CNN for next-day wildfire spread prediction (Family A: no SAR, Family B: + Sentinel-1 SAR).

Input: stacked rasters at day t (fire mask, weather, topo, optional SAR).
Output: per-pixel probability of burn at day t+1 (same grid).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# Channel counts matching dataset.py
FAMILY_A_CHANNELS = 11   # fire_mask(1) + weather(7) + elevation, slope, aspect(3)
FAMILY_B_CHANNELS = 13   # Family A + VV, VH (2)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU x2."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpreadCNN(nn.Module):
    """
    Small U-Net-style CNN for binary next-day spread prediction.

    - in_channels: 11 for Family A (no SAR), 13 for Family B (+ SAR).
    - Preserves spatial size (same H, W in and out) for arbitrary patch/grid sizes.
    """

    def __init__(
        self,
        in_channels: int = FAMILY_A_CHANNELS,
        base_channels: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        # Encoder: depth blocks, each (in_c -> out_c) then pool (except last)
        enc_chans = [base_channels * (2**i) for i in range(depth)]
        self.encoder = nn.ModuleList()
        c = in_channels
        for out_c in enc_chans:
            self.encoder.append(ConvBlock(c, out_c))
            c = out_c
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        bottleneck_c = enc_chans[-1] * 2
        self.bottleneck = ConvBlock(c, bottleneck_c)

        # Decoder: up then concat skip then conv
        self.ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev_c = bottleneck_c
        for i in range(depth):
            # Upsample to the resolution of the corresponding skip connection.
            skip_c = enc_chans[depth - 1 - i]
            self.ups.append(nn.ConvTranspose2d(prev_c, skip_c, 2, stride=2))

            # After concatenating with the skip features, reduce channels.
            dec_in = skip_c * 2
            dec_out = enc_chans[depth - 2 - i] if i < depth - 1 else enc_chans[0]
            self.decoder_blocks.append(ConvBlock(dec_in, dec_out))
            prev_c = dec_out

        # Head
        self.head = nn.Conv2d(enc_chans[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for i in range(self.depth):
            h = self.ups[i](h)
            s = skips[self.depth - 1 - i]
            if h.shape[2:] != s.shape[2:]:
                h = nn.functional.interpolate(h, size=s.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, s], dim=1)
            h = self.decoder_blocks[i](h)

        return self.head(h)


def build_spread_cnn(
    family: str = "A",
    in_channels: Optional[int] = None,
    base_channels: int = 32,
    depth: int = 3,
) -> SpreadCNN:
    """Build SpreadCNN for Family A (no SAR) or B (+ SAR)."""
    if in_channels is None:
        in_channels = FAMILY_B_CHANNELS if family.upper() == "B" else FAMILY_A_CHANNELS
    return SpreadCNN(in_channels=in_channels, base_channels=base_channels, depth=depth)
