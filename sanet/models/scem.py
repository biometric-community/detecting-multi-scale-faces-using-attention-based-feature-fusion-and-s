"""Smoothed Context Enhancement Module (SCEM; Fig. 4, Sec. III-C)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedBlock(nn.Module):
    """Dilated Block: d=3 (3×3) → 3×3 → d=2 (2×2) → 2×2 (Fig. 4).

    Paper padding (3 and 2) can shift spatial size for even kernels; we
    bilinear-resize back to the input HxW after the block.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # dilated 3×3, rate 3, pad 3 (paper)
        self.d3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=True)
        self.c3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        # dilated 2×2, rate 2, pad 2 (paper)
        self.d2 = nn.Conv2d(channels, channels, kernel_size=2, padding=2, dilation=2, bias=True)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=2, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        y = self.act(self.d3(x))
        y = self.act(self.c3(y))
        y = self.act(self.d2(y))
        y = self.act(self.c2(y))
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y


class SCEM(nn.Module):
    """Split channels, one vs two Dilated Blocks, concatenate (Fig. 4)."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        assert channels % 2 == 0
        half = channels // 2
        self.block1 = DilatedBlock(half)
        self.block2a = DilatedBlock(half)
        self.block2b = DilatedBlock(half)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        a, b = x[:, :c], x[:, c:]
        a = self.block1(a)
        b = self.block2b(self.block2a(b))
        return self.out(torch.cat([a, b], dim=1))
