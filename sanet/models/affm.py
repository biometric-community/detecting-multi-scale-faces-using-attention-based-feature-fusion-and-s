"""Attention-guided Feature Fusion Module (AFFM; Fig. 3, Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionModule


class AFFM(nn.Module):
    """Fuse low-level and high-level maps with attention on the upsampled high-level path.

    Two 1×1 (or 3×3) projections → 512 channels; upsample high; AM; element-wise sum.
    """

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        out_channels: int = 512,
        attention_mode: str = "sa",
    ) -> None:
        super().__init__()
        self.proj_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.am = AttentionModule(out_channels, mode=attention_mode)
        self.attention_mode = attention_mode

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        fl = self.proj_low(low)
        fh = self.proj_high(high)
        fh = F.interpolate(fh, size=fl.shape[-2:], mode="nearest")
        if self.attention_mode == "ca_plus_sa":
            return self.am(fh, low=fl)
        attended = self.am(fh)
        return fl + attended


class SmoothLayer(nn.Module):
    """Lateral smooth conv for deepest backbone levels without AFFM (Fig. 2)."""

    def __init__(self, in_channels: int, out_channels: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
