"""Attention modules for AFFM (Fig. 3; Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel-wise attention: F_c = σ(G(F)) ⊙ F (Fig. 3 CAM; G = global avg pool)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # G(F): C×1×1
        w = torch.sigmoid(F.adaptive_avg_pool2d(x, 1))
        return x * w


class SpatialAttention(nn.Module):
    """Spatial-wise attention: F_s = σ(Conv(F)) ⊙ F (Fig. 3 SAM)."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.conv(x))
        return x * w


class AttentionModule(nn.Module):
    """AFFM attention variants: ca | sa | ca_sa (series) | ca_plus_sa (separate)."""

    def __init__(self, channels: int = 512, mode: str = "sa") -> None:
        super().__init__()
        self.mode = mode
        self.ca = ChannelAttention()
        self.sa = SpatialAttention(channels)

    def forward(self, x: torch.Tensor, low: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "ca":
            return self.ca(x)
        if self.mode == "sa":
            return self.sa(x)
        if self.mode == "ca_sa":
            # Eq. (3): spatial after channel — F_m = S(C_v(F_c)) ⊙ F_c
            return self.sa(self.ca(x))
        if self.mode == "ca_plus_sa":
            # Eq. (4)-style separate: SA on low-level, CA on high-level, then sum
            if low is None:
                raise ValueError("ca_plus_sa requires low-level features")
            return self.sa(low) + self.ca(x)
        raise ValueError(f"Unknown attention mode: {self.mode}")
