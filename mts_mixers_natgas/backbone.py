"""DLinear backbone (Zeng et al., "Are Transformers Effective for Time Series Forecasting?",
AAAI 2023): decompose into a moving-average trend and a seasonal residual, forecast each with
its own per-channel linear layer, then recombine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MovingAverageDecomposition(nn.Module):
    """Split x into (trend, seasonal) via a centered moving average, edge-padded by replication."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for a centered moving average")
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B, L, C) -> (trend (B, L, C), seasonal (B, L, C))."""
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)  # (B, C, L)
        x_padded = F.pad(x_t, (pad, pad), mode="replicate")
        trend = self.avg_pool(x_padded).transpose(1, 2)  # (B, L, C)
        seasonal = x - trend
        return trend, seasonal


class DLinear(nn.Module):
    """Per-channel linear forecaster over a trend/seasonal decomposition."""

    def __init__(
        self, seq_len: int, pred_len: int, num_channels: int, kernel_size: int = 25
    ) -> None:
        super().__init__()
        kernel_size = min(kernel_size, seq_len - 1 if seq_len % 2 == 0 else seq_len)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(kernel_size, 1)
        self.decompose = MovingAverageDecomposition(kernel_size)
        self.trend_linear = nn.ModuleList(
            nn.Linear(seq_len, pred_len) for _ in range(num_channels)
        )
        self.seasonal_linear = nn.ModuleList(
            nn.Linear(seq_len, pred_len) for _ in range(num_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, C) -> (B, H, C)."""
        trend, seasonal = self.decompose(x)
        trend_out = torch.stack(
            [linear(trend[:, :, c]) for c, linear in enumerate(self.trend_linear)],
            dim=-1,
        )
        seasonal_out = torch.stack(
            [
                linear(seasonal[:, :, c])
                for c, linear in enumerate(self.seasonal_linear)
            ],
            dim=-1,
        )
        return trend_out + seasonal_out
