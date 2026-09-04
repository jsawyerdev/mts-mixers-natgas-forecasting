"""Reversible instance normalization (Kim et al., ICLR 2021): subtract the per-window mean,
divide by the per-window std, apply a learnable affine transform. Used here as the optional
normalization stage the MTS-Mixers repo itself supports, isolated from the mixing architecture
so the comparison in cli.py is "does normalization help" independent of "does mixing help".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class InstanceStats:
    """Per-window mean/std captured on normalize(), needed to invert denormalize()."""

    mean: Tensor
    std: Tensor


class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def normalize(self, x: Tensor) -> tuple[Tensor, InstanceStats]:
        """x: (B, L, C) -> normalized x, stats to pass to denormalize()."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
        x_norm = (x - mean) / std * self.weight + self.bias
        return x_norm, InstanceStats(mean=mean, std=std)

    def denormalize(self, x: Tensor, stats: InstanceStats) -> Tensor:
        """x: (B, H, C) in the normalized scale -> back to the original price scale."""
        x = (x - self.bias) / (self.weight + self.eps)
        return x * stats.std + stats.mean
