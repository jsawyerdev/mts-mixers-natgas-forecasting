"""MTS-Mixers backbone (Li et al., "MTS-Mixers: Multivariate Time Series Forecasting via
Factorized Temporal and Channel Mixing," arXiv:2302.04501): separate "how do timestamps in one
series relate to each other" from "how do variables relate to each other" into two lightweight
MLP-mixer blocks instead of a Transformer attention stack.

This is a from-scratch reimplementation of the paper's two central ideas -- factorized temporal
mixing and low-rank channel mixing -- not a port of the official repo at plumprc/MTS-Mixers. The
exact factorization scheme and channel-compression mechanism here are simplified.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class TemporalMixing(nn.Module):
    """Factorized temporal mixing: interleave the L timestamps into `stride` subsequences of
    length L/stride, mix each subsequence with one shared MLP along the (shorter) time axis,
    then merge back in original order.

    Interleaving (not contiguous chunking) is deliberate: neighbouring hourly bars are often
    highly redundant, so striding spreads each subsequence across the whole window instead of
    handing the mixer eight cheap-to-predict-from-each-other consecutive hours.
    """

    def __init__(self, seq_len: int, stride: int, hidden_dim: int) -> None:
        super().__init__()
        if seq_len % stride != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by stride={stride}")
        self.stride = stride
        self.sub_len = seq_len // stride
        self.mlp = nn.Sequential(
            nn.Linear(self.sub_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.sub_len),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, C) -> (B, L, C), residual."""
        batch, seq_len, channels = x.shape
        # x.reshape(B, sub_len, stride, C): element at time t = i*stride + k lands at [i, k],
        # so slicing axis=2 at k recovers exactly x[:, k::stride, :] -- the k-th subsequence.
        x_sub = x.reshape(batch, self.sub_len, self.stride, channels)
        x_sub = x_sub.permute(0, 2, 3, 1)  # (B, stride, C, sub_len)
        # Linear applies to the last dim regardless of leading dims.
        mixed = cast(Tensor, self.mlp(x_sub))
        mixed = mixed.permute(0, 3, 1, 2).reshape(batch, seq_len, channels)
        return x + mixed


class ChannelMixing(nn.Module):
    """Low-rank channel mixing: project C channels down to a smaller rank r, mix, project back.

    Correlated markets (WTI/Brent, gold/silver, DXY and the FX majors) carry overlapping
    information rather than being fully independent sources; a full C x C mixing matrix doesn't
    exploit that, and this bottleneck does. Applied identically at every timestep.
    """

    def __init__(self, num_channels: int, rank: int) -> None:
        super().__init__()
        rank = max(1, min(rank, num_channels))
        self.down = nn.Linear(num_channels, rank)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, num_channels)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, C) -> (B, L, C), residual."""
        return x + cast(Tensor, self.up(self.act(self.down(x))))


class MTSMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        *,
        stride: int,
        temporal_hidden: int,
        channel_rank: int,
    ) -> None:
        super().__init__()
        self.temporal = TemporalMixing(seq_len, stride, temporal_hidden)
        self.channel = ChannelMixing(num_channels, channel_rank)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.channel(self.temporal(x)))


class MTSMixers(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_channels: int,
        *,
        stride: int = 2,
        temporal_hidden: int = 64,
        channel_rank: int = 4,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MTSMixerBlock(
                seq_len,
                num_channels,
                stride=stride,
                temporal_hidden=temporal_hidden,
                channel_rank=channel_rank,
            )
            for _ in range(num_blocks)
        )
        # Per-channel time projection head, same style as DLinear's forecast heads.
        self.head = nn.ModuleList(
            nn.Linear(seq_len, pred_len) for _ in range(num_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, C) -> (B, H, C)."""
        for block in self.blocks:
            x = block(x)
        return torch.stack(
            [head(x[:, :, c]) for c, head in enumerate(self.head)],
            dim=-1,
        )
