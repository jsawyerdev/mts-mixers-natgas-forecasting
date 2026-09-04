"""Forecaster: a DLinear or MTS-Mixers backbone, optionally wrapped in RevIN."""

from __future__ import annotations

from enum import StrEnum
from typing import cast

from torch import Tensor, nn

from mts_mixers_natgas.backbone import DLinear
from mts_mixers_natgas.mixer import MTSMixers
from mts_mixers_natgas.norm import RevIN


class Backbone(StrEnum):
    DLINEAR = "dlinear"
    MTSMIXERS = "mtsmixers"


class Strategy(StrEnum):
    RAW = "raw"
    REVIN = "revin"


class Forecaster(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        strategy: Strategy,
        seq_len: int,
        pred_len: int,
        num_channels: int,
        *,
        mixer_stride: int = 2,
        mixer_channel_rank: int = 4,
        mixer_num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.body: nn.Module
        if backbone == Backbone.DLINEAR:
            self.body = DLinear(seq_len, pred_len, num_channels)
        else:
            self.body = MTSMixers(
                seq_len,
                pred_len,
                num_channels,
                stride=mixer_stride,
                channel_rank=mixer_channel_rank,
                num_blocks=mixer_num_blocks,
            )
        self.revin = RevIN(num_channels) if strategy == Strategy.REVIN else None

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, C) -> (B, pred_len, C)."""
        if self.strategy == Strategy.RAW:
            return cast(Tensor, self.body(x))
        assert self.revin is not None
        x_norm, stats = self.revin.normalize(x)
        y_norm = cast(Tensor, self.body(x_norm))
        return self.revin.denormalize(y_norm, stats)
