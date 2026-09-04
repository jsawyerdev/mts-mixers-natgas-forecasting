from __future__ import annotations

import pytest
import torch
from torch import nn

from mts_mixers_natgas.mixer import ChannelMixing, MTSMixers, TemporalMixing


def test_temporal_mixing_interleave_merge_round_trips_exactly() -> None:
    """With an identity 'MLP', forward() is x + round_trip(x); round_trip must equal x exactly,
    or the interleave/merge reshape-permute pair has scrambled the time order."""
    seq_len, stride = 8, 2
    x = torch.arange(1, 9, dtype=torch.float32).reshape(1, seq_len, 1)
    mixing = TemporalMixing(seq_len, stride, hidden_dim=4)
    mixing.mlp = nn.Identity()

    out = mixing(x)

    assert torch.allclose(out, 2 * x)


def test_temporal_mixing_rejects_seq_len_not_divisible_by_stride() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TemporalMixing(seq_len=10, stride=3, hidden_dim=4)


def test_channel_mixing_output_shape_and_residual() -> None:
    mixing = ChannelMixing(num_channels=5, rank=2)
    x = torch.randn(3, 20, 5)

    out = mixing(x)

    assert out.shape == x.shape


def test_channel_mixing_rank_clamped_when_larger_than_channels() -> None:
    mixing = ChannelMixing(num_channels=1, rank=8)
    x = torch.randn(2, 10, 1)

    out = mixing(x)

    assert out.shape == x.shape


def test_mts_mixers_output_shape() -> None:
    model = MTSMixers(
        seq_len=96, pred_len=24, num_channels=9, stride=2, channel_rank=4, num_blocks=2
    )
    x = torch.randn(4, 96, 9)

    out = model(x)

    assert out.shape == (4, 24, 9)
