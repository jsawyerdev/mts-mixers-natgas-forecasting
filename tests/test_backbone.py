from __future__ import annotations

import pytest
import torch

from mts_mixers_natgas.backbone import DLinear, MovingAverageDecomposition


def test_moving_average_decomposition_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        MovingAverageDecomposition(kernel_size=4)


def test_moving_average_decomposition_seasonal_plus_trend_equals_input() -> None:
    decomposition = MovingAverageDecomposition(kernel_size=5)
    x = torch.randn(2, 20, 3)

    trend, seasonal = decomposition(x)

    assert torch.allclose(trend + seasonal, x, atol=1e-5)
    assert trend.shape == x.shape


def test_dlinear_output_shape() -> None:
    model = DLinear(seq_len=96, pred_len=24, num_channels=1)
    x = torch.randn(8, 96, 1)

    output = model(x)

    assert output.shape == (8, 24, 1)


def test_dlinear_handles_short_sequences_without_crashing() -> None:
    model = DLinear(seq_len=6, pred_len=2, num_channels=1)
    x = torch.randn(2, 6, 1)

    output = model(x)

    assert output.shape == (2, 2, 1)
