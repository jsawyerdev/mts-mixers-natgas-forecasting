from __future__ import annotations

import pytest
import torch

from mts_mixers_natgas.models import Backbone, Forecaster, Strategy


@pytest.mark.parametrize("backbone", [Backbone.DLINEAR, Backbone.MTSMIXERS])
@pytest.mark.parametrize("strategy", [Strategy.RAW, Strategy.REVIN])
def test_forecaster_output_shape(backbone: Backbone, strategy: Strategy) -> None:
    seq_len, pred_len, num_channels = 96, 24, 5
    model = Forecaster(backbone, strategy, seq_len, pred_len, num_channels)
    x = torch.randn(3, seq_len, num_channels)

    out = model(x)

    assert out.shape == (3, pred_len, num_channels)


def test_forecaster_single_channel_natgas_only() -> None:
    model = Forecaster(
        Backbone.DLINEAR, Strategy.REVIN, seq_len=96, pred_len=24, num_channels=1
    )
    x = torch.randn(2, 96, 1)

    out = model(x)

    assert out.shape == (2, 24, 1)
