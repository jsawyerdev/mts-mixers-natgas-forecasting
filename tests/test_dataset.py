from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mts_mixers_natgas.dataset import WindowDataset, fit_scaler


def _frame(n: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "natgas": np.arange(n, dtype=np.float64),
            "wti": np.arange(n, dtype=np.float64) * 10,
        },
        index=index,
    )


def test_scaler_is_per_channel_and_fit_on_the_given_frame_only() -> None:
    train = _frame(100)

    scaler = fit_scaler(train)

    assert scaler.columns == ("natgas", "wti")
    assert np.allclose(scaler.mean, train.mean().to_numpy())
    assert np.allclose(scaler.std, train.std().to_numpy())
    assert scaler.channel_index("wti") == 1


def test_scaler_round_trips() -> None:
    train = _frame(50)
    scaler = fit_scaler(train)

    values = train.to_numpy()
    assert np.allclose(scaler.inverse_transform(scaler.transform(values)), values)


def test_window_dataset_windows_are_contiguous_aligned_and_multichannel() -> None:
    frame = _frame(30)
    scaler = fit_scaler(frame)

    dataset = WindowDataset(frame, seq_len=5, pred_len=2, scaler=scaler)

    assert len(dataset) == 30 - 5 - 2 + 1
    x0, y0 = dataset[0]
    assert x0.shape == (5, 2)
    assert y0.shape == (2, 2)
    unscaled_x0 = scaler.inverse_transform(x0.numpy())
    unscaled_y0 = scaler.inverse_transform(y0.numpy())
    assert np.allclose(unscaled_x0[:, 0], [0, 1, 2, 3, 4])  # natgas column
    assert np.allclose(unscaled_x0[:, 1], [0, 10, 20, 30, 40])  # wti column
    assert np.allclose(unscaled_y0[:, 0], [5, 6])
    assert dataset.origin_index[0] == frame.index[4]


def test_window_dataset_raises_when_frame_too_short() -> None:
    frame = _frame(5)
    scaler = fit_scaler(frame)

    with pytest.raises(ValueError, match="too short"):
        WindowDataset(frame, seq_len=5, pred_len=2, scaler=scaler)


def test_window_dataset_rejects_column_mismatch() -> None:
    frame = _frame(30)
    scaler = fit_scaler(frame)

    with pytest.raises(ValueError, match="columns"):
        WindowDataset(frame[["natgas"]], seq_len=5, pred_len=2, scaler=scaler)
