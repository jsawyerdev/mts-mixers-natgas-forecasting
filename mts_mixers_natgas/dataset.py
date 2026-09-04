"""Sliding-window (x, y) pairs over a multivariate price frame, with a per-channel scaler fit
on the train split only (same train-only-scaling discipline as the FAN companion repo, for the
same reason: fitting on train+val+test would leak test-set statistics into training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MultiScaler:
    """Per-channel mean/std computed on the training split, applied to every split."""

    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)
    columns: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, (values - self.mean) / self.std)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, values * self.std + self.mean)

    def channel_index(self, name: str) -> int:
        return self.columns.index(name)


def fit_scaler(train_frame: pd.DataFrame) -> MultiScaler:
    # float32, matching the dtype WindowDataset casts window values to: mixing float64 stats
    # into a float32 array promotes the whole array to float64, and torch's default Linear
    # weights are float32, so that mismatch would crash training with a dtype error.
    mean = train_frame.mean().to_numpy(dtype=np.float32)
    std = train_frame.std().to_numpy(dtype=np.float32)
    if np.any(std <= 0):
        zero_var = [c for c, s in zip(train_frame.columns, std, strict=True) if s <= 0]
        raise ValueError(f"train split has zero variance in channels: {zero_var}")
    return MultiScaler(mean=mean, std=std, columns=tuple(train_frame.columns))


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """seq_len-in / pred_len-out windows over a scaled multivariate frame, all channels."""

    def __init__(
        self, frame: pd.DataFrame, seq_len: int, pred_len: int, scaler: MultiScaler
    ) -> None:
        if tuple(frame.columns) != scaler.columns:
            raise ValueError(
                "frame columns do not match the columns the scaler was fit on"
            )
        values = scaler.transform(frame.to_numpy(dtype=np.float32))  # (T, C)
        n_windows = len(values) - seq_len - pred_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"frame of length {len(values)} is too short for seq_len={seq_len} + "
                f"pred_len={pred_len}"
            )
        x_view = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)[
            :n_windows
        ]
        y_view = np.lib.stride_tricks.sliding_window_view(
            values[seq_len:], pred_len, axis=0
        )[:n_windows]
        # sliding_window_view(..., axis=0) appends the window as the last axis: (N, C, L) ->
        # move it to (N, L, C) to match the (batch, time, channel) convention used everywhere.
        self.x = np.ascontiguousarray(x_view.transpose(0, 2, 1))
        self.y = np.ascontiguousarray(y_view.transpose(0, 2, 1))
        # Timestamp of the last input bar in each window -- the forecast's origin.
        self.origin_index = pd.DatetimeIndex(
            frame.index[seq_len - 1 : seq_len - 1 + n_windows]
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
