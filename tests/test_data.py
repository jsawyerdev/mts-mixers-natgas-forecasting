from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mts_mixers_natgas.data import split_chronological


def _frame(n: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"natgas": np.arange(n, dtype=np.float64)}, index=index)


def test_split_chronological_keeps_time_order_and_covers_the_frame() -> None:
    frame = _frame(1000)

    split = split_chronological(frame, train_frac=0.7, val_frac=0.15)

    assert len(split.train) + len(split.val) + len(split.test) == len(frame)
    assert split.train.index[-1] < split.val.index[0]
    assert split.val.index[-1] < split.test.index[0]
    assert len(split.train) == pytest.approx(700, abs=1)


def test_split_chronological_rejects_fractions_summing_past_one() -> None:
    frame = _frame(100)

    with pytest.raises(ValueError, match="train_frac"):
        split_chronological(frame, train_frac=0.7, val_frac=0.4)
