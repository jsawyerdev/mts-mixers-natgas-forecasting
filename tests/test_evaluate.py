from __future__ import annotations

import numpy as np
import pandas as pd

from mts_mixers_natgas.evaluate import Predictions, evaluate


def _predictions(
    actual: np.ndarray, predicted: np.ndarray, last_input: np.ndarray
) -> Predictions:
    n, seq_len = len(last_input), 8
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # A random walk per row, not a constant, so realized volatility differs across samples
    # and evaluate()'s pd.qcut tercile split has distinct bin edges to work with.
    rng = np.random.default_rng(0)
    steps = rng.normal(scale=0.5, size=(n, seq_len)).cumsum(axis=1)
    input_window = last_input[:, None] + steps
    return Predictions(
        origin_index=index,
        last_input=last_input,
        actual=actual,
        predicted=predicted,
        input_window=input_window,
    )


def test_perfect_directional_predictions_score_full_accuracy() -> None:
    last_input = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([[101.0], [99.0], [102.0], [98.0]])
    predicted = np.array(
        [[101.5], [98.5], [103.0], [97.0]]
    )  # same sign as actual, every row

    report = evaluate(_predictions(actual, predicted, last_input), run_name="test")

    assert report.direction_accuracy_1h == 1.0
    assert report.mae > 0


def test_inverted_directional_predictions_score_zero_accuracy() -> None:
    last_input = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([[101.0], [99.0], [102.0], [98.0]])
    predicted = np.array([[99.0], [101.0], [98.0], [102.0]])  # opposite sign, every row

    report = evaluate(_predictions(actual, predicted, last_input), run_name="test")

    assert report.direction_accuracy_1h == 0.0
    assert report.mean_pnl_bps_1h < 0


def test_evaluation_report_fields_are_finite_on_a_larger_sample() -> None:
    rng = np.random.default_rng(0)
    n = 200
    last_input = 100 + rng.normal(size=n)
    actual = last_input[:, None] + rng.normal(scale=1.0, size=(n, 4))
    predicted = last_input[:, None] + rng.normal(scale=1.0, size=(n, 4))

    report = evaluate(_predictions(actual, predicted, last_input), run_name="test")

    assert 0.0 <= report.direction_accuracy_1h <= 1.0
    assert 0.0 <= report.top_decile_direction_accuracy_1h <= 1.0
    assert np.isfinite(report.mae)
    assert np.isfinite(report.mse)
    assert not report.by_volatility_regime.empty
    assert not report.by_direction.empty
