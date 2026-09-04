"""Turn a trained model's test-set predictions into price-scale forecasts and trading metrics,
sliced down to one target channel (NATGAS) regardless of how many channels the model saw.

PnL figures are illustrative only: spread_bps/slippage_bps are configurable placeholders, not a
calibrated NATGAS CFD cost model, and the "strategy" is just sign(1-step prediction) with no
position sizing, funding, or execution-latency modeling. Treat them as a sanity check on whether
a model's predictions carry directional information, not as a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mts_mixers_natgas.dataset import MultiScaler, WindowDataset
from mts_mixers_natgas.models import Forecaster


@dataclass(frozen=True)
class Predictions:
    """Price-scale predictions and ground truth for the target channel, every test window."""

    origin_index: pd.DatetimeIndex
    last_input: np.ndarray  # (N,) last observed close before the forecast horizon
    actual: np.ndarray  # (N, H)
    predicted: np.ndarray  # (N, H)
    input_window: np.ndarray  # (N, L), price scale, for volatility-regime bucketing


def predict(
    model: Forecaster,
    dataset: WindowDataset,
    scaler: MultiScaler,
    *,
    target_channel: str,
) -> Predictions:
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    predicted_batches = []
    with torch.no_grad():
        for x, _ in loader:
            predicted_batches.append(model(x).numpy())
    predicted_all = scaler.inverse_transform(
        np.concatenate(predicted_batches, axis=0)
    )  # (N,H,C)
    actual_all = scaler.inverse_transform(dataset.y)  # (N,H,C)
    input_all = scaler.inverse_transform(dataset.x)  # (N,L,C)

    channel = scaler.channel_index(target_channel)
    predicted = predicted_all[:, :, channel]
    actual = actual_all[:, :, channel]
    input_window = input_all[:, :, channel]
    return Predictions(
        origin_index=dataset.origin_index,
        last_input=input_window[:, -1],
        actual=actual,
        predicted=predicted,
        input_window=input_window,
    )


@dataclass(frozen=True)
class EvaluationReport:
    run_name: str
    mae: float
    mse: float
    direction_accuracy_1h: float
    direction_accuracy_mean: float
    return_correlation_1h: float
    top_decile_direction_accuracy_1h: float
    mean_pnl_bps_1h: float
    mean_pnl_after_costs_bps_1h: float
    by_volatility_regime: pd.DataFrame
    by_direction: pd.DataFrame


def evaluate(
    predictions: Predictions,
    *,
    run_name: str,
    spread_bps: float = 3.0,
    slippage_bps: float = 2.0,
) -> EvaluationReport:
    actual, predicted, last_input = (
        predictions.actual,
        predictions.predicted,
        predictions.last_input,
    )

    error = predicted - actual
    mae = float(np.abs(error).mean())
    mse = float((error**2).mean())

    predicted_return_1h = predicted[:, 0] - last_input
    actual_return_1h = actual[:, 0] - last_input
    direction_accuracy_1h = _direction_accuracy(predicted_return_1h, actual_return_1h)

    predicted_return_all = predicted - last_input[:, None]
    actual_return_all = actual - last_input[:, None]
    direction_accuracy_mean = _direction_accuracy(
        predicted_return_all.ravel(), actual_return_all.ravel()
    )

    return_correlation_1h = float(
        np.corrcoef(predicted_return_1h, actual_return_1h)[0, 1]
    )

    decile_cutoff = np.quantile(np.abs(predicted_return_1h), 0.9)
    top_decile_mask = np.abs(predicted_return_1h) >= decile_cutoff
    top_decile_direction_accuracy_1h = _direction_accuracy(
        predicted_return_1h[top_decile_mask], actual_return_1h[top_decile_mask]
    )

    position = np.sign(predicted_return_1h)
    actual_pct_return_1h = actual_return_1h / last_input
    gross_pnl_bps = position * actual_pct_return_1h * 10_000
    round_turn_cost_bps = spread_bps + slippage_bps
    net_pnl_bps = np.where(position != 0, gross_pnl_bps - round_turn_cost_bps, 0.0)

    by_volatility_regime = _by_volatility_regime(
        predictions, gross_pnl_bps, net_pnl_bps
    )
    by_direction = _by_direction(
        position, direction_correct=(position == np.sign(actual_return_1h))
    )

    return EvaluationReport(
        run_name=run_name,
        mae=mae,
        mse=mse,
        direction_accuracy_1h=direction_accuracy_1h,
        direction_accuracy_mean=direction_accuracy_mean,
        return_correlation_1h=return_correlation_1h,
        top_decile_direction_accuracy_1h=top_decile_direction_accuracy_1h,
        mean_pnl_bps_1h=float(gross_pnl_bps.mean()),
        mean_pnl_after_costs_bps_1h=float(net_pnl_bps.mean()),
        by_volatility_regime=by_volatility_regime,
        by_direction=by_direction,
    )


def _direction_accuracy(
    predicted_return: np.ndarray, actual_return: np.ndarray
) -> float:
    moved = actual_return != 0
    if not np.any(moved):
        return float("nan")
    correct = np.sign(predicted_return[moved]) == np.sign(actual_return[moved])
    return float(correct.mean())


def _by_volatility_regime(
    predictions: Predictions, gross_pnl_bps: np.ndarray, net_pnl_bps: np.ndarray
) -> pd.DataFrame:
    input_returns = np.diff(predictions.input_window, axis=1)
    realized_vol = input_returns.std(axis=1)
    regime = pd.qcut(realized_vol, q=3, labels=["low_vol", "mid_vol", "high_vol"])

    predicted_return_1h = predictions.predicted[:, 0] - predictions.last_input
    actual_return_1h = predictions.actual[:, 0] - predictions.last_input
    frame = pd.DataFrame(
        {
            "regime": regime,
            "direction_correct": np.sign(predicted_return_1h)
            == np.sign(actual_return_1h),
            "gross_pnl_bps": gross_pnl_bps,
            "net_pnl_bps": net_pnl_bps,
        }
    )
    return (
        frame.groupby("regime", observed=True)
        .agg(
            n=("direction_correct", "size"),
            direction_accuracy=("direction_correct", "mean"),
            mean_gross_pnl_bps=("gross_pnl_bps", "mean"),
            mean_net_pnl_bps=("net_pnl_bps", "mean"),
        )
        .reset_index()
    )


def _by_direction(
    position: np.ndarray, *, direction_correct: np.ndarray
) -> pd.DataFrame:
    labels = np.where(position > 0, "long", np.where(position < 0, "short", "flat"))
    frame = pd.DataFrame({"side": labels, "direction_correct": direction_correct})
    return (
        frame.groupby("side", observed=True)
        .agg(
            n=("direction_correct", "size"),
            direction_accuracy=("direction_correct", "mean"),
        )
        .reset_index()
    )
