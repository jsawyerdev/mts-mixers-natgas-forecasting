"""End-to-end experiment: fetch NATGAS + 8 cross-market channels from Dukascopy, run the
four-way baseline comparison (DLinear/RevIN x DLinear/MTS-Mixers), then run the five-rung
channel-ablation ladder (A: NATGAS-only through E: NATGAS + commodities + macro) with whichever
MTS-Mixers configuration won the baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from mts_mixers_natgas.data import (
    CHANNEL_LADDER,
    TARGET_CHANNEL,
    ChronologicalSplit,
    fetch_market_data,
    split_chronological,
)
from mts_mixers_natgas.dataset import MultiScaler, WindowDataset, fit_scaler
from mts_mixers_natgas.evaluate import EvaluationReport, Predictions, evaluate, predict
from mts_mixers_natgas.models import Backbone, Forecaster, Strategy
from mts_mixers_natgas.train import TrainConfig, train_forecaster
from mts_mixers_natgas.visualize import (
    plot_ablation_ladder,
    plot_baseline_comparison,
    plot_correlation_heatmap,
    plot_cross_market_overview,
    plot_forecast_comparison,
)

LOGGER = logging.getLogger(__name__)


def _baseline_runs(
    all_columns: list[str],
) -> list[tuple[str, Backbone, Strategy, list[str]]]:
    """The four-way comparison: DLinear only ever sees NATGAS (its per-channel heads never mix
    across channels, so feeding it the other 8 would change nothing about its NATGAS forecast
    while wasting compute); MTS-Mixers sees every channel, since testing its channel-mixing
    block on the full cross-market set is the whole point of comparing it to DLinear here.
    """
    natgas_only = [TARGET_CHANNEL]
    return [
        ("dlinear", Backbone.DLINEAR, Strategy.RAW, natgas_only),
        ("dlinear_revin", Backbone.DLINEAR, Strategy.REVIN, natgas_only),
        ("mtsmixers", Backbone.MTSMIXERS, Strategy.RAW, all_columns),
        ("mtsmixers_revin", Backbone.MTSMIXERS, Strategy.REVIN, all_columns),
    ]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--history-days", type=int, default=730)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--spread-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--mixer-stride", type=int, default=2)
    parser.add_argument("--mixer-channel-rank", type=int, default=4)
    parser.add_argument("--mixer-num-blocks", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    torch.manual_seed(0)

    end = datetime.now(UTC)
    start = end - timedelta(days=args.history_days)
    frame = fetch_market_data(
        start=start, end=end, cache_dir=args.data_dir, refresh=args.refresh_data
    )
    LOGGER.info(
        "loaded %d aligned hourly rows across %d channels (%s -> %s)",
        len(frame),
        len(frame.columns),
        frame.index[0],
        frame.index[-1],
    )
    all_columns = list(frame.columns)
    split = split_chronological(frame)
    LOGGER.info(
        "split train=%d val=%d test=%d rows",
        len(split.train),
        len(split.val),
        len(split.test),
    )

    train_config = TrainConfig(epochs=args.epochs)

    baseline_reports: dict[str, EvaluationReport] = {}
    baseline_predictions: dict[str, Predictions] = {}
    for name, backbone, strategy, columns in _baseline_runs(all_columns):
        report, preds = _run(
            name, backbone, strategy, columns, split, args, train_config
        )
        baseline_reports[name] = report
        baseline_predictions[name] = preds

    winning_mixer_strategy = (
        Strategy.RAW
        if baseline_reports["mtsmixers"].mse <= baseline_reports["mtsmixers_revin"].mse
        else Strategy.REVIN
    )
    LOGGER.info(
        "channel ladder will use MTS-Mixers + strategy=%s", winning_mixer_strategy.value
    )

    ladder_reports: dict[str, EvaluationReport] = {}
    ladder_predictions: dict[str, Predictions] = {}
    for step_name, columns in CHANNEL_LADDER.items():
        report, preds = _run(
            step_name,
            Backbone.MTSMIXERS,
            winning_mixer_strategy,
            columns,
            split,
            args,
            train_config,
        )
        ladder_reports[step_name] = report
        ladder_predictions[step_name] = preds

    _print_comparison("Baseline comparison", baseline_reports)
    _print_comparison("Channel-ablation ladder", ladder_reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_results(baseline_reports, ladder_reports, args.output_dir / "results.json")

    LOGGER.info("writing figures to %s", args.figures_dir)
    _write_figures(
        frame,
        split,
        baseline_reports,
        baseline_predictions,
        ladder_reports,
        ladder_predictions,
        args,
    )


def _run(
    run_name: str,
    backbone: Backbone,
    strategy: Strategy,
    columns: list[str],
    split: ChronologicalSplit,
    args: argparse.Namespace,
    train_config: TrainConfig,
) -> tuple[EvaluationReport, Predictions]:
    LOGGER.info(
        "training run=%s backbone=%s strategy=%s channels=%s",
        run_name,
        backbone.value,
        strategy.value,
        columns,
    )
    scaler, train_ds, val_ds, test_ds = _build_datasets(
        split, columns, args.seq_len, args.pred_len
    )
    model = Forecaster(
        backbone,
        strategy,
        args.seq_len,
        args.pred_len,
        len(columns),
        mixer_stride=args.mixer_stride,
        mixer_channel_rank=args.mixer_channel_rank,
        mixer_num_blocks=args.mixer_num_blocks,
    )
    result = train_forecaster(model, train_ds, val_ds, train_config, run_name=run_name)
    preds = predict(result.model, test_ds, scaler, target_channel=TARGET_CHANNEL)
    report = evaluate(
        preds,
        run_name=run_name,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )
    return report, preds


def _build_datasets(
    split: ChronologicalSplit, columns: list[str], seq_len: int, pred_len: int
) -> tuple[MultiScaler, WindowDataset, WindowDataset, WindowDataset]:
    train_frame = split.train[columns]
    val_frame = split.val[columns]
    test_frame = split.test[columns]
    scaler = fit_scaler(train_frame)
    train_ds = WindowDataset(train_frame, seq_len, pred_len, scaler)
    val_ds = WindowDataset(val_frame, seq_len, pred_len, scaler)
    test_ds = WindowDataset(test_frame, seq_len, pred_len, scaler)
    return scaler, train_ds, val_ds, test_ds


def _print_comparison(title: str, reports: dict[str, EvaluationReport]) -> None:
    table = Table(title=title)
    table.add_column("run")
    table.add_column("MAE", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("1h dir. acc.", justify="right")
    table.add_column("1h return corr.", justify="right")
    table.add_column("mean PnL after costs (bps)", justify="right")
    for report in reports.values():
        table.add_row(
            report.run_name,
            f"{report.mae:.4f}",
            f"{report.mse:.4f}",
            f"{report.direction_accuracy_1h:.3f}",
            f"{report.return_correlation_1h:.3f}",
            f"{report.mean_pnl_after_costs_bps_1h:.2f}",
        )
    Console().print(table)


def _write_results(
    baseline_reports: dict[str, EvaluationReport],
    ladder_reports: dict[str, EvaluationReport],
    path: Path,
) -> None:
    def _serialize(reports: dict[str, EvaluationReport]) -> dict[str, object]:
        return {
            name: {
                "mae": r.mae,
                "mse": r.mse,
                "direction_accuracy_1h": r.direction_accuracy_1h,
                "direction_accuracy_mean": r.direction_accuracy_mean,
                "return_correlation_1h": r.return_correlation_1h,
                "top_decile_direction_accuracy_1h": r.top_decile_direction_accuracy_1h,
                "mean_pnl_bps_1h": r.mean_pnl_bps_1h,
                "mean_pnl_after_costs_bps_1h": r.mean_pnl_after_costs_bps_1h,
                "by_volatility_regime": r.by_volatility_regime.to_dict(
                    orient="records"
                ),
                "by_direction": r.by_direction.to_dict(orient="records"),
            }
            for name, r in reports.items()
        }

    payload = {
        "baseline": _serialize(baseline_reports),
        "channel_ladder": _serialize(ladder_reports),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote results to %s", path)


def _write_figures(
    frame: pd.DataFrame,
    split: ChronologicalSplit,
    baseline_reports: dict[str, EvaluationReport],
    baseline_predictions: dict[str, Predictions],
    ladder_reports: dict[str, EvaluationReport],
    ladder_predictions: dict[str, Predictions],
    args: argparse.Namespace,
) -> None:
    figures_dir = args.figures_dir
    plot_cross_market_overview(
        frame,
        (split.train.index[-1], split.val.index[-1]),
        figures_dir / "cross_market_overview.png",
    )
    plot_correlation_heatmap(frame, figures_dir / "correlation_heatmap.png")

    baseline_frame = pd.DataFrame(
        {
            "run": list(baseline_reports),
            "mae": [baseline_reports[r].mae for r in baseline_reports],
            "direction_accuracy_1h": [
                baseline_reports[r].direction_accuracy_1h for r in baseline_reports
            ],
            "return_correlation_1h": [
                baseline_reports[r].return_correlation_1h for r in baseline_reports
            ],
        }
    )
    plot_baseline_comparison(baseline_frame, figures_dir / "baseline_comparison.png")

    ladder_frame = pd.DataFrame(
        {
            "ladder_step": list(ladder_reports),
            "mae": [ladder_reports[r].mae for r in ladder_reports],
            "direction_accuracy_1h": [
                ladder_reports[r].direction_accuracy_1h for r in ladder_reports
            ],
        }
    )
    plot_ablation_ladder(ladder_frame, figures_dir / "ablation_ladder.png")

    sample_idx = len(baseline_predictions["dlinear"].actual) // 2
    widest_ladder_step = list(ladder_predictions)[-1]
    predicted_future = {
        "dlinear": baseline_predictions["dlinear"].predicted[sample_idx],
        widest_ladder_step: ladder_predictions[widest_ladder_step].predicted[
            sample_idx
        ],
    }
    colors = {"dlinear": "#2a78d6", widest_ladder_step: "#1baf7a"}
    labels = {
        "dlinear": "DLinear (NATGAS-only)",
        widest_ladder_step: "MTS-Mixers (all channels)",
    }
    plot_forecast_comparison(
        baseline_predictions["dlinear"].input_window[sample_idx],
        baseline_predictions["dlinear"].actual[sample_idx],
        predicted_future,
        colors,
        labels,
        figures_dir / "forecast_comparison.png",
    )


if __name__ == "__main__":
    main()
