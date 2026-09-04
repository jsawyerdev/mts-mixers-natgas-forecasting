"""Figures for the README: cross-market overview, return-correlation heatmap, the baseline
comparison (DLinear/RevIN x DLinear/MTS-Mixers), the channel-ablation ladder, and one test-set
forecast comparison.

Colors follow the dataviz skill's validated categorical palette and its sequential blue ramp.
The four baseline configurations get stable categorical colors (an unordered comparison); the
five-rung channel ladder gets a light-to-dark sequential ramp, because it's an ordered
progression (narrower to wider channel set), not a set of unordered categories.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RUN_COLORS = {
    "dlinear": "#2a78d6",
    "dlinear_revin": "#eb6834",
    "mtsmixers": "#1baf7a",
    "mtsmixers_revin": "#eda100",
}
RUN_LABELS = {
    "dlinear": "DLinear",
    "dlinear_revin": "DLinear + RevIN",
    "mtsmixers": "MTS-Mixers",
    "mtsmixers_revin": "MTS-Mixers + RevIN",
}
LADDER_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_cross_market_overview(
    frame: pd.DataFrame, split_points: tuple[pd.Timestamp, pd.Timestamp], path: Path
) -> None:
    """NATGAS vs. WTI vs. DXY, each indexed to 100 at the start, to show what the channel-mixer
    has to work with: energy co-movement and a mostly-independent dollar channel."""
    shown = {"natgas": "#2a78d6", "wti": "#eb6834", "dxy": "#eda100"}
    fig, ax = _new_figure((10, 4))
    for column, color in shown.items():
        indexed = frame[column] / frame[column].iloc[0] * 100
        ax.plot(
            indexed.index,
            indexed.to_numpy(),
            color=color,
            linewidth=1.3,
            label=column.upper(),
        )
    for boundary in split_points:
        ax.axvline(
            cast(float, boundary), color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3))
        )
    ax.set_title(
        "NATGAS, WTI, and DXY, indexed to 100",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )
    ax.set_ylabel("indexed level", color=INK_SECONDARY, fontsize=9)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _save(fig, path)


def plot_correlation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    """Pearson correlation of hourly returns across every channel."""
    returns = frame.pct_change().dropna()
    corr = returns.corr()
    labels = [c.upper() for c in corr.columns]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
        color=INK_SECONDARY,
        fontsize=8,
    )
    ax.set_yticks(range(len(labels)), labels, color=INK_SECONDARY, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.to_numpy()[i, j]
            text_color = INK_PRIMARY if abs(value) < 0.6 else SURFACE
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    ax.set_title(
        "Hourly return correlation, all channels",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
        pad=12,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    _save(fig, path)


def plot_baseline_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """Grouped bars: MAE, 1h direction accuracy, 1h return correlation across the four
    DLinear/RevIN x DLinear/MTS-Mixers baseline configurations."""
    columns = ["mae", "direction_accuracy_1h", "return_correlation_1h"]
    titles = [
        "MAE (USD, lower is better)",
        "1h direction accuracy",
        "1h return correlation",
    ]
    runs = list(metrics["run"])
    colors = [RUN_COLORS[r] for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150, facecolor=SURFACE)
    for ax, column, title in zip(axes, columns, titles, strict=True):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="y")
        ax.set_axisbelow(True)
        bars = ax.bar(
            [RUN_LABELS[r] for r in runs], metrics[column], color=colors, width=0.6
        )
        ax.bar_label(bars, fmt="%.3f", color=INK_SECONDARY, fontsize=8, padding=2)
        ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
        ax.tick_params(axis="x", rotation=18)
    _save(fig, path)


def plot_ablation_ladder(metrics: pd.DataFrame, path: Path) -> None:
    """MAE and 1h direction accuracy across the five-rung channel ladder (A: NATGAS-only through
    E: NATGAS + commodities + macro), sequential ramp light-to-dark for the ordered progression.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, facecolor=SURFACE)
    labels = [
        name.split("_", 1)[1].replace("_", " ") for name in metrics["ladder_step"]
    ]
    for ax, column, title in zip(
        axes,
        ["mae", "direction_accuracy_1h"],
        ["MAE (USD, lower is better)", "1h direction accuracy"],
        strict=True,
    ):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="y")
        ax.set_axisbelow(True)
        bars = ax.bar(
            labels, metrics[column], color=LADDER_RAMP[: len(labels)], width=0.6
        )
        ax.bar_label(bars, fmt="%.3f", color=INK_SECONDARY, fontsize=8, padding=2)
        ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_forecast_comparison(
    input_window: np.ndarray,
    actual_future: np.ndarray,
    predicted_future: dict[str, np.ndarray],
    colors: dict[str, str],
    labels: dict[str, str],
    path: Path,
) -> None:
    seq_len, pred_len = len(input_window), len(actual_future)
    fig, ax = _new_figure((9, 4.5))
    history_x = np.arange(-seq_len, 0)
    future_x = np.arange(0, pred_len)

    ax.plot(
        history_x,
        input_window,
        color=INK_SECONDARY,
        linewidth=1.4,
        label="observed history",
    )
    ax.plot(
        np.concatenate([[history_x[-1]], future_x]),
        np.concatenate([[input_window[-1]], actual_future]),
        color=INK_PRIMARY,
        linewidth=2.0,
        label="actual",
    )
    for name, values in predicted_future.items():
        ax.plot(
            np.concatenate([[history_x[-1]], future_x]),
            np.concatenate([[input_window[-1]], values]),
            color=colors[name],
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label=labels[name],
        )
    ax.axvline(0, color=BASELINE, linewidth=1.0)
    ax.set_title(
        "Forecast comparison on one test-set window",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )
    ax.set_xlabel("hours relative to forecast origin", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("USD", color=INK_SECONDARY, fontsize=9)
    legend = ax.legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _save(fig, path)
