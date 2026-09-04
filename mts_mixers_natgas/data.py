"""Dukascopy hourly candles for NATGAS and eight cross-market channels: fetch, cache, align,
and chronological splitting.

Every channel is fetched from Dukascopy, the same single provider used throughout, rather than
mixing in a second API (FRED for yields, a weather feed for storage) for a couple of extra
channels: DOLLAR.IDX/USD and VOL.IDX/USD stand in for a dollar index and a VIX-style volatility
gauge, and USTBOND.TR/USD stands in for a rates channel (a bond CFD, not literally a 10Y yield
series -- named accordingly rather than mislabeled).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import dukascopy_python
import pandas as pd
from dukascopy_python import instruments as di

LOGGER = logging.getLogger(__name__)

INTERVAL = dukascopy_python.INTERVAL_HOUR_1
OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# Column name -> Dukascopy instrument constant. "natgas" is always the forecast target.
INSTRUMENTS: dict[str, str] = {
    "natgas": di.INSTRUMENT_CMD_ENERGY_GAS_CMD_USD,
    "wti": di.INSTRUMENT_CMD_ENERGY_E_LIGHT,
    "brent": di.INSTRUMENT_CMD_ENERGY_E_BRENT,
    "gold": di.INSTRUMENT_FX_METALS_XAU_USD,
    "silver": di.INSTRUMENT_FX_METALS_XAG_USD,
    "dxy": di.INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD,
    "vix": di.INSTRUMENT_IDX_AMERICA_VOL_IDX_USD,
    "ustbond": di.INSTRUMENT_BND_CFD_USTBOND_TR_USD,
    "spx": di.INSTRUMENT_IDX_AMERICA_E_SANDP_500,
}

TARGET_CHANNEL = "natgas"

# The channel-ablation ladder from narrowest to widest -- see README.
CHANNEL_LADDER: dict[str, list[str]] = {
    "A_natgas_only": ["natgas"],
    "B_plus_wti": ["natgas", "wti"],
    "C_plus_brent": ["natgas", "wti", "brent"],
    "D_all_commodities": ["natgas", "wti", "brent", "gold", "silver"],
    "E_commodities_and_macro": [
        "natgas",
        "wti",
        "brent",
        "gold",
        "silver",
        "dxy",
        "vix",
        "ustbond",
        "spx",
    ],
}


def fetch_market_data(
    *,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return hourly close prices for every channel in INSTRUMENTS, aligned on a shared index.

    Each instrument keeps its own trading calendar (NATGAS and equities/FX don't share exact
    session hours), so this inner-joins on timestamp: only hours where every channel has a
    printed close survive. Forward-filling would invent prices for channels that didn't trade.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, start, end)
    if cache_path.exists() and not refresh:
        LOGGER.info("loading cached market data from %s", cache_path)
        return pd.read_parquet(cache_path)

    closes: dict[str, pd.Series] = {}
    for label, instrument in INSTRUMENTS.items():
        LOGGER.info(
            "fetching %s (%s) hourly candles %s -> %s", label, instrument, start, end
        )
        candles: pd.DataFrame = dukascopy_python.fetch(
            instrument, INTERVAL, OFFER_SIDE, start, end
        )
        candles.index = pd.to_datetime(candles.index, utc=True)
        close = candles["close"].sort_index()
        close = close[~close.index.duplicated(keep="last")]
        closes[label] = close
        LOGGER.info("%s: %d candles", label, len(close))

    frame = pd.DataFrame(closes).dropna(how="any")
    frame.index.name = "timestamp"
    if frame.empty:
        raise ValueError("no timestamps have a printed close for every channel")

    frame.to_parquet(cache_path)
    LOGGER.info(
        "aligned market data: %d rows x %d channels", len(frame), len(frame.columns)
    )
    return frame


@dataclass(frozen=True)
class ChronologicalSplit:
    """Train/val/test slices of a multivariate frame, split in time order."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_chronological(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> ChronologicalSplit:
    """Split a time-ordered frame into contiguous train/val/test blocks, oldest first."""
    if (
        not 0.0 < train_frac < 1.0
        or not 0.0 < val_frac < 1.0
        or train_frac + val_frac >= 1.0
    ):
        raise ValueError(
            "train_frac and val_frac must be in (0, 1) and sum to less than 1"
        )
    n = len(frame)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError(
            f"frame of length {n} is too short for the requested split fractions"
        )
    return ChronologicalSplit(
        train=frame.iloc[:train_end],
        val=frame.iloc[train_end:val_end],
        test=frame.iloc[val_end:],
    )


def _cache_path(cache_dir: Path, start: datetime, end: datetime) -> Path:
    start_key = start.astimezone(UTC).strftime("%Y%m%d")
    end_key = end.astimezone(UTC).strftime("%Y%m%d")
    channels_key = "-".join(INSTRUMENTS)
    return cache_dir / f"market_1h_{channels_key}_{start_key}_{end_key}.parquet"
