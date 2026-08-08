"""Shared helpers for synthetic OHLC candlestick generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 1_000_000.0

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.open, self.high, self.low, self.close, self.volume)


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_base_price(rng: np.random.Generator, low: float = 20.0, high: float = 400.0) -> float:
    return float(rng.uniform(low, high))


def random_volume(rng: np.random.Generator, mean: float = 1_500_000.0) -> float:
    return float(max(10_000.0, rng.lognormal(mean=np.log(mean), sigma=0.35)))


def body(open_: float, close: float) -> float:
    return abs(close - open_)


def make_candle(
    open_: float,
    close: float,
    upper_wick: float,
    lower_wick: float,
    volume: float,
) -> Candle:
    high = max(open_, close) + max(0.0, upper_wick)
    low = min(open_, close) - max(0.0, lower_wick)
    # Numerical safety: keep high/low consistent.
    high = max(high, open_, close)
    low = min(low, open_, close)
    return Candle(open=open_, high=high, low=low, close=close, volume=volume)


def candles_to_dataframe(candles: Iterable[Candle], start: str = "2020-01-01") -> pd.DataFrame:
    rows = [c.as_tuple() for c in candles]
    idx = pd.bdate_range(start=start, periods=len(rows))
    df = pd.DataFrame(rows, columns=OHLC_COLUMNS, index=idx)
    df.index.name = "date"
    return df


def prepend_context(
    pattern_candles: list[Candle],
    rng: np.random.Generator,
    context_len: int = 8,
    trend: str = "down",
) -> list[Candle]:
    """Build a short leading context trend before the pattern candles."""
    if not pattern_candles:
        return []

    first = pattern_candles[0]
    price = first.open
    step = price * rng.uniform(0.004, 0.012)
    context: list[Candle] = []

    for i in range(context_len, 0, -1):
        noise = rng.uniform(0.3, 1.2)
        if trend == "down":
            open_ = price + step * i * noise
            close = open_ - step * rng.uniform(0.4, 1.1)
        elif trend == "up":
            open_ = price - step * i * noise
            close = open_ + step * rng.uniform(0.4, 1.1)
        else:
            open_ = price + rng.uniform(-step, step)
            close = open_ + rng.uniform(-step, step)

        wick = abs(close - open_) * rng.uniform(0.1, 0.6)
        context.append(
            make_candle(
                open_=open_,
                close=close,
                upper_wick=wick,
                lower_wick=wick,
                volume=random_volume(rng),
            )
        )

    return context + pattern_candles


def normalize_sequence(df: pd.DataFrame) -> np.ndarray:
    """Return relative OHLC features scaled by the first close."""
    values = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    base = values[0, 3]
    if base == 0:
        base = 1.0
    return (values / base).astype(np.float32)
