"""No Pattern — unstructured random-walk OHLC used as a negative training class."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, random_base_price, random_volume


PATTERN_NAME = "no_pattern"
WINDOW = 8


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    n = int(rng.integers(8, 16))
    candles: list[Candle] = []
    for _ in range(n):
        ret = rng.normal(0.0, 0.012)
        open_ = price
        close = max(0.01, price * (1.0 + ret))
        wick = abs(close - open_) * rng.uniform(0.2, 1.2) + price * rng.uniform(0.001, 0.004)
        candles.append(make_candle(open_, close, wick, wick, random_volume(rng)))
        price = close
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    # Negative class is never emitted by heuristic scans.
    return False
