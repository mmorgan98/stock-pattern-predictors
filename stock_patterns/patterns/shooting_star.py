"""Shooting Star — small body near the low with a long upper wick (bearish reversal)."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, body, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "shooting_star"
WINDOW = 1


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    small_body = price * rng.uniform(0.003, 0.01)
    upper_wick = small_body * rng.uniform(2.2, 4.5)
    lower_wick = small_body * rng.uniform(0.0, 0.9)

    open_ = price
    close = open_ - small_body
    if rng.random() < 0.35:
        open_, close = close, open_

    candle = make_candle(
        open_=open_,
        close=close,
        upper_wick=max(upper_wick, small_body * 2.05),
        lower_wick=min(lower_wick, small_body * 0.95),
        volume=random_volume(rng),
    )
    candles: list[Candle] = [candle]
    if with_context:
        candles = prepend_context(candles, rng, trend="up")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    c = df.iloc[-1]
    b = body(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    upper = c["high"] - max(c["open"], c["close"])
    if b <= 0:
        return False
    return bool(upper >= 2.0 * b and lower <= b)
