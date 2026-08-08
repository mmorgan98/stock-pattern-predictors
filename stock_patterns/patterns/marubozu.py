"""Marubozu — long body with little-to-no wicks (bullish or bearish variant)."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, body, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "marubozu"
WINDOW = 1


def generate(seed: int | None = None, with_context: bool = True, direction: str | None = None) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    long_body = price * rng.uniform(0.02, 0.045)
    direction = direction or ("bullish" if rng.random() < 0.5 else "bearish")

    if direction == "bullish":
        open_, close = price, price + long_body
        trend = "down"
    else:
        open_, close = price, price - long_body
        trend = "up"

    candle = make_candle(
        open_=open_,
        close=close,
        upper_wick=long_body * rng.uniform(0.0, 0.04),
        lower_wick=long_body * rng.uniform(0.0, 0.04),
        volume=random_volume(rng, mean=2_500_000),
    )
    candles: list[Candle] = [candle]
    if with_context:
        candles = prepend_context(candles, rng, trend=trend)
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    c = df.iloc[-1]
    b = body(c["open"], c["close"])
    span = c["high"] - c["low"]
    if span <= 0 or b <= 0:
        return False
    return bool(b / span >= 0.9)
