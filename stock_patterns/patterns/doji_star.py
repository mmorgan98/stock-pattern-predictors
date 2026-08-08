"""Doji Star — open ≈ close with wicks on both sides (indecision / reversal hint)."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, body, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "doji_star"
WINDOW = 1


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    tiny = price * rng.uniform(0.0002, 0.0025)
    wick = price * rng.uniform(0.008, 0.025)

    open_ = price
    close = open_ + rng.uniform(-tiny, tiny)
    candle = make_candle(
        open_=open_,
        close=close,
        upper_wick=wick * rng.uniform(0.7, 1.3),
        lower_wick=wick * rng.uniform(0.7, 1.3),
        volume=random_volume(rng),
    )
    candles: list[Candle] = [candle]
    if with_context:
        trend = "up" if rng.random() < 0.5 else "down"
        candles = prepend_context(candles, rng, trend=trend)
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    c = df.iloc[-1]
    rng_span = c["high"] - c["low"]
    if rng_span <= 0:
        return False
    b = body(c["open"], c["close"])
    return bool(b / rng_span <= 0.12)
