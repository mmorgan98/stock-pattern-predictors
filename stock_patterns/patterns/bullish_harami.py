"""Bullish Harami — large bearish candle containing a smaller bullish body."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "bullish_harami"
WINDOW = 2


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body1 = price * rng.uniform(0.02, 0.04)

    c1_open = price
    c1_close = price - body1
    c1 = make_candle(
        open_=c1_open,
        close=c1_close,
        upper_wick=body1 * rng.uniform(0.05, 0.3),
        lower_wick=body1 * rng.uniform(0.1, 0.4),
        volume=random_volume(rng),
    )

    inner = body1 * rng.uniform(0.2, 0.45)
    c2_open = c1_close + body1 * rng.uniform(0.15, 0.35)
    c2_close = c2_open + inner
    c2_close = min(c2_close, c1_open - body1 * 0.08)
    c2 = make_candle(
        open_=c2_open,
        close=c2_close,
        upper_wick=inner * rng.uniform(0.05, 0.4),
        lower_wick=inner * rng.uniform(0.05, 0.4),
        volume=random_volume(rng, mean=1_100_000),
    )

    candles: list[Candle] = [c1, c2]
    if with_context:
        candles = prepend_context(candles, rng, trend="down")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    return bool(
        a["close"] < a["open"]
        and b["close"] > b["open"]
        and b["open"] > a["close"]
        and b["close"] < a["open"]
    )
