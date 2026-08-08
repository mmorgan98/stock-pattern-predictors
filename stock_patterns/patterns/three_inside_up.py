"""Three Inside Up — bearish candle, bullish harami, then confirmation up-close."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "three_inside_up"
WINDOW = 3


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body1 = price * rng.uniform(0.018, 0.035)

    c1_open = price
    c1_close = price - body1
    c1 = make_candle(
        open_=c1_open,
        close=c1_close,
        upper_wick=body1 * rng.uniform(0.05, 0.3),
        lower_wick=body1 * rng.uniform(0.1, 0.4),
        volume=random_volume(rng),
    )

    # Day 2: small bullish body inside day-1 range.
    inner = body1 * rng.uniform(0.25, 0.55)
    c2_open = c1_close + body1 * rng.uniform(0.15, 0.35)
    c2_close = c2_open + inner
    c2_close = min(c2_close, c1_open - body1 * 0.05)
    c2 = make_candle(
        open_=c2_open,
        close=c2_close,
        upper_wick=inner * rng.uniform(0.05, 0.4),
        lower_wick=inner * rng.uniform(0.05, 0.4),
        volume=random_volume(rng, mean=1_200_000),
    )

    # Day 3: closes above day-1 open.
    c3_open = c2_close + body1 * rng.uniform(-0.05, 0.1)
    c3_close = c1_open + body1 * rng.uniform(0.05, 0.35)
    c3 = make_candle(
        open_=c3_open,
        close=c3_close,
        upper_wick=body1 * rng.uniform(0.05, 0.3),
        lower_wick=body1 * rng.uniform(0.05, 0.25),
        volume=random_volume(rng, mean=2_000_000),
    )

    candles: list[Candle] = [c1, c2, c3]
    if with_context:
        candles = prepend_context(candles, rng, trend="down")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    first_bear = a["close"] < a["open"]
    second_bull = b["close"] > b["open"]
    inside = b["open"] > a["close"] and b["close"] < a["open"]
    confirm = c["close"] > a["open"]
    return bool(first_bear and second_bull and inside and confirm)
