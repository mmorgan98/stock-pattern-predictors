"""Dark Cloud Cover — bullish candle followed by bearish close below midpoint."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "dark_cloud_cover"
WINDOW = 2


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body1 = price * rng.uniform(0.015, 0.03)

    c1_open = price
    c1_close = price + body1
    c1 = make_candle(
        open_=c1_open,
        close=c1_close,
        upper_wick=body1 * rng.uniform(0.1, 0.4),
        lower_wick=body1 * rng.uniform(0.05, 0.3),
        volume=random_volume(rng),
    )

    mid = (c1_open + c1_close) / 2
    c2_open = c1_close + body1 * rng.uniform(0.02, 0.15)
    c2_close = mid - body1 * rng.uniform(0.05, 0.25)
    c2_close = max(c2_close, c1_open + body1 * 0.02)
    c2 = make_candle(
        open_=c2_open,
        close=c2_close,
        upper_wick=body1 * rng.uniform(0.05, 0.25),
        lower_wick=body1 * rng.uniform(0.05, 0.3),
        volume=random_volume(rng, mean=2_000_000),
    )

    candles: list[Candle] = [c1, c2]
    if with_context:
        candles = prepend_context(candles, rng, trend="up")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    mid = (a["open"] + a["close"]) / 2
    return bool(
        a["close"] > a["open"]
        and b["close"] < b["open"]
        and b["open"] > a["close"]
        and b["close"] < mid
        and b["close"] > a["open"]
    )
