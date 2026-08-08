"""Bullish Engulfing — large green candle fully engulfs prior red body."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "bullish_engulfing"
WINDOW = 2


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body1 = price * rng.uniform(0.012, 0.03)
    body2 = body1 * rng.uniform(1.35, 2.2)

    c1_open = price
    c1_close = price - body1
    c1 = make_candle(
        open_=c1_open,
        close=c1_close,
        upper_wick=body1 * rng.uniform(0.05, 0.35),
        lower_wick=body1 * rng.uniform(0.05, 0.45),
        volume=random_volume(rng),
    )

    # Second candle opens at/near prior close and closes above prior open.
    c2_open = c1_close - body1 * rng.uniform(0.0, 0.15)
    c2_close = c1_open + body1 * rng.uniform(0.05, 0.35)
    # Guarantee engulfing body size.
    if c2_close - c2_open < body2:
        c2_close = c2_open + body2
    c2 = make_candle(
        open_=c2_open,
        close=c2_close,
        upper_wick=body2 * rng.uniform(0.05, 0.3),
        lower_wick=body2 * rng.uniform(0.05, 0.25),
        volume=random_volume(rng, mean=2_200_000),
    )

    candles: list[Candle] = [c1, c2]
    if with_context:
        candles = prepend_context(candles, rng, trend="down")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    prior_bear = a["close"] < a["open"]
    curr_bull = b["close"] > b["open"]
    engulfs = b["open"] <= a["close"] and b["close"] >= a["open"]
    return bool(prior_bear and curr_bull and engulfs)
