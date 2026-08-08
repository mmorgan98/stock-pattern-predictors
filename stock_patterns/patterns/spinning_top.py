"""Spinning Top — modest body centered with noticeable upper and lower wicks."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, body, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "spinning_top"
WINDOW = 1


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    mid_body = price * rng.uniform(0.004, 0.012)
    wick = mid_body * rng.uniform(1.1, 2.4)

    open_ = price
    close = open_ + mid_body if rng.random() < 0.5 else open_ - mid_body
    candle = make_candle(
        open_=open_,
        close=close,
        upper_wick=wick,
        lower_wick=wick * rng.uniform(0.85, 1.15),
        volume=random_volume(rng),
    )
    candles: list[Candle] = [candle]
    if with_context:
        candles = prepend_context(candles, rng, trend="flat")
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
    return bool(upper >= 0.6 * b and lower >= 0.6 * b and upper < 2.5 * b and lower < 2.5 * b)
