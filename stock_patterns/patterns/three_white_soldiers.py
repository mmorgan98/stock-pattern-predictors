"""Three White Soldiers — three consecutive strong bullish candles."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "three_white_soldiers"
WINDOW = 3


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    candles: list[Candle] = []
    open_ = price
    for _ in range(3):
        body = price * rng.uniform(0.012, 0.025)
        close = open_ + body
        candles.append(
            make_candle(
                open_=open_,
                close=close,
                upper_wick=body * rng.uniform(0.02, 0.2),
                lower_wick=body * rng.uniform(0.02, 0.25),
                volume=random_volume(rng, mean=1_800_000),
            )
        )
        open_ = close - body * rng.uniform(0.05, 0.25)  # open within prior body
        price = close

    if with_context:
        candles = prepend_context(candles, rng, trend="down")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    rows = df.iloc[-3:]
    bullish = all(r["close"] > r["open"] for _, r in rows.iterrows())
    rising = rows["close"].is_monotonic_increasing
    return bool(bullish and rising)
