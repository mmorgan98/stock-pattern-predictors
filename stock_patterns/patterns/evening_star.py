"""Evening Star — bullish candle, small-bodied star, bearish confirmation."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, body, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "evening_star"
WINDOW = 3


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body1 = price * rng.uniform(0.018, 0.035)

    c1 = make_candle(
        open_=price,
        close=price + body1,
        upper_wick=body1 * rng.uniform(0.1, 0.35),
        lower_wick=body1 * rng.uniform(0.05, 0.25),
        volume=random_volume(rng),
    )

    star_body = body1 * rng.uniform(0.08, 0.28)
    gap = body1 * rng.uniform(0.05, 0.2)
    c2_open = c1.close + gap
    c2_close = c2_open + star_body if rng.random() < 0.5 else c2_open - star_body
    c2 = make_candle(
        open_=c2_open,
        close=c2_close,
        upper_wick=star_body * rng.uniform(0.4, 1.5),
        lower_wick=star_body * rng.uniform(0.4, 1.5),
        volume=random_volume(rng, mean=900_000),
    )

    c3_open = min(c2.open, c2.close) - body1 * rng.uniform(0.0, 0.1)
    c3_close = c1.open - body1 * rng.uniform(0.0, 0.25)
    c3 = make_candle(
        open_=c3_open,
        close=c3_close,
        upper_wick=body1 * rng.uniform(0.05, 0.25),
        lower_wick=body1 * rng.uniform(0.05, 0.3),
        volume=random_volume(rng, mean=2_000_000),
    )

    candles: list[Candle] = [c1, c2, c3]
    if with_context:
        candles = prepend_context(candles, rng, trend="up")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return bool(
        a["close"] > a["open"]
        and body(b["open"], b["close"]) < body(a["open"], a["close"]) * 0.5
        and c["close"] < c["open"]
        and c["close"] < (a["open"] + a["close"]) / 2
    )
