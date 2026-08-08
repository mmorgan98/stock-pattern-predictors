"""Tweezer Top / Bottom — matching highs (top) or lows (bottom) across two candles."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "tweezer"
WINDOW = 2


def generate(seed: int | None = None, with_context: bool = True, variant: str | None = None) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    body_size = price * rng.uniform(0.01, 0.025)
    variant = variant or ("top" if rng.random() < 0.5 else "bottom")

    if variant == "top":
        shared_high = price + body_size * rng.uniform(1.1, 1.6)
        c1_open = price
        c1_close = price + body_size
        c1 = make_candle(
            open_=c1_open,
            close=c1_close,
            upper_wick=max(0.0, shared_high - c1_close),
            lower_wick=body_size * rng.uniform(0.1, 0.4),
            volume=random_volume(rng),
        )
        c2_open = c1_close + body_size * rng.uniform(-0.05, 0.15)
        c2_close = c2_open - body_size * rng.uniform(0.8, 1.2)
        c2 = make_candle(
            open_=c2_open,
            close=c2_close,
            upper_wick=max(0.0, shared_high - max(c2_open, c2_close)),
            lower_wick=body_size * rng.uniform(0.1, 0.4),
            volume=random_volume(rng),
        )
        c1 = Candle(c1.open, shared_high, c1.low, c1.close, c1.volume)
        c2 = Candle(c2.open, shared_high, min(c2.low, min(c2.open, c2.close)), c2.close, c2.volume)
        trend = "up"
    else:
        shared_low = price - body_size * rng.uniform(1.1, 1.6)
        c1_open = price
        c1_close = price - body_size
        c1 = make_candle(
            open_=c1_open,
            close=c1_close,
            upper_wick=body_size * rng.uniform(0.1, 0.4),
            lower_wick=max(0.0, c1_close - shared_low),
            volume=random_volume(rng),
        )
        c2_open = c1_close + body_size * rng.uniform(-0.15, 0.05)
        c2_close = c2_open + body_size * rng.uniform(0.8, 1.2)
        c2 = make_candle(
            open_=c2_open,
            close=c2_close,
            upper_wick=body_size * rng.uniform(0.1, 0.4),
            lower_wick=max(0.0, min(c2_open, c2_close) - shared_low),
            volume=random_volume(rng),
        )
        c1 = Candle(c1.open, c1.high, shared_low, c1.close, c1.volume)
        c2 = Candle(c2.open, c2.high, shared_low, c2.close, c2.volume)
        trend = "down"

    candles: list[Candle] = [c1, c2]
    if with_context:
        candles = prepend_context(candles, rng, trend=trend)
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    a_bull = a["close"] > a["open"]
    b_bull = b["close"] > b["open"]
    # Classic tweezer: opposite-colored candles sharing a high or low.
    if a_bull == b_bull:
        return False
    high_match = abs(a["high"] - b["high"]) / max(a["high"], b["high"]) < 0.0015
    low_match = abs(a["low"] - b["low"]) / max(a["low"], b["low"]) < 0.0015
    tweezer_top = high_match and a_bull and (not b_bull)
    tweezer_bottom = low_match and (not a_bull) and b_bull
    return bool(tweezer_top or tweezer_bottom)
