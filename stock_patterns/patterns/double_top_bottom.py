"""Double Top / Double Bottom — two similar swing highs or lows with a mid trough/peak."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "double_top_bottom"
WINDOW = 12


def _move(rng, start: float, end: float, bars: int) -> list[Candle]:
    candles: list[Candle] = []
    for i in range(bars):
        target = start + (end - start) * ((i + 1) / bars)
        open_ = candles[-1].close if candles else start
        close = target + rng.uniform(-abs(end - start) * 0.03, abs(end - start) * 0.03)
        wick = abs(close - open_) * rng.uniform(0.2, 0.9) + abs(end - start) * 0.01
        candles.append(make_candle(open_, close, wick, wick, random_volume(rng)))
    return candles


def generate(seed: int | None = None, with_context: bool = True, variant: str | None = None) -> pd.DataFrame:
    rng = _rng(seed)
    base = random_base_price(rng)
    variant = variant or ("top" if rng.random() < 0.5 else "bottom")

    if variant == "top":
        peak = base * rng.uniform(1.08, 1.14)
        mid = base * rng.uniform(1.01, 1.04)
        candles = []
        candles += _move(rng, base, peak, 3)
        candles += _move(rng, peak, mid, 3)
        candles += _move(rng, mid, peak * rng.uniform(0.995, 1.005), 3)
        candles += _move(rng, peak, base * rng.uniform(0.97, 1.0), 3)
        trend = "up"
    else:
        trough = base * rng.uniform(0.86, 0.92)
        mid = base * rng.uniform(0.96, 0.99)
        candles = []
        candles += _move(rng, base, trough, 3)
        candles += _move(rng, trough, mid, 3)
        candles += _move(rng, mid, trough * rng.uniform(0.995, 1.005), 3)
        candles += _move(rng, trough, base * rng.uniform(1.0, 1.03), 3)
        trend = "down"

    if with_context:
        candles = prepend_context(candles, rng, context_len=4, trend=trend)
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    highs = df["high"].iloc[-WINDOW:].to_numpy()
    lows = df["low"].iloc[-WINDOW:].to_numpy()
    n = len(highs)
    left_high = highs[: n // 2].max()
    right_high = highs[n // 2 :].max()
    left_low = lows[: n // 2].min()
    right_low = lows[n // 2 :].min()
    double_top = abs(left_high - right_high) / max(left_high, right_high) < 0.03
    double_bottom = abs(left_low - right_low) / max(left_low, right_low) < 0.03
    return bool(double_top or double_bottom)
