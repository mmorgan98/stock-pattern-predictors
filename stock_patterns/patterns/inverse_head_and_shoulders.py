"""Inverse Head and Shoulders — left trough, deeper head, right trough (bullish)."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "inverse_head_and_shoulders"
WINDOW = 15


def _swing_down(rng, start: float, trough: float, bars: int) -> list[Candle]:
    candles: list[Candle] = []
    prices = [start + (trough - start) * (i / max(bars - 1, 1)) for i in range(bars)]
    for i, close_target in enumerate(prices):
        open_ = close_target + abs(start - trough) * rng.uniform(0.02, 0.08)
        if i > 0:
            open_ = candles[-1].close + rng.uniform(-abs(start - trough) * 0.05, abs(start - trough) * 0.02)
        close = close_target + rng.uniform(-abs(start - trough) * 0.02, abs(start - trough) * 0.02)
        wick = abs(close - open_) * rng.uniform(0.2, 0.8)
        candles.append(make_candle(open_, close, wick, wick, random_volume(rng)))
    return candles


def _swing_up(rng, start: float, peak: float, bars: int) -> list[Candle]:
    candles: list[Candle] = []
    prices = [start + (peak - start) * (i / max(bars - 1, 1)) for i in range(bars)]
    for i, close_target in enumerate(prices):
        open_ = close_target - abs(peak - start) * rng.uniform(0.02, 0.08)
        if i > 0:
            open_ = candles[-1].close + rng.uniform(-abs(peak - start) * 0.02, abs(peak - start) * 0.05)
        close = close_target + rng.uniform(-abs(peak - start) * 0.02, abs(peak - start) * 0.02)
        wick = abs(close - open_) * rng.uniform(0.2, 0.8)
        candles.append(make_candle(open_, close, wick, wick, random_volume(rng)))
    return candles


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    base = random_base_price(rng)
    neck = base
    left_shoulder = neck * rng.uniform(0.93, 0.96)
    head = neck * rng.uniform(0.84, 0.88)
    # Keep shoulders close so the matcher thirds agree.
    right_shoulder = left_shoulder * rng.uniform(0.99, 1.01)

    candles: list[Candle] = []
    candles += _swing_down(rng, neck, left_shoulder, 3)
    candles += _swing_up(rng, left_shoulder, neck * rng.uniform(0.99, 1.01), 2)
    candles += _swing_down(rng, neck, head, 3)
    candles += _swing_up(rng, head, neck * rng.uniform(0.99, 1.01), 2)
    candles += _swing_down(rng, neck, right_shoulder, 3)
    candles += _swing_up(rng, right_shoulder, neck * rng.uniform(1.01, 1.04), 2)

    if with_context:
        candles = prepend_context(candles, rng, context_len=5, trend="down")
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    if len(df) < WINDOW:
        return False
    lows = df["low"].iloc[-WINDOW:].to_numpy()
    n = len(lows)
    left = lows[: n // 3].min()
    mid = lows[n // 3 : 2 * n // 3].min()
    right = lows[2 * n // 3 :].min()
    shoulders_close = abs(left - right) / max(left, right) < 0.06
    head_lowest = mid < left * 0.985 and mid < right * 0.985
    return bool(shoulders_close and head_lowest)
