"""Double Top / Double Bottom — two similar swing highs or lows with a mid trough/peak."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, prepend_context, random_base_price, random_volume


PATTERN_NAME = "double_top_bottom"
WINDOW = 12


def _candle_at(rng, close: float, prev_close: float | None = None) -> Candle:
    open_ = prev_close if prev_close is not None else close * rng.uniform(0.995, 1.005)
    # Keep body modest so high/low swings dominate structure.
    body = abs(close - open_)
    if body < close * 0.001:
        open_ = close * (0.998 if rng.random() < 0.5 else 1.002)
        body = abs(close - open_)
    wick = max(body * rng.uniform(0.05, 0.25), close * 0.001)
    return make_candle(open_, close, wick * 0.4, wick * 0.4, random_volume(rng))


def generate(seed: int | None = None, with_context: bool = True, variant: str | None = None) -> pd.DataFrame:
    rng = _rng(seed)
    base = random_base_price(rng)
    variant = variant or ("top" if rng.random() < 0.5 else "bottom")

    # Fixed 12-bar skeleton with clear local extrema.
    if variant == "top":
        peak = base * rng.uniform(1.08, 1.12)
        mid = peak * rng.uniform(0.95, 0.97)
        closes = [
            base,
            base * 1.02,
            peak,          # left top (local high)
            peak * 0.99,
            mid,           # trough
            mid * 1.01,
            peak * rng.uniform(0.997, 1.003),  # right top
            peak * 0.99,
            mid,
            mid * 0.99,
            base * 1.01,
            base * 0.99,
        ]
        trend = "up"
    else:
        trough = base * rng.uniform(0.88, 0.92)
        mid = trough * rng.uniform(1.03, 1.05)
        closes = [
            base,
            base * 0.98,
            trough,        # left bottom
            trough * 1.01,
            mid,           # bounce
            mid * 0.99,
            trough * rng.uniform(0.997, 1.003),  # right bottom
            trough * 1.01,
            mid,
            mid * 1.01,
            base * 0.99,
            base * 1.01,
        ]
        trend = "down"

    candles: list[Candle] = []
    prev = None
    for close in closes:
        c = _candle_at(rng, close, prev)
        # Enforce extrema levels so matcher local peaks/troughs are reliable.
        if variant == "top" and close >= peak * 0.996:
            c = Candle(c.open, max(c.high, close), min(c.low, min(c.open, close)), c.close, c.volume)
        if variant == "bottom" and close <= trough * 1.004:
            c = Candle(c.open, max(c.high, max(c.open, close)), min(c.low, close), c.close, c.volume)
        candles.append(c)
        prev = c.close

    if with_context:
        candles = prepend_context(candles, rng, context_len=4, trend=trend)
    return candles_to_dataframe(candles)


def _local_extrema(values: np.ndarray, kind: str) -> list[int]:
    idxs: list[int] = []
    for i in range(1, len(values) - 1):
        if kind == "max" and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            idxs.append(i)
        if kind == "min" and values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            idxs.append(i)
    return idxs


def matches(df: pd.DataFrame) -> bool:
    """
    Require two local swing highs/lows of similar height with a meaningful
    retracement between them. Plain half-window max/min equality is too common
    on real daily bars and dominated earlier detection output.
    """
    if len(df) < WINDOW:
        return False

    highs = df["high"].iloc[-WINDOW:].to_numpy(dtype=float)
    lows = df["low"].iloc[-WINDOW:].to_numpy(dtype=float)

    peak_idxs = _local_extrema(highs, "max")
    for i in range(len(peak_idxs)):
        for j in range(i + 1, len(peak_idxs)):
            p1, p2 = peak_idxs[i], peak_idxs[j]
            if p2 - p1 < 3:
                continue
            h1, h2 = highs[p1], highs[p2]
            level = max(h1, h2)
            if level <= 0:
                continue
            if abs(h1 - h2) / level > 0.012:
                continue
            # Peaks should be near the window's dominant highs.
            if min(h1, h2) < highs.max() * 0.985:
                continue
            trough = lows[p1 : p2 + 1].min()
            retrace = (level - trough) / level
            if retrace >= 0.03:
                return True

    trough_idxs = _local_extrema(lows, "min")
    for i in range(len(trough_idxs)):
        for j in range(i + 1, len(trough_idxs)):
            t1, t2 = trough_idxs[i], trough_idxs[j]
            if t2 - t1 < 3:
                continue
            l1, l2 = lows[t1], lows[t2]
            level = min(l1, l2)
            if level <= 0:
                continue
            if abs(l1 - l2) / max(l1, l2) > 0.012:
                continue
            if max(l1, l2) > lows.min() * 1.015:
                continue
            peak = highs[t1 : t2 + 1].max()
            bounce = (peak - level) / level
            if bounce >= 0.03:
                return True

    return False
