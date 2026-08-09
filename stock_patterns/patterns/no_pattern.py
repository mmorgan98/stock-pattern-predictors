"""No Pattern — hard negatives: trends, chop, gaps, and noisy random walks."""

from __future__ import annotations

import pandas as pd

from .base import Candle, _rng, candles_to_dataframe, make_candle, random_base_price, random_volume


PATTERN_NAME = "no_pattern"
WINDOW = 8


def _walk(rng, price: float, n: int, drift: float, vol: float) -> list[Candle]:
    candles: list[Candle] = []
    for _ in range(n):
        ret = rng.normal(drift, vol)
        open_ = price
        close = max(0.01, price * (1.0 + ret))
        # Occasional gap open.
        if rng.random() < 0.12:
            open_ = price * (1.0 + rng.normal(0.0, vol * 1.8))
        wick = abs(close - open_) * rng.uniform(0.15, 1.4) + price * rng.uniform(0.001, 0.006)
        # Asymmetric wicks, but not pattern-perfect.
        upper = wick * rng.uniform(0.2, 1.3)
        lower = wick * rng.uniform(0.2, 1.3)
        candles.append(make_candle(open_, close, upper, lower, random_volume(rng)))
        price = close
    return candles


def _choppy(rng, price: float, n: int) -> list[Candle]:
    candles: list[Candle] = []
    for i in range(n):
        amp = price * rng.uniform(0.004, 0.015)
        open_ = price + rng.uniform(-amp, amp)
        close = price + rng.uniform(-amp, amp)
        wick = amp * rng.uniform(0.5, 1.8)
        candles.append(make_candle(open_, close, wick, wick, random_volume(rng)))
        price = close
    return candles


def _trend_then_chop(rng, price: float) -> list[Candle]:
    n_trend = int(rng.integers(4, 8))
    n_chop = int(rng.integers(4, 8))
    drift = abs(rng.normal(0.01, 0.004)) * (1 if rng.random() < 0.5 else -1)
    return _walk(rng, price, n_trend, drift=drift, vol=0.008) + _choppy(
        rng, price * (1 + drift * n_trend), n_chop
    )


def generate(seed: int | None = None, with_context: bool = True) -> pd.DataFrame:
    rng = _rng(seed)
    price = random_base_price(rng)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        candles = _walk(rng, price, int(rng.integers(8, 16)), drift=0.0, vol=rng.uniform(0.008, 0.02))
    elif mode == 1:
        drift = rng.uniform(0.006, 0.014) * (1 if rng.random() < 0.5 else -1)
        candles = _walk(rng, price, int(rng.integers(8, 16)), drift=drift, vol=0.007)
    elif mode == 2:
        candles = _choppy(rng, price, int(rng.integers(8, 16)))
    else:
        candles = _trend_then_chop(rng, price)
    return candles_to_dataframe(candles)


def matches(df: pd.DataFrame) -> bool:
    # Negative class is never emitted by heuristic scans.
    return False
