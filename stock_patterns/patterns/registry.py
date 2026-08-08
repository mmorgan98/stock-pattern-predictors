"""Central registry of pattern generators and detectors."""

from __future__ import annotations

from types import ModuleType
from typing import Callable

import pandas as pd

from . import (
    bearish_engulfing,
    bearish_harami,
    bullish_engulfing,
    bullish_harami,
    dark_cloud_cover,
    doji_star,
    double_top_bottom,
    evening_star,
    hammer,
    head_and_shoulders,
    inverse_head_and_shoulders,
    marubozu,
    morning_star,
    piercing_line,
    shooting_star,
    spinning_top,
    three_black_crows,
    three_inside_down,
    three_inside_up,
    three_white_soldiers,
    tweezer,
)

_MODULES: list[ModuleType] = [
    bullish_engulfing,
    bearish_engulfing,
    hammer,
    shooting_star,
    doji_star,
    three_inside_up,
    three_inside_down,
    spinning_top,
    head_and_shoulders,
    inverse_head_and_shoulders,
    morning_star,
    evening_star,
    piercing_line,
    dark_cloud_cover,
    bullish_harami,
    bearish_harami,
    three_white_soldiers,
    three_black_crows,
    marubozu,
    tweezer,
    double_top_bottom,
]

PATTERN_GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    m.PATTERN_NAME: m.generate for m in _MODULES
}
PATTERN_MATCHERS: dict[str, Callable[[pd.DataFrame], bool]] = {
    m.PATTERN_NAME: m.matches for m in _MODULES
}
PATTERN_WINDOWS: dict[str, int] = {m.PATTERN_NAME: m.WINDOW for m in _MODULES}
PATTERN_NAMES: list[str] = list(PATTERN_GENERATORS.keys())


def generate_all(samples_per_pattern: int = 50, seed: int = 42) -> dict[str, list[pd.DataFrame]]:
    """Generate synthetic samples for every registered pattern."""
    out: dict[str, list[pd.DataFrame]] = {}
    for i, name in enumerate(PATTERN_NAMES):
        gen = PATTERN_GENERATORS[name]
        out[name] = [gen(seed=seed + i * 10_000 + j) for j in range(samples_per_pattern)]
    return out


def detect_patterns(df: pd.DataFrame) -> list[dict]:
    """Scan a Yahoo/OHLC dataframe and return heuristic pattern hits."""
    hits: list[dict] = []
    if df is None or df.empty:
        return hits

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    for name, matcher in PATTERN_MATCHERS.items():
        window = PATTERN_WINDOWS[name]
        if len(work) < window:
            continue
        for end in range(window, len(work) + 1):
            slice_df = work.iloc[end - window : end]
            if matcher(slice_df):
                hits.append(
                    {
                        "pattern": name,
                        "end_date": str(slice_df.index[-1]),
                        "start_date": str(slice_df.index[0]),
                        "window": window,
                    }
                )
    return hits
