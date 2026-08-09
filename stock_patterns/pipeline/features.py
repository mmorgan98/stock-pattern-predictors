"""Feature extraction for OHLC pattern sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_patterns.patterns.base import normalize_sequence

FEATURE_MAX_LEN = 20


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.maximum(b, 1e-6)


def sequence_to_feature_vector(df: pd.DataFrame, max_len: int = FEATURE_MAX_LEN) -> np.ndarray:
    """
    Convert an OHLC dataframe into a fixed-length geometry-aware feature vector.

    Includes:
    - relative OHLC (normalized by first close), padded/truncated
    - per-candle body/wick/range ratios
    - close location in range, signed body, gaps
    - short trend / ATR-like context stats
    """
    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    if not {"open", "high", "low", "close"}.issubset(work.columns):
        raise ValueError("DataFrame must include open/high/low/close")

    norm = normalize_sequence(work)
    if len(norm) > max_len:
        norm = norm[-max_len:]
    if len(norm) < max_len:
        pad = np.zeros((max_len - len(norm), 4), dtype=np.float32)
        norm = np.vstack([pad, norm])

    opens, highs, lows, closes = norm.T
    bodies = np.abs(closes - opens)
    signed = closes - opens
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows
    span = np.maximum(highs - lows, 1e-6)
    body_ratio = _safe_div(bodies, span)
    upper_ratio = _safe_div(upper, span)
    lower_ratio = _safe_div(lower, span)
    close_loc = _safe_div(closes - lows, span)
    gaps = np.zeros_like(closes)
    gaps[1:] = opens[1:] - closes[:-1]

    # Focus geometric descriptors on the most recent candles (pattern core).
    tail = 5
    extras = np.asarray(
        [
            bodies.mean(),
            upper.mean(),
            lower.mean(),
            body_ratio.mean(),
            upper_ratio.mean(),
            lower_ratio.mean(),
            close_loc.mean(),
            signed.mean(),
            closes[-1] - closes[0],
            highs.max() - lows.min(),
            # recent geometry
            body_ratio[-tail:].mean(),
            upper_ratio[-tail:].mean(),
            lower_ratio[-tail:].mean(),
            close_loc[-tail:].mean(),
            signed[-tail:].mean(),
            bodies[-1],
            upper[-1],
            lower[-1],
            body_ratio[-1],
            upper_ratio[-1],
            lower_ratio[-1],
            close_loc[-1],
            signed[-1],
            gaps[-1],
            np.abs(gaps[-tail:]).mean(),
            # simple trend / volatility context
            closes[-1] - closes[-min(5, len(closes))],
            closes[-1] - closes[-min(10, len(closes))],
            span[-tail:].mean(),
            span.mean(),
            (closes[-1] - lows[-tail:].min()) / max(span[-tail:].max(), 1e-6),
            (highs[-tail:].max() - closes[-1]) / max(span[-tail:].max(), 1e-6),
            # symmetry-ish stats useful for H&S / double tops
            highs[-tail:].max() / max(highs.max(), 1e-6),
            lows[-tail:].min() / max(max(lows.max(), 1e-6), 1e-6),
            (highs[: max_len // 2].max() - highs[max_len // 2 :].max()),
            (lows[: max_len // 2].min() - lows[max_len // 2 :].min()),
        ],
        dtype=np.float32,
    )

    per_candle = np.column_stack(
        [
            body_ratio,
            upper_ratio,
            lower_ratio,
            close_loc,
            signed,
            gaps,
        ]
    ).astype(np.float32)

    return np.concatenate([norm.reshape(-1), per_candle.reshape(-1), extras])
