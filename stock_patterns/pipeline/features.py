"""Feature extraction for OHLC pattern sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_patterns.patterns.base import normalize_sequence


def sequence_to_feature_vector(df: pd.DataFrame, max_len: int = 20) -> np.ndarray:
    """
    Convert an OHLC dataframe into a fixed-length feature vector.

    Uses relative OHLC (normalized by first close), padded/truncated to max_len,
    plus simple derived stats (body ratios, wick ratios).
    """
    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    norm = normalize_sequence(work)
    if len(norm) > max_len:
        norm = norm[-max_len:]
    if len(norm) < max_len:
        pad = np.zeros((max_len - len(norm), 4), dtype=np.float32)
        norm = np.vstack([pad, norm])

    opens, highs, lows, closes = norm.T
    bodies = np.abs(closes - opens)
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows
    span = np.maximum(highs - lows, 1e-6)

    extras = np.asarray(
        [
            bodies.mean(),
            upper.mean(),
            lower.mean(),
            (bodies / span).mean(),
            closes[-1] - closes[0],
            highs.max() - lows.min(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([norm.reshape(-1), extras])
