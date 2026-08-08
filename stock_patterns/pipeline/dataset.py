"""Build labeled datasets from synthetic generators and Yahoo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_patterns.patterns.registry import PATTERN_NAMES, generate_all
from stock_patterns.pipeline.features import sequence_to_feature_vector


def pad_sequence_features(vectors: list[np.ndarray]) -> np.ndarray:
    return np.vstack(vectors)


def build_synthetic_dataset(
    samples_per_pattern: int = 100,
    seed: int = 42,
    max_len: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, class_names) from random pattern generators."""
    generated = generate_all(samples_per_pattern=samples_per_pattern, seed=seed)
    # Extra negative-class mass so real-market noise is not forced into a label.
    no_pattern_extra = samples_per_pattern
    if "no_pattern" in generated:
        from stock_patterns.patterns.no_pattern import generate as gen_no_pattern

        generated["no_pattern"].extend(
            [gen_no_pattern(seed=seed + 500_000 + i) for i in range(no_pattern_extra)]
        )

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, name in enumerate(PATTERN_NAMES):
        for df in generated[name]:
            xs.append(sequence_to_feature_vector(df, max_len=max_len))
            ys.append(label)
    return pad_sequence_features(xs), np.asarray(ys, dtype=np.int64), list(PATTERN_NAMES)


def save_synthetic_csvs(
    out_dir: str | Path,
    samples_per_pattern: int = 50,
    seed: int = 42,
) -> Path:
    """Write one CSV per generated sample under out_dir/<pattern>/."""
    out = Path(out_dir)
    generated = generate_all(samples_per_pattern=samples_per_pattern, seed=seed)
    for name, frames in generated.items():
        pattern_dir = out / name
        pattern_dir.mkdir(parents=True, exist_ok=True)
        for i, df in enumerate(frames):
            df.to_csv(pattern_dir / f"{name}_{i:04d}.csv")
    return out


def label_yahoo_with_heuristics(df: pd.DataFrame, max_len: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Weakly label Yahoo OHLCV using rule-based matchers.

    Useful for bootstrapping real-market fine-tuning after synthetic pretraining.
    """
    from stock_patterns.patterns.registry import PATTERN_MATCHERS, PATTERN_WINDOWS

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    name_to_idx = {n: i for i, n in enumerate(PATTERN_NAMES)}

    for name, matcher in PATTERN_MATCHERS.items():
        window = PATTERN_WINDOWS[name]
        if len(work) < window:
            continue
        for end in range(window, len(work) + 1):
            slice_df = work.iloc[end - window : end]
            if matcher(slice_df):
                # Include a little lookback context when available.
                start = max(0, end - max(window, min(max_len, window + 5)))
                ctx = work.iloc[start:end]
                xs.append(sequence_to_feature_vector(ctx, max_len=max_len))
                ys.append(name_to_idx[name])

    if not xs:
        return np.empty((0, 1)), np.empty((0,), dtype=np.int64), list(PATTERN_NAMES)
    return pad_sequence_features(xs), np.asarray(ys, dtype=np.int64), list(PATTERN_NAMES)
