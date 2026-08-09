"""Build labeled datasets from synthetic generators and Yahoo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_patterns.patterns.registry import PATTERN_NAMES, generate_all
from stock_patterns.pipeline.features import FEATURE_MAX_LEN, sequence_to_feature_vector


def pad_sequence_features(vectors: list[np.ndarray]) -> np.ndarray:
    return np.vstack(vectors)


def build_synthetic_dataset(
    samples_per_pattern: int = 100,
    seed: int = 42,
    max_len: int = FEATURE_MAX_LEN,
    no_pattern_multiplier: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, class_names) from random pattern generators."""
    generated = generate_all(samples_per_pattern=samples_per_pattern, seed=seed)
    # Extra hard-negative mass.
    if "no_pattern" in generated:
        from stock_patterns.patterns.no_pattern import generate as gen_no_pattern

        extra = samples_per_pattern * max(0, no_pattern_multiplier - 1)
        generated["no_pattern"].extend(
            [gen_no_pattern(seed=seed + 500_000 + i) for i in range(extra)]
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


def label_yahoo_with_heuristics(
    df: pd.DataFrame,
    max_len: int = FEATURE_MAX_LEN,
    include_negatives: bool = True,
    negative_stride: int = 5,
    negative_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Weakly label Yahoo OHLCV using rule-based matchers.

    Positive labels come from heuristic hits. Optional negatives are windows that
    match no pattern (mapped to no_pattern).
    """
    from stock_patterns.patterns.registry import PATTERN_MATCHERS, PATTERN_WINDOWS

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    name_to_idx = {n: i for i, n in enumerate(PATTERN_NAMES)}
    positive_ends: set[tuple[str, int]] = set()

    for name, matcher in PATTERN_MATCHERS.items():
        if name == "no_pattern":
            continue
        window = PATTERN_WINDOWS[name]
        if len(work) < window:
            continue
        for end in range(window, len(work) + 1):
            slice_df = work.iloc[end - window : end]
            if matcher(slice_df):
                start = max(0, end - max(window, min(max_len, window + 5)))
                ctx = work.iloc[start:end]
                xs.append(sequence_to_feature_vector(ctx, max_len=max_len))
                ys.append(name_to_idx[name])
                positive_ends.add((name, end))

    if include_negatives and "no_pattern" in name_to_idx:
        max_window = max(PATTERN_WINDOWS.values())
        neg_added = 0
        for end in range(max_window, len(work) + 1, max(1, negative_stride)):
            # Skip if any pattern matched ending here.
            if any(end == e for _, e in positive_ends):
                continue
            # Confirm no matcher fires on its native window ending here.
            hit = False
            for name, matcher in PATTERN_MATCHERS.items():
                if name == "no_pattern":
                    continue
                window = PATTERN_WINDOWS[name]
                if end < window:
                    continue
                if matcher(work.iloc[end - window : end]):
                    hit = True
                    break
            if hit:
                continue
            ctx = work.iloc[end - max_window : end]
            xs.append(sequence_to_feature_vector(ctx, max_len=max_len))
            ys.append(name_to_idx["no_pattern"])
            neg_added += 1
            if negative_limit is not None and neg_added >= negative_limit:
                break

    if not xs:
        return np.empty((0, 1)), np.empty((0,), dtype=np.int64), list(PATTERN_NAMES)
    return pad_sequence_features(xs), np.asarray(ys, dtype=np.int64), list(PATTERN_NAMES)


def build_mixed_dataset(
    samples_per_pattern: int = 100,
    seed: int = 42,
    tickers: list[str] | None = None,
    period: str = "2y",
    interval: str = "1d",
    max_len: int = FEATURE_MAX_LEN,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic dataset optionally mixed with weakly labeled Yahoo bars."""
    from stock_patterns.data.yahoo_finance import fetch_ohlcv

    X, y, class_names = build_synthetic_dataset(
        samples_per_pattern=samples_per_pattern,
        seed=seed,
        max_len=max_len,
    )
    for ticker in tickers or []:
        try:
            yahoo = fetch_ohlcv(ticker, period=period, interval=interval)
        except Exception as exc:  # noqa: BLE001 - keep training resilient
            print(f"Skipping {ticker}: {exc}")
            continue
        Xy, yy, _ = label_yahoo_with_heuristics(yahoo, max_len=max_len)
        if len(yy):
            X = np.vstack([X, Xy])
            y = np.concatenate([y, yy])
            print(f"Added {len(yy)} weakly labeled Yahoo samples from {ticker.upper()}")
    return X, y, class_names
