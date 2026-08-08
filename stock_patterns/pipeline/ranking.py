"""Rank candlestick / chart patterns inside a date window."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.patterns.registry import PATTERN_WINDOWS, detect_patterns
from stock_patterns.pipeline.features import sequence_to_feature_vector


def slice_date_window(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return rows inside [start, end], inclusive on available timestamps."""
    if df.empty:
        return df
    work = df.sort_index()
    if start is not None:
        work = work[work.index >= pd.Timestamp(start)]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        work = work[work.index <= end_ts]
    return work


def rank_patterns(
    df: pd.DataFrame,
    top: int = 5,
    model: PatternClassifier | None = None,
    step: int = 1,
) -> list[dict]:
    """
    Rank the top patterns in a dataframe window.

    When a model is available, confidence is that pattern's share of mean
    class-probability mass (excluding `no_pattern`). Heuristic hit counts are
    attached for context and used alone when no model is loaded.
    """
    if df is None or df.empty:
        return []

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]

    hits = detect_patterns(work)
    hit_counts = Counter(h["pattern"] for h in hits)
    total_hits = sum(hit_counts.values())

    class_scores: dict[str, dict] = {}

    if model is not None and model.class_names:
        class_names = [n for n in model.class_names if n != "no_pattern"]
        proba_sums = {n: 0.0 for n in class_names}
        proba_max = {n: 0.0 for n in class_names}
        win_counts = {n: 0 for n in class_names}
        windows_scored = 0

        window_sizes = sorted({w for w in PATTERN_WINDOWS.values() if w > 0})
        for window in window_sizes:
            if len(work) < window:
                continue
            for end in range(window, len(work) + 1, max(1, step)):
                window_df = work.iloc[end - window : end]
                feats = sequence_to_feature_vector(window_df).reshape(1, -1)
                proba = model.predict_proba(feats)[0]

                best_name = None
                best_p = -1.0
                for name in class_names:
                    idx = model.class_names.index(name)
                    p = float(proba[idx])
                    proba_sums[name] += p
                    if p > proba_max[name]:
                        proba_max[name] = p
                    if p > best_p:
                        best_p = p
                        best_name = name
                if best_name is not None:
                    win_counts[best_name] += 1
                windows_scored += 1

        if windows_scored:
            avg = {n: proba_sums[n] / windows_scored for n in class_names}
            mass = sum(avg.values()) or 1.0
            for name in class_names:
                relative = avg[name] / mass
                hits_n = int(hit_counts.get(name, 0))
                class_scores[name] = {
                    "pattern": name,
                    "confidence": float(relative),
                    "model_avg_probability": float(avg[name]),
                    "model_max_probability": float(proba_max[name]),
                    "model_window_wins": int(win_counts[name]),
                    "heuristic_hits": hits_n,
                    "windows_scored": windows_scored,
                }
    else:
        for name, count in hit_counts.items():
            class_scores[name] = {
                "pattern": name,
                "confidence": float(count / total_hits) if total_hits else 0.0,
                "model_avg_probability": None,
                "model_max_probability": None,
                "model_window_wins": 0,
                "heuristic_hits": int(count),
                "windows_scored": 0,
            }

    ranked = sorted(
        class_scores.values(),
        key=lambda r: (
            r["confidence"],
            r.get("model_window_wins", 0),
            r["heuristic_hits"],
        ),
        reverse=True,
    )
    return ranked[: max(1, top)]
