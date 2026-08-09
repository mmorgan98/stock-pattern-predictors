"""Rank candlestick / chart patterns inside a date window."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from stock_patterns.models.binary_ensemble import PatternEnsemble
from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.patterns.registry import PATTERN_MATCHERS, PATTERN_WINDOWS, detect_patterns
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


def _heuristic_strengths(df: pd.DataFrame) -> dict[str, float]:
    """
    Pattern strength in [0, 1] from heuristic hit density and native-window match.
    """
    hits = detect_patterns(df)
    counts: dict[str, int] = defaultdict(int)
    for hit in hits:
        counts[hit["pattern"]] += 1

    strengths: dict[str, float] = {}
    n = max(len(df), 1)
    for name, matcher in PATTERN_MATCHERS.items():
        if name == "no_pattern":
            continue
        window = PATTERN_WINDOWS[name]
        trailing = 1.0 if len(df) >= window and matcher(df.iloc[-window:]) else 0.0
        density = min(1.0, counts.get(name, 0) / max(n / max(window, 1), 1.0))
        strengths[name] = float(min(1.0, 0.65 * trailing + 0.35 * density + (0.15 if counts.get(name, 0) else 0.0)))
        if counts.get(name, 0) and strengths[name] < 0.2:
            strengths[name] = 0.2
    return strengths


def _peak_scores_ensemble(
    model: PatternEnsemble,
    df: pd.DataFrame,
    step: int = 1,
    min_confidence: float = 0.45,
) -> dict[str, dict]:
    peaks: dict[str, list[float]] = defaultdict(list)
    windows_scored = 0
    window_sizes = sorted({PATTERN_WINDOWS[n] for n in model.pattern_names if n in PATTERN_WINDOWS})

    for window in window_sizes:
        if len(df) < window:
            continue
        for end in range(window, len(df) + 1, max(1, step)):
            window_df = df.iloc[end - window : end]
            # Prefer native window size for each pattern; also score with context.
            feats = sequence_to_feature_vector(window_df).reshape(1, -1)
            scores = model.score_vector(feats[0])
            windows_scored += 1
            for name, p in scores.items():
                native = PATTERN_WINDOWS.get(name, window)
                # Only accept scores near the pattern's native scale.
                if abs(window - native) > 2 and window != max(window_sizes):
                    continue
                if p >= min_confidence:
                    peaks[name].append(float(p))

    out: dict[str, dict] = {}
    for name, vals in peaks.items():
        vals_sorted = sorted(vals, reverse=True)
        topk = vals_sorted[: min(5, len(vals_sorted))]
        out[name] = {
            "peak_confidence": float(vals_sorted[0]),
            "mean_peak_confidence": float(np.mean(topk)),
            "peak_count": len(vals),
            "windows_scored": windows_scored,
        }
    return out


def _peak_scores_multiclass(
    model: PatternClassifier,
    df: pd.DataFrame,
    step: int = 1,
    min_confidence: float = 0.45,
) -> dict[str, dict]:
    class_names = [n for n in model.class_names if n != "no_pattern"]
    peaks: dict[str, list[float]] = defaultdict(list)
    windows_scored = 0
    window_sizes = sorted({w for w in PATTERN_WINDOWS.values() if w > 0})

    for window in window_sizes:
        if len(df) < window:
            continue
        for end in range(window, len(df) + 1, max(1, step)):
            window_df = df.iloc[end - window : end]
            feats = sequence_to_feature_vector(window_df).reshape(1, -1)
            proba = model.predict_proba(feats)[0]
            windows_scored += 1
            for name in class_names:
                idx = model.class_names.index(name)
                p = float(proba[idx])
                if p >= min_confidence:
                    peaks[name].append(p)

    out: dict[str, dict] = {}
    for name, vals in peaks.items():
        vals_sorted = sorted(vals, reverse=True)
        topk = vals_sorted[: min(5, len(vals_sorted))]
        out[name] = {
            "peak_confidence": float(vals_sorted[0]),
            "mean_peak_confidence": float(np.mean(topk)),
            "peak_count": len(vals),
            "windows_scored": windows_scored,
        }
    return out


def rank_patterns(
    df: pd.DataFrame,
    top: int = 5,
    model: PatternEnsemble | PatternClassifier | None = None,
    step: int = 1,
    min_confidence: float = 0.45,
    model_weight: float = 0.6,
    heuristic_weight: float = 0.4,
) -> list[dict]:
    """
    Rank top patterns using peak-window model scores ensembled with heuristics.

    confidence = model_weight * mean_top_peaks + heuristic_weight * heuristic_strength
    """
    if df is None or df.empty:
        return []

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]
    heur = _heuristic_strengths(work)

    model_peaks: dict[str, dict] = {}
    if isinstance(model, PatternEnsemble):
        model_peaks = _peak_scores_ensemble(model, work, step=step, min_confidence=min_confidence)
    elif isinstance(model, PatternClassifier):
        model_peaks = _peak_scores_multiclass(model, work, step=step, min_confidence=min_confidence)

    names = sorted(set(heur) | set(model_peaks))
    ranked_rows: list[dict] = []
    mw = model_weight
    hw = heuristic_weight
    if model is None:
        mw, hw = 0.0, 1.0
    total_w = mw + hw
    mw, hw = mw / total_w, hw / total_w

    for name in names:
        h = float(heur.get(name, 0.0))
        peak = model_peaks.get(name, {})
        model_score = float(peak.get("mean_peak_confidence", 0.0))
        # If model never cleared the peak threshold, fall back lightly to raw absence.
        if name not in model_peaks and model is not None:
            model_score = 0.0
        confidence = mw * model_score + hw * h
        if confidence <= 0:
            continue
        ranked_rows.append(
            {
                "pattern": name,
                "confidence": float(confidence),
                "model_peak_confidence": float(peak.get("peak_confidence", 0.0)),
                "model_mean_peak_confidence": model_score,
                "model_peak_count": int(peak.get("peak_count", 0)),
                "heuristic_strength": h,
                "windows_scored": int(peak.get("windows_scored", 0)),
            }
        )

    ranked_rows.sort(
        key=lambda r: (r["confidence"], r["model_peak_confidence"], r["heuristic_strength"]),
        reverse=True,
    )
    return ranked_rows[: max(1, top)]
