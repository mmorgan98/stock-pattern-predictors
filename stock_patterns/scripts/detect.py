"""Detect patterns on a Yahoo Finance ticker (heuristic + optional trained model)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stock_patterns.data.yahoo_finance import fetch_ohlcv
from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.patterns.registry import PATTERN_WINDOWS, detect_patterns
from stock_patterns.pipeline.features import sequence_to_feature_vector


def _summarize_hits(hits: list[dict], recent: int = 10) -> dict:
    counts = Counter(h["pattern"] for h in hits)
    latest_by_pattern: dict[str, dict] = {}
    for hit in hits:
        prev = latest_by_pattern.get(hit["pattern"])
        if prev is None or hit["end_date"] >= prev["end_date"]:
            latest_by_pattern[hit["pattern"]] = hit
    recent_hits = sorted(latest_by_pattern.values(), key=lambda h: h["end_date"], reverse=True)[:recent]
    return {
        "total_hits": len(hits),
        "by_pattern": dict(counts.most_common()),
        "latest_per_pattern": recent_hits,
    }


def _model_votes(model: PatternClassifier, df, step: int = 3, min_confidence: float = 0.35) -> list[dict]:
    """
    Score rolling windows across pattern window sizes.

    Uses argmax votes only when confidence clears min_confidence, so unstructured
    market segments do not all collapse into one long-window chart pattern.
    """
    class_names = model.class_names
    if not class_names:
        return []

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    votes = np.zeros(len(class_names), dtype=float)
    conf_sums = np.zeros(len(class_names), dtype=float)
    n_confident = 0

    windows = sorted({w for w in PATTERN_WINDOWS.values() if w > 0})
    for window in windows:
        if len(df) < window:
            continue
        for end in range(window, len(df) + 1, step):
            window_df = df.iloc[end - window : end]
            feats = sequence_to_feature_vector(window_df).reshape(1, -1)
            proba = model.predict_proba(feats)[0]
            top_idx = int(np.argmax(proba))
            top_name = class_names[top_idx]
            top_p = float(proba[top_idx])
            if top_p < min_confidence:
                continue
            if top_name == "no_pattern":
                continue
            votes[top_idx] += 1.0
            conf_sums[top_idx] += top_p
            n_confident += 1

    if n_confident == 0:
        return []

    ranked = sorted(
        (
            {
                "pattern": class_names[i],
                "window_votes": int(votes[i]),
                "avg_confidence": float(conf_sums[i] / votes[i]) if votes[i] else 0.0,
                "vote_share": float(votes[i] / n_confident),
                "confident_windows": n_confident,
            }
            for i in range(len(class_names))
            if votes[i] > 0 and class_names[i] != "no_pattern"
        ),
        key=lambda r: (r["window_votes"], r["avg_confidence"]),
        reverse=True,
    )
    return ranked[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candlestick / chart patterns")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="6mo")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--model", type=Path, default=None, help="Optional trained .joblib model")
    parser.add_argument("--top", type=int, default=10, help="Latest unique patterns to print")
    parser.add_argument("--model-step", type=int, default=3, help="Step size for rolling model windows")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Min model confidence to count a vote")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    hits = detect_patterns(df)
    summary = _summarize_hits(hits, recent=args.top)

    print(f"Heuristic hits for {args.ticker.upper()}: {summary['total_hits']}")
    print("Counts by pattern:")
    for pattern, count in summary["by_pattern"].items():
        print(f"  {pattern}: {count}")
    if not summary["by_pattern"]:
        print("  (none)")

    print("Latest hit per pattern:")
    for hit in summary["latest_per_pattern"]:
        print(json.dumps(hit))
    if not summary["latest_per_pattern"]:
        print("  (none)")

    if args.model and args.model.exists():
        model = PatternClassifier.load(args.model)
        ranked = _model_votes(
            model,
            df,
            step=max(1, args.model_step),
            min_confidence=args.min_confidence,
        )
        print("Model rolling-window ranking:")
        for row in ranked:
            print(json.dumps(row))
        if not ranked:
            print("  (no confident pattern votes)")


if __name__ == "__main__":
    main()
