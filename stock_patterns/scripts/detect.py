"""Detect patterns on a Yahoo Finance ticker (heuristic + optional trained model)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stock_patterns.data.yahoo_finance import fetch_ohlcv
from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.patterns.registry import PATTERN_WINDOWS, detect_patterns
from stock_patterns.pipeline.features import sequence_to_feature_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candlestick / chart patterns")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="6mo")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--model", type=Path, default=None, help="Optional trained .joblib model")
    parser.add_argument("--top", type=int, default=15, help="Max heuristic hits to print")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    hits = detect_patterns(df)
    print(f"Heuristic hits for {args.ticker.upper()}: {len(hits)}")
    for hit in hits[-args.top :]:
        print(json.dumps(hit))

    if args.model and args.model.exists():
        model = PatternClassifier.load(args.model)
        # Score the most recent window large enough for every pattern.
        max_window = max(PATTERN_WINDOWS.values())
        if len(df) >= max_window:
            recent = df.iloc[-max_window:]
            feats = sequence_to_feature_vector(recent).reshape(1, -1)
            label = model.predict_labels(feats)[0]
            proba = model.predict_proba(feats)[0]
            top_idx = int(np.argmax(proba))
            print(
                json.dumps(
                    {
                        "model_prediction": label,
                        "confidence": float(proba[top_idx]),
                        "as_of": str(df.index[-1]),
                    }
                )
            )


if __name__ == "__main__":
    main()
