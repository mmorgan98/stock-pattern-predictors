"""Train a pattern classifier on synthetic (and optional Yahoo) data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_patterns.data.yahoo_finance import fetch_ohlcv
from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.pipeline.dataset import build_synthetic_dataset, label_yahoo_with_heuristics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candlestick pattern classifier")
    parser.add_argument("--samples", type=int, default=200, help="Synthetic samples per pattern")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticker", default=None, help="Optional Yahoo ticker to mix in weak labels")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", type=Path, default=Path("models/pattern_classifier.joblib"))
    args = parser.parse_args()

    X, y, class_names = build_synthetic_dataset(
        samples_per_pattern=args.samples,
        seed=args.seed,
    )

    if args.ticker:
        yahoo = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
        Xy, yy, _ = label_yahoo_with_heuristics(yahoo)
        if len(yy):
            import numpy as np

            X = np.vstack([X, Xy])
            y = np.concatenate([y, yy])
            print(f"Added {len(yy)} weakly labeled Yahoo samples from {args.ticker.upper()}")

    model = PatternClassifier(class_names=class_names, random_state=args.seed)
    report = model.fit(X, y)
    path = model.save(args.out)
    print(f"Saved model -> {path.resolve()}")
    print(json.dumps({k: report[k] for k in ("accuracy", "macro avg", "weighted avg") if k in report}, indent=2))


if __name__ == "__main__":
    main()
