"""Train per-pattern binary ensemble (plus optional multiclass fallback)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_patterns.models.binary_ensemble import PatternEnsemble
from stock_patterns.models.classifier import PatternClassifier
from stock_patterns.pipeline.dataset import build_mixed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candlestick pattern models")
    parser.add_argument("--samples", type=int, default=200, help="Synthetic samples per pattern")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,JPM,XOM,SPY",
        help="Comma-separated Yahoo tickers for weak-label fine-tuning (empty to disable)",
    )
    parser.add_argument("--ticker", default=None, help="Deprecated single-ticker alias")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/pattern_ensemble.joblib"),
        help="Ensemble index path; per-pattern files go to <stem>_patterns/",
    )
    parser.add_argument(
        "--also-multiclass",
        action="store_true",
        help="Also train/save a calibrated multiclass booster next to the ensemble",
    )
    args = parser.parse_args()

    tickers: list[str] = []
    if args.ticker:
        tickers.append(args.ticker)
    if args.tickers.strip():
        tickers.extend([t.strip() for t in args.tickers.split(",") if t.strip()])
    # Preserve order, unique
    seen = set()
    tickers = [t for t in tickers if not (t.upper() in seen or seen.add(t.upper()))]

    X, y, class_names = build_mixed_dataset(
        samples_per_pattern=args.samples,
        seed=args.seed,
        tickers=tickers,
        period=args.period,
        interval=args.interval,
    )
    print(f"Training on {len(y)} samples across {len(class_names)} classes")

    ensemble = PatternEnsemble(pattern_names=class_names, random_state=args.seed)
    reports = ensemble.fit_from_multiclass(X, y, class_names)
    path = ensemble.save(args.out)
    print(f"Saved ensemble index -> {path.resolve()}")
    print(f"Per-pattern models -> {(path.parent / (path.stem + '_patterns')).resolve()}")

    summary = {
        name: {
            "roc_auc": round(m["roc_auc"], 4),
            "average_precision": round(m["average_precision"], 4),
            "n_pos": m["n_pos"],
        }
        for name, m in sorted(reports.items())
    }
    if summary:
        mean_auc = sum(v["roc_auc"] for v in summary.values()) / len(summary)
        mean_ap = sum(v["average_precision"] for v in summary.values()) / len(summary)
        print(json.dumps({"mean_roc_auc": round(mean_auc, 4), "mean_average_precision": round(mean_ap, 4)}, indent=2))
        print(json.dumps(summary, indent=2))

    if args.also_multiclass:
        multi_path = args.out.with_name("pattern_classifier.joblib")
        multi = PatternClassifier(class_names=class_names, random_state=args.seed)
        report = multi.fit(X, y)
        multi.save(multi_path)
        print(f"Saved multiclass fallback -> {multi_path.resolve()}")
        print(
            json.dumps(
                {k: report[k] for k in ("accuracy", "macro avg", "weighted avg") if k in report},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
