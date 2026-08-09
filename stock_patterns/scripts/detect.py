"""Detect top patterns for a ticker inside a date/time window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_patterns.data.yahoo_finance import fetch_ohlcv
from stock_patterns.models.classifier import load_any_model
from stock_patterns.patterns.registry import PATTERN_WINDOWS
from stock_patterns.pipeline.ranking import rank_patterns, slice_date_window


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank top candlestick/chart patterns in a date window"
    )
    parser.add_argument("--ticker", required=True, help="Symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Window start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Window end date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Bar interval, e.g. 1d, 1h")
    parser.add_argument("--top", type=int, default=5, help="Top N patterns (default: 5)")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pattern_ensemble.joblib"),
        help="Ensemble index or multiclass .joblib (optional)",
    )
    parser.add_argument("--step", type=int, default=1, help="Rolling-window step")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Min model probability for a window to count as a peak",
    )
    parser.add_argument("--model-weight", type=float, default=0.6)
    parser.add_argument("--heuristic-weight", type=float, default=0.4)
    parser.add_argument("--json", action="store_true", help="Print JSON object")
    return parser.parse_args()


def _fetch_with_lookback(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    lookback_bars = max(PATTERN_WINDOWS.values())
    start_ts = pd.Timestamp(start)
    if interval.endswith("d"):
        padded_start = (start_ts - pd.Timedelta(days=lookback_bars * 3)).strftime("%Y-%m-%d")
    else:
        padded_start = (start_ts - pd.Timedelta(days=max(7, lookback_bars))).strftime("%Y-%m-%d")
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return fetch_ohlcv(
        ticker=ticker,
        interval=interval,
        start=padded_start,
        end=end_exclusive,
    )


def main() -> None:
    args = _parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")

    raw = _fetch_with_lookback(args.ticker, args.start, args.end, args.interval)
    window_df = slice_date_window(raw, start=args.start, end=args.end)
    if window_df.empty:
        raise SystemExit(f"No bars found for {args.ticker.upper()} in {args.start}..{args.end}")

    model = None
    model_path = args.model
    if model_path and model_path.exists():
        model = load_any_model(model_path)
    elif Path("models/pattern_classifier.joblib").exists():
        model_path = Path("models/pattern_classifier.joblib")
        model = load_any_model(model_path)

    ranked = rank_patterns(
        window_df,
        top=args.top,
        model=model,
        step=max(1, args.step),
        min_confidence=args.min_confidence,
        model_weight=args.model_weight,
        heuristic_weight=args.heuristic_weight,
    )

    payload = {
        "ticker": args.ticker.upper(),
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "bars": int(len(window_df)),
        "model": str(model_path) if model is not None else None,
        "top": ranked,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(
        f"{payload['ticker']}  {payload['start']} -> {payload['end']}  "
        f"({payload['bars']} bars)  top {args.top}"
    )
    if not ranked:
        print("  (no patterns found)")
        return
    for i, row in enumerate(ranked, start=1):
        print(
            f"{i:>2}. {row['pattern']:<28} "
            f"confidence={row['confidence']:.3f}  "
            f"model_peak={row['model_peak_confidence']:.3f}  "
            f"heuristic={row['heuristic_strength']:.3f}  "
            f"peaks={row['model_peak_count']}"
        )


if __name__ == "__main__":
    main()
