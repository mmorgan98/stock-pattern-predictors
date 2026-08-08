"""Detect top patterns for a ticker inside a date/time window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_patterns.data.yahoo_finance import fetch_ohlcv
from stock_patterns.models.classifier import PatternClassifier
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
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top patterns to return (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pattern_classifier.joblib"),
        help="Trained model path (optional; omit or missing file = heuristics only)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Rolling-window step for model scoring",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object instead of a short table",
    )
    return parser.parse_args()


def _fetch_with_lookback(
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Fetch OHLCV with enough pre-window bars for the largest pattern window."""
    lookback_bars = max(PATTERN_WINDOWS.values())
    start_ts = pd.Timestamp(start)
    # Approximate calendar padding; daily uses business days, intraday uses hours.
    if interval.endswith("d"):
        padded_start = (start_ts - pd.Timedelta(days=lookback_bars * 3)).strftime("%Y-%m-%d")
    else:
        padded_start = (start_ts - pd.Timedelta(days=max(7, lookback_bars))).strftime("%Y-%m-%d")

    # yfinance end is exclusive for daily; add one day so --end is inclusive.
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
    if args.model and args.model.exists():
        model = PatternClassifier.load(args.model)

    ranked = rank_patterns(
        window_df,
        top=args.top,
        model=model,
        step=max(1, args.step),
    )

    payload = {
        "ticker": args.ticker.upper(),
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "bars": int(len(window_df)),
        "model": str(args.model) if model is not None else None,
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
            f"hits={row['heuristic_hits']}"
            + (
                f"  model_avg={row['model_avg_probability']:.3f}"
                if row.get("model_avg_probability") is not None
                else ""
            )
        )


if __name__ == "__main__":
    main()
