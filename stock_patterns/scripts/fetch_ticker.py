"""Fetch Yahoo Finance OHLCV for a ticker and save CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_patterns.data.yahoo_finance import fetch_ohlcv, save_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance OHLCV data")
    parser.add_argument("--ticker", required=True, help="Symbol, e.g. AAPL, MSFT, BTC-USD")
    parser.add_argument("--period", default="1y", help="Yahoo period if start/end omitted")
    parser.add_argument("--interval", default="1d", help="Bar interval, e.g. 1d, 1h, 15m")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    df = fetch_ohlcv(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
    path = save_ohlcv(df, args.out, ticker=args.ticker)
    print(f"Fetched {len(df)} bars for {args.ticker.upper()} -> {path.resolve()}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
