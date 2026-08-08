"""Yahoo Finance OHLCV fetch pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV bars for a ticker from Yahoo Finance.

    Prefer `period` (e.g. 1mo, 6mo, 1y, 5y, max) or explicit start/end dates.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("ticker is required")

    kwargs: dict = {
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
    }
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    raw = yf.download(symbol, **kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"No Yahoo Finance data returned for {symbol}")

    # yfinance may return MultiIndex columns for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df = raw.rename(columns=str.lower)
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Yahoo response missing columns {missing} for {symbol}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.attrs["ticker"] = symbol
    df.attrs["interval"] = interval
    return df


def save_ohlcv(df: pd.DataFrame, out_dir: str | Path, ticker: str | None = None) -> Path:
    """Persist OHLCV frame to CSV under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbol = (ticker or df.attrs.get("ticker") or "unknown").upper()
    interval = df.attrs.get("interval", "1d")
    path = out / f"{symbol}_{interval}.csv"
    df.to_csv(path)
    return path


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a previously saved OHLCV CSV."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    return df
