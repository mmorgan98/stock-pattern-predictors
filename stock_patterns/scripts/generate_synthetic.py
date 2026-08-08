"""Generate synthetic OHLC CSVs for every registered pattern."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_patterns.pipeline.dataset import save_synthetic_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic candlestick pattern samples")
    parser.add_argument("--samples", type=int, default=100, help="Samples per pattern")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    out = save_synthetic_csvs(args.out, samples_per_pattern=args.samples, seed=args.seed)
    print(f"Wrote synthetic samples to {out.resolve()}")


if __name__ == "__main__":
    main()
