# Stock Pattern Predictors

Synthetic candlestick-pattern generators plus a Yahoo Finance fetch pipeline for training pattern recognition models.

## Patterns

Each pattern lives in its own file under `stock_patterns/patterns/` and exposes a random OHLC generator used for bootstrapping training data.

| Pattern | File |
|---|---|
| Bullish Engulfing | `bullish_engulfing.py` |
| Bearish Engulfing | `bearish_engulfing.py` |
| Hammer | `hammer.py` |
| Shooting Star | `shooting_star.py` |
| Doji Star | `doji_star.py` |
| Three Inside Up | `three_inside_up.py` |
| Three Inside Down | `three_inside_down.py` |
| Spinning Top | `spinning_top.py` |
| Head and Shoulders | `head_and_shoulders.py` |
| Inverse Head and Shoulders | `inverse_head_and_shoulders.py` |
| Morning Star | `morning_star.py` |
| Evening Star | `evening_star.py` |
| Piercing Line | `piercing_line.py` |
| Dark Cloud Cover | `dark_cloud_cover.py` |
| Bullish Harami | `bullish_harami.py` |
| Bearish Harami | `bearish_harami.py` |
| Three White Soldiers | `three_white_soldiers.py` |
| Three Black Crows | `three_black_crows.py` |
| Marubozu | `marubozu.py` |
| Tweezer Top / Bottom | `tweezer.py` |
| Double Top / Bottom | `double_top_bottom.py` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate synthetic training data

```bash
python -m stock_patterns.scripts.generate_synthetic --samples 200 --out data/synthetic
```

## Fetch Yahoo Finance data

```bash
python -m stock_patterns.scripts.fetch_ticker --ticker AAPL --period 1y --interval 1d --out data/raw
```

## Train a pattern classifier

Uses synthetic OHLC sequences by default (good for cold start). Optionally mix in labeled Yahoo bars later.

```bash
python -m stock_patterns.scripts.train --samples 500 --out models/pattern_classifier.joblib
```

## Detect top patterns in a date window

```bash
python -m stock_patterns.scripts.detect \
  --ticker AAPL \
  --start 2026-01-01 \
  --end 2026-03-31 \
  --top 5 \
  --model models/pattern_classifier.joblib
```

Add `--json` for machine-readable output.

## Package layout

```
stock_patterns/
  patterns/          # one generator file per pattern
  data/              # Yahoo Finance client
  pipeline/          # dataset assembly + feature extraction
  models/            # sklearn classifier wrapper
  scripts/           # CLI entrypoints
```
