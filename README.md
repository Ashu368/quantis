# quantis

[![CI](https://github.com/Ashu368/quantis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashu368/quantis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A backtesting and statistical-edge-detection engine for algorithmic trading strategies.

## What it does

- Ingests historical market data (OHLCV, order book snapshots)
- Runs statistical analysis to find exploitable patterns (mean reversion, momentum, volatility clustering)
- Backtests strategies with realistic assumptions: slippage, transaction costs, position sizing
- Reports risk-adjusted performance: Sharpe ratio, max drawdown, win rate, out-of-sample validation
- Optional: live paper trading via a broker API

## Why

Most "trading bot" projects overfit to historical data and report meaningless backtest returns. This project is built to avoid that:

- Strict train/test split (no lookahead bias)
- Walk-forward validation
- Transaction cost and slippage modeling
- Statistical significance testing on any discovered "edge"

## Structure

```
quantis/
├── data/         # data loaders, caching, historical datasets
├── backtest/     # backtesting engine (portfolio simulation, metrics)
├── strategies/   # strategy implementations
├── analysis/     # exploratory statistical analysis notebooks/scripts
├── api/          # optional live trading / dashboard API
└── tests/        # unit tests
```

## Findings so far (honest, not cherry-picked)

Running both strategies on SPY (2015-2023) with walk-forward validation (252-day train / 63-day test windows, refit every window):

| Strategy | OOS Sharpe | p-value (mean return != 0) | Verdict |
|---|---|---|---|
| Mean reversion (z-score) | -0.09 | 0.82 | Not significant — edge is noise |
| ML classifier (gradient boosting) | -1.09 | 0.0045 | Significant — model reliably underperforms |

Two things worth noting:
- A naive single train/test split on the ML classifier showed an in-sample Sharpe of **7.0** — an obviously overfit, meaningless number. Walk-forward validation exposes this: the model doesn't generalize, and that failure is itself statistically significant (p=0.0045), not just noisy variance.
- This is the whole point of the project: report what the statistics actually say, not the most flattering split. Reproduce with `--walk-forward` yourself (see Usage below).

## Dashboard

An interactive Streamlit dashboard to configure a strategy, pick walk-forward or single-split validation, and see the equity curve plus statistical significance live:

```bash
streamlit run api/dashboard.py
```

## Microstructure analysis

`analysis/` demonstrates order-flow-imbalance analysis (Cont, Kukanov & Stoikov, 2014) — regressing next-tick returns on order flow imbalance. **Real tick-level order book data is paywalled**, so this runs on a synthetic order book with a documented, injected effect (see `analysis/orderbook_sim.py`) rather than claiming a discovery on real markets we don't have data for. The point is the methodology, which applies unchanged to a real LOBSTER-format book:

```bash
python -m analysis.run_microstructure_analysis
```

A correctly-calibrated microstructure effect looks like this: R² ≈ 0.002 (tiny — as expected, these effects explain almost none of the variance) but p < 0.001 (highly significant given enough ticks). A null control dataset with no injected effect correctly shows p ≈ 0.5. Getting R² anywhere near 1.0 here would be a red flag that the "signal" is an artifact, not microstructure noise.

## Status

🚧 Active development — backtesting engine, ML strategy, walk-forward validation, significance testing, CI, and microstructure analysis are done.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# single train/test split
python -m backtest.run --strategy mean_reversion --start 2020-01-01 --end 2023-01-01

# walk-forward validation (recommended — avoids lucky-split overfitting)
python -m backtest.run --ticker SPY --strategy ml_classifier \
    --start 2015-01-01 --end 2023-01-01 --walk-forward \
    --train-window 252 --test-window 63
```

## License

MIT
