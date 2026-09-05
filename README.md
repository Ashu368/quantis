# quantis

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

## Status

🚧 Active development — backtesting engine, walk-forward validation, and significance testing are done. Next: order book/microstructure analysis, CI, packaging.

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
