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

## Status

🚧 Early development — Phase 1 (backtesting framework).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m backtest.run --strategy mean_reversion --start 2020-01-01 --end 2023-01-01
```

## License

MIT
