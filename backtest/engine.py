"""Event-driven backtesting engine with transaction cost and slippage modeling."""
from dataclasses import dataclass

import pandas as pd

from backtest import metrics


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: dict


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Simulate trading `signals` (position sizes in [-1, 1]) against `prices`.

    `signals` must be shifted by the caller if it needs to avoid lookahead bias
    (i.e. signal computed on day t should only be tradable on day t+1).
    """
    close = prices["close"]
    positions = signals.reindex(close.index).fillna(0.0)

    price_returns = close.pct_change().fillna(0.0)
    strategy_returns = positions.shift(1).fillna(0.0) * price_returns

    position_changes = positions.diff().abs().fillna(0.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    costs = position_changes * cost_rate
    net_returns = strategy_returns - costs

    equity_curve = initial_capital * (1 + net_returns).cumprod()
    equity_curve.iloc[0] = initial_capital

    summary = metrics.summarize(equity_curve, net_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        positions=positions,
        metrics=summary,
    )


def train_test_split(prices: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split price data into in-sample / out-of-sample sets at `split_date`."""
    train = prices[prices.index < split_date]
    test = prices[prices.index >= split_date]
    return train, test
