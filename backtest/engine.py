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


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy,
    train_window: int,
    test_window: int,
    **backtest_kwargs,
) -> BacktestResult:
    """Roll a fixed-size train window forward, refit, and backtest the next
    `test_window` bars out-of-sample. Concatenates all out-of-sample segments
    into one continuous equity curve.

    This is what makes a strategy's reported performance honest for anything
    that needs periodic refitting (e.g. an ML classifier) — no single lucky
    train/test split can inflate the result, since every segment is truly
    unseen by the model at fit time.
    """
    all_returns = []
    all_positions = []

    start = 0
    while start + train_window + test_window <= len(prices):
        train = prices.iloc[start : start + train_window]
        test = prices.iloc[start + train_window : start + train_window + test_window]

        strategy.fit(train)
        signals = strategy.generate_signals(test)

        segment_result = run_backtest(test, signals, **backtest_kwargs)
        all_returns.append(segment_result.returns)
        all_positions.append(segment_result.positions)

        start += test_window

    if not all_returns:
        raise ValueError("Not enough data for even one walk-forward window")

    combined_returns = pd.concat(all_returns)
    combined_positions = pd.concat(all_positions)

    initial_capital = backtest_kwargs.get("initial_capital", 100_000.0)
    equity_curve = initial_capital * (1 + combined_returns).cumprod()

    summary = metrics.summarize(equity_curve, combined_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=combined_returns,
        positions=combined_positions,
        metrics=summary,
    )
