import numpy as np
import pandas as pd
import pytest

from backtest import metrics
from backtest.engine import run_backtest, train_test_split


def _fake_prices(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": prices}, index=dates)


def test_flat_signal_produces_zero_return():
    prices = _fake_prices()
    signals = pd.Series(0.0, index=prices.index)
    result = run_backtest(prices, signals)
    assert result.metrics["total_return"] == 0.0


def test_train_test_split_is_disjoint_and_ordered():
    prices = _fake_prices()
    split = prices.index[50]
    train, test = train_test_split(prices, str(split.date()))
    assert train.index[-1] < test.index[0]
    assert len(train) + len(test) == len(prices)


def test_sharpe_ratio_zero_variance_returns_zero():
    returns = pd.Series([0.0] * 10)
    assert metrics.sharpe_ratio(returns) == 0.0


def test_max_drawdown_is_negative_or_zero():
    equity = pd.Series([100, 110, 90, 95, 120])
    dd = metrics.max_drawdown(equity)
    assert dd <= 0.0
    assert dd == pytest.approx((90 - 110) / 110)
