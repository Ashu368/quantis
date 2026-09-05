import numpy as np
import pandas as pd
import pytest

from backtest.engine import walk_forward_backtest
from strategies.mean_reversion import MeanReversionStrategy


def _fake_prices(n=400, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": prices}, index=dates)


def test_walk_forward_produces_continuous_equity_curve():
    prices = _fake_prices()
    strategy = MeanReversionStrategy()
    result = walk_forward_backtest(prices, strategy, train_window=100, test_window=50)

    assert len(result.equity_curve) > 0
    assert result.equity_curve.index.is_monotonic_increasing
    assert "sharpe_ratio" in result.metrics


def test_walk_forward_raises_when_data_too_short():
    prices = _fake_prices(n=50)
    strategy = MeanReversionStrategy()
    with pytest.raises(ValueError):
        walk_forward_backtest(prices, strategy, train_window=100, test_window=50)
