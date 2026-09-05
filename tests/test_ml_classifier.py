import numpy as np
import pandas as pd
import pytest

from strategies.ml_classifier import MLClassifierStrategy


def _fake_prices(n=200, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": prices}, index=dates)


def test_generate_signals_without_fit_raises():
    strategy = MLClassifierStrategy()
    prices = _fake_prices()
    with pytest.raises(RuntimeError):
        strategy.generate_signals(prices)


def test_fit_then_predict_on_disjoint_data_does_not_error():
    prices = _fake_prices()
    train, test = prices.iloc[:150], prices.iloc[150:]

    strategy = MLClassifierStrategy()
    strategy.fit(train)
    signals = strategy.generate_signals(test)

    assert len(signals) == len(test)
    assert signals.isin([-1.0, 0.0, 1.0]).all()


def test_fit_raises_on_too_little_data():
    strategy = MLClassifierStrategy()
    tiny = _fake_prices(n=10)
    with pytest.raises(ValueError):
        strategy.fit(tiny)


def test_model_is_only_fit_once_on_training_data(monkeypatch):
    prices = _fake_prices()
    train, test = prices.iloc[:150], prices.iloc[150:]

    strategy = MLClassifierStrategy()
    fit_calls = []
    original_fit = strategy.model.fit

    def spy_fit(X, y):
        fit_calls.append(len(X))
        return original_fit(X, y)

    monkeypatch.setattr(strategy.model, "fit", spy_fit)

    strategy.fit(train)
    strategy.generate_signals(train)
    strategy.generate_signals(test)

    assert len(fit_calls) == 1
    assert fit_calls[0] <= len(train)
