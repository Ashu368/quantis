import numpy as np
import pandas as pd

from backtest import significance


def test_t_test_zero_returns_not_significant():
    returns = pd.Series(np.zeros(50))
    result = significance.t_test_mean_return(returns)
    assert result["p_value"] == 1.0


def test_t_test_strong_positive_drift_is_significant():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.01, 0.001, 200))
    result = significance.t_test_mean_return(returns)
    assert result["p_value"] < 0.05


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    ci = significance.bootstrap_sharpe_ci(returns, n_bootstrap=200)
    assert ci["lower"] <= ci["point_estimate"] <= ci["upper"]


def test_bootstrap_ci_handles_zero_variance():
    returns = pd.Series(np.zeros(20))
    ci = significance.bootstrap_sharpe_ci(returns, n_bootstrap=50)
    assert ci == {"lower": 0.0, "upper": 0.0, "point_estimate": 0.0}
