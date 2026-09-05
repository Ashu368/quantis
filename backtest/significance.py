"""Statistical significance testing for strategy returns.

A backtest Sharpe ratio alone doesn't tell you whether the strategy found real
signal or got lucky. These tests answer: is the mean return significantly
different from zero, given how noisy it is?
"""
import numpy as np
import pandas as pd
from scipy import stats


def t_test_mean_return(returns: pd.Series) -> dict:
    """One-sample t-test: is the mean daily return significantly != 0?"""
    nonzero = returns[returns != 0]
    if len(nonzero) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "n_obs": len(nonzero)}

    t_stat, p_value = stats.ttest_1samp(nonzero, popmean=0.0)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n_obs": len(nonzero)}


def bootstrap_sharpe_ci(
    returns: pd.Series, n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 42
) -> dict:
    """Bootstrap a confidence interval on the annualized Sharpe ratio.

    If the CI includes zero (or is very wide), the Sharpe estimate isn't
    reliable enough to trust on this sample size.
    """
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    n = len(values)
    if n < 2 or values.std() == 0:
        return {"lower": 0.0, "upper": 0.0, "point_estimate": 0.0}

    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        std = sample.std()
        sharpes[i] = 0.0 if std == 0 else np.sqrt(252) * sample.mean() / std

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(sharpes, [alpha, 1 - alpha])
    point = np.sqrt(252) * values.mean() / values.std()

    return {"lower": float(lower), "upper": float(upper), "point_estimate": float(point)}
