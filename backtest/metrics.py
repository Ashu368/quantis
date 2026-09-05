"""Risk-adjusted performance metrics for a backtested equity curve."""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def win_rate(returns: pd.Series) -> float:
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return float((trades > 0).sum() / len(trades))


def annualized_return(equity_curve: pd.Series) -> float:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def summarize(equity_curve: pd.Series, returns: pd.Series) -> dict:
    return {
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        "annualized_return": annualized_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(returns),
    }
