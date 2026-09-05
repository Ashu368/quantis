"""Shared feature engineering for strategies (technical indicators from OHLCV data)."""
import pandas as pd


def zscore(series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std()
    return (series - rolling_mean) / rolling_std


def rolling_volatility(returns: pd.Series, lookback: int) -> pd.Series:
    return returns.rolling(lookback).std()


def momentum(series: pd.Series, lookback: int) -> pd.Series:
    return series.pct_change(lookback)


def build_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a standard feature set from OHLCV `prices` (must have a 'close' column).

    Every feature here is computed only from data available up to and including
    the current row, so no shifting is needed here — the caller is responsible
    for shifting FEATURES (not the target) if predicting a future label.
    """
    close = prices["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=prices.index)
    features["return_1d"] = returns
    features["return_5d"] = momentum(close, 5)
    features["return_10d"] = momentum(close, 10)
    features["volatility_10d"] = rolling_volatility(returns, 10)
    features["zscore_20d"] = zscore(close, 20)

    return features
