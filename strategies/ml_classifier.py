"""ML-driven strategy: a classifier predicts next-day direction from engineered features.

The model is fit ONLY inside `fit()`, which the caller must invoke on the
training split before calling `generate_signals()` on either split. This
mirrors the anti-lookahead discipline already used by `backtest.engine`.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data.features import build_feature_matrix
from strategies.base import Strategy


class MLClassifierStrategy(Strategy):
    def __init__(self, confidence_threshold: float = 0.55, random_state: int = 42):
        self.confidence_threshold = confidence_threshold
        self.model = GradientBoostingClassifier(random_state=random_state)
        self._is_fitted = False

    def _make_training_set(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = build_feature_matrix(prices)
        # Target: did price go UP over the next bar? Built from prices the
        # features were derived from, so this label only ever uses this same
        # split's own future — fine for fitting, since `fit` is only ever
        # called with the training split.
        next_return = prices["close"].pct_change().shift(-1)
        target = (next_return > 0).astype(int)

        valid = features.dropna().index.intersection(target.dropna().index)
        return features.loc[valid], target.loc[valid]

    def fit(self, prices: pd.DataFrame) -> None:
        X_train, y_train = self._make_training_set(prices)
        if len(X_train) < 30:
            raise ValueError("Not enough training rows to fit MLClassifierStrategy")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("MLClassifierStrategy.fit() must be called before generate_signals()")

        features = build_feature_matrix(prices)
        position = pd.Series(0.0, index=prices.index)

        valid_rows = features.dropna()
        if valid_rows.empty:
            return position

        probs_up = self.model.predict_proba(valid_rows)[:, 1]
        signal = np.where(
            probs_up > self.confidence_threshold,
            1.0,
            np.where(probs_up < 1 - self.confidence_threshold, -1.0, 0.0),
        )
        position.loc[valid_rows.index] = signal
        return position
