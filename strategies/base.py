"""Strategy interface: any strategy turns price data into a position signal."""
from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    def fit(self, prices: pd.DataFrame) -> None:
        """Fit any parameters/models on `prices` (training data only).

        No-op by default. ML-based strategies override this so the model is
        trained exactly once on the training split, never on the data it will
        later be evaluated on — call `fit(train)` before `generate_signals`
        on either the train or test split.
        """
        return None

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return a position series in [-1, 1], indexed like `prices`.

        Signals are shifted by one bar inside the backtest engine, so
        implementations should NOT shift themselves. Must rely only on
        state set up by `fit()`, never fit on `prices` itself.
        """
        raise NotImplementedError
