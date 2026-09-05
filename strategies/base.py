"""Strategy interface: any strategy turns price data into a position signal."""
from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return a position series in [-1, 1], indexed like `prices`.

        Signals are shifted by one bar inside the backtest engine, so
        implementations should NOT shift themselves.
        """
        raise NotImplementedError
