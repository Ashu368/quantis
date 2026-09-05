"""Z-score mean reversion strategy: fade extreme deviations from a rolling mean."""
import pandas as pd

from data.features import zscore
from strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    def __init__(self, lookback: int = 20, entry_z: float = 1.5, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        z_score = zscore(close, self.lookback)

        position = pd.Series(float("nan"), index=close.index)
        position[z_score < -self.entry_z] = 1.0
        position[z_score > self.entry_z] = -1.0
        position[z_score.abs() < self.exit_z] = 0.0

        return position.ffill().fillna(0.0)
