"""One-off script: generate the equity curve comparison plot used in the README."""
import matplotlib.pyplot as plt

from backtest.engine import walk_forward_backtest
from data.loader import load_ohlcv
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_classifier import MLClassifierStrategy


def main():
    prices = load_ohlcv("SPY", "2015-01-01", "2023-01-01")

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, strategy in [
        ("Mean Reversion", MeanReversionStrategy()),
        ("ML Classifier", MLClassifierStrategy()),
    ]:
        result = walk_forward_backtest(prices, strategy, train_window=252, test_window=63)
        normalized = result.equity_curve / result.equity_curve.iloc[0] * 100
        ax.plot(normalized.index, normalized.values, label=name)

    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Break-even")
    ax.set_title("Walk-forward out-of-sample equity curves (SPY, 2015-2023)")
    ax.set_ylabel("Equity (normalized to 100)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/equity_curves.png", dpi=150)
    print("Saved assets/equity_curves.png")


if __name__ == "__main__":
    main()
