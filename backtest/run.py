"""CLI entry point: run a strategy backtest and print performance metrics.

Usage:
    python -m backtest.run --ticker SPY --strategy mean_reversion \
        --start 2018-01-01 --end 2023-01-01 --split 2021-06-01
"""
import argparse

from backtest.engine import run_backtest, train_test_split
from data.loader import load_ohlcv
from strategies.mean_reversion import MeanReversionStrategy

STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--strategy", default="mean_reversion", choices=STRATEGIES.keys())
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-01-01")
    parser.add_argument("--split", default="2021-06-01", help="in-sample/out-of-sample split date")
    args = parser.parse_args()

    prices = load_ohlcv(args.ticker, args.start, args.end)
    train, test = train_test_split(prices, args.split)

    strategy = STRATEGIES[args.strategy]()

    for label, data in [("IN-SAMPLE", train), ("OUT-OF-SAMPLE", test)]:
        signals = strategy.generate_signals(data)
        result = run_backtest(data, signals)

        print(f"\n=== {label} ({data.index[0].date()} to {data.index[-1].date()}) ===")
        for key, value in result.metrics.items():
            print(f"  {key:20s}: {value:.4f}")


if __name__ == "__main__":
    main()
