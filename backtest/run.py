"""CLI entry point: run a strategy backtest and print performance metrics.

Usage:
    python -m backtest.run --ticker SPY --strategy mean_reversion \
        --start 2018-01-01 --end 2023-01-01 --split 2021-06-01

    python -m backtest.run --ticker SPY --strategy ml_classifier \
        --start 2018-01-01 --end 2023-01-01 --walk-forward \
        --train-window 252 --test-window 63
"""
import argparse

from backtest import significance
from backtest.engine import run_backtest, train_test_split, walk_forward_backtest
from data.loader import load_ohlcv
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_classifier import MLClassifierStrategy

STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
    "ml_classifier": MLClassifierStrategy,
}


def print_report(label: str, data, result) -> None:
    print(f"\n=== {label} ({data.index[0].date()} to {data.index[-1].date()}) ===")
    for key, value in result.metrics.items():
        print(f"  {key:20s}: {value:.4f}")

    t_test = significance.t_test_mean_return(result.returns)
    sharpe_ci = significance.bootstrap_sharpe_ci(result.returns)
    print(f"  {'t_stat (mean ret)':20s}: {t_test['t_stat']:.4f}  (p={t_test['p_value']:.4f}, n={t_test['n_obs']})")
    print(
        f"  {'sharpe 95% CI':20s}: [{sharpe_ci['lower']:.2f}, {sharpe_ci['upper']:.2f}] "
        f"(point={sharpe_ci['point_estimate']:.2f})"
    )
    if t_test["p_value"] > 0.05:
        print("  -> NOT statistically significant at p<0.05. Treat this edge as noise.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--strategy", default="mean_reversion", choices=STRATEGIES.keys())
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-01-01")
    parser.add_argument("--split", default="2021-06-01", help="in-sample/out-of-sample split date")
    parser.add_argument("--walk-forward", action="store_true", help="use rolling walk-forward validation instead of a single split")
    parser.add_argument("--train-window", type=int, default=252, help="walk-forward training window size, in bars")
    parser.add_argument("--test-window", type=int, default=63, help="walk-forward test window size, in bars")
    args = parser.parse_args()

    prices = load_ohlcv(args.ticker, args.start, args.end)
    strategy = STRATEGIES[args.strategy]()

    if args.walk_forward:
        result = walk_forward_backtest(prices, strategy, args.train_window, args.test_window)
        print_report("WALK-FORWARD (out-of-sample only)", prices.loc[result.returns.index], result)
        return

    train, test = train_test_split(prices, args.split)
    strategy.fit(train)

    for label, data in [("IN-SAMPLE", train), ("OUT-OF-SAMPLE", test)]:
        signals = strategy.generate_signals(data)
        result = run_backtest(data, signals)
        print_report(label, data, result)


if __name__ == "__main__":
    main()
