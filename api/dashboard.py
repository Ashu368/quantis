"""Interactive dashboard: compare strategies, view equity curves and significance.

Usage:
    streamlit run api/dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from backtest import significance
from backtest.engine import run_backtest, train_test_split, walk_forward_backtest
from data.loader import load_ohlcv
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_classifier import MLClassifierStrategy

STRATEGIES = {
    "Mean Reversion (z-score)": MeanReversionStrategy,
    "ML Classifier (gradient boosting)": MLClassifierStrategy,
}

st.set_page_config(page_title="quantis", layout="wide")
st.title("quantis — backtest & validation dashboard")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="SPY")
    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
    start = st.text_input("Start date", value="2015-01-01")
    end = st.text_input("End date", value="2023-01-01")
    mode = st.radio("Validation mode", ["Walk-forward (recommended)", "Single train/test split"])

    if mode == "Walk-forward (recommended)":
        train_window = st.number_input("Train window (bars)", value=252, min_value=30)
        test_window = st.number_input("Test window (bars)", value=63, min_value=10)
    else:
        split = st.text_input("Split date", value="2021-06-01")

    run_clicked = st.button("Run backtest", type="primary")

if run_clicked:
    with st.spinner("Loading data and running backtest..."):
        prices = load_ohlcv(ticker, start, end)
        strategy = STRATEGIES[strategy_name]()

        if mode == "Walk-forward (recommended)":
            result = walk_forward_backtest(prices, strategy, train_window, test_window)
            label = "Walk-forward (out-of-sample only)"
        else:
            train, test = train_test_split(prices, split)
            strategy.fit(train)
            signals = strategy.generate_signals(test)
            result = run_backtest(test, signals)
            label = f"Out-of-sample ({split} onward)"

    st.subheader(label)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total return", f"{result.metrics['total_return']:.1%}")
    col2.metric("Sharpe ratio", f"{result.metrics['sharpe_ratio']:.2f}")
    col3.metric("Max drawdown", f"{result.metrics['max_drawdown']:.1%}")
    col4.metric("Win rate", f"{result.metrics['win_rate']:.1%}")

    t_test = significance.t_test_mean_return(result.returns)
    sharpe_ci = significance.bootstrap_sharpe_ci(result.returns)

    st.subheader("Statistical significance")
    sig_col1, sig_col2 = st.columns(2)
    sig_col1.metric("p-value (mean return != 0)", f"{t_test['p_value']:.4f}")
    sig_col2.metric(
        "Sharpe 95% CI",
        f"[{sharpe_ci['lower']:.2f}, {sharpe_ci['upper']:.2f}]",
    )

    if t_test["p_value"] > 0.05:
        st.warning("NOT statistically significant at p < 0.05 — treat this edge as noise, not a real strategy.")
    else:
        st.success("Statistically significant at p < 0.05.")

    st.subheader("Equity curve")
    st.line_chart(result.equity_curve)

    st.subheader("Position over time")
    st.line_chart(result.positions)
else:
    st.info("Configure a strategy in the sidebar and click **Run backtest**.")
