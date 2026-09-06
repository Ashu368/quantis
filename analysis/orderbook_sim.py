"""Synthetic limit order book simulator.

Real tick-level order book data (LOBSTER, exchange feeds) is paywalled, so
this generates a synthetic book with a realistic, DOCUMENTED order-flow
imbalance (OFI) effect: when buy-side order flow outweighs sell-side flow,
the mid-price is nudged upward on the next tick, plus noise. This lets us
demonstrate the actual statistical methodology (order flow imbalance as a
short-horizon price predictor, per Cont/Kukanov/Stoikov 2014) without
claiming a discovery on real markets we don't have data for.

Every function in `microstructure.py` that consumes this data would work
identically on a real LOBSTER-format book.
"""
import numpy as np
import pandas as pd


def simulate_order_book(
    n_ticks: int = 20_000,
    initial_mid: float = 100.0,
    spread: float = 0.02,
    ofi_impact: float = 0.02,
    noise_std: float = 0.2,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic tick-by-tick order book.

    `ofi_impact` controls how strongly order flow imbalance at tick t
    pushes the mid-price at tick t+1 — set to 0.0 to generate a null dataset
    with no real effect (useful for testing the analysis doesn't hallucinate
    signal that isn't there).
    """
    rng = np.random.default_rng(seed)

    bid_volume = rng.exponential(scale=100, size=n_ticks)
    ask_volume = rng.exponential(scale=100, size=n_ticks)
    ofi = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    mid_prices = np.empty(n_ticks)
    mid_prices[0] = initial_mid
    noise = rng.normal(0, noise_std, n_ticks)

    for t in range(1, n_ticks):
        mid_prices[t] = mid_prices[t - 1] + ofi_impact * ofi[t - 1] + noise[t]

    return pd.DataFrame(
        {
            "mid_price": mid_prices,
            "bid_price": mid_prices - spread / 2,
            "ask_price": mid_prices + spread / 2,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "order_flow_imbalance": ofi,
        }
    )
