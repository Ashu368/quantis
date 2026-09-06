"""Market microstructure analysis: does order flow imbalance predict short-horizon returns?

Standard technique from market microstructure research (Cont, Kukanov &
Stoikov, 2014 — "The Price Impact of Order Book Events"): regress the next
tick's return on the current order flow imbalance (OFI). A significant
positive coefficient means buy-side pressure now predicts upward price
movement next tick.
"""
import pandas as pd
from scipy import stats


def compute_next_tick_return(book: pd.DataFrame) -> pd.Series:
    return book["mid_price"].pct_change().shift(-1)


def ofi_predictive_regression(book: pd.DataFrame) -> dict:
    """Simple linear regression: next_tick_return ~ order_flow_imbalance.

    Returns the slope, its statistical significance, and R^2 — a
    significant slope with near-zero R^2 is normal and expected for
    microstructure signals (they explain a tiny fraction of variance but
    are still real and exploitable at scale).
    """
    ofi = book["order_flow_imbalance"].iloc[:-1]
    next_return = compute_next_tick_return(book).iloc[:-1]

    valid = ~(ofi.isna() | next_return.isna())
    ofi, next_return = ofi[valid], next_return[valid]

    slope, intercept, r_value, p_value, std_err = stats.linregress(ofi, next_return)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_obs": int(valid.sum()),
    }


def spread_analysis(book: pd.DataFrame) -> dict:
    spread = book["ask_price"] - book["bid_price"]
    return {
        "mean_spread": float(spread.mean()),
        "spread_volatility": float(spread.std()),
    }


def summarize(book: pd.DataFrame) -> dict:
    regression = ofi_predictive_regression(book)
    spread_stats = spread_analysis(book)
    return {**regression, **spread_stats}
