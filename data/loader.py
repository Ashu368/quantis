"""Historical market data loading with local caching."""
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load daily OHLCV data for `ticker` between `start` and `end` (YYYY-MM-DD).

    Caches to a local parquet file so repeated backtests don't re-hit the API.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file, index_col="date", parse_dates=True)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    df.to_csv(cache_file)
    return df
