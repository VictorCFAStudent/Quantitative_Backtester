"""data.py — Clean & prepare data."""

import pandas as pd
import numpy as np
import yfinance as yf


def validate_ticker(symbol: str) -> bool:
    """Check whether a ticker symbol exists on Yahoo Finance.

    Uses fast_info which is a lightweight metadata call —
    much faster than downloading price history.
    """
    try:
        info = yf.Ticker(symbol).fast_info
        # fast_info returns an object; check that it has a valid market cap
        # or last_price — if the ticker doesn't exist, these will error
        _ = info["lastPrice"]
        return True
    except Exception:
        return False


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    # Standardize to a simple wide price DataFrame
    if len(tickers) == 1:
        # yfinance returns a single-level column for one ticker
        if "Close" not in data.columns:
            raise ValueError(
                f"'Close' column missing from yfinance data for {tickers[0]}. "
                f"Available columns: {list(data.columns)}"
            )
        prices = data[["Close"]].copy()
        prices.columns = tickers
    else:
        # multi-ticker: take close prices
        if "Close" not in data.columns:
            raise ValueError(
                "'Close' column missing from yfinance multi-ticker data. "
                f"Available columns: {list(data.columns)}"
            )
        prices = data["Close"].copy()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def download_open_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download open prices for the given tickers.

    Used for daily-rebalancing mode where trades execute at next open.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if len(tickers) == 1:
        if "Open" not in data.columns:
            raise ValueError(
                f"'Open' column missing from yfinance data for {tickers[0]}. "
                f"Available columns: {list(data.columns)}"
            )
        prices = data[["Open"]].copy()
        prices.columns = tickers
    else:
        if "Open" not in data.columns:
            raise ValueError(
                "'Open' column missing from yfinance multi-ticker data. "
                f"Available columns: {list(data.columns)}"
            )
        prices = data["Open"].copy()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame to log returns."""
    prices = prices.astype(float)
    return np.log(prices / prices.shift(1)).dropna(how="all")


def open_to_open_log_returns(open_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute open-to-open log returns.

    Row at index T contains log(Open[T] / Open[T-1]).
    The alignment to the signal date is handled by the engine
    (it reads row i+1 to get the execution return for signal i).
    """
    open_prices = open_prices.astype(float)
    return np.log(open_prices / open_prices.shift(1)).dropna(how="all")


def resample_to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end frequency."""
    return prices.resample("ME").last()


def check_ticker_availability(
    prices: pd.DataFrame, requested_start: str
) -> dict[str, str]:
    """Identify tickers whose data starts after the requested start date.

    Returns a dict mapping ticker -> actual first available date (as string),
    only for tickers that start later than *requested_start*.

    A tolerance of 5 calendar days is applied so that market closures
    (weekends, holidays like January 1st) do not trigger false warnings.
    """
    req = pd.to_datetime(requested_start)
    tolerance = pd.Timedelta(days=5)
    late: dict[str, str] = {}
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid is not None and first_valid > req + tolerance:
            late[col] = str(first_valid.date())
    return late
