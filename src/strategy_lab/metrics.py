"""metrics.py — Evaluate performance."""

import numpy as np
import pandas as pd


def cumulative_return(log_returns: pd.Series) -> float:
    """Total wealth growth from a series of log returns."""
    if len(log_returns) == 0:
        return 0.0
    return float(np.exp(log_returns.sum()) - 1.0)


def sharpe_ratio(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio from a series of log returns."""
    if len(log_returns) == 0 or log_returns.std() < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * log_returns.mean() / log_returns.std())


def sortino_ratio(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sortino ratio (downside deviation) from a series of log returns."""
    if len(log_returns) == 0:
        return 0.0
    downside_returns = log_returns[log_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * log_returns.mean() / downside_returns.std())


def max_drawdown(log_returns: pd.Series) -> float:
    """Maximum drawdown from a series of log returns."""
    wealth = np.exp(log_returns.cumsum())
    peaks = wealth.cummax()
    dd = (wealth - peaks) / peaks
    return float(dd.min())


def calmar_ratio(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Calmar ratio from a series of log returns."""
    mdd = abs(max_drawdown(log_returns))
    if mdd == 0:
        return 0.0
    # Annualised return
    ann_ret = np.exp(log_returns.mean() * periods_per_year) - 1.0
    return float(ann_ret / mdd)
