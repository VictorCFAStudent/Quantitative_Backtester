import inspect
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Return an equal-weight allocation across all assets in returns."""
    n_assets = len(returns.columns)
    if n_assets == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n_assets, index=returns.columns)

def inverse_volatility_weight(returns: pd.DataFrame) -> pd.Series:
    """Return weights proportional to 1/volatility."""
    if returns.empty or len(returns) < 2:
        return equal_weight(returns)
    vol = returns.std()
    vol = vol.replace(0, np.nan).fillna(vol[vol > 0].max() if any(vol > 0) else 1.0)
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return weights

def min_variance_weight(returns: pd.DataFrame, x0: np.ndarray | None = None) -> pd.Series:
    """Find weights that minimize portfolio variance using SLSQP."""
    if returns.empty or len(returns.columns) < 1:
        return pd.Series(dtype=float)
    if len(returns.columns) == 1:
        return pd.Series([1.0], index=returns.columns)
    if len(returns) < 2:                          # need ≥2 rows for a valid cov matrix
        return equal_weight(returns)
    n = len(returns.columns)
    cov = ledoit_wolf(returns.values)[0]
    def portfolio_variance(weights):
        return weights.T @ cov @ weights
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = x0 if (x0 is not None and len(x0) == n) else np.full(n, 1.0 / n)
    res = minimize(portfolio_variance, init_guess, method='SLSQP',
                   bounds=bounds, constraints=cons, options={'ftol': 1e-9})
    if not res.success:                           # silent failure guard
        return equal_weight(returns)
    return pd.Series(res.x, index=returns.columns)

def max_sharpe_weight(returns: pd.DataFrame, rf: float = 0.0, x0: np.ndarray | None = None) -> pd.Series:
    """Find weights that maximize the Sharpe ratio using SLSQP."""
    if returns.empty or len(returns.columns) < 1:
        return pd.Series(dtype=float)
    if len(returns.columns) == 1:
        return pd.Series([1.0], index=returns.columns)
    if len(returns) < 2:                          # need ≥2 rows for a valid cov matrix
        return equal_weight(returns)
    n = len(returns.columns)
    mu = returns.mean().values
    cov = ledoit_wolf(returns.values)[0]
    def neg_sharpe(weights):
        p_ret = weights.T @ mu
        p_vol = np.sqrt(weights.T @ cov @ weights)
        return -(p_ret - rf) / (p_vol + 1e-9)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = x0 if (x0 is not None and len(x0) == n) else np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, init_guess, method='SLSQP',
                   bounds=bounds, constraints=cons, options={'ftol': 1e-9})
    if not res.success:                           # silent failure guard
        return equal_weight(returns)
    return pd.Series(res.x, index=returns.columns)

def momentum_weight(returns: pd.DataFrame) -> pd.Series:
    """Weight assets proportionally to their recent cumulative return (momentum).

    Assets with non-positive momentum receive zero weight; the remaining
    budget is allocated proportionally to positive-momentum assets.
    Falls back to equal weight when all momentum signals are non-positive.
    """
    if returns.empty or len(returns) < 2:
        return equal_weight(returns)
    # Cumulative return over the full lookback window provided
    cum_ret = (np.exp(returns.sum()) - 1.0)  # log-returns → simple cumulative
    positive = cum_ret[cum_ret > 0]
    if positive.empty:
        return equal_weight(returns)
    weights = positive / positive.sum()
    # Pad zeros for excluded assets
    full_weights = pd.Series(0.0, index=returns.columns)
    full_weights[weights.index] = weights
    return full_weights

def walk_forward_backtest(
    returns: pd.DataFrame,
    strategy_func=equal_weight,
    rebalance_months: int = 1,
    window_type: str = "Expanding",
    window_size: int = 12,
    min_window_size: int = 1,
    frequency: str = "Daily",
    fee_bps: float = 0.0,
    rebalance_frequency: str = "Monthly",
    open_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generic walk-forward backtest. Supports daily or monthly data.

    Parameters
    ----------
    frequency : str
        "Daily" or "Monthly". Controls how window_size is interpreted
        when *rebalance_frequency* is "Monthly":
        Daily -> window_size months × 21 trading days; Monthly -> window_size months.
    min_window_size : int
        Minimum look-back periods required before the strategy function is
        called on an expanding window.  Uses the same unit as *window_size*
        (weeks for daily rebalancing, months otherwise).  Until this threshold
        is reached the engine falls back to equal weight.  Ignored for rolling
        windows (the rolling window size already sets the warm-up implicitly).
    fee_bps : float
        One-way transaction cost in basis points (e.g. 10 = 0.10 %).
        Applied proportionally to turnover at each rebalance.
    rebalance_frequency : str
        "Monthly" (default) or "Daily".
        - Monthly: rebalances every *rebalance_months* months (existing behaviour).
        - Daily: rebalances every trading day.  Weights are computed from
          close-to-close log *returns*, but portfolio P&L uses open-to-open
          log returns supplied via *open_returns*.
          *window_size* is interpreted in **weeks** (× 5 trading days).
    open_returns : pd.DataFrame | None
        Open-to-open log returns aligned with the signal date (row T holds
        log(Open[T+1]/Open[T])).  Required when *rebalance_frequency* is
        "Daily"; ignored otherwise.

    Handles tickers whose data starts later than the backtest start date:
    they are excluded (weight = 0) until their first non-NaN observation,
    then included at the next rebalance.
    """
    if returns.empty:
        return pd.DataFrame()

    daily_rebal = rebalance_frequency == "Daily"

    if daily_rebal and open_returns is None:
        raise ValueError(
            "open_returns must be provided when rebalance_frequency='Daily'"
        )

    all_cols = returns.columns.tolist()

    # Scale the rolling look-back window to match the data frequency
    is_daily = frequency == "Daily"
    if daily_rebal:
        # window_size / min_window_size are in weeks when daily rebalancing
        scaled_window = window_size * 5
        scaled_min_window = min_window_size * 5
    else:
        scaled_window = window_size * 21 if is_daily else window_size
        scaled_min_window = min_window_size * 21 if is_daily else min_window_size

    fee_rate = fee_bps / 10_000.0  # convert bps to decimal

    # Check if the index is datetime-like (real market data)
    # or integer/other (e.g., test data)
    is_datetime_index = isinstance(returns.index, pd.DatetimeIndex)

    # If daily rebalancing, align open_returns to the same index as returns
    if daily_rebal:
        common_idx = returns.index.intersection(open_returns.index)
        returns = returns.loc[common_idx]
        open_returns = open_returns.loc[common_idx]

    weights_hist = []
    strat_rets   = []

    # asset_alloc tracks the actual fractional value of each asset in the portfolio.
    # On rebalance: it snaps to the strategy's target weights (sum = 1).
    # Between rebalances: it drifts with daily returns and is re-normalised each day.
    asset_alloc = pd.Series(0.0, index=all_cols)

    prev_date    = None
    months_passed = 0
    prev_active   = set()      # track which columns were active previously

    # Warm-start support: pass previous solution as x0 to optimising strategies
    _accepts_x0 = 'x0' in inspect.signature(strategy_func).parameters
    _prev_x0: np.ndarray | None = None

    for i, (dt, row) in enumerate(returns.iterrows()):
        # --- Determine which tickers are active (have data) this period ---
        active_cols = [c for c in all_cols if pd.notna(row[c])]

        if not active_cols:
            # No data at all for this row — skip
            weights_hist.append(pd.Series(0.0, index=all_cols))
            strat_rets.append(0.0)
            prev_date = dt
            continue

        current_active = set(active_cols)
        new_tickers_appeared = current_active - prev_active

        # --- Rebalancing decision ---
        should_rebalance = False
        if i == 0:
            should_rebalance = True
        elif new_tickers_appeared:
            # Force rebalance when a new ticker becomes available
            should_rebalance = True
        elif daily_rebal:
            # Rebalance every single day
            should_rebalance = True
        elif is_datetime_index:
            if dt.month != prev_date.month:
                months_passed += 1
                if months_passed % rebalance_months == 0:
                    should_rebalance = True
        else:
            if i % rebalance_months == 0:
                should_rebalance = True

        if should_rebalance:
            if window_type == "Rolling":
                start_idx    = max(0, i - scaled_window)
                available_data = returns.iloc[start_idx:i][active_cols].dropna()
            else:
                available_data = returns.iloc[:i][active_cols].dropna()
                # Warm-up: require at least scaled_min_window rows before using strategy
                if len(available_data) < scaled_min_window:
                    available_data = pd.DataFrame(columns=active_cols)

            if not available_data.empty:
                if _accepts_x0:
                    new_weights = strategy_func(available_data, x0=_prev_x0)
                else:
                    new_weights = strategy_func(available_data)
                _prev_x0 = new_weights.reindex(active_cols).fillna(0.0).values
            else:
                new_weights = equal_weight(pd.DataFrame(columns=active_cols))
                _prev_x0 = None

            # Build full-width weight vector (0 for inactive tickers)
            asset_alloc = new_weights.reindex(all_cols).fillna(0.0)
            # Normalise in case of floating-point drift
            total = asset_alloc.sum()
            if total > 0:
                asset_alloc = asset_alloc / total

            # --- Apply transaction cost based on turnover ---
            if fee_rate > 0 and i > 0:
                prev_weights = weights_hist[-1] if weights_hist else pd.Series(0.0, index=all_cols)
                turnover = (asset_alloc - prev_weights).abs().sum() / 2.0
                fee_cost = turnover * fee_rate
                # Scale down allocations to reflect the fee paid out of portfolio value
                asset_alloc = asset_alloc * (1.0 - fee_cost)

        # Record start-of-period weights (before this day's return is applied)
        weights_hist.append(asset_alloc.copy())

        # --- Compute portfolio return for this period ---
        if daily_rebal:
            # Signal from data through D[i-1] → execute at open of D[i].
            # Return earned = Open[D[i+1]] / Open[D[i]] = open_returns.iloc[i+1].
            if i + 1 < len(open_returns):
                exec_row = open_returns.iloc[i + 1].reindex(all_cols).fillna(0.0)
                simple_row = np.expm1(exec_row)
            else:
                simple_row = pd.Series(0.0, index=all_cols)
        else:
            # Monthly rebalancing: use close-to-close returns (original behaviour)
            simple_row = np.expm1(row.reindex(all_cols).fillna(0.0))

        updated_alloc = asset_alloc * (1.0 + simple_row)
        port_value    = updated_alloc.sum()          # new total (was 1.0)

        strat_rets.append(float(port_value - 1.0))   # simple portfolio return

        # Normalise so allocations sum to 1 again for the next period
        if port_value > 0:
            asset_alloc = updated_alloc / port_value
        prev_date    = dt
        prev_active  = current_active

    out = pd.DataFrame(weights_hist, index=returns.index)
    out.insert(0, "Strategy", pd.Series(strat_rets, index=returns.index))
    return out
