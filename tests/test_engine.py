import pandas as pd
import numpy as np
import pytest
from strategy_lab.engine import (
    equal_weight, 
    inverse_volatility_weight, 
    walk_forward_backtest,
    min_variance_weight,
    max_sharpe_weight,
    momentum_weight,
)

def test_min_variance_weight():
    # A is volatile, B is stable
    data = {
        'A': [0.1, -0.1, 0.1, -0.1],
        'B': [0.01, -0.01, 0.01, -0.01]
    }
    df = pd.DataFrame(data)
    weights = min_variance_weight(df)
    
    # Weights for B (stable) should be much higher than A (volatile)
    assert weights['B'] > weights['A']
    assert weights.sum() == pytest.approx(1.0)
    assert all(weights >= 0)

def test_max_sharpe_weight():
    # A has much higher returns and similar vol
    data = {
        'A': [0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02],
        'B': [0.01, 0.005, 0.01, 0.005, 0.01, 0.005, 0.01, 0.005]
    }
    df = pd.DataFrame(data)
    weights = max_sharpe_weight(df)
    
    # Weights for A (higher ret, better sharpe) should be higher than B
    assert weights['A'] > weights['B']
    assert weights.sum() == pytest.approx(1.0)
    assert all(weights >= -1e-5)


def test_equal_weight():
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    weights = equal_weight(df)
    assert len(weights) == 3
    assert all(weights == 1/3)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_volatility_weight():
    # A is volatile, B is stable
    data = {
        'A': [0.1, -0.1, 0.1, -0.1],
        'B': [0.01, -0.01, 0.01, -0.01]
    }
    df = pd.DataFrame(data)
    weights = inverse_volatility_weight(df)
    
    assert weights['B'] > weights['A']
    assert weights.sum() == pytest.approx(1.0)
    
    # Handle empty
    assert inverse_volatility_weight(pd.DataFrame(columns=['A'])).equals(pd.Series([1.0], index=['A']))


def test_walk_forward_backtest():
    # Create data where volatility changes over time differentially
    data = {
        'A': [0.01, -0.01, 0.01, -0.01, 0.05, -0.05, 0.05, -0.05],
        'B': [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    }
    returns = pd.DataFrame(data)
    
    # 1. Expanding window
    results_exp = walk_forward_backtest(
        returns, 
        strategy_func=inverse_volatility_weight, 
        window_type="Expanding",
        frequency="Monthly",
    )
    
    # 2. Rolling window (size 2)
    results_roll = walk_forward_backtest(
        returns, 
        strategy_func=inverse_volatility_weight, 
        rebalance_months=1, 
        window_type="Rolling", 
        window_size=2,
        frequency="Monthly",
    )
    
    # Weights should be different at the last step
    last_weights_exp = results_exp.drop(columns=['Strategy']).iloc[-1]
    last_weights_roll = results_roll.drop(columns=['Strategy']).iloc[-1]
    
    # Use np.allclose to handle float precision but assert they are NOT equal
    assert not np.allclose(last_weights_exp.values, last_weights_roll.values, atol=1e-5)


def test_walk_forward_with_late_ticker():
    """Column B starts as NaN for the first 4 rows (late-starting ticker)."""
    data = {
        'A': [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
        'B': [np.nan, np.nan, np.nan, np.nan, 0.02, -0.02, 0.02, -0.02],
    }
    returns = pd.DataFrame(data)

    results = walk_forward_backtest(
        returns,
        strategy_func=equal_weight,
        window_type="Expanding",
    )

    assert not results.empty
    assert len(results) == len(returns)

    # Before B is available, its weight should be 0
    weights = results.drop(columns=["Strategy"])
    for i in range(4):
        assert weights.iloc[i]["B"] == 0.0
        assert weights.iloc[i]["A"] > 0.0

    # After B becomes available, it should have non-zero weight
    for i in range(4, len(returns)):
        assert weights.iloc[i]["B"] > 0.0


def test_momentum_weight():
    """Higher-momentum asset gets a larger weight."""
    data = {
        'A': [0.02, 0.03, 0.01],   # cum ≈ +6.2%
        'B': [0.01, 0.01, 0.005],  # cum ≈ +2.5%
    }
    df = pd.DataFrame(data)
    weights = momentum_weight(df)

    assert weights['A'] > weights['B']
    assert weights.sum() == pytest.approx(1.0)
    assert all(weights >= 0)


def test_momentum_weight_excludes_negative():
    """Assets with negative cumulative return get zero weight."""
    data = {
        'A': [0.02, 0.03, 0.01],
        'B': [-0.05, -0.04, -0.03],
    }
    df = pd.DataFrame(data)
    weights = momentum_weight(df)

    assert weights['B'] == 0.0
    assert weights['A'] == pytest.approx(1.0)


def test_momentum_weight_all_negative_fallback():
    """When all assets have negative momentum, fall back to equal weight."""
    data = {
        'A': [-0.02, -0.03, -0.01],
        'B': [-0.05, -0.04, -0.03],
    }
    df = pd.DataFrame(data)
    weights = momentum_weight(df)
    expected = equal_weight(df)

    assert weights['A'] == pytest.approx(expected['A'])
    assert weights['B'] == pytest.approx(expected['B'])


def test_daily_rebalancing_with_open_returns():
    """Daily rebalancing uses open-to-open returns for P&L."""
    # Close-to-close returns (used for weight computation / signal)
    close_returns = pd.DataFrame({
        'A': [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
        'B': [0.005, -0.005, 0.01, -0.01, 0.005, -0.005, 0.01, -0.01],
    })
    # Open-to-open returns (used for execution P&L) — different from close
    open_rets = pd.DataFrame({
        'A': [0.008, -0.012, 0.018, -0.022, 0.009, -0.011, 0.019, -0.021],
        'B': [0.004, -0.006, 0.009, -0.011, 0.004, -0.006, 0.009, -0.011],
    })

    result = walk_forward_backtest(
        close_returns,
        strategy_func=equal_weight,
        window_type="Expanding",
        frequency="Daily",
        rebalance_frequency="Daily",
        open_returns=open_rets,
    )

    assert not result.empty
    assert len(result) == len(close_returns)
    # Strategy returns should NOT be zero (actual execution happened)
    assert result["Strategy"].abs().sum() > 0


def test_daily_rebalancing_window_in_weeks():
    """When rebalance_frequency='Daily', window_size is in weeks (× 5 trading days).

    A rolling window of 1 week (5 rows) should produce different weights than
    a rolling window of 4 weeks (20 rows) because the look-back data differs.
    """
    np.random.seed(42)
    n = 40
    close_returns = pd.DataFrame({
        'A': np.random.normal(0.01, 0.02, n),
        'B': np.random.normal(-0.005, 0.01, n),
    })
    open_rets = pd.DataFrame({
        'A': np.random.normal(0.01, 0.02, n),
        'B': np.random.normal(-0.005, 0.01, n),
    })

    result_1w = walk_forward_backtest(
        close_returns,
        strategy_func=inverse_volatility_weight,
        window_type="Rolling",
        window_size=1,   # 1 week = 5 trading days
        frequency="Daily",
        rebalance_frequency="Daily",
        open_returns=open_rets,
    )

    result_4w = walk_forward_backtest(
        close_returns,
        strategy_func=inverse_volatility_weight,
        window_type="Rolling",
        window_size=4,   # 4 weeks = 20 trading days
        frequency="Daily",
        rebalance_frequency="Daily",
        open_returns=open_rets,
    )

    # Different window sizes must produce different weight histories
    weights_1w = result_1w.drop(columns=["Strategy"])
    weights_4w = result_4w.drop(columns=["Strategy"])
    assert not np.allclose(weights_1w.values, weights_4w.values, atol=1e-6)


def test_daily_rebalancing_differs_from_monthly():
    """Daily rebalancing should produce different results from monthly."""
    data = {
        'A': [0.01, -0.01, 0.01, -0.01, 0.05, -0.05, 0.05, -0.05],
        'B': [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
    }
    close_returns = pd.DataFrame(data)
    open_rets = pd.DataFrame({
        'A': [0.009, -0.011, 0.009, -0.011, 0.049, -0.051, 0.049, -0.051],
        'B': [0.009, -0.011, 0.009, -0.011, 0.009, -0.011, 0.009, -0.011],
    })

    result_monthly = walk_forward_backtest(
        close_returns,
        strategy_func=inverse_volatility_weight,
        window_type="Expanding",
        frequency="Monthly",
    )

    result_daily = walk_forward_backtest(
        close_returns,
        strategy_func=inverse_volatility_weight,
        window_type="Expanding",
        frequency="Daily",
        rebalance_frequency="Daily",
        open_returns=open_rets,
    )

    # Results should differ because execution returns are different
    assert not np.allclose(
        result_monthly["Strategy"].values,
        result_daily["Strategy"].values,
        atol=1e-6,
    )
