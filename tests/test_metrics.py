import pandas as pd
import numpy as np
import pytest
from strategy_lab.metrics import (
    cumulative_return, 
    sharpe_ratio, 
    sortino_ratio, 
    max_drawdown, 
    calmar_ratio
)

def test_cumulative_return():
    rets = pd.Series([0.01, 0.02, -0.01])
    expected = np.exp(rets.sum()) - 1.0
    assert cumulative_return(rets) == pytest.approx(expected)
    assert cumulative_return(pd.Series([])) == 0.0

def test_sharpe_ratio():
    rets = pd.Series([0.01, 0.02, 0.015, 0.01])
    std = rets.std()
    mean = rets.mean()
    expected = np.sqrt(12) * mean / std
    assert sharpe_ratio(rets, periods_per_year=12) == pytest.approx(expected)
    assert sharpe_ratio(pd.Series([])) == 0.0

def test_sortino_ratio():
    rets = pd.Series([0.01, -0.01, 0.02, -0.02])
    downside = rets[rets < 0]
    expected = np.sqrt(12) * rets.mean() / downside.std()
    assert sortino_ratio(rets, periods_per_year=12) == pytest.approx(expected)
    
    # Case with no downside
    assert sortino_ratio(pd.Series([0.01, 0.02])) == 0.0

def test_max_drawdown():
    # Wealth: 1, 1.1, 1.0, 1.2
    rets = pd.Series([0.0, np.log(1.1/1.0), np.log(1.0/1.1), np.log(1.2/1.0)])
    # Wealth curve: 1.0, 1.1, 1.0, 1.2
    # Peaks: 1.0, 1.1, 1.1, 1.2
    # DDs: 0, 0, (1.0-1.1)/1.1 = -0.0909, 0
    # Expected min DD = -0.0909...
    assert max_drawdown(rets) == pytest.approx(-0.0909090909)

def test_calmar_ratio():
    # Constant returns 1% monthly
    rets = pd.Series([0.01] * 12)
    # MDD will be 0 here, so expect 0 as per implementation
    assert calmar_ratio(rets) == 0.0
    
    # Simple case with DD
    rets = pd.Series([0.05, -0.1, 0.05])
    mdd = abs(max_drawdown(rets))
    ann_ret = np.exp(rets.mean() * 12) - 1.0
    expected = ann_ret / mdd
    assert calmar_ratio(rets, periods_per_year=12) == pytest.approx(expected)
