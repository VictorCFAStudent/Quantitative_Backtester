# Strategy Lab Premium
### Institutional-Grade Quantitative Backtesting Framework

A multi-strategy portfolio backtester with an interactive Streamlit dashboard. Download historical price data, run walk-forward backtests across five allocation strategies, compare performance metrics, and export professional PDF reports.

---

## Features

- **5 allocation strategies** compared side-by-side
- **Daily or monthly rebalancing** with open-to-open execution for daily mode
- **Rolling or expanding look-back windows**
- **Transaction cost modelling** (basis points per rebalance, applied to turnover)
- **Live ticker validation** before running
- **Interactive Plotly charts**: cumulative wealth, drawdown, portfolio weights, return distributions
- **Target-return leverage framework**: rescales all strategies to the same terminal wealth to isolate path risk
- **PDF report export** with charts, KPIs, and disclaimer

---

## Strategies

| Strategy | Description |
|---|---|
| **Equal Weight** | 1/N allocation across all active assets. Estimation-free benchmark. |
| **Inverse Volatility** | Weights proportional to 1/σ. No covariance estimation required. |
| **Minimum Variance** | Solves min w′Σw (SLSQP) using a Ledoit-Wolf shrinkage covariance matrix. |
| **Maximum Sharpe** | Maximises (w′μ − r_f) / √(w′Σw) (SLSQP), long-only, fully invested. |
| **Momentum** | Weights proportional to cumulative return over the look-back window; zero weight for negative-momentum assets. |

---

## Project Structure

```
CIWP/project/
├── src/strategy_lab/
│   ├── app/
│   │   └── main.py          # Streamlit dashboard
│   ├── engine.py            # Walk-forward backtest engine + strategy functions
│   ├── data.py              # Price download and return computation (yfinance)
│   ├── metrics.py           # Sharpe, Sortino, Calmar, Max Drawdown
│   └── report_builder.py    # PDF report generator (ReportLab + Matplotlib)
├── tests/
│   ├── test_engine.py
│   └── test_metrics.py
├── pyproject.toml
└── uv.lock
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/VictorCFAStudent/Quantitative_Backtester.git
cd Quantitative_Backtester/CIWP/project

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Usage

```bash
# From the CIWP/project/ directory
streamlit run src/strategy_lab/app/main.py
```

Then in the sidebar:
1. Enter tickers (comma-separated, e.g. `SPY, AGG, GLD`)
2. Set start and end dates
3. Choose rebalancing frequency, window type, and window size
4. Select which strategies to run
5. Click **Fetch data & run selected strategies**

---

## Configuration Options

| Parameter | Options | Notes |
|---|---|---|
| Rebalance Frequency | Daily, Monthly | Daily uses open-to-open execution returns |
| Window Type | Rolling, Expanding | Expanding uses all history up to each rebalance date |
| Window Size | 1–52 weeks (daily) / 1–36 months (monthly) | Rolling only |
| Transaction Fee | 0–50 bps | Applied to one-way turnover at each rebalance |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive dashboard |
| `yfinance` | Historical price data |
| `pandas` / `numpy` | Data manipulation |
| `scipy` | SLSQP optimiser |
| `scikit-learn` | Ledoit-Wolf covariance shrinkage |
| `plotly` | Interactive charts |
| `matplotlib` | PDF chart rendering |
| `reportlab` | PDF generation |
