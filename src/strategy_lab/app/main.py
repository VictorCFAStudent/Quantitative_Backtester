"""main.py — Interactive Streamlit dashboard with multi-strategy comparison."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed

from strategy_lab.data import download_prices, download_open_prices, prices_to_log_returns, open_to_open_log_returns, resample_to_month_end, check_ticker_availability, validate_ticker
from strategy_lab.engine import (
    walk_forward_backtest,
    equal_weight,
    inverse_volatility_weight,
    min_variance_weight,
    max_sharpe_weight,
    momentum_weight,
)
from strategy_lab.metrics import (
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    calmar_ratio,
    cumulative_return,
)
from strategy_lab.report_builder import build_pdf_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Strategy Lab", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at top right, #f8f9fa, #e9ecef); }
    .metric-card {
        background: rgba(255,255,255,0.8); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3); border-radius: 12px;
        padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center; transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-label { font-size: 0.9rem; color: #6c757d; font-weight: 600; margin-bottom: 5px; }
    .metric-value { font-size: 1.5rem; color: #1f2c56; font-weight: 700; }
    .section-header {
        font-size: 1.8rem; font-weight: 700; color: #1f2c56;
        margin-top: 2rem; margin-bottom: 1rem;
        border-left: 5px solid #636EFA; padding-left: 15px;
    }
    [data-testid="stSidebar"] { background-color: #f1f3f5; border-right: 1px solid #dee2e6; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
STRATEGIES = {
    "Equal Weight":        equal_weight,
    "Inverse Volatility":  inverse_volatility_weight,
    "Minimum Variance":    min_variance_weight,
    "Maximum Sharpe":      max_sharpe_weight,
    "Momentum":            momentum_weight,
}
STRAT_NAMES = list(STRATEGIES.keys())

STRAT_COLORS = {
    "Equal Weight":        "#636EFA",
    "Inverse Volatility":  "#EF553B",
    "Minimum Variance":    "#00CC96",
    "Maximum Sharpe":      "#AB63FA",
    "Momentum":            "#FFA15A",
}

STRAT_DESCRIPTIONS = {
    "Equal Weight": (
        "Assigns a fixed weight of 1/N to each asset, where N is the number of "
        "assets with available data at the rebalance date. No statistical estimation "
        "is performed — the allocation is purely mechanical. Weights are recomputed "
        "at each rebalance only if the asset universe changes (e.g. a ticker "
        "starts trading later). This strategy serves as a robust, estimation-free "
        "benchmark against which the optimised portfolios can be compared."
    ),
    "Inverse Volatility": (
        "At each rebalance, the program estimates each asset's volatility σᵢ as "
        "the standard deviation of its daily log returns over the look-back window "
        "(expanding or rolling). Weights are then set proportional to 1/σᵢ and "
        "normalised to sum to one. No covariance or expected-return estimation is "
        "required — only marginal volatilities. Assets with zero variance are "
        "capped at the maximum observed volatility to prevent division errors."
    ),
    "Minimum Variance": (
        "Estimates the full N×N covariance matrix Σ from daily log returns over "
        "the look-back window. The optimiser (SciPy SLSQP) then solves: "
        "min w′Σw subject to Σwᵢ = 1 and 0 ≤ wᵢ ≤ 1 (long-only, fully invested). "
        "Expected returns are **not** estimated, which avoids the largest source "
        "of estimation error in mean-variance analysis. If optimisation fails to "
        "converge, the strategy falls back to equal weight."
    ),
    "Maximum Sharpe": (
        "Estimates both the mean return vector μ and the covariance matrix Σ from "
        "daily log returns over the look-back window. The optimiser (SciPy SLSQP) "
        "then solves: max (w′μ − rꜰ) / √(w′Σw) subject to Σwᵢ = 1 and "
        "0 ≤ wᵢ ≤ 1, with rꜰ = 0. Because this strategy depends on both first- "
        "and second-moment estimates, it is the most sensitive to estimation noise "
        "and may concentrate in a small number of assets. Falls back to equal "
        "weight if the solver does not converge."
    ),
    "Momentum": (
        "At each rebalance, computes the cumulative simple return of each asset "
        "over the entire look-back window (expanding or rolling). Assets with "
        "positive momentum are weighted proportionally to their cumulative return: "
        "higher past winners receive larger allocations, implementing a classic "
        "cross-sectional momentum tilt. Assets with zero or negative momentum "
        "are excluded and receive zero weight. If all assets show negative "
        "momentum, the strategy falls back to equal weight as a safety mechanism. "
        "No covariance estimation is involved — only past return magnitudes."
    ),
}

PLOTLY_TEMPLATE = "plotly_white"

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="padding: 20px 0px;">'
    '<h1 style="color: #1f2c56; margin-bottom: 0px;">🚀 Strategy Lab Premium</h1>'
    '<p style="color: #6c757d; font-size: 1.1rem;">'
    'Institutional-Grade Quantitative Framework</p></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("1) Universe & Dates")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "SPY, AGG, GLD")

# ── Live ticker validation ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _check_ticker(symbol: str) -> bool:
    return validate_ticker(symbol)

_parsed_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if _parsed_tickers:
    _status_parts = []
    _all_valid = True
    for _tk in _parsed_tickers:
        if _check_ticker(_tk):
            _status_parts.append(f"✅ **{_tk}**")
        else:
            _status_parts.append(f"❌ **{_tk}**")
            _all_valid = False
    st.sidebar.markdown("&nbsp;&nbsp;".join(_status_parts), unsafe_allow_html=True)
    if not _all_valid:
        st.sidebar.error("Some tickers are invalid — please fix them before running.")

start_date = st.sidebar.date_input(
    "Start Date", pd.to_datetime("2000-01-01"),
    min_value=pd.to_datetime("1990-01-01"),
    max_value=pd.to_datetime("today") - pd.DateOffset(years=1),
)
end_date = st.sidebar.date_input(
    "End Date", pd.to_datetime("today"),
    min_value=pd.to_datetime("1990-01-01"),
    max_value=pd.to_datetime("today"),
)

st.sidebar.header("2) Backtest Parameters")
rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency", ["Monthly", "Daily"],
    help="Monthly: rebalance every calendar month (close-to-close returns). "
         "Daily: rebalance every trading day (open-to-open execution returns)."
)

# Daily rebalancing requires daily data; monthly rebalancing allows either frequency
if rebalance_frequency == "Daily":
    data_frequency = "Daily"
    st.sidebar.caption("Daily rebalancing requires daily data frequency.")
else:
    data_frequency = st.sidebar.selectbox("Data Frequency", ["Daily", "Monthly"])

rebalance_months = 1  # always rebalance every month

window_type = st.sidebar.selectbox("Window Type", ["Expanding", "Rolling"])
window_size = 12
min_window_size = 1
if window_type == "Rolling":
    if rebalance_frequency == "Daily":
        window_size = st.sidebar.slider(
            "Rolling Window Size (weeks)", min_value=1, max_value=52, value=12,
            help="Look-back window in weeks (× 5 trading days)."
        )
    else:
        window_size = st.sidebar.slider("Rolling Window Size (months)", min_value=1, max_value=36, value=12)
else:  # Expanding
    if rebalance_frequency == "Daily":
        min_window_size = st.sidebar.slider(
            "Expanding Window Start Size (weeks)", min_value=1, max_value=52, value=4,
            help="Minimum weeks of data required before computing strategy weights. "
                 "Equal weight is used during the warm-up period."
        )
    else:
        min_window_size = st.sidebar.slider(
            "Expanding Window Start Size (months)", min_value=1, max_value=36, value=6,
            help="Minimum months of data required before computing strategy weights. "
                 "Equal weight is used during the warm-up period."
        )
fee_bps = st.sidebar.slider("Transaction Fee (bps per rebalance)", min_value=0, max_value=50, value=10, step=1,
                             help="One-way cost in basis points applied to portfolio turnover at each rebalance.")

st.sidebar.header("3) Strategy Selection")
_selected_strats = {}
for _sname in STRAT_NAMES:
    _selected_strats[_sname] = st.sidebar.checkbox(_sname, value=True, key=f"chk_{_sname}")
active_strategies = {k: STRATEGIES[k] for k, v in _selected_strats.items() if v}
active_names = list(active_strategies.keys())

if not active_names:
    st.sidebar.error("Select at least one strategy.")

run = st.sidebar.button("Fetch data & run selected strategies", use_container_width=True)

# ── Run all strategies ───────────────────────────────────────────────────────
if run:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()

    with st.spinner("Downloading prices…"):
        prices = download_prices(tickers, str(start_date), str(end_date))
        # Download open prices when daily rebalancing is selected
        open_prices = None
        if rebalance_frequency == "Daily":
            open_prices = download_open_prices(tickers, str(start_date), str(end_date))

    if prices.empty:
        st.error("No data returned. Check tickers and dates.")
        st.stop()

    st.success(f"✅ Prices successfully downloaded for {len(prices.columns)} ticker(s) — {len(prices)} trading days loaded.")

    # Detect tickers whose data starts after the requested start date
    late_tickers = check_ticker_availability(prices, str(start_date))

    # Forward-fill within each column to handle occasional missing days,
    # but keep leading NaNs so the engine knows when each ticker starts.
    prices = prices.ffill()
    if open_prices is not None:
        open_prices = open_prices.ffill()

    # Resample to month-end if user selected Monthly frequency
    if data_frequency == "Monthly":
        prices = resample_to_month_end(prices)

    returns = prices_to_log_returns(prices)

    # Compute open-to-open execution returns for daily rebalancing
    open_rets = None
    if rebalance_frequency == "Daily" and open_prices is not None:
        open_rets = open_to_open_log_returns(open_prices)

    if not active_names:
        st.error("Please select at least one strategy.")
        st.stop()

    all_results: dict[str, pd.DataFrame] = {}
    progress = st.progress(0, text="Running backtests…")

    def _run(name, func):
        return name, walk_forward_backtest(
            returns,
            strategy_func=func,
            rebalance_months=int(rebalance_months),
            window_type=window_type,
            window_size=window_size,
            min_window_size=min_window_size,
            frequency=data_frequency,
            fee_bps=float(fee_bps),
            rebalance_frequency=rebalance_frequency,
            open_returns=open_rets,
        )

    n_strats = len(active_strategies)
    with ThreadPoolExecutor(max_workers=n_strats) as executor:
        futures = {executor.submit(_run, name, func): name for name, func in active_strategies.items()}
        completed = 0
        for future in as_completed(futures):
            name, result = future.result()
            all_results[name] = result
            completed += 1
            progress.progress(completed / n_strats, text=f"Completed {name}")
    progress.empty()

    # Persist across Streamlit reruns
    st.session_state["all_results"] = all_results
    st.session_state["returns"] = returns
    st.session_state["late_tickers"] = late_tickers
    st.session_state["data_frequency"] = data_frequency
    st.session_state["active_names"] = list(all_results.keys())

# ── Display results ──────────────────────────────────────────────────────────
if "all_results" not in st.session_state:
    st.info("Set parameters in the sidebar and click **Fetch data & run selected strategies**.")
    st.stop()

# Show late-ticker warnings (persisted from run)
if "late_tickers" in st.session_state:
    for ticker, avail_date in st.session_state["late_tickers"].items():
        st.warning(
            f"⚠️ **{ticker}** data not available until **{avail_date}**. "
            f"It will be excluded from the strategy before that date."
        )

all_results: dict[str, pd.DataFrame] = st.session_state["all_results"]
run_names: list[str] = st.session_state.get("active_names", list(all_results.keys()))

# Pre-compute per-strategy series
strat_simple: dict[str, pd.Series] = {}
strat_wealth: dict[str, pd.Series] = {}
strat_dd: dict[str, pd.Series] = {}
strat_log: dict[str, pd.Series] = {}

for name, res in all_results.items():
    sr = res["Strategy"]
    strat_simple[name] = sr
    w = (1 + sr).cumprod()
    strat_wealth[name] = w
    strat_dd[name] = (w - w.cummax()) / w.cummax()
    strat_log[name] = np.log(1 + sr)


freq = st.session_state.get("data_frequency", "Daily")
periods_yr = 252 if freq == "Daily" else 12


def compute_metrics(log_rets: pd.Series, dd_series: pd.Series | None = None) -> dict:
    mdd = float(dd_series.min()) if dd_series is not None else max_drawdown(log_rets)
    ann_ret = np.exp(log_rets.mean() * periods_yr) - 1.0
    calmar_val = float(ann_ret / abs(mdd)) if abs(mdd) > 0 else 0.0
    return {
        "Total Return":  f"{cumulative_return(log_rets) * 100:.2f}%",
        "Sharpe":        f"{sharpe_ratio(log_rets, periods_per_year=periods_yr):.2f}",
        "Sortino":       f"{sortino_ratio(log_rets, periods_per_year=periods_yr):.2f}",
        "Calmar":        f"{calmar_val:.2f}",
        "Max Drawdown":  f"{mdd * 100:.2f}%",
    }


# ═══════════════════════════════════ KPIs ═══════════════════════════════════
st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

metrics_df = pd.DataFrame({name: compute_metrics(strat_log[name], strat_dd[name]) for name in run_names})
st.dataframe(
    metrics_df.style.set_properties(**{"text-align": "center", "font-weight": "600"}),
    use_container_width=True,
)

with st.expander("ℹ️  How do these strategies work?", expanded=False):
    for name in run_names:
        color = STRAT_COLORS[name]
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{color};margin-right:6px;"></span>'
            f'**{name}** — {STRAT_DESCRIPTIONS[name]}',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════ Wealth Chart ═══════════════════════════════
st.markdown('<div class="section-header">📊 Cumulative Returns</div>', unsafe_allow_html=True)
sel_wealth = st.multiselect("Strategies to display", run_names, default=run_names, key="sel_wealth")

if sel_wealth:
    fig_w = go.Figure()
    for name in sel_wealth:
        fig_w.add_trace(go.Scatter(
            x=strat_wealth[name].index, y=strat_wealth[name].values,
            mode="lines", name=name,
            line=dict(color=STRAT_COLORS[name], width=2.5),
        ))
    fig_w.update_layout(
        template=PLOTLY_TEMPLATE, hovermode="x unified",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Wealth ($1 invested)",
    )
    st.plotly_chart(fig_w, use_container_width=True)

# ═══════════════════════ Target-Return Leverage Framework ═══════════════════
st.markdown('<div class="section-header">🎯 Target-Return Comparison</div>', unsafe_allow_html=True)
st.caption(
    "Each strategy's returns are scaled (leveraged) so that **all curves end at the same "
    "target value**. This isolates the *path risk* required to achieve a given return — "
    "smoother paths indicate lower risk for the same outcome."
)

target_annual = st.slider(
    "Target annual return (%)", min_value=1, max_value=50, value=10, step=1, key="target_ret"
)

# Compute target total return over the actual period
ref_idx = strat_wealth[run_names[0]].index
n_days = len(ref_idx)
n_years = n_days / periods_yr
target_total = (1 + target_annual / 100.0) ** n_years
total_pct = (target_total - 1) * 100

st.markdown(
    f"""
    <div style="display:flex; gap:20px; margin-bottom:16px;">
        <div class="metric-card" style="flex:1;">
            <div class="metric-label">Target Annual Return</div>
            <div class="metric-value">{target_annual}%</div>
        </div>
        <div class="metric-card" style="flex:1;">
            <div class="metric-label">Cumulative Target Over {n_years:.1f} Years</div>
            <div class="metric-value">{total_pct:.1f}%</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sel_target = st.multiselect("Strategies to display", run_names, default=run_names, key="sel_target")

if sel_target:
    fig_target = go.Figure()

    # ── Linear glide path (dashed) ──
    glide = np.linspace(1.0, target_total, n_days)
    fig_target.add_trace(go.Scatter(
        x=ref_idx, y=glide,
        mode="lines", name="Linear Glide Path",
        line=dict(color="#888888", width=2, dash="dash"),
    ))

    # ── Horizontal target line ──
    fig_target.add_hline(
        y=target_total,
        line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        annotation=dict(
            text=f"Target: ${target_total:.2f}  ({target_annual}% p.a. × {n_years:.1f}y)",
            font=dict(size=11, color="#666666"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#aaaaaa", borderwidth=1, borderpad=4,
        ),
    )

    # ── Leveraged strategy curves ──
    for name in sel_target:
        actual_total = float(strat_wealth[name].iloc[-1])
        if actual_total <= 0 or actual_total == 1.0:
            continue

        # Apply leverage in LOG-return space so terminal value is exact
        leverage = np.log(target_total) / np.log(actual_total)
        levered_log_rets = strat_log[name] * leverage
        levered_wealth = np.exp(levered_log_rets.cumsum())

        fig_target.add_trace(go.Scatter(
            x=levered_wealth.index, y=levered_wealth.values,
            mode="lines", name=f"{name} (λ={leverage:.2f})",
            line=dict(color=STRAT_COLORS[name], width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Wealth: $%{y:.3f}<extra>" + name + "</extra>",
        ))

    fig_target.update_layout(
        template=PLOTLY_TEMPLATE, hovermode="x unified",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Leveraged Wealth ($1 invested)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_target, use_container_width=True)

# ═══════════════════════════════ Drawdown Chart ════════════════════════════
st.markdown('<div class="section-header">📉 Drawdown</div>', unsafe_allow_html=True)
sel_dd = st.multiselect("Strategies to display", run_names, default=run_names, key="sel_dd")

if sel_dd:
    fig_dd = go.Figure()
    for name in sel_dd:
        fig_dd.add_trace(go.Scatter(
            x=strat_dd[name].index, y=strat_dd[name].values,
            mode="lines", name=name, fill="tozeroy",
            line=dict(color=STRAT_COLORS[name], width=1.5),
            fillcolor=STRAT_COLORS[name].replace(")", ", 0.15)").replace("rgb", "rgba")
                         if "rgb" in STRAT_COLORS[name]
                         else f"rgba({int(STRAT_COLORS[name][1:3],16)},{int(STRAT_COLORS[name][3:5],16)},{int(STRAT_COLORS[name][5:7],16)},0.15)",
        ))
    fig_dd.update_layout(
        template=PLOTLY_TEMPLATE, hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ═══════════════════════════════ Weights Chart ═════════════════════════════
st.markdown('<div class="section-header">⚖️ Portfolio Weights</div>', unsafe_allow_html=True)
sel_wt = st.selectbox("Strategy to inspect", run_names, key="sel_wt")

weights_df = all_results[sel_wt].drop(columns=["Strategy"])
fig_wts = go.Figure()
tickers_list = weights_df.columns.tolist()
area_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
for idx, col in enumerate(tickers_list):
    fig_wts.add_trace(go.Scatter(
        x=weights_df.index, y=weights_df[col],
        mode="lines", name=col, stackgroup="one",
        line=dict(width=0.5, color=area_colors[idx % len(area_colors)]),
    ))
fig_wts.update_layout(
    template=PLOTLY_TEMPLATE,
    margin=dict(l=40, r=40, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Weight", yaxis_tickformat=".0%",
)
st.plotly_chart(fig_wts, use_container_width=True)

# ═══════════════════════════ Distribution Chart ════════════════════════════
dist_label = "Daily" if freq == "Daily" else "Monthly"
st.markdown(f'<div class="section-header">📊 {dist_label} Returns Distribution</div>', unsafe_allow_html=True)
sel_dist = st.multiselect("Strategies to display", run_names, default=run_names, key="sel_dist")

if sel_dist:
    fig_dist = go.Figure()
    for name in sel_dist:
        if freq == "Daily":
            rets_dist = strat_simple[name]
        else:
            rets_dist = strat_simple[name]  # already monthly
        kde = gaussian_kde(rets_dist.dropna())
        x_pdf = np.linspace(rets_dist.min() - 0.04, rets_dist.max() + 0.04, 300)
        y_pdf = kde(x_pdf)
        color = STRAT_COLORS[name]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        fig_dist.add_trace(go.Histogram(
            x=rets_dist, histnorm="probability density", nbinsx=40 if freq == "Daily" else 20,
            name=f"{name} hist", showlegend=False,
            marker=dict(color=f"rgba({r},{g},{b},0.25)", line=dict(width=0)),
        ))
        fig_dist.add_trace(go.Scatter(
            x=x_pdf, y=y_pdf, mode="lines", name=name,
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color=color, width=2.5),
        ))

    fig_dist.update_layout(
        template=PLOTLY_TEMPLATE, barmode="overlay", bargap=0.03,
        margin=dict(l=50, r=30, t=30, b=50),
        font=dict(family="Inter, sans-serif", color="#1f2c56", size=12),
        xaxis=dict(title=f"{dist_label} Return", showgrid=False, zeroline=False, tickformat=".1%"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_color="#1f2c56", bordercolor="rgba(99,110,250,0.4)"),
        hovermode="x",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ═══════════════════════════════ PDF Export ═════════════════════════════════
st.markdown('<div class="section-header">📄 Export & Performance Report</div>', unsafe_allow_html=True)
sel_pdf = st.selectbox("Strategy for PDF report", run_names, key="sel_pdf")

col_pdf, _ = st.columns([1, 2])
with col_pdf:
    st.info("Generate a professional analytical report in PDF format for offline review.")
    wealth_pdf = strat_wealth[sel_pdf]
    dd_pdf = strat_dd[sel_pdf]
    metrics_pdf = compute_metrics(strat_log[sel_pdf], strat_dd[sel_pdf])
    weights_pdf = all_results[sel_pdf].drop(columns=["Strategy"])

    with st.spinner("Preparing PDF report…"):
        pdf_buffer = build_pdf_report(
            wealth=wealth_pdf, drawdown=dd_pdf,
            metrics=metrics_pdf, ticker_weights=weights_pdf,
            strategy_name=sel_pdf,
            log_returns=strat_log[sel_pdf],
            simple_returns=strat_simple[sel_pdf],
            target_annual=float(st.session_state.get("target_ret", 10)),
            periods_per_year=periods_yr,
            rebalance_frequency=rebalance_frequency,
            window_type=window_type,
            window_size=window_size if window_type == "Rolling" else None,
        )
    st.download_button(
        label="📥 Download Performance Report (PDF)",
        data=pdf_buffer,
        file_name=f"strategy_report_{sel_pdf.lower().replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

freq_label = "high-fidelity daily" if freq == "Daily" else "monthly"
st.success(f"Analysis complete — {len(run_names)} strateg{'y' if len(run_names) == 1 else 'ies'} processed with {freq_label} resolution.")
