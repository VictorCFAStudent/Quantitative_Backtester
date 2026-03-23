"""report_builder.py — Premium PDF report generator."""

import io
import datetime
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image,
    PageBreak, HRFlowable, KeepTogether,
)

# ── Brand palette ────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#1f2c56")
ACCENT     = colors.HexColor("#636EFA")
LIGHT_GREY = colors.HexColor("#f8f9fa")
MID_GREY   = colors.HexColor("#6c757d")
DARK_TEXT   = colors.HexColor("#212529")
WHITE      = colors.white

# Matplotlib equivalent colours
MPL_NAVY   = "#1f2c56"
MPL_ACCENT = "#636EFA"
MPL_RED    = "#EF553B"
MPL_GREEN  = "#00CC96"
MPL_BG     = "#f8f9fa"
AREA_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]


def _mpl_style():
    """Configure matplotlib for premium-looking charts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": MPL_NAVY,
        "axes.labelsize": 9,
        "axes.labelcolor": MPL_NAVY,
        "axes.edgecolor": "#dee2e6",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e9ecef",
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "xtick.color": "#6c757d",
        "ytick.color": "#6c757d",
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def _plot_to_img(fig, width=480, height=220):
    """Render a matplotlib figure as a reportlab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height)


# ── Paragraph styles ────────────────────────────────────────────────────────
def _build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "ReportTitle", parent=ss["Title"],
        fontName="Helvetica-Bold", fontSize=26, leading=32,
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "ReportSubtitle", parent=ss["Normal"],
        fontName="Helvetica", fontSize=12, leading=16,
        textColor=colors.HexColor("#c5cae9"), alignment=TA_LEFT,
    ))
    ss.add(ParagraphStyle(
        "SectionTitle", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=16, spaceAfter=8,
    ))
    ss.add(ParagraphStyle(
        "BodyText2", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=DARK_TEXT,
    ))
    ss.add(ParagraphStyle(
        "MetricLabel", parent=ss["Normal"],
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=MID_GREY, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "MetricValue", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=NAVY, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "Footer", parent=ss["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=MID_GREY, alignment=TA_RIGHT,
    ))
    return ss


# ── Page callbacks ───────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    """Draw a thin accent bar at the top and footer text at the bottom."""
    w, h = A4
    # Top accent line
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(doc.leftMargin, h - 12 * mm, w - doc.rightMargin, h - 12 * mm)
    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GREY)
    canvas.drawRightString(
        w - doc.rightMargin, 10 * mm,
        f"Strategy Lab  •  Generated {datetime.date.today().strftime('%B %d, %Y')}  •  Page {doc.page}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════
def build_pdf_report(
    wealth: pd.Series,
    drawdown: pd.Series,
    metrics: dict,
    ticker_weights: pd.DataFrame,
    strategy_name: str = "Strategy",
    log_returns: pd.Series | None = None,
    simple_returns: pd.Series | None = None,
    target_annual: float = 10.0,
    periods_per_year: int = 252,
    rebalance_frequency: str = "Monthly",
    window_type: str = "Expanding",
    window_size: int | None = None,
) -> io.BytesIO:
    """Generate a professional PDF performance report."""

    _mpl_style()
    ss = _build_styles()
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    elements: list = []
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin

    # ═══════════════════════════ HEADER BANNER ═══════════════════════════════
    # We draw a dark navy banner as a Table with colored background
    window_label = (
        f"{window_type} window"
        if window_size is None
        else f"{window_type} window  •  {window_size} {'weeks' if rebalance_frequency == 'Daily' else 'months'}"
    )
    banner_data = [[
        Paragraph("Strategy Lab", ss["ReportTitle"]),
    ], [
        Paragraph("Quantitative Performance Report", ss["ReportSubtitle"]),
    ], [
        Paragraph(
            f'<font size="10" color="#c5cae9">'
            f'{strategy_name}  •  '
            f'{wealth.index[0].strftime("%b %Y")} – {wealth.index[-1].strftime("%b %Y")}'
            f'</font>',
            ss["ReportSubtitle"],
        ),
    ], [
        Paragraph(
            f'<font size="9" color="#9fa8da">'
            f'Rebalancing: {rebalance_frequency}  •  {window_label}'
            f'</font>',
            ss["ReportSubtitle"],
        ),
    ]]
    banner = Table(banner_data, colWidths=[usable_w])
    banner.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",  (0, 0), (-1, 0), 18),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 14))

    # ═══════════════════════ KEY METRICS (card row) ══════════════════════════
    elements.append(Paragraph("Key Performance Indicators", ss["SectionTitle"]))

    # Build a row of metric "cards" as a table
    labels = list(metrics.keys())
    values = list(metrics.values())
    n = len(labels)
    card_w = usable_w / n

    label_cells = [Paragraph(l, ss["MetricLabel"]) for l in labels]
    value_cells = [Paragraph(str(v), ss["MetricValue"]) for v in values]

    card_table = Table(
        [value_cells, label_cells],
        colWidths=[card_w] * n,
        rowHeights=[28, 16],
    )
    card_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
        ("TOPPADDING",    (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.5, colors.HexColor("#dee2e6")),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        # Vertical dividers between cards
        *[("LINEAFTER", (i, 0), (i, -1), 0.5, colors.HexColor("#dee2e6"))
          for i in range(n - 1)],
    ]))
    elements.append(card_table)
    elements.append(Spacer(1, 10))

    # ═══════════════════════ CUMULATIVE WEALTH ═══════════════════════════════
    elements.append(Paragraph("Cumulative Wealth", ss["SectionTitle"]))

    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(wealth.index, wealth.values, color=MPL_ACCENT, linewidth=1.8)
    ax1.fill_between(wealth.index, 1, wealth.values,
                     alpha=0.08, color=MPL_ACCENT)
    ax1.axhline(1, color="#adb5bd", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Wealth ($1 invested)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    elements.append(_plot_to_img(fig1, width=usable_w, height=210))
    elements.append(Spacer(1, 8))

    # ═══════════════════════ TARGET-RETURN CHART ═════════════════════════════
    if log_returns is not None:
        n_days = len(wealth)
        n_years = n_days / periods_per_year
        target_total = (1 + target_annual / 100.0) ** n_years

        actual_total = float(wealth.iloc[-1])
        fig_t, ax_t = plt.subplots(figsize=(9, 3.5))

        # Glide path
        glide = np.linspace(1.0, target_total, n_days)
        ax_t.plot(wealth.index, glide, color="#888888", linewidth=1.2,
                  linestyle="--", label="Linear Glide Path")

        # Horizontal target line
        ax_t.axhline(target_total, color="#aaaaaa", linewidth=1, linestyle=":")

        # Leveraged strategy curve
        if actual_total > 0 and actual_total != 1.0:
            leverage = np.log(target_total) / np.log(actual_total)
            levered_log = log_returns * leverage
            levered_wealth = np.exp(levered_log.cumsum())
            ax_t.plot(levered_wealth.index, levered_wealth.values,
                      color=MPL_ACCENT, linewidth=1.8,
                      label=f"{strategy_name} (\u03bb={leverage:.2f})")
            ax_t.fill_between(levered_wealth.index, 1, levered_wealth.values,
                              alpha=0.06, color=MPL_ACCENT)

        ax_t.set_ylabel("Leveraged Wealth ($1 invested)")
        ax_t.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
        ax_t.legend(loc="upper left", fontsize=8, frameon=False)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        fig_t.tight_layout()
        target_img = _plot_to_img(fig_t, width=usable_w, height=210)

        # Keep title + description + chart together on the same page
        elements.append(KeepTogether([
            Paragraph("Target-Return Comparison", ss["SectionTitle"]),
            Paragraph(
                f"Strategy returns scaled so that the final wealth matches a "
                f"{target_annual:.0f}% annual target. A smoother path indicates "
                f"lower risk for the same outcome.",
                ss["BodyText2"],
            ),
            Spacer(1, 4),
            target_img,
        ]))
        elements.append(Spacer(1, 8))

    # ═══════════════════════ DRAWDOWN ════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.fill_between(drawdown.index, 0, drawdown.values,
                     color=MPL_RED, alpha=0.25, linewidth=0)
    ax2.plot(drawdown.index, drawdown.values, color=MPL_RED, linewidth=1.2)
    ax2.axhline(0, color="#adb5bd", linewidth=0.6)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    dd_img = _plot_to_img(fig2, width=usable_w, height=180)

    # Keep title + chart together on the same page
    elements.append(KeepTogether([
        Paragraph("Drawdown", ss["SectionTitle"]),
        dd_img,
    ]))

    # ═══════════════════════ PAGE BREAK ══════════════════════════════════════
    elements.append(PageBreak())

    # ═══════════════════════ PORTFOLIO WEIGHTS ═══════════════════════════════
    elements.append(Paragraph("Portfolio Weights Over Time", ss["SectionTitle"]))

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    cols = ticker_weights.columns.tolist()
    bottoms = np.zeros(len(ticker_weights))
    for idx, col in enumerate(cols):
        vals = ticker_weights[col].values.astype(float)
        c = AREA_COLORS[idx % len(AREA_COLORS)]
        ax3.fill_between(ticker_weights.index, bottoms, bottoms + vals,
                         label=col, color=c, alpha=0.85, linewidth=0.3,
                         edgecolor="white")
        bottoms += vals
    ax3.set_ylabel("Weight")
    ax3.set_ylim(0, 1)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax3.legend(loc="upper center", ncol=min(len(cols), 6),
               bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    elements.append(_plot_to_img(fig3, width=usable_w, height=240))
    elements.append(Spacer(1, 16))

    # ═══════════════════════ RETURNS DISTRIBUTION ═════════════════════════════
    if simple_returns is not None:
        dist_label = "Daily" if periods_per_year >= 252 else "Monthly"
        elements.append(Paragraph(f"{dist_label} Returns Distribution", ss["SectionTitle"]))

        rets_clean = simple_returns.dropna()
        fig4, ax4 = plt.subplots(figsize=(9, 3.5))

        # Histogram
        n_bins = 50 if periods_per_year >= 252 else 25
        ax4.hist(rets_clean, bins=n_bins, density=True,
                 color=MPL_ACCENT, alpha=0.3, edgecolor="none",
                 label="Histogram")

        # KDE
        kde = gaussian_kde(rets_clean)
        x_pdf = np.linspace(rets_clean.min() - 0.02, rets_clean.max() + 0.02, 300)
        y_pdf = kde(x_pdf)
        ax4.plot(x_pdf, y_pdf, color=MPL_ACCENT, linewidth=2, label="KDE")
        ax4.fill_between(x_pdf, y_pdf, alpha=0.08, color=MPL_ACCENT)

        # Mean and zero lines
        mean_ret = float(rets_clean.mean())
        ax4.axvline(0, color="#adb5bd", linewidth=0.8, linestyle="--")
        ax4.axvline(mean_ret, color=MPL_RED, linewidth=1.2, linestyle="-",
                    label=f"Mean ({mean_ret:.4f})")

        ax4.set_xlabel(f"{dist_label} Return")
        ax4.set_ylabel("Density")
        ax4.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
        ax4.legend(loc="upper right", fontsize=7, frameon=False)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        fig4.tight_layout()
        elements.append(_plot_to_img(fig4, width=usable_w, height=210))
        elements.append(Spacer(1, 20))

    # ═══════════════════════ DISCLAIMER ══════════════════════════════════════
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#dee2e6"), spaceAfter=8,
    ))
    elements.append(Paragraph(
        "This report is generated by <b>Strategy Lab</b> for educational and "
        "informational purposes only. Past performance is not indicative of "
        "future results. This does not constitute investment advice.",
        ss["BodyText2"],
    ))

    # ═══════════════════════ BUILD ═══════════════════════════════════════════
    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buffer.seek(0)
    return buffer
