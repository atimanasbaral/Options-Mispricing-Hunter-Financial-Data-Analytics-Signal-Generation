"""
Options Charts — IV Surface, IV Smile, Greeks Heatmap, PCR
=============================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts_output"
CHARTS_DIR.mkdir(exist_ok=True)


def _save(fig, name, is_plotly=True):
    """Save chart as HTML (Plotly) or PNG (Matplotlib)."""
    if is_plotly:
        fig.write_html(str(CHARTS_DIR / f"{name}.html"))
        try:
            fig.write_image(str(CHARTS_DIR / f"{name}.png"), width=1200, height=700, scale=2,
                            engine="kaleido")
        except Exception:
            pass  # kaleido may hang on Windows — HTML is the primary output
    else:
        fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=150, bbox_inches="tight",
                    facecolor="#0e1117")
        plt.close(fig)


def _axis(title, **extra):
    """Standard axis config for dark theme."""
    base = dict(title=dict(text=title, font=dict(color="#e0e0e0")),
                gridcolor="#333", tickfont=dict(color="#aaa"))
    base.update(extra)
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. IV Surface 3D Plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def iv_surface_3d(df=None, save=True):
    """3D surface: strike × expiry × implied volatility."""
    from charts.data_providers import get_iv_surface_data
    if df is None:
        df = get_iv_surface_data()

    df = df[df["option_type"] == "CE"].copy()
    pivot = df.pivot_table(index="strike_price", columns="expiry_date",
                           values="implied_volatility", aggfunc="mean").dropna()

    strikes = pivot.index.values
    expiries = list(pivot.columns)
    iv_matrix = pivot.values

    fig = go.Figure(data=[go.Surface(
        z=iv_matrix,
        x=list(range(len(expiries))),
        y=strikes,
        colorscale="RdYlGn_r",
        colorbar=dict(title=dict(text="IV", font=dict(color="#e0e0e0")),
                      tickfont=dict(color="#e0e0e0")),
        hovertemplate=(
            "Strike: %{y:,.0f}<br>"
            "Expiry: %{text}<br>"
            "IV: %{z:.2%}<extra></extra>"
        ),
        text=[[exp for exp in expiries] for _ in strikes],
    )])

    fig.update_layout(
        title=dict(text="Implied Volatility Surface", font=dict(size=22, color="#e0e0e0")),
        scene=dict(
            xaxis=dict(title=dict(text="Expiry Date", font=dict(color="#e0e0e0")),
                       tickvals=list(range(len(expiries))),
                       ticktext=[e[:10] for e in expiries],
                       backgroundcolor="#1a1a2e", gridcolor="#333",
                       tickfont=dict(color="#aaa")),
            yaxis=dict(title=dict(text="Strike Price", font=dict(color="#e0e0e0")),
                       backgroundcolor="#1a1a2e", gridcolor="#333",
                       tickfont=dict(color="#aaa")),
            zaxis=dict(title=dict(text="Implied Volatility", font=dict(color="#e0e0e0")),
                       backgroundcolor="#1a1a2e", gridcolor="#333",
                       tickfont=dict(color="#aaa")),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        margin=dict(l=0, r=0, t=60, b=0), height=650,
    )
    if save:
        _save(fig, "iv_surface_3d")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. IV Smile / Skew Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def iv_smile_curve(df=None, save=True):
    """Multi-line IV across strikes, one line per expiry."""
    from charts.data_providers import get_iv_surface_data
    if df is None:
        df = get_iv_surface_data()

    df = df[df["option_type"] == "CE"].copy()
    expiries = sorted(df["expiry_date"].unique())
    colors = px.colors.qualitative.Vivid

    fig = go.Figure()
    for i, exp in enumerate(expiries):
        sub = df[df["expiry_date"] == exp].sort_values("strike_price")
        fig.add_trace(go.Scatter(
            x=sub["strike_price"], y=sub["implied_volatility"],
            mode="lines+markers", name=exp[:10],
            line=dict(width=2.5, color=colors[i % len(colors)]),
            marker=dict(size=5),
            hovertemplate="Strike: %{x:,.0f}<br>IV: %{y:.2%}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="IV Smile / Skew by Expiry", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Strike Price"),
        yaxis=_axis("Implied Volatility", tickformat=".1%"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(title=dict(text="Expiry"), font=dict(color="#e0e0e0"),
                    bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), hovermode="x unified", height=500,
    )
    if save:
        _save(fig, "iv_smile_curve")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Greeks Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def greeks_heatmap(df=None, greek="delta", save=True):
    """Delta/Gamma heatmap across strike × expiry using seaborn."""
    from charts.data_providers import get_greeks_data
    if df is None:
        df = get_greeks_data()

    df = df[df["option_type"] == "CE"].copy()
    pivot = df.pivot_table(index="strike_price", columns="expiry_date",
                           values=greek, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    cmap = "RdYlGn" if greek == "delta" else "YlOrRd"
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap=cmap,
                linewidths=0.5, linecolor="#333",
                cbar_kws={"label": greek.capitalize()},
                annot_kws={"fontsize": 8, "color": "#1a1a2e"})

    ax.set_title(f"{greek.capitalize()} Heatmap: Strike × Expiry",
                 fontsize=16, color="#e0e0e0", pad=15)
    ax.set_xlabel("Expiry Date", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Strike Price", color="#e0e0e0", fontsize=12)
    ax.tick_params(colors="#aaa")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#e0e0e0")
    cbar.ax.tick_params(colors="#aaa")

    plt.tight_layout()
    if save:
        _save(fig, f"greeks_heatmap_{greek}", is_plotly=False)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Put-Call Ratio Over Time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pcr_over_time(df=None, save=True):
    """PCR line chart with overbought/oversold bands."""
    from charts.data_providers import get_pcr_data
    if df is None:
        df = get_pcr_data()

    fig = go.Figure()

    fig.add_hrect(y0=1.5, y1=max(df["pcr"].max(), 2.5), fillcolor="rgba(255,60,60,0.12)",
                  line_width=0, annotation_text="Overbought", annotation_position="top left",
                  annotation=dict(font=dict(color="#ff4444", size=11)))
    fig.add_hrect(y0=0, y1=0.7, fillcolor="rgba(0,200,80,0.12)",
                  line_width=0, annotation_text="Oversold", annotation_position="bottom left",
                  annotation=dict(font=dict(color="#00c853", size=11)))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pcr"], mode="lines", name="PCR (OI)",
        line=dict(width=2.5, color="#64b5f6"),
        fill="tozeroy", fillcolor="rgba(100,181,246,0.08)",
        hovertemplate="Date: %{x}<br>PCR: %{y:.2f}<extra></extra>",
    ))

    fig.add_hline(y=1.5, line=dict(color="#ff4444", dash="dash", width=1.5))
    fig.add_hline(y=0.7, line=dict(color="#00c853", dash="dash", width=1.5))
    fig.add_hline(y=1.0, line=dict(color="#aaa", dash="dot", width=1))

    fig.update_layout(
        title=dict(text="Put-Call Ratio Over Time", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Date"), yaxis=_axis("PCR (OI-based)"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#e0e0e0")),
        height=450,
    )
    if save:
        _save(fig, "pcr_over_time")
    return fig
