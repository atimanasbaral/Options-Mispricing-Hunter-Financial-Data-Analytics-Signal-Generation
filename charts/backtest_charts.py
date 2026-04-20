"""
Backtesting Charts — PnL, Drawdown, Signal Heatmap, Win Rate
================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts_output"
CHARTS_DIR.mkdir(exist_ok=True)


def _save(fig, name):
    fig.write_html(str(CHARTS_DIR / f"{name}.html"))
    try:
        fig.write_image(str(CHARTS_DIR / f"{name}.png"), width=1200, height=700, scale=2,
                        engine="kaleido")
    except Exception:
        pass


def _axis(title, **extra):
    base = dict(title=dict(text=title, font=dict(color="#e0e0e0")),
                gridcolor="#333", tickfont=dict(color="#aaa"))
    base.update(extra)
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Cumulative PnL Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cumulative_pnl_curve(df=None, save=True):
    """Strategy vs Nifty benchmark PnL, filled area under strategy."""
    from charts.data_providers import get_backtest_data
    if df is None:
        df = get_backtest_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["strategy_pnl"], mode="lines", name="Mispricing Strategy",
        line=dict(width=2.5, color="#4ecdc4"),
        fill="tozeroy", fillcolor="rgba(78,205,196,0.12)",
        hovertemplate="Date: %{x}<br>PnL: ₹%{y:,.1f}<extra>Strategy</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["benchmark_pnl"], mode="lines", name="Nifty Benchmark",
        line=dict(width=2, color="#ff6b6b", dash="dot"),
        hovertemplate="Date: %{x}<br>PnL: ₹%{y:,.1f}<extra>Benchmark</extra>",
    ))
    fig.add_hline(y=100, line=dict(color="#666", dash="dot", width=1))

    fig.update_layout(
        title=dict(text="Cumulative PnL — Strategy vs Benchmark",
                   font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Date"), yaxis=_axis("Portfolio Value (₹)"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0", size=12), bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), height=500,
    )
    if save:
        _save(fig, "cumulative_pnl")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Drawdown Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def drawdown_chart(df=None, save=True):
    """Filled area chart showing drawdown depth over time."""
    from charts.data_providers import get_backtest_data
    if df is None:
        df = get_backtest_data()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["drawdown"] * 100, mode="lines", name="Drawdown",
        line=dict(width=1.5, color="#ff4444"),
        fill="tozeroy", fillcolor="rgba(255,68,68,0.25)",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    max_dd_idx = df["drawdown"].idxmin()
    max_dd = df.loc[max_dd_idx, "drawdown"] * 100
    max_dd_date = df.loc[max_dd_idx, "date"]
    fig.add_annotation(x=max_dd_date, y=max_dd,
                       text=f"Max DD: {max_dd:.1f}%",
                       showarrow=True, arrowhead=2, arrowcolor="#ff6b6b",
                       font=dict(color="#ff6b6b", size=12),
                       bgcolor="rgba(0,0,0,0.7)", bordercolor="#ff6b6b")

    fig.update_layout(
        title=dict(text="Strategy Drawdown", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Date"), yaxis=_axis("Drawdown (%)"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), showlegend=False, height=400,
    )
    if save:
        _save(fig, "drawdown_chart")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Signal Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def signal_heatmap(df=None, save=True):
    """Mispricing signals across strike × date, color = signal strength."""
    from charts.data_providers import get_signal_heatmap_data
    if df is None:
        df = get_signal_heatmap_data()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No signal data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#aaa"))
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=400)
        if save:
            _save(fig, "signal_heatmap")
        return fig

    pivot = df.pivot_table(index="strike_price", columns="date",
                           values="signal_strength", aggfunc="mean").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title=dict(text="Signal Strength", font=dict(color="#e0e0e0")),
                      tickfont=dict(color="#e0e0e0")),
        hovertemplate="Date: %{x}<br>Strike: %{y:,.0f}<br>Signal: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Mispricing Signal Heatmap", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Date"), yaxis=_axis("Strike Price"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), height=500,
    )
    if save:
        _save(fig, "signal_heatmap")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Win Rate by Moneyness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def win_rate_by_moneyness(df=None, save=True):
    """Grouped bar chart: moneyness category vs win rate %."""
    from charts.data_providers import get_win_rate_data
    if df is None:
        df = get_win_rate_data()

    colors = ["#4ecdc4", "#64b5f6", "#ffd93d", "#ff6b6b", "#e040fb"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["moneyness"], y=df["win_rate"] * 100, name="Win Rate",
        marker=dict(color=colors[:len(df)], line=dict(width=1.5, color="#333")),
        text=[f"{wr:.0f}%<br>({n} trades)" for wr, n in
              zip(df["win_rate"] * 100, df["total_trades"])],
        textposition="outside", textfont=dict(color="#e0e0e0", size=11),
        hovertemplate="%{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line=dict(color="#ff4444", dash="dash", width=1.5),
                  annotation=dict(text="50% baseline", font=dict(color="#aaa")))

    fig.update_layout(
        title=dict(text="Win Rate by Moneyness Category", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Moneyness", tickfont=dict(color="#e0e0e0", size=12)),
        yaxis=_axis("Win Rate (%)", range=[0, 100]),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), showlegend=False, height=450,
    )
    if save:
        _save(fig, "win_rate_by_moneyness")
    return fig
