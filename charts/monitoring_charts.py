"""
Monitoring Charts — PSI Drift, Feature Distribution Shift, Confidence
========================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts_output"
CHARTS_DIR.mkdir(exist_ok=True)


def _save(fig, name, is_plotly=True):
    if is_plotly:
        fig.write_html(str(CHARTS_DIR / f"{name}.html"))
        try:
            fig.write_image(str(CHARTS_DIR / f"{name}.png"), width=1200, height=700, scale=2,
                            engine="kaleido")
        except Exception:
            pass
    else:
        fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=150, bbox_inches="tight",
                    facecolor="#0e1117")
        plt.close(fig)


def _axis(title, **extra):
    base = dict(title=dict(text=title, font=dict(color="#e0e0e0")),
                gridcolor="#333", tickfont=dict(color="#aaa"))
    base.update(extra)
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PSI Drift Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def psi_drift_chart(df=None, save=True):
    """PSI score over time with threshold line and danger zone."""
    from charts.data_providers import get_psi_history
    if df is None:
        df = get_psi_history()

    fig = go.Figure()

    fig.add_hrect(y0=0.2, y1=max(df["psi_score"].max() * 1.3, 0.35),
                  fillcolor="rgba(255,50,50,0.12)", line_width=0,
                  annotation_text="⚠ Critical Drift Zone", annotation_position="top left",
                  annotation=dict(font=dict(color="#ff4444", size=11)))
    fig.add_hrect(y0=0.1, y1=0.2, fillcolor="rgba(255,200,50,0.08)", line_width=0)

    colors = ["#ff4444" if v >= 0.2 else "#ffc107" if v >= 0.1 else "#4ecdc4"
              for v in df["psi_score"]]
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["psi_score"], mode="lines+markers", name="PSI Score",
        line=dict(width=2.5, color="#64b5f6"),
        marker=dict(size=7, color=colors, line=dict(width=1, color="#333")),
        hovertemplate="Date: %{x}<br>PSI: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0.2, line=dict(color="#ff4444", dash="dash", width=2),
                  annotation=dict(text="Critical (0.2)", font=dict(color="#ff4444")))
    fig.add_hline(y=0.1, line=dict(color="#ffc107", dash="dot", width=1.5),
                  annotation=dict(text="Warning (0.1)", font=dict(color="#ffc107")))

    fig.update_layout(
        title=dict(text="PSI Drift Monitoring", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Date"), yaxis=_axis("PSI Score"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), height=450,
    )
    if save:
        _save(fig, "psi_drift_chart")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Feature Distribution Shift (KDE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def feature_distribution_shift(train_df=None, live_df=None, save=True):
    """Training vs live data KDE overlaid, 2×3 grid for top 6 features."""
    top_features = ["implied_volatility", "delta", "gamma", "moneyness",
                    "volume_oi_ratio", "pcp_deviation_zscore"]
    np.random.seed(42)
    n = 1000

    if train_df is None:
        train_df = pd.DataFrame({
            "implied_volatility": np.random.lognormal(-1.8, 0.4, n),
            "delta": np.random.beta(2, 2, n) * 2 - 1,
            "gamma": np.abs(np.random.normal(0.003, 0.001, n)),
            "moneyness": np.random.normal(1.0, 0.05, n),
            "volume_oi_ratio": np.abs(np.random.normal(0.3, 0.15, n)),
            "pcp_deviation_zscore": np.random.normal(0, 1, n),
        })
    if live_df is None:
        live_df = pd.DataFrame({
            "implied_volatility": np.random.lognormal(-1.7, 0.45, n),
            "delta": np.random.beta(2.2, 1.8, n) * 2 - 1,
            "gamma": np.abs(np.random.normal(0.0035, 0.0012, n)),
            "moneyness": np.random.normal(1.01, 0.06, n),
            "volume_oi_ratio": np.abs(np.random.normal(0.35, 0.18, n)),
            "pcp_deviation_zscore": np.random.normal(0.15, 1.1, n),
        })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#0e1117")
    fig.suptitle("Feature Distribution Shift: Training vs Live",
                 fontsize=18, color="#e0e0e0", y=1.02)

    for idx, feat in enumerate(top_features):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#0e1117")
        if feat in train_df.columns and feat in live_df.columns:
            sns.kdeplot(train_df[feat], ax=ax, color="#4ecdc4", linewidth=2,
                        label="Training", fill=True, alpha=0.2)
            sns.kdeplot(live_df[feat], ax=ax, color="#ff6b6b", linewidth=2,
                        label="Live", fill=True, alpha=0.2)
        ax.set_title(feat.replace("_", " ").title(), color="#e0e0e0", fontsize=12, pad=8)
        ax.tick_params(colors="#aaa")
        ax.set_xlabel("", color="#aaa"); ax.set_ylabel("", color="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0")

    plt.tight_layout()
    if save:
        _save(fig, "feature_distribution_shift", is_plotly=False)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Model Confidence Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def confidence_distribution(pred_df=None, save=True):
    """Histogram of prediction probabilities, split by correct vs incorrect."""
    from charts.data_providers import get_model_predictions
    if pred_df is None:
        pred_df = get_model_predictions()

    correct = pred_df[pred_df["xgb_pred"] == pred_df["y_true"]]["ensemble_prob"]
    incorrect = pred_df[pred_df["xgb_pred"] != pred_df["y_true"]]["ensemble_prob"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=correct, name="Correct Predictions",
        marker=dict(color="#4ecdc4", line=dict(width=0.5, color="#333")),
        opacity=0.75, nbinsx=40,
        hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra>Correct</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=incorrect, name="Incorrect Predictions",
        marker=dict(color="#ff6b6b", line=dict(width=0.5, color="#333")),
        opacity=0.75, nbinsx=40,
        hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra>Incorrect</extra>",
    ))
    fig.add_vline(x=0.5, line=dict(color="#ffd93d", dash="dash", width=2),
                  annotation=dict(text="Threshold (0.5)", font=dict(color="#ffd93d")))

    fig.update_layout(
        title=dict(text="Model Confidence Distribution", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Predicted Probability"),
        yaxis=_axis("Count"),
        barmode="overlay",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), height=450,
    )
    if save:
        _save(fig, "confidence_distribution")
    return fig
