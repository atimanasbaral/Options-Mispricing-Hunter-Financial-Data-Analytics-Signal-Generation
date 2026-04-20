"""
Model & Signal Charts — SHAP, ROC, PR, Confusion Matrix, Optuna
==================================================================
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve as sk_pr_curve, average_precision_score

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
# 1. Feature Importance (SHAP-style bar chart)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def feature_importance_shap(df=None, top_n=15, save=True):
    """Grouped horizontal bar chart: XGBoost + LightGBM SHAP importances."""
    from charts.data_providers import get_feature_importance_data
    if df is None:
        df = get_feature_importance_data()

    df = df.head(top_n).sort_values("xgb_importance", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["xgb_importance"], orientation="h", name="XGBoost",
        marker=dict(color="#ff6b6b", line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra>XGBoost</extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["lgb_importance"], orientation="h", name="LightGBM",
        marker=dict(color="#4ecdc4", line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra>LightGBM</extra>",
    ))

    fig.update_layout(
        title=dict(text="Feature Importance (SHAP Values)", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Mean |SHAP Value|"),
        yaxis=dict(tickfont=dict(color="#e0e0e0", size=11)),
        barmode="group",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), height=550, margin=dict(l=160),
    )
    if save:
        _save(fig, "feature_importance_shap")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. SHAP Beeswarm Plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def shap_beeswarm(model=None, X_test=None, top_n=15, save=True):
    """SHAP beeswarm plot for XGBoost or LightGBM."""
    import joblib
    from config import MODEL_ARTIFACTS_DIR, FEATURE_COLUMNS

    if model is None:
        model_path = MODEL_ARTIFACTS_DIR / "xgboost_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.text(0.5, 0.5, "Model not trained yet\nRun training first",
                    ha="center", va="center", fontsize=16, color="#aaa", transform=ax.transAxes)
            ax.set_title("SHAP Beeswarm — Top 15 Features", fontsize=16, color="#e0e0e0")
            if save:
                _save(fig, "shap_beeswarm", is_plotly=False)
            return fig

    if X_test is None:
        from data.database import OptionsDatabase
        db = OptionsDatabase()
        df = db.get_latest_features(limit=500)
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X_test = df[available].fillna(0).head(200)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        shap.plots.beeswarm(shap_values, max_display=top_n, show=False)

        ax = plt.gca()
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaa")
        ax.set_xlabel("SHAP Value", color="#e0e0e0", fontsize=12)
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.title("SHAP Beeswarm — Top 15 Features", fontsize=16, color="#e0e0e0", pad=15)
        plt.tight_layout()
        if save:
            fig = plt.gcf()
            _save(fig, "shap_beeswarm", is_plotly=False)
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.text(0.5, 0.5, f"SHAP unavailable:\n{str(e)[:80]}",
                ha="center", va="center", fontsize=13, color="#aaa", transform=ax.transAxes)
        ax.set_title("SHAP Beeswarm — Top 15 Features", fontsize=16, color="#e0e0e0")
        if save:
            _save(fig, "shap_beeswarm", is_plotly=False)
        return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Confusion Matrix Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def confusion_matrix_heatmap(pred_df=None, save=True):
    """Seaborn heatmap with precision/recall annotations."""
    from charts.data_providers import get_model_predictions
    from sklearn.metrics import confusion_matrix as cm_func, precision_score, recall_score

    if pred_df is None:
        pred_df = get_model_predictions()

    y_true = pred_df["y_true"]
    y_pred = pred_df["xgb_pred"]
    cm = cm_func(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

    labels = np.array([[f"{cm[0,0]}\n(TN)", f"{cm[0,1]}\n(FP)"],
                       [f"{cm[1,0]}\n(FN)", f"{cm[1,1]}\n(TP)"]])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Fair (0)", "Mispriced (1)"],
                yticklabels=["Fair (0)", "Mispriced (1)"],
                linewidths=2, linecolor="#1a1a2e",
                annot_kws={"fontsize": 14, "fontweight": "bold"},
                cbar_kws={"label": "Count"})

    ax.set_title(f"Confusion Matrix\nPrecision: {prec:.3f} | Recall: {rec:.3f}",
                 fontsize=16, color="#e0e0e0", pad=15)
    ax.set_xlabel("Predicted", color="#e0e0e0", fontsize=13)
    ax.set_ylabel("Actual", color="#e0e0e0", fontsize=13)
    ax.tick_params(colors="#aaa", labelsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#e0e0e0")
    cbar.ax.tick_params(colors="#aaa")

    plt.tight_layout()
    if save:
        _save(fig, "confusion_matrix", is_plotly=False)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ROC-AUC Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def roc_auc_curve(pred_df=None, save=True):
    """ROC curve with AUC annotated in legend."""
    from charts.data_providers import get_model_predictions
    if pred_df is None:
        pred_df = get_model_predictions()

    fig = go.Figure()
    for name, prob_col, color in [
        ("XGBoost", "xgb_prob", "#ff6b6b"),
        ("LightGBM", "lgb_prob", "#4ecdc4"),
        ("Ensemble", "ensemble_prob", "#ffd93d"),
    ]:
        fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df[prob_col])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC = {roc_auc:.4f})",
            line=dict(width=2.5, color=color),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="#666", width=1)))

    fig.update_layout(
        title=dict(text="ROC Curve — Receiver Operating Characteristic",
                   font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("False Positive Rate"),
        yaxis=_axis("True Positive Rate"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0", size=12), bgcolor="rgba(0,0,0,0.6)",
                    x=0.55, y=0.05),
        font=dict(color="#e0e0e0"), height=550,
    )
    if save:
        _save(fig, "roc_auc_curve")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Precision-Recall Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def precision_recall_curve(pred_df=None, save=True):
    """PR curve with shaded area under curve."""
    from charts.data_providers import get_model_predictions
    if pred_df is None:
        pred_df = get_model_predictions()

    fig = go.Figure()
    for name, prob_col, color, fill_color in [
        ("XGBoost", "xgb_prob", "#ff6b6b", "rgba(255,107,107,0.15)"),
        ("LightGBM", "lgb_prob", "#4ecdc4", "rgba(78,205,196,0.10)"),
        ("Ensemble", "ensemble_prob", "#ffd93d", "rgba(255,217,61,0.10)"),
    ]:
        prec, rec, _ = sk_pr_curve(pred_df["y_true"], pred_df[prob_col])
        ap = average_precision_score(pred_df["y_true"], pred_df[prob_col])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name} (AP = {ap:.4f})",
            line=dict(width=2.5, color=color),
            fill="tozeroy" if name == "Ensemble" else None,
            fillcolor=fill_color if name == "Ensemble" else None,
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Precision-Recall Curve", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Recall"),
        yaxis=_axis("Precision", range=[0, 1.05]),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0", size=12), bgcolor="rgba(0,0,0,0.6)"),
        font=dict(color="#e0e0e0"), height=550,
    )
    if save:
        _save(fig, "precision_recall_curve")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Optuna Optimization History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optuna_optimization_history(df=None, save=True):
    """Trial score vs trial number, best trial highlighted in gold."""
    from charts.data_providers import get_optuna_trials
    if df is None:
        df = get_optuna_trials()

    best_idx = df["score"].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trial"], y=df["score"], mode="markers", name="Trials",
        marker=dict(size=8, color="#64b5f6", opacity=0.6, line=dict(width=1, color="#333")),
        hovertemplate="Trial %{x}<br>AUC: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["trial"], y=df["best_so_far"], mode="lines", name="Best Score",
        line=dict(width=2.5, color="#ffd93d"),
    ))
    fig.add_trace(go.Scatter(
        x=[df.loc[best_idx, "trial"]], y=[df.loc[best_idx, "score"]],
        mode="markers", name=f"Best (AUC={df.loc[best_idx, 'score']:.4f})",
        marker=dict(size=16, color="#ffd700", symbol="star", line=dict(width=2, color="#fff")),
    ))

    fig.update_layout(
        title=dict(text="Optuna Optimization History", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Trial #"), yaxis=_axis("CV AUC-ROC"),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0"), height=450,
    )
    if save:
        _save(fig, "optuna_optimization_history")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Optuna Hyperparameter Importance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optuna_param_importance(df=None, save=True):
    """Horizontal bar chart of which hyperparams matter most."""
    from charts.data_providers import get_optuna_param_importance
    if df is None:
        df = get_optuna_param_importance()

    df = df.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        y=df["param"], x=df["importance"], orientation="h",
        marker=dict(color=df["importance"], colorscale="Sunset", line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Optuna Hyperparameter Importance", font=dict(size=20, color="#e0e0e0")),
        xaxis=_axis("Importance Score"),
        yaxis=dict(tickfont=dict(color="#e0e0e0", size=12)),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), height=450, margin=dict(l=160),
    )
    if save:
        _save(fig, "optuna_param_importance")
    return fig
