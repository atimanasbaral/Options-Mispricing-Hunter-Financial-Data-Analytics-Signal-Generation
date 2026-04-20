"""
Options Mispricing Hunter — Streamlit Dashboard
=================================================
7-row layout with 18 interactive charts.

Run:  streamlit run dashboard.py --server.port 8501
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Options Mispricing Hunter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Dark premium styling */
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(100,181,246,0.2);
        box-shadow: 0 4px 30px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #e0e0e0;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #90a4ae;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Section dividers */
    .section-header {
        background: linear-gradient(90deg, rgba(100,181,246,0.15) 0%, transparent 100%);
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #64b5f6;
        margin: 1.5rem 0 0.8rem 0;
        color: #e0e0e0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(26,26,46,0.6);
        border: 1px solid rgba(100,181,246,0.15);
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Plotly chart containers */
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0e1117 100%);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🎯 Options Mispricing Hunter</h1>
    <p>Real-time NSE options mispricing detection • XGBoost + LightGBM ensemble • Optuna-tuned • Calibrated probabilities</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")

    try:
        from data.database import OptionsDatabase
        db = OptionsDatabase()
        features_count = db.get_record_count("features")
        signals_count = db.get_record_count("signals")
    except Exception:
        features_count = 0
        signals_count = 0

    st.metric("Feature Records", f"{features_count:,}")
    st.metric("Signals Generated", f"{signals_count:,}")

    st.markdown("---")
    generate = st.button("🔄 Regenerate All Charts", use_container_width=True)


# ── Import chart functions ──
from charts.options_charts import iv_surface_3d, iv_smile_curve, greeks_heatmap, pcr_over_time
from charts.model_charts import (
    feature_importance_shap, shap_beeswarm, confusion_matrix_heatmap,
    roc_auc_curve, precision_recall_curve, optuna_optimization_history,
    optuna_param_importance,
)
from charts.monitoring_charts import psi_drift_chart, feature_distribution_shift, confidence_distribution
from charts.backtest_charts import cumulative_pnl_curve, drawdown_chart, signal_heatmap, win_rate_by_moneyness


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 1: IV Surface 3D (full width)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">🔥 Implied Volatility Surface</div>', unsafe_allow_html=True)
with st.spinner("Rendering IV Surface..."):
    fig_iv = iv_surface_3d(save=False)
    st.plotly_chart(fig_iv, use_container_width=True, key="iv_surface")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 2: IV Smile | Greeks Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">📈 Volatility Analysis</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig_smile = iv_smile_curve(save=False)
    st.plotly_chart(fig_smile, use_container_width=True, key="iv_smile")

with col2:
    greek_choice = st.selectbox("Greek", ["delta", "gamma"], key="greek_sel")
    fig_greeks = greeks_heatmap(greek=greek_choice, save=False)
    st.pyplot(fig_greeks, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 3: PCR Over Time | Signal Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">📊 Market Sentiment & Signals</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig_pcr = pcr_over_time(save=False)
    st.plotly_chart(fig_pcr, use_container_width=True, key="pcr")

with col2:
    fig_sig = signal_heatmap(save=False)
    st.plotly_chart(fig_sig, use_container_width=True, key="signal_hm")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 4: Cumulative PnL | Drawdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">💰 Backtesting Performance</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig_pnl = cumulative_pnl_curve(save=False)
    st.plotly_chart(fig_pnl, use_container_width=True, key="pnl")

with col2:
    fig_dd = drawdown_chart(save=False)
    st.plotly_chart(fig_dd, use_container_width=True, key="drawdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 5: SHAP Beeswarm | Feature Importance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">🤖 Model Explainability</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    with st.spinner("Computing SHAP values..."):
        fig_shap = shap_beeswarm(save=False)
        st.pyplot(fig_shap, use_container_width=True)

with col2:
    fig_imp = feature_importance_shap(save=False)
    st.plotly_chart(fig_imp, use_container_width=True, key="feat_imp")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 5.5: Optuna Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">🔬 Optuna Optimization</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig_opt = optuna_optimization_history(save=False)
    st.plotly_chart(fig_opt, use_container_width=True, key="optuna_hist")

with col2:
    fig_param = optuna_param_importance(save=False)
    st.plotly_chart(fig_param, use_container_width=True, key="optuna_param")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 6: PSI Drift | Confidence Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">📉 Drift Monitoring</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig_psi = psi_drift_chart(save=False)
    st.plotly_chart(fig_psi, use_container_width=True, key="psi_drift")

with col2:
    fig_conf = confidence_distribution(save=False)
    st.plotly_chart(fig_conf, use_container_width=True, key="confidence")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 6.5: Feature Distribution Shift (full width)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig_shift = feature_distribution_shift(save=False)
st.pyplot(fig_shift, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 7: ROC-AUC | Precision-Recall | Confusion Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    fig_roc = roc_auc_curve(save=False)
    st.plotly_chart(fig_roc, use_container_width=True, key="roc")

with col2:
    fig_pr = precision_recall_curve(save=False)
    st.plotly_chart(fig_pr, use_container_width=True, key="pr")

with col3:
    fig_cm = confusion_matrix_heatmap(save=False)
    st.pyplot(fig_cm, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 8: Win Rate by Moneyness (centered)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_, center, _ = st.columns([1, 2, 1])
with center:
    fig_wr = win_rate_by_moneyness(save=False)
    st.plotly_chart(fig_wr, use_container_width=True, key="win_rate")


# ── Save All Charts Button ──
if generate:
    with st.spinner("Saving all charts to /charts_output ..."):
        iv_surface_3d(save=True)
        iv_smile_curve(save=True)
        greeks_heatmap(greek="delta", save=True)
        greeks_heatmap(greek="gamma", save=True)
        pcr_over_time(save=True)
        feature_importance_shap(save=True)
        shap_beeswarm(save=True)
        confusion_matrix_heatmap(save=True)
        roc_auc_curve(save=True)
        precision_recall_curve(save=True)
        optuna_optimization_history(save=True)
        optuna_param_importance(save=True)
        psi_drift_chart(save=True)
        feature_distribution_shift(save=True)
        confidence_distribution(save=True)
        cumulative_pnl_curve(save=True)
        drawdown_chart(save=True)
        signal_heatmap(save=True)
        win_rate_by_moneyness(save=True)
    st.success("✅ All 18 charts saved as .html + .png in /charts_output")


# ── Footer ──
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.85rem;">'
    'Options Mispricing Hunter v1.0 • XGBoost + LightGBM Ensemble • '
    'Optuna-Tuned • Isotonic Calibration • PSI Drift Monitoring'
    '</p>',
    unsafe_allow_html=True,
)
