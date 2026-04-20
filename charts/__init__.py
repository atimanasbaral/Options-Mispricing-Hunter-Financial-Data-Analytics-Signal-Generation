"""
Charts & Visualizations for Options Mispricing Hunter
=======================================================
Organized by domain:
  - options_charts  : IV surface, smile, Greeks, PCR
  - model_charts    : SHAP, ROC, PR, confusion matrix, Optuna
  - monitoring_charts : PSI drift, distribution shift, confidence
  - backtest_charts : PnL, drawdown, signal heatmap, win rate
  - data_providers  : synthetic data generators for all charts
"""

from charts.options_charts import (
    iv_surface_3d,
    iv_smile_curve,
    greeks_heatmap,
    pcr_over_time,
)
from charts.model_charts import (
    feature_importance_shap,
    shap_beeswarm,
    confusion_matrix_heatmap,
    roc_auc_curve,
    precision_recall_curve,
    optuna_optimization_history,
    optuna_param_importance,
)
from charts.monitoring_charts import (
    psi_drift_chart,
    feature_distribution_shift,
    confidence_distribution,
)
from charts.backtest_charts import (
    cumulative_pnl_curve,
    drawdown_chart,
    signal_heatmap,
    win_rate_by_moneyness,
)
