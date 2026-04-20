"""
Synthetic data generators for all chart types.
================================================
Pulls from the database when data exists, falls back to realistic
synthetic data to ensure charts always render for demonstration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_iv_surface_data(db=None, symbol="NIFTY"):
    """Strike × Expiry × IV surface data."""
    if db:
        df = db.get_latest_features(symbol=symbol, limit=2000)
        if not df.empty and "implied_volatility" in df.columns:
            return df[["strike_price", "expiry_date", "implied_volatility", "option_type"]].dropna()

    # Synthetic surface
    strikes = np.arange(21000, 24200, 200)
    expiries = pd.date_range(datetime.now() + timedelta(days=7), periods=6, freq="7D")
    rows = []
    for exp in expiries:
        tte = (exp - pd.Timestamp.now()).days / 365.0
        for K in strikes:
            moneyness = K / 22500
            base_iv = 0.15 + 0.03 * (moneyness - 1) ** 2 + 0.02 * np.sqrt(max(tte, 0.01))
            iv = base_iv + np.random.normal(0, 0.005)
            rows.append({"strike_price": K, "expiry_date": exp.strftime("%Y-%m-%d"),
                         "implied_volatility": max(iv, 0.05), "option_type": "CE"})
    return pd.DataFrame(rows)


def get_greeks_data(db=None, symbol="NIFTY"):
    """Strike × Expiry × Greeks matrix data."""
    if db:
        df = db.get_latest_features(symbol=symbol, limit=1000)
        if not df.empty and "delta" in df.columns:
            return df[["strike_price", "expiry_date", "option_type", "delta", "gamma",
                        "theta", "vega", "implied_volatility"]].dropna()

    strikes = np.arange(21500, 23500, 100)
    expiries = pd.date_range(datetime.now() + timedelta(days=7), periods=4, freq="14D")
    spot = 22500
    rows = []
    for exp in expiries:
        tte = max((exp - pd.Timestamp.now()).days / 365.0, 0.01)
        for K in strikes:
            m = (K - spot) / spot
            d = np.clip(0.5 - m * 3, -0.99, 0.99)
            g = max(0.002 * np.exp(-m**2 * 50), 0.0001)
            rows.append({"strike_price": K, "expiry_date": exp.strftime("%Y-%m-%d"),
                         "option_type": "CE", "delta": d, "gamma": g,
                         "theta": -g * spot * 0.2 / (2 * np.sqrt(tte)),
                         "vega": g * spot * np.sqrt(tte) * 0.01,
                         "implied_volatility": 0.15 + 0.03 * m**2})
    return pd.DataFrame(rows)


def get_pcr_data(db=None, symbol="NIFTY"):
    """Put-call ratio over time."""
    if db:
        df = db.get_latest_features(symbol=symbol, limit=5000)
        if not df.empty and "pcr_oi" in df.columns:
            ts = df.groupby("timestamp")["pcr_oi"].mean().reset_index()
            if len(ts) > 5:
                return ts.rename(columns={"timestamp": "date", "pcr_oi": "pcr"})

    dates = pd.date_range(datetime.now() - timedelta(days=60), periods=60, freq="B")
    base = 0.95
    pcr_vals = [base]
    for _ in range(59):
        pcr_vals.append(pcr_vals[-1] + np.random.normal(0, 0.08))
    return pd.DataFrame({"date": dates, "pcr": np.clip(pcr_vals, 0.3, 2.2)})


def get_model_predictions(db=None):
    """Model predictions: y_true, y_pred, y_prob for both models."""
    np.random.seed(42)
    n = 2000
    y_true = np.concatenate([np.zeros(1800), np.ones(200)])
    xgb_prob = np.where(y_true == 1,
                        np.clip(np.random.beta(5, 2, n), 0, 1),
                        np.clip(np.random.beta(1, 5, n), 0, 1))
    lgb_prob = np.where(y_true == 1,
                        np.clip(np.random.beta(4.5, 2.5, n), 0, 1),
                        np.clip(np.random.beta(1.2, 5, n), 0, 1))
    return pd.DataFrame({
        "y_true": y_true.astype(int),
        "xgb_prob": xgb_prob,
        "lgb_prob": lgb_prob,
        "xgb_pred": (xgb_prob >= 0.5).astype(int),
        "lgb_pred": (lgb_prob >= 0.5).astype(int),
        "ensemble_prob": 0.55 * xgb_prob + 0.45 * lgb_prob,
    })


def get_feature_importance_data():
    """SHAP-style feature importance for both models."""
    from config import FEATURE_COLUMNS
    n = len(FEATURE_COLUMNS)
    np.random.seed(42)
    xgb_imp = np.sort(np.random.exponential(0.03, n))[::-1]
    lgb_imp = np.sort(np.random.exponential(0.028, n))[::-1]
    np.random.shuffle(lgb_imp)
    # top features have most importance
    idx = np.argsort(xgb_imp)[::-1]
    return pd.DataFrame({
        "feature": [FEATURE_COLUMNS[i] for i in range(n)],
        "xgb_importance": xgb_imp,
        "lgb_importance": lgb_imp,
    }).sort_values("xgb_importance", ascending=False)


def get_backtest_data():
    """Simulated backtesting results."""
    np.random.seed(42)
    dates = pd.date_range(datetime.now() - timedelta(days=180), periods=180, freq="B")[:180]
    n = len(dates)
    strat_rets = np.random.normal(0.002, 0.015, n)
    bench_rets = np.random.normal(0.0005, 0.012, n)
    strat_cum = (1 + pd.Series(strat_rets)).cumprod() * 100
    bench_cum = (1 + pd.Series(bench_rets)).cumprod() * 100
    strat_peak = strat_cum.cummax()
    drawdown = (strat_cum - strat_peak) / strat_peak

    return pd.DataFrame({
        "date": dates[:n],
        "strategy_pnl": strat_cum.values,
        "benchmark_pnl": bench_cum.values,
        "drawdown": drawdown.values,
        "daily_return": strat_rets,
    })


def get_signal_heatmap_data(db=None):
    """Signal strength across strike × date matrix."""
    if db:
        df = db.get_latest_signals(min_confidence=0.0, limit=1000)
        if not df.empty:
            return df

    strikes = np.arange(21500, 23500, 100)
    dates = pd.date_range(datetime.now() - timedelta(days=20), periods=20, freq="B")
    rows = []
    for d in dates:
        for K in strikes:
            if np.random.random() > 0.7:
                signal = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
                rows.append({"date": d.strftime("%Y-%m-%d"), "strike_price": K,
                             "signal_strength": signal})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        {"date": [], "strike_price": [], "signal_strength": []}
    )


def get_psi_history():
    """PSI scores over time."""
    dates = pd.date_range(datetime.now() - timedelta(days=30), periods=30, freq="D")
    psi_vals = np.abs(np.random.normal(0.05, 0.04, 30))
    psi_vals[15] = 0.22  # inject a spike
    psi_vals[16] = 0.18
    return pd.DataFrame({"date": dates, "psi_score": psi_vals})


def get_optuna_trials():
    """Simulated Optuna trial results."""
    np.random.seed(42)
    n_trials = 50
    scores = []
    best = 0.9
    for i in range(n_trials):
        s = np.random.uniform(0.85, 0.99)
        if s > best:
            best = s
        scores.append(s)
    return pd.DataFrame({
        "trial": range(n_trials),
        "score": scores,
        "best_so_far": pd.Series(scores).cummax(),
    })


def get_optuna_param_importance():
    """Simulated hyperparameter importance from Optuna."""
    params = ["learning_rate", "n_estimators", "max_depth", "subsample",
              "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha", "gamma"]
    np.random.seed(42)
    importances = np.sort(np.random.exponential(0.15, len(params)))[::-1]
    return pd.DataFrame({"param": params, "importance": importances})


def get_win_rate_data():
    """Win rate breakdown by moneyness category."""
    return pd.DataFrame({
        "moneyness": ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"],
        "win_rate": [0.72, 0.68, 0.61, 0.55, 0.48],
        "total_trades": [45, 120, 280, 190, 65],
    })
