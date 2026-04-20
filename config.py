"""
Options Mispricing Hunter — Central Configuration
===================================================
All project-wide constants, paths, and hyperparameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "options.db"
MODEL_ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()

# Ensure directories exist
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# NSE API configuration
# ──────────────────────────────────────────────
NSE_BASE_URL = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = f"{NSE_BASE_URL}/api/option-chain-indices"
NSE_EQUITY_OPTION_CHAIN_URL = f"{NSE_BASE_URL}/api/option-chain-equities"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
}

# Rate limiting (seconds between requests)
NSE_REQUEST_INTERVAL = 60
NSE_REQUEST_TIMEOUT = 30
NSE_MAX_RETRIES = 3

# Symbols to track
TRACKED_SYMBOLS = ["NIFTY", "BANKNIFTY"]

# ──────────────────────────────────────────────
# Black-Scholes parameters
# ──────────────────────────────────────────────
RISK_FREE_RATE = 0.065  # RBI repo rate ~6.5%
DIVIDEND_YIELD = 0.012  # Approximate Nifty dividend yield

# IV solver bounds
IV_LOWER_BOUND = 0.001
IV_UPPER_BOUND = 5.0
IV_MAX_ITERATIONS = 100
IV_TOLERANCE = 1e-8

# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "moneyness",
    "log_moneyness",
    "time_to_expiry",
    "sqrt_tte",
    "implied_volatility",
    "iv_rank",
    "iv_skew",
    "iv_term_structure",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "gamma_theta_ratio",
    "vega_theta_ratio",
    "put_call_parity_dev",
    "pcp_deviation_zscore",
    "volume",
    "open_interest",
    "volume_oi_ratio",
    "bid_ask_spread",
    "bid_ask_pct",
    "oi_change_pct",
    "volume_zscore",
    "pcr_oi",
    "pcr_volume",
    "intrinsic_value",
    "extrinsic_value",
    "extrinsic_pct",
]

# ──────────────────────────────────────────────
# ML model hyperparameters
# ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Ensemble weights (tuned on validation set)
ENSEMBLE_WEIGHTS = {"xgboost": 0.55, "lightgbm": 0.45}

# ──────────────────────────────────────────────
# Optuna fine-tuning
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS = 50           # Number of Bayesian optimization trials
OPTUNA_TIMEOUT = 600           # Max seconds for the entire study
OPTUNA_CV_SPLITS = 5           # TimeSeriesSplit folds
OPTUNA_EARLY_STOPPING = 50     # Early stopping rounds during tuning

# Search space bounds (shared by XGBoost and LightGBM)
OPTUNA_SEARCH_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 10),
    "learning_rate": (0.005, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}

# Probability calibration method: "isotonic" or "sigmoid" (Platt scaling)
CALIBRATION_METHOD = "isotonic"

# ──────────────────────────────────────────────
# PSI / drift monitoring
# ──────────────────────────────────────────────
PSI_THRESHOLD = 0.2       # Significant drift → retrain
PSI_WARNING_THRESHOLD = 0.1  # Moderate drift → alert
PSI_BUCKETS = 10
PSI_CHECK_INTERVAL_HOURS = 6
PSI_LOOKBACK_DAYS = 7     # Compare last N days vs training distribution

# ──────────────────────────────────────────────
# Mispricing labeling
# ──────────────────────────────────────────────
MISPRICING_ZSCORE_THRESHOLD = 2.0  # |z-score| > 2 → mispriced
MISPRICING_CONFIDENCE_THRESHOLD = 0.6  # Min ensemble probability for signal

# ──────────────────────────────────────────────
# API configuration
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Options Mispricing Hunter"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Real-time NSE options mispricing detection powered by "
    "XGBoost + LightGBM ensemble with PSI-based drift monitoring."
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
