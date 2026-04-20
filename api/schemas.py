"""
Pydantic Schemas (API Models)
================================
Request/response schemas for all FastAPI endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field


# ── Options Chain ──

class OptionContractResponse(BaseModel):
    """Single option contract with computed features."""
    symbol: str
    strike_price: float
    expiry_date: str
    option_type: str
    underlying_value: float
    last_price: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0
    open_interest: int = 0

    # Greeks
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Key features
    moneyness: float = 0.0
    time_to_expiry: float = 0.0
    iv_rank: float = 0.0
    iv_skew: float = 0.0
    put_call_parity_dev: float = 0.0
    volume_oi_ratio: float = 0.0
    bid_ask_spread: float = 0.0
    pcr_oi: float = 0.0

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "symbol": "NIFTY",
            "strike_price": 22500.0,
            "expiry_date": "2026-04-24",
            "option_type": "CE",
            "underlying_value": 22480.0,
            "last_price": 185.5,
            "implied_volatility": 0.18,
            "delta": 0.52,
            "gamma": 0.004,
            "theta": -12.5,
            "vega": 28.3,
        }]
    })


class OptionsChainResponse(BaseModel):
    """Full options chain response."""
    symbol: str
    underlying_value: float
    timestamp: str
    total_contracts: int
    contracts: List[OptionContractResponse]


# ── Signals ──

class SignalResponse(BaseModel):
    """Single mispricing signal."""
    symbol: str
    strike_price: float
    expiry_date: str
    option_type: str
    underlying_value: float = 0.0
    signal_type: str = Field(
        ..., description="OVERPRICED, UNDERPRICED, or FAIR"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Ensemble probability"
    )
    direction: str = Field(
        ..., description="LONG, SHORT, or HOLD"
    )
    xgb_probability: float = 0.0
    lgb_probability: float = 0.0
    ensemble_probability: float = 0.0
    model_version: str = ""
    model_agreement: bool = True
    timestamp: str = ""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "symbol": "NIFTY",
            "strike_price": 22600.0,
            "expiry_date": "2026-04-24",
            "option_type": "CE",
            "signal_type": "UNDERPRICED",
            "confidence": 0.78,
            "direction": "LONG",
            "model_agreement": True,
        }]
    })


class SignalsListResponse(BaseModel):
    """List of current mispricing signals."""
    total_signals: int
    signals: List[SignalResponse]
    generated_at: str


# ── Predict ──

class PredictRequest(BaseModel):
    """On-demand prediction request."""
    symbol: str = Field(default="NIFTY", description="Underlying symbol")
    strike_price: float = Field(..., description="Strike price")
    expiry_date: str = Field(..., description="Expiry date (YYYY-MM-DD)")
    option_type: str = Field(
        default="CE", description="CE (Call) or PE (Put)"
    )
    spot_price: Optional[float] = Field(
        None, description="Current spot price (auto-fetched if not provided)"
    )
    market_price: Optional[float] = Field(
        None, description="Current market price of the option"
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "symbol": "NIFTY",
            "strike_price": 22600.0,
            "expiry_date": "2026-04-24",
            "option_type": "CE",
            "spot_price": 22480.0,
            "market_price": 145.0,
        }]
    })


class PredictResponse(BaseModel):
    """Prediction result for a single option."""
    is_mispriced: bool
    signal_type: str
    direction: str
    confidence: float
    xgb_probability: float = 0.0
    lgb_probability: float = 0.0
    ensemble_probability: float = 0.0
    model_version: str = ""
    model_agreement: bool = True
    computed_features: Dict[str, float] = {}


# ── Model Health ──

class ModelHealthResponse(BaseModel):
    """Model health and drift status."""
    psi_score: float = Field(
        ..., description="Population Stability Index (overall)"
    )
    drift_status: str = Field(
        ..., description="stable, warning, or critical"
    )
    model_version: str = ""
    last_retrain: str = ""
    last_checked: str = ""
    feature_psi_breakdown: Dict[str, float] = {}
    total_retrains: int = 0
    training_records: int = 0
    recommendation: str = ""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "psi_score": 0.08,
            "drift_status": "stable",
            "model_version": "20260420_1200",
            "recommendation": "No action needed.",
        }]
    })


# ── General ──

class HealthCheckResponse(BaseModel):
    """System-level health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    database_records: int = 0
    models_loaded: bool = False
    pipeline_active: bool = False
    uptime_seconds: float = 0.0


# ── Retrain / Fine-Tune ──

class RetrainRequest(BaseModel):
    """Request body for on-demand fine-tuning."""
    n_trials: int = Field(
        default=50, ge=5, le=500,
        description="Number of Optuna optimization trials",
    )
    cv_splits: int = Field(
        default=5, ge=2, le=10,
        description="TimeSeriesSplit folds",
    )
    calibration_method: str = Field(
        default="isotonic",
        description="Probability calibration: 'isotonic' or 'sigmoid' (Platt)",
    )
    seed_fresh_data: bool = Field(
        default=False,
        description="Seed fresh synthetic data before training",
    )
    seed_days: int = Field(
        default=30, ge=7, le=365,
        description="Days of data to seed (if seed_fresh_data=True)",
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "n_trials": 50,
            "cv_splits": 5,
            "calibration_method": "isotonic",
            "seed_fresh_data": False,
        }]
    })


class RetrainResponse(BaseModel):
    """Response from a fine-tuning run."""
    status: str = Field(..., description="success or error")
    message: str = ""
    model_version: str = ""
    xgb_test_auc: float = 0.0
    lgb_test_auc: float = 0.0
    xgb_brier_score: float = 0.0
    lgb_brier_score: float = 0.0
    xgb_best_params: Dict[str, float] = {}
    lgb_best_params: Dict[str, float] = {}
    calibration_method: str = ""
    total_trials: int = 0
    training_records: int = 0
    duration_seconds: float = 0.0
