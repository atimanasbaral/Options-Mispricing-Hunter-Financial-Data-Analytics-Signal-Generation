"""
FastAPI Routes
================
Endpoints for options chain, signals, predictions, and model health.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    OptionsChainResponse,
    OptionContractResponse,
    SignalsListResponse,
    SignalResponse,
    PredictRequest,
    PredictResponse,
    ModelHealthResponse,
    HealthCheckResponse,
    RetrainRequest,
    RetrainResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency references (set by app.py at startup) ──
_db = None
_pipeline = None
_ensemble = None
_drift_monitor = None
_start_time = None


def init_dependencies(db, pipeline, ensemble, drift_monitor):
    """Called by app.py to inject dependencies."""
    global _db, _pipeline, _ensemble, _drift_monitor, _start_time
    _db = db
    _pipeline = pipeline
    _ensemble = ensemble
    _drift_monitor = drift_monitor
    _start_time = datetime.now()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /options-chain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get(
    "/options-chain",
    response_model=OptionsChainResponse,
    summary="Live options chain with computed features",
    tags=["Data"],
)
async def get_options_chain(
    symbol: str = Query(default="NIFTY", description="Symbol (NIFTY, BANKNIFTY)"),
    limit: int = Query(default=200, ge=1, le=1000, description="Max contracts"),
):
    """
    Fetch the latest options chain data with all computed features.
    Runs the data pipeline if no recent data exists.
    """
    try:
        # Try to get latest features from DB
        df = _db.get_latest_features(symbol=symbol, limit=limit)

        if df.empty:
            # Run pipeline to fetch fresh data
            logger.info(f"No cached data for {symbol}, running pipeline...")
            df = _pipeline.run_once(symbol=symbol)

            if df.empty:
                raise HTTPException(
                    status_code=503,
                    detail=f"Unable to fetch data for {symbol}",
                )

        # Build response
        contracts = []
        for _, row in df.iterrows():
            contracts.append(OptionContractResponse(
                symbol=row.get("symbol", symbol),
                strike_price=float(row.get("strike_price", 0)),
                expiry_date=str(row.get("expiry_date", "")),
                option_type=row.get("option_type", ""),
                underlying_value=float(row.get("underlying_value", 0)),
                last_price=float(row.get("last_price", 0)),
                bid_price=float(row.get("bid_price", 0)),
                ask_price=float(row.get("ask_price", 0)),
                volume=int(row.get("volume", 0)),
                open_interest=int(row.get("open_interest", 0)),
                implied_volatility=float(row.get("implied_volatility", 0)),
                delta=float(row.get("delta", 0)),
                gamma=float(row.get("gamma", 0)),
                theta=float(row.get("theta", 0)),
                vega=float(row.get("vega", 0)),
                rho=float(row.get("rho", 0)),
                moneyness=float(row.get("moneyness", 0)),
                time_to_expiry=float(row.get("time_to_expiry", 0)),
                iv_rank=float(row.get("iv_rank", 0)),
                iv_skew=float(row.get("iv_skew", 0)),
                put_call_parity_dev=float(row.get("put_call_parity_dev", 0)),
                volume_oi_ratio=float(row.get("volume_oi_ratio", 0)),
                bid_ask_spread=float(row.get("bid_ask_spread", 0)),
                pcr_oi=float(row.get("pcr_oi", 0)),
            ))

        underlying = float(df["underlying_value"].iloc[0]) if not df.empty else 0
        timestamp = str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else ""

        return OptionsChainResponse(
            symbol=symbol,
            underlying_value=underlying,
            timestamp=timestamp,
            total_contracts=len(contracts),
            contracts=contracts,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /options-chain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get(
    "/signals",
    response_model=SignalsListResponse,
    summary="Current mispricing signals",
    tags=["Signals"],
)
async def get_signals(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    min_confidence: float = Query(
        default=0.6, ge=0.0, le=1.0, description="Minimum confidence"
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Max signals"),
):
    """
    Get current mispricing signals detected by the ensemble model.
    Returns signals above the specified confidence threshold.
    """
    try:
        # Check if we have recent signals in DB
        df = _db.get_latest_signals(
            symbol=symbol, min_confidence=min_confidence, limit=limit
        )

        if df.empty and _ensemble and _ensemble.is_loaded:
            # Generate fresh signals
            logger.info("No cached signals, generating from latest features...")
            features_df = _db.get_latest_features(symbol=symbol, limit=500)

            if not features_df.empty:
                signal_dicts = _ensemble.generate_signals(features_df)
                if signal_dicts:
                    _db.insert_signals(signal_dicts)
                    df = _db.get_latest_signals(
                        symbol=symbol,
                        min_confidence=min_confidence,
                        limit=limit,
                    )

        signals = []
        for _, row in df.iterrows():
            signals.append(SignalResponse(
                symbol=row.get("symbol", ""),
                strike_price=float(row.get("strike_price", 0)),
                expiry_date=str(row.get("expiry_date", "")),
                option_type=row.get("option_type", ""),
                underlying_value=float(row.get("underlying_value", 0)),
                signal_type=row.get("signal_type", "FAIR"),
                confidence=float(row.get("confidence", 0)),
                direction=row.get("direction", "HOLD"),
                xgb_probability=float(row.get("xgb_prob", 0)),
                lgb_probability=float(row.get("lgb_prob", 0)),
                ensemble_probability=float(row.get("ensemble_prob", 0)),
                model_version=row.get("model_version", ""),
                timestamp=str(row.get("timestamp", "")),
            ))

        return SignalsListResponse(
            total_signals=len(signals),
            signals=signals,
            generated_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in /signals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="On-demand prediction for a strike/expiry",
    tags=["Predict"],
)
async def predict(request: PredictRequest):
    """
    Run an on-demand mispricing prediction for a specific option contract.
    Computes features from the provided parameters and returns the model's assessment.
    """
    if not _ensemble or not _ensemble.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Train models first via the CLI.",
        )

    try:
        from data.feature_engine import FeatureEngine
        from data.nse_fetcher import generate_synthetic_data
        import pandas as pd
        from datetime import datetime as dt

        # Build a mini DataFrame for this contract
        expiry = pd.Timestamp(request.expiry_date)
        spot = request.spot_price or 22500.0  # Default if not provided
        market_price = request.market_price or 0.0

        mini_df = pd.DataFrame([{
            "symbol": request.symbol,
            "strike_price": request.strike_price,
            "expiry_date": expiry,
            "option_type": request.option_type,
            "underlying_value": spot,
            "last_price": market_price,
            "bid_price": market_price * 0.995,
            "ask_price": market_price * 1.005,
            "bid_qty": 100,
            "ask_qty": 100,
            "volume": 1000,
            "open_interest": 5000,
            "oi_change": 0,
            "iv": 0,
            "change": 0,
            "pct_change": 0,
            "timestamp": dt.now().isoformat(),
        }])

        # Also create a matching opposite-type contract for PCP
        opp_type = "PE" if request.option_type == "CE" else "CE"
        mini_df = pd.concat([
            mini_df,
            pd.DataFrame([{
                **mini_df.iloc[0].to_dict(),
                "option_type": opp_type,
                "last_price": market_price * 0.8,  # Rough estimate
            }]),
        ], ignore_index=True)

        # Compute features
        engine = FeatureEngine()
        features_df = engine.compute_features(mini_df)

        # Get prediction for the requested option type
        target_mask = features_df["option_type"] == request.option_type
        target_df = features_df[target_mask]

        if target_df.empty:
            raise HTTPException(status_code=400, detail="Could not compute features")

        result = _ensemble.predict_single(target_df.iloc[0].to_dict())

        # Add computed features to response
        from config import FEATURE_COLUMNS
        computed = {}
        for col in FEATURE_COLUMNS:
            if col in target_df.columns:
                val = target_df.iloc[0][col]
                computed[col] = round(float(val), 6) if pd.notna(val) else 0.0

        return PredictResponse(
            is_mispriced=result["is_mispriced"],
            signal_type=result["signal_type"],
            direction=result["direction"],
            confidence=result["confidence"],
            xgb_probability=result["xgb_probability"],
            lgb_probability=result["lgb_probability"],
            ensemble_probability=result["ensemble_probability"],
            model_version=result["model_version"],
            model_agreement=result["model_agreement"],
            computed_features=computed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /retrain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="On-demand fine-tuning with Optuna",
    tags=["Training"],
)
async def retrain(request: RetrainRequest):
    """
    Trigger on-demand fine-tuning of the ensemble models.

    Runs Optuna hyperparameter optimization with TimeSeriesSplit
    cross-validation, early stopping, and probability calibration.
    Every trial is logged to MLflow.
    """
    import time as _time

    start = _time.time()

    try:
        from models.fine_tuner import FineTuner
        from models.dataset import DatasetBuilder
        from data.pipeline import DataPipeline

        # Optionally seed fresh data
        if request.seed_fresh_data:
            logger.info(f"Seeding {request.seed_days} days of fresh data...")
            pipeline = DataPipeline(db=_db, use_synthetic=True)
            pipeline.seed_historical_data(
                days=request.seed_days, samples_per_day=3
            )

        # Build dataset
        builder = DatasetBuilder(_db)
        data = builder.build_dataset()

        # Run fine-tuning
        tuner = FineTuner(
            n_trials=request.n_trials,
            cv_splits=request.cv_splits,
            calibration_method=request.calibration_method,
        )
        result = tuner.fine_tune(data)

        # Hot-reload models into the running ensemble
        if _ensemble:
            _ensemble.xgb_model = result["xgb_model"]
            _ensemble.lgb_model = result["lgb_model"]
            _ensemble.xgb_calibrator = result["xgb_calibrator"]
            _ensemble.lgb_calibrator = result["lgb_calibrator"]
            _ensemble._calibrated = True
            _ensemble.model_version = datetime.now().strftime("%Y%m%d_%H%M")
            logger.info("Live ensemble hot-reloaded with tuned models")

        elapsed = _time.time() - start

        return RetrainResponse(
            status="success",
            message=(
                f"Fine-tuning complete. {request.n_trials} trials, "
                f"{request.cv_splits}-fold TSCV, "
                f"{request.calibration_method} calibration."
            ),
            model_version=_ensemble.model_version if _ensemble else "",
            xgb_test_auc=result["xgb_metrics"]["auc_roc"],
            lgb_test_auc=result["lgb_metrics"]["auc_roc"],
            xgb_brier_score=result["xgb_metrics"].get("brier_score", 0),
            lgb_brier_score=result["lgb_metrics"].get("brier_score", 0),
            xgb_best_params={
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in result["xgb_best_params"].items()
            },
            lgb_best_params={
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in result["lgb_best_params"].items()
            },
            calibration_method=request.calibration_method,
            total_trials=request.n_trials * 2,  # XGB + LGB
            training_records=_db.get_record_count("features") if _db else 0,
            duration_seconds=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = _time.time() - start
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return RetrainResponse(
            status="error",
            message=str(e),
            duration_seconds=round(elapsed, 2),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /model-health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get(
    "/model-health",
    response_model=ModelHealthResponse,
    summary="PSI score and drift status",
    tags=["Monitoring"],
)
async def get_model_health():
    """
    Get the current model health status including PSI drift score,
    per-feature breakdown, and retraining recommendation.
    """
    try:
        # Run a fresh drift check if monitor is available
        if _drift_monitor:
            health = _drift_monitor.check_drift()
        else:
            # Fall back to latest stored health
            stored = _db.get_latest_model_health()
            if stored:
                health = {
                    "psi_score": stored.get("psi_score", 0),
                    "drift_status": stored.get("drift_status", "unknown"),
                    "feature_breakdown": eval(stored.get("feature_psi_breakdown", "{}")),
                    "checked_at": stored.get("checked_at", ""),
                }
            else:
                health = {
                    "psi_score": 0.0,
                    "drift_status": "no_data",
                    "feature_breakdown": {},
                    "checked_at": "",
                }

        # Generate recommendation
        psi = health.get("psi_score", 0)
        if psi >= 0.2:
            recommendation = "CRITICAL: Significant drift detected. Model retraining triggered."
        elif psi >= 0.1:
            recommendation = "WARNING: Moderate drift detected. Monitor closely."
        else:
            recommendation = "STABLE: No significant drift. No action needed."

        return ModelHealthResponse(
            psi_score=health.get("psi_score", 0),
            drift_status=health.get("drift_status", "unknown"),
            model_version=_ensemble.model_version if _ensemble else "none",
            last_retrain=health.get("last_retrain", ""),
            last_checked=health.get("checked_at", datetime.now().isoformat()),
            feature_psi_breakdown=health.get("feature_breakdown", {}),
            total_retrains=_drift_monitor._retrain_count if _drift_monitor else 0,
            training_records=_db.get_record_count("features") if _db else 0,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Error in /model-health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System health check",
    tags=["System"],
)
async def health_check():
    """Basic system health check."""
    uptime = (datetime.now() - _start_time).total_seconds() if _start_time else 0

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        database_records=_db.get_record_count("features") if _db else 0,
        models_loaded=_ensemble.is_loaded if _ensemble else False,
        pipeline_active=_pipeline._running if _pipeline else False,
        uptime_seconds=round(uptime, 1),
    )
