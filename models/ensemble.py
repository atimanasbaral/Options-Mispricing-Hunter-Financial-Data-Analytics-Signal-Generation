"""
Ensemble Predictor
====================
Combines XGBoost and LightGBM predictions with configurable weights.
Produces trading signals: UNDERPRICED, OVERPRICED, or FAIR.
"""

import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from config import (
    ENSEMBLE_WEIGHTS,
    MISPRICING_CONFIDENCE_THRESHOLD,
    MODEL_ARTIFACTS_DIR,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Weighted ensemble of XGBoost + LightGBM for mispricing detection.

    Signal Logic:
    ─────────────
    1. Both models predict P(mispriced)
    2. Ensemble probability = weighted average
    3. If P ≥ threshold → check pcp_deviation_zscore for direction
       - Positive deviation → OVERPRICED
       - Negative deviation → UNDERPRICED
    4. Otherwise → FAIR
    """

    def __init__(
        self,
        xgb_model=None,
        lgb_model=None,
        scaler=None,
        weights: dict = None,
        xgb_calibrator=None,
        lgb_calibrator=None,
    ):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.scaler = scaler
        self.xgb_calibrator = xgb_calibrator
        self.lgb_calibrator = lgb_calibrator
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M")
        self._loaded = False
        self._calibrated = False

    def load_models(self) -> bool:
        """Load serialized models, calibrators, and scaler from disk."""
        xgb_path = MODEL_ARTIFACTS_DIR / "xgboost_model.joblib"
        lgb_path = MODEL_ARTIFACTS_DIR / "lightgbm_model.joblib"
        scaler_path = MODEL_ARTIFACTS_DIR / "feature_scaler.joblib"
        xgb_cal_path = MODEL_ARTIFACTS_DIR / "xgb_calibrator.joblib"
        lgb_cal_path = MODEL_ARTIFACTS_DIR / "lgb_calibrator.joblib"

        try:
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
                logger.info("XGBoost model loaded")
            else:
                logger.warning(f"XGBoost model not found at {xgb_path}")
                return False

            if lgb_path.exists():
                self.lgb_model = joblib.load(lgb_path)
                logger.info("LightGBM model loaded")
            else:
                logger.warning(f"LightGBM model not found at {lgb_path}")
                return False

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded")

            # Load calibrators if available (from fine-tuning)
            if xgb_cal_path.exists() and lgb_cal_path.exists():
                self.xgb_calibrator = joblib.load(xgb_cal_path)
                self.lgb_calibrator = joblib.load(lgb_cal_path)
                self._calibrated = True
                logger.info("Probability calibrators loaded (calibrated mode)")
            else:
                self._calibrated = False
                logger.info("No calibrators found (raw probability mode)")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if both models are available."""
        return self.xgb_model is not None and self.lgb_model is not None

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Make ensemble predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (unscaled — tree models handle raw features).

        Returns
        -------
        dict with keys:
            xgb_probs : np.ndarray — XGBoost P(mispriced)
            lgb_probs : np.ndarray — LightGBM P(mispriced)
            ensemble_probs : np.ndarray — weighted average
            predictions : np.ndarray — binary (0/1)
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Ensure correct feature columns
        available = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_clean = X[available].fillna(0)

        # Use calibrated probabilities if calibrators exist
        if self._calibrated and self.xgb_calibrator and self.lgb_calibrator:
            xgb_probs = self.xgb_calibrator.predict_proba(X_clean)[:, 1]
            lgb_probs = self.lgb_calibrator.predict_proba(X_clean)[:, 1]
        else:
            xgb_probs = self.xgb_model.predict_proba(X_clean)[:, 1]
            lgb_probs = self.lgb_model.predict_proba(X_clean)[:, 1]

        ensemble_probs = (
            self.weights["xgboost"] * xgb_probs
            + self.weights["lightgbm"] * lgb_probs
        )

        predictions = (ensemble_probs >= MISPRICING_CONFIDENCE_THRESHOLD).astype(int)

        return {
            "xgb_probs": xgb_probs,
            "lgb_probs": lgb_probs,
            "ensemble_probs": ensemble_probs,
            "predictions": predictions,
            "calibrated": self._calibrated,
        }

    def generate_signals(
        self,
        features_df: pd.DataFrame,
    ) -> List[dict]:
        """
        Generate mispricing signals from feature-enriched options data.

        Parameters
        ----------
        features_df : pd.DataFrame
            Options data with all computed features.

        Returns
        -------
        List[dict] — signal records ready for database insertion.
        """
        if not self.is_loaded:
            logger.warning("Models not loaded — cannot generate signals")
            return []

        if features_df.empty:
            return []

        # Run ensemble prediction
        result = self.predict(features_df)

        signals = []
        for i, (idx, row) in enumerate(features_df.iterrows()):
            ensemble_prob = float(result["ensemble_probs"][i])
            is_mispriced = result["predictions"][i]

            if not is_mispriced:
                continue  # Only emit signals for detected mispricings

            # Determine direction from PCP deviation
            pcp_z = row.get("pcp_deviation_zscore", 0)
            if pcp_z > 0:
                signal_type = "OVERPRICED"
                direction = "SHORT"
            elif pcp_z < 0:
                signal_type = "UNDERPRICED"
                direction = "LONG"
            else:
                signal_type = "MISPRICED"
                direction = "NEUTRAL"

            signals.append({
                "symbol": row.get("symbol", "UNKNOWN"),
                "strike_price": float(row.get("strike_price", 0)),
                "expiry_date": str(row.get("expiry_date", "")),
                "option_type": row.get("option_type", ""),
                "underlying_value": float(row.get("underlying_value", 0)),
                "signal_type": signal_type,
                "confidence": round(ensemble_prob, 4),
                "direction": direction,
                "xgb_prob": round(float(result["xgb_probs"][i]), 4),
                "lgb_prob": round(float(result["lgb_probs"][i]), 4),
                "ensemble_prob": round(ensemble_prob, 4),
                "model_version": self.model_version,
                "timestamp": row.get(
                    "timestamp", datetime.now().isoformat()
                ),
            })

        logger.info(
            f"Generated {len(signals)} mispricing signals "
            f"(out of {len(features_df)} contracts)"
        )
        return signals

    def predict_single(
        self,
        features: dict,
    ) -> dict:
        """
        Predict mispricing for a single option contract.

        Parameters
        ----------
        features : dict
            Feature values for one contract.

        Returns
        -------
        dict — prediction result with confidence and direction.
        """
        df = pd.DataFrame([features])
        result = self.predict(df)

        ensemble_prob = float(result["ensemble_probs"][0])
        is_mispriced = bool(result["predictions"][0])

        pcp_z = features.get("pcp_deviation_zscore", 0)
        if is_mispriced:
            signal_type = "OVERPRICED" if pcp_z > 0 else "UNDERPRICED"
            direction = "SHORT" if pcp_z > 0 else "LONG"
        else:
            signal_type = "FAIR"
            direction = "HOLD"

        return {
            "is_mispriced": is_mispriced,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": round(ensemble_prob, 4),
            "xgb_probability": round(float(result["xgb_probs"][0]), 4),
            "lgb_probability": round(float(result["lgb_probs"][0]), 4),
            "ensemble_probability": round(ensemble_prob, 4),
            "model_version": self.model_version,
            "model_agreement": abs(
                float(result["xgb_probs"][0]) - float(result["lgb_probs"][0])
            ) < 0.1,
        }
