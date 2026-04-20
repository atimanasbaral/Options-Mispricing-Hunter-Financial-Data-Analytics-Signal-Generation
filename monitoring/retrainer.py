"""
Auto-Retraining Monitor
=========================
Periodically checks for data drift via PSI and triggers
model retraining when drift exceeds the critical threshold.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from data.database import OptionsDatabase
from models.dataset import DatasetBuilder
from models.trainer import ModelTrainer
from models.ensemble import EnsemblePredictor
from monitoring.psi import calculate_overall_psi
from config import (
    PSI_THRESHOLD,
    PSI_CHECK_INTERVAL_HOURS,
    PSI_LOOKBACK_DAYS,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitors data drift and triggers retraining.

    Workflow:
    1. Periodically load recent production features
    2. Compare against training distribution (stored reference)
    3. If PSI ≥ threshold → retrain models
    4. Only deploy new model if it improves on validation metrics
    """

    def __init__(
        self,
        db: Optional[OptionsDatabase] = None,
        ensemble: Optional[EnsemblePredictor] = None,
        psi_threshold: float = PSI_THRESHOLD,
    ):
        self.db = db or OptionsDatabase()
        self.ensemble = ensemble
        self.psi_threshold = psi_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reference_data: Optional[pd.DataFrame] = None
        self._last_check: Optional[datetime] = None
        self._retrain_count = 0

    def set_reference_data(self, df: pd.DataFrame):
        """Set the training data distribution as reference for PSI."""
        self._reference_data = df
        logger.info(f"Reference data set: {len(df)} records")

    def check_drift(self) -> dict:
        """
        Run a single drift check.

        Returns
        -------
        dict with keys: psi_score, drift_status, feature_breakdown,
                        needs_retrain, checked_at
        """
        if self._reference_data is None or self._reference_data.empty:
            logger.warning("No reference data set — building from training data")
            try:
                builder = DatasetBuilder(self.db)
                data = builder.build_dataset()
                self._reference_data = data["X_train_raw"]
            except Exception as e:
                logger.error(f"Cannot build reference: {e}")
                return {
                    "psi_score": 0.0,
                    "drift_status": "unknown",
                    "feature_breakdown": {},
                    "needs_retrain": False,
                    "checked_at": datetime.now().isoformat(),
                }

        # Get recent production data
        recent_df = self.db.get_feature_stats(days=PSI_LOOKBACK_DAYS)

        if recent_df.empty:
            logger.info("No recent production data for drift check")
            return {
                "psi_score": 0.0,
                "drift_status": "no_data",
                "feature_breakdown": {},
                "needs_retrain": False,
                "checked_at": datetime.now().isoformat(),
            }

        # Calculate PSI
        psi_score, drift_status, feature_breakdown = calculate_overall_psi(
            self._reference_data,
            recent_df,
            FEATURE_COLUMNS,
        )

        needs_retrain = drift_status == "critical"

        # Store health record
        health_record = {
            "psi_score": psi_score,
            "drift_status": drift_status,
            "model_version": self.ensemble.model_version if self.ensemble else "unknown",
            "last_retrain": "",
            "feature_psi_breakdown": feature_breakdown,
        }

        try:
            self.db.insert_model_health(health_record)
        except Exception as e:
            logger.error(f"Failed to store health record: {e}")

        self._last_check = datetime.now()

        result = {
            "psi_score": psi_score,
            "drift_status": drift_status,
            "feature_breakdown": feature_breakdown,
            "needs_retrain": needs_retrain,
            "checked_at": self._last_check.isoformat(),
        }

        if needs_retrain:
            logger.warning(
                f"DRIFT DETECTED! PSI={psi_score:.4f} > {self.psi_threshold}. "
                "Triggering retraining..."
            )
            self._trigger_retrain()

        return result

    def _trigger_retrain(self):
        """Retrain models and update ensemble if performance improves."""
        logger.info("=" * 50)
        logger.info("AUTO-RETRAIN: Starting retraining pipeline...")
        logger.info("=" * 50)

        try:
            # Build fresh dataset
            builder = DatasetBuilder(self.db)
            data = builder.build_dataset()

            # Train new models
            trainer = ModelTrainer(experiment_name="auto_retrain")
            results = trainer.train_both(data)

            # Check if new models are better
            old_metrics = getattr(self.ensemble, "_last_metrics", None)
            new_auc = (
                results["xgb_metrics"]["auc_roc"] * 0.55
                + results["lgb_metrics"]["auc_roc"] * 0.45
            )

            if old_metrics:
                old_auc = old_metrics.get("ensemble_auc", 0)
                if new_auc <= old_auc:
                    logger.info(
                        f"New model AUC ({new_auc:.4f}) not better than "
                        f"old ({old_auc:.4f}). Keeping current model."
                    )
                    return

            # Update ensemble
            if self.ensemble:
                self.ensemble.xgb_model = results["xgb_model"]
                self.ensemble.lgb_model = results["lgb_model"]
                self.ensemble.model_version = datetime.now().strftime("%Y%m%d_%H%M")
                self.ensemble._last_metrics = {"ensemble_auc": new_auc}

            # Update reference data
            self._reference_data = data["X_train_raw"]
            self._retrain_count += 1

            logger.info(
                f"AUTO-RETRAIN: Complete! New model AUC: {new_auc:.4f}. "
                f"Total retrains: {self._retrain_count}"
            )

        except Exception as e:
            logger.error(f"AUTO-RETRAIN: Failed — {e}", exc_info=True)

    def start_monitoring(self, interval_hours: float = PSI_CHECK_INTERVAL_HOURS):
        """Start drift monitoring in a background thread."""
        if self._running:
            logger.warning("Drift monitor already running")
            return

        self._running = True
        interval_seconds = interval_hours * 3600

        def _loop():
            while self._running:
                try:
                    self.check_drift()
                except Exception as e:
                    logger.error(f"Drift monitor error: {e}")
                time.sleep(interval_seconds)

        self._thread = threading.Thread(
            target=_loop, daemon=True, name="DriftMonitor"
        )
        self._thread.start()
        logger.info(f"Drift monitor started (check every {interval_hours}h)")

    def stop_monitoring(self):
        """Stop the background drift monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Drift monitor stopped")
