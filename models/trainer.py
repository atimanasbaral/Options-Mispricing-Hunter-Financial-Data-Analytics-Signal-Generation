"""
Model Trainer
===============
Trains XGBoost and LightGBM classifiers with full MLflow experiment tracking.
Logs parameters, metrics, feature importance, and model artifacts.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains XGBoost and LightGBM models with MLflow tracking.

    Usage:
        trainer = ModelTrainer()
        xgb_model, xgb_metrics = trainer.train_xgboost(data)
        lgb_model, lgb_metrics = trainer.train_lightgbm(data)
    """

    def __init__(self, experiment_name: str = "options_mispricing"):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    def train_xgboost(self, data: dict) -> tuple:
        """
        Train XGBoost classifier with MLflow logging.

        Parameters
        ----------
        data : dict
            Output from DatasetBuilder.build_dataset()

        Returns
        -------
        (XGBClassifier, dict) — trained model and metrics dict
        """
        logger.info("Training XGBoost model...")

        # Tree-based models work best with unscaled features
        X_train = data["X_train_raw"]
        X_val = data["X_val_raw"]
        X_test = data["X_test_raw"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

        params = XGBOOST_PARAMS.copy()
        early_stopping = params.pop("early_stopping_rounds", 50)

        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params(params)
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("n_features", len(data["feature_names"]))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))

            model = XGBClassifier(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test, "xgboost")

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=data["feature_names"],
            ).sort_values(ascending=False)

            mlflow.log_text(
                importance.to_string(),
                "feature_importance_xgboost.txt",
            )

            # Save model
            model_path = MODEL_ARTIFACTS_DIR / "xgboost_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            logger.info(f"XGBoost — Accuracy: {metrics['accuracy']:.4f}, "
                        f"AUC: {metrics['auc_roc']:.4f}")

        return model, metrics

    def train_lightgbm(self, data: dict) -> tuple:
        """
        Train LightGBM classifier with MLflow logging.

        Parameters
        ----------
        data : dict
            Output from DatasetBuilder.build_dataset()

        Returns
        -------
        (LGBMClassifier, dict) — trained model and metrics dict
        """
        logger.info("Training LightGBM model...")

        X_train = data["X_train_raw"]
        X_val = data["X_val_raw"]
        X_test = data["X_test_raw"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

        params = LIGHTGBM_PARAMS.copy()

        with mlflow.start_run(run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params(params)
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("n_features", len(data["feature_names"]))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))

            model = LGBMClassifier(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=None,  # LightGBM handles verbosity via params
            )

            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test, "lightgbm")

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=data["feature_names"],
            ).sort_values(ascending=False)

            mlflow.log_text(
                importance.to_string(),
                "feature_importance_lightgbm.txt",
            )

            # Save model
            model_path = MODEL_ARTIFACTS_DIR / "lightgbm_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            logger.info(f"LightGBM — Accuracy: {metrics['accuracy']:.4f}, "
                        f"AUC: {metrics['auc_roc']:.4f}")

        return model, metrics

    def _evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> dict:
        """Compute classification metrics."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0,
        }

        # Log classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"\n{model_name} Classification Report:\n{report}")

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{model_name} Confusion Matrix:\n{cm}")

        return metrics

    def train_both(self, data: dict) -> dict:
        """
        Train both XGBoost and LightGBM, return models and metrics.

        Returns
        -------
        dict with keys: xgb_model, lgb_model, xgb_metrics, lgb_metrics
        """
        xgb_model, xgb_metrics = self.train_xgboost(data)
        lgb_model, lgb_metrics = self.train_lightgbm(data)

        return {
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "xgb_metrics": xgb_metrics,
            "lgb_metrics": lgb_metrics,
        }
