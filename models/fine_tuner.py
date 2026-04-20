"""
Fine-Tuner with Optuna + TimeSeriesSplit + Probability Calibration
====================================================================
Bayesian hyperparameter optimization for XGBoost and LightGBM using
Optuna with time-series cross-validation, early stopping, and
post-hoc probability calibration (isotonic / Platt scaling).

Every trial is logged to MLflow.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import optuna
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACTS_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_CV_SPLITS,
    OPTUNA_EARLY_STOPPING,
    OPTUNA_SEARCH_SPACE,
    CALIBRATION_METHOD,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

# Suppress Optuna info spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


class FineTuner:
    """
    Optuna-based hyperparameter optimizer with:
      - TimeSeriesSplit cross-validation (preserves temporal order)
      - Early stopping on eval set per fold
      - Probability calibration (isotonic regression or Platt scaling)
      - Full MLflow logging for every trial

    Usage:
        tuner = FineTuner()
        result = tuner.fine_tune(data)
        # result contains best models, calibrators, and metrics
    """

    def __init__(
        self,
        n_trials: int = OPTUNA_N_TRIALS,
        timeout: int = OPTUNA_TIMEOUT,
        cv_splits: int = OPTUNA_CV_SPLITS,
        calibration_method: str = CALIBRATION_METHOD,
        experiment_name: str = "optuna_fine_tuning",
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_splits = cv_splits
        self.calibration_method = calibration_method
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        logger.info(
            f"FineTuner initialized: {n_trials} trials, "
            f"{cv_splits}-fold TSCV, calibration={calibration_method}"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def fine_tune(self, data: dict) -> dict:
        """
        Run full fine-tuning pipeline:
          1. Optuna HPO for XGBoost
          2. Optuna HPO for LightGBM
          3. Retrain best models on full train set
          4. Calibrate probabilities
          5. Save everything

        Parameters
        ----------
        data : dict
            Output from DatasetBuilder.build_dataset()

        Returns
        -------
        dict with keys:
            xgb_model, lgb_model — best tuned models
            xgb_calibrator, lgb_calibrator — probability calibrators
            xgb_best_params, lgb_best_params — optimal hyperparameters
            xgb_metrics, lgb_metrics — test set metrics
            xgb_study, lgb_study — Optuna study objects
        """
        logger.info("=" * 60)
        logger.info("  FINE-TUNING: Starting Optuna optimization")
        logger.info("=" * 60)

        # ── Step 1: Optimize XGBoost ──
        logger.info("─── XGBoost Optimization ───")
        xgb_study = self._optimize_xgboost(data)
        xgb_best_params = xgb_study.best_params
        logger.info(f"XGBoost best params: {xgb_best_params}")
        logger.info(f"XGBoost best CV AUC: {xgb_study.best_value:.4f}")

        # ── Step 2: Optimize LightGBM ──
        logger.info("─── LightGBM Optimization ───")
        lgb_study = self._optimize_lightgbm(data)
        lgb_best_params = lgb_study.best_params
        logger.info(f"LightGBM best params: {lgb_best_params}")
        logger.info(f"LightGBM best CV AUC: {lgb_study.best_value:.4f}")

        # ── Step 3: Retrain with best params on full train data ──
        logger.info("─── Retraining with optimal params ───")

        X_train = data["X_train_raw"]
        X_val = data["X_val_raw"]
        X_test = data["X_test_raw"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

        # Combine train + val for final training
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])

        xgb_model = self._train_final_xgb(
            xgb_best_params, X_trainval, y_trainval, X_test, y_test
        )
        lgb_model = self._train_final_lgb(
            lgb_best_params, X_trainval, y_trainval, X_test, y_test
        )

        # ── Step 4: Calibrate probabilities ──
        logger.info("─── Probability Calibration ───")
        xgb_calibrator = self._calibrate_model(
            xgb_model, X_val, y_val, "xgboost"
        )
        lgb_calibrator = self._calibrate_model(
            lgb_model, X_val, y_val, "lightgbm"
        )

        # ── Step 5: Final evaluation with calibrated probabilities ──
        xgb_metrics = self._evaluate_calibrated(
            xgb_model, xgb_calibrator, X_test, y_test, "xgboost"
        )
        lgb_metrics = self._evaluate_calibrated(
            lgb_model, lgb_calibrator, X_test, y_test, "lightgbm"
        )

        # ── Step 6: Save everything ──
        self._save_artifacts(
            xgb_model, lgb_model, xgb_calibrator, lgb_calibrator,
            xgb_best_params, lgb_best_params, xgb_metrics, lgb_metrics,
            data["feature_names"],
        )

        # ── Step 7: Log best run to MLflow ──
        self._log_best_run(
            xgb_best_params, lgb_best_params,
            xgb_metrics, lgb_metrics,
            xgb_study, lgb_study,
        )

        logger.info("=" * 60)
        logger.info("  FINE-TUNING: Complete!")
        logger.info(f"  XGB AUC: {xgb_metrics['auc_roc']:.4f} | "
                    f"LGB AUC: {lgb_metrics['auc_roc']:.4f}")
        logger.info(f"  XGB Brier: {xgb_metrics.get('brier_score', 0):.4f} | "
                    f"LGB Brier: {lgb_metrics.get('brier_score', 0):.4f}")
        logger.info("=" * 60)

        return {
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "xgb_calibrator": xgb_calibrator,
            "lgb_calibrator": lgb_calibrator,
            "xgb_best_params": xgb_best_params,
            "lgb_best_params": lgb_best_params,
            "xgb_metrics": xgb_metrics,
            "lgb_metrics": lgb_metrics,
            "xgb_study": xgb_study,
            "lgb_study": lgb_study,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Optuna objectives
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _optimize_xgboost(self, data: dict) -> optuna.Study:
        """Run Optuna study for XGBoost with TSCV."""
        X_all = pd.concat([data["X_train_raw"], data["X_val_raw"]])
        y_all = pd.concat([data["y_train"], data["y_val"]])

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *OPTUNA_SEARCH_SPACE["n_estimators"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *OPTUNA_SEARCH_SPACE["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *OPTUNA_SEARCH_SPACE["learning_rate"], log=True
                ),
                "subsample": trial.suggest_float(
                    "subsample", *OPTUNA_SEARCH_SPACE["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *OPTUNA_SEARCH_SPACE["colsample_bytree"]
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
            }

            # TimeSeriesSplit cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            fold_aucs = []

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
                X_fold_train, X_fold_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
                y_fold_train, y_fold_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

                es_rounds = OPTUNA_EARLY_STOPPING
                fit_params = params.copy()
                n_est = fit_params.pop("n_estimators")

                model = XGBClassifier(n_estimators=n_est, **fit_params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )

                y_prob = model.predict_proba(X_fold_val)[:, 1]

                if len(np.unique(y_fold_val)) > 1:
                    fold_auc = roc_auc_score(y_fold_val, y_prob)
                else:
                    fold_auc = 0.5

                fold_aucs.append(fold_auc)

                # Prune unpromising trials early
                trial.report(np.mean(fold_aucs), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_auc = np.mean(fold_aucs)

            # Log trial to MLflow
            with mlflow.start_run(
                run_name=f"xgb_trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mean_auc", mean_auc)
                mlflow.log_metric("cv_std_auc", np.std(fold_aucs))
                for i, auc in enumerate(fold_aucs):
                    mlflow.log_metric(f"fold_{i}_auc", auc)

            return mean_auc

        # Wrap the whole study in a parent MLflow run
        with mlflow.start_run(run_name=f"xgb_optuna_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            )
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )

            mlflow.log_metric("best_cv_auc", study.best_value)
            mlflow.log_params({
                f"best_{k}": v for k, v in study.best_params.items()
            })
            mlflow.log_metric("total_trials", len(study.trials))
            mlflow.log_metric("pruned_trials", len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]))

        return study

    def _optimize_lightgbm(self, data: dict) -> optuna.Study:
        """Run Optuna study for LightGBM with TSCV."""
        X_all = pd.concat([data["X_train_raw"], data["X_val_raw"]])
        y_all = pd.concat([data["y_train"], data["y_val"]])

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *OPTUNA_SEARCH_SPACE["n_estimators"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *OPTUNA_SEARCH_SPACE["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *OPTUNA_SEARCH_SPACE["learning_rate"], log=True
                ),
                "subsample": trial.suggest_float(
                    "subsample", *OPTUNA_SEARCH_SPACE["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *OPTUNA_SEARCH_SPACE["colsample_bytree"]
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "objective": "binary",
                "metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            fold_aucs = []

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
                X_fold_train, X_fold_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
                y_fold_train, y_fold_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

                fit_params = params.copy()
                n_est = fit_params.pop("n_estimators")

                model = LGBMClassifier(n_estimators=n_est, **fit_params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                )

                y_prob = model.predict_proba(X_fold_val)[:, 1]

                if len(np.unique(y_fold_val)) > 1:
                    fold_auc = roc_auc_score(y_fold_val, y_prob)
                else:
                    fold_auc = 0.5

                fold_aucs.append(fold_auc)

                trial.report(np.mean(fold_aucs), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_auc = np.mean(fold_aucs)

            with mlflow.start_run(
                run_name=f"lgb_trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mean_auc", mean_auc)
                mlflow.log_metric("cv_std_auc", np.std(fold_aucs))
                for i, auc in enumerate(fold_aucs):
                    mlflow.log_metric(f"fold_{i}_auc", auc)

            return mean_auc

        with mlflow.start_run(run_name=f"lgb_optuna_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            )
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )

            mlflow.log_metric("best_cv_auc", study.best_value)
            mlflow.log_params({
                f"best_{k}": v for k, v in study.best_params.items()
            })
            mlflow.log_metric("total_trials", len(study.trials))
            mlflow.log_metric("pruned_trials", len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]))

        return study

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Final training with best params
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _train_final_xgb(
        self, best_params, X_train, y_train, X_test, y_test
    ) -> XGBClassifier:
        """Retrain XGBoost with best params on full train data."""
        params = {
            **best_params,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        logger.info(f"Final XGBoost test AUC: {auc:.4f}")

        return model

    def _train_final_lgb(
        self, best_params, X_train, y_train, X_test, y_test
    ) -> LGBMClassifier:
        """Retrain LightGBM with best params on full train data."""
        params = {
            **best_params,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        logger.info(f"Final LightGBM test AUC: {auc:.4f}")

        return model

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Probability calibration
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calibrate_model(
        self, model, X_cal, y_cal, model_name: str
    ) -> CalibratedClassifierCV:
        """
        Calibrate model probabilities using isotonic regression or Platt scaling.

        Uses the validation set as the calibration set.
        CalibratedClassifierCV with cv='prefit' wraps the already-trained model.
        """
        logger.info(
            f"Calibrating {model_name} with {self.calibration_method} regression..."
        )

        # Use FrozenEstimator if available (sklearn >= 1.6), fall back to cv='prefit'
        try:
            from sklearn.frozen import FrozenEstimator
            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(model),
                method=self.calibration_method,
            )
        except ImportError:
            calibrator = CalibratedClassifierCV(
                estimator=model,
                method=self.calibration_method,
                cv="prefit",
            )
        calibrator.fit(X_cal, y_cal)

        # Evaluate calibration: compare raw vs calibrated Brier score
        raw_probs = model.predict_proba(X_cal)[:, 1]
        cal_probs = calibrator.predict_proba(X_cal)[:, 1]

        raw_brier = brier_score_loss(y_cal, raw_probs)
        cal_brier = brier_score_loss(y_cal, cal_probs)

        logger.info(
            f"  {model_name} Brier score: "
            f"raw={raw_brier:.4f} → calibrated={cal_brier:.4f} "
            f"({'improved' if cal_brier < raw_brier else 'no improvement'})"
        )

        return calibrator

    def _evaluate_calibrated(
        self, model, calibrator, X_test, y_test, model_name: str
    ) -> dict:
        """Evaluate using calibrated probabilities."""
        y_pred = calibrator.predict(X_test)
        cal_probs = calibrator.predict_proba(X_test)[:, 1]
        raw_probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, cal_probs)
            if len(np.unique(y_test)) > 1 else 0.0,
            "brier_score": brier_score_loss(y_test, cal_probs),
            "brier_score_raw": brier_score_loss(y_test, raw_probs),
            "log_loss": log_loss(y_test, cal_probs),
        }

        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"\n{model_name} Calibrated Classification Report:\n{report}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{model_name} Confusion Matrix:\n{cm}")

        return metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Persistence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_artifacts(
        self, xgb_model, lgb_model, xgb_cal, lgb_cal,
        xgb_params, lgb_params, xgb_metrics, lgb_metrics,
        feature_names,
    ):
        """Save all tuned models, calibrators, and metadata."""
        # Save base models (overwrite existing)
        joblib.dump(xgb_model, MODEL_ARTIFACTS_DIR / "xgboost_model.joblib")
        joblib.dump(lgb_model, MODEL_ARTIFACTS_DIR / "lightgbm_model.joblib")

        # Save calibrators
        joblib.dump(xgb_cal, MODEL_ARTIFACTS_DIR / "xgb_calibrator.joblib")
        joblib.dump(lgb_cal, MODEL_ARTIFACTS_DIR / "lgb_calibrator.joblib")

        # Save best params
        joblib.dump(
            {"xgboost": xgb_params, "lightgbm": lgb_params},
            MODEL_ARTIFACTS_DIR / "best_params.joblib",
        )

        # Save tuning metadata
        metadata = {
            "tuned_at": datetime.now().isoformat(),
            "calibration_method": self.calibration_method,
            "n_trials": self.n_trials,
            "cv_splits": self.cv_splits,
            "xgb_test_auc": xgb_metrics["auc_roc"],
            "lgb_test_auc": lgb_metrics["auc_roc"],
            "xgb_brier": xgb_metrics.get("brier_score", 0),
            "lgb_brier": lgb_metrics.get("brier_score", 0),
            "feature_names": feature_names,
        }
        joblib.dump(metadata, MODEL_ARTIFACTS_DIR / "tuning_metadata.joblib")

        logger.info(f"All tuned artifacts saved to {MODEL_ARTIFACTS_DIR}")

    def _log_best_run(
        self, xgb_params, lgb_params,
        xgb_metrics, lgb_metrics,
        xgb_study, lgb_study,
    ):
        """Log the best overall result as a registered MLflow run."""
        with mlflow.start_run(
            run_name=f"best_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            # XGBoost best
            for k, v in xgb_params.items():
                mlflow.log_param(f"xgb_{k}", v)
            for k, v in xgb_metrics.items():
                mlflow.log_metric(f"xgb_{k}", v)

            # LightGBM best
            for k, v in lgb_params.items():
                mlflow.log_param(f"lgb_{k}", v)
            for k, v in lgb_metrics.items():
                mlflow.log_metric(f"lgb_{k}", v)

            # Study stats
            mlflow.log_metric("xgb_best_cv_auc", xgb_study.best_value)
            mlflow.log_metric("lgb_best_cv_auc", lgb_study.best_value)
            mlflow.log_param("calibration_method", self.calibration_method)
            mlflow.log_param("cv_splits", self.cv_splits)

            # Log model artifacts
            for artifact in MODEL_ARTIFACTS_DIR.glob("*.joblib"):
                mlflow.log_artifact(str(artifact))

            logger.info("Best tuned models registered in MLflow")
