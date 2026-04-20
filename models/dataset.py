"""
Dataset Preparation
=====================
Loads historical features from SQLite, creates mispricing labels,
and prepares train/validation/test splits for model training.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from data.database import OptionsDatabase
from config import (
    FEATURE_COLUMNS,
    MISPRICING_ZSCORE_THRESHOLD,
    MODEL_ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Prepares ML-ready datasets from stored features.

    Labeling strategy:
    ─────────────────
    Uses put-call parity deviation z-score as a proxy for mispricing.
    Options with |z-score| > threshold are labeled as mispriced (1),
    others as fairly priced (0).
    """

    def __init__(self, db: OptionsDatabase = None):
        self.db = db or OptionsDatabase()
        self.scaler = StandardScaler()

    def build_dataset(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> dict:
        """
        Build labeled dataset from historical features.

        Returns
        -------
        dict with keys:
            X_train, X_val, X_test : pd.DataFrame
            y_train, y_val, y_test : pd.Series
            feature_names : list
            scaler : StandardScaler
        """
        logger.info("Building dataset from stored features...")

        # Load all features
        df = self.db.get_all_features()

        if df.empty:
            logger.error("No feature data found in database!")
            raise ValueError(
                "No data in features table. Run pipeline.seed_historical_data() first."
            )

        logger.info(f"Loaded {len(df)} feature records")

        # Create labels
        df = self._create_labels(df)

        # Select feature columns that exist
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = set(FEATURE_COLUMNS) - set(available_features)
        if missing:
            logger.warning(f"Missing features: {missing}")

        # Drop rows with NaN in features or label
        df_clean = df[available_features + ["mispricing_label"]].dropna()
        logger.info(f"Clean records after NaN removal: {len(df_clean)}")

        if len(df_clean) < 100:
            logger.warning("Very few clean records — model performance may be poor")

        X = df_clean[available_features]
        y = df_clean["mispricing_label"].astype(int)

        # Log class distribution
        class_dist = y.value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        # Time-based split (preserve temporal ordering)
        # First split: train+val vs test
        split_idx_test = int(len(X) * (1 - test_size))
        X_trainval, X_test = X.iloc[:split_idx_test], X.iloc[split_idx_test:]
        y_trainval, y_test = y.iloc[:split_idx_test], y.iloc[split_idx_test:]

        # Second split: train vs val
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=relative_val_size,
            random_state=42,
            stratify=y_trainval if len(y_trainval.unique()) > 1 else None,
        )

        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=available_features,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=available_features,
            index=X_val.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=available_features,
            index=X_test.index,
        )

        # Save scaler
        scaler_path = MODEL_ARTIFACTS_DIR / "feature_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        logger.info(
            f"Dataset splits — Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": available_features,
            "scaler": self.scaler,
            # Unscaled versions for tree-based models
            "X_train_raw": X_train,
            "X_val_raw": X_val,
            "X_test_raw": X_test,
        }

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mispricing labels based on put-call parity deviation.

        Strategy: |pcp_deviation_zscore| > threshold → mispriced (1)
        """
        if "mispricing_label" in df.columns and df["mispricing_label"].notna().any():
            logger.info("Using existing mispricing labels")
            return df

        if "pcp_deviation_zscore" not in df.columns:
            logger.warning("pcp_deviation_zscore not found, generating from parity dev")
            if "put_call_parity_dev" in df.columns:
                mean_dev = df["put_call_parity_dev"].mean()
                std_dev = df["put_call_parity_dev"].std()
                if std_dev > 0:
                    df["pcp_deviation_zscore"] = (
                        (df["put_call_parity_dev"] - mean_dev) / std_dev
                    )
                else:
                    df["pcp_deviation_zscore"] = 0.0
            else:
                df["pcp_deviation_zscore"] = 0.0

        df["mispricing_label"] = (
            df["pcp_deviation_zscore"].abs() > MISPRICING_ZSCORE_THRESHOLD
        ).astype(int)

        # Add directional label (for signal generation)
        df["mispricing_direction"] = np.where(
            df["pcp_deviation_zscore"] > MISPRICING_ZSCORE_THRESHOLD,
            1,  # Overpriced
            np.where(
                df["pcp_deviation_zscore"] < -MISPRICING_ZSCORE_THRESHOLD,
                -1,  # Underpriced
                0,   # Fair
            ),
        )

        mispriced_pct = df["mispricing_label"].mean() * 100
        logger.info(f"Labeled {mispriced_pct:.1f}% of contracts as mispriced")

        return df

    def prepare_inference_data(
        self, df: pd.DataFrame, scaler: StandardScaler = None
    ) -> pd.DataFrame:
        """
        Prepare a live DataFrame for model inference.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered options data.
        scaler : StandardScaler, optional
            Pre-fitted scaler. Loads from disk if not provided.

        Returns
        -------
        pd.DataFrame — Scaled features ready for prediction.
        """
        if scaler is None:
            scaler_path = MODEL_ARTIFACTS_DIR / "feature_scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            else:
                logger.warning("No scaler found — returning unscaled data")
                return df[FEATURE_COLUMNS].fillna(0)

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0)

        return pd.DataFrame(
            scaler.transform(X),
            columns=available,
            index=df.index,
        )
