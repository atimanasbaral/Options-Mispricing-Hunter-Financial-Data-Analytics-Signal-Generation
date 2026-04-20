"""
Population Stability Index (PSI)
==================================
Measures distributional shift between training and production features.
Used to detect data drift and trigger model retraining.

Interpretation:
  PSI < 0.1  → Stable (no action needed)
  0.1 ≤ PSI < 0.2 → Moderate shift (investigate)
  PSI ≥ 0.2 → Significant drift (retrain model)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import PSI_BUCKETS, PSI_THRESHOLD, PSI_WARNING_THRESHOLD, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = PSI_BUCKETS,
) -> float:
    """
    Calculate Population Stability Index for a single feature.

    Parameters
    ----------
    expected : np.ndarray
        Reference distribution (training data).
    actual : np.ndarray
        Current distribution (production data).
    buckets : int
        Number of quantile bins.

    Returns
    -------
    float — PSI value
    """
    # Remove NaN and inf
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < buckets or len(actual) < buckets:
        return 0.0

    # Create bins from expected (reference) distribution
    try:
        _, bin_edges = pd.qcut(expected, q=buckets, retbins=True, duplicates="drop")
    except ValueError:
        # Fall back to equal-width bins if quantile fails
        bin_edges = np.linspace(expected.min(), expected.max(), buckets + 1)

    # Count observations in each bin
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Convert to percentages (add small epsilon to avoid log(0))
    epsilon = 1e-4
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize to ensure they sum to ~1
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    return float(np.sum(psi_values))


def calculate_feature_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list = None,
    buckets: int = PSI_BUCKETS,
) -> Dict[str, float]:
    """
    Calculate PSI for each feature column.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training/reference data.
    current_df : pd.DataFrame
        Current production data.
    feature_columns : list, optional
        Columns to evaluate. Defaults to FEATURE_COLUMNS.
    buckets : int
        Number of bins for PSI calculation.

    Returns
    -------
    dict — mapping feature_name → PSI value
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    psi_scores = {}
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_values = reference_df[col].dropna().values.astype(float)
        cur_values = current_df[col].dropna().values.astype(float)

        if len(ref_values) == 0 or len(cur_values) == 0:
            psi_scores[col] = 0.0
            continue

        psi_scores[col] = calculate_psi(ref_values, cur_values, buckets)

    return psi_scores


def calculate_overall_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list = None,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Calculate overall PSI (mean across all features) and determine drift status.

    Returns
    -------
    tuple of (overall_psi, drift_status, feature_breakdown)
        drift_status: 'stable', 'warning', or 'critical'
    """
    feature_psi = calculate_feature_psi(reference_df, current_df, feature_columns)

    if not feature_psi:
        return 0.0, "stable", {}

    overall_psi = float(np.mean(list(feature_psi.values())))

    # Determine status
    if overall_psi >= PSI_THRESHOLD:
        drift_status = "critical"
    elif overall_psi >= PSI_WARNING_THRESHOLD:
        drift_status = "warning"
    else:
        drift_status = "stable"

    # Log drifting features
    drifting = {k: v for k, v in feature_psi.items() if v >= PSI_WARNING_THRESHOLD}
    if drifting:
        logger.warning(
            f"Drifting features (PSI ≥ {PSI_WARNING_THRESHOLD}): "
            f"{', '.join(f'{k}={v:.3f}' for k, v in sorted(drifting.items(), key=lambda x: -x[1]))}"
        )

    logger.info(
        f"Overall PSI: {overall_psi:.4f} — Status: {drift_status.upper()}"
    )

    return overall_psi, drift_status, feature_psi
