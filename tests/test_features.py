"""
Tests for Feature Engineering
================================
Validates that all 29 features are computed correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.feature_engine import FeatureEngine
from data.nse_fetcher import generate_synthetic_data
from config import FEATURE_COLUMNS


@pytest.fixture
def sample_data():
    """Generate synthetic options data for testing."""
    return generate_synthetic_data(symbol="NIFTY", num_strikes=5, num_expiries=2)


@pytest.fixture
def feature_engine():
    return FeatureEngine()


class TestFeatureComputation:
    """Test that all features are computed."""

    def test_all_features_present(self, feature_engine, sample_data):
        """All 29 feature columns should be present after computation."""
        result = feature_engine.compute_features(sample_data)

        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self, feature_engine, sample_data):
        """Features should not contain NaN after computation."""
        result = feature_engine.compute_features(sample_data)

        for col in FEATURE_COLUMNS:
            nan_count = result[col].isna().sum()
            assert nan_count == 0, f"Feature {col} has {nan_count} NaN values"

    def test_moneyness_range(self, feature_engine, sample_data):
        """Moneyness should be positive and reasonable."""
        result = feature_engine.compute_features(sample_data)
        assert (result["moneyness"] > 0).all()
        assert (result["moneyness"] < 5).all()

    def test_time_to_expiry_positive(self, feature_engine, sample_data):
        """Time to expiry should be positive."""
        result = feature_engine.compute_features(sample_data)
        assert (result["time_to_expiry"] > 0).all()

    def test_implied_volatility_range(self, feature_engine, sample_data):
        """IV should be between 0 and reasonable bounds."""
        result = feature_engine.compute_features(sample_data)
        valid_iv = result["implied_volatility"][result["implied_volatility"] > 0]
        if len(valid_iv) > 0:
            assert (valid_iv < 5.0).all(), "IV should be below 500%"

    def test_delta_range(self, feature_engine, sample_data):
        """Delta should be between -1 and 1."""
        result = feature_engine.compute_features(sample_data)
        assert (result["delta"] >= -1.1).all()
        assert (result["delta"] <= 1.1).all()

    def test_gamma_non_negative(self, feature_engine, sample_data):
        """Gamma should be non-negative."""
        result = feature_engine.compute_features(sample_data)
        assert (result["gamma"] >= -0.01).all()

    def test_volume_oi_ratio_non_negative(self, feature_engine, sample_data):
        """Volume/OI ratio should be non-negative."""
        result = feature_engine.compute_features(sample_data)
        assert (result["volume_oi_ratio"] >= 0).all()

    def test_bid_ask_spread_non_negative(self, feature_engine, sample_data):
        """Bid-ask spread should be non-negative."""
        result = feature_engine.compute_features(sample_data)
        assert (result["bid_ask_spread"] >= 0).all()

    def test_extrinsic_pct_bounded(self, feature_engine, sample_data):
        """Extrinsic percentage should be between 0 and 1."""
        result = feature_engine.compute_features(sample_data)
        assert (result["extrinsic_pct"] >= 0).all()
        assert (result["extrinsic_pct"] <= 1).all()

    def test_empty_dataframe(self, feature_engine):
        """Empty DataFrame should return empty."""
        result = feature_engine.compute_features(pd.DataFrame())
        assert result.empty

    def test_output_row_count(self, feature_engine, sample_data):
        """Row count should be preserved."""
        result = feature_engine.compute_features(sample_data)
        assert len(result) == len(sample_data)
