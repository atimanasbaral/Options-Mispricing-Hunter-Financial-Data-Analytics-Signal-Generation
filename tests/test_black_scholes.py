"""
Tests for Black-Scholes Engine
=================================
Validates pricing, IV solver, Greeks, and put-call parity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from data.black_scholes import (
    bs_call_price,
    bs_put_price,
    bs_price,
    implied_volatility,
    delta,
    gamma,
    theta,
    vega,
    rho,
    compute_all_greeks,
    put_call_parity_deviation,
)


class TestBlackScholesPricing:
    """Test BS pricing formulas against known values."""

    def test_call_price_atm(self):
        """ATM call should be roughly S * 0.4 * sigma * sqrt(T) for small sigma."""
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0)
        # ATM 1-year call with 20% vol should be ~$10-12
        assert 8.0 < price < 15.0

    def test_put_price_atm(self):
        """ATM put with no dividends."""
        price = bs_put_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0)
        assert 3.0 < price < 10.0

    def test_deep_itm_call(self):
        """Deep ITM call ≈ intrinsic value."""
        price = bs_call_price(S=150, K=100, T=0.1, r=0.05, sigma=0.2, q=0)
        assert price > 49.0  # At least intrinsic

    def test_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        price = bs_call_price(S=50, K=100, T=0.1, r=0.05, sigma=0.2, q=0)
        assert price < 0.1

    def test_zero_time_call(self):
        """At expiry, call = max(S-K, 0)."""
        assert bs_call_price(S=110, K=100, T=0, r=0.05, sigma=0.2, q=0) == pytest.approx(10.0, abs=1.0)
        assert bs_call_price(S=90, K=100, T=0, r=0.05, sigma=0.2, q=0) == 0.0

    def test_put_call_parity(self):
        """C - P = S*e^{-qT} - K*e^{-rT} must hold."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.0
        call = bs_call_price(S, K, T, r, sigma, q)
        put = bs_put_price(S, K, T, r, sigma, q)
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-6)


class TestImpliedVolatility:
    """Test IV solver accuracy."""

    def test_iv_recovery(self):
        """IV solver should recover the original sigma used to price."""
        true_sigma = 0.25
        S, K, T, r, q = 100, 105, 0.5, 0.05, 0.01

        market_price = bs_call_price(S, K, T, r, true_sigma, q)
        recovered_iv = implied_volatility(market_price, S, K, T, r, q, "CE")

        assert recovered_iv == pytest.approx(true_sigma, abs=1e-4)

    def test_iv_put(self):
        """IV recovery for puts."""
        true_sigma = 0.30
        S, K, T, r, q = 22500, 22000, 0.1, 0.065, 0.012

        market_price = bs_put_price(S, K, T, r, true_sigma, q)
        recovered_iv = implied_volatility(market_price, S, K, T, r, q, "PE")

        assert recovered_iv == pytest.approx(true_sigma, abs=1e-3)

    def test_iv_returns_nan_for_bad_inputs(self):
        """IV should return NaN for impossible prices."""
        iv = implied_volatility(market_price=-5, S=100, K=100, T=1, r=0.05)
        assert np.isnan(iv)

        iv2 = implied_volatility(market_price=10, S=100, K=100, T=0, r=0.05)
        assert np.isnan(iv2)


class TestGreeks:
    """Test Greeks against known properties."""

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        d = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0, option_type="CE")
        assert 0 < d < 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        d = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0, option_type="PE")
        assert -1 < d < 0

    def test_atm_delta_near_half(self):
        """ATM call delta ≈ 0.5 (slightly above due to drift)."""
        d = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0, option_type="CE")
        assert 0.45 < d < 0.65

    def test_gamma_positive(self):
        """Gamma is always positive."""
        g = gamma(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0)
        assert g > 0

    def test_theta_negative_for_long(self):
        """Theta for long options should be negative (time decay)."""
        t = theta(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0, option_type="CE")
        assert t < 0

    def test_vega_positive(self):
        """Vega should be positive (higher vol → higher price)."""
        v = vega(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0)
        assert v > 0

    def test_compute_all_greeks(self):
        """compute_all_greeks should return all 5 Greeks."""
        greeks = compute_all_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0, option_type="CE")
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks


class TestPutCallParity:
    """Test PCP deviation calculation."""

    def test_pcp_deviation_zero_for_fair(self):
        """BS-priced options should have near-zero PCP deviation."""
        S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0

        call = bs_call_price(S, K, T, r, sigma, q)
        put = bs_put_price(S, K, T, r, sigma, q)

        dev = put_call_parity_deviation(call, put, S, K, T, r, q)
        assert abs(dev) < 1e-6

    def test_pcp_deviation_nonzero_for_mispriced(self):
        """Artificially mispriced options should show deviation."""
        S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0
        call = bs_call_price(S, K, T, r, sigma, q) + 5  # Overpriced call
        put = bs_put_price(S, K, T, r, sigma, q)

        dev = put_call_parity_deviation(call, put, S, K, T, r, q)
        assert dev > 4.0  # Should be close to 5
