"""
Black-Scholes Pricing Engine
==============================
Analytical Black-Scholes-Merton pricing, implied volatility solver,
and first-order Greeks — all vectorized with NumPy.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from config import (
    RISK_FREE_RATE,
    DIVIDEND_YIELD,
    IV_LOWER_BOUND,
    IV_UPPER_BOUND,
    IV_TOLERANCE,
)


# ──────────────────────────────────────────────
# Core pricing functions
# ──────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_call_price(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
) -> float:
    """
    Black-Scholes European call option price.

    Parameters
    ----------
    S : float — Spot price of the underlying
    K : float — Strike price
    T : float — Time to expiry in years
    r : float — Risk-free interest rate
    sigma : float — Volatility
    q : float — Continuous dividend yield

    Returns
    -------
    float — Call option price
    """
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * max(T, 0)) - K * np.exp(-r * max(T, 0)), 0.0)

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
) -> float:
    """
    Black-Scholes European put option price.
    Uses put-call parity: P = C - S*e^{-qT} + K*e^{-rT}
    """
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * max(T, 0)) - S * np.exp(-q * max(T, 0)), 0.0)

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_price(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> float:
    """Dispatch to call or put price based on option_type ('CE' or 'PE')."""
    if option_type.upper() in ("CE", "CALL", "C"):
        return bs_call_price(S, K, T, r, sigma, q)
    elif option_type.upper() in ("PE", "PUT", "P"):
        return bs_put_price(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"Unknown option_type: {option_type}. Use 'CE' or 'PE'.")


# ──────────────────────────────────────────────
# Implied Volatility solver
# ──────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> float:
    """
    Solve for implied volatility using Brent's method.

    Returns NaN if the solver fails (e.g., no-arbitrage bounds violated).
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    # Check intrinsic value bounds
    if option_type.upper() in ("CE", "CALL", "C"):
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < intrinsic * 0.99:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        iv = brentq(
            objective,
            IV_LOWER_BOUND,
            IV_UPPER_BOUND,
            xtol=IV_TOLERANCE,
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def implied_volatility_vectorized(
    market_prices: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float = RISK_FREE_RATE,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> np.ndarray:
    """Vectorized IV calculation over arrays of options."""
    ivs = np.full(len(market_prices), np.nan)
    for i in range(len(market_prices)):
        ivs[i] = implied_volatility(
            market_prices[i],
            S[i] if isinstance(S, np.ndarray) else S,
            K[i],
            T[i] if isinstance(T, np.ndarray) else T,
            r, q, option_type,
        )
    return ivs


# ──────────────────────────────────────────────
# Greeks (analytical closed-form)
# ──────────────────────────────────────────────

def delta(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> float:
    """Option Delta — sensitivity of price to underlying price movement."""
    if T <= 0 or sigma <= 0:
        if option_type.upper() in ("CE", "CALL", "C"):
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    if option_type.upper() in ("CE", "CALL", "C"):
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
) -> float:
    """Option Gamma — rate of change of Delta (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> float:
    """Option Theta — time decay (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type.upper() in ("CE", "CALL", "C"):
        th = (
            common
            + q * S * np.exp(-q * T) * norm.cdf(d1)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        th = (
            common
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    return th / 365.0  # Per calendar day


def vega(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
) -> float:
    """Option Vega — sensitivity to volatility (per 1% move, same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def rho(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> float:
    """Option Rho — sensitivity to interest rate (per 1% move)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d2 = _d2(S, K, T, r, sigma, q)

    if option_type.upper() in ("CE", "CALL", "C"):
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def compute_all_greeks(
    S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.2,
    q: float = DIVIDEND_YIELD,
    option_type: str = "CE",
) -> dict:
    """Compute all Greeks in a single pass (reuses d1/d2)."""
    result = {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
    }

    if T <= 0 or sigma <= 0:
        if option_type.upper() in ("CE", "CALL", "C"):
            result["delta"] = 1.0 if S > K else 0.0
        else:
            result["delta"] = -1.0 if S < K else 0.0
        return result

    d1_val = _d1(S, K, T, r, sigma, q)
    d2_val = d1_val - sigma * np.sqrt(T)
    sqrt_t = np.sqrt(T)
    exp_qt = np.exp(-q * T)
    exp_rt = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1_val)

    # Delta
    if option_type.upper() in ("CE", "CALL", "C"):
        result["delta"] = exp_qt * norm.cdf(d1_val)
    else:
        result["delta"] = exp_qt * (norm.cdf(d1_val) - 1)

    # Gamma (same for calls and puts)
    result["gamma"] = exp_qt * pdf_d1 / (S * sigma * sqrt_t)

    # Theta
    common_theta = -(S * exp_qt * pdf_d1 * sigma) / (2 * sqrt_t)
    if option_type.upper() in ("CE", "CALL", "C"):
        result["theta"] = (
            common_theta
            + q * S * exp_qt * norm.cdf(d1_val)
            - r * K * exp_rt * norm.cdf(d2_val)
        ) / 365.0
    else:
        result["theta"] = (
            common_theta
            - q * S * exp_qt * norm.cdf(-d1_val)
            + r * K * exp_rt * norm.cdf(-d2_val)
        ) / 365.0

    # Vega (per 1% move)
    result["vega"] = S * exp_qt * pdf_d1 * sqrt_t / 100.0

    # Rho (per 1% move)
    if option_type.upper() in ("CE", "CALL", "C"):
        result["rho"] = K * T * exp_rt * norm.cdf(d2_val) / 100.0
    else:
        result["rho"] = -K * T * exp_rt * norm.cdf(-d2_val) / 100.0

    return result


# ──────────────────────────────────────────────
# Put-Call Parity
# ──────────────────────────────────────────────

def put_call_parity_deviation(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    q: float = DIVIDEND_YIELD,
) -> float:
    """
    Compute deviation from put-call parity.

    Parity: C - P = S*e^{-qT} - K*e^{-rT}
    Deviation = (C - P) - (S*e^{-qT} - K*e^{-rT})

    Non-zero deviation indicates potential mispricing.
    """
    theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual_diff = call_price - put_price
    return actual_diff - theoretical_diff
