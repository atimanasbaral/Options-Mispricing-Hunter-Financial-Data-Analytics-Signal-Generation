"""
Feature Engineering Engine
============================
Computes 29 quantitative features for each options contract,
spanning Greeks, IV surface, market microstructure, and sentiment.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data.black_scholes import (
    implied_volatility,
    compute_all_greeks,
    put_call_parity_deviation,
)
from config import RISK_FREE_RATE, DIVIDEND_YIELD, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Transforms raw options chain data into ML-ready feature vectors.

    Input:  DataFrame with columns [symbol, strike_price, expiry_date,
            underlying_value, option_type, last_price, bid_price, ask_price,
            volume, open_interest, oi_change, iv, timestamp]

    Output: Same DataFrame enriched with 29 computed features.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE, dividend_yield: float = DIVIDEND_YIELD):
        self.r = risk_free_rate
        self.q = dividend_yield

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Master feature computation pipeline."""
        if df.empty:
            logger.warning("Empty DataFrame — skipping feature computation")
            return df

        df = df.copy()

        # Ensure proper types
        df = self._prepare_dtypes(df)

        # Compute time to expiry
        df = self._compute_time_to_expiry(df)

        # ── Basic features ──
        df = self._compute_moneyness(df)

        # ── Implied Volatility features ──
        df = self._compute_iv_features(df)

        # ── Greeks ──
        df = self._compute_greeks(df)

        # ── Greek ratios ──
        df = self._compute_greek_ratios(df)

        # ── Put-Call Parity ──
        df = self._compute_pcp_features(df)

        # ── Market Microstructure ──
        df = self._compute_market_micro(df)

        # ── Sentiment (PCR) ──
        df = self._compute_pcr(df)

        # ── Valuation ──
        df = self._compute_valuation(df)

        # Fill remaining NaN with 0
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        logger.info(f"Computed {len(FEATURE_COLUMNS)} features for {len(df)} contracts")
        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _prepare_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types for all columns."""
        numeric_cols = [
            "strike_price", "underlying_value", "last_price",
            "bid_price", "ask_price", "volume", "open_interest",
            "oi_change", "iv",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "expiry_date" in df.columns:
            df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    def _compute_time_to_expiry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time to expiry in years."""
        now = pd.Timestamp.now()
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            now = df["timestamp"].iloc[0]
            if isinstance(now, str):
                now = pd.Timestamp(now)

        df["time_to_expiry"] = (
            (df["expiry_date"] - now).dt.total_seconds() / (365.25 * 24 * 3600)
        )
        df["time_to_expiry"] = df["time_to_expiry"].clip(lower=1e-6)
        df["sqrt_tte"] = np.sqrt(df["time_to_expiry"])

        return df

    def _compute_moneyness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moneyness = S/K and log(S/K)."""
        df["moneyness"] = df["underlying_value"] / df["strike_price"].replace(0, np.nan)
        df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))
        df["log_moneyness"] = df["log_moneyness"].fillna(0)
        return df

    def _compute_iv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """IV, IV rank, IV skew, IV term structure."""
        # Use exchange-provided IV or compute from BS
        if "iv" in df.columns:
            # NSE provides IV as percentage
            df["implied_volatility"] = df["iv"] / 100.0
        else:
            df["implied_volatility"] = 0.0

        # Re-compute IV where missing or zero
        mask = (df["implied_volatility"] <= 0) & (df["last_price"] > 0)
        if mask.any():
            for idx in df[mask].index:
                row = df.loc[idx]
                iv = implied_volatility(
                    market_price=row["last_price"],
                    S=row["underlying_value"],
                    K=row["strike_price"],
                    T=row["time_to_expiry"],
                    r=self.r,
                    q=self.q,
                    option_type=row["option_type"],
                )
                df.at[idx, "implied_volatility"] = iv if not np.isnan(iv) else 0.0

        # IV Rank — percentile within same symbol/expiry group
        df["iv_rank"] = df.groupby(["symbol", "expiry_date"])["implied_volatility"].transform(
            lambda x: x.rank(pct=True) * 100
        )

        # IV Skew — difference between ATM and OTM IV per expiry
        df["iv_skew"] = 0.0
        for (sym, exp), group in df.groupby(["symbol", "expiry_date"]):
            if len(group) < 3:
                continue
            spot = group["underlying_value"].iloc[0]
            atm_mask = (group["moneyness"] - 1.0).abs() < 0.02
            atm_iv = group.loc[atm_mask, "implied_volatility"].mean()
            if np.isnan(atm_iv) or atm_iv == 0:
                atm_iv = group["implied_volatility"].median()
            df.loc[group.index, "iv_skew"] = group["implied_volatility"] - atm_iv

        # IV Term Structure — near-term IV / far-term IV
        df["iv_term_structure"] = 0.0
        for sym, sym_group in df.groupby("symbol"):
            expiries = sorted(sym_group["expiry_date"].unique())
            if len(expiries) >= 2:
                near_mask = sym_group["expiry_date"] == expiries[0]
                far_mask = sym_group["expiry_date"] == expiries[-1]
                near_iv = sym_group.loc[near_mask, "implied_volatility"].mean()
                far_iv = sym_group.loc[far_mask, "implied_volatility"].mean()
                if far_iv > 0:
                    df.loc[sym_group.index, "iv_term_structure"] = near_iv / far_iv

        return df

    def _compute_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Delta, Gamma, Theta, Vega, Rho for each contract."""
        greeks_data = {"delta": [], "gamma": [], "theta": [], "vega": [], "rho": []}

        for _, row in df.iterrows():
            sigma = row.get("implied_volatility", 0.2)
            if sigma <= 0:
                sigma = 0.2  # Fallback

            greeks = compute_all_greeks(
                S=row["underlying_value"],
                K=row["strike_price"],
                T=row["time_to_expiry"],
                r=self.r,
                sigma=sigma,
                q=self.q,
                option_type=row["option_type"],
            )
            for key in greeks_data:
                greeks_data[key].append(greeks[key])

        for key, values in greeks_data.items():
            df[key] = values

        return df

    def _compute_greek_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gamma/Theta and Vega/Theta ratios."""
        abs_theta = df["theta"].abs().replace(0, np.nan)
        df["gamma_theta_ratio"] = (df["gamma"] / abs_theta).fillna(0).clip(-100, 100)
        df["vega_theta_ratio"] = (df["vega"] / abs_theta).fillna(0).clip(-100, 100)
        return df

    def _compute_pcp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Put-call parity deviation and its z-score."""
        df["put_call_parity_dev"] = 0.0
        df["pcp_deviation_zscore"] = 0.0

        for (sym, exp, strike), group in df.groupby(
            ["symbol", "expiry_date", "strike_price"]
        ):
            ce_mask = group["option_type"] == "CE"
            pe_mask = group["option_type"] == "PE"

            if ce_mask.any() and pe_mask.any():
                call_price = group.loc[ce_mask, "last_price"].iloc[0]
                put_price = group.loc[pe_mask, "last_price"].iloc[0]
                S = group["underlying_value"].iloc[0]
                T = group["time_to_expiry"].iloc[0]

                dev = put_call_parity_deviation(
                    call_price, put_price, S, strike, T, self.r, self.q
                )
                df.loc[group.index, "put_call_parity_dev"] = dev

        # Z-score of PCP deviation within each symbol
        for sym, sym_group in df.groupby("symbol"):
            mean_dev = sym_group["put_call_parity_dev"].mean()
            std_dev = sym_group["put_call_parity_dev"].std()
            if std_dev > 0:
                df.loc[sym_group.index, "pcp_deviation_zscore"] = (
                    (sym_group["put_call_parity_dev"] - mean_dev) / std_dev
                )

        return df

    def _compute_market_micro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features."""
        # Volume/OI ratio
        df["volume_oi_ratio"] = (
            df["volume"] / df["open_interest"].replace(0, np.nan)
        ).fillna(0).clip(0, 100)

        # Bid-ask spread
        df["bid_ask_spread"] = (df["ask_price"] - df["bid_price"]).clip(lower=0)

        # Bid-ask spread as percentage of mid price
        mid_price = (df["bid_price"] + df["ask_price"]) / 2
        df["bid_ask_pct"] = (
            df["bid_ask_spread"] / mid_price.replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        # OI change percentage
        df["oi_change_pct"] = (
            df["oi_change"] / df["open_interest"].replace(0, np.nan)
        ).fillna(0).clip(-1, 10)

        # Volume z-score within symbol/expiry
        df["volume_zscore"] = df.groupby(["symbol", "expiry_date"])["volume"].transform(
            lambda x: ((x - x.mean()) / x.std()).fillna(0)
        )

        return df

    def _compute_pcr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Put/Call ratios by OI and volume at each strike."""
        df["pcr_oi"] = 0.0
        df["pcr_volume"] = 0.0

        for (sym, exp, strike), group in df.groupby(
            ["symbol", "expiry_date", "strike_price"]
        ):
            ce_mask = group["option_type"] == "CE"
            pe_mask = group["option_type"] == "PE"

            if ce_mask.any() and pe_mask.any():
                ce_oi = group.loc[ce_mask, "open_interest"].iloc[0]
                pe_oi = group.loc[pe_mask, "open_interest"].iloc[0]
                ce_vol = group.loc[ce_mask, "volume"].iloc[0]
                pe_vol = group.loc[pe_mask, "volume"].iloc[0]

                pcr_oi = pe_oi / ce_oi if ce_oi > 0 else 0
                pcr_vol = pe_vol / ce_vol if ce_vol > 0 else 0

                df.loc[group.index, "pcr_oi"] = pcr_oi
                df.loc[group.index, "pcr_volume"] = pcr_vol

        return df

    def _compute_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intrinsic value, extrinsic value, and extrinsic percentage."""
        S = df["underlying_value"]
        K = df["strike_price"]
        price = df["last_price"]

        # Intrinsic value
        call_intrinsic = (S - K).clip(lower=0)
        put_intrinsic = (K - S).clip(lower=0)
        df["intrinsic_value"] = np.where(
            df["option_type"] == "CE", call_intrinsic, put_intrinsic
        )

        # Extrinsic value (time value)
        df["extrinsic_value"] = (price - df["intrinsic_value"]).clip(lower=0)

        # Extrinsic as percentage of market price
        df["extrinsic_pct"] = (
            df["extrinsic_value"] / price.replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        return df
