"""
NSE Options Chain Data Fetcher
================================
Fetches live options chain data from NSE India's public API.
Includes session management, rate limiting, and a synthetic data
fallback for development/testing when NSE is unreachable.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import (
    NSE_BASE_URL,
    NSE_OPTION_CHAIN_URL,
    NSE_HEADERS,
    NSE_REQUEST_INTERVAL,
    NSE_REQUEST_TIMEOUT,
    NSE_MAX_RETRIES,
    TRACKED_SYMBOLS,
)

logger = logging.getLogger(__name__)


class NSEFetcher:
    """
    Fetches NSE options chain data with proper session/cookie management.

    NSE blocks requests without valid cookies. The strategy is:
    1. Hit the main NSE page to obtain session cookies.
    2. Use those cookies to call the options chain API.
    3. Rate-limit to avoid IP bans.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._last_request_time: float = 0.0
        self._cookies_valid = False

    def _refresh_cookies(self) -> bool:
        """Visit NSE homepage to obtain session cookies."""
        try:
            response = self.session.get(
                NSE_BASE_URL,
                timeout=NSE_REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                self._cookies_valid = True
                logger.info("NSE session cookies refreshed successfully")
                return True
            else:
                logger.warning(f"NSE homepage returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to refresh NSE cookies: {e}")
            return False

    def _rate_limit(self):
        """Enforce minimum interval between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < NSE_REQUEST_INTERVAL:
            sleep_time = NSE_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_option_chain(self, symbol: str = "NIFTY") -> Optional[pd.DataFrame]:
        """
        Fetch the options chain for a given symbol.

        Parameters
        ----------
        symbol : str
            Index symbol (e.g., 'NIFTY', 'BANKNIFTY')

        Returns
        -------
        pd.DataFrame or None
            Flattened options chain data, or None if fetch fails.
        """
        for attempt in range(1, NSE_MAX_RETRIES + 1):
            try:
                # Ensure we have cookies
                if not self._cookies_valid:
                    if not self._refresh_cookies():
                        logger.warning(f"Attempt {attempt}: Cookie refresh failed")
                        time.sleep(2 ** attempt)
                        continue

                self._rate_limit()

                response = self.session.get(
                    NSE_OPTION_CHAIN_URL,
                    params={"symbol": symbol},
                    timeout=NSE_REQUEST_TIMEOUT,
                )

                if response.status_code == 401:
                    logger.info("Session expired, refreshing cookies...")
                    self._cookies_valid = False
                    continue

                if response.status_code != 200:
                    logger.warning(
                        f"Attempt {attempt}: API returned status {response.status_code}"
                    )
                    time.sleep(2 ** attempt)
                    continue

                data = response.json()
                df = self._parse_option_chain(data, symbol)
                logger.info(
                    f"Fetched {len(df)} option contracts for {symbol}"
                )
                return df

            except requests.exceptions.JSONDecodeError:
                logger.error(f"Attempt {attempt}: Invalid JSON response")
                self._cookies_valid = False
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt}: Request failed — {e}")
                self._cookies_valid = False
            except Exception as e:
                logger.error(f"Attempt {attempt}: Unexpected error — {e}")

            time.sleep(2 ** attempt)

        logger.error(
            f"All {NSE_MAX_RETRIES} attempts failed for {symbol}. "
            "Falling back to synthetic data."
        )
        return None

    def _parse_option_chain(self, raw_data: dict, symbol: str) -> pd.DataFrame:
        """Parse NSE's nested JSON into a flat DataFrame."""
        records = raw_data.get("records", {})
        data_rows = records.get("data", [])
        underlying_value = records.get("underlyingValue", 0)
        timestamp = records.get("timestamp", datetime.now().isoformat())

        rows = []
        for row in data_rows:
            strike_price = row.get("strikePrice", 0)
            expiry_date = row.get("expiryDate", "")

            # Parse CE (Call) data
            ce = row.get("CE", {})
            pe = row.get("PE", {})

            base = {
                "symbol": symbol,
                "strike_price": strike_price,
                "expiry_date": expiry_date,
                "underlying_value": underlying_value,
                "timestamp": timestamp,
            }

            if ce:
                rows.append({
                    **base,
                    "option_type": "CE",
                    "last_price": ce.get("lastPrice", 0),
                    "bid_price": ce.get("bidprice", 0),
                    "ask_price": ce.get("askPrice", 0),
                    "bid_qty": ce.get("bidQty", 0),
                    "ask_qty": ce.get("askQty", 0),
                    "volume": ce.get("totalTradedVolume", 0),
                    "open_interest": ce.get("openInterest", 0),
                    "oi_change": ce.get("changeinOpenInterest", 0),
                    "iv": ce.get("impliedVolatility", 0),
                    "change": ce.get("change", 0),
                    "pct_change": ce.get("pchangeinOpenInterest", 0),
                    "underlying_value": ce.get("underlyingValue", underlying_value),
                })

            if pe:
                rows.append({
                    **base,
                    "option_type": "PE",
                    "last_price": pe.get("lastPrice", 0),
                    "bid_price": pe.get("bidprice", 0),
                    "ask_price": pe.get("askPrice", 0),
                    "bid_qty": pe.get("bidQty", 0),
                    "ask_qty": pe.get("askQty", 0),
                    "volume": pe.get("totalTradedVolume", 0),
                    "open_interest": pe.get("openInterest", 0),
                    "oi_change": pe.get("changeinOpenInterest", 0),
                    "iv": pe.get("impliedVolatility", 0),
                    "change": pe.get("change", 0),
                    "pct_change": pe.get("pchangeinOpenInterest", 0),
                    "underlying_value": pe.get("underlyingValue", underlying_value),
                })

        df = pd.DataFrame(rows)

        # Parse expiry date
        if not df.empty and "expiry_date" in df.columns:
            df["expiry_date"] = pd.to_datetime(
                df["expiry_date"], format="%d-%b-%Y", errors="coerce"
            )

        return df

    def fetch_all_symbols(self) -> pd.DataFrame:
        """Fetch option chains for all tracked symbols."""
        all_data = []
        for symbol in TRACKED_SYMBOLS:
            df = self.fetch_option_chain(symbol)
            if df is not None and not df.empty:
                all_data.append(df)
            else:
                logger.warning(
                    f"Using synthetic data for {symbol}"
                )
                all_data.append(generate_synthetic_data(symbol))

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Synthetic data generator (fallback / dev mode)
# ──────────────────────────────────────────────

def generate_synthetic_data(
    symbol: str = "NIFTY",
    spot_price: float = 22500.0,
    num_strikes: int = 25,
    num_expiries: int = 4,
) -> pd.DataFrame:
    """
    Generate realistic synthetic options chain data for development.

    Creates a grid of strikes around the spot price with realistic
    IV smiles, bid-ask spreads, volumes, and OI patterns.
    """
    np.random.seed(42)

    now = datetime.now()
    strike_step = 50 if "NIFTY" in symbol and "BANK" not in symbol else 100
    if "BANK" in symbol:
        spot_price = 48500.0
        strike_step = 100

    # Generate strike range centered on spot
    atm_strike = round(spot_price / strike_step) * strike_step
    strikes = np.arange(
        atm_strike - num_strikes // 2 * strike_step,
        atm_strike + (num_strikes // 2 + 1) * strike_step,
        strike_step,
    )

    # Generate expiry dates (weekly + monthly)
    expiries = []
    for i in range(num_expiries):
        exp = now + timedelta(days=7 * (i + 1))
        # Shift to Thursday (NSE options expire on Thursday)
        while exp.weekday() != 3:
            exp += timedelta(days=1)
        expiries.append(exp)

    rows = []
    for expiry in expiries:
        T = max((expiry - now).total_seconds() / (365.25 * 24 * 3600), 1e-6)

        for strike in strikes:
            moneyness = spot_price / strike

            # Realistic IV smile — higher for OTM options
            base_iv = 0.15 + 0.05 * np.random.randn()
            iv_skew = 0.03 * (1 - moneyness) ** 2
            call_iv = max(0.05, base_iv + iv_skew + 0.01 * np.random.randn())
            put_iv = max(0.05, base_iv + iv_skew + 0.015 + 0.01 * np.random.randn())

            # Compute BS prices
            from data.black_scholes import bs_call_price, bs_put_price

            call_price = bs_call_price(spot_price, strike, T, sigma=call_iv)
            put_price = bs_put_price(spot_price, strike, T, sigma=put_iv)

            # Market microstructure noise
            call_ltp = max(0.5, call_price * (1 + 0.02 * np.random.randn()))
            put_ltp = max(0.5, put_price * (1 + 0.02 * np.random.randn()))

            # Bid-ask spread (wider for illiquid, OTM options)
            otm_factor = abs(1 - moneyness) * 5 + 1
            call_spread = max(0.05, call_ltp * 0.005 * otm_factor)
            put_spread = max(0.05, put_ltp * 0.005 * otm_factor)

            # Volume & OI — higher near ATM
            atm_proximity = np.exp(-5 * (moneyness - 1) ** 2)
            base_volume = int(10000 * atm_proximity * np.random.exponential(1))
            base_oi = int(50000 * atm_proximity * np.random.exponential(1))

            base_record = {
                "symbol": symbol,
                "strike_price": float(strike),
                "expiry_date": expiry,
                "underlying_value": spot_price,
                "timestamp": now.isoformat(),
            }

            # Call record
            rows.append({
                **base_record,
                "option_type": "CE",
                "last_price": round(call_ltp, 2),
                "bid_price": round(call_ltp - call_spread / 2, 2),
                "ask_price": round(call_ltp + call_spread / 2, 2),
                "bid_qty": int(np.random.randint(50, 5000)),
                "ask_qty": int(np.random.randint(50, 5000)),
                "volume": int(base_volume * (0.8 + 0.4 * np.random.rand())),
                "open_interest": int(base_oi * (0.7 + 0.6 * np.random.rand())),
                "oi_change": int(np.random.randint(-2000, 3000)),
                "iv": round(call_iv * 100, 2),
                "change": round(np.random.randn() * call_ltp * 0.05, 2),
                "pct_change": round(np.random.randn() * 5, 2),
            })

            # Put record
            rows.append({
                **base_record,
                "option_type": "PE",
                "last_price": round(put_ltp, 2),
                "bid_price": round(put_ltp - put_spread / 2, 2),
                "ask_price": round(put_ltp + put_spread / 2, 2),
                "bid_qty": int(np.random.randint(50, 5000)),
                "ask_qty": int(np.random.randint(50, 5000)),
                "volume": int(base_volume * (0.6 + 0.8 * np.random.rand())),
                "open_interest": int(base_oi * (0.5 + 1.0 * np.random.rand())),
                "oi_change": int(np.random.randint(-2000, 3000)),
                "iv": round(put_iv * 100, 2),
                "change": round(np.random.randn() * put_ltp * 0.05, 2),
                "pct_change": round(np.random.randn() * 5, 2),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} synthetic contracts for {symbol}")
    return df


def generate_historical_synthetic_data(
    symbol: str = "NIFTY",
    days: int = 180,
    samples_per_day: int = 5,
) -> pd.DataFrame:
    """
    Generate 6 months of synthetic historical data for model training.

    Creates multiple snapshots per day with evolving spot prices
    and realistic time-series dynamics.
    """
    np.random.seed(123)
    all_frames = []

    base_spot = 22500.0 if "BANK" not in symbol else 48500.0
    current_spot = base_spot

    start_date = datetime.now() - timedelta(days=days)

    for day in range(days):
        date = start_date + timedelta(days=day)

        # Skip weekends
        if date.weekday() >= 5:
            continue

        # Random walk for spot price (GBM-like)
        daily_return = np.random.normal(0.0002, 0.012)
        current_spot *= np.exp(daily_return)

        for sample in range(samples_per_day):
            # Intraday micro-movements
            intraday_shift = current_spot * np.random.normal(0, 0.002)
            snapshot_spot = current_spot + intraday_shift

            df = generate_synthetic_data(
                symbol=symbol,
                spot_price=round(snapshot_spot, 2),
                num_strikes=15,
                num_expiries=3,
            )

            # Override timestamp for this historical point
            hour = 9 + int(sample * 6 / samples_per_day)
            minute = np.random.randint(0, 60)
            snapshot_time = date.replace(hour=hour, minute=minute, second=0)
            df["timestamp"] = snapshot_time.isoformat()

            all_frames.append(df)

    result = pd.concat(all_frames, ignore_index=True)
    logger.info(
        f"Generated {len(result)} historical records over {days} days for {symbol}"
    )
    return result
