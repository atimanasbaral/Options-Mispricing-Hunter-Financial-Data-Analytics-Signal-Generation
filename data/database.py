"""
SQLite Database Layer
=======================
Stores raw options chain data, engineered features, and model signals.
Provides efficient batch inserts and query methods.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)


class OptionsDatabase:
    """SQLite database for options data persistence."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    underlying_value REAL,
                    last_price REAL,
                    bid_price REAL,
                    ask_price REAL,
                    bid_qty INTEGER,
                    ask_qty INTEGER,
                    volume INTEGER,
                    open_interest INTEGER,
                    oi_change INTEGER,
                    iv REAL,
                    change REAL,
                    pct_change REAL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    underlying_value REAL,
                    last_price REAL,
                    timestamp TEXT NOT NULL,

                    -- Basic
                    moneyness REAL,
                    log_moneyness REAL,
                    time_to_expiry REAL,
                    sqrt_tte REAL,

                    -- IV
                    implied_volatility REAL,
                    iv_rank REAL,
                    iv_skew REAL,
                    iv_term_structure REAL,

                    -- Greeks
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    rho REAL,
                    gamma_theta_ratio REAL,
                    vega_theta_ratio REAL,

                    -- Parity
                    put_call_parity_dev REAL,
                    pcp_deviation_zscore REAL,

                    -- Market micro
                    volume INTEGER,
                    open_interest INTEGER,
                    volume_oi_ratio REAL,
                    bid_ask_spread REAL,
                    bid_ask_pct REAL,
                    oi_change_pct REAL,
                    volume_zscore REAL,

                    -- Sentiment
                    pcr_oi REAL,
                    pcr_volume REAL,

                    -- Valuation
                    intrinsic_value REAL,
                    extrinsic_value REAL,
                    extrinsic_pct REAL,

                    -- Label (filled after labeling)
                    mispricing_label INTEGER,

                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    underlying_value REAL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    direction TEXT,
                    xgb_prob REAL,
                    lgb_prob REAL,
                    ensemble_prob REAL,
                    model_version TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS model_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    psi_score REAL NOT NULL,
                    drift_status TEXT NOT NULL,
                    model_version TEXT,
                    last_retrain TEXT,
                    feature_psi_breakdown TEXT,
                    checked_at TEXT DEFAULT (datetime('now'))
                );

                -- Indexes for fast lookups
                CREATE INDEX IF NOT EXISTS idx_chain_symbol_ts
                    ON options_chain(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_chain_strike
                    ON options_chain(symbol, expiry_date, strike_price);
                CREATE INDEX IF NOT EXISTS idx_features_symbol_ts
                    ON features(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_features_strike
                    ON features(symbol, expiry_date, strike_price);
                CREATE INDEX IF NOT EXISTS idx_signals_ts
                    ON signals(symbol, timestamp);
            """)
        logger.info(f"Database initialized at {self.db_path}")

    # ── Insert operations ──

    def insert_options_chain(self, df: pd.DataFrame):
        """Batch insert raw options chain data."""
        if df.empty:
            return

        cols = [
            "symbol", "strike_price", "expiry_date", "option_type",
            "underlying_value", "last_price", "bid_price", "ask_price",
            "bid_qty", "ask_qty", "volume", "open_interest",
            "oi_change", "iv", "change", "pct_change", "timestamp",
        ]

        records = self._prepare_records(df, cols)
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._get_connection() as conn:
            conn.executemany(
                f"INSERT INTO options_chain ({col_names}) VALUES ({placeholders})",
                records,
            )
        logger.info(f"Inserted {len(records)} raw chain records")

    def insert_features(self, df: pd.DataFrame):
        """Batch insert engineered features."""
        if df.empty:
            return

        cols = [
            "symbol", "strike_price", "expiry_date", "option_type",
            "underlying_value", "last_price", "timestamp",
            "moneyness", "log_moneyness", "time_to_expiry", "sqrt_tte",
            "implied_volatility", "iv_rank", "iv_skew", "iv_term_structure",
            "delta", "gamma", "theta", "vega", "rho",
            "gamma_theta_ratio", "vega_theta_ratio",
            "put_call_parity_dev", "pcp_deviation_zscore",
            "volume", "open_interest", "volume_oi_ratio",
            "bid_ask_spread", "bid_ask_pct", "oi_change_pct", "volume_zscore",
            "pcr_oi", "pcr_volume",
            "intrinsic_value", "extrinsic_value", "extrinsic_pct",
        ]

        # Only include columns that exist in the DataFrame
        available_cols = [c for c in cols if c in df.columns]
        records = self._prepare_records(df, available_cols)

        placeholders = ", ".join(["?"] * len(available_cols))
        col_names = ", ".join(available_cols)

        with self._get_connection() as conn:
            conn.executemany(
                f"INSERT INTO features ({col_names}) VALUES ({placeholders})",
                records,
            )
        logger.info(f"Inserted {len(records)} feature records")

    def insert_signals(self, signals: List[dict]):
        """Insert model prediction signals."""
        if not signals:
            return

        cols = [
            "symbol", "strike_price", "expiry_date", "option_type",
            "underlying_value", "signal_type", "confidence", "direction",
            "xgb_prob", "lgb_prob", "ensemble_prob",
            "model_version", "timestamp",
        ]

        records = []
        for sig in signals:
            records.append(tuple(sig.get(c, None) for c in cols))

        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._get_connection() as conn:
            conn.executemany(
                f"INSERT INTO signals ({col_names}) VALUES ({placeholders})",
                records,
            )
        logger.info(f"Inserted {len(records)} signal records")

    def insert_model_health(self, health: dict):
        """Insert a model health check record."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO model_health
                   (psi_score, drift_status, model_version, last_retrain, feature_psi_breakdown)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    health.get("psi_score", 0),
                    health.get("drift_status", "unknown"),
                    health.get("model_version", ""),
                    health.get("last_retrain", ""),
                    str(health.get("feature_psi_breakdown", {})),
                ),
            )

    # ── Query operations ──

    def get_latest_chain(
        self, symbol: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        """Get the most recent options chain snapshot."""
        query = "SELECT * FROM options_chain"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_latest_features(
        self, symbol: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        """Get the most recent feature set."""
        query = "SELECT * FROM features"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_all_features(self) -> pd.DataFrame:
        """Get all historical features for training."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM features ORDER BY timestamp", conn)
        return df

    def get_latest_signals(
        self,
        symbol: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get recent mispricing signals."""
        query = "SELECT * FROM signals WHERE confidence >= ?"
        params: list = [min_confidence]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_latest_model_health(self) -> Optional[dict]:
        """Get the most recent model health check."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_health ORDER BY checked_at DESC LIMIT 1"
            ).fetchone()

        if row:
            return dict(row)
        return None

    def get_feature_stats(self, days: int = 30) -> pd.DataFrame:
        """Get feature statistics for the last N days (for PSI reference)."""
        query = """
            SELECT * FROM features
            WHERE created_at >= datetime('now', ?)
            ORDER BY timestamp
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[f"-{days} days"])
        return df

    def get_record_count(self, table: str) -> int:
        """Get total record count for a table."""
        with self._get_connection() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0] if result else 0

    # ── Helpers ──

    def _prepare_records(self, df: pd.DataFrame, cols: list) -> list:
        """Convert DataFrame rows to list of tuples for insertion."""
        records = []
        for _, row in df.iterrows():
            values = []
            for col in cols:
                val = row.get(col, None)
                # Convert numpy/pandas types to Python natives
                if isinstance(val, (pd.Timestamp, datetime)):
                    val = val.isoformat()
                elif hasattr(val, "item"):
                    val = val.item()
                values.append(val)
            records.append(tuple(values))
        return records
