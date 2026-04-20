"""
Data Pipeline Orchestrator
============================
Wires together: NSE Fetch → Feature Engineering → Database Storage.
Supports both one-shot and continuous (background loop) execution.
"""

import time
import logging
import threading
from typing import Optional

import pandas as pd

from data.nse_fetcher import NSEFetcher, generate_synthetic_data
from data.feature_engine import FeatureEngine
from data.database import OptionsDatabase
from config import TRACKED_SYMBOLS

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates the full data flow:
      1. Fetch live options chain from NSE (or synthetic fallback)
      2. Compute 29 engineered features
      3. Store raw + features into SQLite
    """

    def __init__(
        self,
        db: Optional[OptionsDatabase] = None,
        use_synthetic: bool = False,
    ):
        self.fetcher = NSEFetcher()
        self.feature_engine = FeatureEngine()
        self.db = db or OptionsDatabase()
        self.use_synthetic = use_synthetic
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def run_once(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a single fetch → compute → store cycle.

        Parameters
        ----------
        symbol : str, optional
            Specific symbol to fetch. If None, fetches all tracked symbols.

        Returns
        -------
        pd.DataFrame
            The feature-enriched DataFrame.
        """
        logger.info("═" * 60)
        logger.info("Pipeline: Starting data cycle")

        # ── Step 1: Fetch data ──
        if symbol:
            symbols = [symbol]
        else:
            symbols = TRACKED_SYMBOLS

        all_data = []
        for sym in symbols:
            if self.use_synthetic:
                df = generate_synthetic_data(sym)
                logger.info(f"Using synthetic data for {sym}")
            else:
                df = self.fetcher.fetch_option_chain(sym)
                if df is None or df.empty:
                    logger.warning(f"Live fetch failed for {sym}, using synthetic")
                    df = generate_synthetic_data(sym)
            all_data.append(df)

        raw_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        if raw_df.empty:
            logger.error("Pipeline: No data obtained")
            return pd.DataFrame()

        logger.info(f"Pipeline: Fetched {len(raw_df)} raw contracts")

        # ── Step 2: Store raw data ──
        try:
            self.db.insert_options_chain(raw_df)
        except Exception as e:
            logger.error(f"Pipeline: Failed to store raw data — {e}")

        # ── Step 3: Compute features ──
        features_df = self.feature_engine.compute_features(raw_df)
        logger.info(f"Pipeline: Computed features for {len(features_df)} contracts")

        # ── Step 4: Store features ──
        try:
            self.db.insert_features(features_df)
        except Exception as e:
            logger.error(f"Pipeline: Failed to store features — {e}")

        logger.info("Pipeline: Cycle complete")
        logger.info("═" * 60)

        return features_df

    def run_continuous(self, interval_seconds: int = 300):
        """
        Run the pipeline in a background thread, fetching at fixed intervals.

        Parameters
        ----------
        interval_seconds : int
            Seconds between fetch cycles (default: 5 minutes).
        """
        if self._running:
            logger.warning("Pipeline already running in background")
            return

        self._running = True

        def _loop():
            while self._running:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"Pipeline error in continuous mode: {e}")
                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True, name="DataPipeline")
        self._thread.start()
        logger.info(
            f"Pipeline: Continuous mode started (interval={interval_seconds}s)"
        )

    def stop(self):
        """Stop the continuous pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Pipeline: Stopped")

    def seed_historical_data(self, days: int = 180, samples_per_day: int = 5):
        """
        Seed the database with synthetic historical data for model training.

        Parameters
        ----------
        days : int
            Number of days of historical data to generate.
        samples_per_day : int
            Snapshots per trading day.
        """
        from data.nse_fetcher import generate_historical_synthetic_data

        logger.info(f"Seeding {days} days of historical data...")

        for symbol in TRACKED_SYMBOLS:
            hist_df = generate_historical_synthetic_data(
                symbol=symbol, days=days, samples_per_day=samples_per_day
            )

            # Compute features in batches (by timestamp to avoid memory issues)
            timestamps = hist_df["timestamp"].unique()
            batch_size = 50

            for i in range(0, len(timestamps), batch_size):
                batch_ts = timestamps[i : i + batch_size]
                batch_df = hist_df[hist_df["timestamp"].isin(batch_ts)]

                # Store raw
                try:
                    self.db.insert_options_chain(batch_df)
                except Exception as e:
                    logger.error(f"Failed to insert raw batch: {e}")
                    continue

                # Compute and store features
                try:
                    features_df = self.feature_engine.compute_features(batch_df)
                    self.db.insert_features(features_df)
                except Exception as e:
                    logger.error(f"Failed to process feature batch: {e}")
                    continue

            logger.info(f"Seeded {len(hist_df)} records for {symbol}")

        total = self.db.get_record_count("features")
        logger.info(f"Historical seeding complete. Total feature records: {total}")
