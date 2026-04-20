"""
Options Mispricing Hunter — Main Entry Point
===============================================
CLI interface to:
  - Train models from synthetic/historical data
  - Start the FastAPI server
  - Seed the database with historical data
  - Run a single pipeline cycle

Usage:
    python main.py                  # Start API server (default)
    python main.py --train          # Seed data + train models + start server
    python main.py --seed-only      # Only seed historical data
    python main.py --train-only     # Only train models (assumes data exists)
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import API_HOST, API_PORT, LOG_LEVEL, LOG_FORMAT


def setup_logging():
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def seed_data(days: int = 180):
    """Seed the database with synthetic historical data."""
    from data.database import OptionsDatabase
    from data.pipeline import DataPipeline

    logger = logging.getLogger("main")
    logger.info(f"Seeding {days} days of synthetic historical data...")

    db = OptionsDatabase()
    pipeline = DataPipeline(db=db, use_synthetic=True)
    pipeline.seed_historical_data(days=days, samples_per_day=3)

    total = db.get_record_count("features")
    logger.info(f"Seeding complete. Total feature records: {total}")


def train_models():
    """Train XGBoost + LightGBM ensemble from stored features."""
    from data.database import OptionsDatabase
    from models.dataset import DatasetBuilder
    from models.trainer import ModelTrainer

    logger = logging.getLogger("main")
    logger.info("Starting model training...")

    db = OptionsDatabase()
    total = db.get_record_count("features")

    if total == 0:
        logger.error("No data in database! Run with --seed-only first.")
        return False

    logger.info(f"Training on {total} feature records")

    # Build dataset
    builder = DatasetBuilder(db)
    data = builder.build_dataset()

    # Train both models
    trainer = ModelTrainer()
    results = trainer.train_both(data)

    # Summary
    logger.info("=" * 60)
    logger.info("  TRAINING RESULTS")
    logger.info("=" * 60)
    for model_name in ["xgb", "lgb"]:
        m = results[f"{model_name}_metrics"]
        logger.info(
            f"  {model_name.upper():>4} → "
            f"Acc: {m['accuracy']:.4f} | "
            f"Prec: {m['precision']:.4f} | "
            f"Rec: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | "
            f"AUC: {m['auc_roc']:.4f}"
        )
    logger.info("=" * 60)

    return True


def start_server(host: str = API_HOST, port: int = API_PORT, reload: bool = False):
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    logger = logging.getLogger("main")
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"API docs at http://localhost:{port}/docs")

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Options Mispricing Hunter — Detect mispriced NSE options with ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start API server
  python main.py --train            Full setup: seed + train + serve
  python main.py --seed-only        Only seed historical data
  python main.py --train-only       Only train models
  python main.py --port 9000        Custom port
        """,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Seed data, train models, then start server",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed the database with synthetic data",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models (data must exist)",
    )
    parser.add_argument(
        "--seed-days",
        type=int,
        default=180,
        help="Days of historical data to generate (default: 180)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=API_HOST,
        help=f"Server host (default: {API_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=API_PORT,
        help=f"Server port (default: {API_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("main")

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   OPTIONS MISPRICING HUNTER v1.0         ║")
    logger.info("║   Detecting mispriced NSE options w/ ML  ║")
    logger.info("╚══════════════════════════════════════════╝")

    if args.seed_only:
        seed_data(days=args.seed_days)
        return

    if args.train_only:
        train_models()
        return

    if args.train:
        seed_data(days=args.seed_days)
        success = train_models()
        if not success:
            logger.error("Training failed. Exiting.")
            sys.exit(1)

    # Start API server
    start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
