"""
FastAPI Application Factory
==============================
Creates and configures the FastAPI application with middleware,
startup/shutdown events, and dependency injection.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_TITLE, API_VERSION, API_DESCRIPTION, LOG_LEVEL, LOG_FORMAT
from data.database import OptionsDatabase
from data.pipeline import DataPipeline
from models.ensemble import EnsemblePredictor
from monitoring.retrainer import DriftMonitor
from api.routes import router, init_dependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Startup: initialize DB, load models, start pipeline.
    Shutdown: stop background tasks.
    """
    # ── STARTUP ──
    logger.info("=" * 60)
    logger.info(f"  {API_TITLE} v{API_VERSION}")
    logger.info("=" * 60)

    # Initialize database
    db = OptionsDatabase()
    logger.info("Database initialized")

    # Initialize data pipeline (use synthetic for now)
    pipeline = DataPipeline(db=db, use_synthetic=True)

    # Initialize ensemble predictor
    ensemble = EnsemblePredictor()
    models_loaded = ensemble.load_models()

    if models_loaded:
        logger.info("ML models loaded successfully")
    else:
        logger.warning(
            "Models not found. Run 'python main.py --train' to train first."
        )

    # Initialize drift monitor
    drift_monitor = DriftMonitor(db=db, ensemble=ensemble)

    # Inject dependencies into routes
    init_dependencies(db, pipeline, ensemble, drift_monitor)

    # Seed initial data if database is empty
    if db.get_record_count("features") == 0:
        logger.info("Database empty — seeding with initial synthetic data...")
        pipeline.run_once()

    logger.info("Startup complete — API ready")

    yield  # ← App is running

    # ── SHUTDOWN ──
    pipeline.stop()
    drift_monitor.stop_monitoring()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


# Module-level app instance for uvicorn
app = create_app()
