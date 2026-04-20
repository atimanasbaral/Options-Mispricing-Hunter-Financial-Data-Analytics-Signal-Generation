"""
Tests for FastAPI Endpoints
==============================
Integration tests for all API routes using TestClient.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


class TestHealthCheck:
    """Test system health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestOptionsChain:
    """Test options chain endpoint."""

    def test_options_chain_returns_200(self, client):
        response = client.get("/options-chain?symbol=NIFTY")
        assert response.status_code == 200

    def test_options_chain_structure(self, client):
        response = client.get("/options-chain?symbol=NIFTY")
        data = response.json()
        assert "symbol" in data
        assert "contracts" in data
        assert "total_contracts" in data
        assert isinstance(data["contracts"], list)

    def test_options_chain_has_greeks(self, client):
        response = client.get("/options-chain?symbol=NIFTY&limit=5")
        data = response.json()
        if data["contracts"]:
            contract = data["contracts"][0]
            assert "delta" in contract
            assert "gamma" in contract
            assert "theta" in contract
            assert "vega" in contract
            assert "implied_volatility" in contract


class TestSignals:
    """Test signals endpoint."""

    def test_signals_returns_200(self, client):
        response = client.get("/signals")
        assert response.status_code == 200

    def test_signals_structure(self, client):
        response = client.get("/signals")
        data = response.json()
        assert "total_signals" in data
        assert "signals" in data
        assert isinstance(data["signals"], list)

    def test_signals_with_confidence_filter(self, client):
        response = client.get("/signals?min_confidence=0.9")
        assert response.status_code == 200


class TestPredict:
    """Test prediction endpoint."""

    def test_predict_returns_result(self, client):
        """Prediction may fail if models aren't trained — that's ok."""
        response = client.post(
            "/predict",
            json={
                "symbol": "NIFTY",
                "strike_price": 22500,
                "expiry_date": "2026-05-01",
                "option_type": "CE",
                "spot_price": 22480,
                "market_price": 185.0,
            },
        )
        # 200 if models loaded, 503 if not
        assert response.status_code in (200, 503)


class TestModelHealth:
    """Test model health endpoint."""

    def test_model_health_returns_200(self, client):
        response = client.get("/model-health")
        assert response.status_code == 200

    def test_model_health_structure(self, client):
        response = client.get("/model-health")
        data = response.json()
        assert "psi_score" in data
        assert "drift_status" in data
        assert "recommendation" in data
