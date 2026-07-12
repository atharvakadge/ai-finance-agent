"""
API endpoint tests.

Tests the HTTP layer — do endpoints return correct status codes,
response shapes, and handle errors properly?

Uses FastAPI's TestClient which creates a fake HTTP client
that calls your API without starting a real server.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Create a test client.

    We import app here (not at module level) to avoid
    loading the ML models during test collection.
    """
    from app.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test basic server health."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["project"] == "AI Financial Research Agent"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestChatEndpoint:
    """Test the chat endpoint."""

    def test_missing_message(self, client):
        """Empty body should return 422 (validation error)."""
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422

    def test_valid_request_shape(self, client):
        """Valid request should return 200 with correct shape."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "What is a P/E ratio?"},
        )
        # Might be 200 or 500 depending on API key availability
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "model" in data


class TestUploadEndpoint:
    """Test the upload endpoint."""

    def test_non_pdf_rejected(self, client):
        """Non-PDF files should be rejected with 400."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


class TestFinancialToolsEndpoint:
    """Test the financial tools endpoint."""

    def test_calculate_ratios(self, client):
        """Should calculate ratios from provided data."""
        response = client.post(
            "/api/v1/ratios/calculate",
            json={
                "market_price": 1265,
                "eps": 51.45,
                "total_debt": 347530,
                "total_equity": 843200,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 2  # at least PE and DE
        assert "ratios" in data

    def test_empty_data_rejected(self, client):
        """No data should return 400."""
        response = client.post("/api/v1/ratios/calculate", json={})
        assert response.status_code == 400

    def test_single_ratio(self, client):
        """Calculate a specific ratio."""
        response = client.post(
            "/api/v1/ratios/single",
            json={
                "ratio_name": "pe",
                "values": {"market_price": 100, "eps": 10},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == 10.0

    def test_unknown_ratio(self, client):
        """Unknown ratio name should return 400."""
        response = client.post(
            "/api/v1/ratios/single",
            json={"ratio_name": "xyz", "values": {}},
        )
        assert response.status_code == 400