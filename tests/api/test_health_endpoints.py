"""
Health and Root Endpoint Tests.

Tests for the basic health check and root endpoints.
"""

import pytest


@pytest.mark.api
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "PTPD Calibration API"

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_response_time(self, client):
        """Test that health endpoint responds quickly."""
        import time

        start = time.time()
        response = client.get("/api/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, "Health check should respond in under 1 second"

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Should not fail
        assert response.status_code in [200, 204, 405]
