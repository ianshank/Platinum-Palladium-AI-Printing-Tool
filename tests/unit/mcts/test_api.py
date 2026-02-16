"""
Unit tests for MCTS API router.
"""

from __future__ import annotations

import pytest

# Check if FastAPI is available
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)

from ptpd_calibration.api.mcts_router import create_mcts_router


@pytest.fixture
def client():
    """Create a test client with just the MCTS router."""
    app = FastAPI()
    router = create_mcts_router()
    app.include_router(router)
    return TestClient(app)


class TestMCTSStatusEndpoint:
    """Tests for GET /api/mcts/status endpoint."""

    def test_status_returns_200(self, client):
        """Test that status endpoint returns 200."""
        response = client.get("/api/mcts/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self, client):
        """Test that status response has all required fields."""
        response = client.get("/api/mcts/status")
        data = response.json()

        assert "engine_ready" in data
        assert "networks_loaded" in data
        assert "torch_available" in data
        assert "parameter_ranges" in data

    def test_status_parameter_ranges(self, client):
        """Test that parameter ranges are returned correctly."""
        response = client.get("/api/mcts/status")
        data = response.json()

        param_ranges = data["parameter_ranges"]
        assert "metal_ratio" in param_ranges
        assert "coating_weight" in param_ranges
        assert "ferric_oxalate_pct" in param_ranges

        # Check structure of a parameter range
        metal_ratio = param_ranges["metal_ratio"]
        assert "min" in metal_ratio
        assert "max" in metal_ratio
        assert "default" in metal_ratio
        assert "unit" in metal_ratio


class TestMCTSEvaluateEndpoint:
    """Tests for POST /api/mcts/evaluate endpoint."""

    def test_evaluate_with_valid_params(self, client):
        """Test evaluation with valid parameters."""
        params = {
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        }
        response = client.post("/api/mcts/evaluate", json={"parameters": params})
        assert response.status_code == 200

        data = response.json()
        assert "density_curve" in data
        assert "dmin" in data
        assert "dmax" in data
        assert "density_range" in data
        assert "gamma" in data
        assert "quality_score" in data

    def test_evaluate_with_empty_params_uses_defaults(self, client):
        """Test evaluation with empty params uses defaults."""
        response = client.post("/api/mcts/evaluate", json={"parameters": {}})
        assert response.status_code == 200

        data = response.json()
        assert "density_curve" in data
        assert len(data["density_curve"]) > 0

    def test_evaluate_density_curve_is_valid(self, client):
        """Test that density curve values are valid."""
        params = {"metal_ratio": 0.5}
        response = client.post("/api/mcts/evaluate", json={"parameters": params})
        data = response.json()

        curve = data["density_curve"]
        assert isinstance(curve, list)
        assert len(curve) > 0
        # All values should be non-negative
        assert all(val >= 0 for val in curve)

    def test_evaluate_quality_score_in_range(self, client):
        """Test that quality score is in [0, 1]."""
        params = {"metal_ratio": 0.5}
        response = client.post("/api/mcts/evaluate", json={"parameters": params})
        data = response.json()

        quality_score = data["quality_score"]
        assert 0.0 <= quality_score <= 1.0


class TestMCTSSearchEndpoint:
    """Tests for POST /api/mcts/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_result(self, client):
        """Test that search returns a result."""
        request_data = {
            "target_aesthetics": {"warmth": 0.6, "contrast": 0.7},
            "fixed_parameters": {},
            "num_simulations": 100,
        }
        response = client.post("/api/mcts/search", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "search_id" in data
        assert "best_parameters" in data
        assert "predicted_curve" in data
        assert "quality_score" in data
        assert "alternatives" in data
        assert "search_time_seconds" in data
        assert "num_simulations" in data

    @pytest.mark.asyncio
    async def test_search_with_fixed_parameters(self, client):
        """Test search with fixed parameters."""
        request_data = {
            "fixed_parameters": {"metal_ratio": 0.5},
            "num_simulations": 50,
        }
        response = client.post("/api/mcts/search", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Fixed parameter should be in result
        assert data["best_parameters"]["metal_ratio"] == 0.5

    @pytest.mark.asyncio
    async def test_search_respects_num_simulations(self, client):
        """Test that search respects num_simulations parameter."""
        request_data = {"num_simulations": 150}
        response = client.post("/api/mcts/search", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["num_simulations"] == 150


class TestMCTSTrainEndpoint:
    """Tests for POST /api/mcts/train endpoint."""

    def test_train_returns_session_id(self, client):
        """Test that train endpoint returns a session ID."""
        # Skip if torch not available
        status_response = client.get("/api/mcts/status")
        if not status_response.json()["torch_available"]:
            pytest.skip("PyTorch not available")

        request_data = {"num_episodes": 10}
        response = client.post("/api/mcts/train", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert "status" in data
        assert "message" in data

    def test_train_without_torch_returns_503(self, client):
        """Test that train endpoint returns 503 if PyTorch not available."""
        status_response = client.get("/api/mcts/status")
        if status_response.json()["torch_available"]:
            pytest.skip("PyTorch is available")

        request_data = {"num_episodes": 10}
        response = client.post("/api/mcts/train", json=request_data)
        assert response.status_code == 503


class TestMCTSExportEndpoint:
    """Tests for POST /api/mcts/export endpoint."""

    def test_export_json_format(self, client):
        """Test export in JSON format."""
        params = {"metal_ratio": 0.5, "coating_weight": 1.5}
        response = client.post(
            "/api/mcts/export",
            params={"format": "json"},
            json=params,
        )
        assert response.status_code == 200

        data = response.json()
        assert "parameters" in data
        assert "density_curve" in data
        assert "dmin" in data
        assert "dmax" in data

    def test_export_csv_format(self, client):
        """Test export in CSV format."""
        params = {"metal_ratio": 0.5}
        response = client.post(
            "/api/mcts/export",
            params={"format": "csv"},
            json=params,
        )
        assert response.status_code == 200

        data = response.json()
        assert "csv" in data
        assert "input,output" in data["csv"]

    def test_export_recipe_format(self, client):
        """Test export in recipe format."""
        params = {"metal_ratio": 0.5}
        response = client.post(
            "/api/mcts/export",
            params={"format": "recipe"},
            json=params,
        )
        assert response.status_code == 200

        data = response.json()
        assert "title" in data
        assert "parameters" in data
        assert "predicted_results" in data


class TestMCTSFeedbackEndpoint:
    """Tests for POST /api/mcts/feedback endpoint."""

    def test_feedback_success(self, client):
        """Test submitting feedback."""
        request_data = {
            "parameters": {"metal_ratio": 0.5},
            "measured_curve": [0.1, 0.5, 1.0, 1.5, 2.0],
            "quality_rating": 0.8,
        }
        response = client.post("/api/mcts/feedback", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "message" in data


class TestMCTSRecommendationsEndpoint:
    """Tests for GET /api/mcts/recommendations endpoint."""

    def test_recommendations_default(self, client):
        """Test getting recommendations with defaults."""
        response = client.get("/api/mcts/recommendations")
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    def test_recommendations_with_limit(self, client):
        """Test getting recommendations with limit."""
        response = client.get("/api/mcts/recommendations?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data["recommendations"]) <= 2

    def test_recommendation_structure(self, client):
        """Test that recommendations have correct structure."""
        response = client.get("/api/mcts/recommendations")
        data = response.json()

        recommendations = data["recommendations"]
        assert len(recommendations) > 0

        first_rec = recommendations[0]
        assert "parameters" in first_rec
        assert "predicted_quality" in first_rec
        assert "rationale" in first_rec


class TestMCTSErrorHandling:
    """Tests for error handling in MCTS API."""

    def test_invalid_request_returns_422(self, client):
        """Test that invalid request data returns 422."""
        # Invalid quality_rating (out of range)
        request_data = {
            "parameters": {},
            "measured_curve": [1.0],
            "quality_rating": 1.5,  # Invalid: should be [0, 1]
        }
        response = client.post("/api/mcts/feedback", json=request_data)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        """Test that missing required field returns 422."""
        # Missing required 'parameters' field
        response = client.post("/api/mcts/evaluate", json={})
        assert response.status_code == 422
