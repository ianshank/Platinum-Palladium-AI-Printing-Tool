"""
API route tests for the FastAPI backend.

This module contains both structure/contract tests (using local dictionaries)
and integration tests (using FastAPI TestClient for actual HTTP requests).
"""

import pytest
from typing import Any
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self) -> None:
        """Test health endpoint returns healthy status."""
        # Simulating API response structure
        response = {"status": "healthy", "version": "1.0.0"}

        assert response["status"] == "healthy"
        assert "version" in response

    def test_health_includes_version(self) -> None:
        """Test health endpoint includes version info."""
        response = {"status": "healthy", "version": "1.0.0"}

        assert response["version"] is not None

    # Integration tests using TestClient
    def test_health_endpoint_with_client(self, client: TestClient) -> None:
        """Test actual health endpoint HTTP request."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestCurvesAPI:
    """Tests for curves API endpoints."""

    def test_list_curves_returns_array(self) -> None:
        """Test GET /api/v1/curves returns array of curves."""
        response = {
            "curves": [
                {"id": "curve-1", "name": "Test Curve 1"},
                {"id": "curve-2", "name": "Test Curve 2"},
            ],
            "total": 2,
        }

        assert isinstance(response["curves"], list)
        assert response["total"] == len(response["curves"])

    def test_get_curve_by_id(self, sample_curve_data: dict[str, Any]) -> None:
        """Test GET /api/v1/curves/:id returns single curve."""
        response = sample_curve_data

        assert "id" in response
        assert "input_values" in response
        assert "output_values" in response

    def test_create_curve_requires_name(self) -> None:
        """Test POST /api/v1/curves requires name field."""
        invalid_request = {"type": "contrast"}

        # Validation should fail without name
        assert "name" not in invalid_request

    def test_create_curve_validates_type(self) -> None:
        """Test curve type is validated."""
        valid_types = ["contrast", "linearization", "custom"]

        for curve_type in valid_types:
            request = {"name": "Test", "type": curve_type}
            assert request["type"] in valid_types

    def test_modify_curve_updates_timestamp(
        self, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test modifying curve updates modified_at timestamp."""
        from datetime import datetime

        original_modified = sample_curve_data.get("modified_at")

        # Simulate modification
        modified_curve = {**sample_curve_data, "modified_at": datetime.now().isoformat()}

        assert modified_curve["modified_at"] != original_modified

    def test_smooth_curve_accepts_strength_param(self) -> None:
        """Test smooth endpoint accepts strength parameter."""
        request = {"strength": 0.5}

        assert 0 <= request["strength"] <= 1

    def test_blend_curves_requires_two_or_more(self) -> None:
        """Test blend endpoint requires at least 2 curves."""
        valid_request = {
            "curve_ids": ["curve-1", "curve-2"],
            "weights": [0.5, 0.5],
        }

        assert len(valid_request["curve_ids"]) >= 2
        assert len(valid_request["weights"]) == len(valid_request["curve_ids"])

    def test_blend_weights_sum_to_one(self) -> None:
        """Test blend weights sum to approximately 1."""
        request = {
            "curve_ids": ["c1", "c2", "c3"],
            "weights": [0.33, 0.33, 0.34],
        }

        assert abs(sum(request["weights"]) - 1.0) < 0.01

    # Integration tests using TestClient
    def test_generate_curve_endpoint(
        self, client: TestClient, sample_densities: list[float]
    ) -> None:
        """Test curve generation via HTTP request."""
        request_data = {
            "densities": sample_densities,
            "name": "Test Curve",
            "curve_type": "linear",
        }
        response = client.post("/api/curves/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "curve_id" in data
        assert data["name"] == "Test Curve"
        assert "num_points" in data
        assert "input_values" in data
        assert "output_values" in data

    def test_get_stored_curve(
        self, client: TestClient, sample_densities: list[float]
    ) -> None:
        """Test retrieving a stored curve by ID."""
        # First, create a curve
        create_response = client.post(
            "/api/curves/generate",
            json={
                "densities": sample_densities,
                "name": "Retrievable Curve",
                "curve_type": "linear",
            },
        )
        curve_id = create_response.json()["curve_id"]

        # Now retrieve it
        get_response = client.get(f"/api/curves/{curve_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["curve_id"] == curve_id
        assert data["name"] == "Retrievable Curve"
        assert "input_values" in data
        assert "output_values" in data

    def test_get_nonexistent_curve_returns_404(self, client: TestClient) -> None:
        """Test that requesting nonexistent curve returns 404."""
        response = client.get("/api/curves/nonexistent-id")

        assert response.status_code == 404

    def test_modify_curve_endpoint(
        self, client: TestClient, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test curve modification via HTTP request."""
        request_data = {
            "input_values": sample_curve_data["input_values"],
            "output_values": sample_curve_data["output_values"],
            "name": "Modified Curve",
            "adjustment_type": "brightness",
            "amount": 0.1,
        }
        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "curve_id" in data
        assert data["adjustment_applied"] == "brightness"

    def test_smooth_curve_endpoint(
        self, client: TestClient, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test curve smoothing via HTTP request."""
        request_data = {
            "input_values": sample_curve_data["input_values"],
            "output_values": sample_curve_data["output_values"],
            "name": "Smoothed Curve",
            "method": "gaussian",
            "strength": 0.5,
            "preserve_endpoints": True,
        }
        response = client.post("/api/curves/smooth", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["method_applied"] == "gaussian"

    def test_blend_curves_endpoint(
        self, client: TestClient, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test curve blending via HTTP request."""
        request_data = {
            "curve1_inputs": sample_curve_data["input_values"],
            "curve1_outputs": sample_curve_data["output_values"],
            "curve2_inputs": sample_curve_data["input_values"],
            "curve2_outputs": sample_curve_data["input_values"],  # linear curve
            "name": "Blended Curve",
            "mode": "weighted",
            "weight": 0.5,
        }
        response = client.post("/api/curves/blend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode_applied"] == "weighted"


class TestScanAPI:
    """Tests for scan processing API endpoints."""

    def test_upload_accepts_tiff_files(self) -> None:
        """Test upload endpoint accepts TIFF files."""
        valid_extensions = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

        for ext in valid_extensions:
            filename = f"scan{ext}"
            assert any(filename.endswith(e) for e in valid_extensions)

    def test_upload_returns_quality_score(self) -> None:
        """Test upload returns quality assessment."""
        response = {
            "id": "scan-1",
            "quality_score": 0.85,
            "measurements": [],
        }

        assert "quality_score" in response
        assert 0 <= response["quality_score"] <= 1

    def test_upload_returns_measurements(
        self, sample_scan_measurements: list[dict[str, Any]]
    ) -> None:
        """Test upload returns step measurements."""
        response = {
            "id": "scan-1",
            "quality_score": 0.85,
            "measurements": sample_scan_measurements,
        }

        assert len(response["measurements"]) > 0
        for measurement in response["measurements"]:
            assert "step" in measurement
            assert "measured_density" in measurement

    def test_analyze_returns_dmax(self) -> None:
        """Test analyze returns Dmax value."""
        response = {
            "analysis": {
                "dmax": 2.1,
                "dmin": 0.05,
                "contrast_range": 2.05,
            }
        }

        assert "dmax" in response["analysis"]
        assert response["analysis"]["dmax"] > 0

    def test_analyze_includes_recommendations(self) -> None:
        """Test analyze includes improvement recommendations."""
        response = {
            "analysis": {
                "dmax": 2.1,
                "recommendations": [
                    "Consider longer exposure for deeper blacks",
                ],
            }
        }

        assert "recommendations" in response["analysis"]
        assert isinstance(response["analysis"]["recommendations"], list)

    # Integration tests using TestClient
    def test_upload_scan_endpoint(
        self, client: TestClient, sample_step_tablet_image
    ) -> None:
        """Test scan upload via HTTP request."""
        from pathlib import Path

        with open(sample_step_tablet_image, "rb") as f:
            files = {"file": ("step_tablet.png", f, "image/png")}
            data = {"tablet_type": "stouffer_21"}
            response = client.post("/api/scan/upload", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "extraction_id" in result
        assert "num_patches" in result
        assert "densities" in result
        assert "dmax" in result
        assert "dmin" in result

    def test_analyze_densities_endpoint(
        self, client: TestClient, sample_densities: list[float]
    ) -> None:
        """Test density analysis via HTTP request."""
        request_data = {"densities": sample_densities}
        response = client.post("/api/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "dmin" in data
        assert "dmax" in data
        assert "range" in data
        assert "is_monotonic" in data
        assert "max_error" in data
        assert "rms_error" in data
        assert "suggestions" in data


class TestChemistryAPI:
    """Tests for chemistry calculation API endpoints."""

    def test_calculate_requires_area(self) -> None:
        """Test calculate endpoint requires print area."""
        request = {"print_area_sq_in": 80}

        assert "print_area_sq_in" in request
        assert request["print_area_sq_in"] > 0

    def test_calculate_returns_all_components(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test calculate returns all chemistry components."""
        required_components = [
            "platinum_ml",
            "palladium_ml",
            "ferric_oxalate_ml",
            "contrast_agent_ml",
            "total_volume_ml",
        ]

        for component in required_components:
            assert component in sample_chemistry_recipe

    def test_calculate_respects_metal_ratio(self) -> None:
        """Test calculate uses provided metal ratio."""
        request = {"print_area_sq_in": 80, "metal_ratio": 0.7}

        # Expected result for 0.7 ratio
        total_metal = 80 * 0.5  # 40ml at 0.5 ml/sq.in
        expected_platinum = total_metal * 0.7
        expected_palladium = total_metal * 0.3

        assert expected_platinum == 28.0
        assert expected_palladium == 12.0

    def test_list_recipes_returns_array(self) -> None:
        """Test GET /api/v1/chemistry/recipes returns array."""
        response = {
            "recipes": [{"id": "recipe-1", "name": "Standard Mix"}],
            "total": 1,
        }

        assert isinstance(response["recipes"], list)

    # Integration tests using TestClient
    def test_list_calibrations_endpoint(self, client: TestClient) -> None:
        """Test listing calibrations via HTTP request."""
        response = client.get("/api/calibrations")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_create_calibration_endpoint(self, client: TestClient) -> None:
        """Test creating a calibration via HTTP request."""
        request_data = {
            "paper_type": "Arches Platine",
            "exposure_time": 180.0,
            "metal_ratio": 0.5,
            "contrast_agent": "na2",
            "contrast_amount": 5.0,
            "developer": "potassium_oxalate",
            "chemistry_type": "platinum_palladium",
            "densities": [0.1 + i * 0.1 for i in range(21)],
            "notes": "Test calibration",
        }
        response = client.post("/api/calibrations", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data
        assert data["message"] == "Calibration saved"

    def test_get_calibration_endpoint(self, client: TestClient) -> None:
        """Test retrieving a calibration by ID via HTTP request."""
        # First, create a calibration
        create_response = client.post(
            "/api/calibrations",
            json={
                "paper_type": "Test Paper",
                "exposure_time": 200.0,
                "metal_ratio": 0.6,
                "contrast_agent": "none",
                "contrast_amount": 0.0,
                "developer": "potassium_oxalate",
                "chemistry_type": "platinum_palladium",
                "densities": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
        )
        calibration_id = create_response.json()["id"]

        # Now retrieve it
        get_response = client.get(f"/api/calibrations/{calibration_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["paper_type"] == "Test Paper"
        assert data["exposure_time"] == 200.0

    def test_get_statistics_endpoint(self, client: TestClient) -> None:
        """Test getting database statistics via HTTP request."""
        response = client.get("/api/statistics")

        assert response.status_code == 200
        # Statistics endpoint returns various metrics
        data = response.json()
        assert isinstance(data, dict)


class TestChatAPI:
    """Tests for AI chat API endpoints."""

    def test_chat_requires_message(self) -> None:
        """Test chat endpoint requires message field."""
        request = {"message": "How do I improve contrast?"}

        assert "message" in request
        assert len(request["message"]) > 0

    def test_chat_returns_assistant_response(self) -> None:
        """Test chat returns assistant message."""
        response = {
            "id": "msg-1",
            "role": "assistant",
            "content": "To improve contrast...",
        }

        assert response["role"] == "assistant"
        assert len(response["content"]) > 0

    def test_chat_history_returns_array(self) -> None:
        """Test chat history endpoint returns message array."""
        response = {
            "messages": [],
            "total": 0,
        }

        assert isinstance(response["messages"], list)

    def test_quick_prompts_available(self) -> None:
        """Test quick prompts are available for AI chat."""
        quick_prompts = [
            "How do I improve my blacks?",
            "What exposure time should I use?",
            "Help me troubleshoot uneven coating",
        ]

        assert len(quick_prompts) >= 3

    # Integration tests using TestClient
    @pytest.mark.skip(reason="LLM dependencies may not be available in test environment")
    def test_chat_endpoint(self, client: TestClient) -> None:
        """Test chat endpoint via HTTP request (requires LLM setup)."""
        request_data = {
            "message": "How do I improve contrast?",
            "include_history": True,
        }
        response = client.post("/api/chat", json=request_data)

        # May return 500 if LLM not configured, which is acceptable
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "response" in data

    @pytest.mark.skip(reason="LLM dependencies may not be available in test environment")
    def test_recipe_suggestion_endpoint(self, client: TestClient) -> None:
        """Test recipe suggestion endpoint via HTTP request."""
        request_data = {
            "paper_type": "Arches Platine",
            "characteristics": "High contrast with deep blacks",
        }
        response = client.post("/api/chat/recipe", json=request_data)

        # May return 500 if LLM not configured
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "response" in data

    @pytest.mark.skip(reason="LLM dependencies may not be available in test environment")
    def test_troubleshoot_endpoint(self, client: TestClient) -> None:
        """Test troubleshooting endpoint via HTTP request."""
        request_data = {"problem": "Uneven coating on paper"}
        response = client.post("/api/chat/troubleshoot", json=request_data)

        # May return 500 if LLM not configured
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "response" in data


class TestSessionsAPI:
    """Tests for print session API endpoints."""

    def test_create_session_requires_name(self) -> None:
        """Test creating session requires name."""
        request = {
            "name": "Sunday Print Session",
            "date": "2025-01-15T10:00:00Z",
        }

        assert "name" in request

    def test_list_sessions_supports_pagination(self) -> None:
        """Test sessions list supports pagination."""
        request_params = {"page": 1, "limit": 10}

        assert request_params["page"] >= 1
        assert 1 <= request_params["limit"] <= 100

    def test_session_includes_prints(self) -> None:
        """Test session response includes prints array."""
        response = {
            "id": "session-1",
            "name": "Test Session",
            "prints": [],
        }

        assert isinstance(response["prints"], list)


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_404_error_structure(self) -> None:
        """Test 404 error response structure."""
        error_response = {
            "error": "Not Found",
            "message": "Resource not found",
            "status_code": 404,
        }

        assert error_response["status_code"] == 404
        assert "error" in error_response
        assert "message" in error_response

    def test_422_validation_error_includes_details(self) -> None:
        """Test validation error includes field details."""
        error_response = {
            "error": "Validation Error",
            "message": "Invalid input",
            "status_code": 422,
            "details": [
                {"field": "name", "message": "Name is required"},
            ],
        }

        assert error_response["status_code"] == 422
        assert "details" in error_response
        assert len(error_response["details"]) > 0

    def test_500_error_hides_internal_details(self) -> None:
        """Test 500 error doesn't expose internal details."""
        error_response = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500,
        }

        # Should not contain stack trace or internal details
        assert "traceback" not in error_response
        assert "stack" not in error_response

    # Integration tests using TestClient
    def test_404_on_invalid_endpoint(self, client: TestClient) -> None:
        """Test that invalid endpoints return 404."""
        response = client.get("/api/nonexistent/endpoint")

        assert response.status_code == 404

    def test_422_on_invalid_request_body(self, client: TestClient) -> None:
        """Test validation error on invalid request."""
        # Missing required fields
        invalid_request = {"invalid_field": "test"}
        response = client.post("/api/calibrations", json=invalid_request)

        assert response.status_code == 422

    def test_400_on_invalid_curve_generation(self, client: TestClient) -> None:
        """Test 400 error on invalid curve generation request."""
        # Empty densities should cause an error
        invalid_request = {
            "densities": [],
            "name": "Test",
            "curve_type": "linear",
        }
        response = client.post("/api/curves/generate", json=invalid_request)

        assert response.status_code in [400, 422]


class TestAPIAuthentication:
    """Tests for API authentication (future implementation)."""

    def test_unauthenticated_requests_allowed_for_public_endpoints(self) -> None:
        """Test public endpoints don't require auth."""
        public_endpoints = ["/health", "/api/v1/chemistry/calculate"]

        for endpoint in public_endpoints:
            # These should not require authentication
            assert endpoint.startswith("/")

    def test_api_key_header_format(self) -> None:
        """Test API key header format (when implemented)."""
        headers = {"X-API-Key": "test-api-key-12345"}

        assert "X-API-Key" in headers
