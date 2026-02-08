"""
Curve Endpoint Tests.

Tests for curve generation, modification, and export endpoints.
"""

import pytest


@pytest.mark.api
class TestCurveGeneration:
    """Test curve generation endpoints."""

    def test_generate_curve_basic(self, client, sample_densities):
        """Test basic curve generation."""
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
        assert data["num_points"] > 0

    def test_generate_curve_with_paper_type(self, client, sample_densities):
        """Test curve generation with paper type."""
        request_data = {
            "densities": sample_densities,
            "name": "Paper Curve",
            "curve_type": "linear",
            "paper_type": "Arches Platine",
        }

        response = client.post("/api/curves/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_generate_curve_different_types(self, client, sample_densities):
        """Test curve generation with different curve types."""
        curve_types = ["linear", "spline", "polynomial"]

        for curve_type in curve_types:
            request_data = {
                "densities": sample_densities,
                "name": f"{curve_type.title()} Curve",
                "curve_type": curve_type,
            }

            response = client.post("/api/curves/generate", json=request_data)

            assert response.status_code == 200, f"Failed for curve_type: {curve_type}"

    def test_generate_curve_empty_densities(self, client):
        """Test curve generation with empty densities."""
        request_data = {
            "densities": [],
            "name": "Empty Curve",
        }

        response = client.post("/api/curves/generate", json=request_data)

        # Should fail with empty densities
        assert response.status_code in [400, 422]


@pytest.mark.api
class TestCurveModification:
    """Test curve modification endpoints."""

    def test_modify_curve_brightness(self, client, sample_curve_data):
        """Test brightness adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Brightness Adjusted",
            "adjustment_type": "brightness",
            "amount": 0.1,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["adjustment_applied"] == "brightness"

    def test_modify_curve_contrast(self, client, sample_curve_data):
        """Test contrast adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Contrast Adjusted",
            "adjustment_type": "contrast",
            "amount": 0.2,
            "pivot": 0.5,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["adjustment_applied"] == "contrast"

    def test_modify_curve_gamma(self, client, sample_curve_data):
        """Test gamma adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Gamma Adjusted",
            "adjustment_type": "gamma",
            "amount": 0.5,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_modify_curve_levels(self, client, sample_curve_data):
        """Test levels adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Levels Adjusted",
            "adjustment_type": "levels",
            "black_point": 0.05,
            "white_point": 0.95,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_modify_curve_highlights(self, client, sample_curve_data):
        """Test highlights adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Highlights Adjusted",
            "adjustment_type": "highlights",
            "amount": 0.15,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200

    def test_modify_curve_shadows(self, client, sample_curve_data):
        """Test shadows adjustment."""
        request_data = {
            **sample_curve_data,
            "name": "Shadows Adjusted",
            "adjustment_type": "shadows",
            "amount": -0.1,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 200

    def test_modify_curve_invalid_adjustment(self, client, sample_curve_data):
        """Test invalid adjustment type."""
        request_data = {
            **sample_curve_data,
            "name": "Invalid Adjustment",
            "adjustment_type": "invalid_type",
            "amount": 0.1,
        }

        response = client.post("/api/curves/modify", json=request_data)

        assert response.status_code == 400


@pytest.mark.api
class TestCurveSmoothing:
    """Test curve smoothing endpoints."""

    def test_smooth_curve_gaussian(self, client, sample_curve_data):
        """Test Gaussian smoothing."""
        request_data = {
            **sample_curve_data,
            "name": "Gaussian Smoothed",
            "method": "gaussian",
            "strength": 0.5,
        }

        response = client.post("/api/curves/smooth", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["method_applied"] == "gaussian"

    def test_smooth_curve_savgol(self, client, sample_curve_data):
        """Test Savitzky-Golay smoothing."""
        request_data = {
            **sample_curve_data,
            "name": "Savgol Smoothed",
            "method": "savgol",
            "strength": 0.5,
        }

        response = client.post("/api/curves/smooth", json=request_data)

        assert response.status_code == 200

    def test_smooth_curve_preserve_endpoints(self, client, sample_curve_data):
        """Test smoothing with endpoint preservation."""
        request_data = {
            **sample_curve_data,
            "name": "Preserved Endpoints",
            "method": "gaussian",
            "strength": 0.8,
            "preserve_endpoints": True,
        }

        response = client.post("/api/curves/smooth", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Endpoints should be preserved
        assert data["output_values"][0] == sample_curve_data["output_values"][0]
        assert data["output_values"][-1] == sample_curve_data["output_values"][-1]


@pytest.mark.api
class TestCurveBlending:
    """Test curve blending endpoints."""

    def test_blend_curves_average(self, client, sample_curve_data):
        """Test curve blending with average mode."""
        # Create second curve with different values
        curve2_outputs = [x**1.1 for x in sample_curve_data["input_values"]]

        request_data = {
            "curve1_inputs": sample_curve_data["input_values"],
            "curve1_outputs": sample_curve_data["output_values"],
            "curve2_inputs": sample_curve_data["input_values"],
            "curve2_outputs": curve2_outputs,
            "name": "Blended Curve",
            "mode": "average",
        }

        response = client.post("/api/curves/blend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode_applied"] == "average"

    def test_blend_curves_weighted(self, client, sample_curve_data):
        """Test curve blending with weighted mode."""
        curve2_outputs = [x**1.1 for x in sample_curve_data["input_values"]]

        request_data = {
            "curve1_inputs": sample_curve_data["input_values"],
            "curve1_outputs": sample_curve_data["output_values"],
            "curve2_inputs": sample_curve_data["input_values"],
            "curve2_outputs": curve2_outputs,
            "name": "Weighted Blend",
            "mode": "weighted",
            "weight": 0.7,
        }

        response = client.post("/api/curves/blend", json=request_data)

        assert response.status_code == 200


@pytest.mark.api
class TestCurveStorage:
    """Test curve storage and retrieval endpoints."""

    def test_get_stored_curve(self, client, sample_curve_data):
        """Test retrieving a stored curve."""
        # First, create a curve
        modify_request = {
            **sample_curve_data,
            "name": "Stored Curve",
            "adjustment_type": "brightness",
            "amount": 0.0,
        }

        create_response = client.post("/api/curves/modify", json=modify_request)
        assert create_response.status_code == 200
        curve_id = create_response.json()["curve_id"]

        # Then retrieve it
        get_response = client.get(f"/api/curves/{curve_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["curve_id"] == curve_id

    def test_get_curve_not_found(self, client):
        """Test getting a non-existent curve."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/curves/{fake_id}")

        assert response.status_code == 404

    def test_enforce_monotonicity(self, client, sample_curve_data):
        """Test enforcing monotonicity on a curve."""
        # Create a curve first
        modify_request = {
            **sample_curve_data,
            "name": "Monotonic Test",
            "adjustment_type": "brightness",
            "amount": 0.0,
        }

        create_response = client.post("/api/curves/modify", json=modify_request)
        curve_id = create_response.json()["curve_id"]

        # Enforce monotonicity
        mono_response = client.post(
            f"/api/curves/{curve_id}/enforce-monotonicity?direction=increasing"
        )

        assert mono_response.status_code == 200
        data = mono_response.json()
        assert data["success"] is True
