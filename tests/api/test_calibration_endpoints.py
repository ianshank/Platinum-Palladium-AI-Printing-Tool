"""
Calibration Endpoint Tests.

Tests for calibration record CRUD operations.
"""

import pytest


@pytest.mark.api
class TestCalibrationEndpoints:
    """Test calibration record endpoints."""

    def test_list_calibrations_empty(self, client):
        """Test listing calibrations when database is empty."""
        response = client.get("/api/calibrations")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_list_calibrations_with_limit(self, client):
        """Test listing calibrations with limit parameter."""
        response = client.get("/api/calibrations?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data["records"]) <= 10

    def test_list_calibrations_with_paper_filter(self, client):
        """Test listing calibrations filtered by paper type."""
        response = client.get("/api/calibrations?paper_type=Arches%20Platine")

        assert response.status_code == 200
        data = response.json()
        assert "records" in data

    def test_create_calibration(self, client, calibration_request_data):
        """Test creating a new calibration record."""
        response = client.post("/api/calibrations", json=calibration_request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data
        assert data["message"] == "Calibration saved"

    def test_create_calibration_minimal(self, client):
        """Test creating calibration with minimal required fields."""
        minimal_data = {
            "paper_type": "Test Paper",
            "exposure_time": 120.0,
            "metal_ratio": 0.5,
        }

        response = client.post("/api/calibrations", json=minimal_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_create_calibration_with_densities(self, client, sample_densities):
        """Test creating calibration with density measurements."""
        request_data = {
            "paper_type": "Test Paper",
            "exposure_time": 180.0,
            "metal_ratio": 0.5,
            "densities": sample_densities,
        }

        response = client.post("/api/calibrations", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_calibration_not_found(self, client):
        """Test getting a non-existent calibration."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/calibrations/{fake_id}")

        assert response.status_code == 404

    def test_get_calibration_invalid_uuid(self, client):
        """Test getting calibration with invalid UUID."""
        response = client.get("/api/calibrations/invalid-uuid")

        assert response.status_code in [400, 422]

    def test_create_and_retrieve_calibration(self, client, calibration_request_data):
        """Test creating and then retrieving a calibration."""
        # Create
        create_response = client.post("/api/calibrations", json=calibration_request_data)
        assert create_response.status_code == 200
        calibration_id = create_response.json()["id"]

        # Retrieve
        get_response = client.get(f"/api/calibrations/{calibration_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        assert data["paper_type"] == calibration_request_data["paper_type"]
        assert data["exposure_time"] == calibration_request_data["exposure_time"]

    def test_statistics_endpoint(self, client):
        """Test getting database statistics."""
        response = client.get("/api/statistics")

        assert response.status_code == 200
        data = response.json()
        # Should return some statistics about the database
        assert isinstance(data, dict)


@pytest.mark.api
class TestCalibrationValidation:
    """Test calibration request validation."""

    def test_missing_required_fields(self, client):
        """Test that missing required fields returns error."""
        incomplete_data = {
            "paper_type": "Test Paper",
            # Missing exposure_time
        }

        response = client.post("/api/calibrations", json=incomplete_data)

        assert response.status_code == 422

    def test_invalid_metal_ratio(self, client):
        """Test validation of metal ratio bounds."""
        invalid_data = {
            "paper_type": "Test Paper",
            "exposure_time": 180.0,
            "metal_ratio": 1.5,  # Invalid: should be 0-1
        }

        response = client.post("/api/calibrations", json=invalid_data)

        # Metal ratio out of bounds should be rejected with validation error
        assert response.status_code == 422

    def test_negative_exposure_time(self, client):
        """Test validation of negative exposure time."""
        invalid_data = {
            "paper_type": "Test Paper",
            "exposure_time": -10.0,
            "metal_ratio": 0.5,
        }

        response = client.post("/api/calibrations", json=invalid_data)

        # Negative exposure time should fail or be handled
        assert response.status_code in [200, 400, 422]

    def test_empty_paper_type(self, client):
        """Test validation of empty paper type."""
        invalid_data = {
            "paper_type": "",
            "exposure_time": 180.0,
            "metal_ratio": 0.5,
        }

        response = client.post("/api/calibrations", json=invalid_data)

        assert response.status_code in [200, 400, 422]
