"""
Scan and Analysis Endpoint Tests.

Tests for step tablet upload and density analysis endpoints.
"""

import pytest


@pytest.mark.api
class TestAnalyzeEndpoints:
    """Test density analysis endpoints."""

    def test_analyze_densities(self, client, sample_densities):
        """Test analyzing density measurements."""
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

    def test_analyze_densities_monotonic(self, client):
        """Test analysis of monotonically increasing densities."""
        monotonic = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        request_data = {"densities": monotonic}

        response = client.post("/api/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["is_monotonic"] is True

    def test_analyze_densities_non_monotonic(self, client):
        """Test analysis of non-monotonic densities."""
        non_monotonic = [0.1, 0.5, 0.3, 0.7, 0.6, 0.9]  # Contains inversions
        request_data = {"densities": non_monotonic}

        response = client.post("/api/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["is_monotonic"] is False

    def test_analyze_densities_empty(self, client):
        """Test analysis with empty densities."""
        request_data = {"densities": []}

        response = client.post("/api/analyze", json=request_data)

        assert response.status_code in [400, 422, 500]

    def test_analyze_densities_single_value(self, client):
        """Test analysis with single density value."""
        request_data = {"densities": [1.5]}

        response = client.post("/api/analyze", json=request_data)

        # Should handle single value case
        assert response.status_code in [200, 400]


@pytest.mark.api
class TestScanUploadEndpoints:
    """Test step tablet scan upload endpoints."""

    def test_upload_scan(self, client, sample_step_tablet_image):
        """Test uploading a step tablet scan."""
        with open(sample_step_tablet_image, "rb") as f:
            files = {"file": ("step_tablet.png", f, "image/png")}
            data = {"tablet_type": "stouffer_21"}

            response = client.post("/api/scan/upload", files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "extraction_id" in data
        assert "num_patches" in data
        assert "densities" in data

    def test_upload_scan_different_tablet_types(self, client, sample_step_tablet_image):
        """Test uploading with different tablet types."""
        tablet_types = ["stouffer_21", "stouffer_31", "custom"]

        for tablet_type in tablet_types:
            with open(sample_step_tablet_image, "rb") as f:
                files = {"file": ("step_tablet.png", f, "image/png")}
                data = {"tablet_type": tablet_type}

                response = client.post("/api/scan/upload", files=files, data=data)

                # Should either succeed or give meaningful error
                assert response.status_code in [200, 400]

    def test_upload_scan_invalid_file(self, client, tmp_path):
        """Test uploading an invalid file."""
        # Create a text file instead of image
        invalid_file = tmp_path / "not_an_image.txt"
        invalid_file.write_text("This is not an image")

        with open(invalid_file, "rb") as f:
            files = {"file": ("not_an_image.txt", f, "text/plain")}
            data = {"tablet_type": "stouffer_21"}

            response = client.post("/api/scan/upload", files=files, data=data)

        assert response.status_code == 400

    def test_upload_scan_no_file(self, client):
        """Test upload without a file."""
        data = {"tablet_type": "stouffer_21"}

        response = client.post("/api/scan/upload", data=data)

        assert response.status_code == 422

    def test_upload_scan_response_fields(self, client, sample_step_tablet_image):
        """Test that scan response contains expected fields."""
        with open(sample_step_tablet_image, "rb") as f:
            files = {"file": ("step_tablet.png", f, "image/png")}
            data = {"tablet_type": "stouffer_21"}

            response = client.post("/api/scan/upload", files=files, data=data)

        if response.status_code == 200:
            data = response.json()
            assert "dmin" in data
            assert "dmax" in data
            assert "range" in data
            assert "quality" in data


@pytest.mark.api
class TestQuadFileEndpoints:
    """Test .quad file parsing endpoints."""

    def test_upload_quad_file(self, client, tmp_path, sample_quad_content):
        """Test uploading a .quad file."""
        quad_file = tmp_path / "test.quad"
        quad_file.write_text(sample_quad_content)

        with open(quad_file, "rb") as f:
            files = {"file": ("test.quad", f, "application/octet-stream")}
            data = {"channel": "K"}

            response = client.post("/api/curves/upload-quad", files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "profile_name" in data
        assert "active_channels" in data

    def test_parse_quad_content(self, client, sample_quad_content):
        """Test parsing .quad content from string."""
        data = {
            "content": sample_quad_content,
            "name": "Parsed Profile",
            "channel": "K",
        }

        response = client.post("/api/curves/parse-quad", data=data)

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["profile_name"] == "Parsed Profile"

    def test_parse_quad_invalid_content(self, client):
        """Test parsing invalid .quad content."""
        data = {
            "content": "This is not valid quad content",
            "name": "Invalid Profile",
            "channel": "K",
        }

        response = client.post("/api/curves/parse-quad", data=data)

        assert response.status_code == 400

    def test_upload_quad_nonexistent_channel(self, client, tmp_path, sample_quad_content):
        """Test requesting a non-existent channel from .quad file."""
        quad_file = tmp_path / "test.quad"
        quad_file.write_text(sample_quad_content)

        with open(quad_file, "rb") as f:
            files = {"file": ("test.quad", f, "application/octet-stream")}
            data = {"channel": "NONEXISTENT"}

            response = client.post("/api/curves/upload-quad", files=files, data=data)

        # Should succeed but with null curve_data for missing channel
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["curve_data"] is None
