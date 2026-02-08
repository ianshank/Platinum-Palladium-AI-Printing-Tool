import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import ptpd_calibration.config
from ptpd_calibration.api.server import create_app
from ptpd_calibration.gcp.config import get_gcp_config


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env(temp_storage_dir):
    """Mock environment variables for local storage."""
    env_vars = {
        "PTPD_GCP_PROJECT_ID": "test-project",
        "PTPD_GCP_REGION": "us-central1",
        "PTPD_GCS_BUCKET": "test-bucket",
        "PTPD_FORCE_LOCAL_STORAGE": "true",
        "PTPD_STAGING_DIR": str(temp_storage_dir),
    }
    with patch.dict(os.environ, env_vars):
        # Clear caches to ensure reload
        ptpd_calibration.config._settings = None
        get_gcp_config.cache_clear()
        yield
        ptpd_calibration.config._settings = None


def test_curve_persistence(mock_env, temp_storage_dir):
    """Test that curves are persisted to local storage."""
    client = TestClient(create_app())

    # 1. Generate a curve
    response = client.post(
        "/api/curves/generate",
        json={
            "densities": [0.1, 0.5, 1.0, 1.5, 2.0],
            "name": "Test Curve",
            "curve_type": "linear",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    curve_id = data["curve_id"]

    # 2. Verify file existence
    expected_path = temp_storage_dir / "curves" / f"{curve_id}.json"
    assert expected_path.exists()

    # Check file content
    with open(expected_path) as f:
        saved_data = json.load(f)
    assert saved_data["id"] == curve_id
    assert saved_data["name"] == "Test Curve"

    # 3. Retrieve curve via API
    response = client.get(f"/api/curves/{curve_id}")
    assert response.status_code == 200
    retrieved_data = response.json()
    assert retrieved_data["curve_id"] == curve_id
    assert retrieved_data["name"] == "Test Curve"


def test_scan_upload_persistence(mock_env, temp_storage_dir):
    """Test that uploaded scans are persisted."""
    client = TestClient(create_app())

    # Create dummy image content
    scan_content = b"fake_image_data"
    files = {"file": ("test_scan.jpg", scan_content, "image/jpeg")}

    # Mock the reader step to avoid actual image processing failure
    with patch("ptpd_calibration.detection.StepTabletReader.read") as mock_read:
        # Create a mock result
        from ptpd_calibration.core.models import ExtractionResult, StepTabletResult

        mock_result = StepTabletResult(
            extraction=ExtractionResult(
                image_size=(100, 100), tablet_bounds=(0, 0, 100, 100), patches=[]
            )
        )
        mock_read.return_value = mock_result

        response = client.post("/api/scan/upload", files=files, data={"tablet_type": "stouffer_21"})

    assert response.status_code == 200

    # Verify file existence in storage
    expected_path = temp_storage_dir / "scans" / "test_scan.jpg"
    assert expected_path.exists()
    assert expected_path.read_bytes() == scan_content


def test_database_persistence(mock_env, temp_storage_dir):
    """Test that calibration records are persisted."""
    client = TestClient(create_app())

    # Add a calibration record
    record_data = {
        "paper_type": "Arches Platine",
        "exposure_time": 120.0,
        "metal_ratio": 0.5,
        "contrast_agent": "none",
        "contrast_amount": 0.0,
        "developer": "potassium_oxalate",
        "chemistry_type": "platinum_palladium",
        "densities": [0.1, 0.5, 1.2],
        "notes": "Test record",
    }

    response = client.post("/api/calibrations", json=record_data)
    assert response.status_code == 200

    # Check database file
    db_path = temp_storage_dir / "database.json"
    assert db_path.exists()

    with open(db_path) as f:
        db_data = json.load(f)

    assert len(db_data["records"]) >= 1
    stored_record = db_data["records"][-1]
    assert stored_record["paper_type"] == "Arches Platine"
