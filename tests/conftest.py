import os
import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Set mock environment variables for testing."""
    os.environ["PTPD_GCP_PROJECT_ID"] = "test-project"
    os.environ["PTPD_GCS_BUCKET"] = "test-bucket"
    os.environ["PTPD_GCP_REGION"] = "us-central1"
    yield
    # Cleanup is handled by os.environ changes being process-local, 
    # but good practice to clear if we wanted strict isolation.
    # For now, let's just leave them as they don't hurt.

@pytest.fixture
def mock_settings(mocker):
    """Mock get_settings to return valid settings."""
    # This can be used if we want to bypass pydantic validation entirely
    pass
