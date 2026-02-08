import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock environment variables for testing.

    Uses monkeypatch to ensure automatic cleanup after each test,
    preventing env var leakage between test modules.
    """
    monkeypatch.setenv("PTPD_GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("PTPD_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PTPD_GCP_REGION", "us-central1")

