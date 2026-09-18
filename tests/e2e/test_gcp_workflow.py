"""
End-to-end tests for GCP integration.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.gcp.config import GCPConfig
from ptpd_calibration.gcp.storage import GOOGLE_CLOUD_AVAILABLE, GCSBackend, LocalBackend
from ptpd_calibration.gcp.vertex import VERTEX_AVAILABLE, VertexClient

# The GCS and Vertex wrappers only exist when the optional [gcp] extra is
# installed; patching their SDK handles is impossible otherwise. The local
# backend tests below need no SDK and always run.
requires_gcs = pytest.mark.skipif(
    not GOOGLE_CLOUD_AVAILABLE, reason="google-cloud-storage is not installed (extra: gcp)"
)
requires_vertex = pytest.mark.skipif(
    not VERTEX_AVAILABLE, reason="google-cloud-aiplatform is not installed (extra: gcp)"
)


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Config for the mocked GCS and Vertex clients.

    Values are literal because the clients they are passed to are mocked; no
    credentials are read and no network call is made. The live integration
    tests below are skipped unless PTPD_RUN_GCP_INTEGRATION=1.
    """
    return GCPConfig(project_id="test-project", bucket_name="test-bucket", region="us-central1")


# --- LOCAL BACKEND TESTS ---


def test_local_backend(tmp_path):
    """Test LocalBackend operations."""
    backend = LocalBackend(tmp_path)

    # Save
    backend.save("test.txt", "hello world")
    assert (tmp_path / "test.txt").read_text() == "hello world"

    # Load
    assert backend.load("test.txt") == b"hello world"

    # Exists
    assert backend.exists("test.txt")
    assert not backend.exists("missing.txt")

    # Nested
    backend.save("folder/test.txt", "nested")
    assert (tmp_path / "folder/test.txt").read_text() == "nested"


# --- GCS MOCK TESTS ---


@requires_gcs
@patch("ptpd_calibration.gcp.storage.storage.Client")
def test_gcs_backend_mock(mock_client_cls, gcp_config):
    """Test GCSBackend with mocked storage client."""
    # Setup mock
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client_cls.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Initialize backend
    backend = GCSBackend(gcp_config)

    # Test Save
    backend.save("models/test.pt", b"model data")
    mock_bucket.blob.assert_called_with("models/test.pt")
    mock_blob.upload_from_string.assert_called_with(b"model data", content_type="text/plain")

    # Test Load
    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = b"model data"
    data = backend.load("models/test.pt")
    assert data == b"model data"

    # Test Exists
    mock_blob.exists.return_value = False
    assert not backend.exists("models/missing.pt")


@requires_vertex
@patch("ptpd_calibration.gcp.vertex.aiplatform")
def test_vertex_client_mock(mock_aiplatform, gcp_config):
    """Test VertexClient initialization and operations."""
    client = VertexClient(gcp_config)

    # Test Initialize
    client.initialize()
    mock_aiplatform.init.assert_called_once_with(
        project=gcp_config.project_id,
        location=gcp_config.region,
        staging_bucket=gcp_config.storage_bucket_uri,
    )

    # Test Get Model
    client.get_model("ptpd-model-v1")
    mock_aiplatform.Model.assert_called_with(model_name="ptpd-model-v1", version=None)


# --- INTEGRATION TESTS (SKIPPED BY DEFAULT) ---


def has_gcp_credentials():
    """Check if GCP credentials are available."""
    # Simple check: is gcloud installed and authenticated?
    # Or strict check: GOOGLE_APPLICATION_CREDENTIALS set.
    # For safe CI, we check for a specific env var.
    return os.getenv("PTPD_RUN_GCP_INTEGRATION") == "1"


@requires_gcs
@pytest.mark.skipif(not has_gcp_credentials(), reason="Requires PTPD_RUN_GCP_INTEGRATION=1")
def test_gcs_integration(gcp_config):
    """Live test against a real GCS bucket."""
    # WARNING: This interacts with real cloud resources
    backend = GCSBackend(gcp_config)

    test_content = "Integration Test Payload"
    path = f"integration_tests/{os.urandom(4).hex()}.txt"

    try:
        backend.save(path, test_content)
        assert backend.exists(path)
        loaded = backend.load(path).decode("utf-8")
        assert loaded == test_content
    finally:
        # Cleanup if possible (backend doesn't expose delete, requires raw client)
        blob = backend.bucket.blob(path)
        if blob.exists():
            blob.delete()


@requires_vertex
@pytest.mark.skipif(not has_gcp_credentials(), reason="Requires PTPD_RUN_GCP_INTEGRATION=1")
def test_vertex_integration(gcp_config):
    """Live test against Vertex AI."""
    client = VertexClient(gcp_config)
    client.initialize()
    # Just verify init doesn't crash.
    # Calling get_model might fail if model doesn't exist, which is expected behavior.
