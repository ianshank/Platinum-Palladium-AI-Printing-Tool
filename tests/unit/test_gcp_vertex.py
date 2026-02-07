"""
Unit tests for Vertex AI client wrapper.
"""
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.gcp.config import GCPConfig


@pytest.fixture
def gcp_config() -> GCPConfig:
    return GCPConfig(project_id="test-project", bucket_name="test-bucket", region="us-central1")


class TestVertexClient:
    """Tests for the VertexClient wrapper."""

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", False)
    def test_raises_import_error_without_dep(self, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        with pytest.raises(ImportError, match="google-cloud-aiplatform"):
            VertexClient(gcp_config)

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", True)
    def test_init_sets_config(self, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        client = VertexClient(gcp_config)
        assert client.config is gcp_config
        assert client._initialized is False

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", True)
    @patch("ptpd_calibration.gcp.vertex.aiplatform")
    def test_initialize_calls_sdk(self, mock_aiplatform: MagicMock, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        client = VertexClient(gcp_config)
        client.initialize()

        mock_aiplatform.init.assert_called_once_with(
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket",
        )
        assert client._initialized is True

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", True)
    @patch("ptpd_calibration.gcp.vertex.aiplatform")
    def test_initialize_is_idempotent(self, mock_aiplatform: MagicMock, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        client = VertexClient(gcp_config)
        client.initialize()
        client.initialize()

        assert mock_aiplatform.init.call_count == 1

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", True)
    @patch("ptpd_calibration.gcp.vertex.aiplatform")
    def test_get_model_initializes_first(self, mock_aiplatform: MagicMock, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        client = VertexClient(gcp_config)
        client.get_model("projects/p/locations/l/models/m")

        mock_aiplatform.init.assert_called_once()
        mock_aiplatform.Model.assert_called_once_with(model_name="projects/p/locations/l/models/m", version=None)

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", True)
    @patch("ptpd_calibration.gcp.vertex.aiplatform")
    def test_submit_custom_job(self, mock_aiplatform: MagicMock, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        client = VertexClient(gcp_config)
        client.submit_custom_job(display_name="test-job", container_uri="gcr.io/test/image:latest")

        mock_aiplatform.init.assert_called_once()
        mock_aiplatform.CustomContainerTrainingJob.assert_called_once_with(
            display_name="test-job",
            container_uri="gcr.io/test/image:latest",
        )
