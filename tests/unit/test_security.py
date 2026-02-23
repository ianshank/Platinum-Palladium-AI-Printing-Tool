"""
Security tests for the Pt/Pd Calibration Studio.

Validates that API key handling, input sanitization, filesystem boundaries,
and configuration security follow best practices.
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ptpd_calibration.gcp.config import GCPConfig
from ptpd_calibration.gcp.storage import LocalBackend


class TestCredentialSafety:
    """Ensure API keys / credentials are not exposed."""

    def test_gcp_config_service_account_as_path_type(self) -> None:
        config = GCPConfig(
            project_id="p",
            bucket_name="b",
            service_account_path=Path("/secret/sa-key.json"),
        )
        assert isinstance(config.service_account_path, Path)

    def test_no_hardcoded_secrets_in_config_module(self) -> None:
        """Ensure config module does not contain hardcoded secrets."""
        import inspect

        import ptpd_calibration.config as cfg

        source = inspect.getsource(cfg)
        patterns = ["sk-ant-", "sk-proj-", "AKIA", "password123"]
        for pat in patterns:
            assert pat not in source, f"Hardcoded secret pattern found: {pat}"

    def test_no_hardcoded_secrets_in_gcp_config(self) -> None:
        import inspect

        import ptpd_calibration.gcp.config as gcfg

        source = inspect.getsource(gcfg)
        patterns = ["sk-ant-", "AKIA", "password123"]
        for pat in patterns:
            assert pat not in source, f"Hardcoded secret pattern found: {pat}"


class TestPathTraversal:
    """Ensure storage backends handle path inputs safely."""

    def test_local_backend_save_under_root(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("subdir/file.txt", "safe data")
        assert (tmp_path / "subdir" / "file.txt").exists()

    def test_local_backend_strips_leading_slash(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("/leading.txt", "data")
        assert (tmp_path / "leading.txt").exists()


class TestInputValidation:
    """Validate that input models handle edge cases."""

    def test_gcp_config_accepts_valid_data(self) -> None:
        config = GCPConfig(project_id="my-project", bucket_name="my-bucket")
        assert config.project_id == "my-project"
        assert config.bucket_name == "my-bucket"

    @patch.dict("os.environ", {}, clear=True)
    def test_llm_client_rejects_missing_key_no_env(self) -> None:
        from ptpd_calibration.config import LLMProvider, LLMSettings
        from ptpd_calibration.llm.client import AnthropicClient

        settings = LLMSettings(provider=LLMProvider.ANTHROPIC)
        with pytest.raises(ValueError, match="API key"):
            AnthropicClient(settings)

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_client_rejects_missing_key_no_env(self) -> None:
        from ptpd_calibration.config import LLMProvider, LLMSettings
        from ptpd_calibration.llm.client import OpenAIClient

        settings = LLMSettings(provider=LLMProvider.OPENAI)
        with pytest.raises(ValueError, match="API key"):
            OpenAIClient(settings)


class TestDependencyIsolation:
    """Ensure optional-dependency guards work correctly."""

    @patch("ptpd_calibration.gcp.storage.GOOGLE_CLOUD_AVAILABLE", False)
    def test_gcs_backend_unavailable_without_sdk(self) -> None:
        from ptpd_calibration.gcp.storage import GCSBackend

        config = GCPConfig(project_id="p", bucket_name="b")
        with pytest.raises(ImportError, match="google-cloud-storage"):
            GCSBackend(config)

    @patch("ptpd_calibration.gcp.vertex.VERTEX_AVAILABLE", False)
    def test_vertex_unavailable_without_sdk(self) -> None:
        from ptpd_calibration.gcp.vertex import VertexClient

        config = GCPConfig(project_id="p", bucket_name="b")
        with pytest.raises(ImportError, match="google-cloud-aiplatform"):
            VertexClient(config)


class TestEnvironmentVariableIsolation:
    """Ensure env-based config does not leak across tests."""

    def test_config_ignores_unrelated_env_vars(self) -> None:
        env = {
            "PTPD_GCP_PROJECT_ID": "proj",
            "PTPD_GCS_BUCKET": "bucket",
            "UNRELATED_VAR": "should-be-ignored",
        }
        with patch.dict(os.environ, env, clear=True):
            config = GCPConfig()  # type: ignore[call-arg]
            assert not hasattr(config, "UNRELATED_VAR")
