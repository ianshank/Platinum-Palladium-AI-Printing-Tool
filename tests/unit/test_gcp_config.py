"""
Unit tests for GCP configuration management.
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ptpd_calibration.gcp.config import GCPConfig, get_gcp_config


class TestGCPConfig:
    """Tests for GCPConfig Pydantic model."""

    def test_creates_with_required_fields(self) -> None:
        config = GCPConfig(
            project_id="test-project",
            bucket_name="test-bucket",
        )
        assert config.project_id == "test-project"
        assert config.bucket_name == "test-bucket"

    def test_default_region(self) -> None:
        config = GCPConfig(
            project_id="test-project",
            bucket_name="test-bucket",
        )
        assert config.region == "us-central1"

    def test_custom_region(self) -> None:
        config = GCPConfig(
            project_id="p",
            bucket_name="b",
            region="europe-west1",
        )
        assert config.region == "europe-west1"

    def test_default_staging_dir(self) -> None:
        config = GCPConfig(project_id="p", bucket_name="b")
        assert config.staging_dir == Path("staging")

    def test_custom_staging_dir(self) -> None:
        config = GCPConfig(
            project_id="p",
            bucket_name="b",
            staging_dir=Path("/tmp/custom"),
        )
        assert config.staging_dir == Path("/tmp/custom")

    def test_default_force_local_storage_false(self) -> None:
        config = GCPConfig(project_id="p", bucket_name="b")
        assert config.force_local_storage is False

    def test_service_account_path_optional(self) -> None:
        config = GCPConfig(project_id="p", bucket_name="b")
        assert config.service_account_path is None

    def test_storage_bucket_uri_property(self) -> None:
        config = GCPConfig(project_id="p", bucket_name="my-bucket")
        assert config.storage_bucket_uri == "gs://my-bucket"

    def test_reads_from_env_aliases(self) -> None:
        env = {
            "PTPD_GCP_PROJECT_ID": "env-proj",
            "PTPD_GCS_BUCKET": "env-bucket",
            "PTPD_GCP_REGION": "asia-east1",
        }
        with patch.dict(os.environ, env, clear=False):
            config = GCPConfig()  # type: ignore[call-arg]
            assert config.project_id == "env-proj"
            assert config.bucket_name == "env-bucket"
            assert config.region == "asia-east1"


class TestGetGCPConfig:
    """Tests for the cached factory function."""

    def setup_method(self) -> None:
        get_gcp_config.cache_clear()

    def teardown_method(self) -> None:
        get_gcp_config.cache_clear()

    def test_raises_without_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(Exception):
            get_gcp_config()

    def test_returns_config_with_env(self) -> None:
        env = {
            "PTPD_GCP_PROJECT_ID": "p",
            "PTPD_GCS_BUCKET": "b",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_gcp_config()
            assert isinstance(config, GCPConfig)
            assert config.project_id == "p"
        get_gcp_config.cache_clear()

    def test_caches_result(self) -> None:
        env = {
            "PTPD_GCP_PROJECT_ID": "cached",
            "PTPD_GCS_BUCKET": "bucket",
        }
        with patch.dict(os.environ, env, clear=True):
            c1 = get_gcp_config()
            c2 = get_gcp_config()
            assert c1 is c2
        get_gcp_config.cache_clear()
