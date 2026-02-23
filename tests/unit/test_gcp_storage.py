"""
Unit tests for storage backends (LocalBackend and GCSBackend).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.gcp.config import GCPConfig
from ptpd_calibration.gcp.storage import (
    LocalBackend,
    StorageBackend,
    get_storage_backend,
)


@pytest.fixture
def gcp_config(tmp_path: Path) -> GCPConfig:
    return GCPConfig(
        project_id="test-project",
        bucket_name="test-bucket",
        staging_dir=tmp_path / "staging",
    )


class TestLocalBackend:
    """Tests for LocalBackend file system operations."""

    def test_implements_protocol(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        assert isinstance(backend, StorageBackend)

    def test_creates_root_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "new_dir"
        assert not root.exists()
        LocalBackend(root)
        assert root.exists()

    def test_save_and_load_text(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("test.txt", "Hello, World!")
        data = backend.load("test.txt")
        assert data == b"Hello, World!"

    def test_save_and_load_bytes(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        payload = b"\x89PNG\r\n\x1a\n"
        backend.save("image.png", payload, content_type="image/png")
        assert backend.load("image.png") == payload

    def test_save_creates_subdirectories(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("a/b/c/file.txt", "nested")
        assert backend.exists("a/b/c/file.txt")

    def test_exists_returns_false_for_missing(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        assert backend.exists("nonexistent.txt") is False

    def test_exists_returns_true_after_save(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("present.txt", "data")
        assert backend.exists("present.txt") is True

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        with pytest.raises(FileNotFoundError):
            backend.load("missing.txt")

    def test_strips_leading_slash(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("/leading/slash.txt", "data")
        assert backend.exists("leading/slash.txt")

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        backend = LocalBackend(tmp_path)
        backend.save("overwrite.txt", "original")
        backend.save("overwrite.txt", "updated")
        assert backend.load("overwrite.txt") == b"updated"


class TestGetStorageBackend:
    """Tests for the storage factory function."""

    def test_returns_local_when_forced(self, gcp_config: GCPConfig, tmp_path: Path) -> None:
        config = GCPConfig(
            project_id="p",
            bucket_name="b",
            staging_dir=tmp_path,
            force_local_storage=True,
        )
        backend = get_storage_backend(config)
        assert isinstance(backend, LocalBackend)

    @patch("ptpd_calibration.gcp.storage.GOOGLE_CLOUD_AVAILABLE", False)
    def test_gcs_backend_raises_without_dependency(self, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.storage import GCSBackend

        with pytest.raises(ImportError, match="google-cloud-storage"):
            GCSBackend(gcp_config)

    @patch("ptpd_calibration.gcp.storage.GOOGLE_CLOUD_AVAILABLE", True)
    @patch("ptpd_calibration.gcp.storage.storage")
    def test_gcs_backend_initializes(self, mock_storage: MagicMock, gcp_config: GCPConfig) -> None:
        from ptpd_calibration.gcp.storage import GCSBackend

        mock_client = MagicMock()
        mock_storage.Client.return_value = mock_client
        backend = GCSBackend(gcp_config)
        assert backend.bucket_name == "test-bucket"
        mock_storage.Client.assert_called_once_with(project="test-project")
