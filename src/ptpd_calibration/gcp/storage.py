"""
Storage abstractions for Google Cloud Storage and local file system.
"""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    from google.cloud import storage

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

from ptpd_calibration.gcp.config import GCPConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the storage interface."""

    def save(self, path: str, data: str | bytes, content_type: str = "text/plain") -> None:
        """Save data to the specified path."""
        ...

    def load(self, path: str) -> bytes:
        """Load data from the specified path."""
        ...

    def exists(self, path: str) -> bool:
        """Check if a file exists at the specified path."""
        ...


class LocalBackend:
    """Storage backend for local file system."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a storage path to an absolute local path (pure — no side effects)."""
        return self.root_dir / path.lstrip("/")

    def save(self, path: str, data: str | bytes, content_type: str = "text/plain") -> None:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(data, bytes) else "w"
        encoding = None if isinstance(data, bytes) else "utf-8"

        with open(target, mode, encoding=encoding) as f:
            f.write(data)
        logger.info(f"Saved local file: {target}")

    def load(self, path: str) -> bytes:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        return target.read_bytes()

    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()


class GCSBackend:
    """Storage backend for Google Cloud Storage."""

    def __init__(self, config: GCPConfig):
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCSBackend")

        self.client = storage.Client(project=config.project_id)
        self.bucket_name = config.bucket_name
        self.bucket = self.client.bucket(self.bucket_name)

    def save(self, path: str, data: str | bytes, content_type: str = "text/plain") -> None:
        blob = self.bucket.blob(path.lstrip("/"))
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"Uploaded to gs://{self.bucket_name}/{path}")

    def load(self, path: str) -> bytes:
        blob = self.bucket.blob(path.lstrip("/"))
        if not blob.exists():
            raise FileNotFoundError(f"Blob not found: gs://{self.bucket_name}/{path}")
        return blob.download_as_bytes()

    def exists(self, path: str) -> bool:
        return self.bucket.blob(path.lstrip("/")).exists()


def get_storage_backend(config: GCPConfig) -> StorageBackend:
    """Factory to get the appropriate storage backend."""
    if config.force_local_storage:
        return LocalBackend(config.staging_dir)
    return GCSBackend(config)
