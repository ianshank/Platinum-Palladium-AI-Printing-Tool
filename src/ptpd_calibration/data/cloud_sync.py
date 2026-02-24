"""
Cloud synchronization abstraction layer.

Provides a unified interface for syncing data to cloud storage providers,
with support for local storage (testing), S3, and extensible provider system.
"""

import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SyncStatus(str, Enum):
    """Synchronization status."""

    IDLE = "idle"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConflictStrategy(str, Enum):
    """Conflict resolution strategies."""

    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    NEWEST_WINS = "newest_wins"
    MANUAL = "manual"


class SyncRecord(BaseModel):
    """Record of a sync operation."""

    timestamp: datetime = Field(default_factory=datetime.now)
    status: SyncStatus = Field(...)
    direction: str = Field(...)  # "upload", "download", "bidirectional"
    files_synced: int = Field(default=0)
    bytes_transferred: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float | None = Field(default=None)


class FileMetadata(BaseModel):
    """Metadata for a synced file."""

    path: str = Field(...)
    size: int = Field(...)
    modified_time: datetime = Field(...)
    checksum: str = Field(...)
    local_version: str | None = Field(default=None)
    remote_version: str | None = Field(default=None)


class CloudProvider(ABC):
    """Abstract base class for cloud storage providers."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the cloud provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """
        Upload a file to cloud storage.

        Args:
            local_path: Path to local file
            remote_path: Destination path in cloud storage

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """
        Download a file from cloud storage.

        Args:
            remote_path: Path in cloud storage
            local_path: Destination path for downloaded file

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in cloud storage.

        Args:
            prefix: Optional path prefix to filter by

        Returns:
            List of file paths
        """
        pass

    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """
        Delete a file from cloud storage.

        Args:
            remote_path: Path in cloud storage

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_file_metadata(self, remote_path: str) -> FileMetadata | None:
        """
        Get metadata for a file in cloud storage.

        Args:
            remote_path: Path in cloud storage

        Returns:
            FileMetadata if file exists, None otherwise
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to cloud storage.

        Returns:
            True if connection successful, False otherwise
        """
        pass


class LocalStorageProvider(CloudProvider):
    """Local filesystem provider for testing and local sync."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize local storage provider.

        Args:
            config: Configuration with 'sync_dir' key
        """
        super().__init__(config)
        self.sync_dir = Path(config.get("sync_dir", "/tmp/ptpd_sync"))
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload file by copying to sync directory."""
        try:
            dest = self.sync_dir / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True
        except Exception:
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download file by copying from sync directory."""
        try:
            src = self.sync_dir / remote_path
            if not src.exists():
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_path)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in sync directory."""
        base = self.sync_dir / prefix if prefix else self.sync_dir
        if not base.exists():
            return []

        files = []
        for path in base.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self.sync_dir)
                files.append(str(rel_path))

        return files

    def delete_file(self, remote_path: str) -> bool:
        """Delete file from sync directory."""
        try:
            path = self.sync_dir / remote_path
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False

    def get_file_metadata(self, remote_path: str) -> FileMetadata | None:
        """Get metadata for file in sync directory."""
        path = self.sync_dir / remote_path
        if not path.exists():
            return None

        stat = path.stat()
        checksum = self._calculate_checksum(path)

        return FileMetadata(
            path=remote_path,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=checksum,
        )

    def test_connection(self) -> bool:
        """Test that sync directory is accessible."""
        return self.sync_dir.exists() and self.sync_dir.is_dir()

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class S3Provider(CloudProvider):
    """AWS S3 storage provider (requires boto3)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize S3 provider.

        Args:
            config: Configuration with 'bucket', 'region', 'access_key', 'secret_key'

        Raises:
            ImportError: If boto3 is not installed
        """
        super().__init__(config)

        try:
            import boto3
        except ImportError as err:
            raise ImportError(
                "boto3 is required for S3 sync. Install with: pip install boto3"
            ) from err

        self.bucket = config.get("bucket")
        if not self.bucket:
            raise ValueError("S3 bucket not specified in config")

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            region_name=config.get("region", "us-east-1"),
            aws_access_key_id=config.get("access_key"),
            aws_secret_access_key=config.get("secret_key"),
        )

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload file to S3."""
        try:
            self.s3.upload_file(str(local_path), self.bucket, remote_path)
            return True
        except Exception:
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download file from S3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, remote_path, str(local_path))
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in S3 bucket."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" not in response:
                return []
            return [obj["Key"] for obj in response["Contents"]]
        except Exception:
            return []

    def delete_file(self, remote_path: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception:
            return False

    def get_file_metadata(self, remote_path: str) -> FileMetadata | None:
        """Get metadata for file in S3."""
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return FileMetadata(
                path=remote_path,
                size=response["ContentLength"],
                modified_time=response["LastModified"],
                checksum=response.get("ETag", "").strip('"'),
            )
        except Exception:
            return None

    def test_connection(self) -> bool:
        """Test S3 connection."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


class SyncManager:
    """Manages synchronization between local and cloud storage."""

    def __init__(
        self,
        local_path: Path,
        provider: CloudProvider,
        sync_state_path: Path | None = None,
    ) -> None:
        """
        Initialize the sync manager.

        Args:
            local_path: Local directory to sync
            provider: Cloud storage provider
            sync_state_path: Path to store sync state (default: local_path/.sync_state.json)
        """
        self.local_path = local_path
        self.provider = provider
        self.sync_state_path = sync_state_path or (local_path / ".sync_state.json")
        self.sync_history: list[SyncRecord] = []
        self._load_sync_state()

    def sync_to_cloud(self) -> SyncRecord:
        """
        Sync local files to cloud storage.

        Returns:
            SyncRecord with sync results
        """
        start_time = datetime.now()
        record = SyncRecord(
            timestamp=start_time,
            status=SyncStatus.SYNCING,
            direction="upload",
        )

        try:
            # Get all local files
            local_files = self._get_local_files()

            for local_file in local_files:
                rel_path = local_file.relative_to(self.local_path)
                remote_path = str(rel_path)

                # Check if file needs syncing
                if self._needs_upload(local_file, remote_path):
                    success = self.provider.upload_file(local_file, remote_path)
                    if success:
                        record.files_synced += 1
                        record.bytes_transferred += local_file.stat().st_size
                    else:
                        record.errors.append(f"Failed to upload: {remote_path}")

            record.status = SyncStatus.COMPLETED
        except Exception as e:
            record.status = SyncStatus.FAILED
            record.errors.append(str(e))

        record.duration_seconds = (datetime.now() - start_time).total_seconds()
        self.sync_history.append(record)
        self._save_sync_state()

        return record

    def sync_from_cloud(self) -> SyncRecord:
        """
        Sync cloud files to local storage.

        Returns:
            SyncRecord with sync results
        """
        start_time = datetime.now()
        record = SyncRecord(
            timestamp=start_time,
            status=SyncStatus.SYNCING,
            direction="download",
        )

        try:
            # Get all remote files
            remote_files = self.provider.list_files()

            for remote_path in remote_files:
                local_file = self.local_path / remote_path

                # Check if file needs syncing
                if self._needs_download(remote_path, local_file):
                    success = self.provider.download_file(remote_path, local_file)
                    if success:
                        record.files_synced += 1
                        record.bytes_transferred += local_file.stat().st_size
                    else:
                        record.errors.append(f"Failed to download: {remote_path}")

            record.status = SyncStatus.COMPLETED
        except Exception as e:
            record.status = SyncStatus.FAILED
            record.errors.append(str(e))

        record.duration_seconds = (datetime.now() - start_time).total_seconds()
        self.sync_history.append(record)
        self._save_sync_state()

        return record

    def get_sync_status(self) -> dict[str, Any]:
        """
        Get current sync status.

        Returns:
            Dictionary with sync status information
        """
        last_sync = self.get_last_sync_time()

        return {
            "last_sync": last_sync.isoformat() if last_sync else None,
            "total_syncs": len(self.sync_history),
            "recent_syncs": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "direction": r.direction,
                    "status": r.status,
                    "files_synced": r.files_synced,
                }
                for r in self.sync_history[-5:]
            ],
        }

    def resolve_conflicts(self, strategy: ConflictStrategy) -> dict[str, Any]:
        """
        Resolve sync conflicts using specified strategy.

        Args:
            strategy: Conflict resolution strategy

        Returns:
            Dictionary with conflict resolution results
        """
        conflicts = self._detect_conflicts()
        resolved = {"total": len(conflicts), "resolved": 0, "failed": 0}

        for conflict in conflicts:
            try:
                if strategy == ConflictStrategy.LOCAL_WINS:
                    self.provider.upload_file(conflict["local_path"], conflict["remote_path"])
                elif strategy == ConflictStrategy.REMOTE_WINS:
                    self.provider.download_file(conflict["remote_path"], conflict["local_path"])
                elif strategy == ConflictStrategy.NEWEST_WINS:
                    if conflict["local_newer"]:
                        self.provider.upload_file(conflict["local_path"], conflict["remote_path"])
                    else:
                        self.provider.download_file(conflict["remote_path"], conflict["local_path"])
                resolved["resolved"] += 1
            except Exception:
                resolved["failed"] += 1

        return resolved

    def get_last_sync_time(self) -> datetime | None:
        """
        Get timestamp of last successful sync.

        Returns:
            Datetime of last sync, or None if never synced
        """
        successful_syncs = [r for r in self.sync_history if r.status == SyncStatus.COMPLETED]
        if successful_syncs:
            return successful_syncs[-1].timestamp
        return None

    def enable_auto_sync(self, interval_seconds: int) -> None:
        """
        Enable automatic synchronization.

        Args:
            interval_seconds: Sync interval in seconds

        Note:
            This is a placeholder. Actual implementation would require
            a background task scheduler or daemon.
        """
        # Placeholder - would need actual background task implementation
        self._auto_sync_interval = interval_seconds
        # In production, would start a background thread or use a task scheduler

    def _get_local_files(self) -> list[Path]:
        """Get all files in local directory."""
        files = []
        for path in self.local_path.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                files.append(path)
        return files

    def _needs_upload(self, local_file: Path, remote_path: str) -> bool:
        """Check if local file needs to be uploaded."""
        remote_meta = self.provider.get_file_metadata(remote_path)
        if remote_meta is None:
            return True  # File doesn't exist remotely

        # Compare modification times
        local_mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=timezone.utc)
        return local_mtime > remote_meta.modified_time

    def _needs_download(self, remote_path: str, local_file: Path) -> bool:
        """Check if remote file needs to be downloaded."""
        if not local_file.exists():
            return True  # File doesn't exist locally

        remote_meta = self.provider.get_file_metadata(remote_path)
        if remote_meta is None:
            return False  # Remote file doesn't exist

        # Compare modification times
        local_mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=timezone.utc)
        return remote_meta.modified_time > local_mtime

    def _detect_conflicts(self) -> list[dict[str, Any]]:
        """Detect files with conflicts (modified both locally and remotely)."""
        conflicts = []
        local_files = self._get_local_files()

        for local_file in local_files:
            rel_path = local_file.relative_to(self.local_path)
            remote_path = str(rel_path)

            remote_meta = self.provider.get_file_metadata(remote_path)
            if remote_meta is None:
                continue

            local_mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=timezone.utc)

            # Check if both have been modified
            if local_mtime != remote_meta.modified_time:
                conflicts.append(
                    {
                        "local_path": local_file,
                        "remote_path": remote_path,
                        "local_modified": local_mtime,
                        "remote_modified": remote_meta.modified_time,
                        "local_newer": local_mtime > remote_meta.modified_time,
                    }
                )

        return conflicts

    def _load_sync_state(self) -> None:
        """Load sync state from file."""
        if self.sync_state_path.exists():
            try:
                with open(self.sync_state_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.sync_history = [SyncRecord(**record) for record in data.get("history", [])]
            except Exception:
                self.sync_history = []

    def _save_sync_state(self) -> None:
        """Save sync state to file."""
        try:
            data = {"history": [record.model_dump() for record in self.sync_history[-100:]]}
            self.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sync_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass  # Fail silently for sync state saves
