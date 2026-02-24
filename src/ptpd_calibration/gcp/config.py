"""
Configuration management for Google Cloud Platform integration.
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class GCPConfig(BaseSettings):
    """
    Configuration settings for Google Cloud Platform.

    Attributes:
        project_id: The GCP Project ID.
        region: The GCP region for Vertex AI and other regional services.
        bucket_name: The GCS bucket name for storage.
        service_account_path: Optional path to a service account JSON key file.
        credentials_json: Optional JSON string of credentials (for CI/CD).
        staging_dir: Local directory for staging files before upload (default: ./staging).
    """

    project_id: str = Field(
        ..., description="GCP Project ID", validation_alias="PTPD_GCP_PROJECT_ID"
    )

    region: str = Field("us-central1", description="GCP Region", validation_alias="PTPD_GCP_REGION")

    bucket_name: str = Field(..., description="GCS Bucket Name", validation_alias="PTPD_GCS_BUCKET")

    service_account_path: Path | None = Field(
        None, description="Path to Service Account JSON Key", validation_alias="PTPD_GCP_SA_PATH"
    )

    staging_dir: Path = Field(
        Path("staging"), description="Local staging directory", validation_alias="PTPD_STAGING_DIR"
    )

    force_local_storage: bool = Field(
        False,
        description="Force usage of local filesystem backend",
        validation_alias="PTPD_FORCE_LOCAL_STORAGE",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    @property
    def storage_bucket_uri(self) -> str:
        """Returns the full gs:// URI for the bucket."""
        return f"gs://{self.bucket_name}"


@lru_cache
def get_gcp_config() -> GCPConfig:
    """
    Returns a cached instance of GCPConfig.

    Raises:
        ValidationError: If required environment variables are missing.
    """
    return GCPConfig()


def try_get_gcp_config() -> GCPConfig | None:
    """Return a GCPConfig if one can be built, or ``None`` otherwise.

    Use this when GCP is optional and the caller can fall back to
    local-only behaviour.
    """
    try:
        return get_gcp_config()
    except Exception as e:
        logger.warning("Failed to load GCP Config: %s. GCP features will be disabled.", e)
        return None
