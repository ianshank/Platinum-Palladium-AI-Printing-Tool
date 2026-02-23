"""
Vertex AI integration wrapper.
"""

import logging
from typing import Any

try:
    from google.cloud import aiplatform

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

from ptpd_calibration.gcp.config import GCPConfig

logger = logging.getLogger(__name__)


class VertexClient:
    """Wrapper for Vertex AI operations."""

    def __init__(self, config: GCPConfig):
        if not VERTEX_AVAILABLE:
            raise ImportError("google-cloud-aiplatform is required for VertexClient")

        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the Vertex AI SDK."""
        if self._initialized:
            return

        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            staging_bucket=self.config.storage_bucket_uri,
        )
        self._initialized = True
        logger.info(
            f"Initialized Vertex AI for project {self.config.project_id} in {self.config.region}"
        )

    def get_model(self, model_name: str) -> Any:
        """
        Retrieve a model from the Vertex AI Model Registry.

        Args:
            model_name: Fully-qualified model resource name or display name.
                        For versioned models, use the resource name that
                        includes the version (e.g.
                        ``projects/.../models/{id}@{version}``).

        Returns:
            The Vertex AI Model resource.
        """
        self.initialize()
        return aiplatform.Model(model_name=model_name)

    def submit_custom_job(
        self,
        display_name: str,
        container_uri: str,
        args: list[str] | None = None,
        machine_type: str = "n1-standard-4",
        replica_count: int = 1,
    ) -> Any:
        """Submit a Custom Container Training Job to Vertex AI.

        Args:
            display_name: Human-readable name for the training job.
            container_uri: URI of the training container image in
                Artifact Registry or Container Registry.
            args: Command-line arguments passed to the container entrypoint.
                If ``None``, no additional arguments are passed.
            machine_type: The machine type to use for each worker replica
                (for example, ``"n1-standard-4"``).
            replica_count: Number of worker replicas to launch for the job.

        Returns:
            The created and submitted CustomContainerTrainingJob instance.
        """
        self.initialize()
        try:
            job = aiplatform.CustomContainerTrainingJob(
                display_name=display_name,
                container_uri=container_uri,
            )
            job.run(
                args=args or [],
                machine_type=machine_type,
                replica_count=replica_count,
            )
        except Exception as exc:
            logger.exception("Failed to submit custom job to Vertex AI: %s", exc)
            raise
        return job
