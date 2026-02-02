"""
Workflow persistence for checkpoint and resume functionality.

Provides durable storage for workflow state, enabling recovery from
failures and long-running workflow management.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.agents.logging import get_agent_logger


class PersistenceSettings(BaseSettings):
    """Settings for workflow persistence."""

    model_config = SettingsConfigDict(env_prefix="PTPD_PERSISTENCE_")

    # Storage
    checkpoint_dir: Path = Field(
        default=Path.home() / ".ptpd" / "checkpoints",
        description="Directory for workflow checkpoints",
    )
    max_checkpoints: int = Field(
        default=100, ge=10, le=10000, description="Maximum checkpoints to retain"
    )

    # Checkpoint behavior
    auto_checkpoint: bool = Field(
        default=True, description="Automatically checkpoint after each step"
    )
    checkpoint_interval_seconds: float = Field(
        default=60.0, ge=10.0, le=600.0, description="Minimum interval between checkpoints"
    )

    # Cleanup
    cleanup_completed: bool = Field(
        default=True, description="Clean up checkpoints for completed workflows"
    )
    retention_hours: float = Field(
        default=24.0, ge=1.0, le=720.0, description="Hours to retain completed workflow checkpoints"
    )


class WorkflowState(str, Enum):
    """State of a persisted workflow."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskCheckpoint:
    """Checkpoint for a single task."""

    task_id: str
    status: str
    started_at: float | None = None
    completed_at: float | None = None
    result: Any | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for a complete workflow."""

    workflow_id: str
    name: str
    state: WorkflowState
    created_at: float
    updated_at: float
    current_task_index: int = 0
    tasks: list[TaskCheckpoint] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowCheckpoint":
        """Create from dictionary."""
        data["state"] = WorkflowState(data["state"])
        data["tasks"] = [TaskCheckpoint(**t) for t in data.get("tasks", [])]
        return cls(**data)


class WorkflowPersistence:
    """
    Persistent storage for workflow state.

    Enables checkpoint and resume functionality for long-running workflows,
    providing durability across process restarts.

    Example:
        ```python
        persistence = WorkflowPersistence()

        # Save checkpoint
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-123",
            name="calibration",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(checkpoint)

        # Resume later
        checkpoint = persistence.load_checkpoint("wf-123")
        if checkpoint:
            # Resume from checkpoint
            pass
        ```
    """

    def __init__(self, settings: PersistenceSettings | None = None):
        """
        Initialize the persistence layer.

        Args:
            settings: Persistence settings.
        """
        self.settings = settings or PersistenceSettings()
        self._logger = get_agent_logger()
        self._last_checkpoint_time: dict[str, float] = {}

        # Ensure checkpoint directory exists
        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, workflow_id: str) -> Path:
        """Get path for a workflow checkpoint file."""
        safe_id = workflow_id.replace("/", "_").replace("\\", "_")
        return self.settings.checkpoint_dir / f"{safe_id}.json"

    def save_checkpoint(
        self,
        checkpoint: WorkflowCheckpoint,
        force: bool = False,
    ) -> bool:
        """
        Save a workflow checkpoint.

        Args:
            checkpoint: Checkpoint to save.
            force: Force save even if interval hasn't passed.

        Returns:
            True if checkpoint was saved, False if skipped.
        """
        # Check checkpoint interval
        if not force and not self._should_checkpoint(checkpoint.workflow_id):
            return False

        checkpoint.updated_at = time.time()
        path = self._get_checkpoint_path(checkpoint.workflow_id)

        try:
            # Write atomically using temp file
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

            # Atomic rename
            temp_path.rename(path)

            self._last_checkpoint_time[checkpoint.workflow_id] = time.time()

            self._logger.debug(
                f"Saved checkpoint: {checkpoint.workflow_id}",
                data={
                    "workflow_id": checkpoint.workflow_id,
                    "state": checkpoint.state.value,
                    "task_index": checkpoint.current_task_index,
                },
            )

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return True

        except Exception as e:
            self._logger.error(
                f"Failed to save checkpoint: {checkpoint.workflow_id}",
                error=str(e),
            )
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """
        Load a workflow checkpoint.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            WorkflowCheckpoint if found, None otherwise.
        """
        path = self._get_checkpoint_path(workflow_id)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return WorkflowCheckpoint.from_dict(data)

        except Exception as e:
            self._logger.error(
                f"Failed to load checkpoint: {workflow_id}",
                error=str(e),
            )
            return None

    def delete_checkpoint(self, workflow_id: str) -> bool:
        """
        Delete a workflow checkpoint.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            True if deleted, False if not found.
        """
        path = self._get_checkpoint_path(workflow_id)

        if not path.exists():
            return False

        try:
            path.unlink()
            self._last_checkpoint_time.pop(workflow_id, None)

            self._logger.debug(f"Deleted checkpoint: {workflow_id}")
            return True

        except Exception as e:
            self._logger.error(
                f"Failed to delete checkpoint: {workflow_id}",
                error=str(e),
            )
            return False

    def list_checkpoints(
        self,
        state: WorkflowState | None = None,
    ) -> list[WorkflowCheckpoint]:
        """
        List all checkpoints, optionally filtered by state.

        Args:
            state: Filter by workflow state.

        Returns:
            List of checkpoints.
        """
        checkpoints = []

        for path in self.settings.checkpoint_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                checkpoint = WorkflowCheckpoint.from_dict(data)

                if state is None or checkpoint.state == state:
                    checkpoints.append(checkpoint)

            except Exception as e:
                self._logger.warning(
                    f"Failed to load checkpoint: {path}",
                    error=str(e),
                )

        # Sort by updated_at descending
        checkpoints.sort(key=lambda c: c.updated_at, reverse=True)
        return checkpoints

    def list_incomplete_workflows(self) -> list[WorkflowCheckpoint]:
        """
        List workflows that can be resumed.

        Returns:
            List of incomplete workflow checkpoints.
        """
        incomplete_states = {
            WorkflowState.PENDING,
            WorkflowState.RUNNING,
            WorkflowState.PAUSED,
        }

        return [
            cp
            for cp in self.list_checkpoints()
            if cp.state in incomplete_states
        ]

    def update_task_status(
        self,
        workflow_id: str,
        task_id: str,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> bool:
        """
        Update the status of a task in a checkpoint.

        Args:
            workflow_id: ID of the workflow.
            task_id: ID of the task.
            status: New task status.
            result: Task result if completed.
            error: Error message if failed.

        Returns:
            True if updated successfully.
        """
        checkpoint = self.load_checkpoint(workflow_id)
        if not checkpoint:
            return False

        # Find and update task
        for task in checkpoint.tasks:
            if task.task_id == task_id:
                task.status = status
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if status == "completed" or status == "failed":
                    task.completed_at = time.time()
                break

        return self.save_checkpoint(checkpoint, force=True)

    def advance_workflow(
        self,
        workflow_id: str,
        new_state: WorkflowState | None = None,
    ) -> bool:
        """
        Advance workflow to next task.

        Args:
            workflow_id: ID of the workflow.
            new_state: Optional new workflow state.

        Returns:
            True if advanced successfully.
        """
        checkpoint = self.load_checkpoint(workflow_id)
        if not checkpoint:
            return False

        checkpoint.current_task_index += 1

        if new_state:
            checkpoint.state = new_state

        return self.save_checkpoint(checkpoint, force=True)

    def mark_completed(self, workflow_id: str) -> bool:
        """
        Mark a workflow as completed.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            True if marked successfully.
        """
        checkpoint = self.load_checkpoint(workflow_id)
        if not checkpoint:
            return False

        checkpoint.state = WorkflowState.COMPLETED

        saved = self.save_checkpoint(checkpoint, force=True)

        # Optionally clean up
        if saved and self.settings.cleanup_completed:
            # Don't delete immediately, let retention handle it
            pass

        return saved

    def mark_failed(self, workflow_id: str, error: str) -> bool:
        """
        Mark a workflow as failed.

        Args:
            workflow_id: ID of the workflow.
            error: Error message.

        Returns:
            True if marked successfully.
        """
        checkpoint = self.load_checkpoint(workflow_id)
        if not checkpoint:
            return False

        checkpoint.state = WorkflowState.FAILED
        checkpoint.error = error

        return self.save_checkpoint(checkpoint, force=True)

    def _should_checkpoint(self, workflow_id: str) -> bool:
        """Check if enough time has passed for a new checkpoint."""
        if not self.settings.auto_checkpoint:
            return True

        last_time = self._last_checkpoint_time.get(workflow_id, 0)
        return (time.time() - last_time) >= self.settings.checkpoint_interval_seconds

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        checkpoints = self.list_checkpoints()

        # Check count limit
        if len(checkpoints) > self.settings.max_checkpoints:
            # Remove oldest checkpoints beyond limit
            for checkpoint in checkpoints[self.settings.max_checkpoints :]:
                self.delete_checkpoint(checkpoint.workflow_id)

        # Check retention for completed workflows
        if self.settings.cleanup_completed:
            retention_seconds = self.settings.retention_hours * 3600
            current_time = time.time()

            completed_states = {WorkflowState.COMPLETED, WorkflowState.CANCELLED}

            for checkpoint in checkpoints:
                if checkpoint.state in completed_states:
                    age = current_time - checkpoint.updated_at
                    if age > retention_seconds:
                        self.delete_checkpoint(checkpoint.workflow_id)


# Global persistence instance
_persistence: WorkflowPersistence | None = None


def get_persistence() -> WorkflowPersistence:
    """Get the global persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = WorkflowPersistence()
    return _persistence


def create_workflow_checkpoint(
    workflow_id: str,
    name: str,
    tasks: list[dict] | None = None,
    context: dict | None = None,
    metadata: dict | None = None,
) -> WorkflowCheckpoint:
    """
    Create a new workflow checkpoint.

    Args:
        workflow_id: Unique workflow identifier.
        name: Workflow name.
        tasks: Optional list of task definitions.
        context: Optional workflow context.
        metadata: Optional metadata.

    Returns:
        New WorkflowCheckpoint instance.
    """
    current_time = time.time()

    task_checkpoints = []
    if tasks:
        for task in tasks:
            task_checkpoints.append(
                TaskCheckpoint(
                    task_id=task.get("id", f"task-{len(task_checkpoints)}"),
                    status="pending",
                    metadata=task.get("metadata", {}),
                )
            )

    return WorkflowCheckpoint(
        workflow_id=workflow_id,
        name=name,
        state=WorkflowState.PENDING,
        created_at=current_time,
        updated_at=current_time,
        tasks=task_checkpoints,
        context=context or {},
        metadata=metadata or {},
    )
