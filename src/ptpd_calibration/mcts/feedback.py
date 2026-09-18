"""
Persistent store for real measurement feedback (SCI-06).

``POST /api/mcts/feedback`` used to log and discard the measured curve a
printer sent back. This module gives those measurements a durable home: an
append-only JSON Lines file whose location comes from ``MCTSSettings`` so it
can be redirected with ``PTPD_MCTS_FEEDBACK_PATH``. Every record carries
``provenance="measured"`` so downstream consumers can never confuse a real
densitometer reading with simulator output.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from ptpd_calibration.mcts.config import MCTSSettings

logger = logging.getLogger(__name__)

FEEDBACK_FILENAME = "feedback.jsonl"


class FeedbackRecord(BaseModel):
    """One measured-print feedback entry as persisted on disk."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: dict[str, float]
    measured_curve: list[float] = Field(min_length=1)
    quality_rating: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
    provenance: Literal["measured"] = "measured"


def resolve_feedback_path(settings: MCTSSettings | None = None) -> Path:
    """Return the feedback file path from settings.

    ``MCTSSettings.feedback_path`` wins when set; otherwise the file lives next
    to the checkpoints at ``<checkpoint_dir>/feedback.jsonl``.
    """
    resolved_settings = settings or MCTSSettings()
    if resolved_settings.feedback_path:
        return Path(resolved_settings.feedback_path)
    return Path(resolved_settings.checkpoint_dir) / FEEDBACK_FILENAME


class FeedbackStore:
    """Append-only JSON Lines store for :class:`FeedbackRecord` entries."""

    def __init__(
        self,
        path: Path | str | None = None,
        settings: MCTSSettings | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else resolve_feedback_path(settings)
        self._lock = threading.Lock()
        logger.debug("FeedbackStore bound to %s", self._path)

    @property
    def path(self) -> Path:
        """Location of the JSON Lines file."""
        return self._path

    def append(self, record: FeedbackRecord) -> FeedbackRecord:
        """Persist ``record`` and return it (with its server-assigned id)."""
        line = record.model_dump_json()
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        logger.debug(
            "Stored feedback %s (%d curve points, rating %.2f)",
            record.id,
            len(record.measured_curve),
            record.quality_rating,
        )
        return record

    def list(self, limit: int | None = None) -> list[FeedbackRecord]:
        """Read stored records in insertion order, skipping malformed lines.

        Args:
            limit: When given, return only the most recent ``limit`` records.
        """
        if not self._path.exists():
            return []
        records: list[FeedbackRecord] = []
        with self._lock, open(self._path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(FeedbackRecord.model_validate_json(stripped))
                except (ValidationError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Skipping malformed feedback line %d in %s: %s",
                        line_number,
                        self._path,
                        exc,
                    )
        if limit is not None:
            records = records[-limit:] if limit > 0 else []
        logger.debug("Read %d feedback records from %s", len(records), self._path)
        return records

    def count(self) -> int:
        """Number of well-formed records currently stored.

        Deliberately not ``__len__``: a store must never be falsy, otherwise an
        ``injected_store or default`` expression would silently discard an
        empty injected store.
        """
        return len(self.list())
