"""
Print session logger for tracking prints and building process knowledge.

Logs prints with metadata including chemistry, exposure, paper, and results.
Enables learning from past prints to improve future outcomes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4


class PrintResult(str, Enum):
    """Result/quality of a print."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ChemistryUsed:
    """Chemistry used for a print."""

    ferric_oxalate_drops: float = 0.0
    ferric_oxalate_contrast_drops: float = 0.0
    palladium_drops: float = 0.0
    platinum_drops: float = 0.0
    na2_drops: float = 0.0

    # Additional chemistry info
    developer: str = "Potassium Oxalate"
    developer_temperature_f: float = 68.0
    clearing_agent: str = "EDTA"

    @property
    def total_drops(self) -> float:
        """Get total drops used."""
        return (
            self.ferric_oxalate_drops +
            self.ferric_oxalate_contrast_drops +
            self.palladium_drops +
            self.platinum_drops +
            self.na2_drops
        )

    @property
    def platinum_ratio(self) -> float:
        """Get platinum ratio."""
        total_metal = self.palladium_drops + self.platinum_drops
        if total_metal == 0:
            return 0.0
        return self.platinum_drops / total_metal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ferric_oxalate_drops": self.ferric_oxalate_drops,
            "ferric_oxalate_contrast_drops": self.ferric_oxalate_contrast_drops,
            "palladium_drops": self.palladium_drops,
            "platinum_drops": self.platinum_drops,
            "na2_drops": self.na2_drops,
            "total_drops": self.total_drops,
            "platinum_ratio": f"{self.platinum_ratio * 100:.0f}%",
            "developer": self.developer,
            "developer_temperature_f": self.developer_temperature_f,
            "clearing_agent": self.clearing_agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChemistryUsed":
        """Create from dictionary."""
        return cls(
            ferric_oxalate_drops=data.get("ferric_oxalate_drops", 0.0),
            ferric_oxalate_contrast_drops=data.get("ferric_oxalate_contrast_drops", 0.0),
            palladium_drops=data.get("palladium_drops", 0.0),
            platinum_drops=data.get("platinum_drops", 0.0),
            na2_drops=data.get("na2_drops", 0.0),
            developer=data.get("developer", "Potassium Oxalate"),
            developer_temperature_f=data.get("developer_temperature_f", 68.0),
            clearing_agent=data.get("clearing_agent", "EDTA"),
        )


@dataclass
class PrintRecord:
    """Record of a single print."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Image info
    image_name: str = ""
    negative_path: str | None = None

    # Paper info
    paper_type: str = ""
    paper_size: str = ""  # e.g., "8x10"

    # Chemistry
    chemistry: ChemistryUsed = field(default_factory=ChemistryUsed)

    # Exposure
    exposure_time_minutes: float = 0.0
    light_source: str = ""
    uv_unit: str = ""  # e.g., "NuArc 26-1K"

    # Curve used
    curve_name: str | None = None
    curve_file: str | None = None

    # Results
    result: PrintResult = PrintResult.ACCEPTABLE
    dmax_achieved: float | None = None
    dmin_achieved: float | None = None

    # Notes
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # Environmental
    humidity_percent: float | None = None
    temperature_f: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "image_name": self.image_name,
            "negative_path": self.negative_path,
            "paper_type": self.paper_type,
            "paper_size": self.paper_size,
            "chemistry": self.chemistry.to_dict(),
            "exposure_time_minutes": self.exposure_time_minutes,
            "light_source": self.light_source,
            "uv_unit": self.uv_unit,
            "curve_name": self.curve_name,
            "curve_file": self.curve_file,
            "result": self.result.value,
            "dmax_achieved": self.dmax_achieved,
            "dmin_achieved": self.dmin_achieved,
            "notes": self.notes,
            "tags": self.tags,
            "humidity_percent": self.humidity_percent,
            "temperature_f": self.temperature_f,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrintRecord":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            image_name=data.get("image_name", ""),
            negative_path=data.get("negative_path"),
            paper_type=data.get("paper_type", ""),
            paper_size=data.get("paper_size", ""),
            chemistry=ChemistryUsed.from_dict(data.get("chemistry", {})),
            exposure_time_minutes=data.get("exposure_time_minutes", 0.0),
            light_source=data.get("light_source", ""),
            uv_unit=data.get("uv_unit", ""),
            curve_name=data.get("curve_name"),
            curve_file=data.get("curve_file"),
            result=PrintResult(data.get("result", "acceptable")),
            dmax_achieved=data.get("dmax_achieved"),
            dmin_achieved=data.get("dmin_achieved"),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            humidity_percent=data.get("humidity_percent"),
            temperature_f=data.get("temperature_f"),
        )


@dataclass
class PrintSession:
    """A printing session containing multiple prints."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    records: list[PrintRecord] = field(default_factory=list)
    notes: str = ""

    def add_record(self, record: PrintRecord) -> None:
        """Add a print record to the session."""
        self.records.append(record)

    def end_session(self) -> None:
        """Mark session as ended."""
        self.ended_at = datetime.now()

    @property
    def duration_hours(self) -> float | None:
        """Get session duration in hours."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds() / 3600

    @property
    def success_rate(self) -> float:
        """Get percentage of successful prints."""
        if not self.records:
            return 0.0
        successful = sum(
            1 for r in self.records
            if r.result in (PrintResult.EXCELLENT, PrintResult.GOOD, PrintResult.ACCEPTABLE)
        )
        return (successful / len(self.records)) * 100

    def get_statistics(self) -> dict:
        """Get session statistics."""
        if not self.records:
            return {"total_prints": 0}

        results_count: dict[str, int] = {}
        for r in self.records:
            results_count[r.result.value] = results_count.get(r.result.value, 0) + 1

        return {
            "total_prints": len(self.records),
            "results": results_count,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration_hours": round(self.duration_hours or 0, 2),
            "papers_used": list({r.paper_type for r in self.records if r.paper_type}),
            "avg_exposure_minutes": sum(r.exposure_time_minutes for r in self.records) / len(self.records),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "records": [r.to_dict() for r in self.records],
            "notes": self.notes,
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrintSession":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            name=data.get("name", ""),
            started_at=datetime.fromisoformat(data["started_at"]) if "started_at" in data else datetime.now(),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            records=[PrintRecord.from_dict(r) for r in data.get("records", [])],
            notes=data.get("notes", ""),
        )


class SessionLogger:
    """Logger for managing print sessions and records.

    Stores session data in JSON files for persistence.
    """

    def __init__(self, storage_dir: Path | None = None):
        """Initialize session logger.

        Args:
            storage_dir: Directory for storing session files.
                        Defaults to ~/.ptpd/sessions/
        """
        self.storage_dir = storage_dir or Path.home() / ".ptpd" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: PrintSession | None = None

    def start_session(self, name: str = "") -> PrintSession:
        """Start a new printing session.

        Args:
            name: Optional name for the session

        Returns:
            New PrintSession instance
        """
        if not name:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        self._current_session = PrintSession(name=name)
        return self._current_session

    def get_current_session(self) -> PrintSession | None:
        """Get the current active session."""
        return self._current_session

    def log_print(self, record: PrintRecord) -> None:
        """Log a print to the current session.

        Args:
            record: Print record to log
        """
        if not self._current_session:
            self.start_session()
        self._current_session.add_record(record)
        self._auto_save()

    def end_session(self) -> PrintSession | None:
        """End the current session and save.

        Returns:
            The ended session, or None if no session active
        """
        if self._current_session is None:
            return None

        self._current_session.end_session()
        self.save_session(self._current_session)

        session = self._current_session
        self._current_session = None
        return session

    def save_session(self, session: PrintSession) -> Path:
        """Save a session to file.

        Args:
            session: Session to save

        Returns:
            Path to saved file
        """
        filename = f"session_{session.started_at.strftime('%Y%m%d_%H%M%S')}_{session.id.hex[:8]}.json"
        filepath = self.storage_dir / filename

        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        return filepath

    def load_session(self, filepath: Path) -> PrintSession:
        """Load a session from file.

        Args:
            filepath: Path to session file

        Returns:
            Loaded PrintSession
        """
        with open(filepath) as f:
            data = json.load(f)

        return PrintSession.from_dict(data)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        sessions = []
        files = sorted(self.storage_dir.glob("session_*.json"), reverse=True)

        for filepath in files[:limit]:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                sessions.append({
                    "filepath": str(filepath),
                    "name": data.get("name", ""),
                    "started_at": data.get("started_at"),
                    "total_prints": len(data.get("records", [])),
                })
            except Exception as e:
                import logging
                logging.warning(f"Failed to list session {filepath}: {e}")
                continue

        return sessions

    def search_records(
        self,
        paper_type: str | None = None,
        result: PrintResult | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[PrintRecord]:
        """Search print records across all sessions.

        Args:
            paper_type: Filter by paper type
            result: Filter by result
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            List of matching print records
        """
        records = []

        for filepath in self.storage_dir.glob("session_*.json"):
            try:
                session = self.load_session(filepath)
                for record in session.records:
                    if paper_type and record.paper_type != paper_type:
                        continue
                    if result and record.result != result:
                        continue
                    if tags and not any(t in record.tags for t in tags):
                        continue

                    records.append(record)

                    if len(records) >= limit:
                        return records
            except Exception as e:
                import logging
                logging.warning(f"Failed to search session {filepath}: {e}")
                continue

        return records

    def get_paper_statistics(self) -> dict[str, dict]:
        """Get statistics grouped by paper type.

        Returns:
            Dictionary of paper types with their statistics
        """
        stats: dict[str, dict[str, Any]] = {}

        for filepath in self.storage_dir.glob("session_*.json"):
            try:
                session = self.load_session(filepath)
                for record in session.records:
                    if not record.paper_type:
                        continue

                    if record.paper_type not in stats:
                        stats[record.paper_type] = {
                            "total_prints": 0,
                            "excellent": 0,
                            "good": 0,
                            "failed": 0,
                            "avg_exposure": [],
                        }

                    stats[record.paper_type]["total_prints"] += 1
                    if record.result == PrintResult.EXCELLENT:
                        stats[record.paper_type]["excellent"] += 1
                    elif record.result == PrintResult.GOOD:
                        stats[record.paper_type]["good"] += 1
                    elif record.result == PrintResult.FAILED:
                        stats[record.paper_type]["failed"] += 1

                    if record.exposure_time_minutes > 0:
                        stats[record.paper_type]["avg_exposure"].append(record.exposure_time_minutes)
            except Exception as e:
                import logging
                logging.warning(f"Failed to get stats for session {filepath}: {e}")
                continue

        # Calculate averages
        for paper in stats:
            exposures_list = cast(list[float], stats[paper]["avg_exposure"])
            stats[paper]["avg_exposure"] = sum(exposures_list) / len(exposures_list) if exposures_list else 0

        return stats

    def _auto_save(self) -> None:
        """Auto-save current session."""
        if self._current_session and len(self._current_session.records) > 0:
            self.save_session(self._current_session)
