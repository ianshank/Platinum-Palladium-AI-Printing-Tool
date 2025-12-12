"""
Calibration history database for tracking calibration sessions and iterations.

Provides persistent storage for calibration sessions including print analysis
results, curve adjustments, and session metadata.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class CalibrationRecord(BaseModel):
    """
    Record of a single calibration iteration.

    Stores the measurements and adjustments for one test print
    in a calibration session.
    """

    # Identifiers
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(...)
    iteration_number: int = Field(default=1, ge=1)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Paper and chemistry
    paper_type: str = Field(..., min_length=1)
    chemistry_ratio: Optional[str] = Field(default=None)  # e.g., "6Pd:2Pt"
    fo_ratio: Optional[str] = Field(default=None)  # e.g., "7:1"

    # Exposure
    exposure_time: str = Field(...)  # e.g., "3:15" for 3 min 15 sec
    uv_source: Optional[str] = Field(default=None)

    # Curve info
    base_curve_name: str = Field(default="")
    curve_version: str = Field(default="v1")

    # Measured densities
    highlight_density: Optional[float] = Field(default=None, ge=0.0)
    midtone_density: Optional[float] = Field(default=None, ge=0.0)
    shadow_density: Optional[float] = Field(default=None, ge=0.0)
    tonal_range: Optional[float] = Field(default=None, ge=0.0)
    midtone_separation: Optional[float] = Field(default=None, ge=0.0)

    # Applied adjustments (what was changed for this iteration)
    highlight_adj: float = Field(default=0.0, ge=-0.20, le=0.20)
    midtone_adj: float = Field(default=0.0, ge=-0.20, le=0.20)
    shadow_adj: float = Field(default=0.0, ge=-0.20, le=0.20)

    # Recommended adjustments (for next iteration)
    recommended_highlight_adj: float = Field(default=0.0)
    recommended_midtone_adj: float = Field(default=0.0)
    recommended_shadow_adj: float = Field(default=0.0)

    # Files
    scan_path: Optional[str] = Field(default=None)
    result_curve_path: Optional[str] = Field(default=None)

    # Metadata
    notes: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "session_id", mode="before")
    @classmethod
    def convert_uuid(cls, v: Any) -> Optional[UUID]:
        """Convert string UUIDs to UUID objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return UUID(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = self.model_dump()
        data["id"] = str(self.id)
        data["session_id"] = str(self.session_id)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class CalibrationSessionRecord(BaseModel):
    """
    Record of a complete calibration session.

    Groups multiple calibration iterations for a paper/chemistry combination.
    """

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Session details
    name: str = Field(default="")
    paper_type: str = Field(..., min_length=1)
    chemistry: Optional[str] = Field(default=None)
    target_curve_name: Optional[str] = Field(default=None)

    # Status
    is_complete: bool = Field(default=False)
    iteration_count: int = Field(default=0)

    # Cumulative adjustments
    total_highlight_adj: float = Field(default=0.0)
    total_midtone_adj: float = Field(default=0.0)
    total_shadow_adj: float = Field(default=0.0)

    # Final results
    final_curve_path: Optional[str] = Field(default=None)
    final_dmax: Optional[float] = Field(default=None)
    final_dmin: Optional[float] = Field(default=None)

    # Metadata
    notes: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def convert_uuid(cls, v: Any) -> Optional[UUID]:
        """Convert string UUIDs to UUID objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return UUID(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = self.model_dump()
        data["id"] = str(self.id)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data


class CalibrationDatabase:
    """SQLite-based calibration history database."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the calibration database.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            db_path = Path.home() / ".ptpd" / "calibration.db"
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Create calibration sessions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                -- Session details
                name TEXT,
                paper_type TEXT NOT NULL,
                chemistry TEXT,
                target_curve_name TEXT,

                -- Status
                is_complete INTEGER NOT NULL DEFAULT 0,
                iteration_count INTEGER NOT NULL DEFAULT 0,

                -- Cumulative adjustments
                total_highlight_adj REAL NOT NULL DEFAULT 0,
                total_midtone_adj REAL NOT NULL DEFAULT 0,
                total_shadow_adj REAL NOT NULL DEFAULT 0,

                -- Final results
                final_curve_path TEXT,
                final_dmax REAL,
                final_dmin REAL,

                -- Metadata
                notes TEXT,
                tags TEXT,
                metadata TEXT
            )
            """
        )

        # Create calibration records table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_records (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                iteration_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,

                -- Paper and chemistry
                paper_type TEXT NOT NULL,
                chemistry_ratio TEXT,
                fo_ratio TEXT,

                -- Exposure
                exposure_time TEXT NOT NULL,
                uv_source TEXT,

                -- Curve
                base_curve_name TEXT,
                curve_version TEXT,

                -- Measured densities
                highlight_density REAL,
                midtone_density REAL,
                shadow_density REAL,
                tonal_range REAL,
                midtone_separation REAL,

                -- Applied adjustments
                highlight_adj REAL NOT NULL DEFAULT 0,
                midtone_adj REAL NOT NULL DEFAULT 0,
                shadow_adj REAL NOT NULL DEFAULT 0,

                -- Recommended adjustments
                recommended_highlight_adj REAL NOT NULL DEFAULT 0,
                recommended_midtone_adj REAL NOT NULL DEFAULT 0,
                recommended_shadow_adj REAL NOT NULL DEFAULT 0,

                -- Files
                scan_path TEXT,
                result_curve_path TEXT,

                -- Metadata
                notes TEXT,
                metadata TEXT,

                FOREIGN KEY (session_id) REFERENCES calibration_sessions(id)
            )
            """
        )

        # Create indices
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_id ON calibration_records(session_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_type ON calibration_records(paper_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON calibration_records(timestamp DESC)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_paper ON calibration_sessions(paper_type)"
        )

        self.conn.commit()

    # Session methods

    def create_session(
        self,
        paper_type: str,
        chemistry: Optional[str] = None,
        name: Optional[str] = None,
        target_curve_name: Optional[str] = None
    ) -> CalibrationSessionRecord:
        """
        Create a new calibration session.

        Args:
            paper_type: Paper being calibrated
            chemistry: Chemistry description
            name: Optional session name
            target_curve_name: Target curve for calibration

        Returns:
            Created session record
        """
        session = CalibrationSessionRecord(
            paper_type=paper_type,
            chemistry=chemistry,
            name=name or f"Calibration - {paper_type}",
            target_curve_name=target_curve_name
        )

        self.conn.execute(
            """
            INSERT INTO calibration_sessions VALUES (
                :id, :created_at, :updated_at,
                :name, :paper_type, :chemistry, :target_curve_name,
                :is_complete, :iteration_count,
                :total_highlight_adj, :total_midtone_adj, :total_shadow_adj,
                :final_curve_path, :final_dmax, :final_dmin,
                :notes, :tags, :metadata
            )
            """,
            {
                **session.to_dict(),
                "tags": json.dumps(session.tags),
                "metadata": json.dumps(session.metadata),
            }
        )
        self.conn.commit()

        return session

    def get_session(self, session_id: UUID) -> Optional[CalibrationSessionRecord]:
        """Get a session by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM calibration_sessions WHERE id = ?",
            (str(session_id),)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_session(row)

    def update_session(
        self,
        session_id: UUID,
        updates: dict[str, Any]
    ) -> bool:
        """Update a calibration session."""
        existing = self.get_session(session_id)
        if existing is None:
            return False

        # Always update updated_at
        updates["updated_at"] = datetime.now().isoformat()

        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ("tags", "metadata"):
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif isinstance(value, datetime):
                set_clauses.append(f"{key} = ?")
                values.append(value.isoformat())
            elif isinstance(value, bool):
                set_clauses.append(f"{key} = ?")
                values.append(1 if value else 0)
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)

        values.append(str(session_id))

        self.conn.execute(
            f"UPDATE calibration_sessions SET {', '.join(set_clauses)} WHERE id = ?",
            values
        )
        self.conn.commit()

        return True

    def list_sessions(
        self,
        paper_type: Optional[str] = None,
        is_complete: Optional[bool] = None,
        limit: int = 100
    ) -> list[CalibrationSessionRecord]:
        """
        List calibration sessions with optional filters.

        Args:
            paper_type: Filter by paper type
            is_complete: Filter by completion status
            limit: Maximum number to return

        Returns:
            List of session records
        """
        where_clauses = []
        values: list[Any] = []

        if paper_type:
            where_clauses.append("paper_type = ?")
            values.append(paper_type)

        if is_complete is not None:
            where_clauses.append("is_complete = ?")
            values.append(1 if is_complete else 0)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        values.append(limit)

        cursor = self.conn.execute(
            f"""
            SELECT * FROM calibration_sessions
            WHERE {where_sql}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            values
        )

        return [self._row_to_session(row) for row in cursor.fetchall()]

    def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its records."""
        # Delete records first
        self.conn.execute(
            "DELETE FROM calibration_records WHERE session_id = ?",
            (str(session_id),)
        )

        cursor = self.conn.execute(
            "DELETE FROM calibration_sessions WHERE id = ?",
            (str(session_id),)
        )
        self.conn.commit()

        return cursor.rowcount > 0

    # Record methods

    def add_record(self, record: CalibrationRecord) -> UUID:
        """
        Add a calibration record to the database.

        Also updates the parent session's iteration count and totals.

        Args:
            record: CalibrationRecord to add

        Returns:
            UUID of the added record
        """
        data = record.to_dict()

        self.conn.execute(
            """
            INSERT INTO calibration_records VALUES (
                :id, :session_id, :iteration_number, :timestamp,
                :paper_type, :chemistry_ratio, :fo_ratio,
                :exposure_time, :uv_source,
                :base_curve_name, :curve_version,
                :highlight_density, :midtone_density, :shadow_density,
                :tonal_range, :midtone_separation,
                :highlight_adj, :midtone_adj, :shadow_adj,
                :recommended_highlight_adj, :recommended_midtone_adj,
                :recommended_shadow_adj,
                :scan_path, :result_curve_path,
                :notes, :metadata
            )
            """,
            {
                **data,
                "metadata": json.dumps(record.metadata),
            }
        )

        # Update session
        self.conn.execute(
            """
            UPDATE calibration_sessions SET
                iteration_count = iteration_count + 1,
                total_highlight_adj = total_highlight_adj + ?,
                total_midtone_adj = total_midtone_adj + ?,
                total_shadow_adj = total_shadow_adj + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                record.highlight_adj,
                record.midtone_adj,
                record.shadow_adj,
                datetime.now().isoformat(),
                str(record.session_id)
            )
        )

        self.conn.commit()
        return record.id

    def get_record(self, record_id: UUID) -> Optional[CalibrationRecord]:
        """Get a single calibration record by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM calibration_records WHERE id = ?",
            (str(record_id),)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_session_records(
        self,
        session_id: UUID
    ) -> list[CalibrationRecord]:
        """Get all records for a session."""
        cursor = self.conn.execute(
            """
            SELECT * FROM calibration_records
            WHERE session_id = ?
            ORDER BY iteration_number ASC
            """,
            (str(session_id),)
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_records_by_paper(
        self,
        paper_type: str,
        limit: int = 100
    ) -> list[CalibrationRecord]:
        """Get recent records for a specific paper type."""
        cursor = self.conn.execute(
            """
            SELECT * FROM calibration_records
            WHERE paper_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (paper_type, limit)
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics about calibration history."""
        stats: dict[str, Any] = {}

        # Total counts
        cursor = self.conn.execute(
            "SELECT COUNT(*) as count FROM calibration_sessions"
        )
        stats["total_sessions"] = cursor.fetchone()["count"]

        cursor = self.conn.execute(
            "SELECT COUNT(*) as count FROM calibration_records"
        )
        stats["total_records"] = cursor.fetchone()["count"]

        # Completion rate
        cursor = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN is_complete = 1 THEN 1 ELSE 0 END) as complete,
                SUM(CASE WHEN is_complete = 0 THEN 1 ELSE 0 END) as incomplete
            FROM calibration_sessions
            """
        )
        row = cursor.fetchone()
        stats["complete_sessions"] = row["complete"] or 0
        stats["incomplete_sessions"] = row["incomplete"] or 0

        # Paper type distribution
        cursor = self.conn.execute(
            """
            SELECT paper_type, COUNT(*) as count
            FROM calibration_sessions
            GROUP BY paper_type
            ORDER BY count DESC
            """
        )
        stats["papers"] = {row["paper_type"]: row["count"] for row in cursor}

        # Average iterations per session
        cursor = self.conn.execute(
            """
            SELECT AVG(iteration_count) as avg_iterations
            FROM calibration_sessions
            WHERE iteration_count > 0
            """
        )
        stats["avg_iterations"] = cursor.fetchone()["avg_iterations"]

        # Average adjustments
        cursor = self.conn.execute(
            """
            SELECT
                AVG(ABS(total_highlight_adj)) as avg_highlight,
                AVG(ABS(total_midtone_adj)) as avg_midtone,
                AVG(ABS(total_shadow_adj)) as avg_shadow
            FROM calibration_sessions
            WHERE is_complete = 1
            """
        )
        row = cursor.fetchone()
        stats["avg_adjustments"] = {
            "highlight": row["avg_highlight"],
            "midtone": row["avg_midtone"],
            "shadow": row["avg_shadow"]
        }

        return stats

    def _row_to_session(self, row: sqlite3.Row) -> CalibrationSessionRecord:
        """Convert database row to session record."""
        data = dict(row)
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []
        data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
        data["is_complete"] = bool(data["is_complete"])
        return CalibrationSessionRecord(**data)

    def _row_to_record(self, row: sqlite3.Row) -> CalibrationRecord:
        """Convert database row to calibration record."""
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
        return CalibrationRecord(**data)

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "CalibrationDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
