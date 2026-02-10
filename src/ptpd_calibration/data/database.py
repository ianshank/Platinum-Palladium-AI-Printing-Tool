"""
Print database with searchable metadata using SQLite.

Provides persistent storage for print records with full-text search,
filtering, and backup/restore capabilities.
"""

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class PrintRecord(BaseModel):
    """Complete print record with metadata, results, and tags."""

    # Identifiers
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=256)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Paper information
    paper_type: str = Field(..., min_length=1)
    paper_weight: int | None = Field(default=None, ge=50, le=1000)
    paper_sizing: str | None = Field(default=None)

    # Chemistry
    chemistry_type: str = Field(default="platinum_palladium")
    metal_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    contrast_agent: str | None = Field(default=None)
    contrast_amount: float = Field(default=0.0, ge=0.0)
    developer: str | None = Field(default=None)

    # Process parameters
    exposure_time: float = Field(..., ge=0.0)
    uv_source: str | None = Field(default=None)
    humidity: float | None = Field(default=None, ge=0.0, le=100.0)
    temperature: float | None = Field(default=None, ge=-20.0, le=50.0)

    # Results
    dmin: float | None = Field(default=None, ge=0.0)
    dmax: float | None = Field(default=None, ge=0.0)
    density_range: float | None = Field(default=None, ge=0.0)
    overall_quality: float = Field(default=1.0, ge=0.0, le=1.0)

    # Recipe reference
    recipe_id: UUID | None = Field(default=None)
    curve_id: UUID | None = Field(default=None)

    # Image paths (stored as JSON list of strings)
    image_paths: list[str] = Field(default_factory=list)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None)

    # Custom metadata (JSON serializable dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "recipe_id", "curve_id", mode="before")
    @classmethod
    def convert_uuid(cls, v: Any) -> UUID | None:
        """Convert string UUIDs to UUID objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return UUID(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = self.model_dump()
        # Convert UUIDs to strings for JSON serialization
        data["id"] = str(self.id)
        data["recipe_id"] = str(self.recipe_id) if self.recipe_id else None
        data["curve_id"] = str(self.curve_id) if self.curve_id else None
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PrintDatabase:
    """SQLite-based print database with searchable metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialize the print database.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or Path(":memory:")
        self.conn: sqlite3.Connection = None  # type: ignore[assignment]  # Set by _initialize_db
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Create main prints table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prints (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Paper info
                paper_type TEXT NOT NULL,
                paper_weight INTEGER,
                paper_sizing TEXT,

                -- Chemistry
                chemistry_type TEXT NOT NULL,
                metal_ratio REAL NOT NULL,
                contrast_agent TEXT,
                contrast_amount REAL NOT NULL,
                developer TEXT,

                -- Process
                exposure_time REAL NOT NULL,
                uv_source TEXT,
                humidity REAL,
                temperature REAL,

                -- Results
                dmin REAL,
                dmax REAL,
                density_range REAL,
                overall_quality REAL NOT NULL,

                -- References
                recipe_id TEXT,
                curve_id TEXT,

                -- Data (JSON)
                image_paths TEXT,
                tags TEXT,
                notes TEXT,
                metadata TEXT
            )
            """
        )

        # Create indices for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON prints(timestamp DESC)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_type ON prints(paper_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_recipe_id ON prints(recipe_id)")

        # Create full-text search virtual table
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS prints_fts USING fts5(
                id,
                name,
                paper_type,
                chemistry_type,
                notes,
                tags,
                content=prints
            )
            """
        )

        self.conn.commit()

    def add_print(self, record: PrintRecord) -> UUID:
        """
        Add a print record to the database.

        Args:
            record: PrintRecord to add

        Returns:
            UUID of the added record
        """
        data = record.to_dict()

        self.conn.execute(
            """
            INSERT INTO prints VALUES (
                :id, :name, :timestamp,
                :paper_type, :paper_weight, :paper_sizing,
                :chemistry_type, :metal_ratio, :contrast_agent, :contrast_amount, :developer,
                :exposure_time, :uv_source, :humidity, :temperature,
                :dmin, :dmax, :density_range, :overall_quality,
                :recipe_id, :curve_id,
                :image_paths, :tags, :notes, :metadata
            )
            """,
            {
                **data,
                "image_paths": json.dumps(record.image_paths),
                "tags": json.dumps(record.tags),
                "metadata": json.dumps(record.metadata),
            },
        )

        # Update FTS index
        self.conn.execute(
            """
            INSERT INTO prints_fts(id, name, paper_type, chemistry_type, notes, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.id),
                record.name,
                record.paper_type,
                record.chemistry_type,
                record.notes or "",
                " ".join(record.tags),
            ),
        )

        self.conn.commit()
        return record.id

    def get_print(self, print_id: UUID) -> PrintRecord | None:
        """
        Get a single print record by ID.

        Args:
            print_id: UUID of the print to retrieve

        Returns:
            PrintRecord if found, None otherwise
        """
        cursor = self.conn.execute("SELECT * FROM prints WHERE id = ?", (str(print_id),))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def update_print(self, print_id: UUID, updates: dict[str, Any]) -> bool:
        """
        Update a print record.

        Args:
            print_id: UUID of the print to update
            updates: Dictionary of fields to update

        Returns:
            True if record was updated, False if not found
        """
        # Check if record exists
        existing = self.get_print(print_id)
        if existing is None:
            return False

        # Build UPDATE query dynamically
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ("image_paths", "tags", "metadata"):
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == "timestamp" and isinstance(value, datetime):
                set_clauses.append(f"{key} = ?")
                values.append(value.isoformat())
            elif key in ("id", "recipe_id", "curve_id") and isinstance(value, UUID):
                set_clauses.append(f"{key} = ?")
                values.append(str(value))
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)

        values.append(str(print_id))

        self.conn.execute(f"UPDATE prints SET {', '.join(set_clauses)} WHERE id = ?", values)

        # Update FTS if relevant fields changed
        if any(k in updates for k in ("name", "paper_type", "notes", "tags")):
            updated = self.get_print(print_id)
            assert updated is not None
            self.conn.execute(
                """
                UPDATE prints_fts SET
                    name = ?, paper_type = ?, chemistry_type = ?, notes = ?, tags = ?
                WHERE id = ?
                """,
                (
                    updated.name,
                    updated.paper_type,
                    updated.chemistry_type,
                    updated.notes or "",
                    " ".join(updated.tags),
                    str(print_id),
                ),
            )

        self.conn.commit()
        return True

    def delete_print(self, print_id: UUID) -> bool:
        """
        Delete a print record.

        Args:
            print_id: UUID of the print to delete

        Returns:
            True if record was deleted, False if not found
        """
        cursor = self.conn.execute("DELETE FROM prints WHERE id = ?", (str(print_id),))
        self.conn.execute("DELETE FROM prints_fts WHERE id = ?", (str(print_id),))
        self.conn.commit()
        return cursor.rowcount > 0

    def search_prints(self, query: str) -> list[PrintRecord]:
        """
        Full-text search for prints.

        Args:
            query: Search query string

        Returns:
            List of matching PrintRecords
        """
        cursor = self.conn.execute(
            """
            SELECT p.* FROM prints p
            JOIN prints_fts fts ON p.id = fts.id
            WHERE prints_fts MATCH ?
            ORDER BY rank
            """,
            (query,),
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def filter_prints(self, filters: dict[str, Any]) -> list[PrintRecord]:
        """
        Filter prints by criteria.

        Args:
            filters: Dictionary of field:value pairs to filter by
                    Supports comparison operators: field__gt, field__lt, field__gte, field__lte

        Returns:
            List of matching PrintRecords
        """
        where_clauses = []
        values = []

        for key, value in filters.items():
            if "__" in key:
                field, op = key.rsplit("__", 1)
                op_map = {
                    "gt": ">",
                    "lt": "<",
                    "gte": ">=",
                    "lte": "<=",
                    "ne": "!=",
                }
                if op in op_map:
                    where_clauses.append(f"{field} {op_map[op]} ?")
                    values.append(value)
            else:
                where_clauses.append(f"{key} = ?")
                values.append(value)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        cursor = self.conn.execute(
            f"SELECT * FROM prints WHERE {where_sql} ORDER BY timestamp DESC",
            values,
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_prints_by_date_range(self, start: datetime, end: datetime) -> list[PrintRecord]:
        """
        Get prints within a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of PrintRecords in the date range
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM prints
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            """,
            (start.isoformat(), end.isoformat()),
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_prints_by_paper(self, paper_type: str) -> list[PrintRecord]:
        """
        Get all prints for a specific paper type.

        Args:
            paper_type: Paper type to filter by

        Returns:
            List of PrintRecords for the paper type
        """
        return self.filter_prints({"paper_type": paper_type})

    def get_prints_by_recipe(self, recipe_id: UUID) -> list[PrintRecord]:
        """
        Get all prints using a specific recipe.

        Args:
            recipe_id: Recipe UUID to filter by

        Returns:
            List of PrintRecords using the recipe
        """
        cursor = self.conn.execute(
            "SELECT * FROM prints WHERE recipe_id = ? ORDER BY timestamp DESC",
            (str(recipe_id),),
        )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get aggregate statistics about prints.

        Returns:
            Dictionary of statistics including counts, averages, etc.
        """
        stats = {}

        # Total count
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM prints")
        stats["total_prints"] = cursor.fetchone()["count"]

        # Paper type distribution
        cursor = self.conn.execute(
            """
            SELECT paper_type, COUNT(*) as count
            FROM prints
            GROUP BY paper_type
            ORDER BY count DESC
            """
        )
        stats["paper_types"] = {row["paper_type"]: row["count"] for row in cursor}

        # Chemistry distribution
        cursor = self.conn.execute(
            """
            SELECT chemistry_type, COUNT(*) as count
            FROM prints
            GROUP BY chemistry_type
            """
        )
        stats["chemistry_types"] = {row["chemistry_type"]: row["count"] for row in cursor}

        # Average exposure time
        cursor = self.conn.execute(
            "SELECT AVG(exposure_time) as avg_exp FROM prints WHERE exposure_time > 0"
        )
        stats["avg_exposure_time"] = cursor.fetchone()["avg_exp"]

        # Density statistics
        cursor = self.conn.execute(
            """
            SELECT
                AVG(dmin) as avg_dmin,
                AVG(dmax) as avg_dmax,
                AVG(density_range) as avg_range,
                AVG(overall_quality) as avg_quality
            FROM prints
            WHERE dmin IS NOT NULL AND dmax IS NOT NULL
            """
        )
        row = cursor.fetchone()
        stats.update(
            {
                "avg_dmin": row["avg_dmin"],
                "avg_dmax": row["avg_dmax"],
                "avg_density_range": row["avg_range"],
                "avg_quality": row["avg_quality"],
            }
        )

        # Date range
        cursor = self.conn.execute(
            "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM prints"
        )
        row = cursor.fetchone()
        if row["first"]:
            stats["first_print"] = row["first"]
            stats["last_print"] = row["last"]

        return stats

    def backup_database(self, backup_path: Path) -> None:
        """
        Create a backup of the database.

        Args:
            backup_path: Path where backup should be saved
        """
        if self.db_path == Path(":memory:"):
            raise ValueError("Cannot backup in-memory database")

        # Ensure parent directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Close connection to avoid locks
        self.conn.close()

        # Copy database file
        shutil.copy2(self.db_path, backup_path)

        # Reopen connection
        self._initialize_db()

    def restore_database(self, backup_path: Path) -> None:
        """
        Restore database from a backup.

        Args:
            backup_path: Path to backup file
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        if self.db_path == Path(":memory:"):
            raise ValueError("Cannot restore to in-memory database")

        # Close connection
        self.conn.close()

        # Replace current database with backup
        shutil.copy2(backup_path, self.db_path)

        # Reopen connection
        self._initialize_db()

    def _row_to_record(self, row: sqlite3.Row) -> PrintRecord:
        """Convert database row to PrintRecord."""
        data = dict(row)

        # Parse JSON fields
        data["image_paths"] = json.loads(data["image_paths"]) if data["image_paths"] else []
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []
        data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}

        return PrintRecord(**data)

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "PrintDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
