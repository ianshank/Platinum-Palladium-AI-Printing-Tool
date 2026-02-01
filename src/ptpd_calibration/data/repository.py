"""
Repository pattern for data access in PTPD Calibration System.

Provides abstract base classes and implementations for CRUD operations
on calibration records, curves, and recipes. Supports SQLite (default)
and can be extended for other backends.

Usage:
    from ptpd_calibration.data.repository import CalibrationRepository

    repo = CalibrationRepository()

    # Add a record
    record = CalibrationRecord(...)
    saved = repo.add(record)

    # Query records
    results = repo.find(paper_type="Arches Platine")

    # Get by ID
    record = repo.get(record_id)
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
ID = TypeVar("ID")


class Repository(ABC, Generic[T, ID]):
    """Abstract base repository for CRUD operations.

    Subclasses must implement all abstract methods for
    specific data types and storage backends.
    """

    @abstractmethod
    def get(self, id: ID) -> T | None:
        """Get an entity by its ID.

        Args:
            id: Unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[T]:
        """Get all entities with pagination.

        Args:
            limit: Maximum number of entities to return.
            offset: Number of entities to skip.

        Returns:
            Sequence of entities.
        """
        ...

    @abstractmethod
    def add(self, entity: T) -> T:
        """Add a new entity.

        Args:
            entity: Entity to add.

        Returns:
            The added entity (may have ID assigned).
        """
        ...

    @abstractmethod
    def update(self, id: ID, entity: T) -> T | None:
        """Update an existing entity.

        Args:
            id: ID of the entity to update.
            entity: Updated entity data.

        Returns:
            The updated entity, or None if not found.
        """
        ...

    @abstractmethod
    def delete(self, id: ID) -> bool:
        """Delete an entity.

        Args:
            id: ID of the entity to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def find(self, **criteria: Any) -> Sequence[T]:
        """Find entities matching criteria.

        Args:
            **criteria: Field-value pairs to match.

        Returns:
            Sequence of matching entities.
        """
        ...

    @abstractmethod
    def count(self, **criteria: Any) -> int:
        """Count entities matching criteria.

        Args:
            **criteria: Field-value pairs to match.

        Returns:
            Number of matching entities.
        """
        ...


class SQLiteRepository(Repository[T, str], Generic[T]):
    """SQLite-based repository implementation.

    Provides a generic SQLite storage backend for Pydantic models.
    Data is stored as JSON in a single table with indexed fields
    for common queries.
    """

    def __init__(
        self,
        model_class: type[T],
        table_name: str,
        db_path: Path | None = None,
        indexed_fields: list[str] | None = None,
    ):
        """Initialize SQLite repository.

        Args:
            model_class: Pydantic model class for this repository.
            table_name: Name of the SQLite table.
            db_path: Path to database file. Defaults to settings.
            indexed_fields: Fields to index for faster queries.
        """
        self.model_class = model_class
        self.table_name = table_name
        self.indexed_fields = indexed_fields or []

        settings = get_settings()
        self.db_path = db_path or (settings.data_dir / "ptpd.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create main table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indexed field columns
            for field in self.indexed_fields:
                with suppress(sqlite3.OperationalError):
                    # Column may already exist
                    cursor.execute(f"""
                        ALTER TABLE {self.table_name}
                        ADD COLUMN {field} TEXT
                    """)

                # Create index
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_{field}
                    ON {self.table_name}({field})
                """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection as context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        import uuid

        return str(uuid.uuid4())

    def _extract_indexed_values(self, entity: T) -> dict[str, Any]:
        """Extract values for indexed fields from entity."""
        data = entity.model_dump()
        return {f: data.get(f) for f in self.indexed_fields if f in data}

    def get(self, id: str) -> T | None:
        """Get entity by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT data FROM {self.table_name} WHERE id = ?", (id,))
            row = cursor.fetchone()

            if row:
                return self.model_class.model_validate_json(row["data"])
            return None

    def get_all(self, limit: int = 100, offset: int = 0) -> Sequence[T]:
        """Get all entities with pagination."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT data FROM {self.table_name}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()

            return [self.model_class.model_validate_json(row["data"]) for row in rows]

    def add(self, entity: T) -> T:
        """Add a new entity."""
        entity_id = self._generate_id()

        # Try to get existing ID from entity
        if hasattr(entity, "id") and entity.id:
            entity_id = str(entity.id)

        now = datetime.now(timezone.utc).isoformat()

        # Set ID on entity if it has the field
        if hasattr(entity, "id"):
            entity_dict = entity.model_dump()
            entity_dict["id"] = entity_id
            entity = self.model_class.model_validate(entity_dict)

        data = entity.model_dump_json()
        indexed_values = self._extract_indexed_values(entity)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build insert statement with indexed fields
            fields = ["id", "data", "created_at", "updated_at"] + self.indexed_fields
            placeholders = ["?"] * len(fields)
            values = [entity_id, data, now, now] + [
                indexed_values.get(f) for f in self.indexed_fields
            ]

            cursor.execute(
                f"""
                INSERT INTO {self.table_name} ({", ".join(fields)})
                VALUES ({", ".join(placeholders)})
                """,
                values,
            )
            conn.commit()

        logger.debug(f"Added {self.model_class.__name__} with ID {entity_id}")
        return entity

    def update(self, id: str, entity: T) -> T | None:
        """Update an existing entity."""
        if not self.get(id):
            return None

        now = datetime.now(timezone.utc).isoformat()
        data = entity.model_dump_json()
        indexed_values = self._extract_indexed_values(entity)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build update statement
            set_clauses = ["data = ?", "updated_at = ?"]
            values = [data, now]

            for field in self.indexed_fields:
                set_clauses.append(f"{field} = ?")
                values.append(indexed_values.get(field))

            values.append(id)

            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET {", ".join(set_clauses)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()

        logger.debug(f"Updated {self.model_class.__name__} with ID {id}")
        return entity

    def delete(self, id: str) -> bool:
        """Delete an entity."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug(f"Deleted {self.model_class.__name__} with ID {id}")
        return deleted

    def find(self, **criteria: Any) -> Sequence[T]:
        """Find entities matching criteria."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build WHERE clause from criteria
            where_clauses = []
            values = []

            for field, value in criteria.items():
                if field in self.indexed_fields:
                    where_clauses.append(f"{field} = ?")
                    values.append(value)
                else:
                    # For non-indexed fields, use JSON extraction
                    where_clauses.append(f"json_extract(data, '$.{field}') = ?")
                    values.append(json.dumps(value) if not isinstance(value, str) else value)

            sql = f"SELECT data FROM {self.table_name}"
            if where_clauses:
                sql += f" WHERE {' AND '.join(where_clauses)}"
            sql += " ORDER BY created_at DESC"

            cursor.execute(sql, values)
            rows = cursor.fetchall()

            return [self.model_class.model_validate_json(row["data"]) for row in rows]

    def count(self, **criteria: Any) -> int:
        """Count entities matching criteria."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            where_clauses = []
            values = []

            for field, value in criteria.items():
                if field in self.indexed_fields:
                    where_clauses.append(f"{field} = ?")
                    values.append(value)
                else:
                    where_clauses.append(f"json_extract(data, '$.{field}') = ?")
                    values.append(json.dumps(value) if not isinstance(value, str) else value)

            sql = f"SELECT COUNT(*) as count FROM {self.table_name}"
            if where_clauses:
                sql += f" WHERE {' AND '.join(where_clauses)}"

            cursor.execute(sql, values)
            row = cursor.fetchone()
            return row["count"] if row else 0

    def search(
        self,
        query: str,
        fields: list[str] | None = None,
        limit: int = 50,
    ) -> Sequence[T]:
        """Full-text search across fields.

        Args:
            query: Search query string.
            fields: Fields to search (defaults to all indexed fields).
            limit: Maximum results to return.

        Returns:
            Matching entities.
        """
        fields = fields or self.indexed_fields

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build LIKE clauses for each field
            like_clauses = []
            values = []
            pattern = f"%{query}%"

            for field in fields:
                if field in self.indexed_fields:
                    like_clauses.append(f"{field} LIKE ?")
                else:
                    like_clauses.append(f"json_extract(data, '$.{field}') LIKE ?")
                values.append(pattern)

            sql = f"""
                SELECT data FROM {self.table_name}
                WHERE {" OR ".join(like_clauses)}
                ORDER BY created_at DESC
                LIMIT ?
            """
            values.append(limit)

            cursor.execute(sql, values)
            rows = cursor.fetchall()

            return [self.model_class.model_validate_json(row["data"]) for row in rows]


class InMemoryRepository(Repository[T, str], Generic[T]):
    """In-memory repository for testing.

    Stores entities in a dictionary for fast access without
    database overhead. Not persistent across restarts.
    """

    def __init__(self, model_class: type[T]):
        """Initialize in-memory repository.

        Args:
            model_class: Pydantic model class for this repository.
        """
        self.model_class = model_class
        self._store: dict[str, T] = {}
        self._counter = 0

    def _generate_id(self) -> str:
        self._counter += 1
        return str(self._counter)

    def get(self, id: str) -> T | None:
        return self._store.get(id)

    def get_all(self, limit: int = 100, offset: int = 0) -> Sequence[T]:
        items = list(self._store.values())
        return items[offset : offset + limit]

    def add(self, entity: T) -> T:
        entity_id = self._generate_id()
        if hasattr(entity, "id"):
            entity_dict = entity.model_dump()
            entity_dict["id"] = entity_id
            entity = self.model_class.model_validate(entity_dict)
        self._store[entity_id] = entity
        return entity

    def update(self, id: str, entity: T) -> T | None:
        if id not in self._store:
            return None
        self._store[id] = entity
        return entity

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            return True
        return False

    def find(self, **criteria: Any) -> Sequence[T]:
        results = []
        for entity in self._store.values():
            entity_dict = entity.model_dump()
            if all(entity_dict.get(k) == v for k, v in criteria.items()):
                results.append(entity)
        return results

    def count(self, **criteria: Any) -> int:
        return len(self.find(**criteria))

    def clear(self) -> None:
        """Clear all entities (for testing)."""
        self._store.clear()
        self._counter = 0
