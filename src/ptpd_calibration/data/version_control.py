"""
Version control system for recipes and settings.

Provides Git-like versioning with commits, branches, merging, and history
for tracking changes to calibration recipes and application settings.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class VersionedItem(BaseModel):
    """A versioned item with content and metadata."""

    # Version identifiers
    version_id: UUID = Field(default_factory=uuid4)
    item_id: str = Field(..., min_length=1)
    version_number: int = Field(..., ge=1)

    # Version metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    author: str | None = Field(default=None)
    message: str = Field(..., min_length=1)

    # Content
    content: dict[str, Any] = Field(...)
    content_hash: str = Field(...)

    # Lineage
    parent_version_id: UUID | None = Field(default=None)
    branch: str = Field(default="main")

    # Tags
    tags: list[str] = Field(default_factory=list)

    @staticmethod
    def calculate_hash(content: dict[str, Any]) -> str:
        """Calculate SHA256 hash of content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class VersionDiff(BaseModel):
    """Differences between two versions."""

    version1_id: UUID
    version2_id: UUID
    added_keys: list[str] = Field(default_factory=list)
    removed_keys: list[str] = Field(default_factory=list)
    modified_keys: list[str] = Field(default_factory=list)
    changes: dict[str, dict[str, Any]] = Field(default_factory=dict)


class MergeConflict(BaseModel):
    """A merge conflict between branches."""

    key: str
    branch1_value: Any
    branch2_value: Any
    base_value: Any | None = Field(default=None)


class MergeResult(BaseModel):
    """Result of a merge operation."""

    success: bool
    merged_version_id: UUID | None = Field(default=None)
    conflicts: list[MergeConflict] = Field(default_factory=list)
    message: str = Field(default="")


class VersionController:
    """Version control system for items."""

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialize the version controller.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or Path(":memory:")
        self.conn: sqlite3.Connection | None = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Create versions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                author TEXT,
                message TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                parent_version_id TEXT,
                branch TEXT NOT NULL DEFAULT 'main',
                tags TEXT,
                UNIQUE(item_id, version_number, branch)
            )
            """
        )

        # Create indices
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_item_id ON versions(item_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_branch ON versions(item_id, branch)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON versions(timestamp DESC)"
        )

        # Create branches table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                item_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                head_version_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (item_id, branch_name)
            )
            """
        )

        self.conn.commit()

    def commit(
        self,
        item_id: str,
        content: dict[str, Any],
        message: str,
        author: str | None = None,
        branch: str = "main",
    ) -> VersionedItem:
        """
        Create a new version of an item.

        Args:
            item_id: Unique identifier for the item
            content: Item content as dictionary
            message: Commit message
            author: Optional author name
            branch: Branch name (default: 'main')

        Returns:
            Created VersionedItem
        """
        # Get current version number for this item/branch
        cursor = self.conn.execute(
            """
            SELECT MAX(version_number) as max_ver
            FROM versions
            WHERE item_id = ? AND branch = ?
            """,
            (item_id, branch),
        )
        row = cursor.fetchone()
        next_version = (row["max_ver"] or 0) + 1

        # Get parent version ID (last commit on this branch)
        cursor = self.conn.execute(
            """
            SELECT version_id
            FROM versions
            WHERE item_id = ? AND branch = ?
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (item_id, branch),
        )
        row = cursor.fetchone()
        parent_version_id = UUID(row["version_id"]) if row else None

        # Create versioned item
        content_hash = VersionedItem.calculate_hash(content)
        version = VersionedItem(
            item_id=item_id,
            version_number=next_version,
            timestamp=datetime.now(),
            author=author,
            message=message,
            content=content,
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            branch=branch,
        )

        # Save to database
        self.conn.execute(
            """
            INSERT INTO versions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                str(version.version_id),
                version.item_id,
                version.version_number,
                version.timestamp.isoformat(),
                version.author,
                version.message,
                json.dumps(version.content),
                version.content_hash,
                str(version.parent_version_id) if version.parent_version_id else None,
                version.branch,
                json.dumps(version.tags),
            ),
        )

        # Update branch head
        self.conn.execute(
            """
            INSERT OR REPLACE INTO branches VALUES (?, ?, ?, ?)
            """,
            (
                item_id,
                branch,
                str(version.version_id),
                datetime.now().isoformat(),
            ),
        )

        self.conn.commit()
        return version

    def get_version(self, version_id: UUID) -> VersionedItem | None:
        """
        Get a specific version by ID.

        Args:
            version_id: UUID of the version

        Returns:
            VersionedItem if found, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT * FROM versions WHERE version_id = ?",
            (str(version_id),),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_version(row)

    def get_history(
        self, item_id: str, branch: str = "main", limit: int | None = None
    ) -> list[VersionedItem]:
        """
        Get version history for an item.

        Args:
            item_id: Item identifier
            branch: Branch name (default: 'main')
            limit: Optional limit on number of versions to return

        Returns:
            List of VersionedItems in reverse chronological order
        """
        query = """
            SELECT * FROM versions
            WHERE item_id = ? AND branch = ?
            ORDER BY version_number DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query, (item_id, branch))
        return [self._row_to_version(row) for row in cursor.fetchall()]

    def diff(self, version1_id: UUID, version2_id: UUID) -> VersionDiff:
        """
        Compare two versions.

        Args:
            version1_id: First version UUID
            version2_id: Second version UUID

        Returns:
            VersionDiff describing differences

        Raises:
            ValueError: If either version is not found
        """
        v1 = self.get_version(version1_id)
        v2 = self.get_version(version2_id)

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        diff = VersionDiff(
            version1_id=version1_id,
            version2_id=version2_id,
        )

        # Compare keys
        keys1 = set(v1.content.keys())
        keys2 = set(v2.content.keys())

        diff.added_keys = list(keys2 - keys1)
        diff.removed_keys = list(keys1 - keys2)

        # Find modified keys
        common_keys = keys1 & keys2
        for key in common_keys:
            if v1.content[key] != v2.content[key]:
                diff.modified_keys.append(key)
                diff.changes[key] = {
                    "old": v1.content[key],
                    "new": v2.content[key],
                }

        return diff

    def rollback(self, version_id: UUID, message: str | None = None) -> VersionedItem:
        """
        Rollback to a previous version by creating a new commit.

        Args:
            version_id: UUID of version to rollback to
            message: Optional commit message

        Returns:
            New VersionedItem with rolled-back content

        Raises:
            ValueError: If version is not found
        """
        version = self.get_version(version_id)
        if version is None:
            raise ValueError(f"Version not found: {version_id}")

        rollback_message = message or f"Rollback to version {version.version_number}"

        return self.commit(
            item_id=version.item_id,
            content=version.content,
            message=rollback_message,
            branch=version.branch,
        )

    def branch(self, item_id: str, branch_name: str, from_branch: str = "main") -> bool:
        """
        Create a new branch.

        Args:
            item_id: Item identifier
            branch_name: Name for new branch
            from_branch: Branch to branch from (default: 'main')

        Returns:
            True if branch was created, False if it already exists

        Raises:
            ValueError: If source branch doesn't exist
        """
        # Check if branch already exists
        cursor = self.conn.execute(
            "SELECT 1 FROM branches WHERE item_id = ? AND branch_name = ?",
            (item_id, branch_name),
        )
        if cursor.fetchone():
            return False

        # Get head of source branch
        cursor = self.conn.execute(
            "SELECT head_version_id FROM branches WHERE item_id = ? AND branch_name = ?",
            (item_id, from_branch),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Source branch not found: {from_branch}")

        head_version_id = row["head_version_id"]

        # Create new branch pointing to same head
        self.conn.execute(
            "INSERT INTO branches VALUES (?, ?, ?, ?)",
            (item_id, branch_name, head_version_id, datetime.now().isoformat()),
        )
        self.conn.commit()

        return True

    def merge(
        self,
        item_id: str,
        source_branch: str,
        target_branch: str,
        strategy: str = "auto",
        author: str | None = None,
    ) -> MergeResult:
        """
        Merge two branches.

        Args:
            item_id: Item identifier
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy ('auto', 'ours', 'theirs')
            author: Optional author name

        Returns:
            MergeResult with merge status and conflicts

        Raises:
            ValueError: If branches not found
        """
        # Get heads of both branches
        source_version = self._get_branch_head(item_id, source_branch)
        target_version = self._get_branch_head(item_id, target_branch)

        if not source_version or not target_version:
            raise ValueError("One or both branches not found")

        # Find common ancestor
        ancestor = self._find_common_ancestor(source_version, target_version)

        # Detect conflicts
        conflicts = self._detect_merge_conflicts(
            source_version.content,
            target_version.content,
            ancestor.content if ancestor else {},
        )

        if conflicts and strategy == "auto":
            return MergeResult(
                success=False,
                conflicts=conflicts,
                message=f"Merge has {len(conflicts)} conflict(s)",
            )

        # Merge content based on strategy
        if strategy == "ours":
            merged_content = target_version.content.copy()
        elif strategy == "theirs":
            merged_content = source_version.content.copy()
        else:  # auto with no conflicts
            merged_content = self._auto_merge(
                source_version.content,
                target_version.content,
                ancestor.content if ancestor else {},
            )

        # Create merge commit
        message = f"Merge branch '{source_branch}' into '{target_branch}'"
        merged_version = self.commit(
            item_id=item_id,
            content=merged_content,
            message=message,
            author=author,
            branch=target_branch,
        )

        return MergeResult(
            success=True,
            merged_version_id=merged_version.version_id,
            message=message,
        )

    def list_branches(self, item_id: str) -> list[dict[str, Any]]:
        """
        List all branches for an item.

        Args:
            item_id: Item identifier

        Returns:
            List of branch information dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT b.branch_name, b.created_at, v.version_number, v.message, v.timestamp
            FROM branches b
            JOIN versions v ON b.head_version_id = v.version_id
            WHERE b.item_id = ?
            ORDER BY b.created_at DESC
            """,
            (item_id,),
        )

        return [
            {
                "name": row["branch_name"],
                "created_at": row["created_at"],
                "head_version": row["version_number"],
                "last_message": row["message"],
                "last_commit": row["timestamp"],
            }
            for row in cursor
        ]

    def tag_version(self, version_id: UUID, tag: str) -> bool:
        """
        Add a tag to a version.

        Args:
            version_id: Version UUID
            tag: Tag to add

        Returns:
            True if tag was added, False if version not found
        """
        version = self.get_version(version_id)
        if version is None:
            return False

        if tag not in version.tags:
            version.tags.append(tag)
            self.conn.execute(
                "UPDATE versions SET tags = ? WHERE version_id = ?",
                (json.dumps(version.tags), str(version_id)),
            )
            self.conn.commit()

        return True

    def _row_to_version(self, row: sqlite3.Row) -> VersionedItem:
        """Convert database row to VersionedItem."""
        data = dict(row)
        data["content"] = json.loads(data["content"])
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []
        if data["parent_version_id"]:
            data["parent_version_id"] = UUID(data["parent_version_id"])
        return VersionedItem(**data)

    def _get_branch_head(self, item_id: str, branch: str) -> VersionedItem | None:
        """Get the head version of a branch."""
        cursor = self.conn.execute(
            """
            SELECT v.* FROM versions v
            JOIN branches b ON v.version_id = b.head_version_id
            WHERE b.item_id = ? AND b.branch_name = ?
            """,
            (item_id, branch),
        )
        row = cursor.fetchone()
        return self._row_to_version(row) if row else None

    def _find_common_ancestor(
        self, v1: VersionedItem, v2: VersionedItem
    ) -> VersionedItem | None:
        """Find common ancestor of two versions."""
        # Simple implementation - traverse v1's parents until we find one in v2's lineage
        v1_ancestors = self._get_ancestors(v1)
        v2_ancestors = self._get_ancestors(v2)

        # Find first common ancestor
        for ancestor in v1_ancestors:
            if ancestor.version_id in [v.version_id for v in v2_ancestors]:
                return ancestor

        return None

    def _get_ancestors(self, version: VersionedItem) -> list[VersionedItem]:
        """Get all ancestors of a version."""
        ancestors = []
        current = version

        while current.parent_version_id:
            parent = self.get_version(current.parent_version_id)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent

        return ancestors

    def _detect_merge_conflicts(
        self, source: dict[str, Any], target: dict[str, Any], base: dict[str, Any]
    ) -> list[MergeConflict]:
        """Detect conflicts between source and target branches."""
        conflicts = []
        all_keys = set(source.keys()) | set(target.keys()) | set(base.keys())

        for key in all_keys:
            source_val = source.get(key)
            target_val = target.get(key)
            base_val = base.get(key)

            # Conflict if both changed from base and values differ
            if (
                source_val != base_val
                and target_val != base_val
                and source_val != target_val
            ):
                conflicts.append(
                    MergeConflict(
                        key=key,
                        branch1_value=source_val,
                        branch2_value=target_val,
                        base_value=base_val,
                    )
                )

        return conflicts

    def _auto_merge(
        self, source: dict[str, Any], target: dict[str, Any], base: dict[str, Any]
    ) -> dict[str, Any]:
        """Automatically merge changes without conflicts."""
        merged = target.copy()

        for key in source:
            source_val = source.get(key)
            target_val = target.get(key)
            base_val = base.get(key)

            # If only source changed, use source value
            if source_val != base_val and target_val == base_val or source_val == target_val:
                merged[key] = source_val

        return merged

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "VersionController":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
