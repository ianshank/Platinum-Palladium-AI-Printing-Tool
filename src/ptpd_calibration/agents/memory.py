"""
Agent memory system for persistent context and learning.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4


@dataclass
class MemoryItem:
    """A single memory item."""

    id: UUID = field(default_factory=uuid4)
    key: str = ""
    content: str = ""
    category: str = "general"
    importance: float = 0.5  # 0-1
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "key": self.key,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            key=data["key"],
            content=data["content"],
            category=data.get("category", "general"),
            importance=data.get("importance", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


class AgentMemory:
    """
    Persistent memory system for the calibration agent.

    Stores facts, preferences, insights, and context across sessions.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_items: int = 1000,
        working_memory_size: int = 10,
    ):
        """
        Initialize agent memory.

        Args:
            storage_path: Path for persistent storage.
            max_items: Maximum items in long-term memory.
            working_memory_size: Size of working memory window.
        """
        self.storage_path = storage_path
        self.max_items = max_items
        self.working_memory_size = working_memory_size

        self._long_term: dict[str, MemoryItem] = {}
        self._working: list[str] = []  # Recent context
        self._categories: dict[str, list[str]] = {}

        if storage_path and storage_path.exists():
            self._load()

    def remember(
        self,
        key: str,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> MemoryItem:
        """
        Store a memory item.

        Args:
            key: Unique key for the memory.
            content: The content to remember.
            category: Category (fact, preference, insight, etc.).
            importance: Importance score (0-1).
            metadata: Optional additional metadata.

        Returns:
            The created MemoryItem.
        """
        item = MemoryItem(
            key=key,
            content=content,
            category=category,
            importance=importance,
            metadata=metadata or {},
        )

        # Check if updating existing
        if key in self._long_term:
            old_item = self._long_term[key]
            item.id = old_item.id
            item.created_at = old_item.created_at
            item.access_count = old_item.access_count

        self._long_term[key] = item

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if key not in self._categories[category]:
            self._categories[category].append(key)

        # Prune if over limit
        if len(self._long_term) > self.max_items:
            self._prune()

        # Auto-save
        if self.storage_path:
            self._save()

        return item

    def recall(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """
        Recall memories matching a query.

        Args:
            query: Search query.
            category: Optional category filter.
            limit: Maximum items to return.

        Returns:
            List of matching MemoryItems.
        """
        candidates = []

        # Filter by category if specified
        if category:
            keys = self._categories.get(category, [])
            candidates = [self._long_term[k] for k in keys if k in self._long_term]
        else:
            candidates = list(self._long_term.values())

        # Score by relevance
        query_lower = query.lower()
        scored = []

        for item in candidates:
            score = 0.0

            # Key match
            if query_lower in item.key.lower():
                score += 0.4

            # Content match
            if query_lower in item.content.lower():
                score += 0.3

            # Importance boost
            score += item.importance * 0.2

            # Recency boost
            age_days = (datetime.now() - item.last_accessed).days
            recency = max(0, 1 - age_days / 30)
            score += recency * 0.1

            if score > 0:
                scored.append((item, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access times
        results = []
        for item, _ in scored[:limit]:
            item.last_accessed = datetime.now()
            item.access_count += 1
            results.append(item)

        if self.storage_path:
            self._save()

        return results

    def get(self, key: str) -> MemoryItem | None:
        """Get a specific memory by key."""
        item = self._long_term.get(key)
        if item:
            item.last_accessed = datetime.now()
            item.access_count += 1
        return item

    def forget(self, key: str) -> bool:
        """Remove a memory."""
        if key in self._long_term:
            item = self._long_term[key]
            del self._long_term[key]

            # Remove from category index
            if item.category in self._categories:
                if key in self._categories[item.category]:
                    self._categories[item.category].remove(key)

            if self.storage_path:
                self._save()
            return True
        return False

    def add_to_working_memory(self, content: str) -> None:
        """Add content to working memory."""
        self._working.append(content)
        if len(self._working) > self.working_memory_size:
            self._working.pop(0)

    def get_working_memory(self) -> list[str]:
        """Get current working memory."""
        return self._working.copy()

    def clear_working_memory(self) -> None:
        """Clear working memory."""
        self._working = []

    def get_categories(self) -> list[str]:
        """Get all categories."""
        return list(self._categories.keys())

    def get_by_category(self, category: str) -> list[MemoryItem]:
        """Get all memories in a category."""
        keys = self._categories.get(category, [])
        return [self._long_term[k] for k in keys if k in self._long_term]

    def summary(self) -> str:
        """Generate a memory summary."""
        total = len(self._long_term)
        categories = ", ".join(f"{k}:{len(v)}" for k, v in self._categories.items())
        working = len(self._working)

        return f"Memory: {total} items ({categories}), working: {working}"

    def _prune(self) -> None:
        """Prune least important/accessed memories."""
        if len(self._long_term) <= self.max_items:
            return

        # Score items by importance and access
        scored = []
        for key, item in self._long_term.items():
            age_days = (datetime.now() - item.last_accessed).days
            score = (
                item.importance * 0.5
                + (1 - age_days / 365) * 0.3
                + min(item.access_count / 10, 1) * 0.2
            )
            scored.append((key, score))

        # Sort by score and remove lowest
        scored.sort(key=lambda x: x[1])
        to_remove = len(self._long_term) - self.max_items

        for key, _ in scored[:to_remove]:
            self.forget(key)

    def _save(self) -> None:
        """Save to storage."""
        if not self.storage_path:
            return

        data = {
            "version": "1.0",
            "items": [item.to_dict() for item in self._long_term.values()],
            "categories": self._categories,
            "working": self._working,
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        for item_data in data.get("items", []):
            item = MemoryItem.from_dict(item_data)
            self._long_term[item.key] = item

        self._categories = data.get("categories", {})
        self._working = data.get("working", [])
