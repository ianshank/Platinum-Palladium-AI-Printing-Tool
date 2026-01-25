"""
Agent Memory System

Provides memory management for agents with:
- Working memory (short-term)
- Long-term memory with persistence
- Semantic search
- Memory consolidation
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory entries."""

    OBSERVATION = "observation"
    ACTION = "action"
    RESULT = "result"
    THOUGHT = "thought"
    FACT = "fact"
    CONTEXT = "context"
    ERROR = "error"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str = Field(default_factory=lambda: hashlib.sha256(
        f"{time.time()}".encode()
    ).hexdigest()[:16])

    type: MemoryType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Metadata
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str | None = None
    task_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Retrieval tracking
    access_count: int = 0
    last_accessed: datetime | None = None

    def access(self) -> None:
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

    def to_context_string(self) -> str:
        """Convert to string for context injection."""
        return f"[{self.type.value.upper()}] ({self.timestamp.strftime('%H:%M:%S')}): {self.content}"


class MemoryStore(BaseModel):
    """Persistent memory storage."""

    entries: list[MemoryEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentMemory:
    """
    Agent memory management system.

    Provides:
    - Working memory (recent entries, fast access)
    - Long-term memory (persisted, searchable)
    - Importance-based retrieval
    - Memory consolidation

    Usage:
        memory = AgentMemory(working_size=20)

        # Add memories
        memory.add("User asked about weather", MemoryType.USER_INPUT)
        memory.add("Retrieved weather data for NYC", MemoryType.OBSERVATION)

        # Retrieve relevant memories
        context = memory.get_context(max_entries=5)

        # Search memories
        results = memory.search("weather", limit=3)
    """

    def __init__(
        self,
        working_size: int = 20,
        long_term_path: Path | None = None,
        auto_consolidate: bool = True,
        consolidate_threshold: int = 50,
    ):
        """
        Initialize agent memory.

        Args:
            working_size: Size of working memory (recent entries)
            long_term_path: Path for persistent storage
            auto_consolidate: Auto-consolidate when threshold reached
            consolidate_threshold: Number of entries before consolidation
        """
        self.working_size = working_size
        self.long_term_path = long_term_path
        self.auto_consolidate = auto_consolidate
        self.consolidate_threshold = consolidate_threshold

        # Working memory (fast, recent)
        self._working: deque[MemoryEntry] = deque(maxlen=working_size)

        # Long-term memory (searchable, persistent)
        self._long_term: list[MemoryEntry] = []

        # Index for fast lookup
        self._by_type: dict[MemoryType, list[MemoryEntry]] = {
            t: [] for t in MemoryType
        }
        self._by_task: dict[str, list[MemoryEntry]] = {}

        # Load existing long-term memory
        if long_term_path and long_term_path.exists():
            self._load_from_file()

        self._logger = get_logger("agent.memory")

    def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        task_id: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Add a memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            task_id: Associated task ID
            source: Memory source
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Created memory entry
        """
        entry = MemoryEntry(
            type=memory_type,
            content=content,
            importance=importance,
            task_id=task_id,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Add to working memory
        self._working.append(entry)

        # Add to indexes
        self._by_type[memory_type].append(entry)
        if task_id:
            if task_id not in self._by_task:
                self._by_task[task_id] = []
            self._by_task[task_id].append(entry)

        self._logger.debug(
            f"Added memory: {memory_type.value}",
            importance=importance,
            task_id=task_id,
        )

        # Auto-consolidate if needed
        if self.auto_consolidate and len(self._working) >= self.consolidate_threshold:
            self.consolidate()

        return entry

    def add_observation(self, content: str, **kwargs: Any) -> MemoryEntry:
        """Add an observation memory."""
        return self.add(content, MemoryType.OBSERVATION, **kwargs)

    def add_action(self, content: str, **kwargs: Any) -> MemoryEntry:
        """Add an action memory."""
        return self.add(content, MemoryType.ACTION, **kwargs)

    def add_thought(self, content: str, **kwargs: Any) -> MemoryEntry:
        """Add a thought memory."""
        return self.add(content, MemoryType.THOUGHT, **kwargs)

    def add_fact(self, content: str, importance: float = 0.8, **kwargs: Any) -> MemoryEntry:
        """Add a fact memory (high importance by default)."""
        return self.add(content, MemoryType.FACT, importance=importance, **kwargs)

    def get_recent(
        self,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """
        Get recent memories from working memory.

        Args:
            limit: Maximum entries to return
            memory_type: Filter by type

        Returns:
            List of recent memory entries
        """
        entries = list(self._working)

        if memory_type:
            entries = [e for e in entries if e.type == memory_type]

        # Return most recent
        return entries[-limit:]

    def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Get memories by type."""
        entries = self._by_type.get(memory_type, [])
        if limit:
            entries = entries[-limit:]
        return entries

    def get_by_task(
        self,
        task_id: str,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Get memories for a specific task."""
        entries = self._by_task.get(task_id, [])
        if limit:
            entries = entries[-limit:]
        return entries

    def get_important(
        self,
        threshold: float = 0.7,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Get important memories above threshold.

        Args:
            threshold: Minimum importance score
            limit: Maximum entries

        Returns:
            List of important memories
        """
        all_entries = list(self._working) + self._long_term
        important = [e for e in all_entries if e.importance >= threshold]

        # Sort by importance, then by recency
        important.sort(
            key=lambda e: (e.importance, e.timestamp),
            reverse=True,
        )

        return important[:limit]

    def search(
        self,
        query: str,
        limit: int = 5,
        include_long_term: bool = True,
    ) -> list[MemoryEntry]:
        """
        Search memories by content.

        Simple keyword search. Can be extended with semantic search.

        Args:
            query: Search query
            limit: Maximum results
            include_long_term: Include long-term memory

        Returns:
            Matching memory entries
        """
        query_lower = query.lower()
        all_entries = list(self._working)

        if include_long_term:
            all_entries.extend(self._long_term)

        # Simple keyword matching with scoring
        results: list[tuple[float, MemoryEntry]] = []

        for entry in all_entries:
            content_lower = entry.content.lower()

            # Calculate relevance score
            if query_lower in content_lower:
                # Exact match gets higher score
                score = 1.0
            else:
                # Partial word matching
                query_words = query_lower.split()
                matches = sum(1 for w in query_words if w in content_lower)
                score = matches / len(query_words) if query_words else 0

            if score > 0:
                # Boost by importance
                final_score = score * (0.5 + 0.5 * entry.importance)
                results.append((final_score, entry))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)

        # Track access
        found_entries = [e for _, e in results[:limit]]
        for entry in found_entries:
            entry.access()

        return found_entries

    def get_context(
        self,
        max_entries: int = 10,
        max_chars: int = 2000,
        include_types: list[MemoryType] | None = None,
    ) -> str:
        """
        Get formatted context string for agent prompt.

        Args:
            max_entries: Maximum entries to include
            max_chars: Maximum character limit
            include_types: Types to include (all if None)

        Returns:
            Formatted context string
        """
        # Get recent memories
        entries = self.get_recent(limit=max_entries * 2)

        # Filter by type
        if include_types:
            entries = [e for e in entries if e.type in include_types]

        # Build context string
        lines: list[str] = []
        total_chars = 0

        for entry in entries[-max_entries:]:
            line = entry.to_context_string()
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def consolidate(self) -> int:
        """
        Consolidate working memory to long-term.

        Moves important entries from working to long-term memory.

        Returns:
            Number of entries consolidated
        """
        # Get entries to consolidate (older half of working memory)
        to_consolidate = list(self._working)[: len(self._working) // 2]

        # Filter by importance
        important = [e for e in to_consolidate if e.importance >= 0.5]

        # Move to long-term
        self._long_term.extend(important)

        # Save to file if path configured
        if self.long_term_path:
            self._save_to_file()

        self._logger.debug(f"Consolidated {len(important)} memories to long-term")

        return len(important)

    def clear_working(self) -> None:
        """Clear working memory."""
        self._working.clear()
        self._by_type = {t: [] for t in MemoryType}
        self._by_task.clear()

    def clear_all(self) -> None:
        """Clear all memory."""
        self.clear_working()
        self._long_term.clear()

    def _save_to_file(self) -> None:
        """Save long-term memory to file."""
        if not self.long_term_path:
            return

        store = MemoryStore(
            entries=self._long_term,
            metadata={"working_size": self.working_size},
            updated_at=datetime.utcnow(),
        )

        self.long_term_path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_path.write_text(store.model_dump_json(indent=2))

    def _load_from_file(self) -> None:
        """Load long-term memory from file."""
        if not self.long_term_path or not self.long_term_path.exists():
            return

        try:
            data = json.loads(self.long_term_path.read_text())
            store = MemoryStore.model_validate(data)
            self._long_term = store.entries
            self._logger.info(f"Loaded {len(self._long_term)} memories from file")
        except Exception as e:
            self._logger.error(f"Failed to load memory: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "working_count": len(self._working),
            "working_capacity": self.working_size,
            "long_term_count": len(self._long_term),
            "by_type": {
                t.value: len(entries)
                for t, entries in self._by_type.items()
            },
            "task_count": len(self._by_task),
        }

    def __len__(self) -> int:
        """Total memory count."""
        return len(self._working) + len(self._long_term)

    def __repr__(self) -> str:
        """String representation."""
        return f"AgentMemory(working={len(self._working)}, long_term={len(self._long_term)})"
