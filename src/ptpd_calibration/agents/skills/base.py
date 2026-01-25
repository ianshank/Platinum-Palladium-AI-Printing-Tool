"""
Base classes for skill modules.

Skills are focused capability modules that encapsulate domain expertise
and provide structured interfaces for the agent to use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar, Generic

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SkillCategory(str, Enum):
    """Categories of skills."""

    CALIBRATION = "calibration"
    CHEMISTRY = "chemistry"
    QUALITY = "quality"
    TROUBLESHOOTING = "troubleshooting"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    WORKFLOW = "workflow"


class SkillSettings(BaseSettings):
    """Base settings for skills."""

    model_config = SettingsConfigDict(env_prefix="PTPD_SKILL_")

    # Common settings
    enabled: bool = Field(default=True, description="Whether the skill is enabled")
    log_operations: bool = Field(
        default=True, description="Log skill operations for debugging"
    )
    cache_results: bool = Field(
        default=True, description="Cache results for performance"
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=0, le=3600, description="Cache TTL in seconds"
    )


@dataclass
class SkillContext:
    """
    Context provided to skill operations.

    Contains information about the current state, user preferences,
    and any relevant data from previous operations.
    """

    # Current state
    paper_type: Optional[str] = None
    metal_ratio: Optional[float] = None
    exposure_time: Optional[float] = None
    densities: Optional[list[float]] = None

    # Environment
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    uv_intensity: Optional[float] = None

    # User preferences
    user_level: str = "intermediate"  # beginner, intermediate, advanced
    preferred_format: str = "qtr"  # qtr, piezography, csv, json

    # History
    previous_results: list[dict[str, Any]] = field(default_factory=list)
    conversation_context: Optional[str] = None

    # Metadata
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class SkillResult:
    """
    Result from a skill operation.

    Provides structured output that the agent can interpret and
    present to the user.
    """

    success: bool
    data: dict[str, Any]
    message: str
    confidence: float = 1.0  # 0.0 to 1.0
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        data: dict[str, Any],
        message: str,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> "SkillResult":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            message=message,
            confidence=confidence,
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        message: str,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> "SkillResult":
        """Create a failure result."""
        data = {"error": error} if error else {}
        return cls(
            success=False,
            data=data,
            message=message,
            confidence=0.0,
            **kwargs,
        )


T = TypeVar("T", bound=SkillSettings)


class Skill(ABC, Generic[T]):
    """
    Abstract base class for skills.

    Skills encapsulate domain-specific expertise and provide
    focused functionality for the agent. Each skill should:
    - Have one primary responsibility
    - Use existing module functions (no logic duplication)
    - Return structured SkillResult objects
    - Be independently testable
    """

    def __init__(self, settings: Optional[T] = None) -> None:
        """
        Initialize the skill.

        Args:
            settings: Optional settings for the skill.
        """
        self._settings = settings or self._default_settings()
        self._cache: dict[str, tuple[SkillResult, float]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for the skill."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the skill does."""
        ...

    @property
    @abstractmethod
    def category(self) -> SkillCategory:
        """Category of the skill."""
        ...

    @property
    def settings(self) -> T:
        """Get the skill settings."""
        return self._settings

    @abstractmethod
    def _default_settings(self) -> T:
        """Get default settings for this skill."""
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """
        List the capabilities this skill provides.

        Returns:
            List of capability descriptions.
        """
        ...

    @abstractmethod
    def can_handle(self, task: str, context: Optional[SkillContext] = None) -> float:
        """
        Determine if this skill can handle the given task.

        Args:
            task: The task description.
            context: Optional context information.

        Returns:
            Confidence score (0.0 to 1.0) that this skill can handle the task.
        """
        ...

    @abstractmethod
    def execute(
        self,
        task: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute the skill for the given task.

        Args:
            task: The task description.
            context: Optional context information.
            **kwargs: Additional task-specific parameters.

        Returns:
            SkillResult with the operation outcome.
        """
        ...

    def validate_context(self, context: Optional[SkillContext]) -> list[str]:
        """
        Validate that the context has required information.

        Args:
            context: The context to validate.

        Returns:
            List of missing or invalid field names.
        """
        return []  # Override in subclasses as needed

    def _cache_key(self, task: str, **kwargs: Any) -> str:
        """Generate a cache key for a task."""
        import hashlib
        import json

        key_data = {"task": task, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[SkillResult]:
        """Get a cached result if valid."""
        import time

        if not self._settings.cache_results:
            return None

        if key in self._cache:
            result, cached_time = self._cache[key]
            if time.time() - cached_time < self._settings.cache_ttl_seconds:
                return result
            else:
                del self._cache[key]

        return None

    def _set_cached(self, key: str, result: SkillResult) -> None:
        """Cache a result."""
        import time

        if self._settings.cache_results:
            self._cache[key] = (result, time.time())

    def clear_cache(self) -> None:
        """Clear the skill's cache."""
        self._cache.clear()


class SkillRegistry:
    """
    Registry for managing available skills.

    Provides methods to register, retrieve, and find appropriate
    skills for given tasks.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """
        Register a skill.

        Args:
            skill: The skill to register.
        """
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        """
        Unregister a skill by name.

        Args:
            name: The skill name.

        Returns:
            True if skill was unregistered, False if not found.
        """
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: The skill name.

        Returns:
            The skill or None if not found.
        """
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """
        List all registered skills.

        Returns:
            List of all skills.
        """
        return list(self._skills.values())

    def list_by_category(self, category: SkillCategory) -> list[Skill]:
        """
        List skills in a category.

        Args:
            category: The category to filter by.

        Returns:
            List of skills in the category.
        """
        return [s for s in self._skills.values() if s.category == category]

    def find_best_skill(
        self,
        task: str,
        context: Optional[SkillContext] = None,
    ) -> Optional[tuple[Skill, float]]:
        """
        Find the best skill to handle a task.

        Args:
            task: The task description.
            context: Optional context information.

        Returns:
            Tuple of (skill, confidence) or None if no skill can handle it.
        """
        best_skill: Optional[Skill] = None
        best_confidence: float = 0.0

        for skill in self._skills.values():
            if not skill.settings.enabled:
                continue

            confidence = skill.can_handle(task, context)
            if confidence > best_confidence:
                best_confidence = confidence
                best_skill = skill

        if best_skill and best_confidence > 0:
            return (best_skill, best_confidence)

        return None

    def find_capable_skills(
        self,
        task: str,
        context: Optional[SkillContext] = None,
        min_confidence: float = 0.3,
    ) -> list[tuple[Skill, float]]:
        """
        Find all skills that can handle a task.

        Args:
            task: The task description.
            context: Optional context information.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of (skill, confidence) tuples sorted by confidence.
        """
        capable: list[tuple[Skill, float]] = []

        for skill in self._skills.values():
            if not skill.settings.enabled:
                continue

            confidence = skill.can_handle(task, context)
            if confidence >= min_confidence:
                capable.append((skill, confidence))

        return sorted(capable, key=lambda x: x[1], reverse=True)

    def execute_best(
        self,
        task: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> Optional[SkillResult]:
        """
        Execute the best skill for a task.

        Args:
            task: The task description.
            context: Optional context information.
            **kwargs: Additional task parameters.

        Returns:
            SkillResult from the best skill, or None if no skill can handle it.
        """
        result = self.find_best_skill(task, context)
        if result:
            skill, _ = result
            return skill.execute(task, context, **kwargs)
        return None
