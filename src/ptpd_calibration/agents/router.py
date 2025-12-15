"""
Task Router for intelligent task classification and routing.

Routes incoming tasks to appropriate capability layers based on complexity,
keywords, and context. Uses configurable patterns rather than hardcoded values.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TaskComplexity(str, Enum):
    """Task complexity levels determining the capability layer to use."""

    SIMPLE = "simple"  # Direct tool execution, single step
    ASSISTED = "assisted"  # Analysis and recommendations, multi-step with guidance
    AUTONOMOUS = "autonomous"  # Full workflow automation with planning


class TaskCategory(str, Enum):
    """Categories of tasks for routing to appropriate skills."""

    CALIBRATION = "calibration"
    CHEMISTRY = "chemistry"
    QUALITY = "quality"
    TROUBLESHOOTING = "troubleshooting"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    RECIPE = "recipe"
    EXPORT = "export"
    GENERAL = "general"


@dataclass
class RoutingPattern:
    """A pattern for matching tasks to categories and complexity."""

    keywords: list[str]
    category: TaskCategory
    default_complexity: TaskComplexity
    complexity_modifiers: dict[str, TaskComplexity] = field(default_factory=dict)
    priority: int = 0  # Higher priority patterns are checked first

    def matches(self, task: str) -> bool:
        """Check if this pattern matches the given task."""
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in self.keywords)

    def determine_complexity(self, task: str) -> TaskComplexity:
        """Determine the complexity based on task content."""
        task_lower = task.lower()
        for modifier_keyword, complexity in self.complexity_modifiers.items():
            if modifier_keyword in task_lower:
                return complexity
        return self.default_complexity


class RouterSettings(BaseSettings):
    """Configuration settings for the task router."""

    model_config = SettingsConfigDict(env_prefix="PTPD_ROUTER_")

    # Complexity classification thresholds
    simple_max_words: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Maximum words for a task to be considered simple",
    )
    autonomous_min_words: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Minimum words for a task to potentially be autonomous",
    )

    # LLM-based classification
    use_llm_classification: bool = Field(
        default=False,
        description="Use LLM for ambiguous task classification",
    )
    llm_classification_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which to use LLM classification",
    )

    # Pattern matching
    case_sensitive_matching: bool = Field(
        default=False,
        description="Use case-sensitive pattern matching",
    )

    # Capability escalation
    allow_escalation: bool = Field(
        default=True,
        description="Allow automatic escalation to higher complexity if needed",
    )
    escalation_on_failure: bool = Field(
        default=True,
        description="Escalate to higher complexity on task failure",
    )

    # Logging
    log_routing_decisions: bool = Field(
        default=True,
        description="Log routing decisions for debugging",
    )


@dataclass
class RoutingResult:
    """Result of task routing decision."""

    category: TaskCategory
    complexity: TaskComplexity
    confidence: float  # 0.0 to 1.0
    matched_pattern: Optional[str] = None
    suggested_template: Optional[str] = None
    suggested_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_use_planning(self) -> bool:
        """Whether this task should use the planning system."""
        return self.complexity in (TaskComplexity.ASSISTED, TaskComplexity.AUTONOMOUS)

    @property
    def should_use_reflection(self) -> bool:
        """Whether this task should enable reflection."""
        return self.complexity == TaskComplexity.AUTONOMOUS


class PatternRegistry:
    """Registry of routing patterns with configurable patterns."""

    def __init__(self) -> None:
        """Initialize with default patterns."""
        self._patterns: list[RoutingPattern] = []
        self._load_default_patterns()

    def _load_default_patterns(self) -> None:
        """Load the default routing patterns."""
        # Calibration patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "calibrate",
                    "calibration",
                    "step tablet",
                    "linearize",
                    "linearization",
                ],
                category=TaskCategory.CALIBRATION,
                default_complexity=TaskComplexity.AUTONOMOUS,
                complexity_modifiers={
                    "quick": TaskComplexity.ASSISTED,
                    "simple": TaskComplexity.ASSISTED,
                    "full": TaskComplexity.AUTONOMOUS,
                    "complete": TaskComplexity.AUTONOMOUS,
                },
                priority=10,
            )
        )

        # Chemistry patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "chemistry",
                    "coating",
                    "formula",
                    "solution",
                    "drops",
                    "platinum ratio",
                    "palladium ratio",
                    "na2",
                    "ferric oxalate",
                ],
                category=TaskCategory.CHEMISTRY,
                default_complexity=TaskComplexity.ASSISTED,
                complexity_modifiers={
                    "calculate": TaskComplexity.SIMPLE,
                    "how many": TaskComplexity.SIMPLE,
                    "optimize": TaskComplexity.AUTONOMOUS,
                },
                priority=8,
            )
        )

        # Quality patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "quality",
                    "qa",
                    "check",
                    "validate",
                    "verify",
                    "assess",
                    "dmax",
                    "dmin",
                    "density range",
                ],
                category=TaskCategory.QUALITY,
                default_complexity=TaskComplexity.ASSISTED,
                complexity_modifiers={
                    "quick check": TaskComplexity.SIMPLE,
                    "full audit": TaskComplexity.AUTONOMOUS,
                },
                priority=7,
            )
        )

        # Troubleshooting patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "troubleshoot",
                    "problem",
                    "issue",
                    "wrong",
                    "not working",
                    "fix",
                    "help",
                    "why",
                    "blocked",
                    "muddy",
                    "faded",
                ],
                category=TaskCategory.TROUBLESHOOTING,
                default_complexity=TaskComplexity.ASSISTED,
                complexity_modifiers={
                    "diagnose": TaskComplexity.AUTONOMOUS,
                    "investigate": TaskComplexity.AUTONOMOUS,
                },
                priority=9,
            )
        )

        # Analysis patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "analyze",
                    "analysis",
                    "compare",
                    "evaluate",
                    "measure",
                    "densit",
                ],
                category=TaskCategory.ANALYSIS,
                default_complexity=TaskComplexity.ASSISTED,
                complexity_modifiers={
                    "detailed": TaskComplexity.AUTONOMOUS,
                    "comprehensive": TaskComplexity.AUTONOMOUS,
                    "quick": TaskComplexity.SIMPLE,
                },
                priority=6,
            )
        )

        # Prediction patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "predict",
                    "estimate",
                    "forecast",
                    "suggest",
                    "recommend",
                    "exposure time",
                ],
                category=TaskCategory.PREDICTION,
                default_complexity=TaskComplexity.ASSISTED,
                complexity_modifiers={
                    "quick": TaskComplexity.SIMPLE,
                },
                priority=5,
            )
        )

        # Recipe patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "recipe",
                    "workflow",
                    "batch",
                    "process multiple",
                    "automate",
                ],
                category=TaskCategory.RECIPE,
                default_complexity=TaskComplexity.AUTONOMOUS,
                complexity_modifiers={
                    "save": TaskComplexity.SIMPLE,
                    "load": TaskComplexity.SIMPLE,
                    "apply": TaskComplexity.ASSISTED,
                },
                priority=4,
            )
        )

        # Export patterns
        self.register(
            RoutingPattern(
                keywords=[
                    "export",
                    "save curve",
                    "qtr",
                    "piezography",
                    "csv",
                    "download",
                ],
                category=TaskCategory.EXPORT,
                default_complexity=TaskComplexity.SIMPLE,
                complexity_modifiers={
                    "all formats": TaskComplexity.ASSISTED,
                },
                priority=3,
            )
        )

    def register(self, pattern: RoutingPattern) -> None:
        """Register a new routing pattern."""
        self._patterns.append(pattern)
        # Keep patterns sorted by priority (descending)
        self._patterns.sort(key=lambda p: p.priority, reverse=True)

    def unregister(self, keywords: list[str]) -> bool:
        """Unregister a pattern by its keywords."""
        original_length = len(self._patterns)
        self._patterns = [
            p for p in self._patterns if set(p.keywords) != set(keywords)
        ]
        return len(self._patterns) < original_length

    def find_matching_patterns(self, task: str) -> list[RoutingPattern]:
        """Find all patterns that match the given task."""
        return [p for p in self._patterns if p.matches(task)]

    def get_best_match(self, task: str) -> Optional[RoutingPattern]:
        """Get the highest priority matching pattern."""
        matches = self.find_matching_patterns(task)
        return matches[0] if matches else None

    def list_patterns(self) -> list[RoutingPattern]:
        """List all registered patterns."""
        return self._patterns.copy()


class TaskRouter:
    """
    Routes tasks to appropriate capability layers.

    Uses pattern matching and optional LLM classification to determine
    the appropriate category and complexity for incoming tasks.
    """

    def __init__(
        self,
        settings: Optional[RouterSettings] = None,
        pattern_registry: Optional[PatternRegistry] = None,
    ) -> None:
        """
        Initialize the task router.

        Args:
            settings: Router configuration settings.
            pattern_registry: Custom pattern registry (uses default if None).
        """
        self.settings = settings or RouterSettings()
        self.patterns = pattern_registry or PatternRegistry()

    def route(self, task: str, context: Optional[dict[str, Any]] = None) -> RoutingResult:
        """
        Route a task to the appropriate capability layer.

        Args:
            task: The task description to route.
            context: Optional context (e.g., current state, history).

        Returns:
            RoutingResult with category, complexity, and routing metadata.
        """
        context = context or {}

        # Step 1: Pattern-based matching
        best_pattern = self.patterns.get_best_match(task)

        if best_pattern:
            complexity = best_pattern.determine_complexity(task)
            confidence = self._calculate_confidence(task, best_pattern)

            # Check if we need LLM classification for low confidence
            if (
                self.settings.use_llm_classification
                and confidence < self.settings.llm_classification_threshold
            ):
                # LLM classification would go here
                # For now, we proceed with pattern-based result
                pass

            return RoutingResult(
                category=best_pattern.category,
                complexity=complexity,
                confidence=confidence,
                matched_pattern=", ".join(best_pattern.keywords[:3]),
                suggested_template=self._get_template_for_category(best_pattern.category),
                suggested_tools=self._get_tools_for_category(best_pattern.category),
                metadata={
                    "pattern_priority": best_pattern.priority,
                    "word_count": len(task.split()),
                    "context_provided": bool(context),
                },
            )

        # Step 2: Heuristic-based fallback
        return self._heuristic_routing(task, context)

    def _calculate_confidence(self, task: str, pattern: RoutingPattern) -> float:
        """Calculate confidence score for a pattern match."""
        task_lower = task.lower()
        matched_keywords = sum(1 for kw in pattern.keywords if kw in task_lower)
        total_keywords = len(pattern.keywords)

        # Base confidence from keyword matches
        base_confidence = min(matched_keywords / min(total_keywords, 3), 1.0)

        # Boost confidence for longer matching phrases
        phrase_boost = 0.0
        for kw in pattern.keywords:
            if len(kw.split()) > 1 and kw in task_lower:
                phrase_boost += 0.1

        return min(base_confidence + phrase_boost, 1.0)

    def _heuristic_routing(
        self, task: str, context: dict[str, Any]
    ) -> RoutingResult:
        """Fallback heuristic routing when no pattern matches."""
        word_count = len(task.split())

        # Determine complexity based on task length
        if word_count <= self.settings.simple_max_words:
            complexity = TaskComplexity.SIMPLE
        elif word_count >= self.settings.autonomous_min_words:
            complexity = TaskComplexity.AUTONOMOUS
        else:
            complexity = TaskComplexity.ASSISTED

        # Check for question indicators
        if task.strip().endswith("?") or task.lower().startswith(
            ("what", "how", "why", "when", "where", "can", "is", "are", "do", "does")
        ):
            complexity = TaskComplexity.ASSISTED

        return RoutingResult(
            category=TaskCategory.GENERAL,
            complexity=complexity,
            confidence=0.3,  # Low confidence for heuristic routing
            suggested_template=None,
            suggested_tools=[],
            metadata={
                "routing_method": "heuristic",
                "word_count": word_count,
            },
        )

    def _get_template_for_category(self, category: TaskCategory) -> Optional[str]:
        """Get the suggested plan template for a category."""
        templates = {
            TaskCategory.CALIBRATION: "calibrate",
            TaskCategory.ANALYSIS: "analyze",
            TaskCategory.TROUBLESHOOTING: "troubleshoot",
            TaskCategory.RECIPE: "recipe",
            TaskCategory.PREDICTION: "predict",
        }
        return templates.get(category)

    def _get_tools_for_category(self, category: TaskCategory) -> list[str]:
        """Get suggested tools for a category."""
        tool_mapping = {
            TaskCategory.CALIBRATION: [
                "analyze_densities",
                "generate_curve",
                "search_calibrations",
            ],
            TaskCategory.CHEMISTRY: [
                "suggest_parameters",
                "search_calibrations",
            ],
            TaskCategory.QUALITY: [
                "analyze_densities",
                "compare_calibrations",
            ],
            TaskCategory.TROUBLESHOOTING: [
                "analyze_densities",
                "search_calibrations",
            ],
            TaskCategory.ANALYSIS: [
                "analyze_densities",
                "compare_calibrations",
            ],
            TaskCategory.PREDICTION: [
                "predict_response",
                "suggest_parameters",
            ],
            TaskCategory.RECIPE: [
                "search_calibrations",
                "suggest_parameters",
                "create_test_plan",
            ],
            TaskCategory.EXPORT: [
                "generate_curve",
            ],
        }
        return tool_mapping.get(category, [])

    def escalate(self, result: RoutingResult) -> RoutingResult:
        """
        Escalate a routing result to higher complexity.

        Args:
            result: The current routing result.

        Returns:
            New RoutingResult with escalated complexity.
        """
        if not self.settings.allow_escalation:
            return result

        escalation_map = {
            TaskComplexity.SIMPLE: TaskComplexity.ASSISTED,
            TaskComplexity.ASSISTED: TaskComplexity.AUTONOMOUS,
            TaskComplexity.AUTONOMOUS: TaskComplexity.AUTONOMOUS,
        }

        new_complexity = escalation_map[result.complexity]

        return RoutingResult(
            category=result.category,
            complexity=new_complexity,
            confidence=result.confidence,
            matched_pattern=result.matched_pattern,
            suggested_template=result.suggested_template,
            suggested_tools=result.suggested_tools,
            metadata={
                **result.metadata,
                "escalated_from": result.complexity.value,
            },
        )


def create_router(
    settings: Optional[RouterSettings] = None,
    custom_patterns: Optional[list[RoutingPattern]] = None,
) -> TaskRouter:
    """
    Factory function to create a configured task router.

    Args:
        settings: Optional router settings.
        custom_patterns: Optional list of custom patterns to add.

    Returns:
        Configured TaskRouter instance.
    """
    registry = PatternRegistry()

    if custom_patterns:
        for pattern in custom_patterns:
            registry.register(pattern)

    return TaskRouter(settings=settings, pattern_registry=registry)
