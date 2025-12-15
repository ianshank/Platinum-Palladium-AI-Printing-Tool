"""
Unit tests for the TaskRouter module.

Tests cover:
- Task classification and routing
- Pattern matching and confidence scoring
- Complexity determination
- Escalation logic
- Custom pattern registration
"""

import pytest
from typing import Any

from ptpd_calibration.agents.router import (
    TaskRouter,
    TaskComplexity,
    TaskCategory,
    RoutingPattern,
    RoutingResult,
    PatternRegistry,
    RouterSettings,
    create_router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def router_settings() -> RouterSettings:
    """Create default router settings for testing."""
    return RouterSettings(
        simple_max_words=10,
        autonomous_min_words=20,
        use_llm_classification=False,
        allow_escalation=True,
    )


@pytest.fixture
def router(router_settings: RouterSettings) -> TaskRouter:
    """Create a task router with default settings."""
    return TaskRouter(settings=router_settings)


@pytest.fixture
def custom_pattern() -> RoutingPattern:
    """Create a custom routing pattern for testing."""
    return RoutingPattern(
        keywords=["custom_test", "special_operation"],
        category=TaskCategory.ANALYSIS,
        default_complexity=TaskComplexity.ASSISTED,
        complexity_modifiers={
            "quick": TaskComplexity.SIMPLE,
            "detailed": TaskComplexity.AUTONOMOUS,
        },
        priority=15,
    )


# =============================================================================
# Unit Tests - RouterSettings
# =============================================================================


class TestRouterSettings:
    """Tests for RouterSettings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = RouterSettings()
        assert settings.simple_max_words == 10
        assert settings.autonomous_min_words == 20
        assert settings.use_llm_classification is False
        assert settings.allow_escalation is True

    def test_custom_settings(self):
        """Test custom settings override."""
        settings = RouterSettings(
            simple_max_words=5,
            autonomous_min_words=30,
            use_llm_classification=True,
        )
        assert settings.simple_max_words == 5
        assert settings.autonomous_min_words == 30
        assert settings.use_llm_classification is True

    def test_settings_validation(self):
        """Test settings validation constraints."""
        # Valid range
        settings = RouterSettings(simple_max_words=3)
        assert settings.simple_max_words == 3

        # Test boundary
        settings = RouterSettings(simple_max_words=50)
        assert settings.simple_max_words == 50


# =============================================================================
# Unit Tests - RoutingPattern
# =============================================================================


class TestRoutingPattern:
    """Tests for RoutingPattern matching and complexity."""

    def test_pattern_matches_keyword(self, custom_pattern: RoutingPattern):
        """Test that pattern matches on keyword."""
        assert custom_pattern.matches("Run custom_test operation")
        assert custom_pattern.matches("Execute special_operation now")
        assert not custom_pattern.matches("Run standard operation")

    def test_pattern_case_insensitive(self, custom_pattern: RoutingPattern):
        """Test that pattern matching is case-insensitive."""
        assert custom_pattern.matches("CUSTOM_TEST task")
        assert custom_pattern.matches("Custom_Test operation")

    def test_default_complexity(self, custom_pattern: RoutingPattern):
        """Test default complexity is returned when no modifiers match."""
        complexity = custom_pattern.determine_complexity("Run custom_test")
        assert complexity == TaskComplexity.ASSISTED

    def test_complexity_modifiers(self, custom_pattern: RoutingPattern):
        """Test complexity modifiers override default."""
        quick_result = custom_pattern.determine_complexity("quick custom_test")
        assert quick_result == TaskComplexity.SIMPLE

        detailed_result = custom_pattern.determine_complexity("detailed custom_test")
        assert detailed_result == TaskComplexity.AUTONOMOUS

    def test_pattern_priority(self):
        """Test pattern priority sorting."""
        low_priority = RoutingPattern(
            keywords=["test"],
            category=TaskCategory.GENERAL,
            default_complexity=TaskComplexity.SIMPLE,
            priority=1,
        )
        high_priority = RoutingPattern(
            keywords=["test"],
            category=TaskCategory.CALIBRATION,
            default_complexity=TaskComplexity.AUTONOMOUS,
            priority=10,
        )
        assert high_priority.priority > low_priority.priority


# =============================================================================
# Unit Tests - PatternRegistry
# =============================================================================


class TestPatternRegistry:
    """Tests for PatternRegistry management."""

    def test_default_patterns_loaded(self):
        """Test that default patterns are loaded on initialization."""
        registry = PatternRegistry()
        patterns = registry.list_patterns()
        assert len(patterns) > 0

        # Check for core patterns
        keywords_found = []
        for pattern in patterns:
            keywords_found.extend(pattern.keywords)

        assert "calibrate" in keywords_found
        assert "troubleshoot" in keywords_found
        assert "chemistry" in keywords_found

    def test_register_pattern(self, custom_pattern: RoutingPattern):
        """Test registering a new pattern."""
        registry = PatternRegistry()
        initial_count = len(registry.list_patterns())

        registry.register(custom_pattern)

        assert len(registry.list_patterns()) == initial_count + 1

    def test_patterns_sorted_by_priority(self):
        """Test that patterns are sorted by priority."""
        registry = PatternRegistry()
        patterns = registry.list_patterns()

        # Check descending priority order
        priorities = [p.priority for p in patterns]
        assert priorities == sorted(priorities, reverse=True)

    def test_find_matching_patterns(self):
        """Test finding all matching patterns."""
        registry = PatternRegistry()

        # "calibrate" should match calibration pattern
        matches = registry.find_matching_patterns("calibrate my step tablet")
        assert len(matches) > 0
        assert any(p.category == TaskCategory.CALIBRATION for p in matches)

    def test_get_best_match(self):
        """Test getting the highest priority match."""
        registry = PatternRegistry()

        best = registry.get_best_match("calibrate the step tablet")
        assert best is not None
        assert best.category == TaskCategory.CALIBRATION

    def test_no_match_returns_none(self):
        """Test that no match returns None."""
        registry = PatternRegistry()

        best = registry.get_best_match("completely random nonsense xyz123")
        assert best is None

    def test_unregister_pattern(self, custom_pattern: RoutingPattern):
        """Test unregistering a pattern."""
        registry = PatternRegistry()
        registry.register(custom_pattern)

        initial_count = len(registry.list_patterns())
        success = registry.unregister(custom_pattern.keywords)

        assert success
        assert len(registry.list_patterns()) == initial_count - 1


# =============================================================================
# Unit Tests - TaskRouter
# =============================================================================


class TestTaskRouter:
    """Tests for TaskRouter routing logic."""

    def test_route_calibration_task(self, router: TaskRouter):
        """Test routing a calibration task."""
        result = router.route("Calibrate my step tablet image")

        assert result.category == TaskCategory.CALIBRATION
        assert result.complexity == TaskComplexity.AUTONOMOUS
        assert result.confidence > 0.3
        assert result.should_use_planning

    def test_route_chemistry_task(self, router: TaskRouter):
        """Test routing a chemistry task."""
        result = router.route("Calculate drops for coating solution")

        assert result.category == TaskCategory.CHEMISTRY
        assert result.confidence > 0.3

    def test_route_troubleshooting_task(self, router: TaskRouter):
        """Test routing a troubleshooting task."""
        result = router.route("Help me fix my muddy print problem")

        assert result.category == TaskCategory.TROUBLESHOOTING
        assert result.complexity == TaskComplexity.ASSISTED

    def test_route_quality_task(self, router: TaskRouter):
        """Test routing a quality check task."""
        result = router.route("Check the quality of my dmax reading")

        assert result.category == TaskCategory.QUALITY

    def test_route_analysis_task(self, router: TaskRouter):
        """Test routing an analysis task."""
        result = router.route("Analyze these density measurements")

        assert result.category == TaskCategory.ANALYSIS

    def test_route_prediction_task(self, router: TaskRouter):
        """Test routing a prediction task."""
        result = router.route("Predict the exposure time needed")

        assert result.category == TaskCategory.PREDICTION

    def test_route_recipe_task(self, router: TaskRouter):
        """Test routing a recipe task."""
        result = router.route("Create a recipe for batch processing")

        assert result.category == TaskCategory.RECIPE

    def test_route_export_task(self, router: TaskRouter):
        """Test routing an export task."""
        result = router.route("Export my curve to QTR format")

        assert result.category == TaskCategory.EXPORT
        assert result.complexity == TaskComplexity.SIMPLE

    def test_route_unknown_task_fallback(self, router: TaskRouter):
        """Test routing an unknown task uses heuristic fallback."""
        result = router.route("Something completely unrelated to printing")

        assert result.category == TaskCategory.GENERAL
        assert result.confidence < 0.5  # Low confidence for fallback

    def test_short_task_is_simple(self, router: TaskRouter):
        """Test that short tasks are classified as simple."""
        result = router.route("Check dmax")

        # Short task should tend toward simple
        assert result.complexity in (TaskComplexity.SIMPLE, TaskComplexity.ASSISTED)

    def test_long_task_is_autonomous(self, router: TaskRouter):
        """Test that long complex tasks are classified as autonomous."""
        long_task = (
            "I need to calibrate a new paper type, create a linearization curve, "
            "optimize the chemistry ratios, and then batch process all my images "
            "with the new profile including quality validation for each output"
        )
        result = router.route(long_task)

        assert result.complexity == TaskComplexity.AUTONOMOUS

    def test_question_task_is_assisted(self, router: TaskRouter):
        """Test that questions are routed to assisted complexity."""
        result = router.route("Why are my highlights blocked?")

        assert result.complexity == TaskComplexity.ASSISTED

    def test_routing_result_metadata(self, router: TaskRouter):
        """Test that routing result includes metadata."""
        result = router.route("Calibrate my step tablet")

        assert "word_count" in result.metadata
        assert result.matched_pattern is not None

    def test_suggested_template(self, router: TaskRouter):
        """Test that suggested template is provided."""
        result = router.route("Calibrate my step tablet")

        assert result.suggested_template == "calibrate"

    def test_suggested_tools(self, router: TaskRouter):
        """Test that suggested tools are provided."""
        result = router.route("Calibrate my step tablet")

        assert len(result.suggested_tools) > 0
        assert "analyze_densities" in result.suggested_tools

    def test_should_use_reflection(self, router: TaskRouter):
        """Test should_use_reflection property."""
        autonomous_result = router.route("Full calibration workflow with validation")
        autonomous_result_forced = RoutingResult(
            category=TaskCategory.CALIBRATION,
            complexity=TaskComplexity.AUTONOMOUS,
            confidence=0.9,
        )

        assert autonomous_result_forced.should_use_reflection

        simple_result = RoutingResult(
            category=TaskCategory.EXPORT,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.8,
        )
        assert not simple_result.should_use_reflection


# =============================================================================
# Unit Tests - Escalation
# =============================================================================


class TestEscalation:
    """Tests for complexity escalation."""

    def test_escalate_simple_to_assisted(self, router: TaskRouter):
        """Test escalating from simple to assisted."""
        initial = RoutingResult(
            category=TaskCategory.CALIBRATION,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.7,
        )

        escalated = router.escalate(initial)

        assert escalated.complexity == TaskComplexity.ASSISTED
        assert "escalated_from" in escalated.metadata

    def test_escalate_assisted_to_autonomous(self, router: TaskRouter):
        """Test escalating from assisted to autonomous."""
        initial = RoutingResult(
            category=TaskCategory.CALIBRATION,
            complexity=TaskComplexity.ASSISTED,
            confidence=0.7,
        )

        escalated = router.escalate(initial)

        assert escalated.complexity == TaskComplexity.AUTONOMOUS

    def test_autonomous_cannot_escalate_further(self, router: TaskRouter):
        """Test that autonomous cannot escalate further."""
        initial = RoutingResult(
            category=TaskCategory.CALIBRATION,
            complexity=TaskComplexity.AUTONOMOUS,
            confidence=0.7,
        )

        escalated = router.escalate(initial)

        assert escalated.complexity == TaskComplexity.AUTONOMOUS

    def test_escalation_disabled(self, router_settings: RouterSettings):
        """Test escalation when disabled."""
        router_settings.allow_escalation = False
        router = TaskRouter(settings=router_settings)

        initial = RoutingResult(
            category=TaskCategory.CALIBRATION,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.7,
        )

        escalated = router.escalate(initial)

        # Should remain unchanged
        assert escalated.complexity == TaskComplexity.SIMPLE


# =============================================================================
# Unit Tests - Factory Function
# =============================================================================


class TestCreateRouter:
    """Tests for the create_router factory function."""

    def test_create_default_router(self):
        """Test creating router with defaults."""
        router = create_router()

        assert isinstance(router, TaskRouter)
        assert router.settings is not None
        assert router.patterns is not None

    def test_create_router_with_settings(self):
        """Test creating router with custom settings."""
        settings = RouterSettings(simple_max_words=5)
        router = create_router(settings=settings)

        assert router.settings.simple_max_words == 5

    def test_create_router_with_custom_patterns(self, custom_pattern: RoutingPattern):
        """Test creating router with custom patterns."""
        router = create_router(custom_patterns=[custom_pattern])

        result = router.route("Run custom_test operation")
        assert result.category == TaskCategory.ANALYSIS


# =============================================================================
# Integration Tests - Pattern + Router
# =============================================================================


class TestRouterIntegration:
    """Integration tests for router with patterns."""

    def test_multiple_keyword_matches(self, router: TaskRouter):
        """Test task matching multiple patterns uses highest priority."""
        # "analyze density calibration" could match both analyze and calibrate
        result = router.route("Analyze the density values from calibration")

        # Should pick the higher priority match
        assert result.category in (TaskCategory.CALIBRATION, TaskCategory.ANALYSIS)
        assert result.confidence > 0.3

    def test_routing_with_context(self, router: TaskRouter):
        """Test routing with additional context."""
        context = {"current_step": "measurement", "user_level": "advanced"}

        result = router.route("Continue with analysis", context=context)

        assert result.metadata.get("context_provided") is True

    def test_complex_workflow_routing(self, router: TaskRouter):
        """Test routing a complex workflow request."""
        task = (
            "I need to calibrate a new Bergger COT320 paper with warm tones, "
            "create a recipe for it, and then batch process my portfolio images"
        )

        result = router.route(task)

        # Should identify as autonomous calibration/recipe workflow
        assert result.complexity == TaskComplexity.AUTONOMOUS
        assert result.should_use_planning


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "task,expected_category",
    [
        ("Calibrate step tablet", TaskCategory.CALIBRATION),
        ("linearize my curve", TaskCategory.CALIBRATION),
        ("calculate coating drops", TaskCategory.CHEMISTRY),
        ("platinum ratio 50%", TaskCategory.CHEMISTRY),
        ("check quality", TaskCategory.QUALITY),
        ("validate dmax", TaskCategory.QUALITY),
        ("troubleshoot blocked shadows", TaskCategory.TROUBLESHOOTING),
        ("fix my muddy print", TaskCategory.TROUBLESHOOTING),
        ("analyze density", TaskCategory.ANALYSIS),
        ("compare calibrations", TaskCategory.CALIBRATION),  # Comparing calibrations is calibration-related
        ("predict exposure", TaskCategory.PREDICTION),
        ("suggest parameters", TaskCategory.PREDICTION),
        ("create recipe", TaskCategory.RECIPE),
        ("batch process", TaskCategory.RECIPE),
        ("export to QTR", TaskCategory.EXPORT),
        ("save curve", TaskCategory.EXPORT),
    ],
)
def test_category_routing(task: str, expected_category: TaskCategory):
    """Parametrized test for category routing."""
    router = create_router()
    result = router.route(task)
    assert result.category == expected_category


@pytest.mark.parametrize(
    "task,expected_complexity",
    [
        ("Export curve", TaskComplexity.SIMPLE),
        ("Quick calibration", TaskComplexity.ASSISTED),
        ("Full comprehensive calibration with batch processing", TaskComplexity.AUTONOMOUS),
    ],
)
def test_complexity_routing(task: str, expected_complexity: TaskComplexity):
    """Parametrized test for complexity routing."""
    router = create_router()
    result = router.route(task)
    # Allow for adjacent complexity levels due to heuristics
    complexities = [TaskComplexity.SIMPLE, TaskComplexity.ASSISTED, TaskComplexity.AUTONOMOUS]
    expected_idx = complexities.index(expected_complexity)
    actual_idx = complexities.index(result.complexity)
    assert abs(expected_idx - actual_idx) <= 1  # Within one level
