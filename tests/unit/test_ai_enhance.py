"""
Tests for the AI curve enhancement module.
"""

import numpy as np
import pytest

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.ai_enhance import (
    CurveAIEnhancer,
    CurveIssue,
    EnhancementGoal,
    EnhancementResult,
    enhance_curve,
)


@pytest.fixture
def linear_curve():
    """Create a linear curve for testing."""
    inputs = list(np.linspace(0, 1, 256))
    outputs = list(np.linspace(0, 1, 256))
    return CurveData(
        name="Linear Test",
        input_values=inputs,
        output_values=outputs,
    )


@pytest.fixture
def nonlinear_curve():
    """Create a non-linear (gamma 2.2) curve for testing."""
    inputs = list(np.linspace(0, 1, 256))
    outputs = list(np.power(np.linspace(0, 1, 256), 2.2))
    return CurveData(
        name="Gamma 2.2",
        input_values=inputs,
        output_values=outputs,
    )


@pytest.fixture
def noisy_curve():
    """Create a noisy curve for testing."""
    np.random.seed(42)  # For reproducibility
    inputs = list(np.linspace(0, 1, 256))
    base = np.linspace(0, 1, 256)
    noise = np.random.normal(0, 0.05, 256)
    outputs = list(np.clip(base + noise, 0, 1))
    return CurveData(
        name="Noisy Curve",
        input_values=inputs,
        output_values=outputs,
    )


@pytest.fixture
def non_monotonic_curve():
    """Create a non-monotonic curve for testing."""
    inputs = list(np.linspace(0, 1, 20))
    outputs = [
        0.0,
        0.1,
        0.08,
        0.2,
        0.18,
        0.3,
        0.35,
        0.4,
        0.38,
        0.5,
        0.55,
        0.6,
        0.58,
        0.7,
        0.75,
        0.8,
        0.78,
        0.9,
        0.95,
        1.0,
    ]
    return CurveData(
        name="Non-Monotonic",
        input_values=inputs,
        output_values=outputs,
    )


class TestEnhancementGoal:
    """Tests for EnhancementGoal enum."""

    def test_all_goals_exist(self):
        """Test all expected goals are defined."""
        assert EnhancementGoal.LINEARIZATION
        assert EnhancementGoal.MAXIMIZE_RANGE
        assert EnhancementGoal.SMOOTH_GRADATION
        assert EnhancementGoal.HIGHLIGHT_DETAIL
        assert EnhancementGoal.SHADOW_DETAIL
        assert EnhancementGoal.PORTRAIT_AESTHETIC
        assert EnhancementGoal.LANDSCAPE_AESTHETIC
        assert EnhancementGoal.HIGH_KEY
        assert EnhancementGoal.LOW_KEY

    def test_goal_values(self):
        """Test goal string values."""
        assert EnhancementGoal.LINEARIZATION.value == "linearization"
        assert EnhancementGoal.MAXIMIZE_RANGE.value == "maximize_range"


class TestCurveIssue:
    """Tests for CurveIssue dataclass."""

    def test_issue_creation(self):
        """Test creating a curve issue."""
        issue = CurveIssue(
            issue_type="non_monotonic",
            severity="high",
            location="overall",
            description="Curve has 5 reversals",
            suggested_fix="Apply monotonicity enforcement",
        )

        assert issue.issue_type == "non_monotonic"
        assert issue.severity == "high"
        assert issue.location == "overall"


class TestCurveAIEnhancer:
    """Tests for CurveAIEnhancer class."""

    def test_enhancer_initialization(self):
        """Test enhancer initialization."""
        enhancer = CurveAIEnhancer()

        assert enhancer.modifier is not None
        assert enhancer.analyzer is not None

    @pytest.mark.asyncio
    async def test_analyze_and_enhance_linear(self, linear_curve):
        """Test analysis of a linear curve."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            linear_curve,
            goal=EnhancementGoal.LINEARIZATION,
            auto_apply=True,
        )

        assert isinstance(result, EnhancementResult)
        assert result.original_curve == linear_curve
        assert result.enhanced_curve is not None
        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_analyze_and_enhance_nonlinear(self, nonlinear_curve):
        """Test analysis of a non-linear curve."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            nonlinear_curve,
            goal=EnhancementGoal.LINEARIZATION,
            auto_apply=True,
        )

        assert result.analysis is not None
        assert "max_linearity_error" in result.analysis

    @pytest.mark.asyncio
    async def test_analyze_and_enhance_noisy(self, noisy_curve):
        """Test analysis of a noisy curve."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            noisy_curve,
            goal=EnhancementGoal.SMOOTH_GRADATION,
            auto_apply=True,
        )

        # Should apply smoothing
        assert len(result.adjustments_applied) > 0

        # Enhanced curve should be smoother
        orig_rough = np.std(np.diff(np.diff(noisy_curve.output_values)))
        new_rough = np.std(np.diff(np.diff(result.enhanced_curve.output_values)))
        assert new_rough < orig_rough

    @pytest.mark.asyncio
    async def test_analyze_and_enhance_non_monotonic(self, non_monotonic_curve):
        """Test analysis of a non-monotonic curve."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            non_monotonic_curve,
            goal=EnhancementGoal.LINEARIZATION,
            auto_apply=True,
        )

        # Should fix monotonicity
        diffs = np.diff(result.enhanced_curve.output_values)
        assert all(d >= -0.001 for d in diffs), "Should be monotonic after enhancement"

    @pytest.mark.asyncio
    async def test_analyze_without_auto_apply(self, nonlinear_curve):
        """Test analysis without auto-applying enhancements."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            nonlinear_curve,
            goal=EnhancementGoal.LINEARIZATION,
            auto_apply=False,
        )

        # Enhanced curve should be same as original
        assert np.allclose(
            result.enhanced_curve.output_values,
            nonlinear_curve.output_values,
        )
        assert len(result.adjustments_applied) == 0


class TestAnalysisForLLM:
    """Tests for LLM-formatted analysis."""

    def test_analyze_for_llm(self, nonlinear_curve):
        """Test LLM-formatted analysis output."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer.analyze_for_llm(nonlinear_curve)

        assert "Curve Analysis" in analysis
        assert "Basic Statistics" in analysis
        assert "Linearity Analysis" in analysis
        assert "Tonal Distribution" in analysis

    def test_analyze_for_llm_with_recommendations(self, noisy_curve):
        """Test LLM analysis includes recommendations."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer.analyze_for_llm(noisy_curve, include_recommendations=True)

        # May or may not have recommendations depending on issues found
        assert len(analysis) > 0

    def test_analyze_for_llm_without_recommendations(self, linear_curve):
        """Test LLM analysis without recommendations."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer.analyze_for_llm(linear_curve, include_recommendations=False)

        assert "Recommendations" not in analysis or "Issues" not in analysis


class TestEnhancementPrompt:
    """Tests for enhancement prompt generation."""

    def test_get_enhancement_prompt(self, nonlinear_curve):
        """Test prompt generation."""
        enhancer = CurveAIEnhancer()
        prompt = enhancer.get_enhancement_prompt(
            nonlinear_curve,
            EnhancementGoal.LINEARIZATION,
        )

        assert "Enhancement Goal" in prompt
        assert "Linearization" in prompt
        assert "Task" in prompt
        assert "Brightness" in prompt

    def test_get_enhancement_prompt_with_requirements(self, nonlinear_curve):
        """Test prompt generation with user requirements."""
        enhancer = CurveAIEnhancer()
        prompt = enhancer.get_enhancement_prompt(
            nonlinear_curve,
            EnhancementGoal.PORTRAIT_AESTHETIC,
            user_requirements="Optimize for skin tones",
        )

        assert "User Requirements" in prompt
        assert "skin tones" in prompt


class TestGoalSpecificEnhancements:
    """Tests for goal-specific enhancement behaviors."""

    @pytest.mark.asyncio
    async def test_highlight_detail_goal(self, nonlinear_curve):
        """Test highlight detail enhancement goal."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            nonlinear_curve,
            goal=EnhancementGoal.HIGHLIGHT_DETAIL,
            auto_apply=True,
        )

        assert result.enhanced_curve is not None

    @pytest.mark.asyncio
    async def test_shadow_detail_goal(self, nonlinear_curve):
        """Test shadow detail enhancement goal."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            nonlinear_curve,
            goal=EnhancementGoal.SHADOW_DETAIL,
            auto_apply=True,
        )

        assert result.enhanced_curve is not None

    @pytest.mark.asyncio
    async def test_portrait_aesthetic_goal(self, linear_curve):
        """Test portrait aesthetic enhancement."""
        enhancer = CurveAIEnhancer()
        result = await enhancer.analyze_and_enhance(
            linear_curve,
            goal=EnhancementGoal.PORTRAIT_AESTHETIC,
            auto_apply=True,
        )

        # Should add some contrast
        if result.adjustments_applied:
            assert any("contrast" in adj.lower() for adj in result.adjustments_applied)


class TestLLMResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_llm_response_basic(self):
        """Test parsing basic LLM response."""
        enhancer = CurveAIEnhancer()

        response = """
        Based on my analysis, I recommend:
        - brightness: 0.1
        - contrast: 0.2
        - gamma: 1.1
        - smoothing: 0.3
        """

        adjustments = enhancer._parse_llm_response(response)

        assert adjustments["brightness"] == pytest.approx(0.1)
        assert adjustments["contrast"] == pytest.approx(0.2)
        assert adjustments["gamma"] == pytest.approx(1.1)
        assert adjustments["smoothing"] == pytest.approx(0.3)

    def test_parse_llm_response_with_equals(self):
        """Test parsing LLM response with equals sign."""
        enhancer = CurveAIEnhancer()

        response = "brightness = -0.15, contrast = 0.25"

        adjustments = enhancer._parse_llm_response(response)

        assert adjustments["brightness"] == pytest.approx(-0.15)
        assert adjustments["contrast"] == pytest.approx(0.25)

    def test_parse_llm_response_clamping(self):
        """Test that parsed values are clamped to valid ranges."""
        enhancer = CurveAIEnhancer()

        response = "brightness: 5.0, gamma: 100.0, smoothing: 10.0"

        adjustments = enhancer._parse_llm_response(response)

        assert adjustments["brightness"] == 1.0  # Clamped from 5.0
        assert adjustments["gamma"] == 10.0  # Clamped from 100.0
        assert adjustments["smoothing"] == 1.0  # Clamped from 10.0

    def test_parse_llm_response_no_matches(self):
        """Test parsing when no values are found."""
        enhancer = CurveAIEnhancer()

        response = "This response has no adjustment values."

        adjustments = enhancer._parse_llm_response(response)

        assert adjustments["brightness"] == 0.0
        assert adjustments["gamma"] == 1.0


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_confidence_high_for_simple_curve(self, linear_curve):
        """Test high confidence for simple, clean curve."""
        enhancer = CurveAIEnhancer()
        issues = []
        adjustments = []

        confidence = enhancer._calculate_confidence(issues, adjustments)

        assert confidence >= 0.9

    def test_confidence_lower_for_issues(self):
        """Test confidence decreases with issues."""
        enhancer = CurveAIEnhancer()
        issues = [
            CurveIssue("test", "high", "overall", "test", "test"),
            CurveIssue("test", "high", "overall", "test", "test"),
        ]
        adjustments = []

        confidence = enhancer._calculate_confidence(issues, adjustments)

        assert confidence < 0.8

    def test_confidence_minimum(self):
        """Test confidence has minimum value."""
        enhancer = CurveAIEnhancer()
        issues = [CurveIssue("test", "high", "overall", "test", "test") for _ in range(10)]
        adjustments = ["adj" for _ in range(20)]

        confidence = enhancer._calculate_confidence(issues, adjustments)

        assert confidence >= 0.3


class TestConvenienceFunction:
    """Tests for enhance_curve convenience function."""

    @pytest.mark.asyncio
    async def test_enhance_curve_basic(self, noisy_curve):
        """Test basic curve enhancement."""
        result = await enhance_curve(noisy_curve, goal="linearization")

        assert isinstance(result, EnhancementResult)
        assert result.enhanced_curve is not None

    @pytest.mark.asyncio
    async def test_enhance_curve_all_goals(self, linear_curve):
        """Test enhancement with different goals."""
        goals = ["linearization", "maximize_range", "smooth_gradation"]

        for goal in goals:
            result = await enhance_curve(linear_curve, goal=goal)
            assert result.enhanced_curve is not None


class TestAnalysisDetails:
    """Tests for detailed curve analysis."""

    def test_analysis_contains_all_fields(self, linear_curve):
        """Test that analysis contains all expected fields."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer._analyze_curve(linear_curve, None)

        assert "num_points" in analysis
        assert "min_output" in analysis
        assert "max_output" in analysis
        assert "is_monotonic" in analysis
        assert "max_linearity_error" in analysis
        assert "rms_linearity_error" in analysis
        assert "shape" in analysis
        assert "highlight_response" in analysis
        assert "midtone_response" in analysis
        assert "shadow_response" in analysis
        assert "roughness" in analysis

    def test_shape_detection_linear(self, linear_curve):
        """Test shape detection for linear curve."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer._analyze_curve(linear_curve, None)

        assert "linear" in analysis["shape"].lower()

    def test_shape_detection_gamma(self, nonlinear_curve):
        """Test shape detection for gamma curve."""
        enhancer = CurveAIEnhancer()
        analysis = enhancer._analyze_curve(nonlinear_curve, None)

        # Gamma 2.2 curve is concave (darkens midtones)
        assert "concave" in analysis["shape"].lower()
