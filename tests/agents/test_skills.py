"""
Unit and integration tests for skill modules.

Tests cover:
- Skill base class functionality
- CalibrationSkill operations
- ChemistrySkill calculations
- QualitySkill assessments
- TroubleshootingSkill diagnosis
- SkillRegistry management
"""

import pytest
from typing import Any

from ptpd_calibration.agents.skills import (
    Skill,
    SkillResult,
    SkillContext,
    SkillRegistry,
    CalibrationSkill,
    ChemistrySkill,
    QualitySkill,
    TroubleshootingSkill,
    create_default_skills,
)
from ptpd_calibration.agents.skills.base import SkillCategory


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_densities():
    """Create sample density measurements."""
    return [
        0.08, 0.15, 0.25, 0.38, 0.52, 0.68, 0.85, 1.02,
        1.20, 1.38, 1.55, 1.72, 1.88, 2.02, 2.15, 2.26,
        2.35, 2.42, 2.48, 2.52, 2.55
    ]


@pytest.fixture
def poor_densities():
    """Create poor density measurements."""
    return [
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        1.00, 1.05, 1.10, 1.15, 1.20
    ]


@pytest.fixture
def skill_context(sample_densities):
    """Create a skill context with sample data."""
    return SkillContext(
        paper_type="Arches Platine",
        metal_ratio=0.5,
        exposure_time=180.0,
        densities=sample_densities,
        humidity=50.0,
        temperature=21.0,
        user_level="intermediate",
    )


@pytest.fixture
def calibration_skill():
    """Create a calibration skill instance."""
    return CalibrationSkill()


@pytest.fixture
def chemistry_skill():
    """Create a chemistry skill instance."""
    return ChemistrySkill()


@pytest.fixture
def quality_skill():
    """Create a quality skill instance."""
    return QualitySkill()


@pytest.fixture
def troubleshooting_skill():
    """Create a troubleshooting skill instance."""
    return TroubleshootingSkill()


@pytest.fixture
def skill_registry():
    """Create a skill registry with default skills."""
    return create_default_skills()


# =============================================================================
# Unit Tests - SkillResult
# =============================================================================


class TestSkillResult:
    """Tests for SkillResult class."""

    def test_success_result(self):
        """Test creating a success result."""
        result = SkillResult.success_result(
            data={"key": "value"},
            message="Success",
            confidence=0.9,
        )
        assert result.success is True
        assert result.confidence == 0.9
        assert result.data["key"] == "value"

    def test_failure_result(self):
        """Test creating a failure result."""
        result = SkillResult.failure_result(
            message="Failed",
            error="Something went wrong",
        )
        assert result.success is False
        assert result.confidence == 0.0
        assert result.data["error"] == "Something went wrong"


# =============================================================================
# Unit Tests - SkillContext
# =============================================================================


class TestSkillContext:
    """Tests for SkillContext class."""

    def test_context_defaults(self):
        """Test context default values."""
        context = SkillContext()
        assert context.paper_type is None
        assert context.user_level == "intermediate"
        assert context.preferred_format == "qtr"
        assert context.previous_results == []

    def test_context_with_values(self, sample_densities):
        """Test context with provided values."""
        context = SkillContext(
            paper_type="Bergger COT320",
            metal_ratio=0.7,
            densities=sample_densities,
        )
        assert context.paper_type == "Bergger COT320"
        assert context.metal_ratio == 0.7
        assert len(context.densities) == 21


# =============================================================================
# Unit Tests - CalibrationSkill
# =============================================================================


class TestCalibrationSkill:
    """Tests for CalibrationSkill."""

    def test_skill_properties(self, calibration_skill):
        """Test skill property values."""
        assert calibration_skill.name == "calibration"
        assert calibration_skill.category == SkillCategory.CALIBRATION
        assert len(calibration_skill.get_capabilities()) > 0

    def test_can_handle_calibration_task(self, calibration_skill):
        """Test can_handle for calibration tasks."""
        assert calibration_skill.can_handle("Calibrate my step tablet") >= 0.9
        assert calibration_skill.can_handle("Linearize the curve") >= 0.9
        assert calibration_skill.can_handle("Random task") < 0.3

    def test_extract_densities(self, calibration_skill, skill_context):
        """Test density extraction."""
        result = calibration_skill.execute(
            "Extract density measurements",
            context=skill_context,
        )
        assert result.success is True
        assert "densities" in result.data
        assert "dmin" in result.data
        assert "dmax" in result.data

    def test_generate_curve(self, calibration_skill, sample_densities):
        """Test curve generation."""
        result = calibration_skill.execute(
            "Generate linearization curve",
            densities=sample_densities,
            name="Test Curve",
        )
        assert result.success is True
        assert "curve_id" in result.data
        assert result.data["curve_name"] == "Test Curve"

    def test_analyze_calibration(self, calibration_skill, sample_densities):
        """Test calibration analysis."""
        result = calibration_skill.execute(
            "Analyze calibration quality",
            densities=sample_densities,
        )
        assert result.success is True
        assert "quality_grade" in result.data
        assert "is_monotonic" in result.data

    def test_full_calibration(self, calibration_skill, skill_context, sample_densities):
        """Test full calibration workflow."""
        result = calibration_skill.execute(
            "Full calibration",
            context=skill_context,
            densities=sample_densities,
            paper_type="Arches Platine",
        )
        assert result.success is True
        assert "analysis" in result.data
        assert "curve" in result.data
        assert result.data["workflow_complete"] is True


# =============================================================================
# Unit Tests - ChemistrySkill
# =============================================================================


class TestChemistrySkill:
    """Tests for ChemistrySkill."""

    def test_skill_properties(self, chemistry_skill):
        """Test skill property values."""
        assert chemistry_skill.name == "chemistry"
        assert chemistry_skill.category == SkillCategory.CHEMISTRY

    def test_can_handle_chemistry_task(self, chemistry_skill):
        """Test can_handle for chemistry tasks."""
        assert chemistry_skill.can_handle("Calculate coating drops") >= 0.9
        assert chemistry_skill.can_handle("What platinum ratio for warm tones") >= 0.5
        assert chemistry_skill.can_handle("Random task") < 0.3

    def test_calculate_coating(self, chemistry_skill):
        """Test coating calculation."""
        result = chemistry_skill.execute(
            "Calculate coating for 8x10 print",
            print_width=8.0,
            print_height=10.0,
            metal_ratio=0.5,
        )
        assert result.success is True
        assert "components" in result.data
        assert "ferric_oxalate" in result.data["components"]
        assert "platinum" in result.data["components"]
        assert "palladium" in result.data["components"]

    def test_recommend_ratio_warm(self, chemistry_skill):
        """Test warm tone ratio recommendation."""
        result = chemistry_skill.execute(
            "What ratio for warm tones",
            target_tone="warm",
        )
        assert result.success is True
        assert result.data["recommended_ratio"] == 0.2

    def test_recommend_ratio_cool(self, chemistry_skill):
        """Test cool tone ratio recommendation."""
        result = chemistry_skill.execute(
            "What ratio for cool tones",
            target_tone="cool",
        )
        assert result.success is True
        assert result.data["recommended_ratio"] == 0.8

    def test_estimate_cost(self, chemistry_skill):
        """Test cost estimation."""
        result = chemistry_skill.execute(
            "Estimate cost",
            print_width=8.0,
            print_height=10.0,
            num_prints=5,
        )
        assert result.success is True
        assert "cost_per_print" in result.data
        assert "total_cost" in result.data
        assert result.data["num_prints"] == 5


# =============================================================================
# Unit Tests - QualitySkill
# =============================================================================


class TestQualitySkill:
    """Tests for QualitySkill."""

    def test_skill_properties(self, quality_skill):
        """Test skill property values."""
        assert quality_skill.name == "quality"
        assert quality_skill.category == SkillCategory.QUALITY

    def test_can_handle_quality_task(self, quality_skill):
        """Test can_handle for quality tasks."""
        assert quality_skill.can_handle("Check quality") >= 0.9
        assert quality_skill.can_handle("Validate pre-print conditions") >= 0.9
        assert quality_skill.can_handle("Random task") < 0.3

    def test_pre_print_check_pass(self, quality_skill, skill_context):
        """Test pre-print check with good conditions."""
        result = quality_skill.execute(
            "Pre-print check",
            context=skill_context,
        )
        assert result.success is True
        assert "checks" in result.data
        assert result.data["recommendation"] in ("go", "caution")

    def test_post_print_analysis_good(self, quality_skill, sample_densities):
        """Test post-print analysis with good data."""
        result = quality_skill.execute(
            "Post-print analysis",
            densities=sample_densities,
        )
        assert result.success is True
        assert "quality_grade" in result.data
        assert result.data["quality_grade"] in ("A", "B")

    def test_post_print_analysis_poor(self, quality_skill, poor_densities):
        """Test post-print analysis with poor data."""
        result = quality_skill.execute(
            "Post-print analysis",
            densities=poor_densities,
        )
        assert result.success is True
        assert len(result.data["issues"]) > 0

    def test_check_environment_good(self, quality_skill):
        """Test environment check with good conditions."""
        result = quality_skill.execute(
            "Check environment",
            humidity=50.0,
            temperature=21.0,
        )
        assert result.success is True
        assert result.data["passed"] is True

    def test_check_environment_bad(self, quality_skill):
        """Test environment check with bad conditions."""
        result = quality_skill.execute(
            "Check environment",
            humidity=30.0,  # Too low
            temperature=28.0,  # Too high
        )
        assert result.success is True
        assert result.data["passed"] is False
        assert len(result.data["issues"]) >= 2


# =============================================================================
# Unit Tests - TroubleshootingSkill
# =============================================================================


class TestTroubleshootingSkill:
    """Tests for TroubleshootingSkill."""

    def test_skill_properties(self, troubleshooting_skill):
        """Test skill property values."""
        assert troubleshooting_skill.name == "troubleshooting"
        assert troubleshooting_skill.category == SkillCategory.TROUBLESHOOTING

    def test_can_handle_troubleshooting_task(self, troubleshooting_skill):
        """Test can_handle for troubleshooting tasks."""
        assert troubleshooting_skill.can_handle("Troubleshoot my print") >= 0.9
        assert troubleshooting_skill.can_handle("Why is my print muddy") >= 0.8
        assert troubleshooting_skill.can_handle("Random task") < 0.3

    def test_diagnose_muddy(self, troubleshooting_skill):
        """Test diagnosing muddy print."""
        result = troubleshooting_skill.execute(
            "My print is muddy",
            symptoms=["muddy shadows"],
        )
        assert result.success is True
        assert len(result.data["diagnoses"]) > 0
        assert any("exposure" in d["solution"].lower() for d in result.data["diagnoses"])

    def test_diagnose_faded(self, troubleshooting_skill):
        """Test diagnosing faded print."""
        result = troubleshooting_skill.execute(
            "Print looks faded",
            symptoms=["faded"],
        )
        assert result.success is True
        assert len(result.data["diagnoses"]) > 0

    def test_diagnose_from_densities(self, troubleshooting_skill, poor_densities):
        """Test diagnosis from density data."""
        result = troubleshooting_skill.execute(
            "Why does my print have issues",
            symptoms=["problems"],
            densities=poor_densities,
        )
        assert result.success is True
        assert result.data["density_analysis"] is not None
        # Poor densities have high dmin and low dmax
        assert any("Dmax" in d["symptom"] or "Dmin" in d["symptom"] for d in result.data["diagnoses"])

    def test_diagnose_with_context(self, troubleshooting_skill):
        """Test diagnosis with context."""
        context = SkillContext(humidity=35.0, temperature=16.0)
        result = troubleshooting_skill.execute(
            "Why isn't my print working",
            context=context,
        )
        assert result.success is True
        # Should include environment-related diagnoses
        assert any("humidity" in d["symptom"].lower() for d in result.data["diagnoses"])

    def test_priority_actions(self, troubleshooting_skill):
        """Test that priority actions are generated."""
        result = troubleshooting_skill.execute(
            "Print has multiple issues",
            symptoms=["muddy", "uneven", "staining"],
        )
        assert result.success is True
        assert len(result.data["priority_actions"]) > 0


# =============================================================================
# Unit Tests - SkillRegistry
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_create_default_skills(self, skill_registry):
        """Test creating default skills."""
        skills = skill_registry.list_skills()
        assert len(skills) == 4
        skill_names = [s.name for s in skills]
        assert "calibration" in skill_names
        assert "chemistry" in skill_names
        assert "quality" in skill_names
        assert "troubleshooting" in skill_names

    def test_get_skill(self, skill_registry):
        """Test getting a skill by name."""
        skill = skill_registry.get("calibration")
        assert skill is not None
        assert skill.name == "calibration"

    def test_get_nonexistent_skill(self, skill_registry):
        """Test getting a nonexistent skill."""
        skill = skill_registry.get("nonexistent")
        assert skill is None

    def test_find_best_skill(self, skill_registry):
        """Test finding the best skill for a task."""
        result = skill_registry.find_best_skill("Calibrate my step tablet")
        assert result is not None
        skill, confidence = result
        assert skill.name == "calibration"
        assert confidence >= 0.8

    def test_find_capable_skills(self, skill_registry):
        """Test finding all capable skills."""
        capable = skill_registry.find_capable_skills("Check quality of calibration")
        assert len(capable) > 0
        # Should find quality and possibly calibration
        skill_names = [s.name for s, _ in capable]
        assert "quality" in skill_names

    def test_execute_best(self, skill_registry, sample_densities):
        """Test executing the best skill for a task."""
        result = skill_registry.execute_best(
            "Analyze calibration quality",
            densities=sample_densities,
        )
        assert result is not None
        assert result.success is True

    def test_list_by_category(self, skill_registry):
        """Test listing skills by category."""
        calibration_skills = skill_registry.list_by_category(SkillCategory.CALIBRATION)
        assert len(calibration_skills) == 1
        assert calibration_skills[0].name == "calibration"

    def test_register_unregister(self):
        """Test registering and unregistering skills."""
        registry = SkillRegistry()
        skill = CalibrationSkill()

        registry.register(skill)
        assert registry.get("calibration") is not None

        success = registry.unregister("calibration")
        assert success is True
        assert registry.get("calibration") is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkillsIntegration:
    """Integration tests for skills working together."""

    def test_full_workflow(self, skill_registry, sample_densities):
        """Test a full workflow using multiple skills."""
        # Step 1: Quality pre-check
        quality_result = skill_registry.execute_best(
            "Check pre-print conditions",
            humidity=50.0,
            temperature=21.0,
        )
        assert quality_result.success is True

        # Step 2: Calculate chemistry
        chemistry_result = skill_registry.execute_best(
            "Calculate coating for 8x10 print",
            print_width=8.0,
            print_height=10.0,
        )
        assert chemistry_result.success is True

        # Step 3: Calibration analysis
        calibration_result = skill_registry.execute_best(
            "Analyze calibration data",
            densities=sample_densities,
        )
        assert calibration_result.success is True

    def test_troubleshooting_workflow(self, skill_registry, poor_densities):
        """Test troubleshooting workflow with poor data."""
        # Step 1: Quality analysis identifies problems (use quality-specific task)
        quality_result = skill_registry.execute_best(
            "Check quality assessment post-print",
            densities=poor_densities,
        )
        assert quality_result.success is True
        # Quality skill returns quality_grade, calibration returns quality_score
        grade = quality_result.data.get("quality_grade") or quality_result.data.get("quality", {}).get("grade")
        assert grade is not None

        # Step 2: Troubleshoot issues (use troubleshooting-specific task)
        trouble_result = skill_registry.execute_best(
            "Why is my print having problems",
            symptoms=["poor density range"],
            densities=poor_densities,
        )
        assert trouble_result.success is True
        # Either troubleshooting skill (diagnoses) or quality skill (recommendations)
        assert "diagnoses" in trouble_result.data or "recommendations" in trouble_result.data


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "task,expected_skill",
    [
        ("Calibrate step tablet", "calibration"),
        ("Calculate coating drops", "chemistry"),
        ("Check print quality", "quality"),
        ("Why is my print muddy", "troubleshooting"),
        ("Linearize curve", "calibration"),
        ("Metal ratio for warm tones", "chemistry"),
        ("Pre-print validation", "quality"),
        ("Fix blocked shadows", "troubleshooting"),
    ],
)
def test_skill_routing(task: str, expected_skill: str):
    """Parametrized test for skill routing."""
    registry = create_default_skills()
    result = registry.find_best_skill(task)
    assert result is not None
    skill, _ = result
    assert skill.name == expected_skill


@pytest.mark.parametrize(
    "tone,expected_ratio",
    [
        ("warm", 0.2),
        ("neutral", 0.5),
        ("cool", 0.8),
    ],
)
def test_chemistry_tone_ratios(tone: str, expected_ratio: float):
    """Parametrized test for tone ratio recommendations."""
    skill = ChemistrySkill()
    result = skill.execute(f"Ratio for {tone} tones", target_tone=tone)
    assert result.data["recommended_ratio"] == expected_ratio


@pytest.mark.parametrize(
    "symptom",
    ["muddy", "faded", "blocked", "flat", "uneven", "staining", "bronzing", "fog"],
)
def test_troubleshooting_symptoms(symptom: str):
    """Parametrized test for symptom diagnosis."""
    skill = TroubleshootingSkill()
    result = skill.execute(f"Print is {symptom}", symptoms=[symptom])
    assert result.success is True
    assert len(result.data["diagnoses"]) > 0
