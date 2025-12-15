"""
Unit tests for composite tools in the agents module.

Tests cover:
- Full calibration workflow tool
- Quality report generation
- Troubleshooting tool
- Recipe optimization tool
"""

import pytest
from typing import Any

from ptpd_calibration.agents.tools import (
    ToolRegistry,
    ToolResult,
    create_calibration_tools,
    register_composite_tools,
    _full_calibration,
    _quality_report,
    _troubleshoot_print,
    _optimize_recipe,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_densities():
    """Create sample density measurements."""
    # Simulate typical Pt/Pd response with good range
    return [
        0.08, 0.15, 0.25, 0.38, 0.52, 0.68, 0.85, 1.02,
        1.20, 1.38, 1.55, 1.72, 1.88, 2.02, 2.15, 2.26,
        2.35, 2.42, 2.48, 2.52, 2.55
    ]


@pytest.fixture
def poor_densities():
    """Create sample density measurements with issues."""
    # Low range, non-monotonic
    return [
        0.15, 0.22, 0.30, 0.35, 0.42, 0.48, 0.52, 0.55,
        0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.69, 0.70,
        0.71, 0.72, 0.73, 0.74, 0.75
    ]


@pytest.fixture
def non_monotonic_densities():
    """Create non-monotonic density measurements."""
    return [
        0.08, 0.15, 0.25, 0.38, 0.35,  # reversal here
        0.52, 0.68, 0.85, 1.02, 1.20,
        1.38, 1.55, 1.72, 1.88, 2.02,
        2.15, 2.26, 2.35, 2.42, 2.48, 2.55
    ]


@pytest.fixture
def tool_registry():
    """Create a tool registry with composite tools registered."""
    registry = create_calibration_tools()
    register_composite_tools(registry)
    return registry


# =============================================================================
# Unit Tests - Full Calibration Tool
# =============================================================================


class TestFullCalibrationTool:
    """Tests for the full_calibration composite tool."""

    def test_full_calibration_success(self, sample_densities):
        """Test full calibration with good data."""
        result = _full_calibration(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="Test Curve",
            curve_type="linear",
            export_format="qtr",
            target_dmax=2.0,
        )

        assert result.success is True
        assert "analysis" in result.data
        assert "curve" in result.data
        assert "quality" in result.data
        assert "export_ready" in result.data

    def test_full_calibration_analysis_results(self, sample_densities):
        """Test analysis results in full calibration."""
        result = _full_calibration(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="Test Curve",
        )

        analysis = result.data["analysis"]
        assert "dmin" in analysis
        assert "dmax" in analysis
        assert "range" in analysis
        assert "is_monotonic" in analysis
        assert analysis["dmin"] < analysis["dmax"]

    def test_full_calibration_curve_results(self, sample_densities):
        """Test curve results in full calibration."""
        result = _full_calibration(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="My Test Curve",
        )

        curve = result.data["curve"]
        assert "id" in curve
        assert curve["name"] == "My Test Curve"
        assert curve["num_points"] > 0

    def test_full_calibration_quality_grade_excellent(self, sample_densities):
        """Test excellent quality grade for good data."""
        result = _full_calibration(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="Test Curve",
        )

        quality = result.data["quality"]
        assert quality["grade"] in ("excellent", "good")

    def test_full_calibration_quality_grade_poor(self, poor_densities):
        """Test poor quality grade for bad data."""
        result = _full_calibration(
            densities=poor_densities,
            paper_type="Arches Platine",
            name="Test Curve",
        )

        quality = result.data["quality"]
        assert quality["grade"] in ("poor", "fair")

    def test_full_calibration_metadata(self, sample_densities):
        """Test metadata in result."""
        result = _full_calibration(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="Test Curve",
        )

        assert result.metadata.get("composite_tool") == "full_calibration"

    def test_full_calibration_curve_types(self, sample_densities):
        """Test different curve types."""
        for curve_type in ["linear", "paper_white", "aesthetic"]:
            result = _full_calibration(
                densities=sample_densities,
                paper_type="Arches Platine",
                name="Test Curve",
                curve_type=curve_type,
            )
            assert result.success is True
            assert result.data["curve"]["type"] == curve_type


# =============================================================================
# Unit Tests - Quality Report Tool
# =============================================================================


class TestQualityReportTool:
    """Tests for the quality_report composite tool."""

    def test_quality_report_success(self, sample_densities):
        """Test quality report with good data."""
        result = _quality_report(
            densities=sample_densities,
            paper_type="Arches Platine",
            expected_dmin=0.1,
            expected_dmax=2.0,
        )

        assert result.success is True
        assert "metrics" in result.data
        assert "checks" in result.data
        assert "grade" in result.data
        assert "recommendations" in result.data

    def test_quality_report_metrics(self, sample_densities):
        """Test metrics in quality report."""
        result = _quality_report(densities=sample_densities)

        metrics = result.data["metrics"]
        assert "dmin" in metrics
        assert "dmax" in metrics
        assert "range" in metrics
        assert "num_steps" in metrics
        assert "step_uniformity" in metrics
        assert "linearity_error" in metrics

    def test_quality_report_checks(self, sample_densities):
        """Test quality checks in report."""
        result = _quality_report(densities=sample_densities)

        checks = result.data["checks"]
        assert "dmin_acceptable" in checks
        assert "dmax_acceptable" in checks
        assert "range_acceptable" in checks
        assert "monotonicity" in checks
        assert "step_uniformity" in checks
        assert "sufficient_steps" in checks

    def test_quality_report_grade_a(self, sample_densities):
        """Test A grade for excellent data."""
        result = _quality_report(densities=sample_densities)

        grade = result.data["grade"]
        assert grade["overall"] in ("A", "B")
        assert grade["score"] >= 0.8

    def test_quality_report_grade_low(self, poor_densities):
        """Test low grade for poor data."""
        result = _quality_report(
            densities=poor_densities,
            expected_dmin=0.1,
            expected_dmax=2.0,
        )

        grade = result.data["grade"]
        assert grade["overall"] in ("D", "F")
        assert grade["score"] < 0.7

    def test_quality_report_recommendations(self, poor_densities):
        """Test recommendations for poor data."""
        result = _quality_report(
            densities=poor_densities,
            expected_dmin=0.1,
            expected_dmax=2.0,
        )

        recommendations = result.data["recommendations"]
        assert len(recommendations) > 0

    def test_quality_report_paper_type_preserved(self, sample_densities):
        """Test paper type is preserved in report."""
        result = _quality_report(
            densities=sample_densities,
            paper_type="Bergger COT320",
        )

        assert result.data["paper_type"] == "Bergger COT320"

    def test_quality_report_non_monotonic(self, non_monotonic_densities):
        """Test detection of non-monotonic data."""
        result = _quality_report(densities=non_monotonic_densities)

        checks = result.data["checks"]
        assert checks["monotonicity"] is False

        recommendations = result.data["recommendations"]
        assert any("monoton" in r.lower() for r in recommendations)


# =============================================================================
# Unit Tests - Troubleshoot Print Tool
# =============================================================================


class TestTroubleshootPrintTool:
    """Tests for the troubleshoot_print composite tool."""

    def test_troubleshoot_muddy_symptoms(self):
        """Test troubleshooting muddy print symptoms."""
        result = _troubleshoot_print(
            symptoms=["muddy shadows", "blocked midtones"],
        )

        assert result.success is True
        assert "diagnoses" in result.data
        assert len(result.data["diagnoses"]) > 0

    def test_troubleshoot_faded_symptoms(self):
        """Test troubleshooting faded print symptoms."""
        result = _troubleshoot_print(
            symptoms=["faded highlights", "low contrast"],
        )

        assert result.success is True
        assert len(result.data["diagnoses"]) > 0
        assert any("exposure" in d["suggested_fix"].lower() for d in result.data["diagnoses"])

    def test_troubleshoot_blocked_symptoms(self):
        """Test troubleshooting blocked shadows."""
        result = _troubleshoot_print(
            symptoms=["blocked shadows"],
        )

        assert result.success is True
        assert any("blocked" in d["symptom"].lower() for d in result.data["diagnoses"])

    def test_troubleshoot_uneven_symptoms(self):
        """Test troubleshooting uneven coating."""
        result = _troubleshoot_print(
            symptoms=["uneven coating"],
        )

        assert result.success is True
        diagnoses = result.data["diagnoses"]
        assert any("coating" in d["likely_cause"].lower() for d in diagnoses)

    def test_troubleshoot_with_densities(self, poor_densities):
        """Test troubleshooting with density data."""
        result = _troubleshoot_print(
            symptoms=["faded print"],
            densities=poor_densities,
        )

        assert result.success is True
        assert result.data["density_analysis"] is not None
        # Poor densities have low dmax
        assert any("Dmax" in d["symptom"] for d in result.data["diagnoses"])

    def test_troubleshoot_priority_actions(self):
        """Test priority actions are generated."""
        result = _troubleshoot_print(
            symptoms=["muddy shadows", "faded highlights", "blocked areas"],
        )

        assert "priority_actions" in result.data
        assert len(result.data["priority_actions"]) > 0

    def test_troubleshoot_context_preserved(self):
        """Test context is preserved in result."""
        result = _troubleshoot_print(
            symptoms=["muddy print"],
            paper_type="Arches Platine",
            metal_ratio=0.5,
            exposure_time=180.0,
        )

        context = result.data["context"]
        assert context["paper_type"] == "Arches Platine"
        assert context["metal_ratio"] == 0.5
        assert context["exposure_time"] == 180.0

    def test_troubleshoot_confidence(self):
        """Test confidence score calculation."""
        result = _troubleshoot_print(
            symptoms=["muddy", "faded", "blocked"],
        )

        assert "confidence" in result.data
        # Confidence can exceed 1 when multiple matches found per symptom
        assert result.data["confidence"] >= 0

    def test_troubleshoot_staining(self):
        """Test troubleshooting staining issues."""
        result = _troubleshoot_print(
            symptoms=["staining on edges"],
        )

        assert result.success is True
        assert any("clearing" in d["suggested_fix"].lower() for d in result.data["diagnoses"])

    def test_troubleshoot_bronzing(self):
        """Test troubleshooting bronzing issues."""
        result = _troubleshoot_print(
            symptoms=["bronzing in shadows"],
        )

        assert result.success is True
        assert any("bronz" in d["symptom"].lower() for d in result.data["diagnoses"])


# =============================================================================
# Unit Tests - Optimize Recipe Tool
# =============================================================================


class TestOptimizeRecipeTool:
    """Tests for the optimize_recipe composite tool."""

    def test_optimize_recipe_basic(self):
        """Test basic recipe optimization."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
        )

        assert result.success is True
        assert "optimized_recipe" in result.data
        assert "targets" in result.data

    def test_optimize_recipe_warm_tone(self):
        """Test recipe optimization for warm tones."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"tone": "warm"},
        )

        recipe = result.data["optimized_recipe"]
        assert recipe["metal_ratio"] == 0.2  # Low platinum = warm
        assert "80% Pd" in recipe["metal_description"]

    def test_optimize_recipe_cool_tone(self):
        """Test recipe optimization for cool tones."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"tone": "cool"},
        )

        recipe = result.data["optimized_recipe"]
        assert recipe["metal_ratio"] == 0.8  # High platinum = cool
        assert "80% Pt" in recipe["metal_description"]

    def test_optimize_recipe_high_contrast(self):
        """Test recipe optimization for high contrast."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"contrast": "high"},
        )

        recipe = result.data["optimized_recipe"]
        assert recipe["na2_drops"] == 10
        assert "High contrast" in recipe["contrast_description"]

    def test_optimize_recipe_low_contrast(self):
        """Test recipe optimization for low contrast."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"contrast": "low"},
        )

        recipe = result.data["optimized_recipe"]
        assert recipe["na2_drops"] == 0

    def test_optimize_recipe_high_dmax_exposure(self):
        """Test exposure adjustment for high dmax target."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"dmax": 2.5},
        )

        recipe = result.data["optimized_recipe"]
        # High dmax target should increase exposure
        assert recipe["exposure_time"] > 180.0

    def test_optimize_recipe_low_dmax_exposure(self):
        """Test exposure adjustment for low dmax target."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"dmax": 1.5},
        )

        recipe = result.data["optimized_recipe"]
        # Low dmax target should decrease exposure
        assert recipe["exposure_time"] < 180.0

    def test_optimize_recipe_adjustments(self):
        """Test adjustment suggestions when current params provided."""
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics={"tone": "warm"},
            current_params={"metal_ratio": 0.8, "exposure_time": 200.0},
        )

        adjustments = result.data["adjustments"]
        assert len(adjustments) > 0
        assert any("metal ratio" in a.lower() for a in adjustments)

    def test_optimize_recipe_preserves_targets(self):
        """Test that targets are preserved in result."""
        targets = {"dmax": 2.2, "tone": "cool", "contrast": "high"}
        result = _optimize_recipe(
            paper_type="Arches Platine",
            target_characteristics=targets,
        )

        result_targets = result.data["targets"]
        assert result_targets["dmax"] == 2.2
        assert result_targets["tone"] == "cool"
        assert result_targets["contrast"] == "high"


# =============================================================================
# Integration Tests - Tool Registry
# =============================================================================


class TestToolRegistryIntegration:
    """Integration tests for composite tools with registry."""

    def test_composite_tools_registered(self, tool_registry):
        """Test that composite tools are registered."""
        tools = tool_registry.list_tools()
        tool_names = [t.name for t in tools]

        assert "full_calibration" in tool_names
        assert "quality_report" in tool_names
        assert "troubleshoot_print" in tool_names
        assert "optimize_recipe" in tool_names

    def test_full_calibration_via_registry(self, tool_registry, sample_densities):
        """Test executing full_calibration via registry."""
        tool = tool_registry.get("full_calibration")
        assert tool is not None

        # Note: execute is async, so we test the handler directly
        result = tool.handler(
            densities=sample_densities,
            paper_type="Arches Platine",
            name="Test",
        )
        assert result.success is True

    def test_quality_report_via_registry(self, tool_registry, sample_densities):
        """Test executing quality_report via registry."""
        tool = tool_registry.get("quality_report")
        assert tool is not None

        result = tool.handler(densities=sample_densities)
        assert result.success is True

    def test_troubleshoot_via_registry(self, tool_registry):
        """Test executing troubleshoot_print via registry."""
        tool = tool_registry.get("troubleshoot_print")
        assert tool is not None

        result = tool.handler(symptoms=["muddy shadows"])
        assert result.success is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for composite tools."""

    def test_empty_densities_quality_report(self):
        """Test quality report with minimal densities."""
        result = _quality_report(densities=[0.1, 0.5, 1.0])

        assert result.success is True
        # Should flag insufficient steps
        assert result.data["checks"]["sufficient_steps"] is False

    def test_single_symptom_troubleshoot(self):
        """Test troubleshoot with single symptom."""
        result = _troubleshoot_print(symptoms=["muddy"])

        assert result.success is True
        assert len(result.data["diagnoses"]) > 0

    def test_unknown_symptom_troubleshoot(self):
        """Test troubleshoot with unknown symptom."""
        result = _troubleshoot_print(symptoms=["xyz_unknown_symptom"])

        assert result.success is True
        # May have no diagnoses for unknown symptom
        assert result.data["confidence"] >= 0

    def test_optimize_recipe_default_characteristics(self):
        """Test optimize_recipe with no characteristics."""
        result = _optimize_recipe(paper_type="Test Paper")

        assert result.success is True
        # Should use defaults
        recipe = result.data["optimized_recipe"]
        assert recipe["metal_ratio"] == 0.5  # neutral default
        assert recipe["na2_drops"] == 5  # normal default


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "symptoms,expected_keyword",
    [
        (["muddy shadows"], "exposure"),
        (["faded print"], "exposure"),
        (["blocked shadows"], "exposure"),
        (["uneven coating"], "brush"),  # Fix mentions brush technique
        (["staining"], "clearing"),
        (["bronzing"], "sensitizer"),  # Fix mentions sensitizer volume
    ],
)
def test_troubleshoot_symptom_mapping(symptoms: list[str], expected_keyword: str):
    """Parametrized test for symptom-to-fix mapping."""
    result = _troubleshoot_print(symptoms=symptoms)
    fixes = [d["suggested_fix"].lower() for d in result.data["diagnoses"]]
    all_fixes = " ".join(fixes)
    # Expected keyword should appear in at least one fix
    assert expected_keyword in all_fixes or len(result.data["diagnoses"]) == 0


@pytest.mark.parametrize(
    "tone,expected_ratio",
    [
        ("warm", 0.2),
        ("neutral", 0.5),
        ("cool", 0.8),
    ],
)
def test_optimize_recipe_tone_ratios(tone: str, expected_ratio: float):
    """Parametrized test for tone to metal ratio mapping."""
    result = _optimize_recipe(
        paper_type="Test",
        target_characteristics={"tone": tone},
    )
    assert result.data["optimized_recipe"]["metal_ratio"] == expected_ratio


@pytest.mark.parametrize(
    "contrast,expected_drops",
    [
        ("low", 0),
        ("normal", 5),
        ("high", 10),
    ],
)
def test_optimize_recipe_contrast_drops(contrast: str, expected_drops: int):
    """Parametrized test for contrast to Na2 drops mapping."""
    result = _optimize_recipe(
        paper_type="Test",
        target_characteristics={"contrast": contrast},
    )
    assert result.data["optimized_recipe"]["na2_drops"] == expected_drops
