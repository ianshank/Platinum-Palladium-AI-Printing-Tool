"""
Tests for curve generation and export.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.generator import (
    CurveGenerator,
    TargetCurve,
    generate_linearization_curve,
)
from ptpd_calibration.curves.export import (
    QTRExporter,
    PiezographyExporter,
    CSVExporter,
    JSONExporter,
    save_curve,
    load_curve,
)
from ptpd_calibration.curves.analysis import CurveAnalyzer


class TestTargetCurve:
    """Tests for TargetCurve."""

    def test_linear_target(self):
        """Test linear target curve generation."""
        target = TargetCurve.linear(21)

        assert len(target.input_values) == 21
        assert len(target.output_values) == 21
        assert target.input_values[0] == pytest.approx(0.0)
        assert target.input_values[-1] == pytest.approx(1.0)
        assert target.output_values[0] == pytest.approx(0.0)
        assert target.output_values[-1] == pytest.approx(1.0)

    def test_paper_white_preserve_target(self):
        """Test paper white preserving target curve."""
        target = TargetCurve.paper_white_preserve(21, highlight_hold=0.1)

        assert len(target.input_values) == 21
        # First few values should be held at 0
        assert target.output_values[0] == pytest.approx(0.0)
        assert target.output_values[1] == pytest.approx(0.0)

    def test_aesthetic_target(self):
        """Test aesthetic S-curve target."""
        target = TargetCurve.aesthetic(21, shadow_boost=0.1)

        assert len(target.input_values) == 21
        # Should have S-curve characteristics
        mid_idx = 10
        # Midpoint should be slightly different from linear
        assert target.output_values[mid_idx] != pytest.approx(0.5, abs=0.1)


class TestCurveGenerator:
    """Tests for CurveGenerator."""

    @pytest.fixture
    def sample_densities(self):
        """Generate sample density measurements."""
        # Simulate typical Pt/Pd response (toe, linear, shoulder)
        steps = np.linspace(0, 1, 21)
        densities = 0.1 + 2.0 * (steps ** 0.8)  # Non-linear response
        return list(densities)

    def test_generate_linear_curve(self, sample_densities):
        """Test linear curve generation."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            curve_type=CurveType.LINEAR,
            name="Test Linear",
        )

        assert curve.name == "Test Linear"
        assert curve.curve_type == CurveType.LINEAR
        assert len(curve.input_values) > 0
        assert len(curve.output_values) > 0
        assert curve.input_values[0] == pytest.approx(0.0)
        assert curve.input_values[-1] == pytest.approx(1.0)

    def test_generate_paper_white_curve(self, sample_densities):
        """Test paper white preserving curve."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            curve_type=CurveType.PAPER_WHITE,
            name="Test Paper White",
        )

        assert curve.curve_type == CurveType.PAPER_WHITE
        # Output should be 0 at input 0
        assert curve.output_values[0] == pytest.approx(0.0, abs=0.01)

    def test_curve_monotonicity(self, sample_densities):
        """Test that generated curves are monotonic."""
        generator = CurveGenerator()
        curve = generator.generate(sample_densities)

        output = np.array(curve.output_values)
        diffs = np.diff(output)

        # All differences should be >= 0 (monotonically increasing)
        assert np.all(diffs >= -0.001)

    def test_generate_with_metadata(self, sample_densities):
        """Test curve generation with metadata."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            name="Calibration",
            paper_type="Arches Platine",
            chemistry="50% Pt, 5 drops Na2",
        )

        assert curve.paper_type == "Arches Platine"
        assert curve.chemistry == "50% Pt, 5 drops Na2"

    def test_insufficient_measurements(self):
        """Test error with too few measurements."""
        generator = CurveGenerator()

        with pytest.raises(ValueError):
            generator.generate([0.1])  # Only 1 measurement

    def test_flat_measurements(self):
        """Test error with flat measurements (no range)."""
        generator = CurveGenerator()

        with pytest.raises(ValueError):
            generator.generate([1.0, 1.0, 1.0, 1.0, 1.0])


class TestCurveExporters:
    """Tests for curve exporters."""

    @pytest.fixture
    def sample_curve(self):
        """Create a sample curve for testing."""
        return CurveData(
            name="Test Export Curve",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.linspace(0, 1, 256) ** 0.9),
            paper_type="Test Paper",
            curve_type=CurveType.LINEAR,
        )

    def test_qtr_export(self, sample_curve, tmp_path):
        """Test QTR curve export."""
        exporter = QTRExporter()
        output_path = tmp_path / "test_curve.txt"

        exporter.export(sample_curve, output_path, format="curve")

        assert output_path.exists()
        content = output_path.read_text()
        assert "CURVE=" in content
        assert "0=" in content

    def test_qtr_quad_export(self, sample_curve, tmp_path):
        """Test QTR .quad profile export."""
        exporter = QTRExporter()
        output_path = tmp_path / "test_profile.quad"

        exporter.export(sample_curve, output_path, format="quad")

        assert output_path.exists()
        content = output_path.read_text()
        assert "[General]" in content
        assert "ProfileName=" in content

    def test_piezography_export(self, sample_curve, tmp_path):
        """Test Piezography export."""
        exporter = PiezographyExporter()
        output_path = tmp_path / "test_curve.ppt"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "[Linearization]" in content

    def test_csv_export(self, sample_curve, tmp_path):
        """Test CSV export."""
        exporter = CSVExporter()
        output_path = tmp_path / "test_curve.csv"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "input,output" in content
        lines = content.strip().split("\n")
        assert len(lines) > 1  # Header + data

    def test_json_export(self, sample_curve, tmp_path):
        """Test JSON export."""
        exporter = JSONExporter()
        output_path = tmp_path / "test_curve.json"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["name"] == "Test Export Curve"
        assert "input_values" in data
        assert "output_values" in data

    def test_save_and_load_json(self, sample_curve, tmp_path):
        """Test round-trip JSON save/load."""
        output_path = tmp_path / "roundtrip.json"

        save_curve(sample_curve, output_path, format="json")
        loaded = load_curve(output_path)

        assert loaded.name == sample_curve.name
        assert len(loaded.input_values) == len(sample_curve.input_values)
        assert loaded.input_values[0] == pytest.approx(sample_curve.input_values[0])

    def test_save_and_load_csv(self, sample_curve, tmp_path):
        """Test round-trip CSV save/load."""
        output_path = tmp_path / "roundtrip.csv"

        save_curve(sample_curve, output_path, format="csv")
        loaded = load_curve(output_path)

        assert len(loaded.input_values) == len(sample_curve.input_values)


class TestCurveAnalyzer:
    """Tests for CurveAnalyzer."""

    def test_analyze_linearity_perfect(self):
        """Test linearity analysis with perfect linear response."""
        densities = list(np.linspace(0.1, 2.1, 21))

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert analysis.is_monotonic
        assert analysis.max_error < 0.1
        assert analysis.rms_error < 0.05
        assert len(analysis.problem_regions) == 0

    def test_analyze_linearity_nonlinear(self):
        """Test linearity analysis with non-linear response."""
        # Create non-linear response (typical film/paper curve)
        steps = np.linspace(0, 1, 21)
        densities = list(0.1 + 2.0 * (steps ** 1.5))

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert analysis.is_monotonic
        assert analysis.max_error > 0.05  # Should have deviation

    def test_analyze_non_monotonic(self):
        """Test detection of non-monotonic response."""
        # Create response with reversal (solarization)
        densities = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0, 1.9, 1.85, 1.8, 1.75]

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert not analysis.is_monotonic

    def test_suggest_adjustments_low_range(self):
        """Test suggestions for low density range."""
        # Low density range
        densities = list(np.linspace(0.1, 1.0, 21))

        suggestions = CurveAnalyzer.suggest_adjustments(densities)

        assert len(suggestions) > 0
        assert any("range" in s.lower() for s in suggestions)

    def test_suggest_adjustments_high_dmin(self):
        """Test suggestions for high Dmin."""
        # High Dmin (fog or paper issue)
        densities = list(np.linspace(0.25, 2.25, 21))

        suggestions = CurveAnalyzer.suggest_adjustments(densities)

        assert any("dmin" in s.lower() or "clear" in s.lower() for s in suggestions)

    def test_analyze_curve(self):
        """Test comprehensive curve analysis."""
        curve = CurveData(
            name="Test",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.linspace(0, 1, 256) ** 0.9),
        )

        analysis = CurveAnalyzer.analyze_curve(curve)

        assert "name" in analysis
        assert "num_points" in analysis
        assert "is_monotonic" in analysis
        assert analysis["is_monotonic"]
        assert analysis["shape"] in ["approximately linear", "convex (lifts midtones)", "concave (darkens midtones)"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_linearization_curve(self):
        """Test convenience function for linearization."""
        densities = list(np.linspace(0.1, 2.1, 21))

        curve = generate_linearization_curve(
            densities,
            name="Quick Cal",
            paper_type="Test Paper",
        )

        assert curve.name == "Quick Cal"
        assert curve.paper_type == "Test Paper"
        assert curve.curve_type == CurveType.LINEAR
