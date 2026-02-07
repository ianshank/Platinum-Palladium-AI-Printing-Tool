"""
Integration tests for the complete curve display and step wedge analysis workflow.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.analysis import (
    QualityGrade,
    StepWedgeAnalyzer,
    WedgeAnalysisConfig,
)
from ptpd_calibration.config import TabletType, get_settings
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves import (
    ColorScheme,
    CurveModifier,
    CurveVisualizer,
    PlotStyle,
    VisualizationConfig,
    load_curve,
    save_curve,
)


class TestEndToEndStepWedgeWorkflow:
    """
    End-to-end tests for step wedge analysis workflow.

    Tests the complete flow from:
    1. Step wedge scan → density extraction → curve generation → export
    """

    @pytest.fixture
    def synthetic_step_tablet(self, tmp_path):
        """Create a realistic synthetic step tablet image."""
        width, height = 630, 100  # 30px per patch for 21 patches
        num_patches = 21
        patch_width = width // num_patches

        # Create grayscale gradient with typical Pt/Pd response
        img = np.zeros((height, width), dtype=np.uint8)

        for i in range(num_patches):
            # Simulate typical Pt/Pd toe curve
            normalized = i / (num_patches - 1)
            # Non-linear response (toe + shoulder)
            response = normalized ** 0.85
            value = int(255 * (1 - response))

            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img[:, x_start:x_end] = value

            # Add some noise for realism
            noise = np.random.normal(0, 2, (height, x_end - x_start))
            img[:, x_start:x_end] = np.clip(img[:, x_start:x_end] + noise, 0, 255).astype(np.uint8)

        # Add white margin (paper)
        full_img = np.full((height + 60, width + 60, 3), 248, dtype=np.uint8)
        full_img[30:height + 30, 30:width + 30, 0] = img
        full_img[30:height + 30, 30:width + 30, 1] = img
        full_img[30:height + 30, 30:width + 30, 2] = img

        image_path = tmp_path / "step_tablet_synthetic.png"
        Image.fromarray(full_img).save(image_path)

        return image_path

    def test_complete_workflow_from_image(self, synthetic_step_tablet, tmp_path):
        """Test complete workflow: image → analysis → curve → export."""
        # Step 1: Configure analyzer
        config = WedgeAnalysisConfig(
            tablet_type=TabletType.STOUFFER_21,
            auto_fix_reversals=True,
            outlier_rejection=True,
        )
        analyzer = StepWedgeAnalyzer(config)

        # Step 2: Analyze image
        result = analyzer.analyze(
            synthetic_step_tablet,
            curve_name="Integration Test Curve",
            paper_type="Test Paper",
            chemistry="Test Chemistry",
            generate_curve=True,
            curve_type=CurveType.LINEAR,
        )

        # Verify analysis succeeded
        assert result.detection_success is True
        assert len(result.densities) == 21
        assert result.quality is not None
        assert result.quality.grade != QualityGrade.FAILED

        # Step 3: Verify curve was generated
        assert result.curve_generated is True
        assert result.curve is not None
        assert result.curve.name == "Integration Test Curve"
        assert result.curve.paper_type == "Test Paper"

        # Step 4: Export curve
        export_path = tmp_path / "test_curve.json"
        save_curve(result.curve, export_path, format="json")

        assert export_path.exists()

        # Step 5: Reload and verify
        loaded_curve = load_curve(export_path)
        assert loaded_curve.name == result.curve.name
        assert len(loaded_curve.input_values) == len(result.curve.input_values)

    def test_workflow_with_modification(self, synthetic_step_tablet, tmp_path):
        """Test workflow with curve modification step."""
        # Analyze
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze(
            synthetic_step_tablet,
            curve_name="Base Curve",
            generate_curve=True,
        )

        assert result.curve is not None
        original_curve = result.curve

        # Modify curve
        modifier = CurveModifier()

        # Apply brightness adjustment
        brightened = modifier.adjust_brightness(original_curve, 0.1)
        assert brightened is not original_curve

        # Apply contrast adjustment
        contrasted = modifier.adjust_contrast(brightened, 0.15)

        # Apply gamma
        gamma_adjusted = modifier.adjust_gamma(contrasted, 0.9)

        # Apply smoothing
        smoothed = modifier.smooth(gamma_adjusted, strength=0.3)

        # Verify modifications changed the curve
        original_mid = original_curve.output_values[len(original_curve.output_values) // 2]
        modified_mid = smoothed.output_values[len(smoothed.output_values) // 2]
        assert original_mid != modified_mid

        # Export modified curve
        export_path = tmp_path / "modified_curve.csv"
        save_curve(smoothed, export_path, format="csv")
        assert export_path.exists()

    def test_workflow_with_visualization(self, synthetic_step_tablet, tmp_path):
        """Test workflow with visualization output."""
        # Analyze
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze(
            synthetic_step_tablet,
            curve_name="Viz Test Curve",
            generate_curve=True,
        )

        assert result.curve is not None

        # Create visualizer
        vis_config = VisualizationConfig(
            figure_width=12.0,
            figure_height=8.0,
            dpi=150,
            color_scheme=ColorScheme.PLATINUM,
            show_statistics=True,
            show_reference_line=True,
        )
        visualizer = CurveVisualizer(vis_config)

        # Generate various plots
        single_plot = visualizer.plot_single_curve(result.curve, show_stats=True)
        assert single_plot is not None

        slope_plot = visualizer.plot_slope_analysis(result.curve)
        assert slope_plot is not None

        histogram = visualizer.plot_histogram(result.curve)
        assert histogram is not None

        # Save plot
        plot_path = tmp_path / "curve_visualization.png"
        visualizer.save_figure(single_plot, plot_path)
        assert plot_path.exists()

        # Get statistics
        stats = visualizer.compute_statistics(result.curve)
        assert stats is not None
        assert stats.is_monotonic is True


class TestMultiCurveComparison:
    """Tests for comparing multiple curves."""

    @pytest.fixture
    def curve_set(self):
        """Create a set of curves for comparison."""
        curves = []

        # Linear curve
        inputs = list(np.linspace(0, 1, 256))
        curves.append(CurveData(
            name="Linear Reference",
            input_values=inputs,
            output_values=inputs.copy(),
        ))

        # Gamma 0.9 curve
        curves.append(CurveData(
            name="Gamma 0.9",
            input_values=inputs,
            output_values=list(np.array(inputs) ** 0.9),
        ))

        # Gamma 1.2 curve
        curves.append(CurveData(
            name="Gamma 1.2",
            input_values=inputs,
            output_values=list(np.array(inputs) ** 1.2),
        ))

        # S-curve
        t = np.array(inputs)
        s_curve = t * t * (3 - 2 * t)
        curves.append(CurveData(
            name="S-Curve",
            input_values=inputs,
            output_values=list(s_curve),
        ))

        return curves

    def test_multi_curve_comparison_plot(self, curve_set, tmp_path):
        """Test plotting multiple curves for comparison."""
        visualizer = CurveVisualizer()

        # Plot all curves together
        fig = visualizer.plot_multiple_curves(
            curve_set,
            title="Curve Family Comparison",
            style=PlotStyle.LINE,
            show_difference=True,
        )

        assert fig is not None

        # Save comparison plot
        plot_path = tmp_path / "comparison.png"
        visualizer.save_figure(fig, plot_path)
        assert plot_path.exists()

    def test_multi_curve_statistics(self, curve_set):
        """Test statistics for multiple curves."""
        visualizer = CurveVisualizer()

        # Compute statistics for all curves
        all_stats = [visualizer.compute_statistics(c) for c in curve_set]

        # Linear should have gamma ~1.0
        linear_stats = all_stats[0]
        assert linear_stats.gamma == pytest.approx(1.0, abs=0.1)

        # Gamma 0.9 should lift midtones
        gamma_09_stats = all_stats[1]
        assert gamma_09_stats.midpoint_value > 0.5

        # Gamma 1.2 should darken midtones
        gamma_12_stats = all_stats[2]
        assert gamma_12_stats.midpoint_value < 0.5

    def test_curve_comparison_metrics(self, curve_set):
        """Test curve comparison metrics."""
        visualizer = CurveVisualizer()

        # Compare all to linear reference
        comparison = visualizer.compare_curves(curve_set, reference_idx=0)

        assert comparison.curve_names == [c.name for c in curve_set]
        assert comparison.max_difference > 0  # Should have differences
        assert 0 < comparison.correlation < 1  # Some correlation


class TestCurveExportFormats:
    """Tests for exporting curves in various formats."""

    @pytest.fixture
    def sample_curve(self):
        """Create a sample curve."""
        inputs = list(np.linspace(0, 1, 256))
        return CurveData(
            name="Export Test Curve",
            input_values=inputs,
            output_values=list(np.array(inputs) ** 0.9),
            paper_type="Test Paper",
            chemistry="Test Chemistry",
            curve_type=CurveType.LINEAR,
        )

    def test_export_all_formats(self, sample_curve, tmp_path):
        """Test exporting to all supported formats."""
        formats = [
            ("json", ".json"),
            ("csv", ".csv"),
            ("qtr", ".txt"),
        ]

        for format_name, extension in formats:
            export_path = tmp_path / f"curve{extension}"
            save_curve(sample_curve, export_path, format=format_name)

            assert export_path.exists(), f"Export failed for {format_name}"
            assert export_path.stat().st_size > 0, f"Empty file for {format_name}"

    def test_json_roundtrip(self, sample_curve, tmp_path):
        """Test JSON export/import roundtrip."""
        export_path = tmp_path / "roundtrip.json"

        save_curve(sample_curve, export_path, format="json")
        loaded = load_curve(export_path)

        assert loaded.name == sample_curve.name
        assert len(loaded.input_values) == len(sample_curve.input_values)
        assert loaded.input_values[0] == pytest.approx(sample_curve.input_values[0])
        assert loaded.output_values[-1] == pytest.approx(sample_curve.output_values[-1])

    def test_csv_roundtrip(self, sample_curve, tmp_path):
        """Test CSV export/import roundtrip."""
        export_path = tmp_path / "roundtrip.csv"

        save_curve(sample_curve, export_path, format="csv")
        loaded = load_curve(export_path)

        assert len(loaded.input_values) == len(sample_curve.input_values)


class TestConfigurationIntegration:
    """Tests for configuration-driven behavior."""

    def test_settings_affect_analyzer(self):
        """Test that settings properly configure analyzer."""
        settings = get_settings()

        # Create config from settings
        config = WedgeAnalysisConfig(
            min_density_range=settings.wedge_analysis.min_density_range,
            max_dmin=settings.wedge_analysis.max_dmin,
            auto_fix_reversals=settings.wedge_analysis.auto_fix_reversals,
        )

        analyzer = StepWedgeAnalyzer(config)

        assert analyzer.config.min_density_range == settings.wedge_analysis.min_density_range

    def test_settings_affect_visualizer(self):
        """Test that settings properly configure visualizer."""
        settings = get_settings()

        vis_config = VisualizationConfig(
            figure_width=settings.visualization.figure_width,
            figure_height=settings.visualization.figure_height,
            dpi=settings.visualization.dpi,
            show_grid=settings.visualization.show_grid,
        )

        visualizer = CurveVisualizer(vis_config)

        assert visualizer.config.figure_width == settings.visualization.figure_width
        assert visualizer.config.dpi == settings.visualization.dpi


class TestQualityGradeScenarios:
    """Tests for different quality grade scenarios."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default config."""
        return StepWedgeAnalyzer()

    def test_excellent_quality(self, analyzer):
        """Test excellent quality detection."""
        # Ideal density range, low Dmin, good monotonicity
        densities = [0.08 + 2.2 * (i / 20) ** 0.9 for i in range(21)]
        result = analyzer.analyze_from_densities(densities)

        assert result.quality.grade in [QualityGrade.EXCELLENT, QualityGrade.GOOD]
        assert result.quality.score >= 70

    def test_poor_quality_low_range(self, analyzer):
        """Test poor quality due to low density range."""
        # Very low density range
        densities = [0.1 + 0.5 * (i / 20) for i in range(21)]
        result = analyzer.analyze_from_densities(densities)

        assert result.quality.grade in [QualityGrade.POOR, QualityGrade.ACCEPTABLE]
        warning_codes = [w.code for w in result.quality.warnings]
        assert "LOW_DENSITY_RANGE" in warning_codes

    def test_poor_quality_high_dmin(self, analyzer):
        """Test poor quality due to high Dmin (fog)."""
        # High Dmin
        densities = [0.4 + 1.6 * (i / 20) for i in range(21)]
        result = analyzer.analyze_from_densities(densities)

        warning_codes = [w.code for w in result.quality.warnings]
        assert "HIGH_DMIN" in warning_codes

    def test_recommendations_present(self, analyzer):
        """Test that recommendations are provided."""
        densities = [0.1 + 1.8 * (i / 20) ** 0.9 for i in range(21)]
        result = analyzer.analyze_from_densities(densities)

        assert len(result.quality.recommendations) > 0
