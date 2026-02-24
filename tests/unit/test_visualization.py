"""
Tests for curve visualization module.
"""

import numpy as np
import pytest

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.visualization import (
    ColorScheme,
    CurveComparisonResult,
    CurveStatistics,
    CurveVisualizer,
    PlotStyle,
    VisualizationConfig,
)


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()

        assert config.figure_width == 10.0
        assert config.figure_height == 6.0
        assert config.dpi == 100
        assert config.line_width == 2.0
        assert config.show_grid is True
        assert config.show_legend is True
        assert config.color_scheme == ColorScheme.PLATINUM

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            figure_width=12.0,
            figure_height=8.0,
            dpi=150,
            color_scheme=ColorScheme.ACCESSIBLE,
            show_grid=False,
        )

        assert config.figure_width == 12.0
        assert config.figure_height == 8.0
        assert config.dpi == 150
        assert config.color_scheme == ColorScheme.ACCESSIBLE
        assert config.show_grid is False

    def test_get_color_palette_platinum(self):
        """Test platinum color palette generation."""
        config = VisualizationConfig(color_scheme=ColorScheme.PLATINUM)
        colors = config.get_color_palette(5)

        assert len(colors) == 5
        assert all(isinstance(c, str) for c in colors)
        assert all(c.startswith("#") for c in colors)

    def test_get_color_palette_accessible(self):
        """Test accessible color palette generation."""
        config = VisualizationConfig(color_scheme=ColorScheme.ACCESSIBLE)
        colors = config.get_color_palette(8)

        assert len(colors) == 8
        # Accessible palette should have high contrast colors
        assert "#0072B2" in colors  # Blue

    def test_get_color_palette_custom(self):
        """Test custom color palette."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        config = VisualizationConfig(custom_colors=custom_colors)
        colors = config.get_color_palette(3)

        assert colors == custom_colors

    def test_get_color_palette_wrapping(self):
        """Test color palette wrapping for large requests."""
        config = VisualizationConfig(color_scheme=ColorScheme.MONOCHROME)
        colors = config.get_color_palette(20)

        assert len(colors) == 20


class TestCurveStatistics:
    """Tests for CurveStatistics."""

    def test_to_dict(self):
        """Test statistics dictionary conversion."""
        stats = CurveStatistics(
            name="Test Curve",
            num_points=256,
            input_min=0.0,
            input_max=1.0,
            output_min=0.0,
            output_max=1.0,
            gamma=1.0,
            midpoint_value=0.5,
            is_monotonic=True,
            max_slope=1.2,
            min_slope=0.8,
            average_slope=1.0,
            linearity_error=0.01,
        )

        d = stats.to_dict()

        assert d["name"] == "Test Curve"
        assert d["num_points"] == 256
        assert d["gamma"] == 1.0
        assert d["is_monotonic"] is True
        assert "input_range" in d
        assert "output_range" in d


class TestCurveComparisonResult:
    """Tests for CurveComparisonResult."""

    def test_to_dict(self):
        """Test comparison result dictionary conversion."""
        result = CurveComparisonResult(
            curve_names=["Curve A", "Curve B"],
            max_difference=0.05,
            average_difference=0.02,
            rms_difference=0.03,
            correlation=0.98,
        )

        d = result.to_dict()

        assert d["curves_compared"] == ["Curve A", "Curve B"]
        assert d["max_difference"] == 0.05
        assert d["correlation"] == 0.98


class TestCurveVisualizer:
    """Tests for CurveVisualizer."""

    @pytest.fixture
    def sample_curve(self):
        """Create a sample curve for testing."""
        inputs = list(np.linspace(0, 1, 256))
        outputs = list(np.linspace(0, 1, 256) ** 0.9)
        return CurveData(
            name="Test Curve",
            input_values=inputs,
            output_values=outputs,
            curve_type=CurveType.LINEAR,
        )

    @pytest.fixture
    def linear_curve(self):
        """Create a perfectly linear curve."""
        inputs = list(np.linspace(0, 1, 100))
        return CurveData(
            name="Linear Curve",
            input_values=inputs,
            output_values=inputs.copy(),
        )

    @pytest.fixture
    def gamma_curve(self):
        """Create a gamma curve."""
        inputs = list(np.linspace(0, 1, 100))
        outputs = list(np.array(inputs) ** 2.2)
        return CurveData(
            name="Gamma 2.2 Curve",
            input_values=inputs,
            output_values=outputs,
        )

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        return CurveVisualizer()

    def test_init_default_config(self):
        """Test visualizer initialization with default config."""
        viz = CurveVisualizer()
        assert viz.config is not None
        assert isinstance(viz.config, VisualizationConfig)

    def test_init_custom_config(self):
        """Test visualizer initialization with custom config."""
        config = VisualizationConfig(dpi=200, show_grid=False)
        viz = CurveVisualizer(config)

        assert viz.config.dpi == 200
        assert viz.config.show_grid is False

    def test_compute_statistics(self, visualizer, sample_curve):
        """Test statistics computation."""
        stats = visualizer.compute_statistics(sample_curve)

        assert stats.name == "Test Curve"
        assert stats.num_points == 256
        assert stats.input_min == pytest.approx(0.0)
        assert stats.input_max == pytest.approx(1.0)
        assert stats.output_min == pytest.approx(0.0, abs=0.01)
        assert stats.output_max == pytest.approx(1.0, abs=0.01)
        assert stats.is_monotonic is True

    def test_compute_statistics_linear(self, visualizer, linear_curve):
        """Test statistics for linear curve."""
        stats = visualizer.compute_statistics(linear_curve)

        assert stats.gamma == pytest.approx(1.0, abs=0.1)
        assert stats.midpoint_value == pytest.approx(0.5, abs=0.1)
        assert stats.linearity_error < 0.01
        assert stats.average_slope == pytest.approx(1.0, abs=0.1)

    def test_compute_statistics_gamma(self, visualizer, gamma_curve):
        """Test statistics for gamma curve."""
        stats = visualizer.compute_statistics(gamma_curve)

        # Gamma 2.2 curve should have gamma close to 2.2
        assert stats.gamma > 1.5
        assert stats.midpoint_value < 0.5  # Darker midtones
        assert stats.is_monotonic is True

    def test_compare_curves_same(self, visualizer, linear_curve):
        """Test comparison of identical curves."""
        result = visualizer.compare_curves([linear_curve, linear_curve])

        assert result.max_difference == pytest.approx(0.0)
        assert result.average_difference == pytest.approx(0.0)
        assert result.correlation == pytest.approx(1.0)

    def test_compare_curves_different(self, visualizer, linear_curve, gamma_curve):
        """Test comparison of different curves."""
        result = visualizer.compare_curves([linear_curve, gamma_curve])

        assert result.max_difference > 0
        assert result.average_difference > 0
        assert result.correlation < 1.0
        assert result.difference_curve is not None

    def test_compare_curves_insufficient(self, visualizer, linear_curve):
        """Test comparison with insufficient curves."""
        with pytest.raises(ValueError, match="At least 2 curves"):
            visualizer.compare_curves([linear_curve])

    def test_plot_single_curve(self, visualizer, sample_curve):
        """Test single curve plotting."""
        fig = visualizer.plot_single_curve(sample_curve)

        assert fig is not None
        # Check that figure was created
        assert hasattr(fig, "axes")

    def test_plot_single_curve_with_options(self, visualizer, sample_curve):
        """Test single curve plotting with options."""
        fig = visualizer.plot_single_curve(
            sample_curve,
            title="Custom Title",
            style=PlotStyle.LINE_MARKERS,
            color="#FF0000",
            show_stats=True,
        )

        assert fig is not None

    def test_plot_multiple_curves(self, visualizer, linear_curve, gamma_curve):
        """Test multiple curve plotting."""
        fig = visualizer.plot_multiple_curves(
            [linear_curve, gamma_curve],
            title="Comparison",
        )

        assert fig is not None

    def test_plot_multiple_curves_with_difference(self, visualizer, linear_curve, gamma_curve):
        """Test multiple curve plotting with difference."""
        fig = visualizer.plot_multiple_curves(
            [linear_curve, gamma_curve],
            show_difference=True,
        )

        assert fig is not None

    def test_plot_multiple_curves_empty(self, visualizer):
        """Test plotting with no curves."""
        with pytest.raises(ValueError, match="No curves provided"):
            visualizer.plot_multiple_curves([])

    def test_plot_with_statistics(self, visualizer, linear_curve, gamma_curve):
        """Test plotting with statistics panel."""
        fig = visualizer.plot_with_statistics(
            [linear_curve, gamma_curve],
            title="Analysis",
        )

        assert fig is not None

    def test_plot_histogram(self, visualizer, sample_curve):
        """Test histogram plotting."""
        fig = visualizer.plot_histogram(sample_curve, bins=50)

        assert fig is not None

    def test_plot_slope_analysis(self, visualizer, sample_curve):
        """Test slope analysis plotting."""
        fig = visualizer.plot_slope_analysis(sample_curve)

        assert fig is not None

    def test_figure_to_bytes(self, visualizer, sample_curve):
        """Test figure to bytes conversion."""
        fig = visualizer.plot_single_curve(sample_curve)
        bytes_data = visualizer.figure_to_bytes(fig, format="png")

        assert isinstance(bytes_data, bytes)
        assert len(bytes_data) > 0
        # PNG magic bytes
        assert bytes_data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_figure(self, visualizer, sample_curve, tmp_path):
        """Test figure saving."""
        fig = visualizer.plot_single_curve(sample_curve)
        output_path = tmp_path / "test_curve.png"

        result_path = visualizer.save_figure(fig, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPlotStyles:
    """Tests for different plot styles."""

    @pytest.fixture
    def sample_curve(self):
        """Create sample curve."""
        inputs = list(np.linspace(0, 1, 50))
        outputs = list(np.array(inputs) ** 0.9)
        return CurveData(
            name="Test",
            input_values=inputs,
            output_values=outputs,
        )

    @pytest.fixture
    def visualizer(self):
        """Create visualizer."""
        return CurveVisualizer()

    @pytest.mark.parametrize(
        "style",
        [
            PlotStyle.LINE,
            PlotStyle.LINE_MARKERS,
            PlotStyle.SCATTER,
            PlotStyle.AREA,
            PlotStyle.STEP,
        ],
    )
    def test_all_plot_styles(self, visualizer, sample_curve, style):
        """Test all plot styles work without error."""
        fig = visualizer.plot_single_curve(sample_curve, style=style)
        assert fig is not None


class TestColorSchemes:
    """Tests for color schemes."""

    @pytest.mark.parametrize(
        "scheme",
        [
            ColorScheme.PLATINUM,
            ColorScheme.MONOCHROME,
            ColorScheme.VIBRANT,
            ColorScheme.PASTEL,
            ColorScheme.ACCESSIBLE,
        ],
    )
    def test_all_color_schemes(self, scheme):
        """Test all color schemes provide valid colors."""
        config = VisualizationConfig(color_scheme=scheme)
        colors = config.get_color_palette(5)

        assert len(colors) == 5
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format
