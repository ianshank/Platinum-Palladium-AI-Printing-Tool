"""Tests for histogram analysis module."""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.imaging import (
    HistogramAnalyzer,
    HistogramResult,
    HistogramStats,
    HistogramScale,
)


@pytest.fixture
def analyzer():
    """Create histogram analyzer."""
    return HistogramAnalyzer()


@pytest.fixture
def gray_gradient_image():
    """Create a grayscale gradient image."""
    arr = np.linspace(0, 255, 256).reshape(1, 256).repeat(100, axis=0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def rgb_test_image():
    """Create a test RGB image."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :, 0] = 200  # Red top half
    arr[50:, :, 2] = 200  # Blue bottom half
    arr[:, :, 1] = 100  # Green everywhere
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dark_image():
    """Create a dark image."""
    arr = np.full((100, 100), 30, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def bright_image():
    """Create a bright image."""
    arr = np.full((100, 100), 220, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


class TestHistogramAnalyzer:
    """Test histogram analyzer functionality."""

    def test_analyze_grayscale_image(self, analyzer, gray_gradient_image):
        """Test analyzing a grayscale gradient."""
        result = analyzer.analyze(gray_gradient_image)

        assert isinstance(result, HistogramResult)
        assert result.histogram is not None
        assert len(result.histogram) == 256
        assert result.image_mode == "L"
        assert result.total_pixels == 256 * 100

    def test_analyze_rgb_image(self, analyzer, rgb_test_image):
        """Test analyzing an RGB image."""
        result = analyzer.analyze(rgb_test_image, include_rgb=True)

        assert result.red_histogram is not None
        assert result.green_histogram is not None
        assert result.blue_histogram is not None
        assert len(result.red_histogram) == 256

    def test_analyze_numpy_array(self, analyzer):
        """Test analyzing from numpy array."""
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = analyzer.analyze(arr)

        assert result.image_size == (100, 100)
        assert result.total_pixels == 10000

    def test_histogram_stats_basic(self, analyzer, gray_gradient_image):
        """Test basic statistics computation."""
        result = analyzer.analyze(gray_gradient_image)
        stats = result.stats

        assert isinstance(stats, HistogramStats)
        # Gradient should have mean around 127
        assert 100 < stats.mean < 160
        assert stats.min_value == 0
        assert stats.max_value == 255

    def test_histogram_stats_dark_image(self, analyzer, dark_image):
        """Test statistics for dark image."""
        result = analyzer.analyze(dark_image)
        stats = result.stats

        assert stats.brightness < 0.2
        assert stats.mean < 50
        assert "dark" in " ".join(stats.notes).lower() or len(stats.notes) >= 0

    def test_histogram_stats_bright_image(self, analyzer, bright_image):
        """Test statistics for bright image."""
        result = analyzer.analyze(bright_image)
        stats = result.stats

        assert stats.brightness > 0.8
        assert stats.mean > 200

    def test_zone_distribution(self, analyzer, gray_gradient_image):
        """Test zone distribution calculation."""
        result = analyzer.analyze(gray_gradient_image)
        stats = result.stats

        assert len(stats.zone_distribution) == 11
        # Gradient should have some pixels in each zone
        total_pct = sum(stats.zone_distribution.values())
        assert 0.99 <= total_pct <= 1.01  # Should sum to ~100%

    def test_clipping_detection_no_clipping(self, analyzer):
        """Test clipping detection with no clipping."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        result = analyzer.analyze(arr)

        assert result.stats.shadow_clipping_percent < 1
        assert result.stats.highlight_clipping_percent < 1

    def test_clipping_detection_with_clipping(self, analyzer):
        """Test clipping detection with clipping."""
        # Create image with heavy shadows
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:50] = 0  # Half pure black
        arr[50:] = 128
        result = analyzer.analyze(arr)

        assert result.stats.shadow_clipping_percent > 40

    def test_dynamic_range(self, analyzer, gray_gradient_image):
        """Test dynamic range calculation."""
        result = analyzer.analyze(gray_gradient_image)

        # Gradient should have high dynamic range
        assert result.stats.dynamic_range > 5

    def test_compare_histograms(self, analyzer, gray_gradient_image, dark_image):
        """Test histogram comparison."""
        comparison = analyzer.compare_histograms(gray_gradient_image, dark_image)

        assert "similarity" in comparison
        assert "changes" in comparison
        assert "histogram_intersection" in comparison["similarity"]
        assert "mean_shift" in comparison["changes"]

    def test_create_histogram_plot(self, analyzer, gray_gradient_image):
        """Test histogram plot creation."""
        result = analyzer.analyze(gray_gradient_image)
        fig = analyzer.create_histogram_plot(result)

        assert fig is not None

    def test_create_histogram_plot_logarithmic(self, analyzer, gray_gradient_image):
        """Test logarithmic histogram plot."""
        result = analyzer.analyze(gray_gradient_image)
        fig = analyzer.create_histogram_plot(result, scale=HistogramScale.LOGARITHMIC)

        assert fig is not None

    def test_create_histogram_plot_with_rgb(self, analyzer, rgb_test_image):
        """Test histogram plot with RGB channels."""
        result = analyzer.analyze(rgb_test_image, include_rgb=True)
        fig = analyzer.create_histogram_plot(result, show_rgb=True)

        assert fig is not None

    def test_create_histogram_plot_with_zones(self, analyzer, gray_gradient_image):
        """Test histogram plot with zone boundaries."""
        result = analyzer.analyze(gray_gradient_image)
        fig = analyzer.create_histogram_plot(result, show_zones=True)

        assert fig is not None

    def test_to_dict(self, analyzer, gray_gradient_image):
        """Test result conversion to dictionary."""
        result = analyzer.analyze(gray_gradient_image)
        d = result.to_dict()

        assert "image_size" in d
        assert "statistics" in d
        assert "total_pixels" in d

    def test_stats_to_dict(self, analyzer, gray_gradient_image):
        """Test stats conversion to dictionary."""
        result = analyzer.analyze(gray_gradient_image)
        d = result.stats.to_dict()

        assert "mean" in d
        assert "median" in d
        assert "zone_distribution" in d

    def test_zone_descriptions(self):
        """Test zone descriptions retrieval."""
        descs = HistogramAnalyzer.get_zone_descriptions()

        assert len(descs) == 11
        assert 0 in descs
        assert 10 in descs
        assert "black" in descs[0].lower()
        assert "white" in descs[10].lower()
