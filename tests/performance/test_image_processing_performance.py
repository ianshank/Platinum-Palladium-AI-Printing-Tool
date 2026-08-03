"""
Image Processing Performance Tests.

Benchmark tests for image processing operations.
"""

import numpy as np
import pytest

try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.benchmark
class TestImageProcessingPerformance:
    """Benchmark tests for image processing operations."""

    @pytest.fixture
    def processor(self):
        """Create image processor."""
        from ptpd_calibration.imaging import ImageProcessor

        return ImageProcessor()

    @pytest.fixture
    def small_image(self, tmp_path):
        """Create a small test image (800x600)."""
        from PIL import Image

        img_array = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        image_path = tmp_path / "small_image.png"
        Image.fromarray(img_array, mode="L").save(image_path)
        return image_path

    @pytest.fixture
    def medium_image(self, tmp_path):
        """Create a medium test image (2000x1500)."""
        from PIL import Image

        img_array = np.random.randint(0, 255, (1500, 2000), dtype=np.uint8)
        image_path = tmp_path / "medium_image.png"
        Image.fromarray(img_array, mode="L").save(image_path)
        return image_path

    @pytest.fixture
    def sample_curve(self):
        """Create a sample curve for image processing."""
        from ptpd_calibration.core.models import CurveData

        input_values = list(np.linspace(0, 1, 256))
        output_values = [x**0.9 for x in input_values]

        return CurveData(
            name="Test Curve",
            input_values=input_values,
            output_values=output_values,
        )

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_load_image(self, benchmark, processor, small_image):  # noqa: ARG002
        """Benchmark image loading."""
        from PIL import Image

        result = benchmark(Image.open, small_image)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_apply_curve_small(self, benchmark, processor, small_image, sample_curve):
        """Benchmark curve application on small image."""
        result = benchmark(processor.apply_curve, small_image, sample_curve)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_apply_curve_medium(self, benchmark, processor, medium_image, sample_curve):
        """Benchmark curve application on medium image."""
        result = benchmark(processor.apply_curve, medium_image, sample_curve)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_invert_image(self, benchmark, processor, small_image):
        """Benchmark image inversion."""
        result = benchmark(processor.invert, small_image)
        assert result is not None


@pytest.mark.performance
@pytest.mark.benchmark
class TestHistogramPerformance:
    """Benchmark tests for histogram analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create histogram analyzer."""
        from ptpd_calibration.imaging import HistogramAnalyzer

        return HistogramAnalyzer()

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data."""
        return np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_compute_histogram(self, benchmark, analyzer, sample_image_data):
        """Benchmark histogram computation."""
        from PIL import Image

        img = Image.fromarray(sample_image_data, mode="L")
        result = benchmark(analyzer.compute_histogram, img)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_analyze_distribution(self, benchmark, analyzer, sample_image_data):
        """Benchmark distribution analysis."""
        from PIL import Image

        img = Image.fromarray(sample_image_data, mode="L")
        result = benchmark(analyzer.analyze, img)
        assert result is not None


@pytest.mark.performance
@pytest.mark.benchmark
class TestStepTabletPerformance:
    """Benchmark tests for step tablet processing."""

    @pytest.fixture
    def reader(self):
        """Create step tablet reader."""
        from ptpd_calibration.detection import StepTabletReader

        return StepTabletReader()

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_read_step_tablet(self, benchmark, reader, sample_step_tablet_image):
        """Benchmark step tablet reading."""
        result = benchmark(reader.read, sample_step_tablet_image)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_extract_densities(self, benchmark, reader, sample_step_tablet_image):
        """Benchmark density extraction."""

        def extract():
            result = reader.read(sample_step_tablet_image)
            return result.extraction.get_densities()

        densities = benchmark(extract)
        assert len(densities) > 0


@pytest.mark.performance
class TestImagePerformanceThresholds:
    """Test that image processing meets performance thresholds."""

    def test_small_image_processing_under_500ms(self, tmp_path):
        """Small image processing should complete in under 500ms."""
        import time

        from PIL import Image

        from ptpd_calibration.core.models import CurveData
        from ptpd_calibration.imaging import ImageProcessor

        # Create test image
        img_array = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        image_path = tmp_path / "test_image.png"
        Image.fromarray(img_array, mode="L").save(image_path)

        # Create curve
        curve = CurveData(
            name="Test",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=[x**0.9 for x in np.linspace(0, 1, 256)],
        )

        processor = ImageProcessor()

        start = time.perf_counter()
        processor.apply_curve(image_path, curve)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Processing took {elapsed:.3f}s (>500ms)"

    def test_step_tablet_read_under_2s(self, sample_step_tablet_image):
        """Step tablet reading should complete in under 2 seconds."""
        import time

        from ptpd_calibration.detection import StepTabletReader

        reader = StepTabletReader()

        start = time.perf_counter()
        reader.read(sample_step_tablet_image)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Step tablet read took {elapsed:.3f}s (>2s)"
