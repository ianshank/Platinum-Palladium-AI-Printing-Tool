"""
Image Processing Performance Tests.

Benchmark tests for image processing operations.

These are excluded from PR CI (``-m "not performance"`` plus an explicit
``--ignore``) because wall-clock thresholds flake on a shared runner. That
exclusion is also why they went stale: every test here called an API that had
since changed, so the suite measured nothing at all. The calls below are the
ones the application actually makes, so the next signature change breaks them
loudly.

The thresholds are wall-clock and therefore hardware-dependent. Each reads an
environment override so a slower machine can relax it without editing the file.
"""

import os

import numpy as np
import pytest

try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


def _threshold_seconds(name: str, default: float) -> float:
    """Read a wall-clock budget from the environment, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        pytest.fail(f"{name} must be a number of seconds, got {raw!r}")


#: Budget for decoding a small image and applying one curve to it.
SMALL_IMAGE_BUDGET_S = _threshold_seconds("PTPD_PERF_SMALL_IMAGE_SECONDS", 0.5)
#: Budget for reading a step tablet end to end.
STEP_TABLET_BUDGET_S = _threshold_seconds("PTPD_PERF_STEP_TABLET_SECONDS", 2.0)


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
        Image.fromarray(img_array).save(image_path)
        return image_path

    @pytest.fixture
    def medium_image(self, tmp_path):
        """Create a medium test image (2000x1500)."""
        from PIL import Image

        img_array = np.random.randint(0, 255, (1500, 2000), dtype=np.uint8)
        image_path = tmp_path / "medium_image.png"
        Image.fromarray(img_array).save(image_path)
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
    def test_load_image(self, benchmark, processor, small_image):
        """Benchmark image loading through the path the application uses.

        This measured ``Image.open`` before, which is lazy: it parsed a header
        and left the file open, so the benchmark's several hundred rounds each
        leaked a handle. The resulting ``ResourceWarning`` is unraisable, so
        ``filterwarnings = error`` attributed it to whichever test the collector
        happened to run during, failing unrelated tests elsewhere in the
        session. ``load_image`` decodes eagerly and closes the handle.
        """
        result = benchmark(processor.load_image, small_image)
        assert result.image is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_apply_curve_small(self, benchmark, processor, small_image, sample_curve):
        """Benchmark curve application on small image.

        ``apply_curve`` takes the ``ProcessingResult`` that ``load_image``
        returns, not a path. Loading outside the benchmark also stops the decode
        time being counted as curve time.
        """
        loaded = processor.load_image(small_image)

        result = benchmark(processor.apply_curve, loaded, sample_curve)

        assert result.image is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_apply_curve_medium(self, benchmark, processor, medium_image, sample_curve):
        """Benchmark curve application on medium image."""
        loaded = processor.load_image(medium_image)

        result = benchmark(processor.apply_curve, loaded, sample_curve)

        assert result.image is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_invert_image(self, benchmark, processor, small_image):
        """Benchmark image inversion."""
        loaded = processor.load_image(small_image)

        result = benchmark(processor.invert, loaded)

        assert result.image is not None


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
    def test_compare_histograms(self, benchmark, analyzer, sample_image_data):
        """Benchmark comparing two histograms.

        This called ``compute_histogram``, which the analyser has never had, so
        it raised on every run and measured nothing. ``compare_histograms`` is
        the operation of that shape that does exist, and it is the one public
        method here that ``test_analyze_distribution`` does not already cover.
        """
        from PIL import Image

        first = Image.fromarray(sample_image_data)
        second = Image.fromarray(np.roll(sample_image_data, 1, axis=0))

        result = benchmark(analyzer.compare_histograms, first, second)

        assert result

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_analyze_distribution(self, benchmark, analyzer, sample_image_data):
        """Benchmark distribution analysis."""
        from PIL import Image

        img = Image.fromarray(sample_image_data)

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

    def test_small_image_processing_within_budget(self, tmp_path):
        """Decoding a small image and applying a curve stays within budget.

        The decode is timed with the curve because that is the unit a caller
        experiences. ``apply_curve`` was previously handed the path directly,
        which raised before anything was measured.
        """
        import time

        from PIL import Image

        from ptpd_calibration.core.models import CurveData
        from ptpd_calibration.imaging import ImageProcessor

        img_array = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        image_path = tmp_path / "test_image.png"
        Image.fromarray(img_array).save(image_path)

        curve = CurveData(
            name="Test",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=[x**0.9 for x in np.linspace(0, 1, 256)],
        )

        processor = ImageProcessor()

        start = time.perf_counter()
        processor.apply_curve(processor.load_image(image_path), curve)
        elapsed = time.perf_counter() - start

        assert elapsed < SMALL_IMAGE_BUDGET_S, (
            f"Processing took {elapsed:.3f}s, over the "
            f"{SMALL_IMAGE_BUDGET_S:.3f}s budget "
            f"(raise PTPD_PERF_SMALL_IMAGE_SECONDS on slower hardware)"
        )

    def test_step_tablet_read_within_budget(self, sample_step_tablet_image):
        """Step tablet reading stays within budget."""
        import time

        from ptpd_calibration.detection import StepTabletReader

        reader = StepTabletReader()

        start = time.perf_counter()
        reader.read(sample_step_tablet_image)
        elapsed = time.perf_counter() - start

        assert elapsed < STEP_TABLET_BUDGET_S, (
            f"Step tablet read took {elapsed:.3f}s, over the "
            f"{STEP_TABLET_BUDGET_S:.3f}s budget "
            f"(raise PTPD_PERF_STEP_TABLET_SECONDS on slower hardware)"
        )
