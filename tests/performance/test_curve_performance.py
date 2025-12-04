"""
Curve Generation and Modification Performance Tests.

Benchmark tests for curve-related operations.
"""

import pytest

# Try to import benchmark, skip if not available
try:
    import pytest_benchmark

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.benchmark
class TestCurveGenerationPerformance:
    """Benchmark tests for curve generation."""

    @pytest.fixture
    def generator(self):
        """Create curve generator."""
        from ptpd_calibration.curves import CurveGenerator

        return CurveGenerator()

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_generate_linear_curve_small(
        self, benchmark, generator, small_densities
    ):
        """Benchmark linear curve generation with small dataset."""
        from ptpd_calibration.core.types import CurveType

        result = benchmark(
            generator.generate,
            small_densities,
            curve_type=CurveType.LINEAR,
            name="Benchmark Curve",
        )

        assert result is not None
        assert len(result.input_values) > 0

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_generate_linear_curve_large(
        self, benchmark, generator, large_densities
    ):
        """Benchmark linear curve generation with large dataset."""
        from ptpd_calibration.core.types import CurveType

        result = benchmark(
            generator.generate,
            large_densities,
            curve_type=CurveType.LINEAR,
            name="Benchmark Curve",
        )

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_generate_spline_curve(self, benchmark, generator, medium_densities):
        """Benchmark spline curve generation."""
        from ptpd_calibration.core.types import CurveType

        result = benchmark(
            generator.generate,
            medium_densities,
            curve_type=CurveType.SPLINE,
            name="Benchmark Curve",
        )

        assert result is not None


@pytest.mark.performance
@pytest.mark.benchmark
class TestCurveModificationPerformance:
    """Benchmark tests for curve modification."""

    @pytest.fixture
    def modifier(self):
        """Create curve modifier."""
        from ptpd_calibration.curves import CurveModifier

        return CurveModifier()

    @pytest.fixture
    def sample_curve(self, small_curve):
        """Create a sample curve for modification tests."""
        from ptpd_calibration.core.models import CurveData

        return CurveData(
            name="Test Curve",
            input_values=small_curve["input"],
            output_values=small_curve["output"],
        )

    @pytest.fixture
    def large_sample_curve(self, large_curve):
        """Create a large curve for modification tests."""
        from ptpd_calibration.core.models import CurveData

        return CurveData(
            name="Large Test Curve",
            input_values=large_curve["input"],
            output_values=large_curve["output"],
        )

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_brightness_adjustment(self, benchmark, modifier, sample_curve):
        """Benchmark brightness adjustment."""
        result = benchmark(modifier.adjust_brightness, sample_curve, 0.1)

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_contrast_adjustment(self, benchmark, modifier, sample_curve):
        """Benchmark contrast adjustment."""
        result = benchmark(modifier.adjust_contrast, sample_curve, 0.2)

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_gamma_adjustment(self, benchmark, modifier, sample_curve):
        """Benchmark gamma adjustment."""
        result = benchmark(modifier.adjust_gamma, sample_curve, 1.2)

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_smooth_gaussian(self, benchmark, modifier, sample_curve):
        """Benchmark Gaussian smoothing."""
        from ptpd_calibration.curves import SmoothingMethod

        result = benchmark(
            modifier.smooth,
            sample_curve,
            method=SmoothingMethod.GAUSSIAN,
            strength=0.5,
        )

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_smooth_large_curve(self, benchmark, modifier, large_sample_curve):
        """Benchmark smoothing on large curve."""
        from ptpd_calibration.curves import SmoothingMethod

        result = benchmark(
            modifier.smooth,
            large_sample_curve,
            method=SmoothingMethod.GAUSSIAN,
            strength=0.5,
        )

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_enforce_monotonicity(self, benchmark, modifier, sample_curve):
        """Benchmark monotonicity enforcement."""
        result = benchmark(modifier.enforce_monotonicity, sample_curve)

        assert result is not None


@pytest.mark.performance
@pytest.mark.benchmark
class TestCurveBlendingPerformance:
    """Benchmark tests for curve blending."""

    @pytest.fixture
    def modifier(self):
        """Create curve modifier."""
        from ptpd_calibration.curves import CurveModifier

        return CurveModifier()

    @pytest.fixture
    def curve_pair(self, small_curve):
        """Create a pair of curves for blending tests."""
        from ptpd_calibration.core.models import CurveData

        curve1 = CurveData(
            name="Curve 1",
            input_values=small_curve["input"],
            output_values=small_curve["output"],
        )

        curve2 = CurveData(
            name="Curve 2",
            input_values=small_curve["input"],
            output_values=[x**1.1 for x in small_curve["input"]],
        )

        return curve1, curve2

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_blend_average(self, benchmark, modifier, curve_pair):
        """Benchmark average blending."""
        from ptpd_calibration.curves import BlendMode

        curve1, curve2 = curve_pair
        result = benchmark(
            modifier.blend, curve1, curve2, mode=BlendMode.AVERAGE
        )

        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_blend_weighted(self, benchmark, modifier, curve_pair):
        """Benchmark weighted blending."""
        from ptpd_calibration.curves import BlendMode

        curve1, curve2 = curve_pair
        result = benchmark(
            modifier.blend, curve1, curve2, mode=BlendMode.WEIGHTED, weight=0.7
        )

        assert result is not None


@pytest.mark.performance
class TestCurvePerformanceThresholds:
    """Test that curve operations meet performance thresholds."""

    def test_generate_curve_under_100ms(self, small_densities):
        """Curve generation should complete in under 100ms."""
        import time

        from ptpd_calibration.curves import CurveGenerator
        from ptpd_calibration.core.types import CurveType

        generator = CurveGenerator()

        start = time.perf_counter()
        for _ in range(10):
            generator.generate(small_densities, curve_type=CurveType.LINEAR)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.1, f"Curve generation took {elapsed:.3f}s (>100ms)"

    def test_modification_under_50ms(self, small_curve):
        """Curve modification should complete in under 50ms."""
        import time

        from ptpd_calibration.curves import CurveModifier
        from ptpd_calibration.core.models import CurveData

        modifier = CurveModifier()
        curve = CurveData(
            name="Test",
            input_values=small_curve["input"],
            output_values=small_curve["output"],
        )

        start = time.perf_counter()
        for _ in range(10):
            modifier.adjust_brightness(curve, 0.1)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.05, f"Modification took {elapsed:.3f}s (>50ms)"
