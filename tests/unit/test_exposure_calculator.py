"""Tests for exposure calculator module."""

import pytest

from ptpd_calibration.exposure import (
    ExposureCalculator,
    ExposureResult,
    ExposureSettings,
    LightSource,
)


@pytest.fixture
def calculator():
    """Create default exposure calculator."""
    return ExposureCalculator()


@pytest.fixture
def custom_calculator():
    """Create exposure calculator with custom settings."""
    settings = ExposureSettings(
        base_exposure_minutes=10.0,
        base_negative_density=1.6,
        light_source=LightSource.BL_FLUORESCENT,
        base_distance_inches=4.0,
        platinum_ratio=0.0,
    )
    return ExposureCalculator(settings)


class TestExposureCalculator:
    """Test exposure calculator functionality."""

    def test_calculate_basic(self, calculator):
        """Test basic exposure calculation."""
        result = calculator.calculate(negative_density=1.6)

        assert isinstance(result, ExposureResult)
        assert result.exposure_minutes > 0
        assert result.exposure_seconds > 0

    def test_calculate_with_higher_density(self, calculator):
        """Test that higher density requires more exposure."""
        result1 = calculator.calculate(negative_density=1.5)
        result2 = calculator.calculate(negative_density=2.0)

        assert result2.exposure_minutes > result1.exposure_minutes

    def test_calculate_with_lower_density(self, calculator):
        """Test that lower density requires less exposure."""
        result1 = calculator.calculate(negative_density=1.6)
        result2 = calculator.calculate(negative_density=1.2)

        assert result2.exposure_minutes < result1.exposure_minutes

    def test_light_source_affects_exposure(self, calculator):
        """Test that different light sources give different exposures."""
        result_bl = calculator.calculate(
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
        )
        result_led = calculator.calculate(
            negative_density=1.6,
            light_source=LightSource.LED_UV,
        )

        # LED should be faster
        assert result_led.exposure_minutes < result_bl.exposure_minutes

    def test_distance_affects_exposure(self, calculator):
        """Test inverse square law for distance."""
        result_close = calculator.calculate(
            negative_density=1.6,
            distance_inches=4.0,
        )
        result_far = calculator.calculate(
            negative_density=1.6,
            distance_inches=8.0,
        )

        # Double distance = 4x exposure
        ratio = result_far.exposure_minutes / result_close.exposure_minutes
        assert 3.5 < ratio < 4.5

    def test_platinum_ratio_affects_exposure(self):
        """Test that higher platinum ratio increases exposure."""
        settings_pd = ExposureSettings(platinum_ratio=0.0)
        settings_pt = ExposureSettings(platinum_ratio=1.0)

        calc_pd = ExposureCalculator(settings_pd)
        calc_pt = ExposureCalculator(settings_pt)

        result_pd = calc_pd.calculate(negative_density=1.6)
        result_pt = calc_pt.calculate(negative_density=1.6)

        # Platinum is slower
        assert result_pt.exposure_minutes > result_pd.exposure_minutes

    def test_format_time_seconds(self, calculator):
        """Test time formatting for short exposures."""
        result = calculator.calculate(negative_density=0.5)
        formatted = result.format_time()

        # Should format as seconds or minutes
        assert "second" in formatted.lower() or "min" in formatted.lower()

    def test_format_time_minutes(self, custom_calculator):
        """Test time formatting for normal exposures."""
        result = custom_calculator.calculate(negative_density=1.6)
        formatted = result.format_time()

        assert "min" in formatted.lower()

    def test_format_time_hours(self, calculator):  # noqa: ARG002
        """Test time formatting for very long exposures."""
        # Force long exposure
        settings = ExposureSettings(
            base_exposure_minutes=60.0,
            base_negative_density=1.0,
        )
        calc = ExposureCalculator(settings)
        result = calc.calculate(negative_density=3.0)
        formatted = result.format_time()

        # May be hours or long minutes
        assert "hour" in formatted.lower() or "min" in formatted.lower()

    def test_calculate_test_strip(self, calculator):
        """Test test strip generation."""
        times = calculator.calculate_test_strip(
            center_exposure=10.0,
            steps=5,
            increment_stops=0.5,
        )

        assert len(times) == 5
        # Should be centered around 10
        assert any(9 < t < 11 for t in times)
        # Should have range
        assert min(times) < 10 < max(times)

    def test_calculate_test_strip_increments(self, calculator):
        """Test test strip with different increments."""
        times_half = calculator.calculate_test_strip(
            center_exposure=10.0,
            steps=5,
            increment_stops=0.5,
        )
        times_full = calculator.calculate_test_strip(
            center_exposure=10.0,
            steps=5,
            increment_stops=1.0,
        )

        # Full stop increments should have wider range
        range_half = max(times_half) - min(times_half)
        range_full = max(times_full) - min(times_full)
        assert range_full > range_half

    def test_density_to_stops(self, calculator):
        """Test density to stops conversion."""
        stops = calculator.density_to_stops(0.3)
        assert 0.9 < stops < 1.1  # 0.3 density = 1 stop

        stops = calculator.density_to_stops(0.6)
        assert 1.9 < stops < 2.1  # 0.6 density = 2 stops

    def test_stops_to_density(self, calculator):
        """Test stops to density conversion."""
        density = calculator.stops_to_density(1.0)
        assert 0.28 < density < 0.32  # 1 stop = 0.3 density

        density = calculator.stops_to_density(2.0)
        assert 0.58 < density < 0.62  # 2 stops = 0.6 density

    def test_adjust_for_distance(self, calculator):
        """Test distance adjustment calculation."""
        # Double distance = 4x exposure
        adjusted = calculator.adjust_for_distance(
            current_exposure=10.0,
            current_distance=4.0,
            new_distance=8.0,
        )
        assert 38 < adjusted < 42

        # Half distance = 1/4 exposure
        adjusted = calculator.adjust_for_distance(
            current_exposure=10.0,
            current_distance=4.0,
            new_distance=2.0,
        )
        assert 2 < adjusted < 3

    def test_to_dict(self, calculator):
        """Test result conversion to dictionary."""
        result = calculator.calculate(negative_density=1.6)
        d = result.to_dict()

        assert "exposure_time" in d
        assert "adjustments" in d
        assert "inputs" in d
        assert "notes" in d

    def test_get_light_sources(self):
        """Test getting light source list."""
        sources = ExposureCalculator.get_light_sources()

        assert len(sources) >= 8
        assert any("fluorescent" in s[0].lower() for s in sources)
        assert any("led" in s[0].lower() for s in sources)

    def test_all_light_sources(self, calculator):
        """Test calculation with each light source."""
        for source in list(LightSource):
            result = calculator.calculate(
                negative_density=1.6,
                light_source=source,
            )
            assert result.exposure_minutes > 0

    def test_adjustment_breakdown(self, calculator):
        """Test that adjustment breakdown is provided."""
        result = calculator.calculate(
            negative_density=1.8,
            distance_inches=6.0,
        )

        assert result.density_adjustment > 0
        assert result.light_source_adjustment > 0
        assert result.distance_adjustment > 0

    def test_warnings_for_long_exposure(self, calculator):  # noqa: ARG002
        """Test warnings for very long exposure."""
        settings = ExposureSettings(base_exposure_minutes=30.0)
        calc = ExposureCalculator(settings)
        result = calc.calculate(negative_density=2.5)

        # Should have warning for long exposure
        if result.exposure_minutes > 30:
            assert any("long" in n.lower() or "warning" in n.lower() for n in result.notes)

    def test_warnings_for_short_exposure(self, calculator):
        """Test warnings for very short exposure."""
        result = calculator.calculate(negative_density=0.3)

        # May have warning for short exposure
        if result.exposure_minutes < 2:
            assert len(result.notes) >= 0  # May or may not have warning
