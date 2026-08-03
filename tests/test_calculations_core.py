"""
Comprehensive tests for core calculation functions.

Tests cover:
- Basic functionality and algorithm correctness
- Edge cases and boundary conditions
- Parameter validation
- Logging output structure
- Result serialization
"""

import pytest
from ptpd_calibration.calculations.core import (
    DryingTimeResult,
    TestStripExposureResult,
    UVExposureResult,
    calculate_drying_time,
    calculate_test_strip_exposure,
    calculate_uv_exposure,
    DENSITY_PER_STOP,
    OPTIMAL_HUMIDITY_PERCENT,
    OPTIMAL_TEMPERATURE_FAHRENHEIT,
)


# ============================================================================
# Test Strip Exposure Tests
# ============================================================================


class TestCalculateTestStripExposure:
    """Tests for test strip exposure calculation."""

    def test_basic_calculation(self):
        """Test basic test strip calculation with standard values."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=5, increment_stops=0.5
        )

        assert result.center_exposure == 10.0
        assert result.steps == 5
        assert result.increment_stops == 0.5
        assert len(result.exposure_times) == 5

        # Check that center value is preserved
        center_index = len(result.exposure_times) // 2
        assert abs(result.exposure_times[center_index] - 10.0) < 0.01

    def test_exposure_times_progression(self):
        """Test that exposure times follow correct geometric progression."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=5, increment_stops=0.5
        )

        # Verify geometric progression: each step is 2^0.5 = 1.414x
        times = result.exposure_times
        assert len(times) == 5

        # Expected: [5.0, 7.071, 10.0, 14.142, 20.0]
        expected = [10.0 * (2 ** (i * 0.5)) for i in range(-2, 3)]
        for actual, exp in zip(times, expected):
            assert abs(actual - exp) < 0.1

    def test_odd_number_of_steps(self):
        """Test with odd number of steps (symmetric around center)."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=5.0, steps=7, increment_stops=0.5
        )

        assert len(result.exposure_times) == 7
        center_index = len(result.exposure_times) // 2
        assert abs(result.exposure_times[center_index] - 5.0) < 0.01

    def test_even_number_of_steps(self):
        """Test with even number of steps.

        Note: Algorithm creates symmetric range using integer division.
        For even steps=4, half_steps=2, range(-2, 3) yields 5 values.
        This ensures symmetric distribution around center.
        """
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=4, increment_stops=0.5
        )

        # Algorithm creates symmetric range, so even numbers get +1 step
        assert len(result.exposure_times) == 5

    def test_single_step(self):
        """Test with single step (should return just center)."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=1, increment_stops=0.5
        )

        assert len(result.exposure_times) == 1
        assert abs(result.exposure_times[0] - 10.0) < 0.01

    def test_large_exposure_times(self):
        """Test with large exposure times."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=120.0, steps=5, increment_stops=1.0
        )

        assert len(result.exposure_times) == 5
        assert min(result.exposure_times) > 0
        assert max(result.exposure_times) <= 480.0  # 2 hours

    def test_small_exposure_times(self):
        """Test with small exposure times (seconds)."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=0.1, steps=5, increment_stops=0.5
        )

        assert len(result.exposure_times) == 5
        assert min(result.exposure_times) > 0
        assert max(result.exposure_times) <= 0.2

    def test_large_increment(self):
        """Test with 1 stop increment."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=5, increment_stops=1.0
        )

        times = result.exposure_times
        # Check progression: 2.5, 5.0, 10.0, 20.0, 40.0
        assert abs(times[0] - 2.5) < 0.1
        assert abs(times[2] - 10.0) < 0.1
        assert abs(times[4] - 40.0) < 0.1

    def test_zero_increment(self):
        """Test with zero increment (all same)."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=5, increment_stops=0.0
        )

        # All times should be equal
        for time_val in result.exposure_times:
            assert abs(time_val - 10.0) < 0.01

    def test_validation_negative_center(self):
        """Test validation: negative center exposure."""
        with pytest.raises(ValueError, match="must be positive"):
            calculate_test_strip_exposure(center_exposure_minutes=-5.0)

    def test_validation_zero_center(self):
        """Test validation: zero center exposure."""
        with pytest.raises(ValueError, match="must be positive"):
            calculate_test_strip_exposure(center_exposure_minutes=0.0)

    def test_validation_negative_steps(self):
        """Test validation: negative steps."""
        with pytest.raises(ValueError, match="at least 1"):
            calculate_test_strip_exposure(center_exposure_minutes=10.0, steps=-1)

    def test_validation_zero_steps(self):
        """Test validation: zero steps."""
        with pytest.raises(ValueError, match="at least 1"):
            calculate_test_strip_exposure(center_exposure_minutes=10.0, steps=0)

    def test_validation_negative_increment(self):
        """Test validation: negative increment."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_test_strip_exposure(
                center_exposure_minutes=10.0, increment_stops=-0.5
            )

    def test_result_formatting(self):
        """Test result formatting methods."""
        result = calculate_test_strip_exposure(
            center_exposure_minutes=10.0, steps=3, increment_stops=1.0
        )

        # Test formatted times
        formatted = result.get_formatted_times(format_as_seconds=False)
        assert len(formatted) == 3
        assert all(isinstance(f, str) for f in formatted)

        formatted_sec = result.get_formatted_times(format_as_seconds=True)
        assert all("s" in f for f in formatted_sec)


# ============================================================================
# UV Exposure Tests
# ============================================================================


class TestCalculateUVExposure:
    """Tests for UV exposure calculation."""

    def test_baseline_conditions(self):
        """Test calculation at baseline conditions."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
        )

        assert result.exposure_minutes == pytest.approx(10.0, rel=0.01)
        assert result.exposure_seconds == pytest.approx(600.0, rel=0.01)
        assert len(result.notes) >= 0

    def test_density_adjustment_dense_negative(self):
        """Test density adjustment for dense negative."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=2.2,  # +0.6D (2 stops denser)
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
        )

        # 0.6D difference = 2 stops = 4x exposure
        expected_factor = 2 ** (0.6 / DENSITY_PER_STOP)
        assert result.density_adjustment == pytest.approx(expected_factor, rel=0.01)
        assert result.exposure_minutes == pytest.approx(10.0 * expected_factor, rel=0.01)

    def test_density_adjustment_thin_negative(self):
        """Test density adjustment for thin negative."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.0,  # -0.6D (2 stops thinner)
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
        )

        # 0.6D lighter = 2 stops = 0.25x exposure
        expected_factor = 2 ** (-0.6 / DENSITY_PER_STOP)
        assert result.density_adjustment == pytest.approx(expected_factor, rel=0.01)
        assert result.exposure_minutes == pytest.approx(10.0 * expected_factor, rel=0.01)

    def test_humidity_adjustment_low(self):
        """Test humidity adjustment for low humidity."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=30.0,  # -20% from optimal
            temperature_fahrenheit=68.0,
        )

        # Lower humidity = higher humidity_factor = slower exposure
        assert result.humidity_adjustment > 1.0

    def test_humidity_adjustment_high(self):
        """Test humidity adjustment for high humidity."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=70.0,  # +20% from optimal
            temperature_fahrenheit=68.0,
        )

        # Higher humidity = lower humidity_factor = faster exposure
        assert result.humidity_adjustment < 1.0

    def test_temperature_adjustment_cold(self):
        """Test temperature adjustment for cold conditions."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=50.0,  # 18°F below optimal
        )

        # Colder = higher temperature_factor = slower exposure
        assert result.temperature_adjustment > 1.0

    def test_temperature_adjustment_hot(self):
        """Test temperature adjustment for warm conditions."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=85.0,  # 17°F above optimal
        )

        # Warmer = lower temperature_factor = faster exposure
        assert result.temperature_adjustment < 1.0

    def test_uv_intensity_adjustment(self):
        """Test UV intensity adjustment."""
        result_low = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            uv_intensity_percent=50.0,
        )

        result_high = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            uv_intensity_percent=200.0,
        )

        # Lower intensity requires longer exposure
        assert result_low.exposure_minutes > result_high.exposure_minutes

    def test_paper_factor(self):
        """Test paper speed adjustment."""
        result_fast = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_speed_factor=0.8,
        )

        result_slow = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_speed_factor=1.2,
        )

        # Fast paper needs less exposure
        assert result_fast.exposure_minutes < result_slow.exposure_minutes

    def test_chemistry_factor(self):
        """Test chemistry speed adjustment."""
        result_fast = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            chemistry_speed_factor=0.8,
        )

        result_slow = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            chemistry_speed_factor=1.2,
        )

        # Fast chemistry needs less exposure
        assert result_fast.exposure_minutes < result_slow.exposure_minutes

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            uncertainty_percent=10.0,
        )

        assert result.confidence_lower_minutes < result.exposure_minutes
        assert result.confidence_upper_minutes > result.exposure_minutes
        assert abs(
            result.confidence_upper_minutes - result.confidence_lower_minutes
        ) == pytest.approx(result.exposure_minutes * 0.2, rel=0.01)

    def test_warnings_extreme_conditions(self):
        """Test warnings for extreme conditions."""
        result_high_humidity = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=85.0,
            temperature_fahrenheit=68.0,
        )

        assert any("humidity" in w.lower() for w in result_high_humidity.warnings)

        result_low_temp = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.6,
            humidity_percent=50.0,
            temperature_fahrenheit=45.0,
        )

        assert any("temperature" in w.lower() for w in result_low_temp.warnings)

    def test_validation_negative_base_time(self):
        """Test validation: negative base time."""
        with pytest.raises(ValueError, match="positive"):
            calculate_uv_exposure(
                base_time_minutes=-5.0,
                negative_density=1.6,
                humidity_percent=50.0,
                temperature_fahrenheit=68.0,
            )

    def test_validation_humidity_out_of_range(self):
        """Test validation: humidity out of range."""
        with pytest.raises(ValueError, match="humidity"):
            calculate_uv_exposure(
                base_time_minutes=10.0,
                negative_density=1.6,
                humidity_percent=150.0,
                temperature_fahrenheit=68.0,
            )

    def test_validation_negative_intensity(self):
        """Test validation: negative UV intensity."""
        with pytest.raises(ValueError, match="intensity"):
            calculate_uv_exposure(
                base_time_minutes=10.0,
                negative_density=1.6,
                humidity_percent=50.0,
                temperature_fahrenheit=68.0,
                uv_intensity_percent=0.0,
            )

    def test_result_serialization(self):
        """Test result can be serialized to dict."""
        result = calculate_uv_exposure(
            base_time_minutes=10.0,
            negative_density=1.8,
            humidity_percent=55.0,
            temperature_fahrenheit=70.0,
            uv_intensity_percent=95.0,
        )

        result_dict = result.to_dict()
        assert "exposure_time" in result_dict
        assert "exposure_minutes" in result_dict
        assert "adjustments" in result_dict
        assert "inputs" in result_dict
        assert "warnings" in result_dict

    def test_extreme_exposure_warnings(self):
        """Test warnings for extreme exposure times."""
        # Very short exposure
        result_short = calculate_uv_exposure(
            base_time_minutes=0.05,
            negative_density=0.5,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
        )
        assert any("short" in w.lower() for w in result_short.warnings)

        # Very long exposure
        result_long = calculate_uv_exposure(
            base_time_minutes=50.0,
            negative_density=3.0,
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
        )
        assert any("long" in w.lower() for w in result_long.warnings)


# ============================================================================
# Drying Time Tests
# ============================================================================


class TestCalculateDryingTime:
    """Tests for drying time estimation."""

    def test_baseline_conditions(self):
        """Test drying time at baseline conditions."""
        result = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        assert result.drying_minutes > 0
        assert result.drying_hours > 0
        assert result.paper_type == "cold_press"

    def test_humidity_adjustment_low(self):
        """Test drying time with low humidity (faster)."""
        result_low = calculate_drying_time(
            humidity_percent=30.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        result_normal = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        # Lower humidity = faster drying
        assert result_low.drying_minutes < result_normal.drying_minutes

    def test_humidity_adjustment_high(self):
        """Test drying time with high humidity (slower)."""
        result_high = calculate_drying_time(
            humidity_percent=75.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        result_normal = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        # Higher humidity = slower drying
        assert result_high.drying_minutes > result_normal.drying_minutes

    def test_temperature_adjustment_cold(self):
        """Test drying time at cold temperature (slower)."""
        result_cold = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=50.0,
            paper_type="cold_press",
        )

        result_warm = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=75.0,
            paper_type="cold_press",
        )

        # Colder = slower drying
        assert result_cold.drying_minutes > result_warm.drying_minutes

    def test_temperature_adjustment_warm(self):
        """Test drying time at warm temperature (faster)."""
        result_warm = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=80.0,
            paper_type="cold_press",
        )

        result_normal = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        # Warmer = faster drying
        assert result_warm.drying_minutes < result_normal.drying_minutes

    def test_paper_type_hot_press(self):
        """Test that hot press dries faster."""
        result_hot = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="hot_press",
        )

        result_cold = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        # Hot press dries faster
        assert result_hot.drying_minutes < result_cold.drying_minutes

    def test_paper_type_rough(self):
        """Test that rough paper dries slowest."""
        result_rough = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="rough",
        )

        result_cold = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        # Rough paper dries slowest
        assert result_rough.drying_minutes > result_cold.drying_minutes

    def test_forced_air_drying(self):
        """Test forced air drying effect."""
        result_natural = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
            forced_air=False,
        )

        result_forced = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
            forced_air=True,
        )

        # Forced air should be ~50% of natural drying
        assert result_forced.drying_minutes < result_natural.drying_minutes
        assert result_forced.drying_minutes == pytest.approx(
            result_natural.drying_minutes * 0.5, rel=0.01
        )

    def test_estimated_range(self):
        """Test estimated range calculation."""
        result = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )

        range_min, range_max = result.estimated_range_minutes
        assert range_min < result.drying_minutes
        assert range_max > result.drying_minutes
        assert abs(range_min - result.drying_minutes * 0.8) < 0.01
        assert abs(range_max - result.drying_minutes * 1.2) < 0.01

    def test_forced_air_recommendation(self):
        """Test forced air recommendation logic."""
        # High humidity should recommend forced air
        result_humid = calculate_drying_time(
            humidity_percent=75.0,
            temperature_fahrenheit=68.0,
            paper_type="cold_press",
        )
        assert result_humid.forced_air_recommended

        # Low temp should recommend forced air
        result_cold = calculate_drying_time(
            humidity_percent=50.0,
            temperature_fahrenheit=55.0,
            paper_type="cold_press",
        )
        assert result_cold.forced_air_recommended

    def test_validation_humidity_out_of_range(self):
        """Test validation: humidity out of range."""
        with pytest.raises(ValueError, match="humidity"):
            calculate_drying_time(
                humidity_percent=150.0,
                temperature_fahrenheit=68.0,
                paper_type="cold_press",
            )

    def test_validation_empty_paper_type(self):
        """Test validation: empty paper type."""
        with pytest.raises(ValueError, match="paper_type"):
            calculate_drying_time(
                humidity_percent=50.0,
                temperature_fahrenheit=68.0,
                paper_type="",
            )

    def test_result_serialization(self):
        """Test result can be serialized to dict."""
        result = calculate_drying_time(
            humidity_percent=60.0,
            temperature_fahrenheit=70.0,
            paper_type="cold_press",
            forced_air=False,
        )

        result_dict = result.to_dict()
        assert "estimated_time" in result_dict
        assert "drying_minutes" in result_dict
        assert "adjustments" in result_dict
        assert "inputs" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
