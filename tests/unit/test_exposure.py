"""
Tests for exposure calculator module.

Tests UV exposure time calculations for alternative printing.
"""

import pytest

from ptpd_calibration.exposure.calculator import (
    LightSource,
    ExposureSettings,
    ExposureResult,
    ExposureCalculator,
    LIGHT_SOURCE_SPEEDS,
)


class TestLightSource:
    """Tests for LightSource enum."""

    def test_all_sources_exist(self):
        """All light source types should exist."""
        assert LightSource.NUARC_26_1K.value == "nuarc_26_1k"
        assert LightSource.BL_FLUORESCENT.value == "bl_fluorescent"
        assert LightSource.LED_UV.value == "led_uv"
        assert LightSource.SUNLIGHT.value == "sunlight"
        assert LightSource.CUSTOM.value == "custom"

    def test_light_source_speeds(self):
        """All light sources should have speed multipliers."""
        for source in list(LightSource):
            assert source in LIGHT_SOURCE_SPEEDS


class TestExposureSettings:
    """Tests for ExposureSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = ExposureSettings()
        assert settings.base_exposure_minutes == 10.0
        assert settings.base_negative_density == 1.6
        assert settings.light_source == LightSource.BL_FLUORESCENT
        assert settings.platinum_ratio == 0.0

    def test_custom_settings(self):
        """Custom settings should be applied."""
        settings = ExposureSettings(
            base_exposure_minutes=15.0,
            light_source=LightSource.LED_UV,
            platinum_ratio=0.5,
        )
        assert settings.base_exposure_minutes == 15.0
        assert settings.light_source == LightSource.LED_UV
        assert settings.platinum_ratio == 0.5


class TestExposureResult:
    """Tests for ExposureResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample exposure result."""
        return ExposureResult(
            exposure_minutes=12.5,
            exposure_seconds=750.0,
            base_exposure=10.0,
            density_adjustment=1.2,
            light_source_adjustment=1.0,
            distance_adjustment=1.0,
            paper_adjustment=1.0,
            chemistry_adjustment=1.04,
            environmental_adjustment=1.0,
            negative_density=1.7,
            light_source=LightSource.BL_FLUORESCENT,
            distance_inches=4.0,
        )

    def test_format_time_seconds(self):
        """Should format times under 1 minute as seconds."""
        result = ExposureResult(
            exposure_minutes=0.5,
            exposure_seconds=30.0,
            base_exposure=10.0,
            density_adjustment=1.0,
            light_source_adjustment=1.0,
            distance_adjustment=1.0,
            paper_adjustment=1.0,
            chemistry_adjustment=1.0,
            environmental_adjustment=1.0,
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
            distance_inches=4.0,
        )
        assert "30 seconds" in result.format_time()

    def test_format_time_minutes(self):
        """Should format times in minutes."""
        result = ExposureResult(
            exposure_minutes=5.0,
            exposure_seconds=300.0,
            base_exposure=10.0,
            density_adjustment=1.0,
            light_source_adjustment=1.0,
            distance_adjustment=1.0,
            paper_adjustment=1.0,
            chemistry_adjustment=1.0,
            environmental_adjustment=1.0,
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
            distance_inches=4.0,
        )
        assert "5 minutes" in result.format_time()

    def test_format_time_minutes_and_seconds(self):
        """Should format mixed minutes and seconds."""
        result = ExposureResult(
            exposure_minutes=5.5,
            exposure_seconds=330.0,
            base_exposure=10.0,
            density_adjustment=1.0,
            light_source_adjustment=1.0,
            distance_adjustment=1.0,
            paper_adjustment=1.0,
            chemistry_adjustment=1.0,
            environmental_adjustment=1.0,
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
            distance_inches=4.0,
        )
        formatted = result.format_time()
        assert "5 min" in formatted
        assert "30 sec" in formatted

    def test_format_time_hours(self):
        """Should format long times as hours."""
        result = ExposureResult(
            exposure_minutes=90.0,
            exposure_seconds=5400.0,
            base_exposure=10.0,
            density_adjustment=1.0,
            light_source_adjustment=1.0,
            distance_adjustment=1.0,
            paper_adjustment=1.0,
            chemistry_adjustment=1.0,
            environmental_adjustment=1.0,
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
            distance_inches=4.0,
        )
        formatted = result.format_time()
        assert "1 hour" in formatted
        assert "30 min" in formatted

    def test_to_dict(self, sample_result):
        """Should serialize to dictionary."""
        d = sample_result.to_dict()
        assert "exposure_time" in d
        assert "exposure_minutes" in d
        assert "adjustments" in d
        assert "inputs" in d
        assert d["adjustments"]["density"] == pytest.approx(1.2)


class TestExposureCalculator:
    """Tests for ExposureCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create exposure calculator with default settings."""
        return ExposureCalculator()

    @pytest.fixture
    def custom_calculator(self):
        """Create calculator with custom settings."""
        settings = ExposureSettings(
            base_exposure_minutes=12.0,
            base_negative_density=1.5,
            base_distance_inches=6.0,
        )
        return ExposureCalculator(settings=settings)

    def test_calculator_default_settings(self, calculator):
        """Calculator should use default settings."""
        assert calculator.settings.base_exposure_minutes == 10.0

    def test_calculator_custom_settings(self, custom_calculator):
        """Calculator should use custom settings."""
        assert custom_calculator.settings.base_exposure_minutes == 12.0

    def test_calculate_same_density(self, calculator):
        """Same density as base should give base exposure."""
        result = calculator.calculate(negative_density=1.6)  # Same as base
        # With all factors at 1.0, should be close to base
        assert result.exposure_minutes == pytest.approx(10.0, rel=0.01)

    def test_calculate_higher_density(self, calculator):
        """Higher density should increase exposure."""
        result = calculator.calculate(negative_density=1.9)
        assert result.exposure_minutes > calculator.settings.base_exposure_minutes
        assert result.density_adjustment > 1.0

    def test_calculate_lower_density(self, calculator):
        """Lower density should decrease exposure."""
        result = calculator.calculate(negative_density=1.3)
        assert result.exposure_minutes < calculator.settings.base_exposure_minutes
        assert result.density_adjustment < 1.0

    def test_calculate_with_distance(self, calculator):
        """Increased distance should increase exposure (inverse square)."""
        base_result = calculator.calculate(negative_density=1.6)
        far_result = calculator.calculate(negative_density=1.6, distance_inches=8.0)  # Double distance

        # Double distance = 4x exposure (inverse square)
        assert far_result.exposure_minutes == pytest.approx(
            base_result.exposure_minutes * 4, rel=0.01
        )

    def test_calculate_with_led_uv(self, calculator):
        """LED UV should be faster than fluorescent."""
        fluor_result = calculator.calculate(
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
        )
        led_result = calculator.calculate(
            negative_density=1.6,
            light_source=LightSource.LED_UV,
        )

        assert led_result.exposure_minutes < fluor_result.exposure_minutes

    def test_calculate_with_custom_light_source(self):
        """Custom light source should use custom multiplier."""
        settings = ExposureSettings(
            light_source=LightSource.CUSTOM,
            custom_speed_multiplier=0.5,
        )
        calculator = ExposureCalculator(settings=settings)

        result = calculator.calculate(negative_density=1.6)
        # Custom 0.5x multiplier = half the time
        assert result.light_source_adjustment == 0.5

    def test_calculate_with_platinum(self, calculator):
        """Higher platinum ratio should increase exposure."""
        palladium_result = calculator.calculate(
            negative_density=1.6,
            platinum_ratio=0.0,
        )
        platinum_result = calculator.calculate(
            negative_density=1.6,
            platinum_ratio=1.0,
        )

        assert platinum_result.exposure_minutes > palladium_result.exposure_minutes

    def test_calculate_with_humidity(self, calculator):
        """Higher humidity should decrease exposure."""
        dry_result = calculator.calculate(
            negative_density=1.6,
            humidity_factor=0.8,
        )
        humid_result = calculator.calculate(
            negative_density=1.6,
            humidity_factor=1.2,
        )

        assert humid_result.exposure_minutes < dry_result.exposure_minutes

    def test_calculate_long_exposure_warning(self, calculator):
        """Long exposure should generate warning."""
        result = calculator.calculate(negative_density=2.5)  # Very dense
        assert any("Warning" in note and "Long exposure" in note for note in result.notes)

    def test_calculate_short_exposure_warning(self, calculator):
        """Short exposure should generate warning."""
        result = calculator.calculate(
            negative_density=1.0,  # Thin negative
            light_source=LightSource.METAL_HALIDE,  # Fast light
        )
        assert any("Warning" in note and "Short exposure" in note for note in result.notes)

    def test_calculate_test_strip(self, calculator):
        """Should generate test strip times."""
        times = calculator.calculate_test_strip(
            center_exposure=10.0,
            steps=5,
            increment_stops=0.5,
        )

        assert len(times) == 5
        assert times[2] == pytest.approx(10.0)  # Center
        assert times[0] < times[2] < times[4]  # Increasing

    def test_calculate_test_strip_half_stops(self, calculator):
        """Half-stop increments should work correctly."""
        times = calculator.calculate_test_strip(
            center_exposure=8.0,
            steps=3,
            increment_stops=0.5,
        )

        # -0.5 stop, center, +0.5 stop
        assert times[0] == pytest.approx(8.0 / 1.414, rel=0.01)  # -0.5 stop
        assert times[1] == pytest.approx(8.0)  # Center
        assert times[2] == pytest.approx(8.0 * 1.414, rel=0.01)  # +0.5 stop

    def test_density_to_stops(self, calculator):
        """Should convert density to stops correctly."""
        # 0.3 density = 1 stop
        assert calculator.density_to_stops(0.3) == pytest.approx(1.0)
        assert calculator.density_to_stops(0.6) == pytest.approx(2.0)
        assert calculator.density_to_stops(0.15) == pytest.approx(0.5)

    def test_stops_to_density(self, calculator):
        """Should convert stops to density correctly."""
        # 1 stop = 0.3 density
        assert calculator.stops_to_density(1.0) == pytest.approx(0.3)
        assert calculator.stops_to_density(2.0) == pytest.approx(0.6)
        assert calculator.stops_to_density(0.5) == pytest.approx(0.15)

    def test_adjust_for_distance(self, calculator):
        """Should adjust for distance using inverse square law."""
        # Double distance = 4x time
        result = calculator.adjust_for_distance(
            current_exposure=10.0,
            current_distance=4.0,
            new_distance=8.0,
        )
        assert result == pytest.approx(40.0)

        # Half distance = 0.25x time
        result = calculator.adjust_for_distance(
            current_exposure=10.0,
            current_distance=4.0,
            new_distance=2.0,
        )
        assert result == pytest.approx(2.5)

    def test_get_light_sources(self):
        """Should return light source list."""
        sources = ExposureCalculator.get_light_sources()

        assert len(sources) > 0
        assert all(isinstance(s, tuple) for s in sources)
        assert all(len(s) == 2 for s in sources)

        # Check some known sources
        values = [s[0] for s in sources]
        assert "nuarc_26_1k" in values
        assert "led_uv" in values
        assert "sunlight" in values


class TestExposureCalculatorEdgeCases:
    """Tests for edge cases in exposure calculation."""

    def test_very_thin_negative(self):
        """Should handle very thin negatives."""
        calculator = ExposureCalculator()
        result = calculator.calculate(negative_density=0.5)
        assert result.exposure_minutes > 0
        assert result.density_adjustment < 1.0

    def test_very_dense_negative(self):
        """Should handle very dense negatives."""
        calculator = ExposureCalculator()
        result = calculator.calculate(negative_density=3.0)
        assert result.exposure_minutes > 0
        assert result.density_adjustment > 1.0

    def test_zero_distance_protection(self):
        """Calculator should handle edge distance values."""
        calculator = ExposureCalculator()
        # Very close distance
        result = calculator.calculate(negative_density=1.6, distance_inches=1.0)
        assert result.exposure_minutes > 0

    def test_all_light_sources_calculate(self):
        """All light sources should produce valid results."""
        calculator = ExposureCalculator()

        for source in list(LightSource):
            result = calculator.calculate(
                negative_density=1.6,
                light_source=source,
            )
            assert result.exposure_minutes > 0
            assert result.light_source == source

    def test_extreme_platinum_ratio(self):
        """Should handle extreme platinum ratios."""
        calculator = ExposureCalculator()

        # 100% palladium
        result_pd = calculator.calculate(negative_density=1.6, platinum_ratio=0.0)
        assert result_pd.chemistry_adjustment == pytest.approx(1.0)

        # 100% platinum
        result_pt = calculator.calculate(negative_density=1.6, platinum_ratio=1.0)
        assert result_pt.chemistry_adjustment == pytest.approx(2.0)
