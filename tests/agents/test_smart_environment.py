"""Tests for the environmental adaptation system.

Tests cover:
- Environmental conditions handling
- Exposure adaptation calculations
- Seasonal profiles
- Trend analysis
- Paper-specific compensation
"""

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from ptpd_calibration.agents.smart.environment import (
    AdaptationResult,
    EnvironmentAdapter,
    EnvironmentConditions,
    EnvironmentSettings,
    Season,
    SeasonalProfile,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env_settings() -> EnvironmentSettings:
    """Create environment settings for tests."""
    return EnvironmentSettings(
        base_exposure_seconds=180.0,
        reference_temperature=21.0,
        reference_humidity=50.0,
        temperature_coefficient=0.02,
        humidity_coefficient=0.005,
        min_adjustment=0.7,
        max_adjustment=1.5,
    )


@pytest.fixture
def env_adapter(env_settings: EnvironmentSettings) -> EnvironmentAdapter:
    """Create environment adapter for tests."""
    return EnvironmentAdapter(settings=env_settings)


@pytest.fixture
def optimal_conditions() -> EnvironmentConditions:
    """Create optimal environmental conditions."""
    return EnvironmentConditions(
        temperature_celsius=21.0,
        humidity_percent=50.0,
    )


@pytest.fixture
def hot_humid_conditions() -> EnvironmentConditions:
    """Create hot and humid conditions."""
    return EnvironmentConditions(
        temperature_celsius=28.0,
        humidity_percent=70.0,
    )


@pytest.fixture
def cold_dry_conditions() -> EnvironmentConditions:
    """Create cold and dry conditions."""
    return EnvironmentConditions(
        temperature_celsius=15.0,
        humidity_percent=35.0,
    )


# ============================================================================
# Season Tests
# ============================================================================


class TestSeason:
    """Tests for Season enum."""

    def test_seasons_exist(self) -> None:
        """Test all seasons are defined."""
        assert Season.SPRING == "spring"
        assert Season.SUMMER == "summer"
        assert Season.FALL == "fall"
        assert Season.WINTER == "winter"

    def test_season_count(self) -> None:
        """Test correct number of seasons."""
        assert len(Season) == 4


# ============================================================================
# EnvironmentConditions Tests
# ============================================================================


class TestEnvironmentConditions:
    """Tests for EnvironmentConditions dataclass."""

    def test_creation_celsius(self) -> None:
        """Test creation with Celsius temperature."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
        )

        assert conditions.temperature_celsius == 21.0
        assert conditions.humidity_percent == 50.0

    def test_temperature_fahrenheit(self) -> None:
        """Test Fahrenheit conversion."""
        conditions = EnvironmentConditions(
            temperature_celsius=20.0,
            humidity_percent=50.0,
        )

        assert conditions.temperature_fahrenheit == 68.0

    def test_from_fahrenheit(self) -> None:
        """Test creation from Fahrenheit."""
        conditions = EnvironmentConditions.from_fahrenheit(
            temperature_f=68.0,
            humidity_percent=50.0,
        )

        assert abs(conditions.temperature_celsius - 20.0) < 0.01

    def test_optimal_conditions(self, optimal_conditions: EnvironmentConditions) -> None:
        """Test optimal condition detection."""
        is_optimal, warnings = optimal_conditions.is_optimal()

        assert is_optimal is True
        assert len(warnings) == 0

    def test_cold_conditions_warning(self) -> None:
        """Test cold condition warnings."""
        conditions = EnvironmentConditions(
            temperature_celsius=15.0,
            humidity_percent=50.0,
        )

        is_optimal, warnings = conditions.is_optimal()

        assert is_optimal is False
        assert any("low" in w.lower() for w in warnings)

    def test_hot_conditions_warning(self) -> None:
        """Test hot condition warnings."""
        conditions = EnvironmentConditions(
            temperature_celsius=28.0,
            humidity_percent=50.0,
        )

        is_optimal, warnings = conditions.is_optimal()

        assert is_optimal is False
        assert any("high" in w.lower() for w in warnings)

    def test_low_humidity_warning(self) -> None:
        """Test low humidity warnings."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=30.0,
        )

        is_optimal, warnings = conditions.is_optimal()

        assert is_optimal is False
        assert any("humidity" in w.lower() for w in warnings)

    def test_high_humidity_warning(self) -> None:
        """Test high humidity warnings."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=75.0,
        )

        is_optimal, warnings = conditions.is_optimal()

        assert is_optimal is False
        assert any("humidity" in w.lower() for w in warnings)

    def test_with_optional_fields(self) -> None:
        """Test creation with optional fields."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
            uv_index=6.0,
            altitude_meters=1500.0,
            metadata={"location": "Denver"},
        )

        assert conditions.uv_index == 6.0
        assert conditions.altitude_meters == 1500.0
        assert conditions.metadata["location"] == "Denver"


# ============================================================================
# SeasonalProfile Tests
# ============================================================================


class TestSeasonalProfile:
    """Tests for SeasonalProfile dataclass."""

    def test_creation(self) -> None:
        """Test seasonal profile creation."""
        profile = SeasonalProfile(
            season=Season.SUMMER,
            avg_temperature=25.0,
            avg_humidity=55.0,
            exposure_adjustment=0.95,
            coating_adjustment=0.95,
            drying_adjustment=0.9,
            notes=["Test note"],
        )

        assert profile.season == Season.SUMMER
        assert profile.exposure_adjustment == 0.95
        assert len(profile.notes) == 1


# ============================================================================
# EnvironmentSettings Tests
# ============================================================================


class TestEnvironmentSettings:
    """Tests for EnvironmentSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = EnvironmentSettings()

        assert settings.base_exposure_seconds == 180.0
        assert settings.reference_temperature == 21.0
        assert settings.reference_humidity == 50.0
        assert settings.min_adjustment == 0.7
        assert settings.max_adjustment == 1.5

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = EnvironmentSettings(
            base_exposure_seconds=200.0,
            reference_temperature=22.0,
        )

        assert settings.base_exposure_seconds == 200.0
        assert settings.reference_temperature == 22.0


# ============================================================================
# AdaptationResult Tests
# ============================================================================


class TestAdaptationResult:
    """Tests for AdaptationResult dataclass."""

    def test_exposure_change_percent(
        self,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test exposure change percentage calculation."""
        result = AdaptationResult(
            original_exposure=180.0,
            adjusted_exposure=200.0,
            adjustment_factor=200.0 / 180.0,
            conditions=optimal_conditions,
            adjustments_applied={"temperature": 1.0},
            recommendations=[],
            warnings=[],
        )

        change = result.exposure_change_percent
        assert abs(change - 11.11) < 0.1  # About 11% increase

    def test_zero_original_exposure(
        self,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test handling zero original exposure."""
        result = AdaptationResult(
            original_exposure=0.0,
            adjusted_exposure=0.0,
            adjustment_factor=1.0,
            conditions=optimal_conditions,
            adjustments_applied={},
            recommendations=[],
            warnings=[],
        )

        assert result.exposure_change_percent == 0.0


# ============================================================================
# EnvironmentAdapter Tests
# ============================================================================


class TestEnvironmentAdapter:
    """Tests for EnvironmentAdapter."""

    def test_creation(self, env_adapter: EnvironmentAdapter) -> None:
        """Test adapter creation."""
        assert env_adapter.settings is not None
        assert len(env_adapter._seasonal_profiles) == 4

    def test_adapt_exposure_optimal(
        self,
        env_adapter: EnvironmentAdapter,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test exposure adaptation at optimal conditions."""
        result = env_adapter.adapt_exposure(180.0, optimal_conditions)

        # At reference conditions, adjustment should be minimal
        assert 0.95 <= result.adjustment_factor <= 1.05
        assert len(result.warnings) == 0

    def test_adapt_exposure_hot_humid(
        self,
        env_adapter: EnvironmentAdapter,
        hot_humid_conditions: EnvironmentConditions
    ) -> None:
        """Test exposure adaptation in hot humid conditions."""
        result = env_adapter.adapt_exposure(180.0, hot_humid_conditions)

        # Hot conditions should reduce exposure (faster reaction)
        # But high humidity increases it
        assert "temperature" in result.adjustments_applied
        assert "humidity" in result.adjustments_applied
        assert len(result.recommendations) > 0

    def test_adapt_exposure_cold_dry(
        self,
        env_adapter: EnvironmentAdapter,
        cold_dry_conditions: EnvironmentConditions
    ) -> None:
        """Test exposure adaptation in cold dry conditions."""
        result = env_adapter.adapt_exposure(180.0, cold_dry_conditions)

        # Cold conditions should increase exposure
        assert result.adjustment_factor > 1.0
        assert len(result.recommendations) > 0

    def test_adapt_exposure_with_altitude(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test exposure adaptation with high altitude."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
            altitude_meters=2000.0,
        )

        result = env_adapter.adapt_exposure(180.0, conditions)

        # High altitude should reduce exposure (more UV)
        assert "altitude" in result.adjustments_applied
        assert result.adjustments_applied["altitude"] < 1.0

    def test_adapt_exposure_with_uv_index(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test exposure adaptation with UV index."""
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
            uv_index=10.0,  # High UV
        )

        result = env_adapter.adapt_exposure(180.0, conditions)

        assert "uv_index" in result.adjustments_applied

    def test_adaptation_bounds(self, env_adapter: EnvironmentAdapter) -> None:
        """Test adjustment is bounded."""
        # Extreme conditions
        extreme_conditions = EnvironmentConditions(
            temperature_celsius=40.0,
            humidity_percent=95.0,
        )

        result = env_adapter.adapt_exposure(180.0, extreme_conditions)

        # Should be clamped to bounds
        assert result.adjustment_factor >= env_adapter.settings.min_adjustment
        assert result.adjustment_factor <= env_adapter.settings.max_adjustment

    def test_get_seasonal_profile_spring(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test getting spring profile."""
        profile = env_adapter.get_seasonal_profile(month=4)

        assert profile.season == Season.SPRING

    def test_get_seasonal_profile_summer(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test getting summer profile."""
        profile = env_adapter.get_seasonal_profile(month=7)

        assert profile.season == Season.SUMMER

    def test_get_seasonal_profile_fall(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test getting fall profile."""
        profile = env_adapter.get_seasonal_profile(month=10)

        assert profile.season == Season.FALL

    def test_get_seasonal_profile_winter(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test getting winter profile."""
        profile = env_adapter.get_seasonal_profile(month=1)

        assert profile.season == Season.WINTER

    def test_set_custom_seasonal_profile(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test setting custom seasonal profile."""
        custom_profile = SeasonalProfile(
            season=Season.SUMMER,
            avg_temperature=30.0,
            avg_humidity=60.0,
            exposure_adjustment=0.85,
            coating_adjustment=0.9,
            drying_adjustment=0.85,
            notes=["Custom summer profile"],
        )

        env_adapter.set_seasonal_profile(custom_profile)

        profile = env_adapter.get_seasonal_profile(month=7)
        assert profile.exposure_adjustment == 0.85

    def test_get_trend_no_data(self, env_adapter: EnvironmentAdapter) -> None:
        """Test trend analysis with no data."""
        trend = env_adapter.get_trend()

        assert trend["has_data"] is False

    def test_get_trend_with_data(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test trend analysis with data."""
        # Add some conditions
        for i in range(5):
            conditions = EnvironmentConditions(
                temperature_celsius=20.0 + i,
                humidity_percent=50.0,
            )
            env_adapter._conditions_history.append(conditions)

        trend = env_adapter.get_trend()

        assert trend["has_data"] is True
        assert "temperature" in trend
        assert trend["temperature"]["trend"] == "rising"

    def test_recommend_schedule_coating(
        self,
        env_adapter: EnvironmentAdapter,
        hot_humid_conditions: EnvironmentConditions
    ) -> None:
        """Test schedule recommendations for coating."""
        recommendations = env_adapter.recommend_schedule(
            hot_humid_conditions,
            task="coating"
        )

        assert "current_suitable" in recommendations
        assert "advice" in recommendations
        assert "humidity" in recommendations["advice"].lower()

    def test_recommend_schedule_printing(
        self,
        env_adapter: EnvironmentAdapter,
        cold_dry_conditions: EnvironmentConditions
    ) -> None:
        """Test schedule recommendations for printing."""
        recommendations = env_adapter.recommend_schedule(
            cold_dry_conditions,
            task="printing"
        )

        assert "advice" in recommendations
        assert "temperature" in recommendations["advice"].lower()

    def test_recommend_schedule_developing(
        self,
        env_adapter: EnvironmentAdapter,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test schedule recommendations for developing."""
        recommendations = env_adapter.recommend_schedule(
            optimal_conditions,
            task="developing"
        )

        assert "advice" in recommendations
        assert "temperature" in recommendations["advice"].lower()

    def test_get_compensation_for_paper_known(
        self,
        env_adapter: EnvironmentAdapter,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test paper compensation for known paper."""
        compensation = env_adapter.get_compensation_for_paper(
            "platine",
            optimal_conditions
        )

        assert "exposure_compensation" in compensation
        assert "drying_time_factor" in compensation
        assert compensation["paper_type"] == "platine"

    def test_get_compensation_for_paper_unknown(
        self,
        env_adapter: EnvironmentAdapter,
        optimal_conditions: EnvironmentConditions
    ) -> None:
        """Test paper compensation for unknown paper."""
        compensation = env_adapter.get_compensation_for_paper(
            "unknown_paper",
            optimal_conditions
        )

        # Should use default sensitivities
        assert compensation["sensitivity"]["temp"] == 1.0
        assert compensation["sensitivity"]["humidity"] == 1.0

    def test_get_compensation_non_optimal_conditions(
        self,
        env_adapter: EnvironmentAdapter,
        hot_humid_conditions: EnvironmentConditions
    ) -> None:
        """Test paper compensation in non-optimal conditions."""
        compensation = env_adapter.get_compensation_for_paper(
            "bergger_cot320",
            hot_humid_conditions
        )

        # Bergger has higher sensitivity
        assert compensation["sensitivity"]["temp"] == 1.1
        assert compensation["exposure_compensation"] != 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestEnvironmentAdapterIntegration:
    """Integration tests for EnvironmentAdapter."""

    def test_full_adaptation_workflow(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test complete adaptation workflow."""
        # Get current conditions
        conditions = EnvironmentConditions(
            temperature_celsius=23.0,
            humidity_percent=55.0,
            altitude_meters=300.0,
        )

        # Check optimal
        is_optimal, warnings = conditions.is_optimal()

        # Adapt exposure
        base_exposure = 180.0
        result = env_adapter.adapt_exposure(base_exposure, conditions)

        # Get seasonal profile
        season = env_adapter.get_seasonal_profile()

        # Get paper compensation
        compensation = env_adapter.get_compensation_for_paper("platine", conditions)

        # Combined adjustment
        total_adjustment = result.adjustment_factor * compensation["exposure_compensation"]
        final_exposure = base_exposure * total_adjustment

        assert final_exposure > 0
        assert isinstance(result, AdaptationResult)

    def test_history_accumulation(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test conditions history accumulates."""
        for temp in range(18, 25):
            conditions = EnvironmentConditions(
                temperature_celsius=float(temp),
                humidity_percent=50.0,
            )
            env_adapter.adapt_exposure(180.0, conditions)

        assert len(env_adapter._conditions_history) == 7

        trend = env_adapter.get_trend()
        assert trend["has_data"] is True
        assert trend["data_points"] == 7

    def test_conditions_across_seasons(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test adaptation across different seasons."""
        results = {}

        for month, temp, humidity in [
            (1, 10, 40),   # Winter
            (4, 18, 55),   # Spring
            (7, 28, 60),   # Summer
            (10, 15, 50),  # Fall
        ]:
            conditions = EnvironmentConditions(
                temperature_celsius=temp,
                humidity_percent=humidity,
            )
            result = env_adapter.adapt_exposure(180.0, conditions)
            season = env_adapter.get_seasonal_profile(month)
            results[season.season.value] = {
                "adjustment": result.adjustment_factor,
                "exposure": result.adjusted_exposure,
            }

        # Winter should have highest exposure (coldest)
        assert results["winter"]["adjustment"] > results["summer"]["adjustment"]
