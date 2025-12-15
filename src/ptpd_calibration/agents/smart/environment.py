"""Environmental adaptation system for Pt/Pd printing.

This module implements environmental adaptation that:
- Tracks temperature and humidity conditions
- Adjusts exposure times based on environment
- Provides seasonal recommendations
- Compensates for environmental variations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Season(str, Enum):
    """Seasons affecting printing conditions."""

    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


@dataclass
class EnvironmentConditions:
    """Current environmental conditions."""

    temperature_celsius: float
    humidity_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uv_index: Optional[float] = None
    altitude_meters: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def temperature_fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        return self.temperature_celsius * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(
        cls,
        temperature_f: float,
        humidity_percent: float,
        **kwargs: Any
    ) -> EnvironmentConditions:
        """Create from Fahrenheit temperature."""
        temp_c = (temperature_f - 32) * 5 / 9
        return cls(
            temperature_celsius=temp_c,
            humidity_percent=humidity_percent,
            **kwargs
        )

    def is_optimal(self) -> tuple[bool, list[str]]:
        """Check if conditions are optimal for printing.

        Returns:
            Tuple of (is_optimal, list of warnings)
        """
        warnings = []

        # Ideal: 68-72°F (20-22°C), 45-55% humidity
        if self.temperature_celsius < 18:
            warnings.append(
                f"Temperature too low ({self.temperature_celsius:.1f}°C). "
                "Consider warming workspace to 20-22°C."
            )
        elif self.temperature_celsius > 24:
            warnings.append(
                f"Temperature too high ({self.temperature_celsius:.1f}°C). "
                "Chemistry may be less stable."
            )

        if self.humidity_percent < 40:
            warnings.append(
                f"Humidity too low ({self.humidity_percent:.0f}%). "
                "Paper may be too dry, affecting coating absorption."
            )
        elif self.humidity_percent > 60:
            warnings.append(
                f"Humidity too high ({self.humidity_percent:.0f}%). "
                "Extended drying times needed."
            )

        return len(warnings) == 0, warnings


@dataclass
class SeasonalProfile:
    """Seasonal adjustment profile for a location."""

    season: Season
    avg_temperature: float  # Celsius
    avg_humidity: float  # Percent
    exposure_adjustment: float  # Multiplier (1.0 = no change)
    coating_adjustment: float  # Multiplier for coating amount
    drying_adjustment: float  # Multiplier for drying time
    notes: list[str] = field(default_factory=list)


class EnvironmentSettings(BaseSettings):
    """Settings for environmental adaptation."""

    model_config = SettingsConfigDict(
        env_prefix="PTPD_ENVIRONMENT_",
        env_file=".env",
        extra="ignore",
    )

    # Base exposure time reference (seconds)
    base_exposure_seconds: float = Field(
        default=180.0,
        description="Base exposure time at reference conditions"
    )

    # Reference conditions
    reference_temperature: float = Field(
        default=21.0,
        description="Reference temperature in Celsius"
    )

    reference_humidity: float = Field(
        default=50.0,
        description="Reference humidity percentage"
    )

    # Temperature coefficient (% change per degree C)
    temperature_coefficient: float = Field(
        default=0.02,
        description="Exposure adjustment per degree C from reference"
    )

    # Humidity coefficient (% change per % humidity)
    humidity_coefficient: float = Field(
        default=0.005,
        description="Exposure adjustment per percent humidity from reference"
    )

    # UV coefficient for altitude
    altitude_uv_coefficient: float = Field(
        default=0.0001,
        description="UV intensity increase per meter altitude"
    )

    # Minimum/maximum adjustment bounds
    min_adjustment: float = Field(
        default=0.7,
        description="Minimum exposure adjustment multiplier"
    )

    max_adjustment: float = Field(
        default=1.5,
        description="Maximum exposure adjustment multiplier"
    )


@dataclass
class AdaptationResult:
    """Result of environmental adaptation calculation."""

    original_exposure: float
    adjusted_exposure: float
    adjustment_factor: float
    conditions: EnvironmentConditions
    adjustments_applied: dict[str, float]
    recommendations: list[str]
    warnings: list[str]

    @property
    def exposure_change_percent(self) -> float:
        """Get percentage change in exposure."""
        if self.original_exposure == 0:
            return 0.0
        return ((self.adjusted_exposure - self.original_exposure)
                / self.original_exposure * 100)


class EnvironmentAdapter:
    """Main environmental adaptation system."""

    def __init__(self, settings: Optional[EnvironmentSettings] = None):
        """Initialize environment adapter.

        Args:
            settings: Environmental settings
        """
        self.settings = settings or EnvironmentSettings()

        # Historical conditions for trend analysis
        self._conditions_history: list[EnvironmentConditions] = []

        # Seasonal profiles (can be customized)
        self._seasonal_profiles: dict[Season, SeasonalProfile] = {
            Season.SPRING: SeasonalProfile(
                season=Season.SPRING,
                avg_temperature=18.0,
                avg_humidity=55.0,
                exposure_adjustment=1.05,
                coating_adjustment=1.0,
                drying_adjustment=1.1,
                notes=["Higher humidity may require longer drying"]
            ),
            Season.SUMMER: SeasonalProfile(
                season=Season.SUMMER,
                avg_temperature=25.0,
                avg_humidity=50.0,
                exposure_adjustment=0.95,
                coating_adjustment=0.95,
                drying_adjustment=0.9,
                notes=["Higher temps increase reactivity", "UV typically stronger"]
            ),
            Season.FALL: SeasonalProfile(
                season=Season.FALL,
                avg_temperature=15.0,
                avg_humidity=50.0,
                exposure_adjustment=1.1,
                coating_adjustment=1.0,
                drying_adjustment=1.0,
                notes=["Cooler temps may slow reactions"]
            ),
            Season.WINTER: SeasonalProfile(
                season=Season.WINTER,
                avg_temperature=10.0,
                avg_humidity=40.0,
                exposure_adjustment=1.15,
                coating_adjustment=1.05,
                drying_adjustment=0.95,
                notes=["Low humidity speeds drying", "Keep chemistry warm"]
            ),
        }

    def adapt_exposure(
        self,
        base_exposure: float,
        conditions: EnvironmentConditions
    ) -> AdaptationResult:
        """Adapt exposure time based on environmental conditions.

        Args:
            base_exposure: Base exposure time in seconds
            conditions: Current environmental conditions

        Returns:
            Adaptation result with adjusted exposure
        """
        adjustments = {}
        recommendations = []
        warnings = []

        # Temperature adjustment
        temp_diff = conditions.temperature_celsius - self.settings.reference_temperature
        temp_adjustment = 1.0 - (temp_diff * self.settings.temperature_coefficient)
        adjustments["temperature"] = temp_adjustment

        if temp_diff > 5:
            recommendations.append(
                f"High temperature ({conditions.temperature_celsius:.1f}°C): "
                "Consider cooling workspace for more consistent results."
            )
        elif temp_diff < -5:
            recommendations.append(
                f"Low temperature ({conditions.temperature_celsius:.1f}°C): "
                "Warm chemistry to reference temperature before coating."
            )

        # Humidity adjustment
        humidity_diff = conditions.humidity_percent - self.settings.reference_humidity
        humidity_adjustment = 1.0 + (humidity_diff * self.settings.humidity_coefficient)
        adjustments["humidity"] = humidity_adjustment

        if humidity_diff > 15:
            recommendations.append(
                f"High humidity ({conditions.humidity_percent:.0f}%): "
                "Allow extra drying time before exposure."
            )
            warnings.append("Coating may take longer to dry at high humidity.")
        elif humidity_diff < -15:
            recommendations.append(
                f"Low humidity ({conditions.humidity_percent:.0f}%): "
                "Paper may absorb coating faster."
            )

        # Altitude/UV adjustment
        altitude_adjustment = 1.0
        if conditions.altitude_meters is not None and conditions.altitude_meters > 500:
            # Higher altitude = more UV = less exposure needed
            altitude_factor = conditions.altitude_meters * self.settings.altitude_uv_coefficient
            altitude_adjustment = 1.0 - altitude_factor
            adjustments["altitude"] = altitude_adjustment

            if conditions.altitude_meters > 1500:
                recommendations.append(
                    f"High altitude ({conditions.altitude_meters:.0f}m): "
                    "UV intensity is significantly higher."
                )

        # UV index adjustment if available
        if conditions.uv_index is not None:
            # Reference UV index is around 5-6
            uv_reference = 5.5
            uv_factor = uv_reference / max(conditions.uv_index, 0.1)
            adjustments["uv_index"] = uv_factor

            if conditions.uv_index > 8:
                recommendations.append(
                    f"High UV index ({conditions.uv_index:.1f}): "
                    "Exposure will be faster."
                )

        # Calculate total adjustment
        total_adjustment = 1.0
        for adj in adjustments.values():
            total_adjustment *= adj

        # Clamp to bounds
        total_adjustment = max(
            self.settings.min_adjustment,
            min(self.settings.max_adjustment, total_adjustment)
        )

        adjusted_exposure = base_exposure * total_adjustment

        # Check optimal conditions
        is_optimal, condition_warnings = conditions.is_optimal()
        if not is_optimal:
            warnings.extend(condition_warnings)

        # Record conditions for history
        self._conditions_history.append(conditions)
        if len(self._conditions_history) > 100:
            self._conditions_history = self._conditions_history[-100:]

        return AdaptationResult(
            original_exposure=base_exposure,
            adjusted_exposure=adjusted_exposure,
            adjustment_factor=total_adjustment,
            conditions=conditions,
            adjustments_applied=adjustments,
            recommendations=recommendations,
            warnings=warnings,
        )

    def get_seasonal_profile(
        self,
        month: Optional[int] = None
    ) -> SeasonalProfile:
        """Get seasonal profile for current or specified month.

        Args:
            month: Month number (1-12), or None for current month

        Returns:
            Seasonal profile
        """
        if month is None:
            month = datetime.now().month

        # Map months to seasons (Northern hemisphere)
        if month in (3, 4, 5):
            return self._seasonal_profiles[Season.SPRING]
        elif month in (6, 7, 8):
            return self._seasonal_profiles[Season.SUMMER]
        elif month in (9, 10, 11):
            return self._seasonal_profiles[Season.FALL]
        else:
            return self._seasonal_profiles[Season.WINTER]

    def set_seasonal_profile(self, profile: SeasonalProfile) -> None:
        """Set a custom seasonal profile.

        Args:
            profile: Custom seasonal profile
        """
        self._seasonal_profiles[profile.season] = profile

    def get_trend(self, hours: int = 24) -> dict[str, Any]:
        """Get environmental trend over recent hours.

        Args:
            hours: Number of hours to analyze

        Returns:
            Trend information
        """
        if not self._conditions_history:
            return {
                "has_data": False,
                "message": "No environmental data recorded yet",
            }

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (hours * 3600)

        recent = [
            c for c in self._conditions_history
            if c.timestamp.timestamp() > cutoff
        ]

        if len(recent) < 2:
            return {
                "has_data": False,
                "message": "Insufficient data for trend analysis",
            }

        # Calculate trends
        temps = [c.temperature_celsius for c in recent]
        humidities = [c.humidity_percent for c in recent]

        temp_trend = (temps[-1] - temps[0]) / len(temps)
        humidity_trend = (humidities[-1] - humidities[0]) / len(humidities)

        return {
            "has_data": True,
            "data_points": len(recent),
            "temperature": {
                "current": temps[-1],
                "min": min(temps),
                "max": max(temps),
                "trend": "rising" if temp_trend > 0.1 else "falling" if temp_trend < -0.1 else "stable",
                "change_rate": temp_trend,
            },
            "humidity": {
                "current": humidities[-1],
                "min": min(humidities),
                "max": max(humidities),
                "trend": "rising" if humidity_trend > 0.5 else "falling" if humidity_trend < -0.5 else "stable",
                "change_rate": humidity_trend,
            },
        }

    def recommend_schedule(
        self,
        conditions: EnvironmentConditions,
        task: str = "printing"
    ) -> dict[str, Any]:
        """Recommend optimal schedule based on conditions.

        Args:
            conditions: Current conditions
            task: Type of task (printing, coating, developing)

        Returns:
            Scheduling recommendations
        """
        is_optimal, warnings = conditions.is_optimal()
        season = self.get_seasonal_profile()

        recommendations = {
            "current_suitable": is_optimal,
            "warnings": warnings,
            "season": season.season.value,
            "seasonal_notes": season.notes,
        }

        if task == "coating":
            if conditions.humidity_percent > 60:
                recommendations["advice"] = (
                    "High humidity may extend drying time. "
                    "Consider coating in the morning when humidity is typically lower."
                )
            elif conditions.humidity_percent < 40:
                recommendations["advice"] = (
                    "Low humidity conditions are good for quick drying. "
                    "Proceed with coating."
                )
            else:
                recommendations["advice"] = "Humidity is in optimal range for coating."

        elif task == "printing":
            if conditions.temperature_celsius < 18:
                recommendations["advice"] = (
                    "Cold conditions may slow exposure. "
                    "Allow paper and chemistry to reach room temperature."
                )
            elif conditions.temperature_celsius > 26:
                recommendations["advice"] = (
                    "Warm conditions increase reactivity. "
                    "Consider printing earlier in the day when it's cooler."
                )
            else:
                recommendations["advice"] = "Temperature is suitable for printing."

        elif task == "developing":
            recommendations["advice"] = (
                f"Develop at consistent temperature. "
                f"Current: {conditions.temperature_celsius:.1f}°C. "
                f"Adjust developer dilution if needed."
            )

        return recommendations

    def get_compensation_for_paper(
        self,
        paper_type: str,
        conditions: EnvironmentConditions
    ) -> dict[str, Any]:
        """Get environmental compensation factors for specific paper.

        Different papers react differently to environmental changes.

        Args:
            paper_type: Type of paper
            conditions: Current conditions

        Returns:
            Compensation factors
        """
        # Paper-specific sensitivity factors
        paper_sensitivities = {
            "platine": {"temp": 1.0, "humidity": 1.0},
            "arches_platine": {"temp": 1.0, "humidity": 0.9},
            "bergger_cot320": {"temp": 1.1, "humidity": 1.1},
            "hahnemuhle_platinum": {"temp": 0.9, "humidity": 1.0},
            "revere_platinum": {"temp": 1.0, "humidity": 1.05},
        }

        # Normalize paper type
        paper_key = paper_type.lower().replace(" ", "_").replace("-", "_")

        # Get sensitivity or default
        sensitivity = paper_sensitivities.get(paper_key, {"temp": 1.0, "humidity": 1.0})

        # Calculate compensations
        temp_diff = conditions.temperature_celsius - self.settings.reference_temperature
        humidity_diff = conditions.humidity_percent - self.settings.reference_humidity

        exposure_comp = 1.0 - (temp_diff * self.settings.temperature_coefficient * sensitivity["temp"])
        exposure_comp *= 1.0 + (humidity_diff * self.settings.humidity_coefficient * sensitivity["humidity"])

        drying_comp = 1.0 + (humidity_diff * 0.01 * sensitivity["humidity"])

        return {
            "paper_type": paper_type,
            "exposure_compensation": exposure_comp,
            "drying_time_factor": drying_comp,
            "sensitivity": sensitivity,
            "notes": [
                f"Paper type '{paper_type}' has temperature sensitivity of {sensitivity['temp']:.1f}",
                f"Paper type '{paper_type}' has humidity sensitivity of {sensitivity['humidity']:.1f}",
            ],
        }
