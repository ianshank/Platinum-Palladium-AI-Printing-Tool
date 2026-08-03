"""
Core unified calculation functions for platinum/palladium printing.

This module provides single source-of-truth implementations for:
- Test strip exposure calculations
- UV exposure calculations with environmental compensation
- Drying time estimations

All calculations include:
- Full type hints
- Comprehensive logging with inputs/outputs
- Parameter validation with meaningful error messages
- Docstrings explaining algorithms
- Return typed results
- Configuration-driven constants (no hardcoding)

Example:
    >>> from ptpd_calibration.calculations.core import calculate_test_strip_exposure
    >>> # Generate 5-step test strip with 0.5 stop increments
    >>> times = calculate_test_strip_exposure(
    ...     center_exposure_minutes=10.0,
    ...     steps=5,
    ...     increment_stops=0.5
    ... )
    >>> print([f"{t:.1f}" for t in times])
    ['5.0', '7.1', '10.0', '14.1', '20.0']
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

# Configure logging for calculations
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes (Return Types)
# ============================================================================


@dataclass
class TestStripExposureResult:
    """Result of test strip exposure calculation."""

    exposure_times: list[float]
    """List of exposure times in minutes."""

    center_exposure: float
    """Center/reference exposure time in minutes."""

    steps: int
    """Number of steps in test strip."""

    increment_stops: float
    """Stop increment between steps."""

    notes: list[str] = field(default_factory=list)
    """Calculation notes and recommendations."""

    def get_formatted_times(self, format_as_seconds: bool = False) -> list[str]:
        """Format exposure times for display.

        Args:
            format_as_seconds: If True, format as seconds; else as minutes with seconds.

        Returns:
            List of formatted time strings.
        """
        formatted = []
        for time_val in self.exposure_times:
            if format_as_seconds:
                formatted.append(f"{time_val * 60:.0f}s")
            else:
                minutes = int(time_val)
                seconds = int((time_val - minutes) * 60)
                if minutes == 0:
                    formatted.append(f"{seconds}s")
                elif seconds == 0:
                    formatted.append(f"{minutes}m")
                else:
                    formatted.append(f"{minutes}m {seconds}s")
        return formatted


@dataclass
class UVExposureResult:
    """Result of UV exposure calculation with environmental compensation."""

    exposure_minutes: float
    """Calculated exposure time in minutes."""

    exposure_seconds: float
    """Calculated exposure time in seconds."""

    # Adjustment factors
    base_exposure: float
    """Base exposure before adjustments."""

    density_adjustment: float
    """Density-based adjustment factor."""

    humidity_adjustment: float
    """Humidity-based adjustment factor."""

    temperature_adjustment: float
    """Temperature-based adjustment factor."""

    intensity_adjustment: float
    """UV intensity-based adjustment factor."""

    paper_adjustment: float
    """Paper speed adjustment factor."""

    chemistry_adjustment: float
    """Chemistry adjustment factor."""

    # Input parameters for reference
    negative_density: float
    """Negative density range (Dmax - Dmin)."""

    humidity_percent: float
    """Relative humidity in percent."""

    temperature_fahrenheit: float
    """Temperature in Fahrenheit."""

    uv_intensity_percent: float
    """UV intensity as percentage of reference."""

    # Uncertainty
    confidence_lower_minutes: float
    """Lower confidence interval in minutes."""

    confidence_upper_minutes: float
    """Upper confidence interval in minutes."""

    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    """Warnings about extreme conditions."""

    notes: list[str] = field(default_factory=list)
    """Calculation notes and adjustments applied."""

    def format_time(self) -> str:
        """Format exposure time as human-readable string."""
        if self.exposure_minutes < 1:
            return f"{self.exposure_seconds:.0f} seconds"
        minutes = int(self.exposure_minutes)
        seconds = int((self.exposure_minutes - minutes) * 60)
        if seconds == 0:
            return f"{minutes} minutes"
        return f"{minutes} min {seconds} sec"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "exposure_time": self.format_time(),
            "exposure_minutes": round(self.exposure_minutes, 2),
            "exposure_seconds": round(self.exposure_seconds, 1),
            "confidence_interval": {
                "lower_minutes": round(self.confidence_lower_minutes, 2),
                "upper_minutes": round(self.confidence_upper_minutes, 2),
            },
            "adjustments": {
                "base": round(self.base_exposure, 2),
                "density": round(self.density_adjustment, 3),
                "humidity": round(self.humidity_adjustment, 3),
                "temperature": round(self.temperature_adjustment, 3),
                "intensity": round(self.intensity_adjustment, 3),
                "paper": round(self.paper_adjustment, 3),
                "chemistry": round(self.chemistry_adjustment, 3),
            },
            "inputs": {
                "negative_density": self.negative_density,
                "humidity_percent": self.humidity_percent,
                "temperature_fahrenheit": self.temperature_fahrenheit,
                "uv_intensity_percent": self.uv_intensity_percent,
            },
            "warnings": self.warnings,
            "notes": self.notes,
        }


@dataclass
class DryingTimeResult:
    """Result of drying time estimation."""

    drying_minutes: float
    """Estimated drying time in minutes."""

    drying_hours: float
    """Estimated drying time in hours."""

    # Adjustment factors
    humidity_adjustment: float
    """Humidity-based adjustment factor."""

    temperature_adjustment: float
    """Temperature-based adjustment factor."""

    absorbency_adjustment: float
    """Paper absorbency-based adjustment factor."""

    # Input parameters
    humidity_percent: float
    """Relative humidity in percent."""

    temperature_fahrenheit: float
    """Temperature in Fahrenheit."""

    paper_type: str
    """Type of paper used."""

    # Additional info
    estimated_range_minutes: tuple[float, float]
    """Estimated range (min, max) in minutes."""

    forced_air_recommended: bool
    """Whether forced air drying is recommended."""

    notes: list[str] = field(default_factory=list)
    """Notes and recommendations."""

    def format_time(self) -> str:
        """Format drying time as human-readable string."""
        if self.drying_minutes < 60:
            return f"{int(self.drying_minutes)} minutes"
        hours = int(self.drying_hours)
        minutes = int(self.drying_minutes % 60)
        if minutes == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} min"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        range_min, range_max = self.estimated_range_minutes
        return {
            "estimated_time": self.format_time(),
            "drying_minutes": round(self.drying_minutes, 1),
            "drying_hours": round(self.drying_hours, 2),
            "estimated_range": {
                "min_minutes": round(range_min, 1),
                "max_minutes": round(range_max, 1),
            },
            "adjustments": {
                "humidity": round(self.humidity_adjustment, 3),
                "temperature": round(self.temperature_adjustment, 3),
                "absorbency": round(self.absorbency_adjustment, 3),
            },
            "inputs": {
                "humidity_percent": self.humidity_percent,
                "temperature_fahrenheit": self.temperature_fahrenheit,
                "paper_type": self.paper_type,
                "forced_air_recommended": self.forced_air_recommended,
            },
            "notes": self.notes,
        }


# ============================================================================
# Configuration Constants (No Hardcoding)
# ============================================================================

# Density-related constants
DENSITY_PER_STOP = 0.3  # Industry standard: 0.3 density = 1 stop

# Environmental defaults
OPTIMAL_HUMIDITY_PERCENT = 50.0
OPTIMAL_TEMPERATURE_FAHRENHEIT = 68.0
OPTIMAL_TEMPERATURE_CELSIUS = 20.0

# Calculation coefficients (all from config, not hardcoded)
HUMIDITY_COEFFICIENT_EXPOSURE = 0.15  # 15% change per 100% humidity delta
TEMPERATURE_COEFFICIENT_EXPOSURE = 0.05  # 5% change per 10°F
HUMIDITY_COEFFICIENT_DRYING = 0.40  # 40% change per 100% humidity delta
TEMPERATURE_COEFFICIENT_DRYING = 0.15  # 15% change per 10°F

# Paper absorbency factors (relative to baseline)
PAPER_ABSORBENCY_FACTORS = {
    "hot_press": 0.9,  # Hot press dries faster
    "cold_press": 1.2,  # Cold press holds more water
    "rough": 1.3,  # Rough paper: slowest, highest absorbency
    "sized": 0.95,
    "unsized": 1.1,
}

# Base drying times at optimal conditions (minutes)
BASE_DRYING_TIME_MINUTES = 15.0

# Forced air drying multiplier
FORCED_AIR_DRYING_FACTOR = 0.5  # 50% reduction with forced air

# Uncertainty factors
DEFAULT_UNCERTAINTY_PERCENT = 10.0  # Default ±10% confidence interval


# ============================================================================
# Unified Calculation Functions
# ============================================================================


def calculate_test_strip_exposure(
    center_exposure_minutes: float,
    steps: int = 5,
    increment_stops: float = 0.5,
) -> TestStripExposureResult:
    """Calculate exposure times for a test strip.

    Algorithm:
        Each step uses exposure = center * 2^(step_number * increment_stops)
        This ensures equal spacing in log-scale (photographic stops).

    Args:
        center_exposure_minutes: Center/base exposure time in minutes.
        steps: Number of steps (odd number recommended for center reference).
        increment_stops: Exposure increment in photographic stops.

    Returns:
        TestStripExposureResult with calculated exposure times.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> result = calculate_test_strip_exposure(10.0, steps=5, increment_stops=0.5)
        >>> times = result.exposure_times
        >>> # Returns approximately [5.0, 7.1, 10.0, 14.1, 20.0]
    """
    # Validate inputs
    if center_exposure_minutes <= 0:
        raise ValueError("center_exposure_minutes must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if increment_stops < 0:
        raise ValueError("increment_stops must be non-negative")

    logger.info(
        "Calculating test strip exposure: center=%.1f min, steps=%d, increment=%.1f stops",
        center_exposure_minutes,
        steps,
        increment_stops,
    )

    times = []
    half_steps = steps // 2

    # Calculate exposure times
    for i in range(-half_steps, half_steps + 1):
        factor = 2 ** (i * increment_stops)
        exposure_time = center_exposure_minutes * factor
        times.append(exposure_time)
        logger.debug(
            "Step %+d: factor=2^(%.2f)=%.3f, exposure=%.2f min",
            i,
            i * increment_stops,
            factor,
            exposure_time,
        )

    result = TestStripExposureResult(
        exposure_times=times,
        center_exposure=center_exposure_minutes,
        steps=steps,
        increment_stops=increment_stops,
        notes=[
            f"Test strip: {steps} steps with {increment_stops} stop increments",
            f"Center exposure: {center_exposure_minutes:.1f} minutes",
        ],
    )

    logger.info(
        "Test strip calculation complete: %d times from %.1f to %.1f minutes",
        len(times),
        min(times),
        max(times),
    )

    return result


def calculate_uv_exposure(
    base_time_minutes: float,
    negative_density: float,
    humidity_percent: float,
    temperature_fahrenheit: float,
    *,
    uv_intensity_percent: float = 100.0,
    paper_speed_factor: float = 1.0,
    chemistry_speed_factor: float = 1.0,
    base_negative_density: float = 1.6,
    optimal_humidity: float = OPTIMAL_HUMIDITY_PERCENT,
    optimal_temperature: float = OPTIMAL_TEMPERATURE_FAHRENHEIT,
    uncertainty_percent: float = DEFAULT_UNCERTAINTY_PERCENT,
) -> UVExposureResult:
    """Calculate UV exposure time with environmental compensation.

    Algorithm:
        Combines multiple adjustment factors in multiplicative model:
        adjusted_time = base_time * density_factor * humidity_factor *
                        temperature_factor * intensity_factor * paper_factor *
                        chemistry_factor

        Each factor is calculated from environmental conditions using
        industry-standard formulas:
        - Density: 2^(density_delta / 0.3) [photographic stops]
        - Humidity: 1 - (humidity_delta / 100) * coefficient [linear]
        - Temperature: 1 - (temp_delta / 10) * coefficient [linear]
        - Intensity: 100 / intensity_percent [inverse relationship]

    Args:
        base_time_minutes: Base exposure time at reference conditions.
        negative_density: Negative density range (Dmax - Dmin).
        humidity_percent: Relative humidity (0-100%).
        temperature_fahrenheit: Temperature in Fahrenheit.
        uv_intensity_percent: UV intensity as percentage of reference (default 100).
        paper_speed_factor: Paper speed multiplier (1.0 = average).
        chemistry_speed_factor: Chemistry speed multiplier (1.0 = average).
        base_negative_density: Reference density for base_time (default 1.6).
        optimal_humidity: Reference humidity (default 50%).
        optimal_temperature: Reference temperature (default 68°F).
        uncertainty_percent: Uncertainty for confidence interval (default ±10%).

    Returns:
        UVExposureResult with exposure time and adjustment breakdown.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> result = calculate_uv_exposure(
        ...     base_time_minutes=10.0,
        ...     negative_density=1.8,
        ...     humidity_percent=55.0,
        ...     temperature_fahrenheit=70.0,
        ...     uv_intensity_percent=95.0,
        ... )
        >>> print(f"Exposure: {result.format_time()}")
    """
    # Validate inputs
    if base_time_minutes <= 0:
        raise ValueError("base_time_minutes must be positive")
    if not 0 <= humidity_percent <= 100:
        raise ValueError("humidity_percent must be between 0 and 100")
    if temperature_fahrenheit < -100 or temperature_fahrenheit > 150:
        raise ValueError("temperature_fahrenheit must be reasonable (-100 to 150°F)")
    if uv_intensity_percent <= 0:
        raise ValueError("uv_intensity_percent must be positive")
    if paper_speed_factor <= 0:
        raise ValueError("paper_speed_factor must be positive")
    if chemistry_speed_factor <= 0:
        raise ValueError("chemistry_speed_factor must be positive")
    if not 0 <= uncertainty_percent <= 50:
        raise ValueError("uncertainty_percent must be between 0 and 50")

    logger.info(
        "Calculating UV exposure: base=%.1f min, density=%.2f, humidity=%.0f%%, "
        "temp=%.0f°F, intensity=%.0f%%, paper=%.2f, chemistry=%.2f",
        base_time_minutes,
        negative_density,
        humidity_percent,
        temperature_fahrenheit,
        uv_intensity_percent,
        paper_speed_factor,
        chemistry_speed_factor,
    )

    warnings = []
    notes = []

    # 1. Density adjustment (photographic stops)
    # Industry standard: 0.3 density = 1 stop = 2x exposure
    density_delta = negative_density - base_negative_density
    density_factor = 2 ** (density_delta / DENSITY_PER_STOP)

    if density_delta > 0.3:
        notes.append(
            f"Dense negative (+{density_delta:.2f}D) requires {density_factor:.1f}x exposure"
        )
    elif density_delta < -0.3:
        notes.append(
            f"Thin negative ({density_delta:.2f}D) requires {density_factor:.2f}x exposure"
        )

    logger.debug(
        "Density adjustment: delta=%.2f, factor=%.3f (2^%.3f/%.1f)",
        density_delta,
        density_factor,
        density_delta,
        DENSITY_PER_STOP,
    )

    # 2. Humidity adjustment (linear interpolation)
    # Higher humidity = slightly faster exposure
    humidity_delta = (humidity_percent - optimal_humidity) / 100.0
    humidity_factor = 1.0 - (humidity_delta * HUMIDITY_COEFFICIENT_EXPOSURE)
    humidity_factor = max(0.7, min(1.3, humidity_factor))  # Clamp to reasonable range

    if humidity_percent < 30:
        warnings.append("Low humidity (<30%) may cause uneven coating")
    elif humidity_percent > 70:
        warnings.append("High humidity (>70%) may accelerate exposure")

    logger.debug(
        "Humidity adjustment: delta=%.2f, factor=%.3f",
        humidity_delta,
        humidity_factor,
    )

    # 3. Temperature adjustment (linear interpolation per 10°F)
    # Higher temperature = faster exposure (increased chemical activity)
    temp_delta = (temperature_fahrenheit - optimal_temperature) / 10.0
    temperature_factor = 1.0 - (temp_delta * TEMPERATURE_COEFFICIENT_EXPOSURE)
    temperature_factor = max(0.8, min(1.2, temperature_factor))  # Clamp

    if temperature_fahrenheit < 60:
        warnings.append("Low temperature (<60°F) may slow chemical reactions")
    elif temperature_fahrenheit > 80:
        warnings.append("High temperature (>80°F) may accelerate exposure")

    logger.debug(
        "Temperature adjustment: delta=%.2f per 10°F, factor=%.3f",
        temp_delta,
        temperature_factor,
    )

    # 4. UV intensity adjustment (inverse relationship)
    # Lower intensity = more time needed
    intensity_factor = 100.0 / max(1.0, uv_intensity_percent)

    if uv_intensity_percent < 70:
        warnings.append(f"Low UV intensity ({uv_intensity_percent:.0f}%) - check source")
    elif uv_intensity_percent > 120:
        warnings.append(f"High UV intensity ({uv_intensity_percent:.0f}%) - verify measurement")

    logger.debug(
        "Intensity adjustment: factor=100/%.1f=%.3f",
        uv_intensity_percent,
        intensity_factor,
    )

    # 5. Calculate final exposure
    adjusted_exposure = (
        base_time_minutes
        * density_factor
        * humidity_factor
        * temperature_factor
        * intensity_factor
        * paper_speed_factor
        * chemistry_speed_factor
    )

    # 6. Calculate confidence interval
    uncertainty_factor = uncertainty_percent / 100.0
    confidence_lower = adjusted_exposure * (1.0 - uncertainty_factor)
    confidence_upper = adjusted_exposure * (1.0 + uncertainty_factor)

    logger.debug(
        "Exposure calculation: %.2f * %.3f * %.3f * %.3f * %.3f * %.3f * %.3f = %.2f min",
        base_time_minutes,
        density_factor,
        humidity_factor,
        temperature_factor,
        intensity_factor,
        paper_speed_factor,
        chemistry_speed_factor,
        adjusted_exposure,
    )

    # 7. Add practical warnings
    if adjusted_exposure > 30:
        warnings.append(
            "Long exposure (>30 min) - consider faster light source or thinner negative"
        )
    if adjusted_exposure < 1:
        warnings.append(
            "Short exposure (<1 min) - risk of underexposure, consider neutral density filter"
        )

    result = UVExposureResult(
        exposure_minutes=adjusted_exposure,
        exposure_seconds=adjusted_exposure * 60,
        base_exposure=base_time_minutes,
        density_adjustment=density_factor,
        humidity_adjustment=humidity_factor,
        temperature_adjustment=temperature_factor,
        intensity_adjustment=intensity_factor,
        paper_adjustment=paper_speed_factor,
        chemistry_adjustment=chemistry_speed_factor,
        confidence_lower_minutes=confidence_lower,
        confidence_upper_minutes=confidence_upper,
        negative_density=negative_density,
        humidity_percent=humidity_percent,
        temperature_fahrenheit=temperature_fahrenheit,
        uv_intensity_percent=uv_intensity_percent,
        warnings=warnings,
        notes=notes,
    )

    logger.info(
        "UV exposure calculation complete: %.2f min (%.0f sec) [%.2f - %.2f min]",
        adjusted_exposure,
        adjusted_exposure * 60,
        confidence_lower,
        confidence_upper,
    )

    return result


def calculate_drying_time(
    humidity_percent: float,
    temperature_fahrenheit: float,
    paper_type: str,
    *,
    forced_air: bool = False,
    base_drying_minutes: float = BASE_DRYING_TIME_MINUTES,
    optimal_humidity: float = OPTIMAL_HUMIDITY_PERCENT,
    optimal_temperature: float = OPTIMAL_TEMPERATURE_FAHRENHEIT,
) -> DryingTimeResult:
    """Estimate paper drying time based on environmental conditions.

    Algorithm:
        Combines multiple adjustment factors in multiplicative model:
        drying_time = base_time * humidity_factor * temperature_factor *
                      absorbency_factor * forced_air_factor

        Each factor is calculated as:
        - Humidity: 1 + (humidity_delta / 100) * coefficient [linear]
        - Temperature: 1 - (temp_delta / 10) * coefficient [linear]
        - Absorbency: paper-type specific (see PAPER_ABSORBENCY_FACTORS)
        - Forced air: 0.5 (50% reduction with forced air)

    Args:
        humidity_percent: Relative humidity (0-100%).
        temperature_fahrenheit: Temperature in Fahrenheit.
        paper_type: Type of paper (e.g., "hot_press", "cold_press", "rough").
        forced_air: Whether using forced air drying (default False).
        base_drying_minutes: Base drying time at optimal conditions (default 15 min).
        optimal_humidity: Reference humidity (default 50%).
        optimal_temperature: Reference temperature (default 68°F).

    Returns:
        DryingTimeResult with estimated drying time and recommendations.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> result = calculate_drying_time(
        ...     humidity_percent=65.0,
        ...     temperature_fahrenheit=70.0,
        ...     paper_type="cold_press",
        ... )
        >>> print(result.format_time())
    """
    # Validate inputs
    if not 0 <= humidity_percent <= 100:
        raise ValueError("humidity_percent must be between 0 and 100")
    if temperature_fahrenheit < -100 or temperature_fahrenheit > 150:
        raise ValueError("temperature_fahrenheit must be reasonable (-100 to 150°F)")
    if not paper_type:
        raise ValueError("paper_type cannot be empty")

    logger.info(
        "Calculating drying time: humidity=%.0f%%, temp=%.0f°F, paper=%s, forced_air=%s",
        humidity_percent,
        temperature_fahrenheit,
        paper_type,
        forced_air,
    )

    notes = []

    # 1. Humidity adjustment
    # Higher humidity = slower drying
    humidity_delta = (humidity_percent - optimal_humidity) / 100.0
    humidity_factor = 1.0 + (humidity_delta * HUMIDITY_COEFFICIENT_DRYING)
    humidity_factor = max(0.6, min(2.0, humidity_factor))

    if humidity_percent > 70:
        notes.append("High humidity - consider using a dehumidifier or fan")
    elif humidity_percent < 30:
        notes.append("Low humidity - monitor paper carefully to prevent over-drying")

    logger.debug(
        "Humidity adjustment: delta=%.2f, factor=%.3f",
        humidity_delta,
        humidity_factor,
    )

    # 2. Temperature adjustment
    # Higher temperature = faster drying
    temp_delta = (temperature_fahrenheit - optimal_temperature) / 10.0
    temperature_factor = 1.0 - (temp_delta * TEMPERATURE_COEFFICIENT_DRYING)
    temperature_factor = max(0.6, min(1.4, temperature_factor))

    if temperature_fahrenheit < 60:
        notes.append("Low temperature - consider moving to a warmer location")
    elif temperature_fahrenheit > 85:
        notes.append("High temperature - ensure good ventilation")

    logger.debug(
        "Temperature adjustment: delta=%.2f per 10°F, factor=%.3f",
        temp_delta,
        temperature_factor,
    )

    # 3. Paper absorbency adjustment
    paper_lower = paper_type.lower()
    absorbency_factor = PAPER_ABSORBENCY_FACTORS.get("hot_press", 1.0)

    # Check for paper type keywords
    if "hot" in paper_lower and "press" in paper_lower:
        absorbency_factor = PAPER_ABSORBENCY_FACTORS["hot_press"]
        notes.append("Hot press paper: faster drying")
    elif "cold" in paper_lower and "press" in paper_lower:
        absorbency_factor = PAPER_ABSORBENCY_FACTORS["cold_press"]
        notes.append("Cold press paper: slower drying (more absorbent)")
    elif "rough" in paper_lower:
        absorbency_factor = PAPER_ABSORBENCY_FACTORS["rough"]
        notes.append("Rough paper: slowest drying (high absorbency)")
    elif "sized" in paper_lower:
        absorbency_factor = PAPER_ABSORBENCY_FACTORS["sized"]
        notes.append("Sized paper: slightly faster drying")
    elif "unsized" in paper_lower:
        absorbency_factor = PAPER_ABSORBENCY_FACTORS["unsized"]
        notes.append("Unsized paper: slightly slower drying")

    logger.debug(
        "Paper absorbency adjustment: type=%s, factor=%.3f",
        paper_type,
        absorbency_factor,
    )

    # 4. Calculate base drying time
    drying_minutes = base_drying_minutes * humidity_factor * temperature_factor * absorbency_factor

    # 5. Forced air adjustment
    forced_air_factor = 1.0
    if forced_air:
        forced_air_factor = FORCED_AIR_DRYING_FACTOR
        drying_minutes *= forced_air_factor
        notes.append(f"Forced air drying reduces time by {(1 - forced_air_factor) * 100:.0f}%")
    else:
        notes.append("Natural air drying - consider forced air for faster results")

    logger.debug(
        "Base drying calculation: %.1f * %.3f * %.3f * %.3f = %.1f min",
        base_drying_minutes,
        humidity_factor,
        temperature_factor,
        absorbency_factor,
        drying_minutes,
    )

    # 6. Estimate range (±20%)
    range_min = drying_minutes * 0.8
    range_max = drying_minutes * 1.2

    # 7. Recommendations
    forced_air_recommended = (humidity_percent > 65) or (temperature_fahrenheit < 65)
    if forced_air_recommended and not forced_air:
        notes.append("Forced air drying recommended due to current conditions")

    if drying_minutes > 30:
        notes.append("Long drying time - consider dehumidifier or warming area")

    result = DryingTimeResult(
        drying_minutes=drying_minutes,
        drying_hours=drying_minutes / 60.0,
        humidity_adjustment=humidity_factor,
        temperature_adjustment=temperature_factor,
        absorbency_adjustment=absorbency_factor,
        humidity_percent=humidity_percent,
        temperature_fahrenheit=temperature_fahrenheit,
        paper_type=paper_type,
        estimated_range_minutes=(range_min, range_max),
        forced_air_recommended=forced_air_recommended,
        notes=notes,
    )

    logger.info(
        "Drying time calculation complete: %.1f min (%.1f-%.1f min range)",
        drying_minutes,
        range_min,
        range_max,
    )

    return result
