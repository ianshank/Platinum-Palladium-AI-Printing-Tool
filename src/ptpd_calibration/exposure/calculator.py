"""
Exposure calculator for UV printing processes.

Calculate exposure times based on negative density, light source, and paper type.
Uses industry-standard formulas for alternative printing processes.

References:
- 0.3 density = 1 stop exposure change
- Exposure adjustment = 2^(density_difference / 0.3)
- Inverse square law for distance adjustments
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class LightSource(str, Enum):
    """Common UV light sources for alternative printing."""

    NUARC_26_1K = "nuarc_26_1k"  # NuArc 26-1K platemaker
    NUARC_FT40 = "nuarc_ft40"  # NuArc FT40
    BL_FLUORESCENT = "bl_fluorescent"  # BL fluorescent tubes
    BLB_FLUORESCENT = "blb_fluorescent"  # BLB blacklight tubes
    LED_UV = "led_uv"  # UV LED array
    METAL_HALIDE = "metal_halide"  # Metal halide
    MERCURY_VAPOR = "mercury_vapor"  # Mercury vapor
    SUNLIGHT = "sunlight"  # Direct sunlight
    CUSTOM = "custom"  # User-defined


# Relative speed multipliers (1.0 = standard BL fluorescent)
LIGHT_SOURCE_SPEEDS = {
    LightSource.NUARC_26_1K: 0.7,  # Faster than fluorescent
    LightSource.NUARC_FT40: 0.8,
    LightSource.BL_FLUORESCENT: 1.0,  # Baseline
    LightSource.BLB_FLUORESCENT: 1.2,  # Slightly slower
    LightSource.LED_UV: 0.6,  # Very fast
    LightSource.METAL_HALIDE: 0.5,  # Fastest artificial
    LightSource.MERCURY_VAPOR: 0.6,
    LightSource.SUNLIGHT: 0.4,  # Variable but fast
    LightSource.CUSTOM: 1.0,
}


@dataclass
class ExposureSettings:
    """Settings for exposure calculation."""

    # Reference exposure
    base_exposure_minutes: float = 10.0
    base_negative_density: float = 1.6

    # Light source
    light_source: LightSource = LightSource.BL_FLUORESCENT
    custom_speed_multiplier: float = 1.0

    # Distance
    base_distance_inches: float = 4.0

    # Paper adjustments
    paper_speed_factor: float = 1.0  # 1.0 = average

    # Chemistry adjustments
    # Platinum is slower than palladium
    platinum_ratio: float = 0.0  # 0 = all Pd, 1 = all Pt
    # Chlorate increases exposure time
    contrast_agent_factor: float = 1.0  # 1.0 = none, up to 1.5 for heavy

    # Environmental
    humidity_adjustment: float = 1.0  # Higher humidity = faster


@dataclass
class ExposureResult:
    """Result of exposure calculation."""

    exposure_minutes: float
    exposure_seconds: float

    # Breakdown
    base_exposure: float
    density_adjustment: float
    light_source_adjustment: float
    distance_adjustment: float
    paper_adjustment: float
    chemistry_adjustment: float
    environmental_adjustment: float

    # Input values for reference
    negative_density: float
    light_source: LightSource
    distance_inches: float

    # Notes
    notes: list[str] = field(default_factory=list)

    def format_time(self) -> str:
        """Format exposure time as human-readable string."""
        if self.exposure_minutes < 1:
            return f"{self.exposure_seconds:.0f} seconds"
        elif self.exposure_minutes < 60:
            mins = int(self.exposure_minutes)
            secs = int((self.exposure_minutes - mins) * 60)
            if secs == 0:
                return f"{mins} minutes"
            return f"{mins} min {secs} sec"
        else:
            hours = int(self.exposure_minutes / 60)
            mins = int(self.exposure_minutes % 60)
            return f"{hours} hour{'s' if hours > 1 else ''} {mins} min"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "exposure_time": self.format_time(),
            "exposure_minutes": round(self.exposure_minutes, 2),
            "exposure_seconds": round(self.exposure_seconds, 1),
            "adjustments": {
                "base": round(self.base_exposure, 2),
                "density": round(self.density_adjustment, 3),
                "light_source": round(self.light_source_adjustment, 3),
                "distance": round(self.distance_adjustment, 3),
                "paper": round(self.paper_adjustment, 3),
                "chemistry": round(self.chemistry_adjustment, 3),
                "environmental": round(self.environmental_adjustment, 3),
            },
            "inputs": {
                "negative_density": self.negative_density,
                "light_source": self.light_source.value,
                "distance_inches": self.distance_inches,
            },
            "notes": self.notes,
        }


class ExposureCalculator:
    """Calculate exposure times for alternative printing processes.

    Uses industry-standard formulas:
    - Density-based adjustment: 2^(delta_density / 0.3)
    - Distance adjustment: (new_distance / base_distance)^2
    - Various multipliers for light source, paper, chemistry
    """

    # Standard density for one stop
    DENSITY_PER_STOP = 0.3

    def __init__(self, settings: Optional[ExposureSettings] = None):
        """Initialize exposure calculator.

        Args:
            settings: Exposure settings. If None, uses defaults.
        """
        self.settings = settings or ExposureSettings()

    def calculate(
        self,
        negative_density: float,
        distance_inches: Optional[float] = None,
        light_source: Optional[LightSource] = None,
        paper_speed: Optional[float] = None,
        platinum_ratio: Optional[float] = None,
        humidity_factor: Optional[float] = None,
    ) -> ExposureResult:
        """Calculate exposure time for given conditions.

        Args:
            negative_density: Density range of negative (Dmax - Dmin)
            distance_inches: Distance from light source (if different from base)
            light_source: Light source type (if different from settings)
            paper_speed: Paper speed factor (1.0 = average)
            platinum_ratio: Platinum ratio (0-1, affects speed)
            humidity_factor: Humidity adjustment (higher = faster)

        Returns:
            ExposureResult with calculated exposure time
        """
        notes = []

        # Get effective values
        distance = distance_inches or self.settings.base_distance_inches
        source = light_source or self.settings.light_source
        paper = paper_speed or self.settings.paper_speed_factor
        pt_ratio = platinum_ratio if platinum_ratio is not None else self.settings.platinum_ratio
        humidity = humidity_factor or self.settings.humidity_adjustment

        # Start with base exposure
        base = self.settings.base_exposure_minutes

        # 1. Density adjustment
        # More dense negative = more exposure needed
        density_delta = negative_density - self.settings.base_negative_density
        density_adjustment = 2 ** (density_delta / self.DENSITY_PER_STOP)

        if density_delta > 0.3:
            notes.append(f"Dense negative: +{density_delta:.2f} density requires {density_adjustment:.1f}x exposure")
        elif density_delta < -0.3:
            notes.append(f"Thin negative: {density_delta:.2f} density requires {density_adjustment:.2f}x exposure")

        # 2. Light source adjustment
        if source == LightSource.CUSTOM:
            light_adjustment = self.settings.custom_speed_multiplier
        else:
            light_adjustment = LIGHT_SOURCE_SPEEDS.get(source, 1.0)

        # 3. Distance adjustment (inverse square law)
        distance_adjustment = (distance / self.settings.base_distance_inches) ** 2

        if distance != self.settings.base_distance_inches:
            notes.append(f"Distance {distance}\" vs base {self.settings.base_distance_inches}\": {distance_adjustment:.2f}x")

        # 4. Paper adjustment
        paper_adjustment = paper

        # 5. Chemistry adjustment
        # Platinum is about 2x slower than palladium
        pt_speed_factor = 1.0 + (pt_ratio * 1.0)  # Up to 2x for pure Pt
        chemistry_adjustment = pt_speed_factor * self.settings.contrast_agent_factor

        if pt_ratio > 0.5:
            notes.append(f"High platinum ({pt_ratio*100:.0f}%): slower exposure needed")

        # 6. Environmental adjustment
        environmental_adjustment = 1.0 / humidity  # Higher humidity = faster

        # Calculate final exposure
        exposure_minutes = (
            base *
            density_adjustment *
            light_adjustment *
            distance_adjustment *
            paper_adjustment *
            chemistry_adjustment *
            environmental_adjustment
        )

        # Add warnings for extreme values
        if exposure_minutes > 30:
            notes.append("Warning: Long exposure time. Consider reducing negative density or using faster light source.")
        if exposure_minutes < 2:
            notes.append("Warning: Short exposure. Risk of underexposure. Consider adding neutral density or increasing distance.")

        return ExposureResult(
            exposure_minutes=exposure_minutes,
            exposure_seconds=exposure_minutes * 60,
            base_exposure=base,
            density_adjustment=density_adjustment,
            light_source_adjustment=light_adjustment,
            distance_adjustment=distance_adjustment,
            paper_adjustment=paper_adjustment,
            chemistry_adjustment=chemistry_adjustment,
            environmental_adjustment=environmental_adjustment,
            negative_density=negative_density,
            light_source=source,
            distance_inches=distance,
            notes=notes,
        )

    def calculate_test_strip(
        self,
        center_exposure: float,
        steps: int = 5,
        increment_stops: float = 0.5,
    ) -> list[float]:
        """Calculate exposure times for a test strip.

        Args:
            center_exposure: Center/base exposure in minutes
            steps: Number of steps (odd number recommended)
            increment_stops: Exposure increment in stops

        Returns:
            List of exposure times in minutes
        """
        times = []
        half_steps = steps // 2

        for i in range(-half_steps, half_steps + 1):
            factor = 2 ** (i * increment_stops)
            times.append(center_exposure * factor)

        return times

    def density_to_stops(self, density_change: float) -> float:
        """Convert density change to stops.

        Args:
            density_change: Change in density units

        Returns:
            Equivalent change in stops
        """
        return density_change / self.DENSITY_PER_STOP

    def stops_to_density(self, stops: float) -> float:
        """Convert stops to density change.

        Args:
            stops: Change in stops

        Returns:
            Equivalent change in density units
        """
        return stops * self.DENSITY_PER_STOP

    def adjust_for_distance(
        self,
        current_exposure: float,
        current_distance: float,
        new_distance: float,
    ) -> float:
        """Adjust exposure time for distance change.

        Uses inverse square law.

        Args:
            current_exposure: Current exposure time in minutes
            current_distance: Current distance in inches
            new_distance: New distance in inches

        Returns:
            Adjusted exposure time in minutes
        """
        return current_exposure * (new_distance / current_distance) ** 2

    @staticmethod
    def get_light_sources() -> list[tuple[str, str]]:
        """Get list of light sources with descriptions.

        Returns:
            List of (value, description) tuples
        """
        return [
            (LightSource.NUARC_26_1K.value, "NuArc 26-1K Platemaker"),
            (LightSource.NUARC_FT40.value, "NuArc FT40"),
            (LightSource.BL_FLUORESCENT.value, "BL Fluorescent Tubes"),
            (LightSource.BLB_FLUORESCENT.value, "BLB Blacklight Tubes"),
            (LightSource.LED_UV.value, "UV LED Array"),
            (LightSource.METAL_HALIDE.value, "Metal Halide"),
            (LightSource.MERCURY_VAPOR.value, "Mercury Vapor"),
            (LightSource.SUNLIGHT.value, "Direct Sunlight"),
            (LightSource.CUSTOM.value, "Custom Source"),
        ]
