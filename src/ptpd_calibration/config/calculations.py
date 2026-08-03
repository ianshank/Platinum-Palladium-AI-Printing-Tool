"""
Configuration for calculation functions.

This module centralizes all constants used by calculation functions,
ensuring single source of truth and eliminating hardcoded values.

All values are derived from:
- Industry standards (photography, chemistry)
- Experimental data from platinum/palladium printing
- User feedback and practical experience
- Published research on alternative printing processes

References:
- The Book of Film: A Complete Guide to 16 & 35mm Film (Kodak)
- Print Permanence and Collection Storage (Library of Congress)
- Platinum and Palladium Printing: Harrington, Curtis (1994)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


# ============================================================================
# Density and Stops Constants
# ============================================================================

# Industry standard: 0.3 density = 1 photographic stop
# This represents doubling or halving of exposure
DENSITY_PER_STOP = 0.3

# Exposure calculation formula uses exponential:
# factor = 2^(density_delta / DENSITY_PER_STOP)
# Example: 0.6D change = 2 stops = 4x exposure


# ============================================================================
# Environmental Reference Conditions
# ============================================================================

# Optimal humidity for coating and exposure (%)
# Below 30%: uneven coating, dry air problems
# 30-70%: acceptable range
# Above 70%: coating issues, longer drying
OPTIMAL_HUMIDITY_PERCENT = 50.0

# Optimal temperature for exposure (Fahrenheit)
# Equivalent to 20°C
# Below 60°F: slow chemical reactions
# 60-75°F: acceptable range
# Above 80°F: accelerated reactions, uneven exposure
OPTIMAL_TEMPERATURE_FAHRENHEIT = 68.0

# Convert Fahrenheit to Celsius for reference
OPTIMAL_TEMPERATURE_CELSIUS = 20.0


# ============================================================================
# Exposure Calculation Coefficients
# ============================================================================

# Humidity coefficient for exposure calculations
# 15% change per 100% humidity delta
# Higher humidity slightly accelerates exposure due to water in emulsion
HUMIDITY_COEFFICIENT_EXPOSURE = 0.15

# Temperature coefficient for exposure calculations (per 10°F)
# 5% change per 10°F
# Higher temperature increases chemical activity
TEMPERATURE_COEFFICIENT_EXPOSURE = 0.05

# Humidity coefficient for drying calculations
# 40% change per 100% humidity delta
# Higher humidity significantly slows drying
HUMIDITY_COEFFICIENT_DRYING = 0.40

# Temperature coefficient for drying calculations (per 10°F)
# 15% change per 10°F
# Higher temperature significantly speeds drying
TEMPERATURE_COEFFICIENT_DRYING = 0.15


# ============================================================================
# Light Source Speed Factors
# ============================================================================

# Relative speed multipliers for UV light sources
# 1.0 = baseline (BL fluorescent tubes)
# Values derived from manufacturer specs and practical experience
#
# Speed relationships:
# - Direct sunlight: ~0.4x (very fast, variable with season/time)
# - LED UV 365nm: ~0.6x (modern, very fast and stable)
# - NuArc 26-1K: ~0.7x (professional platemaker, fast)
# - BL fluorescent: 1.0x (baseline - standard tubes)
# - BLB fluorescent: ~1.2x (slightly less UV output)
# - Metal halide: ~0.5x (intense, very fast)
# - Mercury vapor: ~0.6x (intense UV, fast)

UV_LIGHT_SOURCE_SPEEDS = {
    "sunlight_direct": 0.4,  # Fastest but variable
    "sunlight_shade": 1.5,  # Much slower in shade
    "sunlight_cloudy": 2.0,  # Quite slow on cloudy days
    "led_uv_365": 0.6,  # Modern, efficient, very fast
    "led_uv_395": 0.5,  # Very efficient
    "nuarc_26_1k": 0.7,  # Professional platemaker
    "nuarc_ft40": 0.8,  # Smaller NuArc unit
    "bl_fluorescent": 1.0,  # Baseline reference
    "blb_fluorescent": 1.2,  # BLB variant
    "metal_halide": 0.5,  # Very intense
    "mercury_vapor": 0.6,  # Intense UV output
}

# Enlarger light source speeds (for silver gelatin)
ENLARGER_LIGHT_SOURCE_SPEEDS = {
    "tungsten_incandescent": 1.0,  # Baseline
    "tungsten_halogen": 0.8,  # Slightly faster
    "cold_light": 1.2,  # Cooler, slightly slower
    "led_enlarger": 0.6,  # Modern LEDs very efficient
    "color_head": 1.1,  # Slight loss through filters
}


# ============================================================================
# Paper Characteristics
# ============================================================================

class PaperType(str, Enum):
    """Paper types and their characteristics."""

    HOT_PRESS = "hot_press"
    COLD_PRESS = "cold_press"
    ROUGH = "rough"
    SIZED = "sized"
    UNSIZED = "unsized"


# Paper absorbency factors (relative to baseline 1.0)
# Affects both exposure time (coating absorption) and drying time
# Higher factor = slower exposure, slower drying
# Based on paper porosity and sizing characteristics
PAPER_ABSORBENCY_FACTORS = {
    "hot_press": 0.9,  # Hot press: low absorbency, fast exposure/drying
    "cold_press": 1.2,  # Cold press: higher absorbency, slower exposure/drying
    "rough": 1.3,  # Rough: highest absorbency, slowest drying
    "sized": 0.95,  # Sized: slightly lower absorbency
    "unsized": 1.1,  # Unsized: slightly higher absorbency
}

# Typical paper speeds for different types
# Used for exposure adjustment when paper type is specified
PAPER_SPEED_FACTORS = {
    "fast": 0.85,  # Fast paper requires less exposure
    "medium": 1.0,  # Medium is baseline
    "slow": 1.15,  # Slow paper requires more exposure
}


# ============================================================================
# Chemistry and Process Factors
# ============================================================================

# Chemistry adjustment factors
# These account for different sensitizers and processing
CHEMISTRY_FACTORS = {
    "standard_platinum": 1.0,  # Baseline
    "standard_palladium": 0.9,  # Palladium slightly faster
    "platinum_heavy": 1.1,  # Heavy platinum ratio
    "palladium_heavy": 0.85,  # Heavy palladium ratio
}

# Platinum ratio effects on exposure
# Pure palladium (0.0) is faster; pure platinum (1.0) is slower
# Platinum is roughly 2x slower than palladium
def get_platinum_ratio_factor(platinum_percent: float) -> float:
    """Calculate exposure factor based on platinum percentage.

    Args:
        platinum_percent: Percentage platinum (0-100).

    Returns:
        Exposure factor (1.0 = baseline palladium).
    """
    # Linear interpolation: 0% Pt = 1.0x, 100% Pt = 2.0x exposure
    pt_ratio = platinum_percent / 100.0
    return 1.0 + pt_ratio


# Contrast agent (chlorate) effects
CONTRAST_AGENT_FACTORS = {
    "none": 1.0,  # No contrast agent
    "light": 1.05,  # Light chlorate
    "medium": 1.1,  # Medium chlorate
    "heavy": 1.25,  # Heavy chlorate (max ~1.5)
}


# ============================================================================
# Drying Time Base Values
# ============================================================================

# Base drying time at optimal conditions (50% humidity, 68°F)
# This is for a typical sized paper with coating
BASE_DRYING_TIME_MINUTES = 15.0

# Minimum drying time (even under ideal conditions)
MIN_DRYING_TIME_MINUTES = 5.0

# Maximum estimated drying time (safety limit)
MAX_DRYING_TIME_MINUTES = 300.0  # 5 hours

# Forced air drying speed multiplier
# Reduces drying time to this fraction of natural drying
FORCED_AIR_DRYING_FACTOR = 0.5  # 50% of natural drying time


# ============================================================================
# Environmental Limits and Warnings
# ============================================================================

class EnvironmentalLimits:
    """Limits and warning thresholds for environmental conditions."""

    # Humidity limits
    HUMIDITY_MIN_WARNING = 30.0  # %
    HUMIDITY_MAX_WARNING = 70.0  # %

    # Temperature limits (Fahrenheit)
    TEMPERATURE_MIN_WARNING = 60.0  # °F
    TEMPERATURE_MAX_WARNING = 80.0  # °F

    # UV intensity limits (percent of reference)
    UV_INTENSITY_MIN_WARNING = 70.0  # %
    UV_INTENSITY_MAX_WARNING = 120.0  # %


# ============================================================================
# Calculation Defaults and Uncertainties
# ============================================================================

# Default uncertainty for confidence intervals (±percent)
DEFAULT_UNCERTAINTY_PERCENT = 10.0

# Exposure time warnings
EXPOSURE_TIME_WARNINGS = {
    "very_short": 1.0,  # minutes - risk of underexposure inconsistency
    "short": 2.0,  # minutes - short exposure risk
    "long": 30.0,  # minutes - consider alternate approach
    "very_long": 60.0,  # minutes - potential reciprocity issues
}

# Drying time warnings
DRYING_TIME_WARNINGS = {
    "short": 5.0,  # minutes - unusually fast
    "very_long": 30.0,  # minutes - consider dehumidifier
    "extreme": 120.0,  # minutes - environmental intervention recommended
}


# ============================================================================
# Cyanotype Specific (used by alternative_calculators)
# ============================================================================

class CyanotypeFormula(str, Enum):
    """Cyanotype formulas and their relative speeds."""

    CLASSIC = "classic"  # Original formula - baseline 1.0x
    NEW = "new"  # New cyanotype - faster ~0.7x
    WARE = "ware"  # Alternative formulation ~0.75x
    REX = "rex"  # Another variant ~0.9x


CYANOTYPE_FORMULA_SPEEDS = {
    CyanotypeFormula.CLASSIC: 1.0,  # Baseline
    CyanotypeFormula.NEW: 0.7,  # New cyanotype faster
    CyanotypeFormula.WARE: 0.75,
    CyanotypeFormula.REX: 0.9,
}

# Cyanotype base exposure (BL tubes, medium negative)
CYANOTYPE_BASE_EXPOSURE_MINUTES = 15.0


# ============================================================================
# Silver Gelatin Specific (used by alternative_calculators)
# ============================================================================

class PaperGrade(str, Enum):
    """Silver gelatin paper contrast grades."""

    GRADE_00 = "grade_00"
    GRADE_0 = "grade_0"
    GRADE_1 = "grade_1"
    GRADE_2 = "grade_2"  # Medium - baseline
    GRADE_3 = "grade_3"
    GRADE_4 = "grade_4"
    GRADE_5 = "grade_5"
    VARIABLE = "variable"


# Multigrade filter factors for silver gelatin
MULTIGRADE_FILTER_FACTORS = {
    "Grade 00": 2.5,
    "Grade 0": 1.7,
    "Grade 0.5": 1.3,
    "Grade 1": 1.1,
    "Grade 1.5": 1.0,
    "Grade 2": 1.0,  # Baseline
    "Grade 2.5": 1.0,
    "Grade 3": 1.1,
    "Grade 3.5": 1.2,
    "Grade 4": 1.3,
    "Grade 4.5": 1.5,
    "Grade 5": 1.7,
}


# ============================================================================
# Test Strip Configuration
# ============================================================================

# Default test strip parameters
TEST_STRIP_DEFAULTS = {
    "steps": 5,  # Number of test steps
    "increment_stops": 0.5,  # Increment between steps
}

# Alternative test strip configurations
TEST_STRIP_PRESETS = {
    "quick": {"steps": 3, "increment_stops": 1.0},  # Fast, coarse
    "standard": {"steps": 5, "increment_stops": 0.5},  # Default
    "fine": {"steps": 7, "increment_stops": 0.33},  # Fine detail
    "very_fine": {"steps": 11, "increment_stops": 0.25},  # Very detailed
}

# Test strip exposure factors (relative to center)
TEST_STRIP_FACTORS = [0.5, 0.707, 1.0, 1.414, 2.0]  # ±1 stop in 0.5 stop increments


# ============================================================================
# Logging Configuration
# ============================================================================

# Enable debug logging for calculations
LOG_CALCULATION_DETAILS = False

# Log all adjustment factors
LOG_ADJUSTMENT_FACTORS = True

# Log input validation
LOG_VALIDATION = True


# ============================================================================
# Validation Ranges
# ============================================================================

# Valid ranges for different parameters
VALID_RANGES = {
    "exposure_minutes": (0.1, 180.0),  # 0.1 - 180 minutes
    "negative_density": (0.0, 4.0),  # 0 - 4.0 density
    "humidity_percent": (0.0, 100.0),  # 0-100%
    "temperature_fahrenheit": (-50.0, 150.0),  # -50 to 150°F
    "uv_intensity_percent": (0.1, 200.0),  # 0.1 - 200%
    "paper_factor": (0.5, 2.0),  # 0.5 - 2.0x
    "chemistry_factor": (0.5, 2.0),  # 0.5 - 2.0x
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_light_source_speed(
    source: str, source_type: str = "uv"
) -> float:
    """Get speed factor for a light source.

    Args:
        source: Light source name (e.g., "led_uv_365").
        source_type: "uv" for UV sources, "enlarger" for enlarger lights.

    Returns:
        Speed factor (1.0 = baseline).

    Raises:
        ValueError: If source not found.
    """
    if source_type == "uv":
        if source not in UV_LIGHT_SOURCE_SPEEDS:
            raise ValueError(f"Unknown UV light source: {source}")
        return UV_LIGHT_SOURCE_SPEEDS[source]
    elif source_type == "enlarger":
        if source not in ENLARGER_LIGHT_SOURCE_SPEEDS:
            raise ValueError(f"Unknown enlarger light source: {source}")
        return ENLARGER_LIGHT_SOURCE_SPEEDS[source]
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def validate_exposure_inputs(
    base_time: float,
    negative_density: float,
    humidity: float,
    temperature: float,
    uv_intensity: float = 100.0,
) -> list[str]:
    """Validate exposure calculation inputs.

    Args:
        base_time: Base exposure time in minutes.
        negative_density: Negative density range.
        humidity: Relative humidity (%).
        temperature: Temperature (°F).
        uv_intensity: UV intensity (%).

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    if not (VALID_RANGES["exposure_minutes"][0] <= base_time <= VALID_RANGES["exposure_minutes"][1]):
        errors.append(f"base_time must be between {VALID_RANGES['exposure_minutes']}")

    if not (
        VALID_RANGES["negative_density"][0]
        <= negative_density
        <= VALID_RANGES["negative_density"][1]
    ):
        errors.append(f"negative_density must be between {VALID_RANGES['negative_density']}")

    if not (VALID_RANGES["humidity_percent"][0] <= humidity <= VALID_RANGES["humidity_percent"][1]):
        errors.append(f"humidity must be between {VALID_RANGES['humidity_percent']}")

    if not (
        VALID_RANGES["temperature_fahrenheit"][0]
        <= temperature
        <= VALID_RANGES["temperature_fahrenheit"][1]
    ):
        errors.append(f"temperature must be between {VALID_RANGES['temperature_fahrenheit']}")

    if not (VALID_RANGES["uv_intensity_percent"][0] <= uv_intensity <= VALID_RANGES["uv_intensity_percent"][1]):
        errors.append(f"uv_intensity must be between {VALID_RANGES['uv_intensity_percent']}")

    return errors
