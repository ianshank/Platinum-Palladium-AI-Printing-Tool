"""
Exposure calculators for alternative photographic printing processes.

Provides exposure calculations for:
- Cyanotype (UV contact printing)
- Silver Gelatin (enlarger/contact printing)
- Van Dyke Brown
- Kallitype

Each process has unique exposure characteristics based on sensitivity
to different wavelengths, chemistry, and paper types.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from ptpd_calibration.core.types import (
    CyanotypeFormula,
    PaperGrade,
)


class EnlargerLightSource(str, Enum):
    """Light sources for silver gelatin enlarger printing."""

    TUNGSTEN_INCANDESCENT = "tungsten_incandescent"  # Traditional bulb
    TUNGSTEN_HALOGEN = "tungsten_halogen"  # Quartz halogen
    COLD_LIGHT = "cold_light"  # Diffused fluorescent (Aristo)
    LED_ENLARGER = "led_enlarger"  # Modern LED head
    COLOR_HEAD = "color_head"  # Dichroic color head (for filtering)


# Relative speed multipliers for enlarger light sources
ENLARGER_LIGHT_SPEEDS = {
    EnlargerLightSource.TUNGSTEN_INCANDESCENT: 1.0,  # Baseline
    EnlargerLightSource.TUNGSTEN_HALOGEN: 0.8,  # Slightly faster
    EnlargerLightSource.COLD_LIGHT: 1.2,  # Cooler, slightly slower
    EnlargerLightSource.LED_ENLARGER: 0.6,  # Modern LEDs are efficient
    EnlargerLightSource.COLOR_HEAD: 1.1,  # Slight loss through filters
}


class UVSource(str, Enum):
    """UV light sources for alternative process contact printing."""

    DIRECT_SUNLIGHT = "direct_sunlight"  # Midday sun
    SUNLIGHT_SHADE = "sunlight_shade"  # Open shade
    SUNLIGHT_CLOUDY = "sunlight_cloudy"  # Overcast
    BL_TUBES = "bl_tubes"  # BL fluorescent tubes
    BLB_TUBES = "blb_tubes"  # BLB blacklight tubes
    LED_UV_365 = "led_uv_365"  # UV LED at 365nm
    LED_UV_395 = "led_uv_395"  # UV LED at 395nm
    METAL_HALIDE = "metal_halide"  # Metal halide exposure unit
    NUARC = "nuarc"  # NuArc platemaker
    AMERGRAPH = "amergraph"  # Amergraph ULF units


# Relative speeds for UV sources (1.0 = standard BL tubes)
UV_SOURCE_SPEEDS = {
    UVSource.DIRECT_SUNLIGHT: 0.3,  # Very fast
    UVSource.SUNLIGHT_SHADE: 1.5,  # Much slower
    UVSource.SUNLIGHT_CLOUDY: 2.0,  # Quite slow
    UVSource.BL_TUBES: 1.0,  # Baseline
    UVSource.BLB_TUBES: 1.3,  # Slightly slower
    UVSource.LED_UV_365: 0.4,  # Very fast
    UVSource.LED_UV_395: 0.5,  # Fast
    UVSource.METAL_HALIDE: 0.35,  # Very fast
    UVSource.NUARC: 0.5,  # Fast
    UVSource.AMERGRAPH: 0.6,  # Fast
}


@dataclass
class CyanotypeExposureResult:
    """Result of cyanotype exposure calculation."""

    exposure_minutes: float
    exposure_seconds: float

    # Breakdown of adjustments
    base_exposure: float
    negative_density_adjustment: float
    uv_source_adjustment: float
    formula_adjustment: float
    humidity_adjustment: float
    paper_adjustment: float

    # Input parameters
    negative_density: float
    uv_source: UVSource
    formula: CyanotypeFormula

    # Visual indicators
    unexposed_color: str
    properly_exposed_color: str
    overexposed_color: str

    # Notes and recommendations
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
        """Convert to dictionary for JSON serialization."""
        return {
            "exposure_time": self.format_time(),
            "exposure_minutes": round(self.exposure_minutes, 2),
            "exposure_seconds": round(self.exposure_seconds, 1),
            "adjustments": {
                "base": round(self.base_exposure, 2),
                "negative_density": round(self.negative_density_adjustment, 3),
                "uv_source": round(self.uv_source_adjustment, 3),
                "formula": round(self.formula_adjustment, 3),
                "humidity": round(self.humidity_adjustment, 3),
                "paper": round(self.paper_adjustment, 3),
            },
            "visual_indicators": {
                "unexposed": self.unexposed_color,
                "properly_exposed": self.properly_exposed_color,
                "overexposed": self.overexposed_color,
            },
            "notes": self.notes,
        }


@dataclass
class SilverGelatinExposureResult:
    """Result of silver gelatin exposure calculation."""

    exposure_seconds: float
    f_stop: float

    # Breakdown
    base_exposure: float
    magnification_adjustment: float
    filter_adjustment: float
    paper_speed_adjustment: float
    reciprocity_adjustment: float

    # Input parameters
    enlarger_height_cm: float
    paper_grade: PaperGrade
    filter_factor: float

    # Test strip recommendations
    test_strip_times: list[float]

    # Notes
    notes: list[str] = field(default_factory=list)

    def format_time(self) -> str:
        """Format exposure time."""
        if self.exposure_seconds < 1:
            return f"{self.exposure_seconds:.2f} sec"
        elif self.exposure_seconds < 60:
            return f"{self.exposure_seconds:.1f} sec"
        else:
            mins = int(self.exposure_seconds / 60)
            secs = self.exposure_seconds % 60
            return f"{mins}:{secs:05.2f}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "exposure_seconds": round(self.exposure_seconds, 2),
            "f_stop": self.f_stop,
            "formatted_time": self.format_time(),
            "adjustments": {
                "base": round(self.base_exposure, 2),
                "magnification": round(self.magnification_adjustment, 3),
                "filter": round(self.filter_adjustment, 3),
                "paper_speed": round(self.paper_speed_adjustment, 3),
                "reciprocity": round(self.reciprocity_adjustment, 3),
            },
            "test_strip_times": [round(t, 1) for t in self.test_strip_times],
            "notes": self.notes,
        }


class CyanotypeExposureCalculator:
    """Exposure calculator for cyanotype printing.

    Cyanotype is a UV-sensitive iron-based process that requires
    careful exposure calculation based on:
    - UV light source intensity and type
    - Negative density range
    - Cyanotype formula (classic vs new)
    - Environmental conditions (humidity)
    - Paper/substrate type
    """

    # Standard density for one stop
    DENSITY_PER_STOP = 0.3

    def __init__(self):
        """Initialize the cyanotype exposure calculator."""
        # Base exposure in minutes for BL tubes with average negative
        self.base_exposure_minutes = 15.0
        self.base_negative_density = 1.6

    def calculate(
        self,
        negative_density: float = 1.6,
        uv_source: UVSource = UVSource.BL_TUBES,
        formula: CyanotypeFormula = CyanotypeFormula.CLASSIC,
        humidity_percent: float = 50.0,
        paper_factor: float = 1.0,
        distance_inches: float = 4.0,
        base_distance_inches: float = 4.0,
    ) -> CyanotypeExposureResult:
        """Calculate cyanotype exposure time.

        Args:
            negative_density: Density range of digital negative (Dmax - Dmin)
            uv_source: Type of UV light source
            formula: Cyanotype formula (classic or new)
            humidity_percent: Relative humidity (affects sensitivity)
            paper_factor: Paper absorbency factor (1.0 = average)
            distance_inches: Distance from UV source
            base_distance_inches: Reference distance

        Returns:
            CyanotypeExposureResult with calculated exposure

        Raises:
            ValueError: If inputs are invalid
        """
        if negative_density < 0:
            raise ValueError("negative density cannot be negative")
        if not (0 <= humidity_percent <= 100):
            raise ValueError("humidity must be between 0 and 100")
        if paper_factor <= 0:
            raise ValueError("paper factor must be positive")
        if distance_inches <= 0:
            raise ValueError("distance must be positive")

        notes = []
        base = self.base_exposure_minutes

        # 1. Negative density adjustment
        density_delta = negative_density - self.base_negative_density
        density_adjustment = 2 ** (density_delta / self.DENSITY_PER_STOP)

        if density_delta > 0.3:
            notes.append(f"Dense negative (+{density_delta:.2f}) - longer exposure needed")
        elif density_delta < -0.3:
            notes.append(f"Thin negative ({density_delta:.2f}) - shorter exposure")

        # 2. UV source adjustment
        uv_adjustment = UV_SOURCE_SPEEDS.get(uv_source, 1.0)

        if uv_source == UVSource.DIRECT_SUNLIGHT:
            notes.append("Sunlight exposure varies - watch for color shift to bronze/olive")

        # 3. Formula adjustment
        formula_factors = {
            CyanotypeFormula.CLASSIC: 1.0,
            CyanotypeFormula.NEW: 0.7,  # New cyanotype is faster
            CyanotypeFormula.WARE: 0.75,
            CyanotypeFormula.REX: 0.9,
        }
        formula_adjustment = formula_factors.get(formula, 1.0)

        if formula == CyanotypeFormula.NEW:
            notes.append("New Cyanotype formula exposes faster than classic")

        # 4. Humidity adjustment
        # Higher humidity generally increases sensitivity slightly
        humidity_adjustment = 1.0 + (50.0 - humidity_percent) * 0.005

        if humidity_percent > 70:
            notes.append("High humidity - coating may not dry fully, reduce if possible")
        elif humidity_percent < 30:
            notes.append("Low humidity - longer exposure may be needed")

        # 5. Paper factor
        paper_adjustment = paper_factor

        # 6. Distance adjustment (inverse square law)
        distance_adjustment = (distance_inches / base_distance_inches) ** 2

        # Calculate final exposure
        exposure_minutes = (
            base
            * density_adjustment
            * uv_adjustment
            * formula_adjustment
            * humidity_adjustment
            * paper_adjustment
            * distance_adjustment
        )

        # Visual indicators for exposure monitoring
        if formula == CyanotypeFormula.NEW:
            unexposed = "Yellow-green"
            exposed = "Blue-grey (greenish tint gone)"
            over = "Dark grey-blue, possible bronzing"
        else:
            unexposed = "Yellow-green"
            exposed = "Bronze-olive (darker than unexposed)"
            over = "Dark bronze, solarization possible"

        # Add warnings
        if exposure_minutes > 45:
            notes.append("Long exposure - consider more efficient UV source")
        if exposure_minutes < 3:
            notes.append("Very short exposure - risk of underexposure inconsistency")

        notes.append("Exposure complete when color shifts from yellow-green to bronze/olive")

        return CyanotypeExposureResult(
            exposure_minutes=exposure_minutes,
            exposure_seconds=exposure_minutes * 60,
            base_exposure=base,
            negative_density_adjustment=density_adjustment,
            uv_source_adjustment=uv_adjustment,
            formula_adjustment=formula_adjustment,
            humidity_adjustment=humidity_adjustment,
            paper_adjustment=paper_adjustment,
            negative_density=negative_density,
            uv_source=uv_source,
            formula=formula,
            unexposed_color=unexposed,
            properly_exposed_color=exposed,
            overexposed_color=over,
            notes=notes,
        )

    def calculate_test_strip(
        self,
        center_exposure: float,
        strips: int = 5,
        increment_stops: float = 1.0,
    ) -> list[float]:
        """Calculate exposure times for test strip.

        Args:
            center_exposure: Center exposure in minutes
            strips: Number of test strips
            increment_stops: Exposure increment in stops

        Returns:
            List of exposure times in minutes
        """
        times = []
        half_strips = strips // 2

        for i in range(-half_strips, half_strips + 1):
            factor = 2 ** (i * increment_stops)
            times.append(center_exposure * factor)

        return times

    @staticmethod
    def get_uv_sources() -> list[tuple[str, str]]:
        """Get list of UV sources with descriptions."""
        return [
            (UVSource.DIRECT_SUNLIGHT.value, "Direct Sunlight (midday)"),
            (UVSource.SUNLIGHT_SHADE.value, "Open Shade"),
            (UVSource.SUNLIGHT_CLOUDY.value, "Cloudy/Overcast"),
            (UVSource.BL_TUBES.value, "BL Fluorescent Tubes"),
            (UVSource.BLB_TUBES.value, "BLB Blacklight Tubes"),
            (UVSource.LED_UV_365.value, "UV LED (365nm)"),
            (UVSource.LED_UV_395.value, "UV LED (395nm)"),
            (UVSource.METAL_HALIDE.value, "Metal Halide"),
            (UVSource.NUARC.value, "NuArc Platemaker"),
            (UVSource.AMERGRAPH.value, "Amergraph ULF"),
        ]


class SilverGelatinExposureCalculator:
    """Exposure calculator for silver gelatin enlarger printing.

    Calculates exposure times based on:
    - Enlarger height/magnification
    - Lens aperture (f-stop)
    - Paper grade and speed
    - Filter factors (for variable contrast)
    - Reciprocity for long exposures
    """

    def __init__(self):
        """Initialize silver gelatin exposure calculator."""
        # Base exposure for 8x10 print at f/8, grade 2, medium paper
        self.base_exposure_seconds = 10.0
        self.base_height_cm = 30.0  # Reference enlarger height

    def calculate(
        self,
        enlarger_height_cm: float = 30.0,
        f_stop: float = 8.0,
        paper_grade: PaperGrade = PaperGrade.GRADE_2,
        paper_speed_iso: float = 250.0,  # Paper speed (ISO-P)
        filter_factor: float = 1.0,  # Multigrade filter factor
        negative_density: float = 1.0,  # Average negative density
        base_f_stop: float = 8.0,
        light_source: EnlargerLightSource = EnlargerLightSource.TUNGSTEN_INCANDESCENT,
    ) -> SilverGelatinExposureResult:
        """Calculate silver gelatin exposure time.

        Args:
            enlarger_height_cm: Height of enlarger head
            f_stop: Lens aperture
            paper_grade: Paper contrast grade
            paper_speed_iso: Paper ISO-P speed (higher = faster)
            filter_factor: Multigrade filter factor
            negative_density: Average negative density
            base_f_stop: Reference f-stop
            light_source: Type of enlarger light source

        Returns:
            SilverGelatinExposureResult with calculated exposure

        Raises:
            ValueError: If inputs are invalid (negative or zero where not allowed)
        """
        # Input validation
        if enlarger_height_cm <= 0:
            raise ValueError("enlarger height must be positive")
        if f_stop <= 0:
            raise ValueError("f-stop must be positive")
        if paper_speed_iso <= 0:
            raise ValueError("paper speed (ISO) must be positive")
        if filter_factor <= 0:
            raise ValueError("filter factor must be positive")
        if negative_density < 0:
            raise ValueError("negative density cannot be negative")

        notes = []
        base = self.base_exposure_seconds

        # 1. Magnification adjustment (inverse square law approximation)
        height_ratio = enlarger_height_cm / self.base_height_cm
        magnification_adjustment = height_ratio**2

        if height_ratio > 2:
            notes.append("Large magnification - consider opening aperture or longer exposure")

        # 2. F-stop adjustment
        # Each stop doubles/halves exposure
        stop_difference = 2 * math.log2(f_stop / base_f_stop)
        f_stop_adjustment = 2**stop_difference

        # 3. Paper speed adjustment
        # ISO-P 250 is baseline
        paper_speed_adjustment = 250.0 / paper_speed_iso

        # 4. Filter factor for multigrade
        filter_adjustment = filter_factor

        if paper_grade == PaperGrade.VARIABLE and filter_factor > 1.5:
            notes.append("High contrast filter - may need split-grade printing")

        # 5. Light source adjustment
        light_source_speed = ENLARGER_LIGHT_SPEEDS.get(light_source, 1.0)

        # 6. Calculate base exposure before reciprocity
        pre_reciprocity = (
            base
            * magnification_adjustment
            * f_stop_adjustment
            * paper_speed_adjustment
            * filter_adjustment
            * light_source_speed
        )

        # 7. Reciprocity adjustment for long exposures
        reciprocity_adjustment = self._calculate_reciprocity(pre_reciprocity)

        if reciprocity_adjustment > 1.2:
            notes.append(f"Reciprocity correction applied: {reciprocity_adjustment:.2f}x")

        # Final exposure
        exposure_seconds = pre_reciprocity * reciprocity_adjustment

        # Generate test strip times
        test_times = self._generate_test_strip(exposure_seconds)

        # Add notes based on exposure
        if exposure_seconds < 3:
            notes.append("Very short exposure - consider stopping down lens")
        elif exposure_seconds > 60:
            notes.append("Long exposure - reciprocity may affect highlights")

        return SilverGelatinExposureResult(
            exposure_seconds=exposure_seconds,
            f_stop=f_stop,
            base_exposure=base,
            magnification_adjustment=magnification_adjustment,
            filter_adjustment=filter_adjustment * f_stop_adjustment,
            paper_speed_adjustment=paper_speed_adjustment,
            reciprocity_adjustment=reciprocity_adjustment,
            enlarger_height_cm=enlarger_height_cm,
            paper_grade=paper_grade,
            filter_factor=filter_factor,
            test_strip_times=test_times,
            notes=notes,
        )

    def calculate_dodging_burning(
        self,
        base_exposure: float,
        adjustment_stops: float,
    ) -> tuple[float, str]:
        """Calculate time for dodging or burning.

        Args:
            base_exposure: Base exposure time in seconds
            adjustment_stops: Stops to dodge (-) or burn (+)

        Returns:
            Tuple of (additional time, description)
        """
        factor = 2**adjustment_stops
        additional_time = base_exposure * (factor - 1)

        if adjustment_stops < 0:
            # Dodging - time to remove from exposure
            return abs(additional_time), f"Dodge for {abs(additional_time):.1f}s of exposure"
        else:
            # Burning - additional exposure time
            return additional_time, f"Burn for additional {additional_time:.1f}s"

    def _calculate_reciprocity(self, exposure_seconds: float) -> float:
        """Calculate reciprocity correction factor.

        Silver gelatin papers generally don't need much reciprocity
        correction until very long exposures.
        """
        if exposure_seconds < 10:
            return 1.0
        elif exposure_seconds < 30:
            return 1.1
        elif exposure_seconds < 60:
            return 1.2
        elif exposure_seconds < 120:
            return 1.35
        else:
            # For very long exposures
            return 1.5

    def _generate_test_strip(self, center_exposure: float) -> list[float]:
        """Generate test strip times at 1/3 stop increments."""
        times = []
        # 5 strips at 1/3 stop increments
        for i in range(-2, 3):
            factor = 2 ** (i / 3)
            times.append(center_exposure * factor)
        return times

    def calculate_split_grade(
        self,
        total_exposure: float,
        shadow_filter: float = 5.0,
        highlight_filter: float = 0.0,
        shadow_ratio: float = 0.5,
    ) -> dict:
        """Calculate split-grade exposure times.

        Args:
            total_exposure: Total exposure time in seconds
            shadow_filter: Filter grade for shadows (high contrast)
            highlight_filter: Filter grade for highlights (low contrast)
            shadow_ratio: Ratio of shadow to total exposure

        Returns:
            Dictionary with split-grade exposure info
        """
        shadow_time = total_exposure * shadow_ratio
        highlight_time = total_exposure * (1 - shadow_ratio)

        return {
            "shadow_exposure": {
                "time_seconds": shadow_time,
                "filter_grade": shadow_filter,
                "purpose": "Build shadow density and contrast",
            },
            "highlight_exposure": {
                "time_seconds": highlight_time,
                "filter_grade": highlight_filter,
                "purpose": "Control highlights and overall density",
            },
            "total_time": total_exposure,
            "sequence": [
                f"1. Set filter to grade {shadow_filter}",
                f"2. Expose for {shadow_time:.1f} seconds",
                f"3. Change filter to grade {highlight_filter}",
                f"4. Expose for {highlight_time:.1f} seconds",
                "5. Process normally",
            ],
        }

    @staticmethod
    def get_filter_factors() -> dict[str, float]:
        """Get standard multigrade filter factors."""
        return {
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


class VanDykeExposureCalculator:
    """Exposure calculator for Van Dyke brown prints.

    Van Dyke brown (also called brown print or sepia) uses ferric ammonium
    citrate, tartaric acid, and silver nitrate. It's a printing-out process
    with similar characteristics to cyanotype but producing brown tones.
    """

    def __init__(self):
        """Initialize Van Dyke exposure calculator."""
        # Base exposure for BL tubes
        self.base_exposure_minutes = 8.0  # Faster than cyanotype
        self.base_negative_density = 1.4  # Van Dyke works better with lower density

    def calculate(
        self,
        negative_density: float = 1.4,
        uv_source: UVSource = UVSource.BL_TUBES,
        humidity_percent: float = 50.0,
        paper_factor: float = 1.0,
    ) -> dict:
        """Calculate Van Dyke exposure time.

        Args:
            negative_density: Digital negative density range
            uv_source: UV light source
            humidity_percent: Relative humidity
            paper_factor: Paper absorbency factor

        Returns:
            Dictionary with exposure calculation results
        """
        base = self.base_exposure_minutes

        # Density adjustment
        density_delta = negative_density - self.base_negative_density
        density_factor = 2 ** (density_delta / 0.3)

        # UV source
        uv_factor = UV_SOURCE_SPEEDS.get(uv_source, 1.0)

        # Humidity (Van Dyke is sensitive)
        humidity_factor = 1.0 + (50.0 - humidity_percent) * 0.008

        exposure_minutes = base * density_factor * uv_factor * humidity_factor * paper_factor

        return {
            "exposure_minutes": round(exposure_minutes, 2),
            "exposure_seconds": round(exposure_minutes * 60, 1),
            "visual_indicators": {
                "unexposed": "Tan-orange coating",
                "properly_exposed": "Dark purple-brown",
                "overexposed": "Very dark, possible solarization",
            },
            "notes": [
                "Van Dyke exposes faster than cyanotype",
                "Watch for shift from orange to purple-brown",
                "First wash in running water until yellow clears",
                "Fix in dilute fixer (sodium thiosulfate) for 2-3 minutes",
            ],
        }


class KallitypeExposureCalculator:
    """Exposure calculator for kallitype prints.

    Kallitype uses ferric oxalate and silver nitrate, developed in various
    solutions to achieve different tones from warm brown to cool neutral.
    """

    def __init__(self):
        """Initialize kallitype exposure calculator."""
        self.base_exposure_minutes = 6.0  # Relatively fast
        self.base_negative_density = 1.5

    def calculate(
        self,
        negative_density: float = 1.5,
        uv_source: UVSource = UVSource.BL_TUBES,
        developer_type: str = "potassium_oxalate",
        humidity_percent: float = 50.0,
    ) -> dict:
        """Calculate kallitype exposure time.

        Args:
            negative_density: Digital negative density range
            uv_source: UV light source
            developer_type: Developer choice affects required exposure
            humidity_percent: Relative humidity

        Returns:
            Dictionary with exposure results
        """
        base = self.base_exposure_minutes

        # Density adjustment
        density_factor = 2 ** ((negative_density - self.base_negative_density) / 0.3)

        # UV source
        uv_factor = UV_SOURCE_SPEEDS.get(uv_source, 1.0)

        # Developer affects optimal exposure
        developer_factors = {
            "potassium_oxalate": 1.0,  # Neutral tones
            "sodium_acetate": 1.1,  # Warm brown
            "rochelle_salt": 0.95,  # Warm brown
            "borax": 1.15,  # Cool tones
        }
        dev_factor = developer_factors.get(developer_type, 1.0)

        exposure_minutes = base * density_factor * uv_factor * dev_factor

        return {
            "exposure_minutes": round(exposure_minutes, 2),
            "exposure_seconds": round(exposure_minutes * 60, 1),
            "developer": developer_type,
            "expected_tone": self._get_tone_for_developer(developer_type),
            "notes": [
                "Kallitype is a develop-out process",
                "Exposure sets the image, development reveals it",
                "Clear in citric acid bath after development",
                "Gold or platinum toning improves permanence",
            ],
        }

    def _get_tone_for_developer(self, developer: str) -> str:
        """Get expected tone color for developer type."""
        tones = {
            "potassium_oxalate": "Neutral black (similar to platinum)",
            "sodium_acetate": "Warm sepia-brown",
            "rochelle_salt": "Rich warm brown",
            "borax": "Cool blue-black",
        }
        return tones.get(developer, "Variable")
