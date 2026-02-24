"""
Silver gelatin darkroom chemistry calculator for traditional photographic printing.

Calculates processing chemistry volumes and times for silver gelatin papers
in the traditional darkroom workflow. Since silver gelatin papers are pre-sensitized,
this calculator focuses on development, stop bath, fixer, and washing chemistry.

References:
- The Darkroom Cookbook by Steve Anchell
- Way Beyond Monochrome by Ralph Lambrecht
- Kodak Professional Handbook of Photography

Standard Workflow:
1. Developer (1-3 minutes depending on paper/developer)
2. Stop Bath (30 seconds)
3. Fixer (5-10 minutes for FB, 2-5 minutes for RC)
4. Wash (1 hour for FB with hypo clear, 4-5 minutes for RC)
5. Optional: Toner
"""

from dataclasses import dataclass, field
from enum import Enum

from ptpd_calibration.config import SilverGelatinSettings
from ptpd_calibration.core.types import (
    DeveloperType,
    FixerType,
    PaperBase,
)

# Standard development times (seconds) for different developers at 20C
DEFAULT_DEVELOPER_TIMES = {
    DeveloperType.DEKTOL: 90,
    DeveloperType.D_72: 90,
    DeveloperType.SELECTOL: 120,
    DeveloperType.SELECTOL_SOFT: 150,
    DeveloperType.ILFORD_MULTIGRADE: 60,
    DeveloperType.ETHOL_LPD: 90,
    DeveloperType.ANSCO_130: 120,
    DeveloperType.AMIDOL: 120,
}


class DilutionRatio(str, Enum):
    """Common developer dilution ratios."""

    STOCK = "stock"  # 1:0 (undiluted)
    ONE_TO_ONE = "1:1"
    ONE_TO_TWO = "1:2"
    ONE_TO_THREE = "1:3"
    ONE_TO_FOUR = "1:4"
    ONE_TO_SEVEN = "1:7"
    ONE_TO_NINE = "1:9"


# Dilution multipliers for working solution calculation
DILUTION_MULTIPLIERS = {
    DilutionRatio.STOCK: 1.0,
    DilutionRatio.ONE_TO_ONE: 0.5,
    DilutionRatio.ONE_TO_TWO: 0.333,
    DilutionRatio.ONE_TO_THREE: 0.25,
    DilutionRatio.ONE_TO_FOUR: 0.2,
    DilutionRatio.ONE_TO_SEVEN: 0.125,
    DilutionRatio.ONE_TO_NINE: 0.1,
}


class TraySize(str, Enum):
    """Standard darkroom tray sizes."""

    FIVE_BY_SEVEN = "5x7"
    EIGHT_BY_TEN = "8x10"
    ELEVEN_BY_FOURTEEN = "11x14"
    SIXTEEN_BY_TWENTY = "16x20"
    TWENTY_BY_TWENTY_FOUR = "20x24"


# Tray volumes in milliliters (minimum working volume)
TRAY_VOLUMES_ML = {
    TraySize.FIVE_BY_SEVEN: 500,
    TraySize.EIGHT_BY_TEN: 1000,
    TraySize.ELEVEN_BY_FOURTEEN: 2000,
    TraySize.SIXTEEN_BY_TWENTY: 3000,
    TraySize.TWENTY_BY_TWENTY_FOUR: 5000,
}


@dataclass
class DeveloperRecipe:
    """Developer working solution recipe."""

    developer_type: DeveloperType
    stock_ml: float
    water_ml: float
    total_ml: float
    dilution: DilutionRatio
    temperature_c: float
    temperature_f: float
    development_time_seconds: int
    capacity_prints: int  # Number of 8x10 equivalent prints
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessingChemistry:
    """Complete darkroom processing chemistry recipe."""

    # Print info
    print_size: tuple[float, float]
    paper_base: PaperBase
    tray_size: TraySize

    # Developer
    developer: DeveloperRecipe

    # Stop bath
    stop_bath_ml: float
    stop_bath_type: str  # "indicator" or "plain"
    stop_bath_time_seconds: int

    # Fixer
    fixer_type: FixerType
    fixer_ml: float
    fixer_time_seconds: int
    fixer_capacity_prints: int

    # Optional hypo clear
    hypo_clear_ml: float | None
    hypo_clear_time_seconds: int | None

    # Wash
    wash_time_minutes: int
    wash_method: str

    # Estimated costs
    estimated_cost_usd: float | None = None

    # Notes
    notes: list[str] = field(default_factory=list)

    def format_recipe(self) -> str:
        """Format recipe as human-readable text."""
        lines = [
            "=" * 60,
            "SILVER GELATIN DARKROOM PROCESSING RECIPE",
            "=" * 60,
            "",
            f'Print Size: {self.print_size[0]}" x {self.print_size[1]}"',
            f"Paper Base: {self.paper_base.value.replace('_', ' ').title()}",
            f"Tray Size: {self.tray_size.value}",
            "",
            "-" * 60,
            "DEVELOPER",
            "-" * 60,
            f"Type: {self.developer.developer_type.value.replace('_', ' ').title()}",
            f"Dilution: {self.developer.dilution.value}",
            f"Stock Solution: {self.developer.stock_ml:.0f} ml",
            f"Water: {self.developer.water_ml:.0f} ml",
            f"Total Working Solution: {self.developer.total_ml:.0f} ml",
            f"Temperature: {self.developer.temperature_c:.1f}C ({self.developer.temperature_f:.0f}F)",
            f"Development Time: {self.developer.development_time_seconds // 60}:{self.developer.development_time_seconds % 60:02d}",
            f"Capacity: ~{self.developer.capacity_prints} prints (8x10 equivalent)",
            "",
            "-" * 60,
            "STOP BATH",
            "-" * 60,
            f"Type: {self.stop_bath_type.title()}",
            f"Volume: {self.stop_bath_ml:.0f} ml",
            f"Time: {self.stop_bath_time_seconds} seconds",
            "",
            "-" * 60,
            "FIXER",
            "-" * 60,
            f"Type: {self.fixer_type.value.replace('_', ' ').title()}",
            f"Volume: {self.fixer_ml:.0f} ml",
            f"Time: {self.fixer_time_seconds // 60}:{self.fixer_time_seconds % 60:02d}",
            f"Capacity: ~{self.fixer_capacity_prints} prints",
        ]

        if self.hypo_clear_ml and self.hypo_clear_time_seconds is not None:
            minutes = self.hypo_clear_time_seconds // 60
            seconds = self.hypo_clear_time_seconds % 60
            lines.extend(
                [
                    "",
                    "-" * 60,
                    "HYPO CLEAR / WASH AID",
                    "-" * 60,
                    f"Volume: {self.hypo_clear_ml:.0f} ml",
                    f"Time: {minutes}:{seconds:02d}",
                ]
            )

        lines.extend(
            [
                "",
                "-" * 60,
                "WASH",
                "-" * 60,
                f"Method: {self.wash_method}",
                f"Time: {self.wash_time_minutes} minutes",
            ]
        )

        if self.estimated_cost_usd:
            lines.extend(
                [
                    "",
                    "-" * 60,
                    f"ESTIMATED CHEMISTRY COST: ${self.estimated_cost_usd:.2f} USD",
                ]
            )

        if self.notes:
            lines.extend(
                [
                    "",
                    "-" * 60,
                    "NOTES",
                    "-" * 60,
                ]
            )
            for note in self.notes:
                lines.append(f"* {note}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "print_size": {
                "width": self.print_size[0],
                "height": self.print_size[1],
            },
            "paper_base": self.paper_base.value,
            "tray_size": self.tray_size.value,
            "developer": {
                "type": self.developer.developer_type.value,
                "dilution": self.developer.dilution.value,
                "stock_ml": self.developer.stock_ml,
                "water_ml": self.developer.water_ml,
                "total_ml": self.developer.total_ml,
                "temperature_c": self.developer.temperature_c,
                "development_time_seconds": self.developer.development_time_seconds,
                "capacity_prints": self.developer.capacity_prints,
            },
            "stop_bath": {
                "volume_ml": self.stop_bath_ml,
                "type": self.stop_bath_type,
                "time_seconds": self.stop_bath_time_seconds,
            },
            "fixer": {
                "type": self.fixer_type.value,
                "volume_ml": self.fixer_ml,
                "time_seconds": self.fixer_time_seconds,
                "capacity_prints": self.fixer_capacity_prints,
            },
            "hypo_clear": {
                "volume_ml": self.hypo_clear_ml,
                "time_seconds": self.hypo_clear_time_seconds,
            }
            if self.hypo_clear_ml
            else None,
            "wash": {
                "method": self.wash_method,
                "time_minutes": self.wash_time_minutes,
            },
            "estimated_cost_usd": self.estimated_cost_usd,
            "notes": self.notes,
        }


class SilverGelatinCalculator:
    """Calculator for silver gelatin darkroom processing chemistry.

    Since silver gelatin papers are pre-sensitized, this calculator focuses
    on calculating the proper volumes and times for development, stop bath,
    fixer, and washing chemistry.
    """

    def __init__(self, settings: SilverGelatinSettings | None = None):
        """Initialize calculator with optional custom settings.

        Args:
            settings: Custom settings. If None, uses defaults.
        """
        self.settings = settings or SilverGelatinSettings()

    def calculate(
        self,
        width_inches: float,
        height_inches: float,
        paper_base: PaperBase = PaperBase.FIBER,
        developer: DeveloperType | None = None,
        dilution: DilutionRatio | None = None,
        temperature_c: float | None = None,
        fixer: FixerType = FixerType.SODIUM_THIOSULFATE,
        include_hypo_clear: bool = True,
        tray_size: TraySize | None = None,
        num_prints: int = 1,
        include_cost: bool = True,
    ) -> ProcessingChemistry:
        """Calculate processing chemistry for a silver gelatin printing session.

        Args:
            width_inches: Print width
            height_inches: Print height
            paper_base: Paper base type (fiber or RC)
            developer: Developer type to use
            dilution: Developer dilution ratio
            temperature_c: Developer temperature in Celsius
            fixer: Fixer type
            include_hypo_clear: Whether to include hypo clear for FB paper
            tray_size: Tray size (auto-selected if None)
            num_prints: Expected number of prints in session
            include_cost: Whether to calculate cost estimate

        Returns:
            ProcessingChemistry with all chemistry calculations

        Raises:
            ValueError: If inputs are invalid
        """
        if width_inches <= 0:
            raise ValueError("width must be positive")
        if height_inches <= 0:
            raise ValueError("height must be positive")
        if num_prints <= 0:
            raise ValueError("number of prints must be positive")
        # Use defaults if not specified
        # Use defaults if not specified
        if developer is None:
            # Handle string vs Enum if coming from config
            dev_setting = self.settings.default_developer
            developer = DeveloperType(dev_setting) if isinstance(dev_setting, str) else dev_setting

        if dilution is None:
            dil_setting = self.settings.default_dilution
            dilution = DilutionRatio(dil_setting) if isinstance(dil_setting, str) else dil_setting

        temperature_c = temperature_c or self.settings.default_temperature_c

        # Auto-select tray size if not specified
        if tray_size is None:
            tray_size = self._select_tray_size(width_inches, height_inches)

        # Get tray volume
        tray_volume_ml = TRAY_VOLUMES_ML[tray_size]

        # Calculate developer recipe
        developer_recipe = self._calculate_developer(
            developer, dilution, tray_volume_ml, temperature_c, num_prints
        )

        # Calculate stop bath
        stop_bath_ml = tray_volume_ml

        # Calculate fixer
        fixer_time = (
            self.settings.fixer_time_fb_seconds
            if paper_base == PaperBase.FIBER
            else self.settings.fixer_time_rc_seconds
        )
        fixer_capacity = self._calculate_fixer_capacity(tray_volume_ml, paper_base)

        # Hypo clear (only for fiber base)
        hypo_clear_ml = None
        hypo_clear_time = None
        if include_hypo_clear and paper_base == PaperBase.FIBER:
            hypo_clear_ml = tray_volume_ml
            hypo_clear_time = 120  # 2 minutes

        # Calculate wash time and method
        wash_time, wash_method = self._calculate_wash(paper_base, include_hypo_clear)

        # Calculate cost
        estimated_cost = None
        if include_cost:
            estimated_cost = self._calculate_cost(
                developer_recipe.total_ml,
                stop_bath_ml,
                tray_volume_ml,
                hypo_clear_ml,
            )

        # Generate notes
        notes = self._generate_notes(developer, paper_base, temperature_c, num_prints)

        return ProcessingChemistry(
            print_size=(width_inches, height_inches),
            paper_base=paper_base,
            tray_size=tray_size,
            developer=developer_recipe,
            stop_bath_ml=stop_bath_ml,
            stop_bath_type="indicator" if paper_base == PaperBase.FIBER else "plain",
            stop_bath_time_seconds=30,
            fixer_type=fixer,
            fixer_ml=tray_volume_ml,
            fixer_time_seconds=fixer_time,
            fixer_capacity_prints=fixer_capacity,
            hypo_clear_ml=hypo_clear_ml,
            hypo_clear_time_seconds=hypo_clear_time,
            wash_time_minutes=wash_time,
            wash_method=wash_method,
            estimated_cost_usd=estimated_cost,
            notes=notes,
        )

    def calculate_test_strip_times(
        self,
        base_exposure_seconds: float,
        num_strips: int = 5,
        increment_factor: float = 1.4,  # Half-stop increments
    ) -> list[float]:
        """Calculate exposure times for test strips.

        Args:
            base_exposure_seconds: Starting exposure time
            num_strips: Number of test strip segments
            increment_factor: Multiplier between strips (1.4 = 1/2 stop)

        Returns:
            List of exposure times
        """
        if base_exposure_seconds <= 0:
            raise ValueError("base exposure must be positive")
        if num_strips <= 0:
            raise ValueError("number of strips must be positive")

        times = []
        half_strips = num_strips // 2

        for i in range(-half_strips, num_strips - half_strips):
            factor = increment_factor**i
            times.append(base_exposure_seconds * factor)

        return times

    def calculate_split_filter_exposure(
        self,
        base_exposure_seconds: float,
        shadow_grade: float = 5.0,
        highlight_grade: float = 0.0,
        split_ratio: float = 0.5,
    ) -> dict:
        """Calculate split-filter (split-grade) exposure times.

        Split filtering uses separate exposures through different contrast
        filters to achieve precise tonal control.

        Args:
            base_exposure_seconds: Total exposure time
            shadow_grade: Contrast grade for shadows (typically 4-5)
            highlight_grade: Contrast grade for highlights (typically 0-1)
            split_ratio: Ratio of shadow to total exposure

        Returns:
            Dictionary with shadow and highlight exposure times
        """
        if base_exposure_seconds <= 0:
            raise ValueError("base exposure must be positive")
        if not (0 <= shadow_grade <= 5):
            raise ValueError("shadow grade must be between 0 and 5")
        if not (0 <= highlight_grade <= 5):
            raise ValueError("highlight grade must be between 0 and 5")
        if not (0 <= split_ratio <= 1):
            raise ValueError("split ratio must be between 0 and 1")

        shadow_time = base_exposure_seconds * split_ratio
        highlight_time = base_exposure_seconds * (1 - split_ratio)

        return {
            "shadow_exposure": {
                "time_seconds": shadow_time,
                "grade": shadow_grade,
                "filter": self._grade_to_filter(shadow_grade),
            },
            "highlight_exposure": {
                "time_seconds": highlight_time,
                "grade": highlight_grade,
                "filter": self._grade_to_filter(highlight_grade),
            },
            "total_time": base_exposure_seconds,
            "notes": [
                "Expose shadows first with high contrast filter",
                "Then expose highlights with low contrast filter",
                "Adjust split_ratio to control overall contrast",
            ],
        }

    def _calculate_developer(
        self,
        developer: DeveloperType,
        dilution: DilutionRatio,
        volume_ml: float,
        temperature_c: float,
        num_prints: int,
    ) -> DeveloperRecipe:
        """Calculate developer working solution recipe."""
        # Get dilution factor
        stock_ratio = DILUTION_MULTIPLIERS.get(dilution, 0.5)
        stock_ml = volume_ml * stock_ratio
        water_ml = volume_ml - stock_ml

        # Get development time
        base_time = DEFAULT_DEVELOPER_TIMES.get(developer, 90)

        # Adjust for temperature (10% change per degree from 20C)
        temp_factor = 1.0 + (20.0 - temperature_c) * 0.1
        adjusted_time = int(base_time * temp_factor)

        # Calculate capacity (approximately 10-15 8x10 prints per liter)
        capacity = int((volume_ml / 1000) * 12)

        # Convert temperature to Fahrenheit
        temperature_f = temperature_c * 9 / 5 + 32

        notes = []
        if temperature_c != 20.0:
            notes.append(f"Time adjusted for {temperature_c}C (standard is 20C)")
        if dilution in [DilutionRatio.ONE_TO_THREE, DilutionRatio.ONE_TO_FOUR]:
            notes.append("Higher dilution gives softer results with longer times")

        return DeveloperRecipe(
            developer_type=developer,
            stock_ml=stock_ml,
            water_ml=water_ml,
            total_ml=volume_ml,
            dilution=dilution,
            temperature_c=temperature_c,
            temperature_f=temperature_f,
            development_time_seconds=adjusted_time,
            capacity_prints=capacity,
            notes=notes,
        )

    def _select_tray_size(self, width: float, height: float) -> TraySize:
        """Select appropriate tray size for print dimensions."""
        # Add margin for handling
        needed_width = width + 2
        needed_height = height + 2

        # Select smallest tray that fits
        sizes = [
            (TraySize.FIVE_BY_SEVEN, 5, 7),
            (TraySize.EIGHT_BY_TEN, 8, 10),
            (TraySize.ELEVEN_BY_FOURTEEN, 11, 14),
            (TraySize.SIXTEEN_BY_TWENTY, 16, 20),
            (TraySize.TWENTY_BY_TWENTY_FOUR, 20, 24),
        ]

        for tray, tw, th in sizes:
            if needed_width <= tw and needed_height <= th:
                return tray
            # Also check rotated
            if needed_width <= th and needed_height <= tw:
                return tray

        # Default to largest
        return TraySize.TWENTY_BY_TWENTY_FOUR

    def _calculate_fixer_capacity(self, volume_ml: float, paper_base: PaperBase) -> int:
        """Calculate fixer capacity in number of 8x10 equivalent prints."""
        # Standard fixer capacity is about 20-25 8x10s per liter for regular fix
        base_capacity = 22
        liters = volume_ml / 1000

        # RC paper exhausts fixer slightly faster
        if paper_base == PaperBase.RESIN_COATED:
            base_capacity = int(base_capacity * 0.9)

        return int(liters * base_capacity)

    def _calculate_wash(self, paper_base: PaperBase, has_hypo_clear: bool) -> tuple[int, str]:
        """Calculate wash time and method based on paper type."""
        if paper_base == PaperBase.RESIN_COATED:
            return 4, "Running water or 10 changes with agitation"
        elif has_hypo_clear:
            return 30, "Running water with print washer (archival wash)"
        else:
            return 60, "Running water with print washer (archival wash)"

    def _calculate_cost(
        self,
        developer_ml: float,
        stop_ml: float,
        fixer_ml: float,
        hypo_clear_ml: float | None,
    ) -> float:
        """Calculate estimated chemistry cost."""
        cost = (
            (developer_ml / 1000) * self.settings.developer_cost_per_liter
            + (stop_ml / 1000) * self.settings.stop_bath_cost_per_liter
            + (fixer_ml / 1000) * self.settings.fixer_cost_per_liter
        )

        if hypo_clear_ml:
            cost += (hypo_clear_ml / 1000) * self.settings.hypo_clear_cost_per_liter

        return cost

    def _grade_to_filter(self, grade: float) -> str:
        """Convert contrast grade to filter description."""
        if grade <= 0:
            return "Magenta 0 (lowest contrast)"
        elif grade <= 1:
            return "Magenta 1"
        elif grade <= 2:
            return "Magenta 2 (normal)"
        elif grade <= 3:
            return "Yellow 3"
        elif grade <= 4:
            return "Yellow 4"
        else:
            return "Yellow 5 (highest contrast)"

    def _generate_notes(
        self,
        developer: DeveloperType,
        paper_base: PaperBase,
        temperature: float,
        num_prints: int,
    ) -> list[str]:
        """Generate helpful notes for the processing session."""
        notes = []

        # Developer notes
        if developer == DeveloperType.DEKTOL:
            notes.append("Dektol produces neutral to cool black tones")
        elif developer == DeveloperType.SELECTOL:
            notes.append("Selectol produces warm brown-black tones")
        elif developer == DeveloperType.AMIDOL:
            notes.append("Amidol produces cold blue-black tones (archival favorite)")

        # Paper base notes
        if paper_base == PaperBase.FIBER:
            notes.append("Fiber paper requires thorough washing for archival permanence")
            notes.append("Consider two-bath fixing for maximum archival quality")
        else:
            notes.append("RC paper processes faster but is not considered archival")

        # Temperature notes
        if temperature < 18:
            notes.append("Low temperature may cause uneven development")
        elif temperature > 22:
            notes.append("Higher temperature speeds development - watch for fog")

        # General tips
        notes.append("Agitate continuously for first 30 seconds, then every 30 seconds")
        notes.append("Keep prints submerged - air exposure causes oxidation stains")

        return notes

    @staticmethod
    def get_developer_info() -> dict[str, dict]:
        """Get information about available developers."""
        return {
            DeveloperType.DEKTOL.value: {
                "name": "Dektol (Kodak D-72)",
                "tone": "Neutral to slightly cool",
                "contrast": "Normal",
                "typical_dilution": "1:2",
                "typical_time": "1:30 at 20C",
                "notes": "Industry standard paper developer",
            },
            DeveloperType.SELECTOL.value: {
                "name": "Selectol",
                "tone": "Warm brown-black",
                "contrast": "Normal",
                "typical_dilution": "1:1",
                "typical_time": "2:00 at 20C",
                "notes": "Good for portrait and fine art work",
            },
            DeveloperType.SELECTOL_SOFT.value: {
                "name": "Selectol-Soft",
                "tone": "Warm",
                "contrast": "Low",
                "typical_dilution": "1:1",
                "typical_time": "2:30 at 20C",
                "notes": "Reduces contrast by about 1 grade",
            },
            DeveloperType.ETHOL_LPD.value: {
                "name": "Ethol LPD",
                "tone": "Variable (warm to neutral)",
                "contrast": "Variable",
                "typical_dilution": "1:2 to 1:4",
                "typical_time": "1:30 at 20C",
                "notes": "Tone varies with dilution - more dilute = warmer",
            },
            DeveloperType.AMIDOL.value: {
                "name": "Amidol",
                "tone": "Cold blue-black",
                "contrast": "Normal to high",
                "typical_dilution": "Use fresh",
                "typical_time": "2:00 at 20C",
                "notes": "Favored by fine art printers for maximum Dmax",
            },
        }

    @staticmethod
    def get_troubleshooting_guide() -> dict[str, str]:
        """Get troubleshooting guide for common darkroom issues."""
        return {
            "flat_prints": "Increase paper grade or use harder filter. Check developer freshness.",
            "too_contrasty": "Decrease paper grade or use softer filter. Try split-grade printing.",
            "muddy_shadows": "Reduce exposure. Shadows may be blocked in negative.",
            "gray_blacks": "Extend development time. Check developer temperature and freshness.",
            "uneven_development": "Agitate more consistently. Ensure print is fully submerged.",
            "yellow_stains": "Extend fixing time or use fresh fixer. Improve washing.",
            "purple_stains": "Insufficient fixing. Use two-bath fixing for FB paper.",
            "fog": "Check for light leaks. Developer may be contaminated or too warm.",
            "blisters": "Temperature shock. Keep all solutions within 3C of each other.",
        }
