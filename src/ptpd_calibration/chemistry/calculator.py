"""
Chemistry calculator for platinum/palladium printing.

Calculates coating solution amounts based on print dimensions, paper type,
and coating method. Based on Bostick-Sullivan formulas and industry standards.

References:
- Bostick-Sullivan Platinum and Palladium Kit Instructions
- Edward Stapel Drop Chart
- Standard industry practices

Standard formula: A (FO #1) + B (FO #2 contrast) + C (metals) = total coating solution
Rule: Drops of metals (C) should equal drops of ferric oxalate (A + B)
"""

from dataclasses import dataclass, field
from enum import Enum

from ptpd_calibration.config import ChemistrySettings, get_settings


class PaperAbsorbency(str, Enum):
    """Paper absorbency levels affecting coating amounts."""

    LOW = "low"  # Hot press, sized papers
    MEDIUM = "medium"  # Standard art papers
    HIGH = "high"  # Cold press, unsized papers


class CoatingMethod(str, Enum):
    """Coating application methods."""

    BRUSH = "brush"  # Hake brush, requires more solution
    ROD = "rod"  # Glass rod, more efficient
    PUDDLE_PUSHER = "puddle_pusher"  # Similar to rod


class MetalMix(str, Enum):
    """Common metal mixture presets."""

    PURE_PALLADIUM = "pure_palladium"  # 100% Pd - warm tones, economical
    PURE_PLATINUM = "pure_platinum"  # 100% Pt - cooler tones, max Dmax
    CLASSIC_MIX = "classic_mix"  # 50% Pt / 50% Pd - balanced
    WARM_MIX = "warm_mix"  # 25% Pt / 75% Pd - warm with good blacks
    COOL_MIX = "cool_mix"  # 75% Pt / 25% Pd - cooler tones


# Metal mix presets (platinum ratio)
METAL_MIX_RATIOS = {
    MetalMix.PURE_PALLADIUM: 0.0,
    MetalMix.PURE_PLATINUM: 1.0,
    MetalMix.CLASSIC_MIX: 0.5,
    MetalMix.WARM_MIX: 0.25,
    MetalMix.COOL_MIX: 0.75,
}


@dataclass
class ChemistryRecipe:
    """Complete chemistry recipe for a print.

    All measurements are provided in both drops and milliliters.
    """

    # Print dimensions
    print_width_inches: float
    print_height_inches: float
    coating_width_inches: float
    coating_height_inches: float
    coating_area_sq_inches: float

    # Solution amounts in drops
    ferric_oxalate_drops: float
    ferric_oxalate_contrast_drops: float  # FO #2 with potassium chlorate
    palladium_drops: float
    platinum_drops: float
    na2_drops: float
    total_drops: float

    # Solution amounts in milliliters
    ferric_oxalate_ml: float
    ferric_oxalate_contrast_ml: float
    palladium_ml: float
    platinum_ml: float
    na2_ml: float
    total_ml: float

    # Metal ratios
    platinum_ratio: float
    palladium_ratio: float

    # Settings used
    paper_absorbency: PaperAbsorbency
    coating_method: CoatingMethod
    contrast_boost: float  # 0.0-1.0, ratio of FO#2 to total FO

    # Cost estimate (if enabled)
    estimated_cost_usd: float | None = None

    # Notes and recommendations
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert recipe to dictionary for JSON serialization."""
        return {
            "print_dimensions": {
                "width_inches": self.print_width_inches,
                "height_inches": self.print_height_inches,
                "coating_width_inches": self.coating_width_inches,
                "coating_height_inches": self.coating_height_inches,
                "coating_area_sq_inches": round(self.coating_area_sq_inches, 2),
            },
            "drops": {
                "ferric_oxalate_1": round(self.ferric_oxalate_drops, 1),
                "ferric_oxalate_2_contrast": round(self.ferric_oxalate_contrast_drops, 1),
                "palladium": round(self.palladium_drops, 1),
                "platinum": round(self.platinum_drops, 1),
                "na2": round(self.na2_drops, 1),
                "total": round(self.total_drops, 1),
            },
            "milliliters": {
                "ferric_oxalate_1": round(self.ferric_oxalate_ml, 2),
                "ferric_oxalate_2_contrast": round(self.ferric_oxalate_contrast_ml, 2),
                "palladium": round(self.palladium_ml, 2),
                "platinum": round(self.platinum_ml, 2),
                "na2": round(self.na2_ml, 2),
                "total": round(self.total_ml, 2),
            },
            "metal_ratios": {
                "platinum_percent": round(self.platinum_ratio * 100, 1),
                "palladium_percent": round(self.palladium_ratio * 100, 1),
            },
            "settings": {
                "paper_absorbency": self.paper_absorbency.value,
                "coating_method": self.coating_method.value,
                "contrast_boost": round(self.contrast_boost * 100, 1),
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 2)
            if self.estimated_cost_usd
            else None,
            "notes": self.notes,
        }

    def format_recipe(self) -> str:
        """Format recipe as human-readable text."""
        lines = [
            "=" * 50,
            "PLATINUM/PALLADIUM COATING RECIPE",
            "=" * 50,
            "",
            f'Print Size: {self.print_width_inches}" x {self.print_height_inches}"',
            f'Coating Area: {self.coating_width_inches:.1f}" x {self.coating_height_inches:.1f}" '
            f"({self.coating_area_sq_inches:.1f} sq in)",
            "",
            "-" * 50,
            "SOLUTION AMOUNTS (Drops / ml)",
            "-" * 50,
            f"Ferric Oxalate #1:     {self.ferric_oxalate_drops:5.1f} drops  ({self.ferric_oxalate_ml:.2f} ml)",
            f"Ferric Oxalate #2:     {self.ferric_oxalate_contrast_drops:5.1f} drops  ({self.ferric_oxalate_contrast_ml:.2f} ml)",
            f"Palladium:             {self.palladium_drops:5.1f} drops  ({self.palladium_ml:.2f} ml)",
            f"Platinum:              {self.platinum_drops:5.1f} drops  ({self.platinum_ml:.2f} ml)",
            f"Na2 (contrast agent):  {self.na2_drops:5.1f} drops  ({self.na2_ml:.2f} ml)",
            "-" * 50,
            f"TOTAL:                 {self.total_drops:5.1f} drops  ({self.total_ml:.2f} ml)",
            "",
            "-" * 50,
            "METAL RATIO",
            "-" * 50,
            f"Platinum: {self.platinum_ratio * 100:.0f}%  |  Palladium: {self.palladium_ratio * 100:.0f}%",
            "",
            "-" * 50,
            "SETTINGS",
            "-" * 50,
            f"Paper Absorbency: {self.paper_absorbency.value.title()}",
            f"Coating Method: {self.coating_method.value.title()}",
            f"Contrast Boost: {self.contrast_boost * 100:.0f}%",
        ]

        if self.estimated_cost_usd is not None:
            lines.extend(
                [
                    "",
                    "-" * 50,
                    f"ESTIMATED COST: ${self.estimated_cost_usd:.2f} USD",
                ]
            )

        if self.notes:
            lines.extend(
                [
                    "",
                    "-" * 50,
                    "NOTES",
                    "-" * 50,
                ]
            )
            for note in self.notes:
                lines.append(f"• {note}")

        lines.append("=" * 50)

        return "\n".join(lines)


class ChemistryCalculator:
    """Calculator for platinum/palladium coating chemistry.

    Based on Bostick-Sullivan formulas:
    - Standard 8x10" coating uses ~46 drops total
    - Rule: Metal drops = Ferric Oxalate drops
    - Na2 adds contrast (typically ~25% of metal drops)
    """

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with optional custom settings.

        Args:
            settings: Custom chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def calculate(
        self,
        width_inches: float,
        height_inches: float,
        platinum_ratio: float = 0.0,
        paper_absorbency: PaperAbsorbency = PaperAbsorbency.MEDIUM,
        coating_method: CoatingMethod = CoatingMethod.BRUSH,
        contrast_boost: float = 0.0,
        na2_ratio: float | None = None,
        margin_inches: float | None = None,
        include_cost: bool = True,
    ) -> ChemistryRecipe:
        """Calculate chemistry recipe for a given print size.

        Args:
            width_inches: Print width in inches
            height_inches: Print height in inches
            platinum_ratio: Ratio of platinum to total metal (0.0-1.0)
            paper_absorbency: Paper absorbency level
            coating_method: Coating application method
            contrast_boost: Ratio of FO#2 to total FO (0.0-1.0)
            na2_ratio: Na2 drops as ratio of metal drops (default from settings)
            margin_inches: Margin to subtract for coating area (default from settings)
            include_cost: Whether to calculate cost estimate

        Returns:
            ChemistryRecipe with all calculated values
        """
        # Validate inputs
        if width_inches <= 0 or height_inches <= 0:
            raise ValueError("Print dimensions must be positive")
        if not 0.0 <= platinum_ratio <= 1.0:
            raise ValueError("Platinum ratio must be between 0.0 and 1.0")
        if not 0.0 <= contrast_boost <= 1.0:
            raise ValueError("Contrast boost must be between 0.0 and 1.0")

        # Get margin
        margin = margin_inches if margin_inches is not None else self.settings.default_margin_inches

        # Calculate coating area
        coating_width = max(0.5, width_inches - 2 * margin)
        coating_height = max(0.5, height_inches - 2 * margin)
        coating_area = coating_width * coating_height

        # Get absorbency multiplier
        absorbency_multiplier = {
            PaperAbsorbency.LOW: self.settings.low_absorbency_multiplier,
            PaperAbsorbency.MEDIUM: self.settings.medium_absorbency_multiplier,
            PaperAbsorbency.HIGH: self.settings.high_absorbency_multiplier,
        }[paper_absorbency]

        # Get coating method multiplier
        method_multiplier = {
            CoatingMethod.BRUSH: self.settings.brush_coating_multiplier,
            CoatingMethod.ROD: self.settings.rod_coating_multiplier,
            CoatingMethod.PUDDLE_PUSHER: self.settings.rod_coating_multiplier,
        }[coating_method]

        # Calculate base drops
        base_drops = coating_area * self.settings.drops_per_square_inch
        adjusted_drops = base_drops * absorbency_multiplier * method_multiplier

        # Split into ferric oxalate and metals (equal amounts per standard formula)
        ferric_oxalate_total = adjusted_drops / 2
        metal_total = adjusted_drops / 2

        # Split ferric oxalate by contrast boost
        fo_contrast_drops = ferric_oxalate_total * contrast_boost
        fo_standard_drops = ferric_oxalate_total - fo_contrast_drops

        # Split metals by platinum ratio
        platinum_drops = metal_total * platinum_ratio
        palladium_drops = metal_total * (1 - platinum_ratio)

        # Calculate Na2
        na2_ratio_value = (
            na2_ratio if na2_ratio is not None else self.settings.default_na2_drops_ratio
        )
        na2_drops = metal_total * na2_ratio_value

        # Total drops
        total_drops = (
            fo_standard_drops + fo_contrast_drops + palladium_drops + platinum_drops + na2_drops
        )

        # Convert to milliliters
        drops_per_ml = self.settings.drops_per_ml
        fo_standard_ml = fo_standard_drops / drops_per_ml
        fo_contrast_ml = fo_contrast_drops / drops_per_ml
        palladium_ml = palladium_drops / drops_per_ml
        platinum_ml = platinum_drops / drops_per_ml
        na2_ml = na2_drops / drops_per_ml
        total_ml = total_drops / drops_per_ml

        # Calculate cost
        estimated_cost = None
        if include_cost:
            estimated_cost = (
                (fo_standard_ml + fo_contrast_ml) * self.settings.ferric_oxalate_cost_per_ml
                + palladium_ml * self.settings.palladium_cost_per_ml
                + platinum_ml * self.settings.platinum_cost_per_ml
                + na2_ml * self.settings.na2_cost_per_ml
            )

        # Generate notes
        notes = self._generate_notes(
            width_inches,
            height_inches,
            platinum_ratio,
            paper_absorbency,
            coating_method,
            contrast_boost,
        )

        return ChemistryRecipe(
            print_width_inches=width_inches,
            print_height_inches=height_inches,
            coating_width_inches=coating_width,
            coating_height_inches=coating_height,
            coating_area_sq_inches=coating_area,
            ferric_oxalate_drops=fo_standard_drops,
            ferric_oxalate_contrast_drops=fo_contrast_drops,
            palladium_drops=palladium_drops,
            platinum_drops=platinum_drops,
            na2_drops=na2_drops,
            total_drops=total_drops,
            ferric_oxalate_ml=fo_standard_ml,
            ferric_oxalate_contrast_ml=fo_contrast_ml,
            palladium_ml=palladium_ml,
            platinum_ml=platinum_ml,
            na2_ml=na2_ml,
            total_ml=total_ml,
            platinum_ratio=platinum_ratio,
            palladium_ratio=1 - platinum_ratio,
            paper_absorbency=paper_absorbency,
            coating_method=coating_method,
            contrast_boost=contrast_boost,
            estimated_cost_usd=estimated_cost,
            notes=notes,
        )

    def calculate_from_preset(
        self,
        width_inches: float,
        height_inches: float,
        metal_mix: MetalMix = MetalMix.PURE_PALLADIUM,
        paper_absorbency: PaperAbsorbency = PaperAbsorbency.MEDIUM,
        coating_method: CoatingMethod = CoatingMethod.BRUSH,
        contrast_boost: float = 0.0,
    ) -> ChemistryRecipe:
        """Calculate using a metal mix preset.

        Args:
            width_inches: Print width in inches
            height_inches: Print height in inches
            metal_mix: Preset metal mixture
            paper_absorbency: Paper absorbency level
            coating_method: Coating application method
            contrast_boost: Ratio of FO#2 to total FO (0.0-1.0)

        Returns:
            ChemistryRecipe with all calculated values
        """
        platinum_ratio = METAL_MIX_RATIOS[metal_mix]
        return self.calculate(
            width_inches=width_inches,
            height_inches=height_inches,
            platinum_ratio=platinum_ratio,
            paper_absorbency=paper_absorbency,
            coating_method=coating_method,
            contrast_boost=contrast_boost,
        )

    def scale_recipe(
        self,
        recipe: ChemistryRecipe,
        scale_factor: float,
    ) -> ChemistryRecipe:
        """Scale an existing recipe by a factor.

        Useful for batch printing or making test strips.

        Args:
            recipe: Original recipe
            scale_factor: Scaling factor (e.g., 0.5 for half, 2.0 for double)

        Returns:
            Scaled ChemistryRecipe
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive")

        drops_per_ml = self.settings.drops_per_ml

        scaled_fo_drops = recipe.ferric_oxalate_drops * scale_factor
        scaled_fo_contrast_drops = recipe.ferric_oxalate_contrast_drops * scale_factor
        scaled_pd_drops = recipe.palladium_drops * scale_factor
        scaled_pt_drops = recipe.platinum_drops * scale_factor
        scaled_na2_drops = recipe.na2_drops * scale_factor
        scaled_total_drops = recipe.total_drops * scale_factor

        notes = list(recipe.notes)
        notes.append(
            f'Scaled {scale_factor}x from original {recipe.print_width_inches}"x{recipe.print_height_inches}" recipe'
        )

        return ChemistryRecipe(
            print_width_inches=recipe.print_width_inches,
            print_height_inches=recipe.print_height_inches,
            coating_width_inches=recipe.coating_width_inches,
            coating_height_inches=recipe.coating_height_inches,
            coating_area_sq_inches=recipe.coating_area_sq_inches,
            ferric_oxalate_drops=scaled_fo_drops,
            ferric_oxalate_contrast_drops=scaled_fo_contrast_drops,
            palladium_drops=scaled_pd_drops,
            platinum_drops=scaled_pt_drops,
            na2_drops=scaled_na2_drops,
            total_drops=scaled_total_drops,
            ferric_oxalate_ml=scaled_fo_drops / drops_per_ml,
            ferric_oxalate_contrast_ml=scaled_fo_contrast_drops / drops_per_ml,
            palladium_ml=scaled_pd_drops / drops_per_ml,
            platinum_ml=scaled_pt_drops / drops_per_ml,
            na2_ml=scaled_na2_drops / drops_per_ml,
            total_ml=scaled_total_drops / drops_per_ml,
            platinum_ratio=recipe.platinum_ratio,
            palladium_ratio=recipe.palladium_ratio,
            paper_absorbency=recipe.paper_absorbency,
            coating_method=recipe.coating_method,
            contrast_boost=recipe.contrast_boost,
            estimated_cost_usd=(recipe.estimated_cost_usd * scale_factor)
            if recipe.estimated_cost_usd
            else None,
            notes=notes,
        )

    def _generate_notes(
        self,
        width: float,
        height: float,
        pt_ratio: float,
        absorbency: PaperAbsorbency,
        method: CoatingMethod,
        contrast: float,
    ) -> list[str]:
        """Generate helpful notes for the recipe."""
        notes = []

        # Metal ratio notes
        if pt_ratio == 0.0:
            notes.append("Pure palladium produces warm, brown-black tones")
        elif pt_ratio == 1.0:
            notes.append("Pure platinum produces cooler, neutral black tones with maximum Dmax")
        elif pt_ratio >= 0.5:
            notes.append("Higher platinum ratio gives cooler tones and deeper blacks")
        else:
            notes.append("Higher palladium ratio gives warmer tones and finer grain")

        # Paper absorbency notes
        if absorbency == PaperAbsorbency.HIGH:
            notes.append(
                "Cold press paper may need additional solution - adjust if coating appears thin"
            )
        elif absorbency == PaperAbsorbency.LOW:
            notes.append("Hot press paper coats efficiently - watch for pooling")

        # Coating method notes
        if method == CoatingMethod.BRUSH:
            notes.append("Use a clean hake brush and work quickly before solution absorbs")
        else:
            notes.append("Glass rod coating is more economical - use swift, even strokes")

        # Contrast notes
        if contrast > 0:
            notes.append(
                f"FO#2 adds {contrast * 100:.0f}% contrast boost - good for flat negatives"
            )

        # Size notes
        area = width * height
        if area < 20:
            notes.append("Small print - round drops up rather than down")
        elif area > 100:
            notes.append("Large print - consider making in batches for consistent coating")

        return notes

    @staticmethod
    def get_standard_sizes() -> dict[str, tuple[float, float]]:
        """Get dictionary of standard print sizes in inches."""
        return {
            "4x5": (4, 5),
            "5x7": (5, 7),
            "8x10": (8, 10),
            "11x14": (11, 14),
            "16x20": (16, 20),
            "20x24": (20, 24),
        }
