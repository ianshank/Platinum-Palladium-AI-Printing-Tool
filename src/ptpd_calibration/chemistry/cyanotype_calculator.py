"""
Cyanotype chemistry calculator for iron-based alternative photography printing.

Calculates sensitizer solution amounts for cyanotype prints based on print dimensions,
paper type, and formula variant. Supports both classic and new cyanotype formulas.

References:
- Sir John Herschel's original 1842 formula
- Mike Ware's "New Cyanotype" formula
- Alternative Photography: Art and Technique by Christopher James

Classic Cyanotype Formula:
- Solution A: Ferric Ammonium Citrate (green) - 25% solution
- Solution B: Potassium Ferricyanide - 10% solution
- Mix 1:1 ratio just before coating

New Cyanotype Formula (Mike Ware):
- More concentrated, better Dmax, faster exposure
- Uses different iron salt ratios
"""

from dataclasses import dataclass, field
from enum import Enum

from ptpd_calibration.config import CyanotypeSettings
from ptpd_calibration.core.types import CyanotypeFormula


class CyanotypePaperType(str, Enum):
    """Paper types suitable for cyanotype printing."""

    COTTON_RAG = "cotton_rag"  # 100% cotton papers (best quality)
    WATERCOLOR_HOT = "watercolor_hot"  # Hot press watercolor paper
    WATERCOLOR_COLD = "watercolor_cold"  # Cold press watercolor paper
    FABRIC_COTTON = "fabric_cotton"  # Cotton fabric
    FABRIC_SILK = "fabric_silk"  # Silk fabric
    FABRIC_LINEN = "fabric_linen"  # Linen fabric
    MIXED_MEDIA = "mixed_media"  # Mixed media papers
    PRINTMAKING = "printmaking"  # BFK Rives, etc.


# Paper absorbency factors for cyanotype
CYANOTYPE_PAPER_FACTORS = {
    CyanotypePaperType.COTTON_RAG: 1.0,
    CyanotypePaperType.WATERCOLOR_HOT: 0.85,
    CyanotypePaperType.WATERCOLOR_COLD: 1.2,
    CyanotypePaperType.FABRIC_COTTON: 1.3,
    CyanotypePaperType.FABRIC_SILK: 0.9,
    CyanotypePaperType.FABRIC_LINEN: 1.4,
    CyanotypePaperType.MIXED_MEDIA: 1.0,
    CyanotypePaperType.PRINTMAKING: 1.1,
}


@dataclass
class CyanotypeRecipe:
    """Complete cyanotype chemistry recipe for a print.

    All measurements are provided in milliliters and drops.
    """

    # Print dimensions
    print_width_inches: float
    print_height_inches: float
    coating_width_inches: float
    coating_height_inches: float
    coating_area_sq_inches: float

    # Solution amounts in milliliters
    solution_a_ml: float  # Ferric ammonium citrate solution
    solution_b_ml: float  # Potassium ferricyanide solution
    total_sensitizer_ml: float

    # Solution amounts in drops
    solution_a_drops: float
    solution_b_drops: float
    total_drops: float

    # Formula and settings
    formula: CyanotypeFormula
    paper_type: CyanotypePaperType
    concentration_factor: float  # 1.0 = standard

    # Development info
    development_method: str
    estimated_exposure_minutes: float  # Rough estimate for sunlight

    # Cost estimate
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
            "milliliters": {
                "solution_a_fac": round(self.solution_a_ml, 2),
                "solution_b_potassium_ferricyanide": round(self.solution_b_ml, 2),
                "total": round(self.total_sensitizer_ml, 2),
            },
            "drops": {
                "solution_a": round(self.solution_a_drops, 1),
                "solution_b": round(self.solution_b_drops, 1),
                "total": round(self.total_drops, 1),
            },
            "settings": {
                "formula": self.formula.value,
                "paper_type": self.paper_type.value,
                "concentration_factor": round(self.concentration_factor, 2),
            },
            "development": {
                "method": self.development_method,
                "estimated_exposure_minutes": self.estimated_exposure_minutes,
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
            "CYANOTYPE COATING RECIPE",
            "=" * 50,
            "",
            f"Formula: {self.formula.value.title()}",
            f'Print Size: {self.print_width_inches}" x {self.print_height_inches}"',
            f'Coating Area: {self.coating_width_inches:.1f}" x {self.coating_height_inches:.1f}" '
            f"({self.coating_area_sq_inches:.1f} sq in)",
            "",
            "-" * 50,
            "SOLUTION AMOUNTS (ml / drops)",
            "-" * 50,
            "Solution A (Ferric Ammonium Citrate):",
            f"                       {self.solution_a_ml:6.2f} ml  ({self.solution_a_drops:.1f} drops)",
            "Solution B (Potassium Ferricyanide):",
            f"                       {self.solution_b_ml:6.2f} ml  ({self.solution_b_drops:.1f} drops)",
            "-" * 50,
            f"TOTAL:                 {self.total_sensitizer_ml:6.2f} ml  ({self.total_drops:.1f} drops)",
            "",
            "-" * 50,
            "PROCESSING",
            "-" * 50,
            f"Paper Type: {self.paper_type.value.replace('_', ' ').title()}",
            f"Development: {self.development_method}",
            f"Est. Exposure (sunlight): {self.estimated_exposure_minutes:.0f}-{self.estimated_exposure_minutes * 1.5:.0f} minutes",
            "",
            "-" * 50,
            "MIXING INSTRUCTIONS",
            "-" * 50,
            "1. Mix equal parts Solution A and Solution B just before coating",
            "2. Work in subdued light (not direct sunlight or UV)",
            "3. Apply evenly with brush or rod coater",
            "4. Dry in complete darkness (30-60 minutes)",
            "5. Expose under UV until deep bronze-olive color visible",
            "6. Develop in running water for 5-10 minutes",
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
                lines.append(f"* {note}")

        lines.append("=" * 50)

        return "\n".join(lines)


class CyanotypeCalculator:
    """Calculator for cyanotype sensitizer chemistry.

    Calculates the amounts of Solutions A and B needed for coating
    based on print size, paper type, and formula variant.

    Classic Cyanotype:
    - Equal parts ferric ammonium citrate (FAC) and potassium ferricyanide
    - Standard 1:1 mix ratio
    - Typical coating: 1-2ml per 8x10"

    New Cyanotype (Mike Ware):
    - Modified iron salts for better performance
    - Higher Dmax achievable
    - Faster exposure times
    """

    def __init__(self, settings: CyanotypeSettings | None = None):
        """Initialize calculator with optional custom settings.

        Args:
            settings: Custom cyanotype settings. If None, uses defaults.
        """
        self.settings = settings or CyanotypeSettings()

    def calculate(
        self,
        width_inches: float,
        height_inches: float,
        formula: CyanotypeFormula = CyanotypeFormula.CLASSIC,
        paper_type: CyanotypePaperType = CyanotypePaperType.COTTON_RAG,
        concentration_factor: float = 1.0,
        margin_inches: float | None = None,
        include_cost: bool = True,
    ) -> CyanotypeRecipe:
        """Calculate cyanotype sensitizer recipe for a given print size.

        Args:
            width_inches: Print width in inches
            height_inches: Print height in inches
            formula: Cyanotype formula variant
            paper_type: Type of paper/substrate
            concentration_factor: Multiply solution amounts (0.8-1.2 typical)
            margin_inches: Margin to subtract for coating area
            include_cost: Whether to calculate cost estimate

        Returns:
            CyanotypeRecipe with all calculated values
        """
        # Validate inputs
        if width_inches <= 0:
            raise ValueError("width must be positive")
        if height_inches <= 0:
            raise ValueError("height must be positive")
        if not 0.5 <= concentration_factor <= 2.0:
            raise ValueError("concentration factor must be between 0.5 and 2.0")
        if margin_inches is not None and margin_inches < 0:
            raise ValueError("margin cannot be negative")

        # Get margin
        margin = margin_inches if margin_inches is not None else self.settings.default_margin_inches

        # Calculate coating area
        coating_width = max(0.5, width_inches - 2 * margin)
        coating_height = max(0.5, height_inches - 2 * margin)
        coating_area = coating_width * coating_height

        # Get paper absorbency factor
        paper_factor = CYANOTYPE_PAPER_FACTORS.get(paper_type, 1.0)

        # Get formula-specific adjustments
        formula_factor = self._get_formula_factor(formula)

        # Calculate base sensitizer amount
        base_ml = coating_area * self.settings.ml_per_square_inch
        adjusted_ml = base_ml * paper_factor * formula_factor * concentration_factor

        # Split 50/50 between solutions A and B (classic cyanotype)
        solution_a_ml = adjusted_ml / 2
        solution_b_ml = adjusted_ml / 2
        total_ml = adjusted_ml

        # Convert to drops
        drops_per_ml = self.settings.drops_per_ml
        solution_a_drops = solution_a_ml * drops_per_ml
        solution_b_drops = solution_b_ml * drops_per_ml
        total_drops = total_ml * drops_per_ml

        # Calculate exposure estimate
        exposure_estimate = self._estimate_exposure(formula, paper_type, concentration_factor)

        # Calculate cost if requested
        estimated_cost = None
        if include_cost:
            estimated_cost = (
                solution_a_ml * self.settings.solution_a_cost_per_ml
                + solution_b_ml * self.settings.solution_b_cost_per_ml
            )

        # Generate notes
        notes = self._generate_notes(formula, paper_type, concentration_factor, coating_area)

        # Determine development method
        development_method = self._get_development_method(formula)

        return CyanotypeRecipe(
            print_width_inches=width_inches,
            print_height_inches=height_inches,
            coating_width_inches=coating_width,
            coating_height_inches=coating_height,
            coating_area_sq_inches=coating_area,
            solution_a_ml=solution_a_ml,
            solution_b_ml=solution_b_ml,
            total_sensitizer_ml=total_ml,
            solution_a_drops=solution_a_drops,
            solution_b_drops=solution_b_drops,
            total_drops=total_drops,
            formula=formula,
            paper_type=paper_type,
            concentration_factor=concentration_factor,
            development_method=development_method,
            estimated_exposure_minutes=exposure_estimate,
            estimated_cost_usd=estimated_cost,
            notes=notes,
        )

    def calculate_stock_solutions(
        self,
        total_volume_ml: float = 100.0,
        formula: CyanotypeFormula = CyanotypeFormula.CLASSIC,
    ) -> dict:
        """Calculate amounts for making stock solutions.

        Args:
            total_volume_ml: Desired total volume of each solution
            formula: Formula variant

        Returns:
            Dictionary with solution preparation instructions

        Raises:
            ValueError: If volume is invalid
        """
        if total_volume_ml <= 0:
            raise ValueError("total volume must be positive")

        if formula == CyanotypeFormula.CLASSIC:
            return {
                "solution_a": {
                    "name": "Ferric Ammonium Citrate (Green)",
                    "chemical_grams": total_volume_ml * 0.25,  # 25% solution
                    "water_ml": total_volume_ml,
                    "instructions": "Dissolve FAC in warm distilled water, then add water to reach total volume",
                },
                "solution_b": {
                    "name": "Potassium Ferricyanide",
                    "chemical_grams": total_volume_ml * 0.10,  # 10% solution
                    "water_ml": total_volume_ml,
                    "instructions": "Dissolve in distilled water at room temperature",
                },
                "shelf_life": "Solutions keep 6+ months in brown bottles away from light",
                "mixing": "Mix equal parts A and B immediately before coating",
            }
        elif formula == CyanotypeFormula.NEW:
            # Mike Ware's New Cyanotype formula
            return {
                "solution_a": {
                    "name": "Ammonium Iron(III) Oxalate",
                    "chemical_grams": total_volume_ml * 0.30,  # 30% solution
                    "water_ml": total_volume_ml,
                    "instructions": "Dissolve in distilled water with gentle warming",
                },
                "solution_b": {
                    "name": "Potassium Ferricyanide",
                    "chemical_grams": total_volume_ml * 0.10,  # 10% solution
                    "water_ml": total_volume_ml,
                    "instructions": "Dissolve in distilled water at room temperature",
                },
                "shelf_life": "Solutions keep 3-6 months in brown bottles",
                "mixing": "Mix 3 parts A to 1 part B for deeper blues",
            }
        else:
            return self.calculate_stock_solutions(total_volume_ml, CyanotypeFormula.CLASSIC)

    def _get_formula_factor(self, formula: CyanotypeFormula) -> float:
        """Get adjustment factor for different formula variants."""
        factors = {
            CyanotypeFormula.CLASSIC: 1.0,
            CyanotypeFormula.NEW: 0.85,  # New formula is more efficient
            CyanotypeFormula.WARE: 0.9,
            CyanotypeFormula.REX: 1.1,
        }
        return factors.get(formula, 1.0)

    def _estimate_exposure(
        self,
        formula: CyanotypeFormula,
        paper_type: CyanotypePaperType,
        concentration: float,
    ) -> float:
        """Estimate exposure time in direct sunlight."""
        base = self.settings.base_sunlight_exposure_minutes

        # Formula adjustments
        formula_factors = {
            CyanotypeFormula.CLASSIC: 1.0,
            CyanotypeFormula.NEW: 0.7,  # Faster
            CyanotypeFormula.WARE: 0.75,
            CyanotypeFormula.REX: 0.9,
        }
        formula_factor = formula_factors.get(formula, 1.0)

        # Paper adjustments (more absorbent = slightly longer)
        paper_factor = CYANOTYPE_PAPER_FACTORS.get(paper_type, 1.0)

        # Concentration adjustment
        conc_factor = 1.0 / concentration  # More concentrated = slightly faster

        return float(base * formula_factor * (paper_factor**0.5) * conc_factor)

    def _get_development_method(self, formula: CyanotypeFormula) -> str:
        """Get recommended development method for formula."""
        if formula == CyanotypeFormula.NEW:
            return "Running water 10-15 min, optional citric acid bath for deeper blues"
        elif formula == CyanotypeFormula.WARE:
            return "Acidified water bath (1% citric acid), then running water rinse"
        else:
            return "Running water 5-10 minutes until yellow stain clears"

    def _generate_notes(
        self,
        formula: CyanotypeFormula,
        paper_type: CyanotypePaperType,
        concentration: float,
        area: float,
    ) -> list[str]:
        """Generate helpful notes for the recipe."""
        notes = []

        # Formula notes
        if formula == CyanotypeFormula.CLASSIC:
            notes.append("Classic formula produces characteristic Prussian blue tones")
        elif formula == CyanotypeFormula.NEW:
            notes.append("New Cyanotype formula provides deeper Dmax and faster exposure")
            notes.append("Exposure indicated by shift from yellow-green to blue-grey")

        # Paper notes
        if paper_type in [CyanotypePaperType.FABRIC_COTTON, CyanotypePaperType.FABRIC_LINEN]:
            notes.append("Pre-wash fabric with washing soda to remove sizing")
            notes.append("Iron fabric while damp for smooth coating surface")
        elif paper_type == CyanotypePaperType.WATERCOLOR_COLD:
            notes.append("Cold press texture may show in final print - embrace or use hot press")

        # Concentration notes
        if concentration < 0.9:
            notes.append("Reduced concentration may result in lighter prints")
        elif concentration > 1.1:
            notes.append("Higher concentration increases Dmax but watch for bronzing")

        # Size notes
        if area > 100:
            notes.append("Large print - coat quickly and evenly before solution absorbs")
        if area < 20:
            notes.append("Small print - use glass rod coater for even coverage")

        # General tips
        notes.append("Store coated paper in light-tight container - usable for several days")
        notes.append("For deeper blues, try hydrogen peroxide oxidation after development")

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

    @staticmethod
    def get_troubleshooting_guide() -> dict[str, str]:
        """Get troubleshooting guide for common cyanotype issues."""
        return {
            "weak_blues": "Increase exposure time or use fresh chemicals. Try hydrogen peroxide bath.",
            "yellow_stain": "Extend washing time or use slightly acidic water for final rinse.",
            "uneven_coating": "Work faster, use less solution, or try rod coating instead of brush.",
            "bronzing": "Reduce sensitizer concentration or exposure time.",
            "fading": "Ensure complete oxidation; store prints away from alkaline materials.",
            "blotchy_highlights": "Coat more evenly; ensure paper is dry before exposure.",
            "muddy_shadows": "Negative may be too dense; try shorter exposure.",
            "posterization": "Use digital negative with proper curve for cyanotype.",
        }
