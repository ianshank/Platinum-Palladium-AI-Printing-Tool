"""
Chemistry calculation module for platinum/palladium printing.

Provides tools for calculating coating solution amounts based on print dimensions.
"""

from ptpd_calibration.chemistry.calculator import (
    METAL_MIX_RATIOS,
    ChemistryCalculator,
    ChemistryRecipe,
    CoatingMethod,
    MetalMix,
    PaperAbsorbency,
)

__all__ = [
    "ChemistryCalculator",
    "ChemistryRecipe",
    "PaperAbsorbency",
    "CoatingMethod",
    "MetalMix",
    "METAL_MIX_RATIOS",
]
