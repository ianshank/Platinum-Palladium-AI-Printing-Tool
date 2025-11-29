"""
Chemistry calculation module for platinum/palladium printing.

Provides tools for calculating coating solution amounts based on print dimensions.
"""

from ptpd_calibration.chemistry.calculator import (
    ChemistryCalculator,
    ChemistryRecipe,
    PaperAbsorbency,
    CoatingMethod,
    MetalMix,
    METAL_MIX_RATIOS,
)

__all__ = [
    "ChemistryCalculator",
    "ChemistryRecipe",
    "PaperAbsorbency",
    "CoatingMethod",
    "MetalMix",
    "METAL_MIX_RATIOS",
]
