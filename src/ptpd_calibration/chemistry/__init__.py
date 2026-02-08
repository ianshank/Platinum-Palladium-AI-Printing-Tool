"""
Chemistry calculation module for alternative photographic printing processes.

Provides tools for calculating coating solution and processing chemistry amounts
for various alternative photography processes including:
- Platinum/Palladium printing
- Cyanotype (iron-based blue process)
- Silver Gelatin (traditional darkroom)
"""

from ptpd_calibration.chemistry.calculator import (
    METAL_MIX_RATIOS,
    ChemistryCalculator,
    ChemistryRecipe,
    CoatingMethod,
    MetalMix,
    PaperAbsorbency,
)
from ptpd_calibration.chemistry.cyanotype_calculator import (
    CYANOTYPE_PAPER_FACTORS,
    CyanotypeCalculator,
    CyanotypePaperType,
    CyanotypeRecipe,
    CyanotypeSettings,
)
from ptpd_calibration.chemistry.silver_gelatin_calculator import (
    DILUTION_MULTIPLIERS,
    TRAY_VOLUMES_ML,
    DeveloperRecipe,
    DilutionRatio,
    ProcessingChemistry,
    SilverGelatinCalculator,
    SilverGelatinSettings,
    TraySize,
)

__all__ = [
    # Platinum/Palladium
    "ChemistryCalculator",
    "ChemistryRecipe",
    "PaperAbsorbency",
    "CoatingMethod",
    "MetalMix",
    "METAL_MIX_RATIOS",
    # Cyanotype
    "CyanotypeCalculator",
    "CyanotypeRecipe",
    "CyanotypeSettings",
    "CyanotypePaperType",
    "CYANOTYPE_PAPER_FACTORS",
    # Silver Gelatin
    "SilverGelatinCalculator",
    "ProcessingChemistry",
    "SilverGelatinSettings",
    "DeveloperRecipe",
    "DilutionRatio",
    "TraySize",
    "DILUTION_MULTIPLIERS",
    "TRAY_VOLUMES_ML",
]
