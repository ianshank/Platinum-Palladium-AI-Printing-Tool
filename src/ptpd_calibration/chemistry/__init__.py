"""
Chemistry calculation module for alternative photographic printing processes.

Provides tools for calculating coating solution and processing chemistry amounts
for various alternative photography processes including:
- Platinum/Palladium printing
- Cyanotype (iron-based blue process)
- Silver Gelatin (traditional darkroom)
"""

from ptpd_calibration.chemistry.calculator import (
    ChemistryCalculator,
    ChemistryRecipe,
    PaperAbsorbency,
    CoatingMethod,
    MetalMix,
    METAL_MIX_RATIOS,
)

from ptpd_calibration.chemistry.cyanotype_calculator import (
    CyanotypeCalculator,
    CyanotypeRecipe,
    CyanotypeSettings,
    CyanotypePaperType,
    CYANOTYPE_PAPER_FACTORS,
)

from ptpd_calibration.chemistry.silver_gelatin_calculator import (
    SilverGelatinCalculator,
    ProcessingChemistry,
    SilverGelatinSettings,
    DeveloperRecipe,
    DilutionRatio,
    TraySize,
    DILUTION_MULTIPLIERS,
    TRAY_VOLUMES_ML,
)

__all__ = [
    # Platinum/Palladium
    "ChemistryCalculator",
    "ChemistryRecipe",
    "PaperAbsorbency",
    "CoatingMethod",
    "MetalMix",
    # Cyanotype
    "CyanotypeCalculator",
    "CyanotypeRecipe",
    "CyanotypeSettings",
    "CyanotypePaperType",
    # Silver Gelatin
    "SilverGelatinCalculator",
    "ProcessingChemistry",
    "SilverGelatinSettings",
    "DeveloperRecipe",
    "DilutionRatio",
    "TraySize",
]
