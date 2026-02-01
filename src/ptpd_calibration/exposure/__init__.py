"""
Exposure calculators for alternative photographic printing processes.

Calculate exposure times based on negative density, light source, paper type,
and process-specific characteristics for:
- Platinum/Palladium printing
- Cyanotype (iron-based blue process)
- Silver Gelatin (traditional darkroom)
- Van Dyke Brown
- Kallitype
"""

from ptpd_calibration.exposure.alternative_calculators import (
    ENLARGER_LIGHT_SPEEDS,
    UV_SOURCE_SPEEDS,
    # Cyanotype
    CyanotypeExposureCalculator,
    CyanotypeExposureResult,
    EnlargerLightSource,
    # Kallitype
    KallitypeExposureCalculator,
    # Silver Gelatin
    SilverGelatinExposureCalculator,
    SilverGelatinExposureResult,
    UVSource,
    # Van Dyke
    VanDykeExposureCalculator,
)
from ptpd_calibration.exposure.calculator import (
    LIGHT_SOURCE_SPEEDS,
    ExposureCalculator,
    ExposureResult,
    ExposureSettings,
    LightSource,
)

__all__ = [
    # Platinum/Palladium
    "ExposureCalculator",
    "LightSource",
    "ExposureResult",
    "ExposureSettings",
    "LIGHT_SOURCE_SPEEDS",
    # Cyanotype
    "CyanotypeExposureCalculator",
    "CyanotypeExposureResult",
    "UVSource",
    "UV_SOURCE_SPEEDS",
    # Silver Gelatin
    "SilverGelatinExposureCalculator",
    "SilverGelatinExposureResult",
    "EnlargerLightSource",
    "ENLARGER_LIGHT_SPEEDS",
    # Van Dyke
    "VanDykeExposureCalculator",
    # Kallitype
    "KallitypeExposureCalculator",
]
