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

from ptpd_calibration.exposure.calculator import (
    ExposureCalculator,
    LightSource,
    ExposureResult,
    ExposureSettings,
    LIGHT_SOURCE_SPEEDS,
)

from ptpd_calibration.exposure.alternative_calculators import (
    # Cyanotype
    CyanotypeExposureCalculator,
    CyanotypeExposureResult,
    UVSource,
    UV_SOURCE_SPEEDS,
    # Silver Gelatin
    SilverGelatinExposureCalculator,
    SilverGelatinExposureResult,
    EnlargerLightSource,
    ENLARGER_LIGHT_SPEEDS,
    # Van Dyke
    VanDykeExposureCalculator,
    # Kallitype
    KallitypeExposureCalculator,
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
