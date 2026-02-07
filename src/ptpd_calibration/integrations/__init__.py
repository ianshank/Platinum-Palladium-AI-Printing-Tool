"""
Integrations module for Platinum-Palladium AI Printing Tool.

Provides integrations with external hardware and services:
- Spectrophotometer devices for density and color measurements
- Weather APIs for environmental condition monitoring
- Printer drivers for digital negative printing
- ICC profile management for color-accurate workflow
"""

from .icc_profiles import (
    ColorSpace,
    ICCProfileManager,
    ProfileClass,
    ProfileInfo,
    ProfileValidation,
    RenderingIntent,
)
from .printer_drivers import (
    CanonDriver,
    ColorMode,
    EpsonDriver,
    InkLevel,
    MediaType,
    NozzleCheckResult,
    PrinterBrand,
    PrinterInterface,
    PrintJob,
    PrintQuality,
    PrintSettings,
)
from .spectrophotometer import (
    ApertureSize,
    CalibrationResult,
    LABValue,
    MeasurementMode,
    PatchMeasurement,
    SpectralData,
    SpectrophotometerInterface,
    XRiteIntegration,
)
from .spectrophotometer import (
    ExportFormat as SpectroExportFormat,
)
from .weather import (
    CoatingRecommendation,
    CurrentConditions,
    DryingTimeEstimate,
    ForecastPeriod,
    OpenWeatherMapProvider,
    PaperType,
    WeatherCondition,
    WeatherProvider,
)

__all__ = [
    # Spectrophotometer
    "SpectrophotometerInterface",
    "XRiteIntegration",
    "MeasurementMode",
    "ApertureSize",
    "SpectroExportFormat",
    "LABValue",
    "SpectralData",
    "PatchMeasurement",
    "CalibrationResult",
    # Weather
    "WeatherProvider",
    "OpenWeatherMapProvider",
    "WeatherCondition",
    "PaperType",
    "CurrentConditions",
    "ForecastPeriod",
    "DryingTimeEstimate",
    "CoatingRecommendation",
    # Printer Drivers
    "PrinterInterface",
    "EpsonDriver",
    "CanonDriver",
    "PrinterBrand",
    "PrintQuality",
    "MediaType",
    "ColorMode",
    "InkLevel",
    "PrintSettings",
    "NozzleCheckResult",
    "PrintJob",
    # ICC Profiles
    "ICCProfileManager",
    "ColorSpace",
    "RenderingIntent",
    "ProfileClass",
    "ProfileInfo",
    "ProfileValidation",
]

__version__ = "0.1.0"
