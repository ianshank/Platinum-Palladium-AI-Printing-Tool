"""
Integrations module for Platinum-Palladium AI Printing Tool.

Provides integrations with external hardware and services:
- Spectrophotometer devices for density and color measurements
- Weather APIs for environmental condition monitoring
- Printer drivers for digital negative printing
- ICC profile management for color-accurate workflow
"""

from .spectrophotometer import (
    SpectrophotometerInterface,
    XRiteIntegration,
    MeasurementMode,
    ApertureSize,
    ExportFormat as SpectroExportFormat,
    LABValue,
    SpectralData,
    PatchMeasurement,
    CalibrationResult,
)

from .weather import (
    WeatherProvider,
    OpenWeatherMapProvider,
    WeatherCondition,
    PaperType,
    CurrentConditions,
    ForecastPeriod,
    DryingTimeEstimate,
    CoatingRecommendation,
)

from .printer_drivers import (
    PrinterInterface,
    EpsonDriver,
    CanonDriver,
    PrinterBrand,
    PrintQuality,
    MediaType,
    ColorMode,
    InkLevel,
    PrintSettings,
    NozzleCheckResult,
    PrintJob,
)

from .icc_profiles import (
    ICCProfileManager,
    ColorSpace,
    RenderingIntent,
    ProfileClass,
    ProfileInfo,
    ProfileValidation,
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
