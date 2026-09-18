"""
Imaging module for digital negative creation and curve application.

Provides tools for applying calibration curves to images, creating
digital negatives for platinum/palladium printing, and histogram analysis.
"""

from ptpd_calibration.imaging.histogram import (
    HistogramAnalyzer,
    HistogramResult,
    HistogramScale,
    HistogramStats,
)
from ptpd_calibration.imaging.processor import (
    HIGH_DEPTH_GRAY_MODES,
    SIXTEEN_BIT_FORMATS,
    ColorMode,
    ExportSettings,
    ImageFormat,
    ImageProcessor,
    ProcessingResult,
    is_high_depth_gray,
    to_eight_bit_gray,
    to_uint16,
)
from ptpd_calibration.imaging.split_grade import (
    BlendMode,
    ExposureCalculation,
    MetalType,
    SplitGradeSettings,
    SplitGradeSimulator,
    TonalAnalysis,
    TonalCurveAdjuster,
)

__all__ = [
    # Processor
    "ImageProcessor",
    "ImageFormat",
    "ProcessingResult",
    "ExportSettings",
    "ColorMode",
    # Bit depth (ADR-0016)
    "HIGH_DEPTH_GRAY_MODES",
    "SIXTEEN_BIT_FORMATS",
    "is_high_depth_gray",
    "to_eight_bit_gray",
    "to_uint16",
    # Histogram
    "HistogramAnalyzer",
    "HistogramResult",
    "HistogramStats",
    "HistogramScale",
    # Split-grade
    "SplitGradeSettings",
    "SplitGradeSimulator",
    "TonalCurveAdjuster",
    "TonalAnalysis",
    "ExposureCalculation",
    "BlendMode",
    "MetalType",
]
