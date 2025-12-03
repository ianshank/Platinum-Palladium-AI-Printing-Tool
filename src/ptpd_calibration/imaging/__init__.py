"""
Imaging module for digital negative creation and curve application.

Provides tools for applying calibration curves to images, creating
digital negatives for platinum/palladium printing, and histogram analysis.
"""

from ptpd_calibration.imaging.processor import (
    ImageProcessor,
    ImageFormat,
    ProcessingResult,
    ExportSettings,
)
from ptpd_calibration.imaging.histogram import (
    HistogramAnalyzer,
    HistogramResult,
    HistogramStats,
    HistogramScale,
)
from ptpd_calibration.imaging.split_grade import (
    SplitGradeSettings,
    SplitGradeSimulator,
    TonalCurveAdjuster,
    TonalAnalysis,
    ExposureCalculation,
    BlendMode,
    MetalType,
)

__all__ = [
    # Processor
    "ImageProcessor",
    "ImageFormat",
    "ProcessingResult",
    "ExportSettings",
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
