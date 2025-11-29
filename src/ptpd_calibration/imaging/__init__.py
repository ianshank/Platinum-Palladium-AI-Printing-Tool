"""
Imaging module for digital negative creation and curve application.

Provides tools for applying calibration curves to images and creating
digital negatives for platinum/palladium printing.
"""

from ptpd_calibration.imaging.processor import (
    ImageProcessor,
    ImageFormat,
    ProcessingResult,
    ExportSettings,
)

__all__ = [
    "ImageProcessor",
    "ImageFormat",
    "ProcessingResult",
    "ExportSettings",
]
