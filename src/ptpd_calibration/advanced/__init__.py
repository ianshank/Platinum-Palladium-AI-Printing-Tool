"""
Advanced features for platinum/palladium printing.

This module provides sophisticated tools for:
- Alternative process simulation (cyanotype, Van Dyke, etc.)
- Advanced negative blending and masking
- QR code metadata generation for archival labels
- Historic style transfer from master printers
- Print comparison and quality analysis
"""

from ptpd_calibration.advanced.features import (
    AlternativeProcessParams,
    AlternativeProcessSimulator,
    BlendMode,
    HistoricStyle,
    NegativeBlender,
    PrintComparison,
    PrintMetadata,
    QRMetadataGenerator,
    StyleParameters,
    StyleTransfer,
)

__all__ = [
    # Main classes
    "AlternativeProcessSimulator",
    "NegativeBlender",
    "QRMetadataGenerator",
    "StyleTransfer",
    "PrintComparison",
    # Configuration classes
    "AlternativeProcessParams",
    "StyleParameters",
    "PrintMetadata",
    # Enums
    "BlendMode",
    "HistoricStyle",
]
