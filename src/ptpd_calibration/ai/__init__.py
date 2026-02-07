"""
AI tools for platinum-palladium printing.

This module provides comprehensive AI-powered analysis, prediction,
and optimization for the complete Pt/Pd printing workflow.
"""

from ptpd_calibration.ai.platinum_palladium_ai import (
    ChemistryRecommendation,
    ContrastLevel,
    DigitalNegativeResult,
    ExposurePrediction,
    # Main AI class
    PlatinumPalladiumAI,
    PrinterProfile,
    PrintQualityAnalysis,
    ProblemArea,
    # Result models
    TonalityAnalysisResult,
    # Enums
    TonePreference,
    WorkflowOptimization,
)

__all__ = [
    # Main class
    "PlatinumPalladiumAI",

    # Enums
    "TonePreference",
    "ContrastLevel",
    "PrinterProfile",
    "ProblemArea",

    # Result models
    "TonalityAnalysisResult",
    "ExposurePrediction",
    "ChemistryRecommendation",
    "DigitalNegativeResult",
    "PrintQualityAnalysis",
    "WorkflowOptimization",
]
