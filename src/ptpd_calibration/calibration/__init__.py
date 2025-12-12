"""
Calibration module for print analysis and curve refinement.

This module provides tools for:
- Paper-specific curve adjustments based on empirically tested profiles
- Print scan analysis for iterative curve refinement
- Calibration session tracking

Example usage:

    # Adjust a curve for a specific paper
    from ptpd_calibration.calibration import (
        CurveCalibrator,
        QuadCurveParser,
        CALIBRATION_PROFILES
    )

    header, curves = QuadCurveParser.parse("my_curve.quad")
    calibrator = CurveCalibrator(CALIBRATION_PROFILES["arches_platine"])
    adjusted = calibrator.adjust_all_curves(curves)
    QuadCurveParser.write("adjusted.quad", header, adjusted)

    # Analyze a print scan
    from ptpd_calibration.calibration import PrintAnalyzer

    analyzer = PrintAnalyzer()
    analysis = analyzer.analyze_from_file("print_scan.jpg")
    print(analysis.summary())

    # Apply feedback-based refinement
    refined = calibrator.adjust_all_from_feedback(
        curves,
        highlight_delta=analysis.recommended_highlight_adj,
        midtone_delta=analysis.recommended_midtone_adj,
        shadow_delta=analysis.recommended_shadow_adj
    )
"""

from ptpd_calibration.calibration.curve_adjuster import (
    CalibrationProfile,
    CurveCalibrator,
    QuadCurveParser,
    CALIBRATION_PROFILES,
    adjust_curve_for_paper,
    refine_curve_from_print,
    get_available_calibration_profiles,
    get_calibration_profile,
)

from ptpd_calibration.calibration.print_analyzer import (
    PrintAnalysis,
    PrintAnalyzer,
    CalibrationSession,
    CalibrationIteration,
    TargetDensities,
)

from ptpd_calibration.calibration.database import (
    CalibrationDatabase,
    CalibrationRecord,
    CalibrationSessionRecord,
)

__all__ = [
    # Curve adjustment
    "CalibrationProfile",
    "CurveCalibrator",
    "QuadCurveParser",
    "CALIBRATION_PROFILES",
    "adjust_curve_for_paper",
    "refine_curve_from_print",
    "get_available_calibration_profiles",
    "get_calibration_profile",
    # Print analysis
    "PrintAnalysis",
    "PrintAnalyzer",
    "CalibrationSession",
    "CalibrationIteration",
    "TargetDensities",
    # Database
    "CalibrationDatabase",
    "CalibrationRecord",
    "CalibrationSessionRecord",
]
