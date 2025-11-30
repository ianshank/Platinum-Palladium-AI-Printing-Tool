"""
Enhanced calculations module for platinum/palladium printing.

This module provides advanced technical calculators for:
- UV exposure with environmental compensation
- Coating volume optimization
- Cost tracking and analysis
- Solution dilution and replenishment
- Environmental and seasonal adjustments

All calculations are configuration-driven with no hardcoded values.

Example Usage:
    >>> from ptpd_calibration.calculations import UVExposureCalculator, CoatingVolumeCalculator
    >>>
    >>> # Calculate UV exposure with environmental factors
    >>> uv_calc = UVExposureCalculator()
    >>> result = uv_calc.calculate_uv_exposure(
    ...     base_time=10.0,
    ...     negative_density=1.8,
    ...     humidity=55.0,
    ...     temperature=70.0,
    ...     uv_intensity=95.0,
    ...     paper_factor=1.0,
    ...     chemistry_factor=1.2,
    ... )
    >>> print(f"Adjusted exposure: {result.adjusted_exposure_minutes:.2f} minutes")
    >>>
    >>> # Calculate coating volume
    >>> coating_calc = CoatingVolumeCalculator()
    >>> volume = coating_calc.determine_coating_volume(
    ...     paper_area=80.0,  # 8x10 inches
    ...     paper_type="arches_platine",
    ...     coating_method="glass_rod",
    ...     humidity=50.0,
    ... )
    >>> print(f"Required volume: {volume.recommended_ml:.1f} ml ({volume.recommended_drops:.0f} drops)")
"""

from ptpd_calibration.calculations.enhanced import (
    # Calculators
    UVExposureCalculator,
    CoatingVolumeCalculator,
    CostCalculator,
    DilutionCalculator,
    EnvironmentalCompensation,

    # Result Models
    ExposureResult,
    CoatingResult,
    PrintCostResult,
    SessionCostResult,
    SolutionUsageEstimate,
    DilutionResult,
    ReplenishmentResult,
    EnvironmentalAdjustment,
    OptimalConditions,
    DryingTimeEstimate,
)

__all__ = [
    # Calculators
    "UVExposureCalculator",
    "CoatingVolumeCalculator",
    "CostCalculator",
    "DilutionCalculator",
    "EnvironmentalCompensation",

    # Result Models
    "ExposureResult",
    "CoatingResult",
    "PrintCostResult",
    "SessionCostResult",
    "SolutionUsageEstimate",
    "DilutionResult",
    "ReplenishmentResult",
    "EnvironmentalAdjustment",
    "OptimalConditions",
    "DryingTimeEstimate",
]
