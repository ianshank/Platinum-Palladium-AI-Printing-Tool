"""
Quality Assurance module for platinum/palladium printing.

Provides comprehensive QA tools including:
- Negative density validation and analysis
- Chemistry solution freshness tracking
- Paper humidity monitoring for coating readiness
- UV light meter integration and calibration
- Quality report generation (pre-print and post-print)
- Alert system for monitoring printing conditions

All thresholds and settings are configurable via QASettings.

Example usage:
    >>> from ptpd_calibration.qa import (
    ...     NegativeDensityValidator,
    ...     ChemistryFreshnessTracker,
    ...     PaperHumidityChecker,
    ...     UVLightMeterIntegration,
    ...     QualityReport,
    ...     AlertSystem,
    ...     QASettings,
    ... )
    >>>
    >>> # Validate negative density
    >>> validator = NegativeDensityValidator()
    >>> analysis = validator.validate_density_range(image)
    >>> print(analysis.to_dict())
    >>>
    >>> # Track chemistry freshness
    >>> tracker = ChemistryFreshnessTracker()
    >>> solution_id = tracker.register_solution(
    ...     SolutionType.PALLADIUM,
    ...     datetime.now(),
    ...     volume_ml=100.0
    ... )
    >>> is_fresh, msg = tracker.check_freshness(solution_id)
    >>>
    >>> # Generate pre-print checklist
    >>> report = QualityReport()
    >>> checklist = report.generate_pre_print_checklist(
    ...     image=negative_image,
    ...     chemistry_tracker=tracker,
    ...     humidity_checker=humidity_checker,
    ... )
"""

from ptpd_calibration.qa.quality_assurance import (
    Alert,
    AlertSeverity,
    AlertSystem,
    AlertType,
    ChemistryFreshnessTracker,
    ChemistrySolution,
    # Data Models
    DensityAnalysis,
    HumidityReading,
    # Main Components
    NegativeDensityValidator,
    PaperHumidityChecker,
    # Configuration
    QASettings,
    QualityReport,
    ReportFormat,
    SolutionType,
    UVLightMeterIntegration,
    UVReading,
)

__all__ = [
    # Configuration
    "QASettings",
    "SolutionType",
    "AlertSeverity",
    "AlertType",
    "ReportFormat",
    # Data Models
    "DensityAnalysis",
    "ChemistrySolution",
    "HumidityReading",
    "UVReading",
    "Alert",
    # Main Components
    "NegativeDensityValidator",
    "ChemistryFreshnessTracker",
    "PaperHumidityChecker",
    "UVLightMeterIntegration",
    "QualityReport",
    "AlertSystem",
]
