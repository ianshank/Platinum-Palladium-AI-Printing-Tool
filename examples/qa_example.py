"""
Example usage of the Quality Assurance module.

Demonstrates all major QA features for platinum/palladium printing.
"""

from datetime import datetime

import numpy as np

from ptpd_calibration.qa import (
    AlertSeverity,
    AlertSystem,
    AlertType,
    ChemistryFreshnessTracker,
    NegativeDensityValidator,
    PaperHumidityChecker,
    QualityReport,
    ReportFormat,
    SolutionType,
    UVLightMeterIntegration,
)


def example_density_validation():
    """Example: Validate negative density."""
    print("\n" + "=" * 60)
    print("NEGATIVE DENSITY VALIDATION")
    print("=" * 60)

    # Create a test negative image (simulate a density wedge)
    # Creating a gradient from white (255) to black (0)
    test_image = np.linspace(255, 0, 256).reshape(1, -1)
    test_image = np.repeat(test_image, 100, axis=0).astype(np.uint8)

    # Initialize validator
    validator = NegativeDensityValidator()

    # Validate density range
    analysis = validator.validate_density_range(test_image)

    print("\nDensity Range Analysis:")
    print(f"  Min Density: {analysis.min_density:.3f}")
    print(f"  Max Density: {analysis.max_density:.3f}")
    print(f"  Mean Density: {analysis.mean_density:.3f}")
    print(f"  Density Range: {analysis.density_range:.3f}")
    print(f"  Highlight Blocked: {analysis.highlight_blocked}")
    print(f"  Shadow Blocked: {analysis.shadow_blocked}")

    if analysis.warnings:
        print("\nWarnings:")
        for warning in analysis.warnings:
            print(f"  - {warning}")

    if analysis.suggestions:
        print("\nSuggestions:")
        for suggestion in analysis.suggestions:
            print(f"  - {suggestion}")

    # Check specific areas
    has_highlight_detail, msg = validator.check_highlight_detail(test_image)
    print(f"\nHighlight Detail: {msg}")

    has_shadow_detail, msg = validator.check_shadow_detail(test_image)
    print(f"Shadow Detail: {msg}")


def example_chemistry_tracking():
    """Example: Track chemistry freshness."""
    print("\n" + "=" * 60)
    print("CHEMISTRY FRESHNESS TRACKING")
    print("=" * 60)

    # Initialize tracker
    tracker = ChemistryFreshnessTracker()

    # Register solutions
    print("\nRegistering solutions...")
    pd_id = tracker.register_solution(
        solution_type=SolutionType.PALLADIUM,
        date_mixed=datetime.now(),
        volume_ml=100.0,
        notes="Fresh batch for portrait printing"
    )
    print(f"  Palladium solution registered: {pd_id}")

    fo_id = tracker.register_solution(
        solution_type=SolutionType.FERRIC_OXALATE_1,
        date_mixed=datetime.now(),
        volume_ml=200.0,
        notes="Standard ferric oxalate"
    )
    print(f"  Ferric Oxalate solution registered: {fo_id}")

    # Check freshness
    print("\nChecking freshness...")
    is_fresh, msg = tracker.check_freshness(pd_id)
    print(f"  Palladium: {msg}")

    is_fresh, msg = tracker.check_freshness(fo_id)
    print(f"  Ferric Oxalate: {msg}")

    # Log usage
    print("\nLogging usage...")
    tracker.log_usage(pd_id, 10.0)
    tracker.log_usage(fo_id, 15.0)
    print(f"  Palladium remaining: {tracker.get_remaining_volume(pd_id):.1f} ml")
    print(f"  Ferric Oxalate remaining: {tracker.get_remaining_volume(fo_id):.1f} ml")

    # Get alerts
    alerts = tracker.get_alerts()
    if alerts:
        print("\nActive Alerts:")
        for alert in alerts:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("\nNo active alerts")

    # Get recommendations
    rec = tracker.recommend_replenishment(pd_id)
    print(f"\nReplenishment Recommendation: {rec}")


def example_humidity_monitoring():
    """Example: Monitor paper humidity."""
    print("\n" + "=" * 60)
    print("PAPER HUMIDITY MONITORING")
    print("=" * 60)

    # Initialize checker
    checker = PaperHumidityChecker()

    # Log ambient conditions
    print("\nLogging ambient conditions...")
    checker.log_ambient_conditions(
        humidity_percent=45.0,
        temperature_celsius=22.0
    )

    # Measure paper humidity
    print("\nMeasuring paper humidity...")
    reading = checker.measure_paper_humidity(
        humidity_percent=52.0,
        temperature_celsius=21.5,
        paper_type="Hahnemuhle Platinum Rag",
        notes="Paper stored in dry cabinet for 24 hours"
    )
    print(f"  Humidity: {reading.humidity_percent:.1f}%")
    print(f"  Temperature: {reading.temperature_celsius:.1f}°C")

    # Check if ready to coat
    is_ready, msg = checker.is_paper_ready()
    print(f"\nPaper Status: {msg}")

    # Estimate drying time if needed
    if not is_ready:
        hours, msg = checker.estimate_drying_time(reading.humidity_percent)
        print(f"Drying Time Estimate: {msg}")

    # Get recommendation
    recommendation = checker.recommend_humidity_adjustment()
    print(f"\nRecommendation: {recommendation}")


def example_uv_meter():
    """Example: UV light meter integration."""
    print("\n" + "=" * 60)
    print("UV LIGHT METER INTEGRATION")
    print("=" * 60)

    # Initialize UV meter
    uv_meter = UVLightMeterIntegration()

    # Calibrate meter
    print("\nCalibrating meter...")
    status = uv_meter.calibrate_meter(reference_intensity=100.0)
    print(f"  {status}")

    # Take readings
    print("\nTaking UV readings...")
    reading1 = uv_meter.read_intensity(
        intensity=95.0,
        wavelength=365.0,
        bulb_hours=150.0,
        notes="Morning reading before print session"
    )
    print(f"  Reading 1: {reading1.intensity:.2f} at {reading1.wavelength:.0f}nm")

    reading2 = uv_meter.read_intensity(
        intensity=93.5,
        wavelength=365.0,
        bulb_hours=155.0,
        notes="After 2-hour print session"
    )
    print(f"  Reading 2: {reading2.intensity:.2f} at {reading2.wavelength:.0f}nm")

    # Calculate exposure adjustment
    adjustment, msg = uv_meter.calculate_exposure_adjustment()
    print(f"\nExposure Adjustment: {adjustment:.2f}x")
    print(f"  {msg}")

    # Check for bulb degradation
    degraded, msg = uv_meter.check_bulb_degradation()
    print(f"\nBulb Degradation Check: {msg}")

    # Get replacement recommendation
    recommendation = uv_meter.recommend_bulb_replacement()
    print(f"Bulb Replacement: {recommendation}")


def example_pre_print_checklist():
    """Example: Generate pre-print checklist."""
    print("\n" + "=" * 60)
    print("PRE-PRINT CHECKLIST")
    print("=" * 60)

    # Create test image
    test_image = np.linspace(255, 20, 256).reshape(1, -1)
    test_image = np.repeat(test_image, 100, axis=0).astype(np.uint8)

    # Set up tracking systems
    chemistry_tracker = ChemistryFreshnessTracker()
    chemistry_tracker.register_solution(
        SolutionType.PALLADIUM, datetime.now(), 100.0
    )

    humidity_checker = PaperHumidityChecker()
    humidity_checker.measure_paper_humidity(52.0, 21.0, "Arches Platine")

    uv_meter = UVLightMeterIntegration()
    uv_meter.read_intensity(98.0, 365.0, 200.0)

    # Generate checklist
    report = QualityReport()
    checklist = report.generate_pre_print_checklist(
        image=test_image,
        chemistry_tracker=chemistry_tracker,
        humidity_checker=humidity_checker,
        uv_meter=uv_meter,
    )

    print(f"\nReady to Print: {checklist['ready_to_print']}")

    print("\nChecks:")
    for check_name, check_data in checklist['checks'].items():
        print(f"  {check_name}: {check_data.get('status', 'N/A').upper()}")

    if checklist['warnings']:
        print("\nWarnings:")
        for warning in checklist['warnings']:
            print(f"  - {warning}")

    if checklist['errors']:
        print("\nErrors:")
        for error in checklist['errors']:
            print(f"  - {error}")


def example_alert_system():
    """Example: Alert system usage."""
    print("\n" + "=" * 60)
    print("ALERT SYSTEM")
    print("=" * 60)

    # Initialize alert system
    alerts = AlertSystem()

    # Add various alerts
    print("\nAdding alerts...")
    alert1 = alerts.add_alert(
        alert_type=AlertType.DENSITY,
        message="Negative density range below recommended minimum",
        severity=AlertSeverity.WARNING
    )
    print(f"  Added alert: {alert1}")

    alert2 = alerts.add_alert(
        alert_type=AlertType.CHEMISTRY,
        message="Palladium solution expires in 3 days",
        severity=AlertSeverity.ERROR
    )
    print(f"  Added alert: {alert2}")

    alert3 = alerts.add_alert(
        alert_type=AlertType.UV_LIGHT,
        message="UV bulb showing 20% degradation",
        severity=AlertSeverity.CRITICAL
    )
    print(f"  Added alert: {alert3}")

    # Get active alerts
    print("\nActive alerts:")
    active = alerts.get_active_alerts()
    for alert in active:
        print(f"  [{alert.severity.value.upper()}] {alert.message}")

    # Get summary
    summary = alerts.get_alert_summary()
    print("\nAlert Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  Error: {summary['error']}")
    print(f"  Warning: {summary['warning']}")
    print(f"  Info: {summary['info']}")

    # Dismiss an alert
    print("\nDismissing warning alert...")
    alerts.dismiss_alert(alert1)
    active = alerts.get_active_alerts()
    print(f"  Active alerts remaining: {len(active)}")


def example_full_workflow():
    """Example: Complete QA workflow."""
    print("\n" + "=" * 60)
    print("COMPLETE QA WORKFLOW")
    print("=" * 60)

    # 1. Validate negative
    print("\n1. Validating negative...")
    test_negative = np.random.randint(30, 230, (512, 512), dtype=np.uint8)
    validator = NegativeDensityValidator()
    analysis = validator.validate_density_range(test_negative)
    print(f"   Density range: {analysis.density_range:.2f}")

    # 2. Check chemistry
    print("\n2. Checking chemistry freshness...")
    tracker = ChemistryFreshnessTracker()
    pd_id = tracker.register_solution(SolutionType.PALLADIUM, datetime.now(), 50.0)
    is_fresh, msg = tracker.check_freshness(pd_id)
    print(f"   {msg}")

    # 3. Check paper humidity
    print("\n3. Checking paper humidity...")
    humidity_checker = PaperHumidityChecker()
    humidity_checker.measure_paper_humidity(48.0, 20.5)
    is_ready, msg = humidity_checker.is_paper_ready()
    print(f"   {msg}")

    # 4. Check UV intensity
    print("\n4. Checking UV light...")
    uv_meter = UVLightMeterIntegration()
    uv_meter.read_intensity(102.0, 365.0)
    adjustment, msg = uv_meter.calculate_exposure_adjustment()
    print(f"   {msg}")

    # 5. Generate pre-print report
    print("\n5. Generating pre-print checklist...")
    report = QualityReport()
    checklist = report.generate_pre_print_checklist(
        image=test_negative,
        chemistry_tracker=tracker,
        humidity_checker=humidity_checker,
        uv_meter=uv_meter,
    )

    # 6. Export report
    print("\n6. Exporting report...")
    markdown_report = report.export_report(checklist, ReportFormat.MARKDOWN)
    print("   Report generated in Markdown format")
    print("\n" + "-" * 60)
    print(markdown_report)
    print("-" * 60)


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("PLATINUM/PALLADIUM QA MODULE EXAMPLES")
    print("=" * 60)

    # Run all examples
    example_density_validation()
    example_chemistry_tracking()
    example_humidity_monitoring()
    example_uv_meter()
    example_pre_print_checklist()
    example_alert_system()
    example_full_workflow()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60 + "\n")
