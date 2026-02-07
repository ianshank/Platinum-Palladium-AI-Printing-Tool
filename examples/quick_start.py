#!/usr/bin/env python3
"""
Quick Start Guide for Pt/Pd Calibration Studio

This script shows the most common workflows for calibrating
digital negatives for platinum/palladium printing.

Usage:
    python examples/quick_start.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 60)
    print("  Pt/Pd Calibration Studio - Quick Start Guide")
    print("=" * 60)

    # =========================================================================
    # Example 1: Basic Calibration Workflow
    # =========================================================================
    print("\n1. BASIC CALIBRATION WORKFLOW")
    print("-" * 40)


    print("""
    # Read your scanned step tablet
    reader = StepTabletReader()
    result = reader.read("step_tablet_scan.tiff")

    # Check the results
    densities = result.extraction.get_densities()
    print(f"Dmax: {max(densities):.2f}")
    print(f"Dmin: {min(densities):.2f}")

    # Generate linearization curve
    generator = CurveGenerator()
    curve = generator.generate_from_extraction(
        result.extraction,
        curve_type=CurveType.LINEAR,
        name="My Paper - Standard",
        paper_type="Arches Platine",
    )

    # Export for your RIP
    save_curve(curve, "my_curve.txt", format="qtr")
    """)

    # =========================================================================
    # Example 2: Chemistry Calculation
    # =========================================================================
    print("\n2. CHEMISTRY CALCULATION")
    print("-" * 40)

    from ptpd_calibration.chemistry import ChemistryCalculator

    calculator = ChemistryCalculator()

    # Calculate for an 8x10 print
    recipe = calculator.calculate(
        width_inches=8.0,
        height_inches=10.0,
        platinum_ratio=0.5,  # 50/50 Pt/Pd
    )

    print(f"""
    Recipe for 8x10" print (50% Pt / 50% Pd):

    Ferric Oxalate: {recipe.ferric_oxalate_drops:.0f} drops
    Platinum:       {recipe.platinum_drops:.0f} drops
    Palladium:      {recipe.palladium_drops:.0f} drops
    Na2:            {recipe.na2_drops:.0f} drops
    ─────────────────────────
    Total:          {recipe.total_drops:.0f} drops ({recipe.total_ml:.2f} ml)
    """)

    # =========================================================================
    # Example 3: Exposure Calculation
    # =========================================================================
    print("\n3. EXPOSURE CALCULATION")
    print("-" * 40)

    from ptpd_calibration.exposure import ExposureCalculator, ExposureSettings

    settings = ExposureSettings(
        base_exposure_minutes=10.0,
        base_negative_density=1.6,
    )
    calculator = ExposureCalculator(settings)

    # Calculate for your negative
    result = calculator.calculate(negative_density=1.8)
    print(f"""
    For a negative with density 1.8:
    Exposure time: {result.format_time()}

    Test strip times (centered on 10 min):""")

    times = calculator.calculate_test_strip(10.0, steps=5, increment_stops=0.5)
    for i, t in enumerate(times, 1):
        print(f"      Strip {i}: {t:.1f} min")

    # =========================================================================
    # Example 4: Image Analysis
    # =========================================================================
    print("\n4. IMAGE ANALYSIS")
    print("-" * 40)

    print("""
    from ptpd_calibration.imaging import HistogramAnalyzer
    from ptpd_calibration.zones import ZoneMapper

    # Analyze histogram
    analyzer = HistogramAnalyzer()
    hist = analyzer.analyze(your_image)

    print(f"Mean: {hist.stats.mean:.1f}")
    print(f"Dynamic range: {hist.stats.dynamic_range}")

    # Zone system analysis
    mapper = ZoneMapper()
    zones = mapper.analyze_image(your_image)

    print(f"Development: {zones.development_adjustment}")
    """)

    # =========================================================================
    # Example 5: Digital Negative Creation
    # =========================================================================
    print("\n5. DIGITAL NEGATIVE CREATION")
    print("-" * 40)

    print("""
    from ptpd_calibration.imaging import ImageProcessor

    processor = ImageProcessor()

    # Create digital negative with your curve
    result = processor.create_digital_negative(
        "your_image.tiff",
        curve=your_curve,
        invert=True,
    )

    # Export
    result.image.save("negative.tiff")
    """)

    # =========================================================================
    # Example 6: Auto-Linearization
    # =========================================================================
    print("\n6. AUTO-LINEARIZATION")
    print("-" * 40)

    from ptpd_calibration.curves import AutoLinearizer, LinearizationMethod

    # Sample density measurements
    densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

    linearizer = AutoLinearizer()
    result = linearizer.linearize(
        densities,
        method=LinearizationMethod.SPLINE_FIT,
        curve_name="Auto-Linearized Curve",
    )

    print(f"""
    Auto-linearization from 9 density measurements:

    Method: {result.method_used.value}
    Residual error: {result.residual_error:.4f}
    Max deviation: {result.max_deviation:.4f}
    Curve points: {len(result.curve.output_values)}
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("  For full documentation, run: python examples/comprehensive_demo.py")
    print("  Or launch the UI: python -m ptpd_calibration.ui.gradio_app")
    print("=" * 60)


if __name__ == "__main__":
    main()
