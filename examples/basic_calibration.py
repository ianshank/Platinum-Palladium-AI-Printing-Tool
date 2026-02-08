#!/usr/bin/env python3
"""
Basic Calibration Example

This example demonstrates the core calibration workflow:
1. Read a step tablet scan
2. Analyze the density response
3. Generate a linearization curve
4. Export to QTR format
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptpd_calibration import (
    CurveAnalyzer,
    CurveGenerator,
    QTRExporter,
    StepTabletReader,
    TabletType,
)
from ptpd_calibration.core.types import CurveType


def main():
    # Configuration
    scan_path = Path("step_tablet_scan.tiff")  # Replace with your scan
    output_path = Path("linearization_curve.txt")

    # Paper and chemistry info
    paper_type = "Arches Platine"
    chemistry = "50% Pt / 50% Pd, 5 drops Na2"

    print("=" * 60)
    print("Pt/Pd Calibration - Basic Example")
    print("=" * 60)

    # Check if scan exists, otherwise use demo densities
    if scan_path.exists():
        print(f"\nReading step tablet from: {scan_path}")

        # Initialize reader
        reader = StepTabletReader(tablet_type=TabletType.STOUFFER_21)

        # Read the scan
        result = reader.read(scan_path)

        print("\nExtraction Results:")
        print(f"  Patches detected: {result.extraction.num_patches}")
        print(f"  Dmin (paper base): {result.extraction.dmin:.3f}")
        print(f"  Dmax: {result.extraction.dmax:.3f}")
        print(f"  Density range: {result.extraction.density_range:.3f}")
        print(f"  Quality score: {result.extraction.overall_quality:.1%}")

        if result.extraction.warnings:
            print("\nWarnings:")
            for warning in result.extraction.warnings:
                print(f"  - {warning}")

        densities = result.extraction.get_densities()
    else:
        print(f"\nNo scan found at {scan_path}")
        print("Using demo density values...")

        # Demo densities (typical Pt/Pd response)
        import numpy as np
        steps = np.linspace(0, 1, 21)
        densities = list(0.1 + 2.0 * (steps ** 0.85))

        print("\nDemo Densities:")
        print(f"  Dmin: {min(densities):.3f}")
        print(f"  Dmax: {max(densities):.3f}")
        print(f"  Range: {max(densities) - min(densities):.3f}")

    # Analyze the response
    print("\n" + "-" * 60)
    print("Density Analysis")
    print("-" * 60)

    analysis = CurveAnalyzer.analyze_linearity(densities)
    print(f"  Monotonic: {analysis.is_monotonic}")
    print(f"  Max linearity error: {analysis.max_error:.3f}")
    print(f"  RMS error: {analysis.rms_error:.3f}")

    suggestions = CurveAnalyzer.suggest_adjustments(densities)
    print("\nRecommendations:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")

    # Generate curve
    print("\n" + "-" * 60)
    print("Generating Calibration Curve")
    print("-" * 60)

    generator = CurveGenerator()
    curve = generator.generate(
        densities,
        curve_type=CurveType.LINEAR,
        name=f"{paper_type} - Linear",
        paper_type=paper_type,
        chemistry=chemistry,
    )

    print(f"  Curve name: {curve.name}")
    print(f"  Curve type: {curve.curve_type.value}")
    print(f"  Output points: {len(curve.output_values)}")

    # Analyze the generated curve
    curve_analysis = CurveAnalyzer.analyze_curve(curve)
    print(f"  Curve shape: {curve_analysis['shape']}")
    print(f"  Max deviation: {curve_analysis['max_deviation']:.3f}")

    # Export
    print("\n" + "-" * 60)
    print("Exporting Curve")
    print("-" * 60)

    exporter = QTRExporter()
    exporter.export(curve, output_path)

    print(f"  Exported to: {output_path}")
    print("  Format: QuadTone RIP")

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)

    return curve


if __name__ == "__main__":
    main()
