#!/usr/bin/env python3
"""
Comprehensive Demonstration of Pt/Pd Calibration Studio

This script demonstrates all major features of the platinum/palladium
calibration system. Run this to see the full capabilities in action.

Usage:
    python examples/comprehensive_demo.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Add project to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_synthetic_step_tablet(num_patches: int = 21) -> Image.Image:
    """Create a synthetic step tablet image for demonstration."""
    width, height = num_patches * 20, 100
    patch_width = width // num_patches

    img = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_patches):
        # Create realistic density progression
        value = 255 - int(255 * (i / (num_patches - 1)) ** 0.85)
        x_start = i * patch_width
        x_end = (i + 1) * patch_width
        img[:, x_start:x_end] = value

    # Add border
    full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
    full_img[20:height + 20, 20:width + 20, 0] = img
    full_img[20:height + 20, 20:width + 20, 1] = img
    full_img[20:height + 20, 20:width + 20, 2] = img

    return Image.fromarray(full_img)


def create_test_image() -> Image.Image:
    """Create a test grayscale image for processing demos."""
    arr = np.linspace(30, 220, 256).reshape(16, 16).astype(np.uint8)
    arr = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)
    return Image.fromarray(arr, mode="L")


def demo_separator(title: str):
    """Print a visual separator for demo sections."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# =============================================================================
# DEMO 1: Step Tablet Reading and Density Extraction
# =============================================================================
def demo_step_tablet_reading():
    """Demonstrate step tablet reading capabilities."""
    demo_separator("Step Tablet Reading & Density Extraction")

    from ptpd_calibration.detection import StepTabletReader

    # Create synthetic step tablet
    step_tablet = create_synthetic_step_tablet(21)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        step_tablet.save(f.name)

        # Read step tablet
        reader = StepTabletReader()
        result = reader.read(f.name)

        print("Step tablet analysis complete!")
        print(f"  - Patches detected: {result.extraction.num_patches}")
        print(f"  - Quality score: {result.extraction.overall_quality:.2%}")

        densities = result.extraction.get_densities()
        if densities:
            print(f"  - Dmin (paper white): {min(densities):.3f}")
            print(f"  - Dmax (maximum black): {max(densities):.3f}")
            print(f"  - Density range: {max(densities) - min(densities):.3f}")

        if result.extraction.warnings:
            print(f"  - Warnings: {', '.join(result.extraction.warnings)}")

    return result


# =============================================================================
# DEMO 2: Curve Generation
# =============================================================================
def demo_curve_generation(extraction_result=None):
    """Demonstrate curve generation capabilities."""
    demo_separator("Curve Generation")

    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.curves import CurveGenerator

    generator = CurveGenerator()

    if extraction_result and extraction_result.extraction:
        # Generate from extraction
        curve = generator.generate_from_extraction(
            extraction_result.extraction,
            curve_type=CurveType.LINEAR,
            name="Demo Calibration Curve",
            paper_type="Arches Platine",
        )
    else:
        # Generate from sample densities
        from ptpd_calibration.core.models import CurveData
        curve = CurveData(
            name="Demo Calibration Curve",
            input_values=[i / 20.0 for i in range(21)],
            output_values=[min(1.0, (i / 20.0) ** 1.1) for i in range(21)],
        )

    print(f"Generated curve: {curve.name}")
    print(f"  - Input points: {len(curve.input_values)}")
    print(f"  - Output points: {len(curve.output_values)}")
    print(f"  - Input range: {curve.input_values[0]:.2f} - {curve.input_values[-1]:.2f}")
    print(f"  - Output range: {curve.output_values[0]:.2f} - {curve.output_values[-1]:.2f}")

    return curve


# =============================================================================
# DEMO 3: Auto-Linearization
# =============================================================================
def demo_auto_linearization():
    """Demonstrate auto-linearization algorithms."""
    demo_separator("Auto-Linearization")

    from ptpd_calibration.curves import AutoLinearizer, LinearizationMethod, TargetResponse

    # Sample density measurements from a step wedge
    densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

    linearizer = AutoLinearizer()

    print("Testing different linearization methods:\n")

    for method in [
        LinearizationMethod.DIRECT_INVERSION,
        LinearizationMethod.SPLINE_FIT,
        LinearizationMethod.POLYNOMIAL_FIT,
    ]:
        result = linearizer.linearize(
            densities,
            method=method,
            curve_name=f"Test {method.value}",
        )
        print(f"  {method.value}:")
        print(f"    - Residual error: {result.residual_error:.4f}")
        print(f"    - Max deviation: {result.max_deviation:.4f}")

    print("\nTesting different target responses:\n")

    for target in [TargetResponse.LINEAR, TargetResponse.GAMMA_22]:
        result = linearizer.linearize(
            densities,
            target=target,
            curve_name=f"Test {target.value}",
        )
        print(f"  {target.value}: Generated {len(result.curve.output_values)} point curve")


# =============================================================================
# DEMO 4: Quad File Import
# =============================================================================
def demo_quad_import():
    """Demonstrate importing a real-world QuadTone RIP .quad file."""
    demo_separator("Quad File Import")

    from ptpd_calibration.curves import load_quad_file

    # Path to example data
    data_dir = Path(__file__).parent / "data"
    quad_path = data_dir / "Platinum_Palladium_V6-CC.quad"

    # Check if file exists (it should be created by our setup)
    if not quad_path.exists():
        # Try to find it in tests/fixtures as fallback for dev environment
        alt_path = Path(__file__).parent.parent / "tests" / "fixtures" / "Platinum_Palladium_V6-CC.quad"
        if alt_path.exists():
            quad_path = alt_path
        else:
            print(f"Note: Example .quad file not found at {quad_path}. Skipping detail demo.")
            return

    print(f"Loading profile: {quad_path.name}")
    try:
        profile = load_quad_file(quad_path)

        print("\nProfile loaded successfully:")
        print(f"  - Name: {profile.profile_name}")
        print(f"  - Channels found: {', '.join(profile.channels.keys())}")

        # Show active channels
        active_channels = [name for name, ch in profile.channels.items() if any(v > 0 for v in ch.values)]
        print(f"  - Active channels: {', '.join(active_channels)}")

        # Inspect K channel
        if "K" in profile.channels:
            k_curve = profile.channels["K"]
            print("\nK Channel Analysis:")
            print(f"  - Points: {len(k_curve.values)}")
            print(f"  - Max output: {max(k_curve.values)} (on 0-255 scale)")

            # Show sample points
            print(f"  - First 5 values: {k_curve.values[:5]}")
            print(f"  - Last 5 values: {k_curve.values[-5:]}")

        # Show comments/metadata
        if profile.comments:
            print("\nMetadata/Comments:")
            for comment in profile.comments[:5]:  # Show first 5 comments
                print(f"  # {comment}")
            if len(profile.comments) > 5:
                print(f"  ... and {len(profile.comments) - 5} more comments")

    except Exception as e:
        print(f"Error loading quad file: {e}")
        raise


# =============================================================================
# DEMO 5: Chemistry Calculator
# =============================================================================
def demo_chemistry_calculator():
    """Demonstrate chemistry calculation."""
    demo_separator("Chemistry Calculator")

    from ptpd_calibration.chemistry import ChemistryCalculator

    calculator = ChemistryCalculator()

    # Calculate for standard 8x10 print
    recipe = calculator.calculate(
        width_inches=8.0,
        height_inches=10.0,
        platinum_ratio=0.5,  # 50% Pt / 50% Pd
    )

    print("Chemistry Recipe for 8x10\" print (50% Pt / 50% Pd):\n")
    print(f"  Coating area: {recipe.coating_area_sq_inches:.1f} sq inches")
    print("\nSolution drops:")
    print(f"  - Ferric Oxalate: {recipe.ferric_oxalate_drops:.0f} drops")
    print(f"  - Platinum: {recipe.platinum_drops:.0f} drops")
    print(f"  - Palladium: {recipe.palladium_drops:.0f} drops")
    print(f"  - Na2 (contrast): {recipe.na2_drops:.0f} drops")
    print(f"  - Total: {recipe.total_drops:.0f} drops")

    print("\nIn milliliters:")
    print(f"  - Total solution: {recipe.total_ml:.2f} ml")

    # Show standard sizes
    print("\nStandard print sizes available:")
    for name, (w, h) in ChemistryCalculator.get_standard_sizes().items():
        print(f"  - {name}: {w}\" x {h}\"")


# =============================================================================
# DEMO 5: Exposure Calculator
# =============================================================================
def demo_exposure_calculator():
    """Demonstrate exposure calculation."""
    demo_separator("Exposure Calculator")

    from ptpd_calibration.exposure import ExposureCalculator, ExposureSettings, LightSource

    # Set up base exposure settings
    settings = ExposureSettings(
        base_exposure_minutes=10.0,
        base_negative_density=1.6,
        light_source=LightSource.BL_FLUORESCENT,
        platinum_ratio=0.5,
    )

    calculator = ExposureCalculator(settings)

    # Calculate exposure for different negative densities
    print("Exposure calculations for different negative densities:\n")
    print(f"  Base: {settings.base_exposure_minutes} min at density {settings.base_negative_density}")
    print()

    for density in [1.4, 1.6, 1.8, 2.0]:
        result = calculator.calculate(negative_density=density)
        print(f"  Density {density}: {result.format_time()}")

    # Generate test strip
    print("\nTest strip exposures (centered on 10 min, 0.5 stop increments):")
    test_times = calculator.calculate_test_strip(
        center_exposure=10.0,
        steps=5,
        increment_stops=0.5,
    )
    for i, time in enumerate(test_times, 1):
        print(f"  Strip {i}: {time:.1f} min")

    # Show available light sources
    print("\nSupported light sources:")
    for source_name, _multiplier in calculator.get_light_sources()[:5]:
        print(f"  - {source_name}")


# =============================================================================
# DEMO 6: Zone System Analysis
# =============================================================================
def demo_zone_system():
    """Demonstrate zone system analysis."""
    demo_separator("Zone System Analysis")

    from ptpd_calibration.zones import Zone, ZoneMapper

    # Create test image
    test_image = create_test_image()

    mapper = ZoneMapper()
    analysis = mapper.analyze_image(test_image)

    print("Zone System Analysis:\n")
    print(f"  Development recommendation: {analysis.development_adjustment}")
    print("\nZone distribution:")

    for zone in Zone:
        count = analysis.zone_histogram.get(zone, 0)
        bar = "█" * int(count * 50)
        print(f"  Zone {zone.value:>2}: {bar} ({count:.1%})")


# =============================================================================
# DEMO 7: Histogram Analysis
# =============================================================================
def demo_histogram_analysis():
    """Demonstrate histogram analysis."""
    demo_separator("Histogram Analysis")

    from ptpd_calibration.imaging import HistogramAnalyzer

    test_image = create_test_image()

    analyzer = HistogramAnalyzer()
    result = analyzer.analyze(test_image)

    print("Histogram Analysis:\n")
    print(f"  Image size: {result.image_size}")
    print(f"  Mean: {result.stats.mean:.1f}")
    print(f"  Median: {result.stats.median:.1f}")
    print(f"  Std dev: {result.stats.std_dev:.1f}")
    print(f"  Dynamic range: {result.stats.dynamic_range:.0f} levels")
    print(f"  Brightness: {result.stats.brightness:.2f}")
    print(f"  Contrast: {result.stats.contrast:.2f}")

    if result.stats.highlight_clipping_percent > 0:
        print(f"  Highlight clipping: {result.stats.highlight_clipping_percent:.1f}%")
    if result.stats.shadow_clipping_percent > 0:
        print(f"  Shadow clipping: {result.stats.shadow_clipping_percent:.1f}%")


# =============================================================================
# DEMO 8: Soft Proofing
# =============================================================================
def demo_soft_proofing():
    """Demonstrate soft proofing simulation."""
    demo_separator("Soft Proofing")

    from ptpd_calibration.proofing import PaperSimulation, ProofSettings, SoftProofer

    test_image = create_test_image()

    print("Soft proofing on different papers:\n")

    for paper in [
        PaperSimulation.ARCHES_PLATINE,
        PaperSimulation.BERGGER_COT320,
        PaperSimulation.STONEHENGE,
    ]:
        settings = ProofSettings.from_paper_preset(paper)
        proofer = SoftProofer(settings)
        result = proofer.proof(test_image)

        print(f"  {paper.value}:")
        print(f"    - Dmax: {settings.paper_dmax}")
        print(f"    - Output size: {result.image.size}")
        print(f"    - Notes: {result.notes[0] if result.notes else 'None'}")


# =============================================================================
# DEMO 9: Paper Profiles
# =============================================================================
def demo_paper_profiles():
    """Demonstrate paper profiles database."""
    demo_separator("Paper Profiles Database")

    from ptpd_calibration.papers import PaperDatabase

    db = PaperDatabase()

    papers = db.list_papers()
    print(f"Available paper profiles ({len(papers)} papers):\n")

    for paper in papers[:6]:  # Show first 6
        print(f"  {paper.name}")
        if paper.characteristics:
            print(f"    - Dmax: {paper.characteristics.typical_dmax}")
            print(f"    - Sizing: {paper.characteristics.sizing}")
            print(f"    - Surface: {paper.characteristics.surface}")


# =============================================================================
# DEMO 10: Digital Negative Creation
# =============================================================================
def demo_digital_negative():
    """Demonstrate digital negative creation."""
    demo_separator("Digital Negative Creation")

    from ptpd_calibration.core.models import CurveData
    from ptpd_calibration.imaging import ImageProcessor

    test_image = create_test_image()

    # Create a simple correction curve
    curve = CurveData(
        name="Demo Curve",
        input_values=[i / 10.0 for i in range(11)],
        output_values=[min(1.0, (i / 10.0) ** 1.1) for i in range(11)],
    )

    processor = ImageProcessor()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_image.save(f.name)

        # Preview curve effect
        original, processed = processor.preview_curve_effect(f.name, curve)

        print("Curve preview generated:")
        print(f"  - Original: {original.size}")
        print(f"  - Processed: {processed.size}")

        # Create digital negative
        result = processor.create_digital_negative(
            f.name,
            curve=curve,
            invert=True,
        )

        print("\nDigital negative created:")
        print(f"  - Size: {result.image.size}")
        print(f"  - Mode: {result.image.mode}")
        print("  - Inverted: Yes")


# =============================================================================
# DEMO 11: Session Logging
# =============================================================================
def demo_session_logging():
    """Demonstrate print session logging."""
    demo_separator("Print Session Logging")

    from ptpd_calibration.session import PrintRecord, SessionLogger
    from ptpd_calibration.session.logger import ChemistryUsed, PrintResult

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = SessionLogger(storage_dir=Path(tmp_dir) / "sessions")

        # Start a session
        session = logger.start_session("Demo Print Session")

        # Log some prints
        for i, (paper, rating) in enumerate([
            ("Arches Platine", PrintResult.EXCELLENT),
            ("Bergger COT320", PrintResult.GOOD),
            ("Stonehenge", PrintResult.ACCEPTABLE),
        ]):
            record = PrintRecord(
                image_name=f"Test Image {i+1}",
                paper_type=paper,
                exposure_time_minutes=10.0 + i * 2,
                chemistry=ChemistryUsed(
                    ferric_oxalate_drops=12.0,
                    platinum_drops=6.0,
                    palladium_drops=6.0,
                ),
                result=rating,
            )
            logger.log_print(record)

        # Get statistics
        current = logger.get_current_session()
        stats = current.get_statistics()

        print(f"Session: {session.name}")
        print("\nSession statistics:")
        print(f"  - Total prints: {stats['total_prints']}")
        print(f"  - Success rate: {stats['success_rate']}")
        print(f"  - Avg exposure: {stats['avg_exposure_minutes']:.1f} min")
        print(f"  - Papers used: {', '.join(stats['papers_used'])}")


# =============================================================================
# DEMO 12: Curve Export Formats
# =============================================================================
def demo_curve_export():
    """Demonstrate curve export to different formats."""
    demo_separator("Curve Export Formats")

    from ptpd_calibration.core.models import CurveData
    from ptpd_calibration.curves import save_curve

    curve = CurveData(
        name="Demo Export Curve",
        input_values=[i / 255.0 for i in range(256)],
        output_values=[min(1.0, (i / 255.0) ** 1.1) for i in range(256)],
        paper_type="Arches Platine",
        chemistry_notes="50% Pt / 50% Pd",
    )

    print("Supported export formats:\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for fmt in ["qtr", "csv", "json"]:
            export_path = tmp_path / f"demo_curve.{fmt}"
            save_curve(curve, export_path, format=fmt)
            size = export_path.stat().st_size
            print(f"  - {fmt.upper()}: {size} bytes")

            # Show preview
            content = export_path.read_text()[:200]
            print(f"    Preview: {content[:100]}...")


# =============================================================================
# MAIN DEMO RUNNER
# =============================================================================
def run_all_demos():
    """Run all demonstration functions."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PLATINUM/PALLADIUM CALIBRATION STUDIO - COMPREHENSIVE DEMO  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    demos = [
        ("Step Tablet Reading", demo_step_tablet_reading),
        ("Curve Generation", demo_curve_generation),
        ("Auto-Linearization", demo_auto_linearization),
        ("Quad File Import", demo_quad_import),
        ("Chemistry Calculator", demo_chemistry_calculator),
        ("Exposure Calculator", demo_exposure_calculator),
        ("Zone System Analysis", demo_zone_system),
        ("Histogram Analysis", demo_histogram_analysis),
        ("Soft Proofing", demo_soft_proofing),
        ("Paper Profiles", demo_paper_profiles),
        ("Digital Negative Creation", demo_digital_negative),
        ("Session Logging", demo_session_logging),
        ("Curve Export Formats", demo_curve_export),
    ]

    results = {}
    last_result = None
    for name, demo_func in demos:
        try:
            if name == "Curve Generation" and last_result and hasattr(last_result, "extraction"):
                result = demo_func(last_result)
            else:
                result = demo_func()

            if name == "Step Tablet Reading":
                last_result = result

            results[name] = "✓ Success"
        except Exception as e:
            results[name] = f"✗ Error: {e}"
            print(f"\nError in {name}: {e}")

    # Summary
    demo_separator("DEMO SUMMARY")
    print("Demo Results:\n")
    for name, status in results.items():
        print(f"  {name}: {status}")

    successes = sum(1 for s in results.values() if s.startswith("✓"))
    print(f"\n  Total: {successes}/{len(demos)} demos completed successfully")

    print("\n" + "=" * 70)
    print("  Demo complete! For more information, see README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()
