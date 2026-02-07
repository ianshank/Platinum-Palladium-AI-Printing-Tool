#!/usr/bin/env python3
"""
Example usage of the split-grade printing simulation module.

This example demonstrates the complete split-grade workflow:
1. Analyze an image for optimal split-grade parameters
2. Create shadow and highlight masks
3. Apply split-grade simulation
4. Calculate exposure times
5. Preview results
"""

import numpy as np

from ptpd_calibration.imaging.split_grade import (
    BlendMode,
    SplitGradeSettings,
    SplitGradeSimulator,
    TonalCurveAdjuster,
)


def example_basic_usage():
    """Basic split-grade simulation example."""
    print("=" * 60)
    print("BASIC SPLIT-GRADE SIMULATION EXAMPLE")
    print("=" * 60)

    # Create simulator with default settings
    simulator = SplitGradeSimulator()

    # Load an image (or create a test image)
    # In practice, you would load your image:
    # image = Image.open("path/to/your/image.jpg")

    # Create a test gradient image for demonstration
    test_image = create_test_image()

    # 1. Analyze the image
    print("\n1. Analyzing image...")
    analysis = simulator.analyze_image(test_image)

    print("\nTonal Analysis Results:")
    print(f"  Mean Luminance: {analysis.mean_luminance:.3f}")
    print(f"  Median Luminance: {analysis.median_luminance:.3f}")
    print(f"  Standard Deviation: {analysis.std_luminance:.3f}")
    print(f"  Tonal Range: {analysis.tonal_range:.3f}")
    print(f"  Contrast Score: {analysis.contrast_score:.3f}")
    print("\nTonal Distribution:")
    print(f"  Shadows: {analysis.shadow_percentage * 100:.1f}%")
    print(f"  Midtones: {analysis.midtone_percentage * 100:.1f}%")
    print(f"  Highlights: {analysis.highlight_percentage * 100:.1f}%")
    print("\nRecommendations:")
    print(f"  Shadow Grade: {analysis.recommended_shadow_grade:.1f}")
    print(f"  Highlight Grade: {analysis.recommended_highlight_grade:.1f}")
    print(f"  Shadow Threshold: {analysis.recommended_shadow_threshold:.2f}")
    print(f"  Highlight Threshold: {analysis.recommended_highlight_threshold:.2f}")
    print(f"  Exposure Ratio: {analysis.recommended_exposure_ratio:.2f}")
    print(f"  Needs Split-Grade: {analysis.needs_split_grade}")

    if analysis.notes:
        print("\nNotes:")
        for note in analysis.notes:
            print(f"  • {note}")

    # 2. Apply split-grade simulation
    print("\n2. Applying split-grade simulation...")
    processed = simulator.simulate_split_grade(test_image)
    print(f"Processed image shape: {processed.shape}")

    # 3. Generate preview with masks
    print("\n3. Generating preview...")
    preview = simulator.preview_result(test_image, include_masks=True)
    print(f"Preview contains: {list(preview.keys())}")

    # 4. Calculate exposure times
    print("\n4. Calculating exposure times...")
    base_exposure = 60.0  # 60 seconds base exposure
    exposure_calc = simulator.calculate_exposure_times(base_exposure)
    print(f"\n{exposure_calc.format_exposure_info()}")


def example_custom_settings():
    """Example with custom split-grade settings."""
    print("\n\n" + "=" * 60)
    print("CUSTOM SETTINGS EXAMPLE")
    print("=" * 60)

    # Create custom settings for high-contrast Pt/Pd printing
    settings = SplitGradeSettings(
        shadow_grade=3.5,  # Harder shadow grade
        highlight_grade=1.0,  # Softer highlight grade
        shadow_exposure_ratio=0.65,  # More shadow exposure
        blend_mode=BlendMode.GAMMA,
        blend_gamma=2.4,
        shadow_threshold=0.35,
        highlight_threshold=0.75,
        platinum_ratio=0.5,  # 50/50 Pt/Pd mix
        mask_blur_radius=15.0,
        preserve_highlights=True,
        preserve_shadows=True,
    )

    print("\nCustom Settings:")
    print(f"  Shadow Grade: {settings.shadow_grade}")
    print(f"  Highlight Grade: {settings.highlight_grade}")
    print(f"  Shadow Exposure: {settings.shadow_exposure_ratio * 100:.0f}%")
    print(f"  Blend Mode: {settings.blend_mode.value}")
    print(f"  Platinum Ratio: {settings.platinum_ratio * 100:.0f}%")

    # Create simulator with custom settings
    simulator = SplitGradeSimulator(settings=settings)

    # Create test image
    test_image = create_test_image(size=(800, 600))

    # Process with custom settings
    processed = simulator.simulate_split_grade(test_image)
    print(f"\nProcessed with custom settings: {processed.shape}")

    # Calculate exposures
    exposure = simulator.calculate_exposure_times(base_time=90.0)
    print(f"\n{exposure.format_exposure_info()}")


def example_tonal_curves():
    """Example of using TonalCurveAdjuster directly."""
    print("\n\n" + "=" * 60)
    print("TONAL CURVE ADJUSTER EXAMPLE")
    print("=" * 60)

    # Create curve adjuster
    curve_adjuster = TonalCurveAdjuster()

    # Generate curves for different grades
    print("\nGenerating contrast curves for grades 0-5:")
    for grade in [0, 1, 2, 3, 4, 5]:
        x, y = curve_adjuster.create_contrast_curve(grade)
        # Show midpoint value
        midpoint_idx = len(y) // 2
        print(f"  Grade {grade}: midpoint input=0.5 → output={y[midpoint_idx]:.3f}")

    # Demonstrate metal characteristics
    print("\nMetal Characteristic Curves:")
    test_image = create_test_image()

    # Apply platinum characteristic
    pt_processed = curve_adjuster.apply_platinum_characteristic(test_image, strength=1.0)
    print(f"  Platinum: mean={pt_processed.mean():.3f}, std={pt_processed.std():.3f}")

    # Apply palladium characteristic
    pd_processed = curve_adjuster.apply_palladium_characteristic(test_image, strength=1.0)
    print(f"  Palladium: mean={pd_processed.mean():.3f}, std={pd_processed.std():.3f}")

    # Blend characteristics
    for pt_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = curve_adjuster.blend_metal_characteristics(
            test_image, pt_ratio=pt_ratio, strength=1.0
        )
        pt_pct = pt_ratio * 100
        pd_pct = (1 - pt_ratio) * 100
        print(f"  {pt_pct:.0f}% Pt / {pd_pct:.0f}% Pd: mean={blended.mean():.3f}")


def example_mask_generation():
    """Example of creating and using masks."""
    print("\n\n" + "=" * 60)
    print("MASK GENERATION EXAMPLE")
    print("=" * 60)

    simulator = SplitGradeSimulator()

    # Create test image with various tonal regions
    test_image = create_complex_test_image()

    # Create shadow mask
    shadow_mask = simulator.create_shadow_mask(test_image, threshold=0.4)
    print("Shadow Mask:")
    print(f"  Shape: {shadow_mask.shape}")
    print(f"  Mean: {shadow_mask.mean():.3f}")
    print(f"  Coverage: {(shadow_mask > 0.5).sum() / shadow_mask.size * 100:.1f}%")

    # Create highlight mask
    highlight_mask = simulator.create_highlight_mask(test_image, threshold=0.7)
    print("\nHighlight Mask:")
    print(f"  Shape: {highlight_mask.shape}")
    print(f"  Mean: {highlight_mask.mean():.3f}")
    print(f"  Coverage: {(highlight_mask > 0.5).sum() / highlight_mask.size * 100:.1f}%")

    # Show how masks complement each other
    overlap = ((shadow_mask > 0.5) & (highlight_mask > 0.5)).sum()
    print(f"\nMask overlap: {overlap} pixels ({overlap / shadow_mask.size * 100:.2f}%)")


def create_test_image(size=(600, 400)):
    """Create a test image with gradient for demonstration."""
    # Create a horizontal gradient
    gradient = np.linspace(0, 1, size[0])
    image = np.tile(gradient, (size[1], 1))
    return image


def create_complex_test_image(size=(600, 400)):
    """Create a more complex test image with various tonal regions."""
    # Create an image with shadows, midtones, and highlights
    image = np.zeros((size[1], size[0]))

    # Dark region (shadows)
    image[:size[1]//3, :size[0]//3] = 0.2

    # Midtone region
    image[size[1]//3:2*size[1]//3, size[0]//3:2*size[0]//3] = 0.5

    # Bright region (highlights)
    image[2*size[1]//3:, 2*size[0]//3:] = 0.85

    # Add a gradient in the center
    center_y = size[1] // 2
    center_x = size[0] // 2
    y, x = np.ogrid[:size[1], :size[0]]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    gradient = 1 - (distance / max_dist)

    # Blend gradient with base image
    image = image * 0.7 + gradient * 0.3

    return np.clip(image, 0, 1)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SPLIT-GRADE PRINTING SIMULATION EXAMPLES")
    print("Platinum/Palladium Digital Negative Tool")
    print("=" * 60)

    try:
        example_basic_usage()
        example_custom_settings()
        example_tonal_curves()
        example_mask_generation()

        print("\n\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
