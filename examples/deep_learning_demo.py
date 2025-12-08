"""
Demonstration of UV Exposure Predictor and Defect Detector.

This script shows how to use the newly implemented deep learning features:
1. UV Exposure Neural Predictor
2. Automated Defect Detection (U-Net + ResNet)

Note: Requires PyTorch to be installed: pip install torch torchvision
"""

import numpy as np
from pathlib import Path

# Import the deep learning components
from ptpd_calibration.deep_learning import (
    # UV Exposure
    UVExposurePredictor,
    UVExposureSettings,
    UVSourceType,
    # Defect Detection
    DefectDetector,
    DefectDetectionSettings,
)


def demo_uv_exposure_predictor():
    """Demonstrate UV exposure prediction."""
    print("=" * 80)
    print("UV Exposure Neural Predictor Demo")
    print("=" * 80)

    # Initialize predictor with default settings
    settings = UVExposureSettings()
    predictor = UVExposurePredictor(settings=settings)

    # Example prediction
    print("\nPredicting UV exposure time...")
    prediction = predictor.predict(
        target_density=0.7,
        paper_type="Arches Platine",
        chemistry_ratio=0.5,  # 50/50 Pt/Pd
        uv_source=UVSourceType.NUARC_26_1K,
        humidity=55.0,
        temperature=20.0,
        coating_thickness=0.6,
        negative_dmax=0.95,
    )

    print(f"\nPredicted exposure time: {prediction.format_time()}")
    print(f"Confidence interval: {prediction.lower_bound_seconds:.1f}s - {prediction.upper_bound_seconds:.1f}s")
    print(f"Confidence level: {prediction.confidence_level * 100:.0f}%")

    print("\nFactor contributions:")
    for factor, contribution in prediction.factor_contributions.items():
        print(f"  {factor}: {contribution:+.3f}")

    if prediction.recommendations:
        print("\nRecommendations:")
        for rec in prediction.recommendations:
            print(f"  - {rec}")

    if prediction.warnings:
        print("\nWarnings:")
        for warning in prediction.warnings:
            print(f"  ⚠ {warning}")

    print(f"\nInference time: {prediction.inference_time_ms:.2f}ms")


def demo_defect_detector():
    """Demonstrate defect detection."""
    print("\n" + "=" * 80)
    print("Automated Defect Detection Demo")
    print("=" * 80)

    # Initialize detector with default settings
    settings = DefectDetectionSettings()
    detector = DefectDetector(settings=settings)

    # Create a sample image (in production, this would be a real print scan)
    print("\nCreating sample image...")
    sample_image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

    # Detect defects
    print("Detecting defects...")
    result = detector.detect(sample_image)

    print(f"\nDefects detected: {result.num_defects}")
    print(f"Overall severity: {result.overall_severity.value}")
    print(f"Print acceptable: {'Yes' if result.print_acceptable else 'No'}")
    print(f"Defect coverage: {result.defect_coverage:.2f}%")

    if result.defects_by_type:
        print("\nDefects by type:")
        for defect_type, count in result.defects_by_type.items():
            print(f"  {defect_type}: {count}")

    if result.defects_by_severity:
        print("\nDefects by severity:")
        for severity, count in result.defects_by_severity.items():
            print(f"  {severity}: {count}")

    if result.defects:
        print("\nTop defects:")
        for i, defect in enumerate(result.defects[:5], 1):
            print(f"  {i}. {defect.defect_type.value}")
            print(f"     Severity: {defect.severity.value}")
            print(f"     Confidence: {defect.confidence:.2%}")
            print(f"     Area: {defect.area_pixels} pixels ({defect.area_percentage:.2f}%)")
            if defect.remediation:
                print(f"     Remediation: {defect.remediation}")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

    print(f"\nInference time: {result.inference_time_ms:.2f}ms")


def demo_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n" + "=" * 80)
    print("Custom Configuration Demo")
    print("=" * 80)

    # Custom UV exposure settings
    uv_settings = UVExposureSettings(
        hidden_layers=[512, 256, 128, 64],  # Larger network
        activation="gelu",
        learning_rate=5e-4,
        ensemble_size=10,  # More models for better uncertainty
        confidence_level=0.99,  # Higher confidence
    )
    print("\nCustom UV Exposure Settings:")
    print(f"  Hidden layers: {uv_settings.hidden_layers}")
    print(f"  Ensemble size: {uv_settings.ensemble_size}")
    print(f"  Confidence level: {uv_settings.confidence_level}")

    # Custom defect detection settings
    defect_settings = DefectDetectionSettings(
        confidence_threshold=0.7,  # Higher threshold for more precision
        min_defect_area=50,  # Smaller defects
        use_multi_scale=True,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5],  # More scales
        apply_morphological_cleanup=True,
        merge_nearby_defects=True,
    )
    print("\nCustom Defect Detection Settings:")
    print(f"  Confidence threshold: {defect_settings.confidence_threshold}")
    print(f"  Multi-scale: {defect_settings.use_multi_scale}")
    print(f"  Scales: {defect_settings.scales}")


if __name__ == "__main__":
    try:
        print("\nPlatinum-Palladium AI Printing Tool")
        print("Deep Learning Features Demo\n")

        # Run demos
        demo_uv_exposure_predictor()
        demo_defect_detector()
        demo_custom_configuration()

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install torch torchvision numpy")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
