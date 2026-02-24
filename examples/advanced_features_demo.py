#!/usr/bin/env python3
"""
Demo script for advanced features in Pt/Pd printing workflow.

Demonstrates usage of:
- AlternativeProcessSimulator
- NegativeBlender
- QRMetadataGenerator (requires qrcode library)
- StyleTransfer
- PrintComparison
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ptpd_calibration.advanced import (
    AlternativeProcessParams,
    AlternativeProcessSimulator,
    BlendMode,
    HistoricStyle,
    NegativeBlender,
    PrintComparison,
    PrintMetadata,
    StyleTransfer,
)

# Note: QRMetadataGenerator requires optional dependency
# Install with: pip install qrcode[pil]
try:
    from ptpd_calibration.advanced import QRMetadataGenerator
    HAS_QR = True
except ImportError:
    HAS_QR = False
    print("QRMetadataGenerator not available. Install with: pip install qrcode[pil]")


def demo_alternative_processes():
    """Demonstrate alternative process simulation."""
    print("\n=== Alternative Process Simulation ===")

    # Create a sample image (or load your own)
    sample_image = Image.new('L', (800, 600), color=128)
    # Add some tonal variation
    arr = np.array(sample_image)
    for i in range(arr.shape[0]):
        arr[i, :] = int(255 * i / arr.shape[0])
    sample_image = Image.fromarray(arr)

    simulator = AlternativeProcessSimulator()

    # Simulate cyanotype
    cyanotype = simulator.simulate_cyanotype(sample_image)
    print("✓ Cyanotype simulation created")

    # Simulate Van Dyke brown
    vandyke = simulator.simulate_vandyke(sample_image)
    print("✓ Van Dyke brown simulation created")

    # Simulate kallitype
    kallitype = simulator.simulate_kallitype(sample_image)
    print("✓ Kallitype simulation created")

    # Simulate gum bichromate with custom pigment (sepia)
    gum = simulator.simulate_gum_bichromate(
        sample_image,
        pigment_color=(80, 60, 40)  # Sepia tone
    )
    print("✓ Gum bichromate simulation created")

    # Simulate salt print
    salt = simulator.simulate_salt_print(sample_image)
    print("✓ Salt print simulation created")

    # Custom process parameters
    custom_params = AlternativeProcessParams(
        gamma=1.15,
        contrast=1.2,
        shadow_color=(30, 25, 20),
        highlight_color=(240, 235, 225),
        dmax=1.75,
        dmin=0.09,
    )
    custom = simulator._apply_process_simulation(
        sample_image, custom_params, "Custom Process"
    )
    print("✓ Custom process simulation created")

    return {
        'cyanotype': cyanotype,
        'vandyke': vandyke,
        'kallitype': kallitype,
        'gum': gum,
        'salt': salt,
        'custom': custom,
    }


def demo_negative_blending():
    """Demonstrate negative blending and masking."""
    print("\n=== Negative Blending ===")

    # Create sample negatives
    neg1 = Image.new('L', (500, 500), color=100)
    neg2 = Image.new('L', (500, 500), color=150)

    # Add some content to negatives
    arr1 = np.array(neg1)
    arr2 = np.array(neg2)

    # Add gradient to neg1
    for i in range(arr1.shape[0]):
        arr1[i, :] = int(255 * i / arr1.shape[0])

    # Add radial gradient to neg2
    center_x, center_y = arr2.shape[1] // 2, arr2.shape[0] // 2
    for i in range(arr2.shape[0]):
        for j in range(arr2.shape[1]):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            arr2[i, j] = int(255 * (1 - dist / (arr2.shape[0] / 2)))

    neg1 = Image.fromarray(arr1)
    neg2 = Image.fromarray(arr2)

    blender = NegativeBlender()

    # Simple blend
    blended = blender.blend_negatives([neg1, neg2])
    print("✓ Simple negative blend created")

    # Blend with custom blend mode
    blended_multiply = blender.blend_negatives(
        [neg1, neg2],
        blend_modes=[BlendMode.NORMAL, BlendMode.MULTIPLY]
    )
    print("✓ Multiply blend created")

    # Create masks
    blender.create_contrast_mask(neg1, threshold=0.5)
    print("✓ Contrast mask created")

    highlight_mask = blender.create_highlight_mask(neg1, threshold=0.7)
    print("✓ Highlight mask created")

    shadow_mask = blender.create_shadow_mask(neg1, threshold=0.3)
    print("✓ Shadow mask created")

    # Apply dodge and burn
    dodged_burned = blender.apply_dodge_burn(
        neg1,
        dodge_mask=shadow_mask,
        burn_mask=highlight_mask,
        dodge_amount=0.3,
        burn_amount=0.2
    )
    print("✓ Dodge and burn applied")

    # Multi-layer mask
    blender.create_multi_layer_mask(
        [highlight_mask, shadow_mask],
        blend_modes=['multiply', 'add']
    )
    print("✓ Multi-layer mask created")

    return {
        'blended': blended,
        'blended_multiply': blended_multiply,
        'dodged_burned': dodged_burned,
    }


def demo_qr_metadata():
    """Demonstrate QR code metadata generation."""
    print("\n=== QR Metadata Generation ===")

    if not HAS_QR:
        print("⚠ QR code generation requires qrcode library")
        print("  Install with: pip install qrcode[pil]")
        return None

    generator = QRMetadataGenerator()

    # Create print metadata
    metadata = PrintMetadata(
        title="Moonrise Over Mountain",
        artist="Ansel Adams",
        date="2024-11-30",
        edition="1/25",
        paper="Arches Platine",
        chemistry="Pure Platinum",
        exposure_time="12 minutes @ UV-A",
        developer="Potassium Oxalate",
        curve_name="Linear_Arches_v1",
        dmax=1.75,
        dmin=0.08,
        notes="First print in new series",
    )

    # Generate QR code
    qr_code = generator.generate_print_qr(metadata, size=200)
    print("✓ QR code generated")

    # Create archival label
    label = generator.create_archival_label(metadata, label_size=(600, 300))
    print("✓ Archival label created")

    # Encode/decode recipe
    encoded = generator.encode_recipe(metadata)
    print(f"✓ Encoded metadata: {encoded[:50]}...")

    return {
        'qr_code': qr_code,
        'label': label,
        'metadata': metadata,
    }


def demo_style_transfer():
    """Demonstrate historic style transfer."""
    print("\n=== Historic Style Transfer ===")

    # Create sample image
    sample = Image.new('L', (600, 400), color=128)
    arr = np.array(sample)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = int(128 + 127 * np.sin(i / 50) * np.cos(j / 50))
    sample = Image.fromarray(arr)

    transfer = StyleTransfer()

    # List available styles
    print(f"Available historic styles: {len(transfer.styles)}")
    for style_name in transfer.styles:
        if hasattr(style_name, 'value'):
            print(f"  - {style_name.value}")

    # Apply Edward Weston style
    weston_style = transfer.apply_style(sample, HistoricStyle.EDWARD_WESTON)
    print("✓ Edward Weston style applied")

    # Apply Irving Penn style
    penn_style = transfer.apply_style(sample, HistoricStyle.IRVING_PENN)
    print("✓ Irving Penn style applied")

    # Apply Sally Mann style
    mann_style = transfer.apply_style(sample, HistoricStyle.SALLY_MANN)
    print("✓ Sally Mann style applied")

    # Analyze a reference image
    analyzed_style = transfer.analyze_style(sample)
    print(f"✓ Style analyzed: gamma={analyzed_style.gamma:.2f}, "
          f"contrast={analyzed_style.contrast:.2f}")

    # Create custom style
    custom_style = transfer.create_custom_style(
        "My Custom Style",
        {
            'description': 'Personal interpretation',
            'gamma': 1.1,
            'contrast': 1.15,
            'shadow_tone': (25, 22, 20),
            'highlight_tone': (245, 242, 238),
            'dmax': 1.7,
            'dmin': 0.08,
        }
    )
    print("✓ Custom style created")

    custom_applied = transfer._apply_style_params(sample, custom_style)
    print("✓ Custom style applied")

    return {
        'weston': weston_style,
        'penn': penn_style,
        'mann': mann_style,
        'custom': custom_applied,
    }


def demo_print_comparison():
    """Demonstrate print comparison tools."""
    print("\n=== Print Comparison ===")

    # Create sample original and "scanned print"
    original = Image.new('L', (400, 400), color=128)
    arr = np.array(original)
    for i in range(arr.shape[0]):
        arr[i, :] = int(255 * i / arr.shape[0])
    original = Image.fromarray(arr)

    # Simulate a scanned print (with slight variations)
    print_scan = original.copy()
    scan_arr = np.array(print_scan)
    scan_arr = (scan_arr * 0.95 + 10).astype(np.uint8)  # Slight shift
    print_scan = Image.fromarray(scan_arr)

    comparison = PrintComparison()

    # Compare before/after
    metrics = comparison.compare_before_after(original, print_scan)
    print("✓ Before/after comparison completed")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Similarity: {metrics['similarity_score']:.4f}")
    print(f"  Histogram correlation: {metrics['histogram_correlation']:.4f}")

    # Generate difference map
    diff_map = comparison.generate_difference_map(original, print_scan, colorize=True)
    print("✓ Difference map generated")

    # Calculate similarity scores
    ssim_score = comparison.calculate_similarity_score(
        original, print_scan, method='ssim'
    )
    corr_score = comparison.calculate_similarity_score(
        original, print_scan, method='correlation'
    )
    print(f"✓ SSIM similarity: {ssim_score:.4f}")
    print(f"✓ Correlation similarity: {corr_score:.4f}")

    # Generate full comparison report
    images = {
        'original': original,
        'print_1': print_scan,
        'print_2': original,  # Use same for demo
    }
    report = comparison.generate_comparison_report(images, reference_key='original')
    print("✓ Comparison report generated")
    print(f"  Compared {report['num_images']} images")
    print(f"  Average similarity: {report['summary']['average_similarity']:.4f}")

    return {
        'difference_map': diff_map,
        'metrics': metrics,
        'report': report,
    }


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Platinum/Palladium Advanced Features Demo")
    print("=" * 60)

    # Run all demos
    alt_processes = demo_alternative_processes()
    blended = demo_negative_blending()
    qr_results = demo_qr_metadata()
    styled = demo_style_transfer()
    compared = demo_print_comparison()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    # Optionally save results
    output_dir = Path("./advanced_features_output")
    output_dir.mkdir(exist_ok=True)

    # Save alternative process examples
    if alt_processes:
        for name, img in alt_processes.items():
            img.save(output_dir / f"alt_process_{name}.jpg")
        print(f"\n✓ Alternative process examples saved to {output_dir}")

    # Save blending examples
    if blended:
        for name, img in blended.items():
            img.save(output_dir / f"blend_{name}.png")
        print(f"✓ Blending examples saved to {output_dir}")

    # Save QR code examples
    if qr_results:
        if 'qr_code' in qr_results:
            qr_results['qr_code'].save(output_dir / "metadata_qr.png")
        if 'label' in qr_results:
            qr_results['label'].save(output_dir / "archival_label.png")
        print(f"✓ QR code examples saved to {output_dir}")

    # Save style examples
    if styled:
        for name, img in styled.items():
            img.save(output_dir / f"style_{name}.jpg")
        print(f"✓ Style transfer examples saved to {output_dir}")

    # Save comparison examples
    if compared and 'difference_map' in compared:
        compared['difference_map'].save(output_dir / "difference_map.png")
        print(f"✓ Comparison examples saved to {output_dir}")

    print(f"\nAll output saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
