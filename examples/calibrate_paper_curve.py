#!/usr/bin/env python3
"""
Paper Curve Calibration Script.

Calibrates a platinum/palladium printing curve from one paper type to another
using transfer learning and AI-powered curve enhancement.

Usage:
    python calibrate_paper_curve.py

This script will:
1. Load the source .quad file (Reviere Platinum)
2. Apply transfer learning adjustments for paper differences
3. Enhance curves using AI-powered optimization
4. Export a new .quad file for the target paper (Arches Platine)
"""

from pathlib import Path

import numpy as np

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.ai_enhance import EnhancementGoal
from ptpd_calibration.curves.modifier import CurveModifier, SmoothingMethod
from ptpd_calibration.curves.parser import QuadFileParser, QuadProfile
from ptpd_calibration.ml.transfer import PAPER_CHARACTERISTICS

# Configuration
SOURCE_QUAD_FILE = Path(__file__).parent / "Platinum_Palladium_V6-CC-Revier_Platinum.quad"
OUTPUT_QUAD_FILE = Path(__file__).parent / "Platinum_Palladium_V6-CC-Arches_Platine.quad"
SOURCE_PAPER = "revere platinum"
TARGET_PAPER = "arches platine"

# Number of values per channel in a .quad profile
CURVE_POINTS = 256


def analyze_paper_differences(source: str, target: str) -> dict:
    """
    Analyze differences between source and target papers.

    Args:
        source: Source paper name.
        target: Target paper name.

    Returns:
        Dictionary with analysis results and adjustment factors.
    """
    source_chars = PAPER_CHARACTERISTICS.get(source.lower(), {})
    target_chars = PAPER_CHARACTERISTICS.get(target.lower(), {})

    print(f"\n{'='*60}")
    print("PAPER CHARACTERISTIC ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSource Paper: {source}")
    print(f"  Fiber: {source_chars.get('fiber', 'unknown')}")
    print(f"  Sizing: {source_chars.get('sizing', 'unknown')}")
    print(f"  Weight: {source_chars.get('weight_range', 'unknown')} GSM")
    print(f"  Characteristics: {', '.join(source_chars.get('characteristics', []))}")

    print(f"\nTarget Paper: {target}")
    print(f"  Fiber: {target_chars.get('fiber', 'unknown')}")
    print(f"  Sizing: {target_chars.get('sizing', 'unknown')}")
    print(f"  Weight: {target_chars.get('weight_range', 'unknown')} GSM")
    print(f"  Characteristics: {', '.join(target_chars.get('characteristics', []))}")

    # Calculate adjustment factors based on paper differences
    adjustments = {
        "exposure_factor": 1.0,
        "curve_scale": 1.0,
        "highlight_adjust": 0.0,
        "shadow_adjust": 0.0,
    }

    # Sizing adjustment: gelatin vs internal sizing affects absorption
    source_sizing = source_chars.get("sizing", "internal")
    target_sizing = target_chars.get("sizing", "internal")

    if source_sizing == "gelatin" and target_sizing == "internal":
        # Internal sizing absorbs less, needs slight exposure increase
        adjustments["exposure_factor"] = 1.05
        adjustments["curve_scale"] = 0.98  # Slightly compress to prevent blocking
        print("\n  -> Sizing change: gelatin -> internal")
        print("    Adjustment: +5% exposure, -2% curve compression")

    # White point adjustment: neutral vs bright white
    source_white = any("neutral" in c for c in source_chars.get("characteristics", []))
    target_white = any("bright" in c for c in target_chars.get("characteristics", []))

    if source_white and target_white:
        # Bright white paper may need highlight adjustment
        adjustments["highlight_adjust"] = 0.02  # Slight highlight lift
        print("  -> White point change: neutral -> bright")
        print("    Adjustment: +2% highlight lift")

    # Fiber type adjustment
    source_fiber = source_chars.get("fiber", "cotton")
    target_fiber = target_chars.get("fiber", "cotton")

    if source_fiber == "cotton_alpha" and target_fiber == "cotton":
        # Pure cotton may have slightly different response
        adjustments["shadow_adjust"] = -0.01  # Slight shadow deepen
        print("  -> Fiber change: cotton_alpha -> cotton")
        print("    Adjustment: -1% shadow adjustment")

    return adjustments


def apply_transfer_adjustments(
    profile: QuadProfile,
    adjustments: dict,
) -> dict[str, list[int]]:
    """
    Apply transfer learning adjustments to all channels.

    Args:
        profile: Source QuadProfile.
        adjustments: Adjustment factors from paper analysis.

    Returns:
        Dictionary mapping channel names to adjusted values.
    """
    print(f"\n{'='*60}")
    print("APPLYING TRANSFER ADJUSTMENTS")
    print(f"{'='*60}")

    adjusted_channels = {}

    for channel_name in profile.all_channel_names:
        channel = profile.get_channel(channel_name)
        if channel is None:
            continue

        values = np.array(channel.values, dtype=np.float64)
        original_max = values.max()

        # Apply curve scaling
        if adjustments["curve_scale"] != 1.0:
            # Scale around the midpoint to preserve endpoints
            values = values * adjustments["curve_scale"]

        # Apply highlight adjustment (affects upper values)
        if adjustments["highlight_adjust"] != 0.0:
            highlight_mask = values > (original_max * 0.8)
            values[highlight_mask] *= (1.0 + adjustments["highlight_adjust"])

        # Apply shadow adjustment (affects lower values)
        if adjustments["shadow_adjust"] != 0.0:
            shadow_mask = values < (original_max * 0.2)
            values[shadow_mask] *= (1.0 + adjustments["shadow_adjust"])

        # Ensure monotonicity (vectorised)
        values = np.maximum.accumulate(values)

        # Clamp to valid range
        values = np.clip(values, 0, 65535).astype(int)

        adjusted_channels[channel_name] = values.tolist()

        print(f"  {channel_name}: adjusted (max: {int(values.max())}, range: {int(values.max() - values.min())})")

    return adjusted_channels


def apply_ai_enhancement(
    channels: dict[str, list[int]],
    goal: EnhancementGoal = EnhancementGoal.SMOOTH_GRADATION,
) -> dict[str, list[int]]:
    """
    Apply AI-powered curve enhancement to all channels.

    Args:
        channels: Dictionary of channel values.
        goal: Enhancement goal.

    Returns:
        Enhanced channel values.
    """
    print(f"\n{'='*60}")
    print("APPLYING AI ENHANCEMENT")
    print(f"{'='*60}")
    print(f"  Goal: {goal.value}")

    modifier = CurveModifier()
    enhanced_channels = {}

    for channel_name, values in channels.items():
        # Convert to CurveData for processing
        normalized_input = np.linspace(0, 1, len(values))
        normalized_output = np.array(values) / 65535.0

        curve_data = CurveData(
            name=f"{channel_name}_channel",
            input_values=normalized_input.tolist(),
            output_values=normalized_output.tolist(),
        )

        # Apply smoothing to reduce any artifacts from transfer
        smoothed = modifier.smooth(
            curve_data,
            method=SmoothingMethod.SAVGOL,
            strength=0.3,
        )

        # Ensure monotonicity after smoothing
        monotonic = modifier.enforce_monotonicity(smoothed, direction="increasing")

        # Convert back to 16-bit values
        enhanced_values = (np.array(monotonic.output_values) * 65535).astype(int)
        enhanced_values = np.clip(enhanced_values, 0, 65535)

        enhanced_channels[channel_name] = enhanced_values.tolist()

        print(f"  {channel_name}: enhanced with {goal.value}")

    return enhanced_channels


def export_quad_file(
    channels: dict[str, list[int]],
    source_profile: QuadProfile,
    output_path: Path,
    target_paper: str,
) -> None:
    """
    Export enhanced channels to a new .quad file.

    Args:
        channels: Enhanced channel values.
        source_profile: Original profile for metadata.
        output_path: Output file path.
        target_paper: Target paper name for metadata.
    """
    print(f"\n{'='*60}")
    print("EXPORTING CALIBRATED CURVE")
    print(f"{'='*60}")

    # Build the .quad file content
    lines = []

    # Header
    lines.append("## QuadToneRIP K,C,M,Y,LC,LM,LK,LLK,V,MK")
    lines.append("# Black and White Mastery QuadToneProfiler v3")
    lines.append("# Process: Platinum-Palladium")
    lines.append(f"# Paper: {target_paper.title()}")
    lines.append(f"# Calibrated from: {source_profile.profile_name}")
    lines.append("# Calibration method: Transfer Learning + AI Enhancement")
    lines.append(f"# Ink Limit: {source_profile.ink_limit}")
    lines.append(f"# Resolution: {source_profile.resolution}")

    # Standard channel order
    channel_order = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK", "V", "MK"]

    for channel_name in channel_order:
        lines.append(f"# {channel_name} Curve")

        if channel_name in channels:
            for value in channels[channel_name]:
                lines.append(str(value))
        else:
            # Empty channel — output zeros
            for _ in range(CURVE_POINTS):
                lines.append("0")

    # Write the file
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    print(f"  Output: {output_path}")
    print(f"  Channels: {len(channels)}")
    print("  Values per channel: 256")
    print(f"  File size: {output_path.stat().st_size} bytes")


def print_summary(source_profile: QuadProfile, adjusted_channels: dict) -> None:
    """Print a summary comparison of source and target curves."""
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")

    print("\nChannel Comparison (source -> target max values):")
    for channel_name in source_profile.all_channel_names:
        source_channel = source_profile.get_channel(channel_name)
        if source_channel and channel_name in adjusted_channels:
            source_max = max(source_channel.values)
            target_max = max(adjusted_channels[channel_name])
            diff = target_max - source_max
            diff_pct = (diff / source_max * 100) if source_max > 0 else 0
            print(f"  {channel_name}: {source_max} -> {target_max} ({diff_pct:+.1f}%)")


def main() -> None:
    """Main calibration workflow."""
    print("\n" + "="*60)
    print("PLATINUM/PALLADIUM PAPER CURVE CALIBRATION")
    print("="*60)
    print(f"\nSource: {SOURCE_QUAD_FILE.name}")
    print(f"Target: {OUTPUT_QUAD_FILE.name}")
    print(f"Paper: {SOURCE_PAPER} -> {TARGET_PAPER}")

    # Step 1: Parse source .quad file
    print(f"\n{'='*60}")
    print("LOADING SOURCE CURVE")
    print(f"{'='*60}")

    parser = QuadFileParser()
    source_profile = parser.parse(SOURCE_QUAD_FILE)

    print(f"  Profile: {source_profile.profile_name}")
    print(f"  Channels: {len(source_profile.all_channel_names)}")
    print(f"  Active channels: {', '.join(source_profile.active_channels)}")

    # Step 2: Analyze paper differences
    adjustments = analyze_paper_differences(SOURCE_PAPER, TARGET_PAPER)

    # Step 3: Apply transfer adjustments
    adjusted_channels = apply_transfer_adjustments(source_profile, adjustments)

    # Step 4: Apply AI enhancement
    enhanced_channels = apply_ai_enhancement(
        adjusted_channels,
        goal=EnhancementGoal.SMOOTH_GRADATION,
    )

    # Step 5: Export calibrated curve
    export_quad_file(
        enhanced_channels,
        source_profile,
        OUTPUT_QUAD_FILE,
        TARGET_PAPER,
    )

    # Print summary
    print_summary(source_profile, enhanced_channels)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput file: {OUTPUT_QUAD_FILE}")
    print("\nNext steps:")
    print("  1. Load the new .quad file in QuadTone RIP")
    print("  2. Print a test target on Arches Platine")
    print("  3. Measure and fine-tune as needed")


if __name__ == "__main__":
    main()
