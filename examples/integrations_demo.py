"""
Demo script showcasing the integrations module.

Demonstrates usage of spectrophotometer, weather, printer, and ICC profile integrations.
"""

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from ptpd_calibration.config import get_settings
from ptpd_calibration.integrations import (
    ApertureSize,
    ColorMode,
    # Printer
    EpsonDriver,
    # ICC Profiles
    ICCProfileManager,
    MeasurementMode,
    MediaType,
    # Weather
    OpenWeatherMapProvider,
    PaperType,
    PrintQuality,
    PrintSettings,
    RenderingIntent,
    SpectroExportFormat,
    # Spectrophotometer
    XRiteIntegration,
)


def demo_spectrophotometer():
    """Demonstrate spectrophotometer integration."""
    print("\n" + "="*60)
    print("SPECTROPHOTOMETER DEMO")
    print("="*60)

    # Initialize X-Rite spectrophotometer (simulated)
    spectro = XRiteIntegration(
        device_id="XRITE-12345",
        mode=MeasurementMode.REFLECTION,
        aperture=ApertureSize.MEDIUM,
        simulate=True
    )

    # Connect to device
    if spectro.connect():
        print(f"✓ Connected to {spectro.device_model}")

        # Calibrate
        cal_result = spectro.calibrate_device()
        if cal_result.success:
            print("✓ Calibration successful")
            print(f"  White reference: L*={cal_result.white_reference.L:.1f}")
            print(f"  Black reference: L*={cal_result.black_reference.L:.1f}")

        # Read a single patch
        print("\nReading single patch...")
        measurement = spectro.read_patch("test_patch_01")
        print(f"  Patch: {measurement.patch_id}")
        print(f"  Density: {measurement.density:.3f}")
        print(f"  L*a*b*: L={measurement.lab.L:.1f}, a={measurement.lab.a:.1f}, b={measurement.lab.b:.1f}")
        print(f"  RGB: {measurement.rgb}")

        # Read a strip of patches
        print("\nReading strip of 21 patches...")
        strip = spectro.read_strip(num_patches=21, patch_prefix="stouffer", delay_seconds=0.1)
        print(f"✓ Read {len(strip)} patches")

        # Show density range
        densities = [m.density for m in strip]
        print(f"  Density range: {min(densities):.3f} - {max(densities):.3f}")

        # Export measurements
        output_path = Path("/tmp/spectro_measurements.csv")
        exported = spectro.export_measurements(
            strip,
            output_path,
            format=SpectroExportFormat.CSV
        )
        print(f"✓ Exported to: {exported}")

        spectro.disconnect()
    else:
        print("✗ Failed to connect to spectrophotometer")


async def demo_weather():
    """Demonstrate weather API integration."""
    print("\n" + "="*60)
    print("WEATHER API DEMO")
    print("="*60)

    settings = get_settings()

    # Initialize weather provider
    weather = OpenWeatherMapProvider(
        api_key=settings.integrations.weather_api_key,  # Will use simulated data if None
        units="metric"
    )

    # Get current conditions
    print("\nFetching current conditions...")
    conditions = await weather.get_current_conditions(
        location="Portland, OR"
    )
    print("✓ Current conditions for Portland, OR:")
    print(f"  Temperature: {conditions.temperature_c:.1f}°C ({conditions.temperature_f:.1f}°F)")
    print(f"  Humidity: {conditions.humidity_percent:.0f}%")
    print(f"  Condition: {conditions.condition.value} - {conditions.description}")
    print(f"  Suitable for coating: {'Yes ✓' if conditions.is_suitable_for_coating else 'No ✗'}")

    # Calculate drying time
    print("\nCalculating paper drying time...")
    drying_estimate = weather.calculate_drying_time(
        conditions,
        paper_type=PaperType.COLD_PRESS
    )
    print("✓ Drying time estimate:")
    print(f"  Paper type: {drying_estimate.paper_type.value}")
    print(f"  Estimated time: {drying_estimate.estimated_hours:.1f} hours")
    print(f"  Confidence: {drying_estimate.confidence:.0%}")
    print("  Recommendations:")
    for rec in drying_estimate.recommendations:
        print(f"    - {rec}")

    # Get forecast and recommend coating time
    print("\nFinding best coating time in next 48 hours...")
    recommendation = await weather.recommend_coating_time(
        location="Portland, OR",
        forecast_hours=48
    )
    print("✓ Best time to coat:")
    print(f"  Recommended: {recommendation.best_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Temperature: {recommendation.forecast.temperature_c:.1f}°C")
    print(f"  Humidity: {recommendation.forecast.humidity_percent:.0f}%")
    print(f"  Reason: {recommendation.reason}")


def demo_printer():
    """Demonstrate printer driver integration."""
    print("\n" + "="*60)
    print("PRINTER DRIVER DEMO")
    print("="*60)

    # Initialize Epson printer (simulated)
    printer = EpsonDriver(
        printer_name="Epson Stylus Photo R2400",
        model="R2400",
        simulate=True
    )

    # Connect
    if printer.connect():
        print(f"✓ Connected to {printer.printer_name}")

        # Get status
        status = printer.get_status()
        print("\nPrinter status:")
        print(f"  Model: {status['model']}")
        print(f"  Connected: {status['connected']}")

        # Check ink levels
        print("\nInk levels:")
        ink_levels = printer.get_ink_levels()
        for color, level in ink_levels.items():
            bar = "█" * int(level.level_percent / 5) + "░" * (20 - int(level.level_percent / 5))
            print(f"  {color:15s}: [{bar}] {level.level_percent:5.1f}% ({level.status})")

        # Run nozzle check
        print("\nRunning nozzle check...")
        nozzle_result = printer.run_nozzle_check()
        if nozzle_result.success:
            print("✓ Nozzle check passed")
            print(f"  Quality: {nozzle_result.pattern_quality:.0%}")
        else:
            print("✗ Nozzle check found issues:")
            for nozzle in nozzle_result.missing_nozzles:
                print(f"    - Missing: {nozzle}")

        # Create a test image (digital negative)
        print("\nCreating test digital negative...")
        test_image = Image.new('L', (800, 600), color=255)
        # Add gradient
        for y in range(600):
            for x in range(800):
                test_image.putpixel((x, y), int(255 * x / 800))

        # Print settings for digital negative
        print_settings = PrintSettings(
            quality=PrintQuality.PHOTO,
            media_type=MediaType.TRANSPARENCY,
            color_mode=ColorMode.GRAYSCALE,
            resolution_dpi=2880,
            invert=True,  # Create negative
            mirror=True,  # Mirror for contact printing
        )

        # Print
        print("\nPrinting digital negative...")
        job = printer.print_negative(test_image, print_settings)
        print(f"✓ Print job {job.job_id} {job.status}")
        print(f"  Settings: {job.settings.quality.value}, {job.settings.resolution_dpi} DPI")

        printer.disconnect()
    else:
        print("✗ Failed to connect to printer")


def demo_icc_profiles():
    """Demonstrate ICC profile management."""
    print("\n" + "="*60)
    print("ICC PROFILE MANAGER DEMO")
    print("="*60)

    # Initialize profile manager
    manager = ICCProfileManager()

    # List installed profiles
    print("\nScanning for installed ICC profiles...")
    profiles = manager.list_installed_profiles()
    print(f"✓ Found {len(profiles)} ICC profiles")

    # Show first few profiles
    if profiles:
        print("\nSample profiles:")
        for profile in profiles[:5]:
            print(f"  - {profile.description}")
            print(f"    Color space: {profile.color_space.value}")
            print(f"    Class: {profile.profile_class.value}")
            print(f"    Size: {profile.size_bytes / 1024:.1f} KB")

    # Get default profiles
    print("\nDefault profiles:")
    srgb_profile = manager.get_default_rgb_profile()
    print("✓ sRGB profile loaded")

    manager.get_default_gray_profile()
    print("✓ Grayscale profile loaded")

    # Create test image
    print("\nCreating test image...")
    test_image = Image.new('RGB', (400, 300))
    pixels = np.array(test_image)
    # Create color gradient
    for y in range(300):
        for x in range(400):
            pixels[y, x] = [int(255 * x / 400), int(255 * y / 300), 128]
    test_image = Image.fromarray(pixels)

    # Apply profile
    print("\nApplying ICC profile to image...")
    profiled_image = manager.apply_profile(
        test_image,
        srgb_profile,
        rendering_intent=RenderingIntent.PERCEPTUAL
    )
    print("✓ Profile applied")
    print(f"  Original size: {test_image.size}")
    print(f"  Profiled size: {profiled_image.size}")

    # Embed profile
    print("\nEmbedding profile in image...")
    embedded_image = manager.embed_profile(profiled_image, srgb_profile)
    has_profile = manager.extract_profile(embedded_image) is not None
    print(f"✓ Profile embedded: {has_profile}")


async def main():
    """Run all integration demos."""
    print("\n")
    print("╔" + "="*60 + "╗")
    print("║" + " "*10 + "PTPD INTEGRATIONS MODULE DEMO" + " "*20 + "║")
    print("╚" + "="*60 + "╝")

    try:
        # Run demos
        demo_spectrophotometer()
        await demo_weather()
        demo_printer()
        demo_icc_profiles()

        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async demo
    asyncio.run(main())
