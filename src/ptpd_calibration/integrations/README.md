# Integrations Module

The integrations module provides interfaces for hardware devices and external services commonly used in platinum/palladium printing workflows.

## Overview

- **Spectrophotometer**: Density and color measurements
- **Weather API**: Environmental condition monitoring
- **Printer Drivers**: Digital negative printing
- **ICC Profiles**: Color management

## Components

### 1. Spectrophotometer Integration

Interface for spectrophotometer devices (X-Rite, etc.) for measuring density and L*a*b* values.

#### Features

- Device calibration (white/black reference)
- Single patch measurement
- Strip/batch measurements
- Export to CGATS, CSV, JSON formats
- Spectral data capture

#### Example Usage

```python
from ptpd_calibration.integrations import (
    XRiteIntegration,
    MeasurementMode,
    ApertureSize,
    SpectroExportFormat
)

# Initialize device (simulated)
spectro = XRiteIntegration(
    device_id="XRITE-12345",
    mode=MeasurementMode.REFLECTION,
    aperture=ApertureSize.MEDIUM,
    simulate=True
)

# Connect and calibrate
spectro.connect()
cal_result = spectro.calibrate_device()

# Measure single patch
measurement = spectro.read_patch("patch_01")
print(f"Density: {measurement.density:.3f}")
print(f"L*a*b*: {measurement.lab.L:.1f}, {measurement.lab.a:.1f}, {measurement.lab.b:.1f}")

# Measure strip (21-step tablet)
strip = spectro.read_strip(num_patches=21, patch_prefix="stouffer")

# Export measurements
spectro.export_measurements(
    strip,
    output_path="measurements.csv",
    format=SpectroExportFormat.CSV
)

spectro.disconnect()
```

### 2. Weather API Integration

Monitor environmental conditions to optimize coating and drying times.

#### Features

- Current conditions (temperature, humidity)
- Weather forecasts
- Paper drying time estimation
- Coating time recommendations

#### Example Usage

```python
import asyncio
from ptpd_calibration.integrations import (
    OpenWeatherMapProvider,
    PaperType
)

async def main():
    # Initialize provider
    weather = OpenWeatherMapProvider(
        api_key="your_api_key",  # Or None for simulated data
        units="metric"
    )

    # Get current conditions
    conditions = await weather.get_current_conditions(
        location="Portland, OR"
    )
    print(f"Temp: {conditions.temperature_c:.1f}°C")
    print(f"Humidity: {conditions.humidity_percent:.0f}%")
    print(f"Suitable for coating: {conditions.is_suitable_for_coating}")

    # Calculate drying time
    drying = weather.calculate_drying_time(
        conditions,
        paper_type=PaperType.COLD_PRESS
    )
    print(f"Estimated drying time: {drying.estimated_hours:.1f} hours")
    print(f"Recommendations: {drying.recommendations}")

    # Find best coating time
    recommendation = await weather.recommend_coating_time(
        location="Portland, OR",
        forecast_hours=48
    )
    print(f"Best time to coat: {recommendation.best_time}")
    print(f"Reason: {recommendation.reason}")

asyncio.run(main())
```

### 3. Printer Driver Integration

Interface for inkjet printers used for digital negatives.

#### Supported Printers

- **Epson**: P800, P900, 3880, 7900, R2400, etc.
- **Canon**: PRO-1000, PRO-2000, PRO-100, iPF series

#### Features

- ICC profile management
- Digital negative printing (inverted, mirrored)
- Ink level monitoring
- Nozzle check diagnostics

#### Example Usage

```python
from ptpd_calibration.integrations import (
    EpsonDriver,
    PrintSettings,
    PrintQuality,
    MediaType,
    ColorMode
)
from PIL import Image

# Initialize printer (simulated)
printer = EpsonDriver(
    printer_name="Epson Stylus Photo R2400",
    model="R2400",
    simulate=True
)

# Connect
printer.connect()

# Check ink levels
ink_levels = printer.get_ink_levels()
for color, level in ink_levels.items():
    print(f"{color}: {level.level_percent:.0f}% ({level.status})")

# Run nozzle check
nozzle_result = printer.run_nozzle_check()
if nozzle_result.success:
    print("Nozzle check passed!")

# Set ICC profile
printer.set_profile("path/to/profile.icc")

# Print digital negative
image = Image.open("negative.tif")
settings = PrintSettings(
    quality=PrintQuality.PHOTO,
    media_type=MediaType.TRANSPARENCY,
    color_mode=ColorMode.GRAYSCALE,
    resolution_dpi=2880,
    invert=True,   # Create negative
    mirror=True,   # Mirror for contact printing
)

job = printer.print_negative(image, settings)
print(f"Print job {job.job_id}: {job.status}")

printer.disconnect()
```

### 4. ICC Profile Management

Load, apply, create, and manage ICC color profiles.

#### Features

- Load system and custom ICC profiles
- Apply profiles to images
- Convert between color spaces
- Create custom profiles from measurements
- Profile validation
- Embed/extract profiles in images

#### Example Usage

```python
from ptpd_calibration.integrations import (
    ICCProfileManager,
    RenderingIntent,
    ColorSpace
)
from PIL import Image

# Initialize manager
manager = ICCProfileManager(
    custom_profile_dir="/path/to/custom/profiles"
)

# List installed profiles
profiles = manager.list_installed_profiles(
    color_space=ColorSpace.RGB
)
print(f"Found {len(profiles)} RGB profiles")

# Load specific profile
profile = manager.load_profile("path/to/profile.icc")

# Apply profile to image
image = Image.open("image.jpg")
profiled_image = manager.apply_profile(
    image,
    profile,
    rendering_intent=RenderingIntent.PERCEPTUAL
)

# Convert between color spaces
source_profile = manager.get_default_rgb_profile()
target_profile = manager.get_default_gray_profile()

converted = manager.convert_colorspace(
    image,
    source_profile,
    target_profile,
    rendering_intent=RenderingIntent.RELATIVE_COLORIMETRIC
)

# Embed profile in image
embedded = manager.embed_profile(image, profile)
embedded.save("output_with_profile.jpg")

# Validate profile
validation = manager.validate_profile("profile.icc")
if validation.is_valid:
    print("Profile is valid!")
else:
    print(f"Errors: {validation.errors}")
```

## Configuration

Integration settings are configured via environment variables or the config system:

```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Weather API
api_key = settings.integrations.weather_api_key
location = settings.integrations.weather_location

# Spectrophotometer
device_id = settings.integrations.spectro_device_id
simulate = settings.integrations.spectro_simulate

# Printer
printer_name = settings.integrations.default_printer_name
printer_brand = settings.integrations.default_printer_brand

# ICC Profiles
custom_dir = settings.integrations.custom_profile_dir
rendering_intent = settings.integrations.default_rendering_intent
```

### Environment Variables

```bash
# Weather API
export PTPD_INTEGRATION_WEATHER_API_KEY="your_openweathermap_key"
export PTPD_INTEGRATION_WEATHER_LOCATION="Portland, OR"

# Spectrophotometer
export PTPD_INTEGRATION_SPECTRO_DEVICE_ID="/dev/ttyUSB0"
export PTPD_INTEGRATION_SPECTRO_SIMULATE=false

# Printer
export PTPD_INTEGRATION_DEFAULT_PRINTER_NAME="Epson R2400"
export PTPD_INTEGRATION_PRINTER_SIMULATE=false

# ICC Profiles
export PTPD_INTEGRATION_CUSTOM_PROFILE_DIR="/home/user/icc_profiles"
```

## Simulation Mode

All hardware integrations support simulation mode for development and testing:

- **Spectrophotometer**: Generates realistic density and L*a*b* measurements
- **Printer**: Simulates printing, ink levels, and nozzle checks
- **Weather**: Returns simulated weather data when no API key provided

This allows complete testing without physical hardware.

## Dependencies

- `numpy` - Numerical operations
- `pillow` - Image processing
- `httpx` - Async HTTP client (weather API)
- `pydantic` - Data validation

## Notes

### Real Hardware Integration

For production use with real hardware:

1. **Spectrophotometer**: Set `simulate=False` and implement device-specific communication (USB/SDK)
2. **Printer**: Set `simulate=False` and use CUPS/Windows Print Spooler
3. **Weather**: Provide valid OpenWeatherMap API key
4. **ICC Profiles**: For custom profile creation, use Argyll CMS or similar tools

### OpenWeatherMap API

Free tier includes:
- Current weather conditions
- 5-day forecast (3-hour intervals)
- 60 calls/minute, 1,000,000 calls/month

Get API key at: https://openweathermap.org/api

## Examples

See `/examples/integrations_demo.py` for comprehensive usage examples.

## License

Part of the Platinum-Palladium AI Printing Tool.
