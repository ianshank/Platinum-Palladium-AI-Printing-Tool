"""Shared constants for hardware integration.

This module centralizes configuration values used across hardware drivers
to ensure consistency and enable easy configuration changes.
"""

from typing import Final

# =============================================================================
# PRINTER CONSTANTS
# =============================================================================

# Standard paper sizes supported across all printer drivers
STANDARD_PAPER_SIZES: Final[list[str]] = [
    "4x5",
    "5x7",
    "8x10",
    "11x14",
    "16x20",
    "letter",
    "a4",
    "a3",
    "roll_13in",
    "roll_17in",
    "roll_24in",
]

# Common photo printer resolutions (DPI)
STANDARD_RESOLUTIONS: Final[list[int]] = [360, 720, 1440, 2880, 5760]

# Default resolution for printing
DEFAULT_RESOLUTION_DPI: Final[int] = 2880

# Ink level thresholds (percentage)
INK_LEVEL_LOW_THRESHOLD: Final[int] = 25
INK_LEVEL_MEDIUM_THRESHOLD: Final[int] = 50

# Print quality options
PRINT_QUALITY_DRAFT: Final[str] = "draft"
PRINT_QUALITY_NORMAL: Final[str] = "normal"
PRINT_QUALITY_HIGH: Final[str] = "high"
PRINT_QUALITY_PHOTO: Final[str] = "photo"
DEFAULT_PRINT_QUALITY: Final[str] = PRINT_QUALITY_HIGH

# Color model options
COLOR_MODEL_GRAY: Final[str] = "Gray"
COLOR_MODEL_RGB: Final[str] = "RGB"
COLOR_MODEL_CMYK: Final[str] = "CMYK"
DEFAULT_COLOR_MODEL: Final[str] = COLOR_MODEL_GRAY

# Paper size validation pattern (alphanumeric, dots, 'x' only)
PAPER_SIZE_VALIDATION_PATTERN: Final[str] = r"^[a-zA-Z0-9.x]+$"

# =============================================================================
# SPECTROPHOTOMETER CONSTANTS
# =============================================================================

# Spectral measurement range defaults
DEFAULT_SPECTRAL_START_NM: Final[float] = 380.0
DEFAULT_SPECTRAL_END_NM: Final[float] = 730.0
DEFAULT_SPECTRAL_INTERVAL_NM: Final[float] = 10.0

# X-Rite USB vendor IDs
XRITE_VENDOR_ID: Final[int] = 0x0765
XRITE_PRODUCT_IDS: Final[dict[str, int]] = {
    "i1Pro": 0x5001,
    "i1Pro2": 0x5020,
    "i1Pro3": 0x5030,
}

# Serial communication defaults
DEFAULT_BAUD_RATE: Final[int] = 9600
DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
DEFAULT_LINE_TERMINATOR: Final[str] = "\r\n"
DEFAULT_LINE_TERMINATOR_BYTES: Final[bytes] = b"\r\n"

# Device protocol patterns
FIRMWARE_VERSION_PATTERN: Final[str] = r"V?(\d+\.\d+(?:\.\d+)?)"
SERIAL_NUMBER_PATTERN: Final[str] = r"SN:?([A-Z0-9]+)"

# =============================================================================
# SIMULATED DEVICE CONSTANTS
# =============================================================================

# Simulation delay times (seconds)
SIMULATED_CONNECT_DELAY_SEC: Final[float] = 0.5
SIMULATED_CALIBRATE_WHITE_DELAY_SEC: Final[float] = 1.0
SIMULATED_CALIBRATE_BLACK_DELAY_SEC: Final[float] = 0.5
SIMULATED_MEASURE_DELAY_SEC: Final[float] = 0.3
SIMULATED_SPECTRAL_DELAY_SEC: Final[float] = 0.5
SIMULATED_PRINT_CONNECT_DELAY_SEC: Final[float] = 0.3

# Simulated print timing
SIMULATED_PRINT_BASE_TIME_SEC: Final[float] = 2.0
SIMULATED_PRINT_MAX_TIME_SEC: Final[float] = 5.0
SIMULATED_PRINT_RESOLUTION_REF_DPI: Final[int] = 1440

# Simulated measurement parameters
SIMULATED_STEP_TABLET_STEPS: Final[int] = 21
SIMULATED_DENSITY_GAMMA: Final[float] = 0.85
SIMULATED_MAX_DENSITY_CLAMP: Final[float] = 4.0

# Simulated Lab value calculation
SIMULATED_LAB_L_BASE: Final[float] = 95.0
SIMULATED_LAB_L_SLOPE: Final[float] = 35.0
SIMULATED_LAB_A_STD_DEV: Final[float] = 1.0
SIMULATED_LAB_B_MEAN: Final[float] = -1.0
SIMULATED_LAB_B_STD_DEV: Final[float] = 1.0

# Simulated spectral reflectance
SIMULATED_REFLECTANCE_BASE: Final[float] = 0.3
SIMULATED_REFLECTANCE_SLOPE: Final[float] = 0.1
SIMULATED_REFLECTANCE_WAVELENGTH_REF: Final[float] = 400.0
SIMULATED_REFLECTANCE_RANGE: Final[float] = 300.0

# Simulated device identifiers
SIMULATED_SPECTRO_VENDOR: Final[str] = "Simulated"
SIMULATED_SPECTRO_MODEL: Final[str] = "Virtual Spectro Pro"
SIMULATED_SPECTRO_SERIAL: Final[str] = "SIM-001"
SIMULATED_SPECTRO_FIRMWARE: Final[str] = "1.0.0"

SIMULATED_PRINTER_VENDOR: Final[str] = "Simulated"
SIMULATED_PRINTER_MODEL: Final[str] = "Virtual Inkjet Pro"
SIMULATED_PRINTER_SERIAL: Final[str] = "SIM-PRT-001"
SIMULATED_PRINTER_FIRMWARE: Final[str] = "2.0.0"

# Simulated ink level ranges (min, max) per color
SIMULATED_INK_LEVEL_RANGES: Final[dict[str, tuple[int, int]]] = {
    "matte_black": (60, 100),
    "photo_black": (50, 95),
    "light_black": (40, 90),
    "light_light_black": (30, 85),
    "cyan": (70, 100),
    "magenta": (65, 100),
    "yellow": (75, 100),
}

# =============================================================================
# DEVICE CAPABILITIES
# =============================================================================

PRINTER_CAPABILITIES: Final[dict[str, str]] = {
    "color": "Full color printing",
    "grayscale": "Grayscale/monochrome printing",
    "high_resolution": "High DPI support (2400+)",
    "duplex": "Two-sided printing",
    "roll_paper": "Roll paper support",
    "sheet_paper": "Sheet paper support",
}

SPECTRO_CAPABILITIES: Final[dict[str, str]] = {
    "density": "Density measurement",
    "lab": "CIE Lab color measurement",
    "spectral": "Full spectral data",
    "reflection": "Reflectance measurement",
    "transmission": "Transmittance measurement",
}
