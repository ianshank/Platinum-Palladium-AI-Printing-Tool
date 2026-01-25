"""
Hardware integration subpackage for PTPD Calibration.

Provides real hardware drivers for:
- X-Rite i1 Pro spectrophotometers (USB serial communication)
- CUPS/IPP printers (Linux/macOS)
- Win32 printers (Windows)

All implementations follow the protocols defined in protocols.py and support
simulation mode for testing.

Usage:
    from ptpd_calibration.integrations.hardware import (
        XRiteI1ProDriver,
        CUPSPrinterDriver,
        get_spectrophotometer_driver,
        get_printer_driver,
    )

    # Auto-detect and connect to spectrophotometer
    driver = get_spectrophotometer_driver()
    if driver.connect():
        measurement = driver.read_density()
"""

from typing import TYPE_CHECKING

# Lazy imports for optional dependencies
__lazy_imports = {
    "XRiteI1ProDriver": "ptpd_calibration.integrations.hardware.xrite_i1pro",
    "CUPSPrinterDriver": "ptpd_calibration.integrations.hardware.cups_printer",
    "HardwareError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceNotFoundError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceConnectionError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceCommunicationError": "ptpd_calibration.integrations.hardware.exceptions",
    "CalibrationError": "ptpd_calibration.integrations.hardware.exceptions",
    "MeasurementError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrinterError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrintJobError": "ptpd_calibration.integrations.hardware.exceptions",
}


def __getattr__(name: str):
    """Lazy import implementation classes."""
    if name in __lazy_imports:
        import importlib
        module_path = __lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_spectrophotometer_driver(
    simulate: bool = True,
    port: str | None = None,
):
    """Get appropriate spectrophotometer driver based on settings.

    Args:
        simulate: If True, return simulated driver. If False, try real hardware.
        port: Serial port for real hardware (auto-detect if None).

    Returns:
        Spectrophotometer driver instance.

    Raises:
        ImportError: If hardware dependencies not installed and simulate=False.
    """
    if simulate:
        from ptpd_calibration.integrations.hardware.simulated import (
            SimulatedSpectrophotometer,
        )
        return SimulatedSpectrophotometer()

    try:
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver
        return XRiteI1ProDriver(port=port)
    except ImportError as e:
        raise ImportError(
            "Hardware drivers not installed. Install with: "
            "pip install ptpd-calibration[hardware]"
        ) from e


def get_printer_driver(
    simulate: bool = True,
    printer_name: str | None = None,
):
    """Get appropriate printer driver based on platform and settings.

    Args:
        simulate: If True, return simulated driver. If False, try real hardware.
        printer_name: Printer name (use system default if None).

    Returns:
        Printer driver instance.

    Raises:
        ImportError: If hardware dependencies not installed and simulate=False.
    """
    if simulate:
        from ptpd_calibration.integrations.hardware.simulated import SimulatedPrinter
        return SimulatedPrinter()

    import platform
    system = platform.system()

    if system in ("Linux", "Darwin"):  # Linux or macOS
        try:
            from ptpd_calibration.integrations.hardware.cups_printer import (
                CUPSPrinterDriver,
            )
            return CUPSPrinterDriver(printer_name=printer_name)
        except ImportError as e:
            raise ImportError(
                "CUPS drivers not installed. Install with: "
                "pip install pycups"
            ) from e
    elif system == "Windows":
        try:
            from ptpd_calibration.integrations.hardware.win32_printer import (
                Win32PrinterDriver,
            )
            return Win32PrinterDriver(printer_name=printer_name)
        except ImportError as e:
            raise ImportError(
                "Windows printing libraries not available."
            ) from e
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


__all__ = [
    "XRiteI1ProDriver",
    "CUPSPrinterDriver",
    "HardwareError",
    "DeviceNotFoundError",
    "DeviceConnectionError",
    "DeviceCommunicationError",
    "CalibrationError",
    "MeasurementError",
    "PrinterError",
    "PrintJobError",
    "get_spectrophotometer_driver",
    "get_printer_driver",
]
