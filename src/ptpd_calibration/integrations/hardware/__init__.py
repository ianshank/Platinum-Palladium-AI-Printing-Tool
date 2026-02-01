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
        Win32PrinterDriver,
        get_spectrophotometer_driver,
        get_printer_driver,
        get_device,
        DeviceRegistry,
        DeviceType,
        ConnectionManager,
    )

    # Auto-detect and connect to spectrophotometer
    driver = get_spectrophotometer_driver()
    if driver.connect():
        measurement = driver.read_density()

    # Use connection manager for automatic lifecycle management
    from ptpd_calibration.integrations.hardware import get_device, DeviceType
    with get_device(DeviceType.SPECTROPHOTOMETER).session() as spectro:
        measurement = spectro.read_density()

    # Access constants for configuration
    from ptpd_calibration.integrations.hardware import constants
    print(constants.STANDARD_RESOLUTIONS)
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any, Literal, overload

# Direct imports for always-available modules
from ptpd_calibration.integrations.hardware import constants
from ptpd_calibration.integrations.hardware.base import (
    HardwareDeviceBase,
    parse_device_response,
)
from ptpd_calibration.integrations.hardware.connection_manager import (
    ConnectionManager,
    ConnectionSettings,
    ConnectionState,
)
from ptpd_calibration.integrations.hardware.discovery import (
    CUPSPrinterDiscovery,
    IPPPrinterDiscovery,
    SimulatedDeviceDiscovery,
    USBDeviceDiscovery,
    Win32PrinterDiscovery,
    discover_all_devices,
    get_platform_discovery_handlers,
)
from ptpd_calibration.integrations.hardware.registry import (
    DeviceRegistry,
    DeviceType,
    DiscoveredDevice,
    RegisteredDevice,
)

if TYPE_CHECKING:
    from ptpd_calibration.integrations.protocols import (
        PrinterProtocol,
        SpectrophotometerProtocol,
    )

# Lazy imports for optional dependencies
__lazy_imports = {
    "XRiteI1ProDriver": "ptpd_calibration.integrations.hardware.xrite_i1pro",
    "CUPSPrinterDriver": "ptpd_calibration.integrations.hardware.cups_printer",
    "Win32PrinterDriver": "ptpd_calibration.integrations.hardware.win32_printer",
    "SimulatedSpectrophotometer": "ptpd_calibration.integrations.hardware.simulated",
    "SimulatedPrinter": "ptpd_calibration.integrations.hardware.simulated",
    # Exceptions
    "HardwareError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceNotFoundError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceConnectionError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceCommunicationError": "ptpd_calibration.integrations.hardware.exceptions",
    "CalibrationError": "ptpd_calibration.integrations.hardware.exceptions",
    "MeasurementError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrinterError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrinterNotFoundError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrintJobError": "ptpd_calibration.integrations.hardware.exceptions",
    "PrinterDriverError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceReconnectionError": "ptpd_calibration.integrations.hardware.exceptions",
    "DeviceTimeoutError": "ptpd_calibration.integrations.hardware.exceptions",
    "DiscoveryError": "ptpd_calibration.integrations.hardware.exceptions",
    "PermissionDeniedError": "ptpd_calibration.integrations.hardware.exceptions",
    # Debugging
    "HardwareDebugger": "ptpd_calibration.integrations.hardware.debug",
    "ProtocolLogger": "ptpd_calibration.integrations.hardware.debug",
    "DebugLevel": "ptpd_calibration.integrations.hardware.debug",
    "debug_mode": "ptpd_calibration.integrations.hardware.debug",
    "debug_hardware_call": "ptpd_calibration.integrations.hardware.debug",
    "get_diagnostic_report": "ptpd_calibration.integrations.hardware.debug",
    "save_debug_session": "ptpd_calibration.integrations.hardware.debug",
}


def __getattr__(name: str) -> Any:
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
) -> Any:
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
) -> Any:
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

    system = platform.system()

    if system in ("Linux", "Darwin"):  # Linux or macOS
        try:
            from ptpd_calibration.integrations.hardware.cups_printer import (
                CUPSPrinterDriver,
            )

            return CUPSPrinterDriver(printer_name=printer_name)
        except ImportError as e:
            raise ImportError(
                "CUPS drivers not installed. Install with: pip install pycups"
            ) from e

    elif system == "Windows":
        try:
            from ptpd_calibration.integrations.hardware.win32_printer import (
                Win32PrinterDriver,
            )

            return Win32PrinterDriver(printer_name=printer_name)
        except ImportError as e:
            raise ImportError(
                "Windows drivers not installed. Install with: pip install pywin32"
            ) from e

    else:
        raise RuntimeError(f"Unsupported platform: {system}")


@overload
def get_device(
    device_type: Literal[DeviceType.SPECTROPHOTOMETER],
    device_id: str | None = None,
    *,
    simulate: bool | None = None,
    auto_connect: bool = False,
    managed: bool = True,
) -> ConnectionManager["SpectrophotometerProtocol"]: ...


@overload
def get_device(
    device_type: Literal[DeviceType.PRINTER],
    device_id: str | None = None,
    *,
    simulate: bool | None = None,
    auto_connect: bool = False,
    managed: bool = True,
) -> ConnectionManager["PrinterProtocol"]: ...


@overload
def get_device(
    device_type: DeviceType,
    device_id: str | None = None,
    *,
    simulate: bool | None = None,
    auto_connect: bool = False,
    managed: bool = True,
) -> ConnectionManager[Any]: ...


def get_device(
    device_type: DeviceType,
    device_id: str | None = None,
    *,
    simulate: bool | None = None,
    auto_connect: bool = False,
    managed: bool = True,
    connection_settings: ConnectionSettings | None = None,
) -> ConnectionManager[Any] | Any:
    """Get a hardware device with connection management.

    This is the recommended way to obtain hardware devices. It provides:
    - Automatic device discovery and selection
    - Connection lifecycle management
    - Automatic reconnection on connection loss
    - Thread-safe operation

    Args:
        device_type: Type of device to get (SPECTROPHOTOMETER, PRINTER).
        device_id: Specific device ID (auto-select if None).
        simulate: Force simulation mode (None = auto-detect real hardware).
        auto_connect: Automatically connect to device.
        managed: Wrap in ConnectionManager (recommended).
        connection_settings: Custom connection settings.

    Returns:
        ConnectionManager wrapping the device (if managed=True), or device directly.

    Example:
        # Get spectrophotometer with auto-detection
        with get_device(DeviceType.SPECTROPHOTOMETER).session() as spectro:
            measurement = spectro.read_density()

        # Get specific printer without auto-connect
        manager = get_device(
            DeviceType.PRINTER,
            device_id="cups-epson_p800",
            simulate=False,
        )
        manager.connect()

        # Get simulated device for testing
        with get_device(DeviceType.SPECTROPHOTOMETER, simulate=True).session() as s:
            data = s.read_spectral()
    """
    from ptpd_calibration.config import get_settings

    settings = get_settings()

    # Determine simulation mode
    if simulate is None:
        # Check settings for force_simulation
        # Default to True if no real hardware settings configured
        if device_type == DeviceType.SPECTROPHOTOMETER:
            simulate = settings.integrations.spectro_simulate
        elif device_type == DeviceType.PRINTER:
            simulate = settings.integrations.printer_simulate
        else:
            simulate = True

    # Get or create device
    device: Any = None

    if device_id:
        # Try to get from registry
        registry = DeviceRegistry()
        device = registry.get_device(device_id, auto_connect=False)

    if device is None:
        # Create new device based on type and simulation mode
        if device_type == DeviceType.SPECTROPHOTOMETER:
            device = get_spectrophotometer_driver(
                simulate=simulate,
                port=settings.integrations.spectrophotometer_port,
            )
        elif device_type == DeviceType.PRINTER:
            device = get_printer_driver(
                simulate=simulate,
                printer_name=settings.integrations.default_printer_name,
            )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    if managed:
        manager = ConnectionManager(
            device,
            settings=connection_settings
            or ConnectionSettings(
                auto_reconnect=True,
                max_reconnect_attempts=3,
                health_check_interval_seconds=60.0,
            ),
        )
        if auto_connect:
            manager.connect()
        return manager

    if auto_connect and hasattr(device, "connect"):
        device.connect()

    return device


def initialize_discovery() -> DeviceRegistry:
    """Initialize the device registry with platform-appropriate discovery handlers.

    Registers all available discovery handlers for the current platform.
    Call this once at application startup to enable device discovery.

    Returns:
        Initialized DeviceRegistry instance.

    Example:
        registry = initialize_discovery()
        devices = registry.discover_all()
        for device_id, device in devices.items():
            print(f"Found: {device.discovered_info.device_info}")
    """
    registry = DeviceRegistry()

    # Register spectrophotometer discovery
    if USBDeviceDiscovery.is_available():
        registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            USBDeviceDiscovery.discover,
        )

    # Register printer discovery (platform-specific)
    if CUPSPrinterDiscovery.is_available():
        registry.register_discovery_handler(
            DeviceType.PRINTER,
            CUPSPrinterDiscovery.discover,
        )
    if Win32PrinterDiscovery.is_available():
        registry.register_discovery_handler(
            DeviceType.PRINTER,
            Win32PrinterDiscovery.discover,
        )

    # Register simulated device discovery (always available)
    registry.register_discovery_handler(
        DeviceType.SPECTROPHOTOMETER,
        lambda: [
            d
            for d in SimulatedDeviceDiscovery.discover()
            if d.device_type == DeviceType.SPECTROPHOTOMETER
        ],
    )
    registry.register_discovery_handler(
        DeviceType.PRINTER,
        lambda: [
            d
            for d in SimulatedDeviceDiscovery.discover()
            if d.device_type == DeviceType.PRINTER
        ],
    )

    return registry


__all__ = [
    # Base classes and utilities
    "HardwareDeviceBase",
    "parse_device_response",
    "constants",
    # Registry and connection management
    "DeviceRegistry",
    "DeviceType",
    "DiscoveredDevice",
    "RegisteredDevice",
    "ConnectionManager",
    "ConnectionSettings",
    "ConnectionState",
    # Discovery
    "USBDeviceDiscovery",
    "CUPSPrinterDiscovery",
    "Win32PrinterDiscovery",
    "IPPPrinterDiscovery",
    "SimulatedDeviceDiscovery",
    "discover_all_devices",
    "get_platform_discovery_handlers",
    "initialize_discovery",
    # Drivers
    "XRiteI1ProDriver",
    "CUPSPrinterDriver",
    "Win32PrinterDriver",
    "SimulatedSpectrophotometer",
    "SimulatedPrinter",
    # Exceptions
    "HardwareError",
    "DeviceNotFoundError",
    "DeviceConnectionError",
    "DeviceCommunicationError",
    "CalibrationError",
    "MeasurementError",
    "PrinterError",
    "PrinterNotFoundError",
    "PrintJobError",
    "PrinterDriverError",
    "DeviceReconnectionError",
    "DeviceTimeoutError",
    "DiscoveryError",
    "PermissionDeniedError",
    # Debugging
    "HardwareDebugger",
    "ProtocolLogger",
    "DebugLevel",
    "debug_mode",
    "debug_hardware_call",
    "get_diagnostic_report",
    "save_debug_session",
    # Factory functions
    "get_spectrophotometer_driver",
    "get_printer_driver",
    "get_device",
]
