"""
Device discovery system for hardware integration.

Provides pluggable discovery handlers for finding available hardware devices.
Supports USB devices, CUPS printers, Windows printers, and network printers.

Usage:
    from ptpd_calibration.integrations.hardware.discovery import (
        USBDeviceDiscovery,
        CUPSPrinterDiscovery,
        Win32PrinterDiscovery,
        discover_all_devices,
    )

    # Discover USB spectrophotometers
    spectros = USBDeviceDiscovery.discover()

    # Discover all available devices
    all_devices = discover_all_devices()
"""

from __future__ import annotations

import platform
from typing import Protocol, runtime_checkable

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.constants import (
    SIMULATED_PRINTER_FIRMWARE,
    SIMULATED_PRINTER_MODEL,
    SIMULATED_PRINTER_SERIAL,
    SIMULATED_PRINTER_VENDOR,
    SIMULATED_SPECTRO_FIRMWARE,
    SIMULATED_SPECTRO_MODEL,
    SIMULATED_SPECTRO_SERIAL,
    SIMULATED_SPECTRO_VENDOR,
    XRITE_PRODUCT_IDS,
    XRITE_VENDOR_ID,
)
from ptpd_calibration.integrations.hardware.registry import (
    DeviceType,
    DiscoveredDevice,
)
from ptpd_calibration.integrations.protocols import DeviceInfo

logger = get_logger(__name__)


# Known USB device vendor/product IDs mapped to driver hints
KNOWN_USB_SPECTROPHOTOMETERS: dict[tuple[int, int], tuple[str, str, str]] = {
    (XRITE_VENDOR_ID, XRITE_PRODUCT_IDS.get("i1Pro", 0x5001)): (
        "X-Rite",
        "i1Pro",
        "XRiteI1ProDriver",
    ),
    (XRITE_VENDOR_ID, XRITE_PRODUCT_IDS.get("i1Pro2", 0x5020)): (
        "X-Rite",
        "i1Pro2",
        "XRiteI1ProDriver",
    ),
    (XRITE_VENDOR_ID, XRITE_PRODUCT_IDS.get("i1Pro3", 0x5030)): (
        "X-Rite",
        "i1Pro3",
        "XRiteI1ProDriver",
    ),
    (0x0971, 0x2000): ("X-Rite", "ColorMunki Photo", "ColorMunkiDriver"),
    (0x0971, 0x2001): ("X-Rite", "ColorMunki Design", "ColorMunkiDriver"),
    (0x0971, 0x2007): ("X-Rite", "ColorMunki Display", "ColorMunkiDriver"),
}


@runtime_checkable
class DeviceDiscoveryProtocol(Protocol):
    """Protocol for device discovery handlers."""

    @staticmethod
    def is_available() -> bool:
        """Check if this discovery method is available on current platform.

        Returns:
            True if discovery can be performed.
        """
        ...

    @staticmethod
    def discover() -> list[DiscoveredDevice]:
        """Discover devices using this method.

        Returns:
            List of discovered devices.
        """
        ...


class USBDeviceDiscovery:
    """Discover USB devices (spectrophotometers).

    Uses pyserial's serial port enumeration to find connected USB devices.
    Matches known vendor/product IDs to identify spectrophotometers.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if USB discovery is available."""
        try:
            import serial.tools.list_ports  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def discover() -> list[DiscoveredDevice]:
        """Scan USB ports for known spectrophotometers.

        Returns:
            List of discovered spectrophotometer devices.
        """
        discovered: list[DiscoveredDevice] = []

        try:
            import serial.tools.list_ports

            ports = serial.tools.list_ports.comports()

            for port in ports:
                # Check if this matches a known device
                vid = port.vid or 0
                pid = port.pid or 0

                if (vid, pid) in KNOWN_USB_SPECTROPHOTOMETERS:
                    vendor, model, driver_hint = KNOWN_USB_SPECTROPHOTOMETERS[(vid, pid)]
                    serial_number = port.serial_number or f"{vid:04x}:{pid:04x}"
                    device_id = f"usb-{vendor.lower()}-{serial_number}"

                    device = DiscoveredDevice(
                        device_type=DeviceType.SPECTROPHOTOMETER,
                        device_id=device_id,
                        device_info=DeviceInfo(
                            vendor=vendor,
                            model=model,
                            serial_number=serial_number,
                            firmware_version=None,  # Will be populated on connect
                            capabilities=["density", "lab", "spectral", "reflection"],
                        ),
                        connection_params={
                            "port": port.device,
                            "vid": vid,
                            "pid": pid,
                        },
                        driver_hint=driver_hint,
                        is_simulated=False,
                    )
                    discovered.append(device)
                    logger.info(f"Discovered USB device: {vendor} {model} on {port.device}")

        except ImportError:
            logger.warning("pyserial not installed, USB discovery unavailable")
        except Exception as e:
            logger.error(f"USB discovery error: {e}")

        return discovered


class CUPSPrinterDiscovery:
    """Discover printers via CUPS (Linux/macOS).

    Uses the pycups library to enumerate available CUPS printers.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if CUPS discovery is available."""
        if platform.system() not in ("Linux", "Darwin"):
            return False

        try:
            import cups  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def discover() -> list[DiscoveredDevice]:
        """Discover printers via CUPS.

        Returns:
            List of discovered printer devices.
        """
        discovered: list[DiscoveredDevice] = []

        try:
            import cups

            conn = cups.Connection()
            printers = conn.getPrinters()

            for name, attrs in printers.items():
                # Extract printer info from CUPS attributes
                info = attrs.get("printer-info", name)
                location = attrs.get("printer-location", "")
                uri = attrs.get("device-uri", "")
                state = attrs.get("printer-state", 0)

                # Determine vendor from URI or info
                vendor = "Unknown"
                model = info
                if "epson" in uri.lower() or "epson" in info.lower():
                    vendor = "Epson"
                elif "canon" in uri.lower() or "canon" in info.lower():
                    vendor = "Canon"
                elif "hp" in uri.lower() or "hp" in info.lower():
                    vendor = "HP"

                device_id = f"cups-{name.replace(' ', '_').lower()}"

                capabilities = ["grayscale", "sheet_paper"]
                if state == 3:  # Idle
                    capabilities.append("ready")

                device = DiscoveredDevice(
                    device_type=DeviceType.PRINTER,
                    device_id=device_id,
                    device_info=DeviceInfo(
                        vendor=vendor,
                        model=model,
                        serial_number=None,
                        firmware_version=None,
                        capabilities=capabilities,
                    ),
                    connection_params={
                        "printer_name": name,
                        "uri": uri,
                        "location": location,
                    },
                    driver_hint="CUPSPrinterDriver",
                    is_simulated=False,
                )
                discovered.append(device)
                logger.info(f"Discovered CUPS printer: {name}")

        except ImportError:
            logger.warning("pycups not installed, CUPS discovery unavailable")
        except Exception as e:
            logger.error(f"CUPS discovery error: {e}")

        return discovered


class Win32PrinterDiscovery:
    """Discover printers via Windows Print Spooler.

    Uses the pywin32 library to enumerate Windows printers.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Windows printer discovery is available."""
        if platform.system() != "Windows":
            return False

        try:
            import win32print  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def discover() -> list[DiscoveredDevice]:
        """Discover printers via Win32 API.

        Returns:
            List of discovered printer devices.
        """
        discovered: list[DiscoveredDevice] = []

        try:
            import win32print

            # Enumerate local and network printers
            flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
            printers = win32print.EnumPrinters(flags)

            for printer in printers:
                # printer tuple: (flags, description, name, comment)
                name = printer[2]
                description = printer[1] if len(printer) > 1 else name
                comment = printer[3] if len(printer) > 3 else ""

                # Determine vendor from name/description
                vendor = "Unknown"
                model = name
                name_lower = name.lower()
                if "epson" in name_lower:
                    vendor = "Epson"
                elif "canon" in name_lower:
                    vendor = "Canon"
                elif "hp" in name_lower:
                    vendor = "HP"
                elif "brother" in name_lower:
                    vendor = "Brother"

                device_id = f"win32-{name.replace(' ', '_').lower()}"

                device = DiscoveredDevice(
                    device_type=DeviceType.PRINTER,
                    device_id=device_id,
                    device_info=DeviceInfo(
                        vendor=vendor,
                        model=model,
                        serial_number=None,
                        firmware_version=None,
                        capabilities=["grayscale", "sheet_paper"],
                    ),
                    connection_params={
                        "printer_name": name,
                        "description": description,
                        "comment": comment,
                    },
                    driver_hint="Win32PrinterDriver",
                    is_simulated=False,
                )
                discovered.append(device)
                logger.info(f"Discovered Windows printer: {name}")

        except ImportError:
            logger.warning("pywin32 not installed, Windows discovery unavailable")
        except Exception as e:
            logger.error(f"Win32 discovery error: {e}")

        return discovered


class IPPPrinterDiscovery:
    """Discover network printers via IPP/Bonjour/mDNS.

    Uses zeroconf to discover IPP printers on the local network.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if IPP/mDNS discovery is available."""
        try:
            import zeroconf  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def discover(timeout_seconds: float = 5.0) -> list[DiscoveredDevice]:
        """Discover IPP printers on network.

        Args:
            timeout_seconds: Discovery timeout.

        Returns:
            List of discovered printer devices.
        """
        discovered: list[DiscoveredDevice] = []

        try:
            import time

            from zeroconf import ServiceBrowser, Zeroconf

            class IPPListener:
                def __init__(self) -> None:
                    self.printers: list[DiscoveredDevice] = []

                def add_service(
                    self,
                    zeroconf: Zeroconf,
                    service_type: str,
                    name: str,
                ) -> None:
                    info = zeroconf.get_service_info(service_type, name)
                    if info:
                        printer_name = name.replace("._ipp._tcp.local.", "")
                        addresses = [".".join(map(str, addr)) for addr in info.addresses]
                        port = info.port

                        device_id = f"ipp-{printer_name.replace(' ', '_').lower()}"

                        device = DiscoveredDevice(
                            device_type=DeviceType.PRINTER,
                            device_id=device_id,
                            device_info=DeviceInfo(
                                vendor="Network",
                                model=printer_name,
                                serial_number=None,
                                firmware_version=None,
                                capabilities=["network", "ipp"],
                            ),
                            connection_params={
                                "printer_name": printer_name,
                                "addresses": addresses,
                                "port": port,
                                "uri": f"ipp://{addresses[0]}:{port}/ipp/print"
                                if addresses
                                else None,
                            },
                            driver_hint="CUPSPrinterDriver",  # IPP can use CUPS
                            is_simulated=False,
                        )
                        self.printers.append(device)
                        logger.info(f"Discovered IPP printer: {printer_name}")

                def remove_service(
                    self,
                    zeroconf: Zeroconf,
                    service_type: str,
                    name: str,
                ) -> None:
                    pass

                def update_service(
                    self,
                    zeroconf: Zeroconf,
                    service_type: str,
                    name: str,
                ) -> None:
                    pass

            zc = Zeroconf()
            listener = IPPListener()

            # Browse for IPP services
            browser = ServiceBrowser(zc, "_ipp._tcp.local.", listener)  # noqa: F841

            # Wait for discovery
            time.sleep(timeout_seconds)

            zc.close()
            discovered = listener.printers

        except ImportError:
            logger.warning("zeroconf not installed, IPP discovery unavailable")
        except Exception as e:
            logger.error(f"IPP discovery error: {e}")

        return discovered


class SimulatedDeviceDiscovery:
    """Discover simulated devices for testing.

    Always available and returns mock devices for development/testing.
    """

    @staticmethod
    def is_available() -> bool:
        """Simulated discovery is always available."""
        return True

    @staticmethod
    def discover() -> list[DiscoveredDevice]:
        """Return simulated devices.

        Returns:
            List with one simulated spectrophotometer and printer.
        """
        return [
            DiscoveredDevice(
                device_type=DeviceType.SPECTROPHOTOMETER,
                device_id="simulated-spectro-001",
                device_info=DeviceInfo(
                    vendor=SIMULATED_SPECTRO_VENDOR,
                    model=SIMULATED_SPECTRO_MODEL,
                    serial_number=SIMULATED_SPECTRO_SERIAL,
                    firmware_version=SIMULATED_SPECTRO_FIRMWARE,
                    capabilities=["density", "lab", "spectral", "reflection", "transmission"],
                ),
                connection_params={},
                driver_hint="SimulatedSpectrophotometer",
                is_simulated=True,
            ),
            DiscoveredDevice(
                device_type=DeviceType.PRINTER,
                device_id="simulated-printer-001",
                device_info=DeviceInfo(
                    vendor=SIMULATED_PRINTER_VENDOR,
                    model=SIMULATED_PRINTER_MODEL,
                    serial_number=SIMULATED_PRINTER_SERIAL,
                    firmware_version=SIMULATED_PRINTER_FIRMWARE,
                    capabilities=["color", "grayscale", "high_resolution", "roll_paper"],
                ),
                connection_params={},
                driver_hint="SimulatedPrinter",
                is_simulated=True,
            ),
        ]


def discover_all_devices(
    include_simulated: bool = True,
    device_types: list[DeviceType] | None = None,
) -> list[DiscoveredDevice]:
    """Discover all available devices using all available methods.

    Args:
        include_simulated: Include simulated devices.
        device_types: Filter to specific device types (all if None).

    Returns:
        List of all discovered devices.
    """
    all_discovered: list[DiscoveredDevice] = []

    # Spectrophotometer discovery
    if (
        device_types is None or DeviceType.SPECTROPHOTOMETER in device_types
    ) and USBDeviceDiscovery.is_available():
        all_discovered.extend(USBDeviceDiscovery.discover())

    # Printer discovery
    if device_types is None or DeviceType.PRINTER in device_types:
        if CUPSPrinterDiscovery.is_available():
            all_discovered.extend(CUPSPrinterDiscovery.discover())
        if Win32PrinterDiscovery.is_available():
            all_discovered.extend(Win32PrinterDiscovery.discover())
        # IPP discovery is optional and slow, skip by default
        # if IPPPrinterDiscovery.is_available():
        #     all_discovered.extend(IPPPrinterDiscovery.discover())

    # Simulated devices
    if include_simulated:
        simulated = SimulatedDeviceDiscovery.discover()
        if device_types:
            simulated = [d for d in simulated if d.device_type in device_types]
        all_discovered.extend(simulated)

    logger.info(f"Total devices discovered: {len(all_discovered)}")
    return all_discovered


def get_platform_discovery_handlers() -> list[type]:
    """Get discovery handlers available on current platform.

    Returns:
        List of discovery handler classes.
    """
    handlers: list[type] = []

    if USBDeviceDiscovery.is_available():
        handlers.append(USBDeviceDiscovery)
    if CUPSPrinterDiscovery.is_available():
        handlers.append(CUPSPrinterDiscovery)
    if Win32PrinterDiscovery.is_available():
        handlers.append(Win32PrinterDiscovery)
    if IPPPrinterDiscovery.is_available():
        handlers.append(IPPPrinterDiscovery)

    # Simulated is always available
    handlers.append(SimulatedDeviceDiscovery)

    return handlers
