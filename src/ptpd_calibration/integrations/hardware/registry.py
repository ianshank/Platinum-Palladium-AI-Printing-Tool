"""
Device registry for hardware device management.

Provides a centralized registry for discovering, registering, and managing
hardware devices. Implements the singleton pattern for application-wide
device management.

Usage:
    from ptpd_calibration.integrations.hardware.registry import (
        DeviceRegistry,
        DeviceType,
    )

    # Get registry instance
    registry = DeviceRegistry()

    # Discover all devices
    devices = registry.discover_all()

    # Get a specific device
    device = registry.get_device("xrite-001")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.protocols import DeviceInfo, DeviceStatus

if TYPE_CHECKING:
    from ptpd_calibration.integrations.hardware.base import HardwareDeviceBase

logger = get_logger(__name__)


class DeviceType(str, Enum):
    """Supported device types."""

    SPECTROPHOTOMETER = "spectrophotometer"
    PRINTER = "printer"
    UV_METER = "uv_meter"


@dataclass
class DiscoveredDevice:
    """Information about a discovered device.

    Attributes:
        device_type: Type of device (spectrophotometer, printer, etc.).
        device_id: Unique identifier for this device.
        device_info: Device information (vendor, model, etc.).
        connection_params: Parameters needed to connect to the device.
        driver_hint: Suggested driver class name.
        is_simulated: Whether this is a simulated device.
    """

    device_type: DeviceType
    device_id: str
    device_info: DeviceInfo
    connection_params: dict[str, Any] = field(default_factory=dict)
    driver_hint: str | None = None
    is_simulated: bool = False

    def __hash__(self) -> int:
        return hash(self.device_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscoveredDevice):
            return False
        return self.device_id == other.device_id


@dataclass
class RegisteredDevice:
    """A device registered in the registry.

    Attributes:
        device_type: Type of device.
        discovered_info: Discovery information.
        driver_class: Class to use for creating device instance.
        instance: Cached device instance (if created).
        is_simulated: Whether this is a simulated device.
    """

    device_type: DeviceType
    discovered_info: DiscoveredDevice
    driver_class: type
    instance: HardwareDeviceBase | None = None
    is_simulated: bool = False


# Type alias for discovery handler function
DiscoveryHandler = Callable[[], list[DiscoveredDevice]]


class DeviceRegistry:
    """Singleton registry for hardware devices.

    Provides:
    - Device discovery across all supported types
    - Caching of discovered devices
    - Factory method for device instantiation
    - Event notification for device connect/disconnect

    Thread-safe implementation using locks.

    Example:
        registry = DeviceRegistry()

        # Register a custom discovery handler
        registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            my_custom_discovery_function
        )

        # Discover all devices
        devices = registry.discover_all()

        # Get specific device
        spectro = registry.get_device("xrite-i1pro-12345")
    """

    _instance: DeviceRegistry | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> DeviceRegistry:
        """Create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        """Initialize registry (only runs once due to singleton)."""
        # Prevent re-initialization
        if DeviceRegistry._initialized:
            return

        self._devices: dict[str, RegisteredDevice] = {}
        self._discovery_handlers: dict[DeviceType, list[DiscoveryHandler]] = {
            DeviceType.SPECTROPHOTOMETER: [],
            DeviceType.PRINTER: [],
            DeviceType.UV_METER: [],
        }
        self._change_callbacks: list[Callable[[str, str], None]] = []
        self._device_lock = threading.RLock()

        DeviceRegistry._initialized = True
        logger.debug("DeviceRegistry initialized")

    def register_discovery_handler(
        self,
        device_type: DeviceType,
        handler: DiscoveryHandler,
    ) -> None:
        """Register a discovery handler for a device type.

        Args:
            device_type: Type of device this handler discovers.
            handler: Callable that returns list of DiscoveredDevice.
        """
        with self._device_lock:
            if handler not in self._discovery_handlers[device_type]:
                self._discovery_handlers[device_type].append(handler)
                logger.debug(
                    f"Registered discovery handler for {device_type.value}: "
                    f"{handler.__name__ if hasattr(handler, '__name__') else handler}"
                )

    def unregister_discovery_handler(
        self,
        device_type: DeviceType,
        handler: DiscoveryHandler,
    ) -> None:
        """Unregister a discovery handler.

        Args:
            device_type: Type of device.
            handler: Handler to remove.
        """
        with self._device_lock:
            if handler in self._discovery_handlers[device_type]:
                self._discovery_handlers[device_type].remove(handler)
                logger.debug(f"Unregistered discovery handler for {device_type.value}")

    def discover_all(
        self,
        refresh: bool = False,
        device_types: list[DeviceType] | None = None,
    ) -> dict[str, RegisteredDevice]:
        """Discover all available devices.

        Args:
            refresh: If True, clear cache and rediscover.
            device_types: Specific types to discover (all if None).

        Returns:
            Dictionary of device_id -> RegisteredDevice.
        """
        with self._device_lock:
            if refresh:
                # Clear existing devices (but preserve running instances)
                for device_id in list(self._devices.keys()):
                    registered = self._devices[device_id]
                    if registered.instance is None or (
                        hasattr(registered.instance, "status")
                        and registered.instance.status == DeviceStatus.DISCONNECTED
                    ):
                        del self._devices[device_id]

            types_to_discover = device_types or list(DeviceType)

            for device_type in types_to_discover:
                handlers = self._discovery_handlers.get(device_type, [])
                for handler in handlers:
                    try:
                        discovered = handler()
                        for device in discovered:
                            self._register_discovered_device(device)
                    except Exception as e:
                        logger.warning(
                            f"Discovery handler failed for {device_type.value}: {e}"
                        )

            logger.info(f"Discovered {len(self._devices)} devices")
            return dict(self._devices)

    def _register_discovered_device(self, device: DiscoveredDevice) -> None:
        """Register a discovered device internally.

        Args:
            device: Discovered device information.
        """
        if device.device_id in self._devices:
            # Update existing registration
            existing = self._devices[device.device_id]
            existing.discovered_info = device
            logger.debug(f"Updated device registration: {device.device_id}")
        else:
            # Get appropriate driver class
            driver_class = self._get_driver_class(device)
            if driver_class:
                self._devices[device.device_id] = RegisteredDevice(
                    device_type=device.device_type,
                    discovered_info=device,
                    driver_class=driver_class,
                    is_simulated=device.is_simulated,
                )
                logger.info(
                    f"Registered device: {device.device_id} "
                    f"({device.device_info.vendor} {device.device_info.model})"
                )
                self._notify_change(device.device_id, "registered")

    def _get_driver_class(self, device: DiscoveredDevice) -> type | None:
        """Get appropriate driver class for a discovered device.

        Args:
            device: Discovered device.

        Returns:
            Driver class or None if not found.
        """
        # Lazy import to avoid circular dependencies
        if device.is_simulated:
            from ptpd_calibration.integrations.hardware.simulated import (
                SimulatedPrinter,
                SimulatedSpectrophotometer,
            )

            if device.device_type == DeviceType.SPECTROPHOTOMETER:
                return SimulatedSpectrophotometer
            elif device.device_type == DeviceType.PRINTER:
                return SimulatedPrinter
        else:
            # Real hardware drivers
            if device.device_type == DeviceType.SPECTROPHOTOMETER:
                if device.driver_hint == "XRiteI1ProDriver":
                    try:
                        from ptpd_calibration.integrations.hardware.xrite_i1pro import (
                            XRiteI1ProDriver,
                        )

                        return XRiteI1ProDriver
                    except ImportError:
                        logger.warning("XRiteI1ProDriver not available")
            elif device.device_type == DeviceType.PRINTER:
                if device.driver_hint == "CUPSPrinterDriver":
                    try:
                        from ptpd_calibration.integrations.hardware.cups_printer import (
                            CUPSPrinterDriver,
                        )

                        return CUPSPrinterDriver
                    except ImportError:
                        logger.warning("CUPSPrinterDriver not available")
                elif device.driver_hint == "Win32PrinterDriver":
                    try:
                        from ptpd_calibration.integrations.hardware.win32_printer import (
                            Win32PrinterDriver,
                        )

                        return Win32PrinterDriver
                    except ImportError:
                        logger.warning("Win32PrinterDriver not available")

        return None

    def register_device(
        self,
        device_id: str,
        device_type: DeviceType,
        driver_class: type,
        device_info: DeviceInfo | None = None,
        connection_params: dict[str, Any] | None = None,
        is_simulated: bool = False,
    ) -> None:
        """Manually register a device.

        Args:
            device_id: Unique device identifier.
            device_type: Type of device.
            driver_class: Class to instantiate.
            device_info: Optional device information.
            connection_params: Optional connection parameters.
            is_simulated: Whether this is a simulated device.
        """
        with self._device_lock:
            discovered = DiscoveredDevice(
                device_type=device_type,
                device_id=device_id,
                device_info=device_info
                or DeviceInfo(vendor="Manual", model="Registration"),
                connection_params=connection_params or {},
                is_simulated=is_simulated,
            )
            self._devices[device_id] = RegisteredDevice(
                device_type=device_type,
                discovered_info=discovered,
                driver_class=driver_class,
                is_simulated=is_simulated,
            )
            logger.info(f"Manually registered device: {device_id}")
            self._notify_change(device_id, "registered")

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device.

        Args:
            device_id: Device to unregister.

        Returns:
            True if device was found and removed.
        """
        with self._device_lock:
            if device_id in self._devices:
                registered = self._devices[device_id]
                # Disconnect if connected
                if registered.instance is not None:
                    try:
                        registered.instance.disconnect()
                    except Exception as e:
                        logger.warning(f"Error disconnecting device {device_id}: {e}")
                del self._devices[device_id]
                logger.info(f"Unregistered device: {device_id}")
                self._notify_change(device_id, "unregistered")
                return True
            return False

    def get_device(
        self,
        device_id: str,
        auto_connect: bool = False,
    ) -> HardwareDeviceBase | None:
        """Get a device instance by ID.

        Args:
            device_id: Device identifier.
            auto_connect: Automatically connect if not connected.

        Returns:
            Device instance or None if not found.
        """
        with self._device_lock:
            if device_id not in self._devices:
                logger.warning(f"Device not found: {device_id}")
                return None

            registered = self._devices[device_id]

            # Create instance if needed
            if registered.instance is None:
                try:
                    instance = registered.driver_class()
                    registered.instance = instance
                    logger.debug(f"Created device instance: {device_id}")
                except Exception as e:
                    logger.error(f"Failed to create device instance {device_id}: {e}")
                    return None

            # Auto-connect if requested
            if auto_connect and registered.instance is not None:
                if (
                    hasattr(registered.instance, "status")
                    and registered.instance.status != DeviceStatus.CONNECTED
                ):
                    try:
                        params = registered.discovered_info.connection_params
                        registered.instance.connect(**params)
                    except Exception as e:
                        logger.error(f"Failed to connect to device {device_id}: {e}")

            return registered.instance

    def get_devices_by_type(
        self,
        device_type: DeviceType,
    ) -> list[RegisteredDevice]:
        """Get all devices of a specific type.

        Args:
            device_type: Type to filter by.

        Returns:
            List of registered devices of that type.
        """
        with self._device_lock:
            return [
                device
                for device in self._devices.values()
                if device.device_type == device_type
            ]

    def get_all_devices(self) -> dict[str, RegisteredDevice]:
        """Get all registered devices.

        Returns:
            Dictionary of device_id -> RegisteredDevice.
        """
        with self._device_lock:
            return dict(self._devices)

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for device changes.

        Args:
            callback: Function(device_id, event_type) to call.
                event_type is one of: "registered", "unregistered", "connected", "disconnected"
        """
        with self._device_lock:
            if callback not in self._change_callbacks:
                self._change_callbacks.append(callback)

    def _notify_change(self, device_id: str, event_type: str) -> None:
        """Notify listeners of device change.

        Thread-safe: Creates a copy of callbacks to prevent issues if
        callbacks modify the registry during iteration.

        Args:
            device_id: Device that changed.
            event_type: Type of change.
        """
        with self._device_lock:
            callbacks = list(self._change_callbacks)

        for callback in callbacks:
            try:
                callback(device_id, event_type)
            except Exception as e:
                logger.warning(f"Change callback error: {e}")

    def clear(self) -> None:
        """Clear all registered devices and disconnect."""
        with self._device_lock:
            for device_id, registered in list(self._devices.items()):
                if registered.instance is not None:
                    try:
                        registered.instance.disconnect()
                    except Exception as e:
                        logger.warning(f"Error disconnecting {device_id}: {e}")
            self._devices.clear()
            logger.info("DeviceRegistry cleared")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (mainly for testing).

        Warning: This will lose all registered devices and handlers.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None
            cls._initialized = False
            logger.debug("DeviceRegistry singleton reset")
