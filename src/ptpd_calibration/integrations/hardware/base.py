"""Base classes for hardware device drivers.

This module provides reusable base classes and utilities for hardware
integration, following the DRY principle and ensuring consistent behavior
across all hardware drivers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.protocols import DeviceInfo, DeviceStatus

if TYPE_CHECKING:
    from ptpd_calibration.integrations.hardware.debug import (
        HardwareDebugger,
        ProtocolLogger,
    )

logger = get_logger(__name__)


class HardwareDeviceBase(ABC):
    """Base class for all hardware device drivers.

    Provides common state management, logging, lifecycle patterns,
    and integrated debugging capabilities.

    Attributes:
        status: Current device connection status.
        device_info: Device information (None if not connected).
        protocol_logger: Protocol-level logger for debugging (lazy initialized).
    """

    def __init__(self, device_type: str = "device") -> None:
        """Initialize hardware device base.

        Args:
            device_type: Type of device for logging (e.g., 'printer', 'spectrophotometer').
        """
        self._status = DeviceStatus.DISCONNECTED
        self._device_info: DeviceInfo | None = None
        self._device_type = device_type
        self._protocol_logger: ProtocolLogger | None = None
        self._debugger: HardwareDebugger | None = None

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information (None if not connected)."""
        return self._device_info

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._status == DeviceStatus.CONNECTED

    @property
    def protocol_logger(self) -> ProtocolLogger:
        """Get protocol logger for this device (lazy initialized).

        Returns:
            Protocol logger instance for debugging communication.
        """
        if self._protocol_logger is None:
            from ptpd_calibration.integrations.hardware.debug import HardwareDebugger

            self._debugger = HardwareDebugger()
            self._protocol_logger = self._debugger.get_protocol_logger(self._device_type)
        return self._protocol_logger

    @contextmanager
    def _track_operation(
        self,
        operation: str,
        **metadata: Any,
    ) -> Generator[None, None, None]:
        """Track a hardware operation with timing for debugging.

        Args:
            operation: Name of the operation.
            **metadata: Additional metadata to record.

        Yields:
            None - context manager for tracking.

        Example:
            with self._track_operation("calibrate", mode="white"):
                # ... perform calibration ...
        """
        if self._debugger is None:
            from ptpd_calibration.integrations.hardware.debug import HardwareDebugger

            self._debugger = HardwareDebugger()

        with self._debugger.track_operation(
            f"{self._device_type}.{operation}",
            **metadata,
        ):
            yield

    def _log_command(self, command: str) -> float:
        """Log an outgoing command and return start time for latency tracking.

        Args:
            command: Command being sent.

        Returns:
            Start time (time.perf_counter) for latency calculation.
        """
        self.protocol_logger.log_send(command)
        return time.perf_counter()

    def _log_response(
        self,
        command: str,
        response: str,
        start_time: float | None = None,
    ) -> None:
        """Log a received response.

        Args:
            command: Original command.
            response: Response received.
            start_time: Start time from _log_command for latency calculation.
        """
        latency_ms = None
        if start_time is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000

        self.protocol_logger.log_receive(command, response, latency_ms)

    def _log_error(
        self,
        command: str,
        error: str,
        start_time: float | None = None,
    ) -> None:
        """Log a command error.

        Args:
            command: Command that failed.
            error: Error description.
            start_time: Start time from _log_command for latency calculation.
        """
        latency_ms = None
        if start_time is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000

        self.protocol_logger.log_error(command, error, latency_ms)

    def _set_status(self, new_status: DeviceStatus, message: str = "") -> None:
        """Safely set device status with logging.

        Args:
            new_status: New status value.
            message: Optional log message.
        """
        old_status = self._status
        self._status = new_status

        if message:
            log_message = f"[{self._device_type}] {message} ({old_status.value} → {new_status.value})"
            if new_status == DeviceStatus.ERROR:
                logger.error(log_message)
            elif new_status == DeviceStatus.DISCONNECTED and old_status != DeviceStatus.DISCONNECTED:
                logger.warning(log_message)
            else:
                logger.info(log_message)
        else:
            logger.debug(
                f"[{self._device_type}] Status change: {old_status.value} → {new_status.value}"
            )

    def _set_device_info(self, info: DeviceInfo) -> None:
        """Set device information with logging.

        Args:
            info: Device information.
        """
        self._device_info = info
        logger.debug(f"[{self._device_type}] Device info: {info}")

    def _clear_device_info(self) -> None:
        """Clear cached device info."""
        self._device_info = None

    @abstractmethod
    def connect(self, **kwargs: Any) -> bool:
        """Connect to device.

        Returns:
            True if connection successful.

        Raises:
            DeviceConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from device."""
        ...


def parse_device_response(
    response: str,
    field_map: dict[str, type] | None = None,
    remove_prefix: str = "OK:",
    delimiter: str = ",",
    key_value_separator: str = "=",
) -> dict[str, Any]:
    """Parse key=value response from device.

    Generic parser for device responses in key=value format.

    Args:
        response: Raw response string from device.
        field_map: Optional dict mapping key names to expected types for conversion.
        remove_prefix: Prefix to strip from response (default "OK:").
        delimiter: Separator between key-value pairs (default ",").
        key_value_separator: Separator between key and value (default "=").

    Returns:
        Dictionary of parsed key-value pairs.

    Example:
        >>> parse_device_response("OK:D=1.234,L=50.12")
        {'D': 1.234, 'L': 50.12}
    """
    response = response.replace(remove_prefix, "").strip()
    result: dict[str, Any] = {}

    for part in response.split(delimiter):
        if key_value_separator in part:
            key, value = part.split(key_value_separator, 1)
            key = key.strip().upper()
            value_str = value.strip()

            try:
                # Convert to appropriate type
                if field_map and key in field_map:
                    result[key] = field_map[key](value_str)
                else:
                    # Try float first, then string
                    try:
                        result[key] = float(value_str)
                    except ValueError:
                        result[key] = value_str
            except ValueError as e:
                logger.debug(f"Failed to parse {key}={value_str}: {e}")

    return result
