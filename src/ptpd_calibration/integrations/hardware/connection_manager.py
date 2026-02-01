"""
Connection manager for hardware device lifecycle management.

Provides automatic reconnection, health monitoring, and resource cleanup
for hardware devices. Implements context manager pattern for RAII-style
resource management.

Usage:
    from ptpd_calibration.integrations.hardware.connection_manager import (
        ConnectionManager,
        ConnectionSettings,
    )

    # Create manager for a device
    manager = ConnectionManager(spectrophotometer, settings)

    # Use context manager for automatic cleanup
    with manager.session() as device:
        measurement = device.read_density()

    # Or manage manually
    manager.connect()
    manager.ensure_connected()
    manager.disconnect()
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.exceptions import (
    DeviceConnectionError,
    DeviceReconnectionError,
)
from ptpd_calibration.integrations.protocols import DeviceStatus

if TYPE_CHECKING:
    from ptpd_calibration.integrations.hardware.base import HardwareDeviceBase

logger = get_logger(__name__)

T = TypeVar("T", bound="HardwareDeviceBase")


@dataclass
class ConnectionSettings:
    """Settings for connection management.

    Attributes:
        auto_reconnect: Automatically reconnect on connection loss.
        max_reconnect_attempts: Maximum number of reconnection attempts.
        reconnect_delay_seconds: Delay between reconnection attempts.
        reconnect_backoff_multiplier: Multiplier for exponential backoff.
        connection_timeout_seconds: Timeout for connection operations.
        health_check_interval_seconds: Interval between health checks.
        enable_health_monitoring: Enable background health monitoring.
    """

    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0
    reconnect_backoff_multiplier: float = 2.0
    connection_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 60.0
    enable_health_monitoring: bool = True


# Type for event callbacks
EventCallback = Callable[[], None]
ErrorCallback = Callable[[Exception], None]


@dataclass
class ConnectionState:
    """Internal connection state tracking.

    Attributes:
        is_connected: Current connection status.
        last_connected_at: Timestamp of last successful connection.
        last_disconnected_at: Timestamp of last disconnection.
        reconnect_attempts: Number of reconnection attempts since last success.
        total_reconnections: Total reconnections during session.
        health_check_failures: Consecutive health check failures.
    """

    is_connected: bool = False
    last_connected_at: float | None = None
    last_disconnected_at: float | None = None
    reconnect_attempts: int = 0
    total_reconnections: int = 0
    health_check_failures: int = 0


class ConnectionManager(Generic[T]):
    """Manages device connection lifecycle.

    Features:
    - Automatic reconnection on connection loss
    - Health monitoring with configurable intervals
    - Thread-safe connection state management
    - Context manager support for RAII pattern
    - Event callbacks for connection state changes

    Example:
        # Basic usage
        manager = ConnectionManager(my_device)
        manager.connect(port="/dev/ttyUSB0")

        # With context manager
        with manager.session() as device:
            result = device.read_density()

        # With event callbacks
        manager.on("connected", lambda: print("Connected!"))
        manager.on("disconnected", lambda: print("Disconnected!"))
        manager.on("error", lambda e: print(f"Error: {e}"))
    """

    def __init__(
        self,
        device: T,
        settings: ConnectionSettings | None = None,
    ) -> None:
        """Initialize connection manager.

        Args:
            device: Device instance to manage.
            settings: Connection settings (uses defaults if None).
        """
        self._device = device
        self._settings = settings or ConnectionSettings()
        self._state = ConnectionState()
        self._lock = threading.RLock()

        # Health monitoring
        self._health_thread: threading.Thread | None = None
        self._health_stop_event = threading.Event()

        # Event callbacks
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "connected": [],
            "disconnected": [],
            "reconnecting": [],
            "error": [],
            "health_check_failed": [],
        }

        # Connection parameters (stored for reconnection)
        self._connection_params: dict[str, Any] = {}

    @property
    def device(self) -> T:
        """Get the managed device."""
        return self._device

    @property
    def settings(self) -> ConnectionSettings:
        """Get connection settings."""
        return self._settings

    @property
    def is_connected(self) -> bool:
        """Check if device is currently connected."""
        with self._lock:
            if hasattr(self._device, "status"):
                return self._device.status == DeviceStatus.CONNECTED
            return self._state.is_connected

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        with self._lock:
            return self._state

    def connect(self, **kwargs: Any) -> bool:
        """Connect to device with configured settings.

        Args:
            **kwargs: Connection parameters passed to device.connect().

        Returns:
            True if connection successful.

        Raises:
            DeviceConnectionError: If connection fails.
        """
        with self._lock:
            # Store params for reconnection
            self._connection_params = kwargs

            try:
                logger.info(f"Connecting to device: {self._device.__class__.__name__}")

                if hasattr(self._device, "connect"):
                    success = self._device.connect(**kwargs)
                else:
                    logger.error("Device does not support connect()")
                    return False

                if success:
                    self._state.is_connected = True
                    self._state.last_connected_at = time.time()
                    self._state.reconnect_attempts = 0
                    self._emit("connected")

                    # Start health monitoring if enabled
                    if self._settings.enable_health_monitoring:
                        self._start_health_monitor()

                    logger.info("Device connected successfully")
                    return True
                else:
                    logger.warning("Device connect() returned False")
                    return False

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._emit_error(e)
                raise DeviceConnectionError(
                    message=str(e),
                    device_type=getattr(self._device, "_device_type", "unknown"),
                ) from e

    def disconnect(self) -> None:
        """Disconnect and stop health monitoring."""
        with self._lock:
            # Stop health monitoring first
            self._stop_health_monitor()

            try:
                if hasattr(self._device, "disconnect"):
                    self._device.disconnect()

                self._state.is_connected = False
                self._state.last_disconnected_at = time.time()
                self._emit("disconnected")
                logger.info("Device disconnected")

            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                self._emit_error(e)

    def ensure_connected(self) -> bool:
        """Ensure device is connected, reconnecting if needed.

        Returns:
            True if device is connected after this call.
        """
        with self._lock:
            if self.is_connected:
                return True

            if self._settings.auto_reconnect:
                return self.reconnect()

            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect to the device.

        Uses exponential backoff between attempts.

        Returns:
            True if reconnection successful.

        Raises:
            DeviceReconnectionError: If all attempts fail.
        """
        with self._lock:
            self._emit("reconnecting")

            delay = self._settings.reconnect_delay_seconds

            for attempt in range(1, self._settings.max_reconnect_attempts + 1):
                self._state.reconnect_attempts = attempt

                logger.info(
                    f"Reconnection attempt {attempt}/{self._settings.max_reconnect_attempts}"
                )

                try:
                    # Disconnect first to clean up
                    if hasattr(self._device, "disconnect"):
                        with suppress(Exception):
                            self._device.disconnect()

                    # Attempt reconnect
                    if hasattr(self._device, "connect"):
                        success = self._device.connect(**self._connection_params)

                        if success:
                            self._state.is_connected = True
                            self._state.last_connected_at = time.time()
                            self._state.total_reconnections += 1
                            self._state.reconnect_attempts = 0
                            self._emit("connected")
                            logger.info(f"Reconnection successful on attempt {attempt}")
                            return True

                except Exception as e:
                    logger.warning(f"Reconnection attempt {attempt} failed: {e}")

                # Wait before next attempt with exponential backoff
                if attempt < self._settings.max_reconnect_attempts:
                    time.sleep(delay)
                    delay *= self._settings.reconnect_backoff_multiplier

            # All attempts failed
            error = DeviceReconnectionError(
                message="All reconnection attempts failed",
                attempts=self._settings.max_reconnect_attempts,
            )
            self._emit_error(error)
            raise error

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:
        """Register event callback.

        Args:
            event: Event name (connected, disconnected, reconnecting, error, health_check_failed).
            callback: Function to call when event occurs.
                For "error" and "health_check_failed", callback receives the exception.
        """
        with self._lock:
            if event in self._callbacks:
                self._callbacks[event].append(callback)
            else:
                logger.warning(f"Unknown event type: {event}")

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:
        """Unregister event callback.

        Args:
            event: Event name.
            callback: Callback to remove.
        """
        with self._lock:
            if event in self._callbacks and callback in self._callbacks[event]:
                self._callbacks[event].remove(callback)

    def _emit(self, event: str) -> None:
        """Emit an event to all registered callbacks.

        Thread-safe: Creates a copy of callbacks to prevent issues if
        callbacks modify the manager during iteration.

        Args:
            event: Event name.
        """
        with self._lock:
            callbacks = list(self._callbacks.get(event, []))

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Event callback error for {event}: {e}")

    def _emit_error(self, error: Exception) -> None:
        """Emit an error event.

        Thread-safe: Creates a copy of callbacks to prevent issues if
        callbacks modify the manager during iteration.

        Args:
            error: Exception to pass to callbacks.
        """
        with self._lock:
            callbacks = list(self._callbacks.get("error", []))

        for callback in callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback exception: {e}")

    @contextmanager
    def session(self, **connect_kwargs: Any) -> Iterator[T]:
        """Context manager for device session.

        Automatically connects on enter and disconnects on exit.

        Args:
            **connect_kwargs: Arguments passed to connect().

        Yields:
            The managed device.

        Example:
            with manager.session(port="/dev/ttyUSB0") as device:
                result = device.read_density()
        """
        try:
            self.connect(**connect_kwargs)
            yield self._device
        finally:
            self.disconnect()

    def _start_health_monitor(self) -> None:
        """Start background health monitoring thread."""
        if self._health_thread is not None and self._health_thread.is_alive():
            return

        self._health_stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name=f"HealthMonitor-{id(self)}",
        )
        self._health_thread.start()
        logger.debug("Health monitoring started")

    def _stop_health_monitor(self) -> None:
        """Stop background health monitoring."""
        self._health_stop_event.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=2.0)
            self._health_thread = None
        logger.debug("Health monitoring stopped")

    def _health_check_loop(self) -> None:
        """Background health check loop.

        Thread-safe: Uses locks when accessing shared state.
        """
        while not self._health_stop_event.is_set():
            # Wait for interval or stop signal
            if self._health_stop_event.wait(timeout=self._settings.health_check_interval_seconds):
                break

            # Perform health check
            try:
                if not self._perform_health_check():
                    with self._lock:
                        self._state.health_check_failures += 1
                        failure_count = self._state.health_check_failures

                    logger.warning(f"Health check failed ({failure_count} consecutive)")

                    # Emit event for consecutive failures
                    if failure_count >= 3:
                        error = Exception(f"Health check failed {failure_count} times")
                        # Copy callbacks under lock for thread safety
                        with self._lock:
                            callbacks = list(self._callbacks.get("health_check_failed", []))

                        for callback in callbacks:
                            try:
                                callback(error)
                            except Exception as e:
                                logger.warning(f"Health callback error: {e}")

                        # Attempt reconnection
                        if self._settings.auto_reconnect:
                            try:
                                self.reconnect()
                                with self._lock:
                                    self._state.health_check_failures = 0
                            except DeviceReconnectionError:
                                pass
                else:
                    with self._lock:
                        self._state.health_check_failures = 0

            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _perform_health_check(self) -> bool:
        """Perform a health check on the device.

        Returns:
            True if device is healthy.
        """
        with self._lock:
            # Check device status property
            if hasattr(self._device, "status"):
                status = self._device.status
                return status == DeviceStatus.CONNECTED

            # Check is_connected property
            if hasattr(self._device, "is_connected"):
                return self._device.is_connected

            # Fallback to internal state
            return self._state.is_connected

    def __enter__(self) -> ConnectionManager[T]:
        """Enter context - connect to device."""
        self.connect(**self._connection_params)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context - disconnect from device."""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        device_name = self._device.__class__.__name__
        connected = "connected" if self.is_connected else "disconnected"
        return f"ConnectionManager({device_name}, {connected})"
