"""
Hardware debugging and diagnostic utilities.

Provides tools for:
- Protocol-level debugging (command/response logging)
- Connection state monitoring
- Performance profiling for hardware operations
- Diagnostic reports
- Debug mode toggling

Usage:
    from ptpd_calibration.integrations.hardware.debug import (
        HardwareDebugger,
        ProtocolLogger,
        debug_mode,
        get_diagnostic_report,
    )

    # Enable debug mode for hardware operations
    with debug_mode():
        driver.connect()
        driver.read_density()

    # Get diagnostic report
    report = get_diagnostic_report()
    print(report.summary)
"""

from __future__ import annotations

import functools
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

from pydantic import BaseModel, Field

from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])

# Thread-local storage for debug context
_debug_context = threading.local()


class DebugLevel(str, Enum):
    """Debug output verbosity levels."""

    OFF = "off"
    MINIMAL = "minimal"
    VERBOSE = "verbose"
    TRACE = "trace"


class ProtocolDirection(str, Enum):
    """Direction of protocol messages."""

    SEND = "send"
    RECEIVE = "receive"


@dataclass
class ProtocolMessage:
    """Recorded protocol message for debugging."""

    timestamp: datetime
    direction: ProtocolDirection
    device_type: str
    command: str
    raw_bytes: bytes | None = None
    response: str | None = None
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "device_type": self.device_type,
            "command": self.command,
            "raw_bytes": self.raw_bytes.hex() if self.raw_bytes else None,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class OperationMetrics:
    """Performance metrics for a hardware operation."""

    operation: str
    start_time: datetime
    end_time: datetime | None = None
    success: bool = False
    error: str | None = None
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Calculate operation duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class DiagnosticReport(BaseModel):
    """Hardware diagnostic report."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_info: dict[str, Any] = Field(default_factory=dict)
    devices: list[dict[str, Any]] = Field(default_factory=list)
    connections: list[dict[str, Any]] = Field(default_factory=list)
    recent_operations: list[dict[str, Any]] = Field(default_factory=list)
    protocol_messages: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Hardware Diagnostic Report ({self.report_id})",
            f"Generated: {self.generated_at.isoformat()}",
            "",
            f"Devices: {len(self.devices)}",
            f"Active connections: {len([c for c in self.connections if c.get('connected')])}",
            f"Recent operations: {len(self.recent_operations)}",
            f"Warnings: {len(self.warnings)}",
            f"Errors: {len(self.errors)}",
        ]

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings[:5]:
                lines.append(f"  - {warning}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors[:5]:
                lines.append(f"  - {error}")

        return "\n".join(lines)


class ProtocolLogger:
    """Logger for protocol-level communication.

    Records all commands sent and responses received from hardware devices.
    Useful for debugging communication issues.

    Example:
        protocol_logger = ProtocolLogger(device_type="spectrophotometer")

        # Log outgoing command
        protocol_logger.log_send("*IDN?")

        # Log response with latency
        protocol_logger.log_receive("*IDN?", "X-Rite i1Pro v1.2", latency_ms=15.3)
    """

    def __init__(
        self,
        device_type: str,
        max_messages: int = 1000,
        log_raw_bytes: bool = False,
    ) -> None:
        """Initialize protocol logger.

        Args:
            device_type: Type of device being logged.
            max_messages: Maximum messages to retain (circular buffer).
            log_raw_bytes: Include raw byte data in logs.
        """
        self.device_type = device_type
        self.max_messages = max_messages
        self.log_raw_bytes = log_raw_bytes
        self._messages: list[ProtocolMessage] = []
        self._lock = threading.Lock()

    def log_send(
        self,
        command: str,
        raw_bytes: bytes | None = None,
    ) -> None:
        """Log an outgoing command.

        Args:
            command: Command string.
            raw_bytes: Raw bytes if applicable.
        """
        message = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.SEND,
            device_type=self.device_type,
            command=command,
            raw_bytes=raw_bytes if self.log_raw_bytes else None,
        )
        self._add_message(message)
        logger.debug(
            f"[{self.device_type}] TX: {command}",
            extra={"direction": "send", "device": self.device_type},
        )

    def log_receive(
        self,
        command: str,
        response: str,
        latency_ms: float | None = None,
        raw_bytes: bytes | None = None,
    ) -> None:
        """Log a received response.

        Args:
            command: Original command.
            response: Response received.
            latency_ms: Round-trip latency in milliseconds.
            raw_bytes: Raw bytes if applicable.
        """
        message = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.RECEIVE,
            device_type=self.device_type,
            command=command,
            response=response,
            latency_ms=latency_ms,
            raw_bytes=raw_bytes if self.log_raw_bytes else None,
        )
        self._add_message(message)

        latency_str = f" ({latency_ms:.1f}ms)" if latency_ms else ""
        logger.debug(
            f"[{self.device_type}] RX: {response[:100]}{'...' if len(response) > 100 else ''}{latency_str}",
            extra={
                "direction": "receive",
                "device": self.device_type,
                "latency_ms": latency_ms,
            },
        )

    def log_error(
        self,
        command: str,
        error: str,
        latency_ms: float | None = None,
    ) -> None:
        """Log a communication error.

        Args:
            command: Command that failed.
            error: Error description.
            latency_ms: Time until error occurred.
        """
        message = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.RECEIVE,
            device_type=self.device_type,
            command=command,
            error=error,
            latency_ms=latency_ms,
        )
        self._add_message(message)
        logger.warning(
            f"[{self.device_type}] ERR: {command} -> {error}",
            extra={
                "direction": "error",
                "device": self.device_type,
                "error": error,
            },
        )

    def get_messages(
        self,
        limit: int | None = None,
        direction: ProtocolDirection | None = None,
    ) -> list[ProtocolMessage]:
        """Get recorded messages.

        Args:
            limit: Maximum messages to return (most recent).
            direction: Filter by direction.

        Returns:
            List of protocol messages.
        """
        with self._lock:
            messages = self._messages.copy()

        if direction:
            messages = [m for m in messages if m.direction == direction]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_statistics(self) -> dict[str, Any]:
        """Get communication statistics.

        Returns:
            Dictionary with message counts, average latency, etc.
        """
        with self._lock:
            messages = self._messages.copy()

        if not messages:
            return {
                "total_messages": 0,
                "sends": 0,
                "receives": 0,
                "errors": 0,
                "avg_latency_ms": None,
            }

        sends = [m for m in messages if m.direction == ProtocolDirection.SEND]
        receives = [m for m in messages if m.direction == ProtocolDirection.RECEIVE]
        errors = [m for m in messages if m.error]
        latencies = [m.latency_ms for m in receives if m.latency_ms is not None]

        return {
            "total_messages": len(messages),
            "sends": len(sends),
            "receives": len(receives),
            "errors": len(errors),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "min_latency_ms": min(latencies) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
        }

    def clear(self) -> None:
        """Clear all recorded messages."""
        with self._lock:
            self._messages.clear()

    def export_to_file(self, path: Path | str) -> None:
        """Export messages to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "device_type": self.device_type,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "statistics": self.get_statistics(),
                "messages": [m.to_dict() for m in self._messages],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Protocol log exported to {path}")

    def _add_message(self, message: ProtocolMessage) -> None:
        """Add message to buffer, maintaining size limit."""
        with self._lock:
            self._messages.append(message)
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages :]


class HardwareDebugger:
    """Central debugger for hardware operations.

    Provides unified debugging capabilities across all hardware drivers.
    Tracks operations, performance metrics, and enables detailed logging.

    Example:
        debugger = HardwareDebugger()
        debugger.enable(level=DebugLevel.VERBOSE)

        with debugger.track_operation("read_density"):
            driver.read_density()

        print(debugger.get_performance_report())
    """

    _instance: HardwareDebugger | None = None
    _lock = threading.Lock()

    def __new__(cls) -> HardwareDebugger:
        """Singleton instance creation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize debugger."""
        if self._initialized:
            return

        self._enabled = False
        self._level = DebugLevel.OFF
        self._protocol_loggers: dict[str, ProtocolLogger] = {}
        self._operations: list[OperationMetrics] = []
        self._max_operations = 10000
        self._initialized = True

    @property
    def enabled(self) -> bool:
        """Check if debugging is enabled."""
        return self._enabled

    @property
    def level(self) -> DebugLevel:
        """Get current debug level."""
        return self._level

    def enable(self, level: DebugLevel = DebugLevel.VERBOSE) -> None:
        """Enable debugging.

        Args:
            level: Debug verbosity level.
        """
        self._enabled = True
        self._level = level
        logger.info(f"Hardware debugging enabled at level: {level.value}")

    def disable(self) -> None:
        """Disable debugging."""
        self._enabled = False
        self._level = DebugLevel.OFF
        logger.info("Hardware debugging disabled")

    def get_protocol_logger(self, device_type: str) -> ProtocolLogger:
        """Get or create protocol logger for device type.

        Args:
            device_type: Type of device.

        Returns:
            Protocol logger instance.
        """
        if device_type not in self._protocol_loggers:
            self._protocol_loggers[device_type] = ProtocolLogger(
                device_type=device_type,
                log_raw_bytes=self._level == DebugLevel.TRACE,
            )
        return self._protocol_loggers[device_type]

    @contextmanager
    def track_operation(
        self,
        operation: str,
        **metadata: Any,
    ) -> Generator[OperationMetrics, None, None]:
        """Track a hardware operation with timing.

        Args:
            operation: Operation name.
            **metadata: Additional metadata to record.

        Yields:
            Operation metrics object.

        Example:
            with debugger.track_operation("calibrate", device="i1Pro") as op:
                driver.calibrate()
            print(f"Calibration took {op.duration_ms}ms")
        """
        metrics = OperationMetrics(
            operation=operation,
            start_time=datetime.now(timezone.utc),
            metadata=metadata,
        )

        if self._enabled and self._level in (DebugLevel.VERBOSE, DebugLevel.TRACE):
            logger.debug(
                f"Starting operation: {operation}",
                extra={"operation": operation, **metadata},
            )

        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.error = str(e)
            raise
        finally:
            metrics.end_time = datetime.now(timezone.utc)
            self._record_operation(metrics)

            if self._enabled and self._level in (DebugLevel.VERBOSE, DebugLevel.TRACE):
                status = "completed" if metrics.success else f"failed: {metrics.error}"
                logger.debug(
                    f"Operation {operation} {status} in {metrics.duration_ms:.1f}ms",
                    extra={
                        "operation": operation,
                        "duration_ms": metrics.duration_ms,
                        "success": metrics.success,
                    },
                )

    def _record_operation(self, metrics: OperationMetrics) -> None:
        """Record operation metrics."""
        self._operations.append(metrics)
        if len(self._operations) > self._max_operations:
            self._operations = self._operations[-self._max_operations :]

    def get_operations(
        self,
        limit: int | None = None,
        operation_filter: str | None = None,
    ) -> list[OperationMetrics]:
        """Get recorded operations.

        Args:
            limit: Maximum operations to return.
            operation_filter: Filter by operation name.

        Returns:
            List of operation metrics.
        """
        ops = self._operations.copy()

        if operation_filter:
            ops = [o for o in ops if operation_filter in o.operation]

        if limit:
            ops = ops[-limit:]

        return ops

    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report for all operations.

        Returns:
            Dictionary with performance statistics.
        """
        if not self._operations:
            return {"total_operations": 0}

        # Group by operation type
        by_operation: dict[str, list[OperationMetrics]] = {}
        for op in self._operations:
            if op.operation not in by_operation:
                by_operation[op.operation] = []
            by_operation[op.operation].append(op)

        report: dict[str, Any] = {
            "total_operations": len(self._operations),
            "success_rate": sum(1 for o in self._operations if o.success)
            / len(self._operations),
            "operations": {},
        }

        for op_name, ops in by_operation.items():
            durations = [o.duration_ms for o in ops if o.duration_ms is not None]
            successful = sum(1 for o in ops if o.success)

            report["operations"][op_name] = {
                "count": len(ops),
                "success_rate": successful / len(ops) if ops else 0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else None,
                "min_duration_ms": min(durations) if durations else None,
                "max_duration_ms": max(durations) if durations else None,
            }

        return report

    def clear(self) -> None:
        """Clear all recorded data."""
        self._operations.clear()
        for pl in self._protocol_loggers.values():
            pl.clear()


def debug_hardware_call(func: F) -> F:
    """Decorator to add debugging to hardware method calls.

    Automatically tracks execution time and logs calls when debugging is enabled.

    Example:
        class MyDriver:
            @debug_hardware_call
            def read_density(self) -> float:
                # Implementation
                pass
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        debugger = HardwareDebugger()

        if not debugger.enabled:
            return func(*args, **kwargs)

        # Get operation name from function
        operation = func.__name__

        # Add class name if this is a method
        if args and hasattr(args[0], "__class__"):
            operation = f"{args[0].__class__.__name__}.{operation}"

        with debugger.track_operation(operation):
            return func(*args, **kwargs)

    return wrapper  # type: ignore


@contextmanager
def debug_mode(level: DebugLevel = DebugLevel.VERBOSE) -> Generator[HardwareDebugger, None, None]:
    """Context manager to temporarily enable debug mode.

    Args:
        level: Debug level to use.

    Yields:
        The hardware debugger instance.

    Example:
        with debug_mode(DebugLevel.TRACE):
            # All hardware operations will be traced
            driver.connect()
            driver.read_density()
    """
    debugger = HardwareDebugger()
    previous_enabled = debugger.enabled
    previous_level = debugger.level

    debugger.enable(level)
    try:
        yield debugger
    finally:
        if previous_enabled:
            debugger.enable(previous_level)
        else:
            debugger.disable()


def get_diagnostic_report() -> DiagnosticReport:
    """Generate a comprehensive diagnostic report.

    Collects information about:
    - System configuration
    - Discovered devices
    - Active connections
    - Recent operations
    - Protocol messages
    - Warnings and errors

    Returns:
        DiagnosticReport instance.

    Example:
        report = get_diagnostic_report()
        print(report.summary)

        # Save to file
        with open("diagnostic.json", "w") as f:
            f.write(report.model_dump_json(indent=2))
    """
    import platform
    import sys

    debugger = HardwareDebugger()
    report = DiagnosticReport()

    # System info
    report.system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "debug_enabled": debugger.enabled,
        "debug_level": debugger.level.value,
    }

    # Try to get device registry info
    try:
        from ptpd_calibration.integrations.hardware.registry import DeviceRegistry

        registry = DeviceRegistry()
        for device_id, device in registry._devices.items():
            report.devices.append({
                "device_id": device_id,
                "device_type": device.device_type.value,
                "is_simulated": device.is_simulated,
                "connected": device.is_connected,
            })
    except Exception as e:
        report.warnings.append(f"Could not get device registry: {e}")

    # Recent operations
    for op in debugger.get_operations(limit=50):
        report.recent_operations.append({
            "operation": op.operation,
            "start_time": op.start_time.isoformat(),
            "duration_ms": op.duration_ms,
            "success": op.success,
            "error": op.error,
            "metadata": op.metadata,
        })

    # Protocol messages from all loggers
    for device_type, pl in debugger._protocol_loggers.items():
        for msg in pl.get_messages(limit=100):
            report.protocol_messages.append(msg.to_dict())

    return report


def save_debug_session(path: Path | str) -> Path:
    """Save current debug session to file.

    Exports all debug data including operations, protocol logs,
    and diagnostic information.

    Args:
        path: Output file path (will use .json extension).

    Returns:
        Path to saved file.

    Example:
        path = save_debug_session("debug_session")
        print(f"Debug session saved to {path}")
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")

    path.parent.mkdir(parents=True, exist_ok=True)

    debugger = HardwareDebugger()
    report = get_diagnostic_report()

    data = {
        "session_id": str(uuid.uuid4())[:8],
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_report": report.model_dump(),
        "performance_report": debugger.get_performance_report(),
        "protocol_statistics": {
            device_type: pl.get_statistics()
            for device_type, pl in debugger._protocol_loggers.items()
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Debug session saved to {path}")
    return path
