"""
Hardware integration exceptions for PTPD Calibration.

Provides a hierarchy of exceptions for hardware-related errors:
- HardwareError (base)
  - DeviceNotFoundError
  - DeviceConnectionError
  - DeviceCommunicationError
  - CalibrationError
  - MeasurementError
  - PrinterError
    - PrintJobError

All exceptions include context about the device and operation that failed.
"""

from typing import Any


class HardwareError(Exception):
    """Base exception for hardware-related errors.

    Attributes:
        device_type: Type of hardware device (spectrophotometer, printer, etc.)
        operation: Operation that failed.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        device_type: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize hardware error.

        Args:
            message: Human-readable error message.
            device_type: Type of hardware device.
            operation: Operation that was being performed.
            details: Additional context as key-value pairs.
        """
        super().__init__(message)
        self.device_type = device_type
        self.operation = operation
        self.details = details or {}

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.device_type:
            parts.append(f"Device: {self.device_type}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        return " | ".join(parts)


class DeviceNotFoundError(HardwareError):
    """Device not found or not accessible.

    Raised when:
    - No device matching criteria is connected
    - Device is connected but not accessible (permissions)
    - Specified port/address does not exist
    """

    def __init__(
        self,
        message: str = "Device not found",
        device_type: str | None = None,
        port: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if port:
            details["port"] = port
        super().__init__(
            message,
            device_type=device_type,
            operation="device_discovery",
            details=details,
            **kwargs,
        )


class DeviceConnectionError(HardwareError):
    """Failed to establish connection with device.

    Raised when:
    - Serial port cannot be opened
    - USB connection fails
    - Device does not respond to initial handshake
    - Connection timeout occurs
    """

    def __init__(
        self,
        message: str = "Failed to connect to device",
        device_type: str | None = None,
        port: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if port:
            details["port"] = port
        if timeout:
            details["timeout_seconds"] = timeout
        super().__init__(
            message,
            device_type=device_type,
            operation="connect",
            details=details,
            **kwargs,
        )


class DeviceCommunicationError(HardwareError):
    """Communication error during device operation.

    Raised when:
    - Command times out waiting for response
    - Invalid or unexpected response received
    - Data corruption detected
    - Device reports error status
    """

    def __init__(
        self,
        message: str = "Communication error with device",
        device_type: str | None = None,
        command: str | None = None,
        response: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if command:
            details["command"] = command
        if response:
            details["response"] = response[:100]  # Truncate long responses
        super().__init__(
            message,
            device_type=device_type,
            operation="communicate",
            details=details,
            **kwargs,
        )


class CalibrationError(HardwareError):
    """Device calibration failed.

    Raised when:
    - White/black reference calibration fails
    - User does not position device correctly
    - Calibration values out of expected range
    - Calibration timeout occurs
    """

    def __init__(
        self,
        message: str = "Calibration failed",
        device_type: str | None = None,
        calibration_type: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if calibration_type:
            details["calibration_type"] = calibration_type
        super().__init__(
            message,
            device_type=device_type,
            operation="calibrate",
            details=details,
            **kwargs,
        )


class MeasurementError(HardwareError):
    """Measurement operation failed.

    Raised when:
    - Density/color measurement fails
    - Measurement values out of valid range
    - Device reports measurement error
    - Insufficient light or positioning issue
    """

    def __init__(
        self,
        message: str = "Measurement failed",
        device_type: str | None = None,
        measurement_type: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if measurement_type:
            details["measurement_type"] = measurement_type
        super().__init__(
            message,
            device_type=device_type,
            operation="measure",
            details=details,
            **kwargs,
        )


class PrinterError(HardwareError):
    """Printer-specific error.

    Base class for printer-related errors.
    """

    def __init__(
        self,
        message: str = "Printer error",
        printer_name: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if printer_name:
            details["printer_name"] = printer_name
        super().__init__(
            message,
            device_type="printer",
            details=details,
            **kwargs,
        )


class PrintJobError(PrinterError):
    """Print job failed.

    Raised when:
    - Print job submission fails
    - Print job is cancelled
    - Paper jam or out of paper
    - Ink/toner exhausted
    - Job timeout
    """

    def __init__(
        self,
        message: str = "Print job failed",
        job_id: str | None = None,
        job_name: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if job_id:
            details["job_id"] = job_id
        if job_name:
            details["job_name"] = job_name
        super().__init__(
            message,
            operation="print",
            details=details,
            **kwargs,
        )


class PrinterNotFoundError(PrinterError):
    """Specified printer not found.

    Raised when the requested printer name does not exist in the system.
    """

    def __init__(
        self,
        message: str = "Printer not found",
        printer_name: str | None = None,
        available_printers: list[str] | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {})
        if available_printers:
            details["available_printers"] = available_printers[:5]  # Limit to 5
        super().__init__(
            message,
            printer_name=printer_name,
            operation="find_printer",
            details=details,
            **kwargs,
        )
