"""
Unit tests for hardware exception classes.

Tests all exception types including the new exceptions added
for connection management and discovery.
"""

import pytest

from ptpd_calibration.integrations.hardware.exceptions import (
    CalibrationError,
    DeviceCommunicationError,
    DeviceConnectionError,
    DeviceNotFoundError,
    DeviceReconnectionError,
    DeviceTimeoutError,
    DiscoveryError,
    HardwareError,
    MeasurementError,
    PermissionDeniedError,
    PrinterDriverError,
    PrinterError,
    PrinterNotFoundError,
    PrintJobError,
)


class TestHardwareErrorBase:
    """Tests for base HardwareError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = HardwareError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.device_type is None
        assert error.operation is None
        assert error.details == {}

    def test_error_with_device_type(self):
        """Test error with device type."""
        error = HardwareError(
            "Error occurred",
            device_type="spectrophotometer",
        )

        assert error.device_type == "spectrophotometer"
        assert "Device: spectrophotometer" in str(error)

    def test_error_with_operation(self):
        """Test error with operation."""
        error = HardwareError(
            "Error occurred",
            operation="calibrate",
        )

        assert error.operation == "calibrate"
        assert "Operation: calibrate" in str(error)

    def test_error_with_details(self):
        """Test error with details."""
        error = HardwareError(
            "Error occurred",
            details={"key": "value", "count": 42},
        )

        assert error.details == {"key": "value", "count": 42}
        assert "Details:" in str(error)

    def test_full_error_message(self):
        """Test complete error message formatting."""
        error = HardwareError(
            "Connection failed",
            device_type="printer",
            operation="connect",
            details={"port": "USB001"},
        )

        message = str(error)
        assert "Connection failed" in message
        assert "Device: printer" in message
        assert "Operation: connect" in message
        assert "port=USB001" in message


class TestDeviceNotFoundError:
    """Tests for DeviceNotFoundError."""

    def test_default_message(self):
        """Test default error message."""
        error = DeviceNotFoundError()
        assert "not found" in str(error).lower()

    def test_with_port(self):
        """Test error with port info."""
        error = DeviceNotFoundError(
            message="Device not found on port",
            port="/dev/ttyUSB0",
        )

        assert error.details["port"] == "/dev/ttyUSB0"

    def test_operation_is_device_discovery(self):
        """Test operation is set to device_discovery."""
        error = DeviceNotFoundError()
        assert error.operation == "device_discovery"


class TestDeviceConnectionError:
    """Tests for DeviceConnectionError."""

    def test_default_message(self):
        """Test default error message."""
        error = DeviceConnectionError()
        assert "connect" in str(error).lower()

    def test_with_port_and_timeout(self):
        """Test error with port and timeout info."""
        error = DeviceConnectionError(
            message="Connection timed out",
            port="/dev/ttyUSB0",
            timeout=5.0,
        )

        assert error.details["port"] == "/dev/ttyUSB0"
        assert error.details["timeout_seconds"] == 5.0


class TestDeviceReconnectionError:
    """Tests for DeviceReconnectionError (new exception)."""

    def test_default_message(self):
        """Test default error message."""
        error = DeviceReconnectionError()
        assert "reconnect" in str(error).lower()

    def test_with_attempts(self):
        """Test error with attempt count."""
        error = DeviceReconnectionError(
            message="All reconnection attempts failed",
            attempts=3,
        )

        assert error.details["reconnect_attempts"] == 3

    def test_inherits_from_connection_error(self):
        """Test inheritance from DeviceConnectionError."""
        error = DeviceReconnectionError()
        assert isinstance(error, DeviceConnectionError)
        assert isinstance(error, HardwareError)


class TestDeviceTimeoutError:
    """Tests for DeviceTimeoutError (new exception)."""

    def test_default_message(self):
        """Test default error message."""
        error = DeviceTimeoutError()
        assert "timeout" in str(error).lower()

    def test_with_timeout_value(self):
        """Test error with timeout value."""
        error = DeviceTimeoutError(
            message="Operation timed out",
            timeout_seconds=30.0,
        )

        assert error.details["timeout_seconds"] == 30.0

    def test_inherits_from_communication_error(self):
        """Test inheritance from DeviceCommunicationError."""
        error = DeviceTimeoutError()
        assert isinstance(error, DeviceCommunicationError)


class TestDeviceCommunicationError:
    """Tests for DeviceCommunicationError."""

    def test_default_message(self):
        """Test default error message."""
        error = DeviceCommunicationError()
        assert "communication" in str(error).lower()

    def test_with_command_and_response(self):
        """Test error with command and response."""
        error = DeviceCommunicationError(
            message="Invalid response",
            command="MEASURE",
            response="ERROR: Invalid state",
        )

        assert error.details["command"] == "MEASURE"
        assert "ERROR" in error.details["response"]

    def test_response_truncation(self):
        """Test long responses are truncated."""
        long_response = "A" * 200  # Longer than 100 chars

        error = DeviceCommunicationError(
            message="Error",
            response=long_response,
        )

        assert len(error.details["response"]) == 100


class TestCalibrationError:
    """Tests for CalibrationError."""

    def test_default_message(self):
        """Test default error message."""
        error = CalibrationError()
        assert "calibration" in str(error).lower()

    def test_with_calibration_type(self):
        """Test error with calibration type."""
        error = CalibrationError(
            message="White calibration failed",
            calibration_type="white_reference",
        )

        assert error.details["calibration_type"] == "white_reference"


class TestMeasurementError:
    """Tests for MeasurementError."""

    def test_default_message(self):
        """Test default error message."""
        error = MeasurementError()
        assert "measurement" in str(error).lower()

    def test_with_measurement_type(self):
        """Test error with measurement type."""
        error = MeasurementError(
            message="Spectral measurement failed",
            measurement_type="spectral",
        )

        assert error.details["measurement_type"] == "spectral"


class TestPrinterErrors:
    """Tests for printer-related exceptions."""

    def test_printer_error_base(self):
        """Test base PrinterError."""
        error = PrinterError(
            message="Printer error",
            printer_name="EPSON P800",
        )

        assert error.device_type == "printer"
        assert error.details["printer_name"] == "EPSON P800"

    def test_printer_not_found_error(self):
        """Test PrinterNotFoundError."""
        error = PrinterNotFoundError(
            printer_name="NonexistentPrinter",
            available_printers=["Printer1", "Printer2"],
        )

        assert error.details["available_printers"] == ["Printer1", "Printer2"]

    def test_printer_not_found_truncates_list(self):
        """Test available_printers list is truncated."""
        many_printers = [f"Printer{i}" for i in range(10)]

        error = PrinterNotFoundError(
            printer_name="Missing",
            available_printers=many_printers,
        )

        assert len(error.details["available_printers"]) == 5

    def test_print_job_error(self):
        """Test PrintJobError."""
        error = PrintJobError(
            message="Print failed",
            job_id="job-001",
            job_name="Test Print",
        )

        assert error.details["job_id"] == "job-001"
        assert error.details["job_name"] == "Test Print"

    def test_printer_driver_error(self):
        """Test PrinterDriverError (new exception)."""
        error = PrinterDriverError(
            message="Driver initialization failed",
            driver_name="Win32PrinterDriver",
        )

        assert error.details["driver_name"] == "Win32PrinterDriver"
        assert isinstance(error, PrinterError)


class TestDiscoveryErrors:
    """Tests for discovery-related exceptions (new)."""

    def test_discovery_error(self):
        """Test DiscoveryError."""
        error = DiscoveryError(
            message="Discovery failed",
            discovery_method="USB",
        )

        assert error.operation == "discover"
        assert error.details["discovery_method"] == "USB"

    def test_permission_denied_error(self):
        """Test PermissionDeniedError."""
        error = PermissionDeniedError(
            message="Cannot access device",
            device_path="/dev/ttyUSB0",
            required_permission="dialout group",
        )

        assert error.details["device_path"] == "/dev/ttyUSB0"
        assert error.details["required_permission"] == "dialout group"

    def test_permission_denied_inherits_from_discovery(self):
        """Test PermissionDeniedError inherits from DiscoveryError."""
        error = PermissionDeniedError()

        assert isinstance(error, DiscoveryError)
        assert isinstance(error, HardwareError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_hardware_error(self):
        """Test all exceptions inherit from HardwareError."""
        exceptions = [
            DeviceNotFoundError,
            DeviceConnectionError,
            DeviceReconnectionError,
            DeviceCommunicationError,
            DeviceTimeoutError,
            CalibrationError,
            MeasurementError,
            PrinterError,
            PrinterNotFoundError,
            PrintJobError,
            PrinterDriverError,
            DiscoveryError,
            PermissionDeniedError,
        ]

        for exc_class in exceptions:
            instance = exc_class()
            assert isinstance(instance, HardwareError)

    def test_printer_errors_inherit_from_printer_error(self):
        """Test printer exceptions inherit from PrinterError."""
        printer_exceptions = [
            PrinterNotFoundError,
            PrintJobError,
            PrinterDriverError,
        ]

        for exc_class in printer_exceptions:
            instance = exc_class()
            assert isinstance(instance, PrinterError)

    def test_all_exceptions_are_catchable_as_exception(self):
        """Test all exceptions can be caught as Exception."""
        exceptions = [
            HardwareError,
            DeviceNotFoundError,
            DeviceConnectionError,
            PrinterError,
            DiscoveryError,
        ]

        for exc_class in exceptions:
            try:
                raise exc_class("Test error")
            except Exception as e:
                assert isinstance(e, exc_class)
