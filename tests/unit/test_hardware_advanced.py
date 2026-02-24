"""
Comprehensive unit tests for hardware and advanced features modules.

Tests modules:
1. hardware.exceptions - Hardware exception classes
2. hardware.simulated - Simulated hardware for testing
3. hardware.base - Base hardware ABC classes
4. hardware.debug - Debug hardware implementation
5. advanced.features - Advanced features (partial coverage)
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest
from PIL import Image

# Import hardware modules directly to avoid weather/httpx import issues
from ptpd_calibration.integrations.hardware.base import (
    HardwareDeviceBase,
    parse_device_response,
)
from ptpd_calibration.integrations.hardware.debug import (
    DebugLevel,
    DiagnosticReport,
    HardwareDebugger,
    OperationMetrics,
    ProtocolDirection,
    ProtocolLogger,
    ProtocolMessage,
    debug_hardware_call,
    debug_mode,
    get_diagnostic_report,
    save_debug_session,
)
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
from ptpd_calibration.integrations.hardware.simulated import (
    SimulatedPrinter,
    SimulatedSpectrophotometer,
)
from ptpd_calibration.integrations.protocols import (
    DeviceInfo,
    DeviceStatus,
    PrintJob,
)

# Import advanced features with qrcode optional
try:
    from ptpd_calibration.advanced.features import (
        AlternativeProcessParams,
        AlternativeProcessSimulator,
        BlendMode,
        HistoricStyle,
        NegativeBlender,
        PrintComparison,
        PrintMetadata,
        QRMetadataGenerator,
        StyleParameters,
        StyleTransfer,
    )

    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


# =============================================================================
# HARDWARE EXCEPTIONS TESTS
# =============================================================================


class TestHardwareError:
    """Test HardwareError base exception class."""

    def test_basic_creation(self):
        """Test creating a basic hardware error."""
        error = HardwareError("Test error")
        assert str(error) == "Test error"
        assert error.device_type is None
        assert error.operation is None
        assert error.details == {}

    def test_with_device_type(self):
        """Test error with device type."""
        error = HardwareError("Test error", device_type="spectrophotometer")
        assert "Device: spectrophotometer" in str(error)
        assert error.device_type == "spectrophotometer"

    def test_with_operation(self):
        """Test error with operation."""
        error = HardwareError("Test error", operation="connect")
        assert "Operation: connect" in str(error)
        assert error.operation == "connect"

    def test_with_details(self):
        """Test error with details."""
        details = {"port": "/dev/ttyUSB0", "timeout": 5.0}
        error = HardwareError("Test error", details=details)
        assert "Details:" in str(error)
        assert "port=/dev/ttyUSB0" in str(error)
        assert "timeout=5.0" in str(error)

    def test_full_error(self):
        """Test error with all fields."""
        error = HardwareError(
            "Connection failed",
            device_type="printer",
            operation="connect",
            details={"port": "USB001", "attempts": 3},
        )
        error_str = str(error)
        assert "Connection failed" in error_str
        assert "Device: printer" in error_str
        assert "Operation: connect" in error_str
        assert "port=USB001" in error_str
        assert "attempts=3" in error_str


class TestDeviceNotFoundError:
    """Test DeviceNotFoundError exception."""

    def test_default_message(self):
        """Test with default message."""
        error = DeviceNotFoundError()
        assert "Device not found" in str(error)
        assert error.operation == "device_discovery"

    def test_with_device_type(self):
        """Test with device type."""
        error = DeviceNotFoundError(device_type="spectrophotometer")
        assert error.device_type == "spectrophotometer"

    def test_with_port(self):
        """Test with port information."""
        error = DeviceNotFoundError(port="/dev/ttyUSB0")
        assert error.details["port"] == "/dev/ttyUSB0"


class TestDeviceConnectionError:
    """Test DeviceConnectionError exception."""

    def test_default_creation(self):
        """Test default connection error."""
        error = DeviceConnectionError()
        assert "Failed to connect" in str(error)
        assert error.operation == "connect"

    def test_with_timeout(self):
        """Test with timeout information."""
        error = DeviceConnectionError(timeout=10.0)
        assert error.details["timeout_seconds"] == 10.0


class TestDeviceCommunicationError:
    """Test DeviceCommunicationError exception."""

    def test_with_command(self):
        """Test with command information."""
        error = DeviceCommunicationError(command="*IDN?")
        assert error.details["command"] == "*IDN?"

    def test_with_response(self):
        """Test with response information."""
        long_response = "x" * 200
        error = DeviceCommunicationError(response=long_response)
        # Response should be truncated to 100 chars
        assert len(error.details["response"]) == 100


class TestCalibrationError:
    """Test CalibrationError exception."""

    def test_with_calibration_type(self):
        """Test with calibration type."""
        error = CalibrationError(calibration_type="white")
        assert error.details["calibration_type"] == "white"
        assert error.operation == "calibrate"


class TestMeasurementError:
    """Test MeasurementError exception."""

    def test_with_measurement_type(self):
        """Test with measurement type."""
        error = MeasurementError(measurement_type="density")
        assert error.details["measurement_type"] == "density"
        assert error.operation == "measure"


class TestPrinterError:
    """Test PrinterError exception."""

    def test_with_printer_name(self):
        """Test with printer name."""
        error = PrinterError(printer_name="Epson P900")
        assert error.details["printer_name"] == "Epson P900"
        assert error.device_type == "printer"


class TestPrintJobError:
    """Test PrintJobError exception."""

    def test_with_job_info(self):
        """Test with job information."""
        error = PrintJobError(job_id="job-001", job_name="test.tif")
        assert error.details["job_id"] == "job-001"
        assert error.details["job_name"] == "test.tif"
        assert error.operation == "print"


class TestPrinterNotFoundError:
    """Test PrinterNotFoundError exception."""

    def test_with_available_printers(self):
        """Test with list of available printers."""
        printers = ["Printer1", "Printer2", "Printer3", "Printer4", "Printer5", "Printer6"]
        error = PrinterNotFoundError(available_printers=printers)
        # Should only include first 5
        assert len(error.details["available_printers"]) == 5


class TestPrinterDriverError:
    """Test PrinterDriverError exception."""

    def test_with_driver_name(self):
        """Test with driver name."""
        error = PrinterDriverError(driver_name="cups")
        assert error.details["driver_name"] == "cups"


class TestDeviceReconnectionError:
    """Test DeviceReconnectionError exception."""

    def test_with_attempts(self):
        """Test with reconnection attempts."""
        error = DeviceReconnectionError(attempts=5)
        assert error.details["reconnect_attempts"] == 5
        assert error.operation == "reconnect"


class TestDeviceTimeoutError:
    """Test DeviceTimeoutError exception."""

    def test_with_timeout_seconds(self):
        """Test with timeout value."""
        error = DeviceTimeoutError(timeout_seconds=30.0)
        assert error.details["timeout_seconds"] == 30.0
        assert error.operation == "timeout"


class TestDiscoveryError:
    """Test DiscoveryError exception."""

    def test_with_discovery_method(self):
        """Test with discovery method."""
        error = DiscoveryError(discovery_method="usb")
        assert error.details["discovery_method"] == "usb"
        assert error.operation == "discover"


class TestPermissionDeniedError:
    """Test PermissionDeniedError exception."""

    def test_with_device_path(self):
        """Test with device path."""
        error = PermissionDeniedError(
            device_path="/dev/ttyUSB0", required_permission="dialout"
        )
        assert error.details["device_path"] == "/dev/ttyUSB0"
        assert error.details["required_permission"] == "dialout"


# =============================================================================
# SIMULATED HARDWARE TESTS
# =============================================================================


class TestSimulatedSpectrophotometer:
    """Test SimulatedSpectrophotometer class."""

    def test_initialization(self):
        """Test spectrophotometer initialization."""
        spectro = SimulatedSpectrophotometer()
        assert spectro.status == DeviceStatus.DISCONNECTED
        assert spectro.device_info is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        spectro = SimulatedSpectrophotometer(
            simulate_delay=False, noise_level=0.01, base_density=0.1, max_density=2.0
        )
        assert spectro._simulate_delay is False
        assert spectro._noise_level == 0.01
        assert spectro._base_density == 0.1
        assert spectro._max_density == 2.0

    def test_connect(self):
        """Test connection to simulated device."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        result = spectro.connect()
        assert result is True
        assert spectro.status == DeviceStatus.CONNECTED
        assert spectro.device_info is not None
        assert spectro.device_info.vendor == "Simulated"
        assert "density" in spectro.device_info.capabilities

    def test_connect_with_delay(self):
        """Test connection with delay simulation."""
        spectro = SimulatedSpectrophotometer(simulate_delay=True)
        start = time.time()
        spectro.connect()
        duration = time.time() - start
        # Should have some delay
        assert duration >= 0.4  # Allow some tolerance

    def test_disconnect(self):
        """Test disconnection."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        spectro.connect()
        spectro.disconnect()
        assert spectro.status == DeviceStatus.DISCONNECTED
        assert spectro.device_info is None

    def test_calibrate_white(self):
        """Test white calibration."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        spectro.connect()
        result = spectro.calibrate_white()
        assert result is True
        assert spectro._calibrated is True

    def test_calibrate_white_not_connected(self):
        """Test white calibration when not connected."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        with pytest.raises(RuntimeError, match="not connected"):
            spectro.calibrate_white()

    def test_calibrate_black(self):
        """Test black calibration."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        spectro.connect()
        result = spectro.calibrate_black()
        assert result is True

    def test_calibrate_black_not_connected(self):
        """Test black calibration when not connected."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        result = spectro.calibrate_black()
        assert result is False

    def test_read_density(self):
        """Test density measurement."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        spectro.connect()
        measurement = spectro.read_density()

        assert measurement.density >= 0
        assert measurement.density <= 2.5
        assert 0 <= measurement.lab_l <= 100
        assert -128 <= measurement.lab_a <= 128
        assert -128 <= measurement.lab_b <= 128
        assert measurement.aperture_size == "medium"
        assert measurement.measurement_mode == "reflection"

    def test_read_density_not_connected(self):
        """Test density reading when not connected."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        with pytest.raises(RuntimeError, match="not connected"):
            spectro.read_density()

    def test_read_density_sequence(self):
        """Test multiple density readings simulate step tablet."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False, noise_level=0.0)
        spectro.connect()

        measurements = [spectro.read_density() for _ in range(21)]
        densities = [m.density for m in measurements]

        # Densities should increase over sequence
        assert densities[0] < densities[10] < densities[20]

    def test_read_spectral(self):
        """Test spectral measurement."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        spectro.connect()
        spectral = spectro.read_spectral()

        assert len(spectral.wavelengths) == len(spectral.values)
        assert spectral.start_nm == 380.0
        assert spectral.end_nm == 730.0
        assert all(0 <= v <= 1 for v in spectral.values)

    def test_read_spectral_not_connected(self):
        """Test spectral reading when not connected."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False)
        with pytest.raises(RuntimeError, match="not connected"):
            spectro.read_spectral()

    def test_reset_measurement_count(self):
        """Test resetting measurement counter."""
        spectro = SimulatedSpectrophotometer(simulate_delay=False, noise_level=0.0)
        spectro.connect()

        m1 = spectro.read_density()
        spectro.reset_measurement_count()
        m2 = spectro.read_density()

        # After reset, should get same density
        assert abs(m1.density - m2.density) < 0.01


class TestSimulatedPrinter:
    """Test SimulatedPrinter class."""

    def test_initialization(self):
        """Test printer initialization."""
        printer = SimulatedPrinter()
        assert printer.status == DeviceStatus.DISCONNECTED
        assert printer.device_info is None

    def test_connect(self):
        """Test connection."""
        printer = SimulatedPrinter(simulate_delay=False)
        result = printer.connect()
        assert result is True
        assert printer.status == DeviceStatus.CONNECTED
        assert printer.device_info is not None
        assert "color" in printer.device_info.capabilities

    def test_disconnect(self):
        """Test disconnection."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()
        printer.disconnect()
        assert printer.status == DeviceStatus.DISCONNECTED

    def test_print_image_success(self):
        """Test successful print job."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()

        job = PrintJob(name="test.tif", image_path="/tmp/test.tif", copies=1)
        result = printer.print_image(job)

        assert result.success is True
        assert result.job_id is not None
        assert result.pages_printed == 1

    def test_print_image_not_connected(self):
        """Test print when not connected."""
        printer = SimulatedPrinter(simulate_delay=False)
        job = PrintJob(name="test.tif", image_path="/tmp/test.tif")
        result = printer.print_image(job)

        assert result.success is False
        assert "not connected" in result.error

    def test_print_image_with_failure(self):
        """Test print with simulated failure."""
        printer = SimulatedPrinter(simulate_delay=False, failure_rate=1.0)
        printer.connect()

        job = PrintJob(name="test.tif", image_path="/tmp/test.tif")
        result = printer.print_image(job)

        assert result.success is False
        assert "Simulated print failure" in result.error

    def test_get_paper_sizes(self):
        """Test getting available paper sizes."""
        printer = SimulatedPrinter()
        sizes = printer.get_paper_sizes()
        assert len(sizes) > 0
        assert "8x10" in sizes

    def test_get_resolutions(self):
        """Test getting available resolutions."""
        printer = SimulatedPrinter()
        resolutions = printer.get_resolutions()
        assert len(resolutions) > 0
        assert 2880 in resolutions

    def test_get_ink_levels(self):
        """Test getting ink levels."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()
        levels = printer.get_ink_levels()

        assert len(levels) > 0
        for color, info in levels.items():
            assert "level" in info
            assert "status" in info
            assert 0 <= info["level"] <= 100

    def test_get_ink_levels_not_connected(self):
        """Test ink levels when not connected."""
        printer = SimulatedPrinter()
        levels = printer.get_ink_levels()
        assert levels == {}


# =============================================================================
# HARDWARE BASE TESTS
# =============================================================================


class ConcreteHardwareDevice(HardwareDeviceBase):
    """Concrete implementation for testing."""

    def connect(self, **kwargs: Any) -> bool:
        self._set_status(DeviceStatus.CONNECTING, "Connecting...")
        info = DeviceInfo(vendor="Test", model="Device", capabilities=["test"])
        self._set_device_info(info)
        self._set_status(DeviceStatus.CONNECTED, "Connected")
        return True

    def disconnect(self) -> None:
        self._set_status(DeviceStatus.DISCONNECTED, "Disconnected")
        self._clear_device_info()


class TestHardwareDeviceBase:
    """Test HardwareDeviceBase abstract class."""

    def test_initialization(self):
        """Test device initialization."""
        device = ConcreteHardwareDevice(device_type="test_device")
        assert device.status == DeviceStatus.DISCONNECTED
        assert device.device_info is None
        assert device.is_connected is False

    def test_connect(self):
        """Test connection flow."""
        device = ConcreteHardwareDevice()
        result = device.connect()
        assert result is True
        assert device.is_connected is True
        assert device.device_info is not None

    def test_disconnect(self):
        """Test disconnection."""
        device = ConcreteHardwareDevice()
        device.connect()
        device.disconnect()
        assert device.status == DeviceStatus.DISCONNECTED
        assert device.device_info is None

    def test_protocol_logger_lazy_init(self):
        """Test protocol logger is lazy initialized."""
        device = ConcreteHardwareDevice()
        logger = device.protocol_logger
        assert isinstance(logger, ProtocolLogger)

    def test_track_operation(self):
        """Test operation tracking."""
        device = ConcreteHardwareDevice()
        with device._track_operation("test_op", param="value"):
            time.sleep(0.01)

    def test_log_command(self):
        """Test command logging."""
        device = ConcreteHardwareDevice()
        start_time = device._log_command("TEST")
        assert isinstance(start_time, float)

    def test_log_response(self):
        """Test response logging."""
        device = ConcreteHardwareDevice()
        start_time = device._log_command("TEST")
        device._log_response("TEST", "OK", start_time)

    def test_log_error(self):
        """Test error logging."""
        device = ConcreteHardwareDevice()
        start_time = device._log_command("TEST")
        device._log_error("TEST", "Error occurred", start_time)


class TestParseDeviceResponse:
    """Test parse_device_response utility function."""

    def test_basic_parsing(self):
        """Test basic key-value parsing."""
        result = parse_device_response("OK:D=1.234,L=50.12")
        assert result["D"] == pytest.approx(1.234)
        assert result["L"] == pytest.approx(50.12)

    def test_with_field_map(self):
        """Test parsing with type conversion."""
        field_map = {"ID": str, "COUNT": int}
        result = parse_device_response("OK:ID=ABC123,COUNT=42", field_map=field_map)
        assert result["ID"] == "ABC123"
        assert result["COUNT"] == 42

    def test_custom_delimiter(self):
        """Test custom delimiter."""
        result = parse_device_response("OK:A=1;B=2", delimiter=";")
        assert result["A"] == 1
        assert result["B"] == 2

    def test_custom_separator(self):
        """Test custom key-value separator."""
        result = parse_device_response("OK:A:1,B:2", key_value_separator=":")
        assert result["A"] == 1
        assert result["B"] == 2

    def test_invalid_values(self):
        """Test handling of invalid values."""
        result = parse_device_response("OK:A=invalid,B=2")
        assert result["A"] == "invalid"
        assert result["B"] == 2


# =============================================================================
# HARDWARE DEBUG TESTS
# =============================================================================


class TestProtocolMessage:
    """Test ProtocolMessage dataclass."""

    def test_creation(self):
        """Test creating a protocol message."""
        msg = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.SEND,
            device_type="test",
            command="*IDN?",
        )
        assert msg.direction == ProtocolDirection.SEND
        assert msg.command == "*IDN?"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.RECEIVE,
            device_type="test",
            command="*IDN?",
            response="Test Device v1.0",
            latency_ms=15.3,
        )
        d = msg.to_dict()
        assert d["direction"] == "receive"
        assert d["command"] == "*IDN?"
        assert d["response"] == "Test Device v1.0"
        assert d["latency_ms"] == 15.3


class TestOperationMetrics:
    """Test OperationMetrics dataclass."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now(timezone.utc)
        metrics = OperationMetrics(operation="test", start_time=start)
        time.sleep(0.01)
        metrics.end_time = datetime.now(timezone.utc)
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0

    def test_duration_none_when_not_ended(self):
        """Test duration is None when operation not ended."""
        metrics = OperationMetrics(operation="test", start_time=datetime.now(timezone.utc))
        assert metrics.duration_ms is None


class TestDiagnosticReport:
    """Test DiagnosticReport model."""

    def test_creation(self):
        """Test creating diagnostic report."""
        report = DiagnosticReport()
        assert report.report_id is not None
        assert len(report.report_id) == 8

    def test_summary_generation(self):
        """Test summary string generation."""
        report = DiagnosticReport()
        report.devices = [{"name": "test"}]
        report.warnings = ["Warning 1"]
        report.errors = ["Error 1"]
        summary = report.summary
        assert "Devices: 1" in summary
        assert "Warnings: 1" in summary
        assert "Errors: 1" in summary


class TestProtocolLogger:
    """Test ProtocolLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = ProtocolLogger(device_type="test")
        assert logger.device_type == "test"
        assert logger.max_messages == 1000

    def test_log_send(self):
        """Test logging sent commands."""
        logger = ProtocolLogger(device_type="test")
        logger.log_send("*IDN?")
        messages = logger.get_messages()
        assert len(messages) == 1
        assert messages[0].direction == ProtocolDirection.SEND
        assert messages[0].command == "*IDN?"

    def test_log_receive(self):
        """Test logging received responses."""
        logger = ProtocolLogger(device_type="test")
        logger.log_receive("*IDN?", "Test Device", latency_ms=10.5)
        messages = logger.get_messages()
        assert len(messages) == 1
        assert messages[0].response == "Test Device"
        assert messages[0].latency_ms == 10.5

    def test_log_error(self):
        """Test logging errors."""
        logger = ProtocolLogger(device_type="test")
        logger.log_error("*IDN?", "Timeout")
        messages = logger.get_messages()
        assert len(messages) == 1
        assert messages[0].error == "Timeout"

    def test_get_messages_with_filter(self):
        """Test filtering messages by direction."""
        logger = ProtocolLogger(device_type="test")
        logger.log_send("CMD1")
        logger.log_receive("CMD1", "OK")
        logger.log_send("CMD2")

        sends = logger.get_messages(direction=ProtocolDirection.SEND)
        assert len(sends) == 2
        receives = logger.get_messages(direction=ProtocolDirection.RECEIVE)
        assert len(receives) == 1

    def test_get_messages_with_limit(self):
        """Test limiting returned messages."""
        logger = ProtocolLogger(device_type="test")
        for i in range(10):
            logger.log_send(f"CMD{i}")
        messages = logger.get_messages(limit=5)
        assert len(messages) == 5

    def test_get_statistics(self):
        """Test statistics generation."""
        logger = ProtocolLogger(device_type="test")
        logger.log_send("CMD1")
        logger.log_receive("CMD1", "OK", latency_ms=10.0)
        logger.log_send("CMD2")
        logger.log_receive("CMD2", "OK", latency_ms=20.0)
        logger.log_error("CMD3", "Error")

        stats = logger.get_statistics()
        assert stats["total_messages"] == 5
        assert stats["sends"] == 2
        # log_error creates RECEIVE direction messages
        assert stats["receives"] == 3
        assert stats["errors"] == 1
        assert stats["avg_latency_ms"] == 15.0

    def test_clear(self):
        """Test clearing messages."""
        logger = ProtocolLogger(device_type="test")
        logger.log_send("CMD")
        logger.clear()
        assert len(logger.get_messages()) == 0

    def test_export_to_file(self, tmp_path):
        """Test exporting to JSON file."""
        logger = ProtocolLogger(device_type="test")
        logger.log_send("CMD")
        logger.log_receive("CMD", "OK", latency_ms=5.0)

        output_file = tmp_path / "protocol.json"
        logger.export_to_file(output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data["device_type"] == "test"
        assert len(data["messages"]) == 2

    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        logger = ProtocolLogger(device_type="test", max_messages=10)
        for i in range(20):
            logger.log_send(f"CMD{i}")
        messages = logger.get_messages()
        assert len(messages) == 10
        assert messages[0].command == "CMD10"


class TestHardwareDebugger:
    """Test HardwareDebugger singleton class."""

    def test_singleton(self):
        """Test singleton pattern."""
        d1 = HardwareDebugger()
        d2 = HardwareDebugger()
        assert d1 is d2

    def test_enable_disable(self):
        """Test enabling/disabling debug mode."""
        debugger = HardwareDebugger()
        debugger.disable()
        assert debugger.enabled is False

        debugger.enable(DebugLevel.VERBOSE)
        assert debugger.enabled is True
        assert debugger.level == DebugLevel.VERBOSE

        debugger.disable()
        assert debugger.enabled is False

    def test_get_protocol_logger(self):
        """Test getting protocol logger."""
        debugger = HardwareDebugger()
        logger = debugger.get_protocol_logger("test_device")
        assert isinstance(logger, ProtocolLogger)
        assert logger.device_type == "test_device"

        # Should return same logger for same device
        logger2 = debugger.get_protocol_logger("test_device")
        assert logger is logger2

    def test_track_operation(self):
        """Test operation tracking."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.VERBOSE)

        with debugger.track_operation("test_op", param="value") as metrics:
            time.sleep(0.01)

        assert metrics.success is True
        assert metrics.duration_ms is not None
        assert metrics.metadata["param"] == "value"

    def test_track_operation_with_error(self):
        """Test tracking failed operation."""
        debugger = HardwareDebugger()
        debugger.enable()

        with pytest.raises(ValueError):
            with debugger.track_operation("test_op"):
                raise ValueError("Test error")

        # Metrics should still be recorded
        ops = debugger.get_operations()
        assert len(ops) > 0

    def test_get_operations(self):
        """Test retrieving operations."""
        debugger = HardwareDebugger()
        debugger.clear()

        with debugger.track_operation("op1"):
            pass
        with debugger.track_operation("op2"):
            pass

        ops = debugger.get_operations()
        assert len(ops) >= 2

    def test_get_operations_with_filter(self):
        """Test filtering operations."""
        debugger = HardwareDebugger()
        debugger.clear()

        with debugger.track_operation("device.connect"):
            pass
        with debugger.track_operation("device.measure"):
            pass

        filtered = debugger.get_operations(operation_filter="connect")
        assert len(filtered) == 1
        assert "connect" in filtered[0].operation

    def test_get_performance_report(self):
        """Test performance report generation."""
        debugger = HardwareDebugger()
        debugger.clear()

        with debugger.track_operation("test_op"):
            time.sleep(0.01)

        report = debugger.get_performance_report()
        assert report["total_operations"] >= 1
        assert "test_op" in report["operations"]

    def test_clear(self):
        """Test clearing all data."""
        debugger = HardwareDebugger()
        with debugger.track_operation("test"):
            pass

        debugger.clear()
        assert len(debugger.get_operations()) == 0


class TestDebugHardwareCall:
    """Test debug_hardware_call decorator."""

    def test_decorator_when_disabled(self):
        """Test decorator when debugging disabled."""
        debugger = HardwareDebugger()
        debugger.disable()

        @debug_hardware_call
        def test_function(x):
            return x * 2

        result = test_function(5)
        assert result == 10

    def test_decorator_when_enabled(self):
        """Test decorator when debugging enabled."""
        debugger = HardwareDebugger()
        debugger.enable()
        debugger.clear()

        @debug_hardware_call
        def test_function(x):
            return x * 2

        result = test_function(5)
        assert result == 10

        ops = debugger.get_operations()
        assert len(ops) > 0


class TestDebugMode:
    """Test debug_mode context manager."""

    def test_context_manager(self):
        """Test debug mode context manager."""
        debugger = HardwareDebugger()
        debugger.disable()

        with debug_mode(DebugLevel.TRACE) as d:
            assert d.enabled is True
            assert d.level == DebugLevel.TRACE

        # Should restore previous state
        assert debugger.enabled is False

    def test_context_manager_preserves_previous(self):
        """Test context manager preserves previous state."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.MINIMAL)

        with debug_mode(DebugLevel.VERBOSE):
            pass

        # Should restore previous enabled state
        assert debugger.enabled is True
        assert debugger.level == DebugLevel.MINIMAL


class TestGetDiagnosticReport:
    """Test get_diagnostic_report function."""

    def test_report_generation(self):
        """Test generating diagnostic report."""
        report = get_diagnostic_report()
        assert isinstance(report, DiagnosticReport)
        assert "platform" in report.system_info
        assert "python_version" in report.system_info


class TestSaveDebugSession:
    """Test save_debug_session function."""

    def test_save_session(self, tmp_path):
        """Test saving debug session."""
        debugger = HardwareDebugger()
        debugger.enable()
        with debugger.track_operation("test"):
            pass

        output_path = tmp_path / "debug_session"
        result_path = save_debug_session(output_path)

        assert result_path.exists()
        assert result_path.suffix == ".json"

        with open(result_path) as f:
            data = json.load(f)
        assert "session_id" in data
        assert "diagnostic_report" in data


# =============================================================================
# ADVANCED FEATURES TESTS
# =============================================================================


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced features not available")
class TestAlternativeProcessSimulator:
    """Test AlternativeProcessSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = AlternativeProcessSimulator()
        assert len(sim._process_presets) > 0

    def test_simulate_cyanotype(self):
        """Test cyanotype simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=128)
        result = sim.simulate_cyanotype(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_simulate_vandyke(self):
        """Test Van Dyke brown simulation."""
        sim = AlternativeProcessSimulator()
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = sim.simulate_vandyke(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_simulate_kallitype(self):
        """Test kallitype simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=200)
        result = sim.simulate_kallitype(img)

        assert isinstance(result, Image.Image)

    def test_simulate_gum_bichromate(self):
        """Test gum bichromate simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=150)
        result = sim.simulate_gum_bichromate(img, pigment_color=(100, 50, 30))

        assert isinstance(result, Image.Image)

    def test_simulate_salt_print(self):
        """Test salt print simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=100)
        result = sim.simulate_salt_print(img)

        assert isinstance(result, Image.Image)

    def test_simulate_silver_gelatin(self):
        """Test silver gelatin simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=128)
        result = sim.simulate_silver_gelatin(img, tone="warm")

        assert isinstance(result, Image.Image)

    def test_simulate_argyrotype(self):
        """Test argyrotype simulation."""
        sim = AlternativeProcessSimulator()
        img = Image.new("L", (100, 100), color=175)
        result = sim.simulate_argyrotype(img)

        assert isinstance(result, Image.Image)

    def test_custom_params(self):
        """Test simulation with custom parameters."""
        sim = AlternativeProcessSimulator()
        params = AlternativeProcessParams(
            gamma=1.5, contrast=1.2, shadow_color=(10, 10, 10), dmax=2.0
        )
        img = Image.new("L", (100, 100), color=128)
        result = sim.simulate_cyanotype(img, params=params)

        assert isinstance(result, Image.Image)


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced features not available")
class TestNegativeBlender:
    """Test NegativeBlender class."""

    def test_blend_two_negatives(self):
        """Test blending two negatives."""
        blender = NegativeBlender()
        neg1 = Image.new("L", (100, 100), color=100)
        neg2 = Image.new("L", (100, 100), color=150)

        result = blender.blend_negatives([neg1, neg2])
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_blend_with_masks(self):
        """Test blending with masks."""
        blender = NegativeBlender()
        neg1 = Image.new("L", (100, 100), color=100)
        neg2 = Image.new("L", (100, 100), color=150)
        mask = Image.new("L", (100, 100), color=128)

        result = blender.blend_negatives([neg1, neg2], masks=[None, mask])
        assert isinstance(result, Image.Image)

    def test_blend_modes(self):
        """Test different blend modes."""
        blender = NegativeBlender()
        neg1 = np.full((100, 100), 0.5, dtype=np.float32)
        neg2 = np.full((100, 100), 0.3, dtype=np.float32)

        modes = [BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN]
        result = blender.blend_negatives([neg1, neg2], blend_modes=modes)
        assert isinstance(result, Image.Image)

    def test_create_contrast_mask(self):
        """Test creating contrast mask."""
        blender = NegativeBlender()
        img = Image.new("L", (100, 100), color=128)
        mask = blender.create_contrast_mask(img, threshold=0.5)

        assert isinstance(mask, Image.Image)
        assert mask.mode == "L"

    def test_create_highlight_mask(self):
        """Test creating highlight mask."""
        blender = NegativeBlender()
        img = Image.new("L", (100, 100), color=200)
        mask = blender.create_highlight_mask(img, threshold=0.7)

        assert isinstance(mask, Image.Image)

    def test_create_shadow_mask(self):
        """Test creating shadow mask."""
        blender = NegativeBlender()
        img = Image.new("L", (100, 100), color=50)
        mask = blender.create_shadow_mask(img, threshold=0.3)

        assert isinstance(mask, Image.Image)

    def test_apply_dodge_burn(self):
        """Test dodge and burn application."""
        blender = NegativeBlender()
        img = Image.new("L", (100, 100), color=128)
        dodge_mask = Image.new("L", (100, 100), color=255)
        burn_mask = Image.new("L", (100, 100), color=255)

        result = blender.apply_dodge_burn(
            img, dodge_mask=dodge_mask, burn_mask=burn_mask, dodge_amount=0.3, burn_amount=0.3
        )
        assert isinstance(result, Image.Image)

    def test_create_multi_layer_mask(self):
        """Test creating multi-layer mask."""
        blender = NegativeBlender()
        layer1 = Image.new("L", (100, 100), color=128)
        layer2 = Image.new("L", (100, 100), color=200)

        result = blender.create_multi_layer_mask([layer1, layer2], blend_modes=["multiply"])
        assert isinstance(result, Image.Image)


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced features not available")
class TestQRMetadataGenerator:
    """Test QRMetadataGenerator class."""

    @pytest.fixture(autouse=True)
    def _skip_without_qrcode(self):
        pytest.importorskip("qrcode")

    def test_initialization(self):
        """Test QR generator initialization."""
        gen = QRMetadataGenerator()
        assert gen is not None

    def test_generate_print_qr(self):
        """Test generating QR code."""
        gen = QRMetadataGenerator()
        metadata = PrintMetadata(
            title="Test Print", artist="Test Artist", date="2024-01-01", paper="Hahnemuhle"
        )
        qr = gen.generate_print_qr(metadata, size=200)

        assert isinstance(qr, Image.Image)
        assert qr.size == (200, 200)

    def test_encode_recipe(self):
        """Test encoding recipe."""
        gen = QRMetadataGenerator()
        metadata = PrintMetadata(title="Test", artist="Artist", dmax=1.8)
        encoded = gen.encode_recipe(metadata)

        assert isinstance(encoded, str)
        assert "title:Test" in encoded
        assert "artist:Artist" in encoded

    def test_encode_recipe_dict(self):
        """Test encoding dictionary."""
        gen = QRMetadataGenerator()
        data = {"title": "Test", "artist": "Artist"}
        encoded = gen.encode_recipe(data)

        assert isinstance(encoded, str)
        assert "title:Test" in encoded

    def test_create_archival_label(self):
        """Test creating archival label."""
        gen = QRMetadataGenerator()
        metadata = PrintMetadata(
            title="Test Print",
            artist="Test Artist",
            date="2024-01-01",
            paper="Platine",
            chemistry="Pt/Pd 1:1",
        )
        label = gen.create_archival_label(metadata, label_size=(600, 300), qr_size=180)

        assert isinstance(label, Image.Image)
        assert label.size == (600, 300)
        assert label.mode == "RGB"


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced features not available")
class TestStyleTransfer:
    """Test StyleTransfer class."""

    def test_initialization(self):
        """Test style transfer initialization."""
        st = StyleTransfer()
        assert len(st.styles) > 0

    def test_load_historic_styles(self):
        """Test loading historic styles."""
        st = StyleTransfer()
        styles = st.load_historic_styles()

        assert HistoricStyle.EDWARD_WESTON in styles
        assert HistoricStyle.IRVING_PENN in styles

    def test_analyze_style(self):
        """Test analyzing style from image."""
        st = StyleTransfer()
        img = Image.new("L", (100, 100), color=128)
        params = st.analyze_style(img)

        assert isinstance(params, StyleParameters)
        assert params.gamma > 0
        assert params.contrast > 0

    def test_apply_style(self):
        """Test applying named style."""
        st = StyleTransfer()
        img = Image.new("L", (100, 100), color=128)
        result = st.apply_style(img, HistoricStyle.EDWARD_WESTON)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_create_custom_style(self):
        """Test creating custom style."""
        st = StyleTransfer()
        params = st.create_custom_style(
            "My Style", {"gamma": 1.2, "contrast": 1.1, "description": "Custom test style"}
        )

        assert isinstance(params, StyleParameters)
        assert params.name == "My Style"
        assert params.gamma == 1.2


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced features not available")
class TestPrintComparison:
    """Test PrintComparison class."""

    def test_compare_before_after(self):
        """Test comparing two images."""
        comp = PrintComparison()
        img1 = Image.new("L", (100, 100), color=128)
        img2 = Image.new("L", (100, 100), color=130)

        result = comp.compare_before_after(img1, img2)

        assert "rmse" in result
        assert "psnr" in result
        assert "similarity_score" in result
        assert "histogram_correlation" in result

    def test_generate_difference_map(self):
        """Test generating difference map."""
        comp = PrintComparison()
        img1 = Image.new("L", (100, 100), color=100)
        img2 = Image.new("L", (100, 100), color=150)

        diff = comp.generate_difference_map(img1, img2, colorize=True)
        assert isinstance(diff, Image.Image)
        assert diff.mode == "RGB"

        diff_gray = comp.generate_difference_map(img1, img2, colorize=False)
        assert diff_gray.mode == "L"

    def test_calculate_similarity_score(self):
        """Test calculating similarity scores."""
        comp = PrintComparison()
        # Use gradient images to avoid NaN from zero-variance in correlation
        arr1 = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
        img1 = Image.fromarray(arr1, mode="L")
        img2 = Image.fromarray(arr1, mode="L")

        # Identical images should have high similarity
        score_mse = comp.calculate_similarity_score(img1, img2, method="mse")
        assert score_mse > 0.9

        score_corr = comp.calculate_similarity_score(img1, img2, method="correlation")
        assert score_corr > 0.9

    def test_generate_comparison_report(self):
        """Test generating comparison report."""
        comp = PrintComparison()
        images = {
            "original": Image.new("L", (100, 100), color=128),
            "print1": Image.new("L", (100, 100), color=130),
            "print2": Image.new("L", (100, 100), color=125),
        }

        report = comp.generate_comparison_report(images, reference_key="original")

        assert "reference" in report
        assert report["reference"] == "original"
        assert "comparisons" in report
        assert "print1" in report["comparisons"]
        assert "print2" in report["comparisons"]
        assert "summary" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
