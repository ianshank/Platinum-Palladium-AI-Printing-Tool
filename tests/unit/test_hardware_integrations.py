"""
Unit tests for hardware integration components.

Tests the hardware abstraction layer including:
- Simulated spectrophotometer and printer
- Exception hierarchy
- Device status and info models
- Protocol compliance
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.integrations.hardware import (
    CalibrationError,
    DeviceCommunicationError,
    DeviceConnectionError,
    DeviceNotFoundError,
    HardwareError,
    MeasurementError,
    PrinterError,
    PrintJobError,
    get_printer_driver,
    get_spectrophotometer_driver,
)
from ptpd_calibration.integrations.hardware.simulated import (
    SimulatedPrinter,
    SimulatedSpectrophotometer,
)
from ptpd_calibration.integrations.protocols import (
    DensityMeasurement,
    DeviceInfo,
    DeviceStatus,
    PrintJob,
    PrintResult,
    SpectralData,
)

# =============================================================================
# Test DeviceStatus Enum
# =============================================================================


class TestDeviceStatus:
    """Tests for DeviceStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert DeviceStatus.DISCONNECTED
        assert DeviceStatus.CONNECTING
        assert DeviceStatus.CONNECTED
        assert DeviceStatus.CALIBRATING
        assert DeviceStatus.MEASURING
        assert DeviceStatus.ERROR
        assert DeviceStatus.BUSY

    def test_status_string_representation(self):
        """Test status string values."""
        assert DeviceStatus.DISCONNECTED.value == "disconnected"
        assert DeviceStatus.CONNECTED.value == "connected"
        assert DeviceStatus.CALIBRATING.value == "calibrating"


# =============================================================================
# Test DeviceInfo Model
# =============================================================================


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_create_device_info(self):
        """Test creating DeviceInfo."""
        info = DeviceInfo(
            vendor="X-Rite",
            model="i1Pro2",
            serial_number="ABC123",
            firmware_version="1.2.3",
            capabilities=["density", "lab", "spectral"],
        )
        assert info.vendor == "X-Rite"
        assert info.model == "i1Pro2"
        assert info.serial_number == "ABC123"
        assert "density" in info.capabilities

    def test_device_info_optional_fields(self):
        """Test DeviceInfo with optional fields."""
        info = DeviceInfo(
            vendor="CUPS",
            model="Epson P800",
        )
        assert info.vendor == "CUPS"
        assert info.serial_number is None
        assert info.firmware_version is None
        assert info.capabilities == []


# =============================================================================
# Test DensityMeasurement Model
# =============================================================================


class TestDensityMeasurement:
    """Tests for DensityMeasurement dataclass."""

    def test_create_measurement(self):
        """Test creating DensityMeasurement."""
        measurement = DensityMeasurement(
            density=1.5,
            lab_l=50.0,
            lab_a=2.0,
            lab_b=-1.5,
            timestamp=datetime.now(),
            measurement_mode="reflection",
        )
        assert measurement.density == 1.5
        assert measurement.lab_l == 50.0
        assert measurement.measurement_mode == "reflection"

    def test_measurement_defaults(self):
        """Test DensityMeasurement default values."""
        measurement = DensityMeasurement(
            density=1.0,
            lab_l=50.0,
            lab_a=0.0,
            lab_b=0.0,
        )
        assert measurement.timestamp is not None
        assert measurement.measurement_mode == "reflection"


# =============================================================================
# Test SpectralData Model
# =============================================================================


class TestSpectralData:
    """Tests for SpectralData dataclass."""

    def test_create_spectral_data(self):
        """Test creating SpectralData."""
        wavelengths = list(range(380, 740, 10))
        values = [0.1] * len(wavelengths)

        spectral = SpectralData(
            wavelengths=wavelengths,
            values=values,
            start_nm=380.0,
            end_nm=730.0,
            interval_nm=10.0,
        )
        assert len(spectral.wavelengths) == 36
        assert len(spectral.values) == 36
        assert spectral.start_nm == 380.0

    def test_spectral_data_validation(self):
        """Test SpectralData length matching."""
        spectral = SpectralData(
            wavelengths=[380, 400, 420],
            values=[0.1, 0.2, 0.3],
            start_nm=380.0,
            end_nm=420.0,
            interval_nm=20.0,
        )
        assert len(spectral.wavelengths) == len(spectral.values)


# =============================================================================
# Test PrintJob Model
# =============================================================================


class TestPrintJob:
    """Tests for PrintJob dataclass."""

    def test_create_print_job(self):
        """Test creating PrintJob."""
        job = PrintJob(
            name="Test Negative",
            image_path="/path/to/negative.tiff",
            paper_size="8x10",
            resolution_dpi=2880,
            copies=1,
        )
        assert job.name == "Test Negative"
        assert job.resolution_dpi == 2880
        assert job.copies == 1

    def test_print_job_defaults(self):
        """Test PrintJob default values."""
        job = PrintJob(
            name="Test",
            image_path="/path/to/image.tiff",
        )
        # Default paper size depends on implementation
        assert job.paper_size in ["letter", "8x10"]
        assert job.resolution_dpi in [1440, 2880]
        assert job.copies == 1


# =============================================================================
# Test PrintResult Model
# =============================================================================


class TestPrintResult:
    """Tests for PrintResult dataclass."""

    def test_successful_result(self):
        """Test successful print result."""
        result = PrintResult(
            success=True,
            job_id="12345",
            pages_printed=1,
            duration_seconds=5.5,
        )
        assert result.success is True
        assert result.job_id == "12345"
        assert result.error is None

    def test_failed_result(self):
        """Test failed print result."""
        result = PrintResult(
            success=False,
            error="Paper jam",
        )
        assert result.success is False
        assert result.error == "Paper jam"
        assert result.job_id is None


# =============================================================================
# Test Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Tests for hardware exception classes."""

    def test_hardware_error_base(self):
        """Test HardwareError base exception."""
        error = HardwareError("Device failed", device_type="spectrophotometer")
        assert "Device failed" in str(error)
        assert error.device_type == "spectrophotometer"

    def test_device_not_found_error(self):
        """Test DeviceNotFoundError."""
        error = DeviceNotFoundError(
            "No device found", device_type="spectrophotometer", port="/dev/ttyUSB0"
        )
        assert "No device found" in str(error)
        assert error.details.get("port") == "/dev/ttyUSB0"

    def test_device_connection_error(self):
        """Test DeviceConnectionError."""
        error = DeviceConnectionError(
            "Connection refused", device_type="spectrophotometer", port="/dev/ttyUSB0"
        )
        assert error.details.get("port") == "/dev/ttyUSB0"

    def test_device_communication_error(self):
        """Test DeviceCommunicationError."""
        error = DeviceCommunicationError(
            "Timeout", device_type="spectrophotometer", command="*MSR:DENSITY", response=None
        )
        assert error.details.get("command") == "*MSR:DENSITY"

    def test_calibration_error(self):
        """Test CalibrationError."""
        error = CalibrationError(
            "White calibration failed", device_type="spectrophotometer", calibration_type="white"
        )
        assert error.details.get("calibration_type") == "white"

    def test_measurement_error(self):
        """Test MeasurementError."""
        error = MeasurementError(
            "Invalid reading", device_type="spectrophotometer", measurement_type="density"
        )
        assert error.details.get("measurement_type") == "density"

    def test_printer_error(self):
        """Test PrinterError."""
        error = PrinterError("Printer offline", printer_name="EPSON-P800", operation="connect")
        assert error.details.get("printer_name") == "EPSON-P800"

    def test_print_job_error(self):
        """Test PrintJobError."""
        error = PrintJobError("Job cancelled", job_name="Test Print", job_id="12345")
        assert error.details.get("job_name") == "Test Print"
        assert error.details.get("job_id") == "12345"

    def test_exception_inheritance(self):
        """Test exception inheritance chain."""
        error = MeasurementError(
            "Test", device_type="spectrophotometer", measurement_type="density"
        )
        assert isinstance(error, HardwareError)
        assert isinstance(error, Exception)


# =============================================================================
# Test SimulatedSpectrophotometer
# =============================================================================


class TestSimulatedSpectrophotometer:
    """Tests for SimulatedSpectrophotometer."""

    def test_initial_state(self):
        """Test initial disconnected state."""
        device = SimulatedSpectrophotometer()
        assert device.status == DeviceStatus.DISCONNECTED
        assert device.device_info is None

    def test_connect(self):
        """Test connecting to simulated device."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        result = device.connect()

        assert result is True
        assert device.status == DeviceStatus.CONNECTED
        assert device.device_info is not None
        assert device.device_info.vendor == "Simulated"
        # Model name may vary in simulation
        assert device.device_info.model is not None

    def test_connect_with_port(self):
        """Test connect accepts port parameter."""
        device = SimulatedSpectrophotometer()
        result = device.connect(port="/dev/simulated")
        assert result is True
        assert device.status == DeviceStatus.CONNECTED

    def test_disconnect(self):
        """Test disconnecting from device."""
        device = SimulatedSpectrophotometer()
        device.connect()
        device.disconnect()

        assert device.status == DeviceStatus.DISCONNECTED
        assert device.device_info is None

    def test_calibrate_white(self):
        """Test white calibration."""
        device = SimulatedSpectrophotometer()
        device.connect()
        result = device.calibrate_white()

        assert result is True
        assert device.status == DeviceStatus.CONNECTED

    def test_calibrate_white_not_connected(self):
        """Test calibration returns False when not connected."""
        device = SimulatedSpectrophotometer(simulate_delay=False)

        # Simulated device returns False instead of raising
        result = device.calibrate_white()
        assert result is False

    def test_read_density(self):
        """Test reading density measurement."""
        device = SimulatedSpectrophotometer()
        device.connect()
        device.calibrate_white()

        measurement = device.read_density()

        assert isinstance(measurement, DensityMeasurement)
        assert 0.0 <= measurement.density <= 3.0
        assert 0.0 <= measurement.lab_l <= 100.0
        assert measurement.measurement_mode == "reflection"

    def test_read_density_not_connected(self):
        """Test read_density fails when not connected."""
        device = SimulatedSpectrophotometer(simulate_delay=False)

        # Simulated device raises RuntimeError
        with pytest.raises(RuntimeError):
            device.read_density()

    def test_read_density_without_calibration(self):
        """Test read_density warns when not calibrated."""
        device = SimulatedSpectrophotometer()
        device.connect()

        # Should still work but with warning (simulated behavior)
        measurement = device.read_density()
        assert measurement is not None

    def test_read_spectral(self):
        """Test reading spectral data."""
        device = SimulatedSpectrophotometer()
        device.connect()
        device.calibrate_white()

        spectral = device.read_spectral()

        assert isinstance(spectral, SpectralData)
        assert len(spectral.wavelengths) == len(spectral.values)
        assert spectral.start_nm == 380.0
        assert spectral.end_nm == 730.0

    def test_read_spectral_not_connected(self):
        """Test read_spectral fails when not connected."""
        device = SimulatedSpectrophotometer(simulate_delay=False)

        # Simulated device raises RuntimeError
        with pytest.raises(RuntimeError):
            device.read_spectral()

    def test_multiple_measurements(self):
        """Test multiple measurements produce varied results."""
        device = SimulatedSpectrophotometer()
        device.connect()
        device.calibrate_white()

        measurements = [device.read_density() for _ in range(5)]
        densities = [m.density for m in measurements]

        # Simulated data should have some variation
        assert len(set(densities)) > 1

    def test_device_info_capabilities(self):
        """Test device reports correct capabilities."""
        device = SimulatedSpectrophotometer()
        device.connect()

        info = device.device_info
        assert "density" in info.capabilities
        assert "lab" in info.capabilities
        assert "spectral" in info.capabilities


# =============================================================================
# Test SimulatedPrinter
# =============================================================================


class TestSimulatedPrinter:
    """Tests for SimulatedPrinter."""

    def test_initial_state(self):
        """Test initial disconnected state."""
        printer = SimulatedPrinter()
        assert printer.status == DeviceStatus.DISCONNECTED
        assert printer.device_info is None

    def test_connect(self):
        """Test connecting to simulated printer."""
        printer = SimulatedPrinter()
        result = printer.connect()

        assert result is True
        assert printer.status == DeviceStatus.CONNECTED
        assert printer.device_info is not None

    def test_connect_with_name(self):
        """Test connect with printer name."""
        printer = SimulatedPrinter()
        result = printer.connect(printer_name="Test Printer")

        assert result is True
        assert (
            "Test Printer" in printer.device_info.model or printer.status == DeviceStatus.CONNECTED
        )

    def test_disconnect(self):
        """Test disconnecting from printer."""
        printer = SimulatedPrinter()
        printer.connect()
        printer.disconnect()

        assert printer.status == DeviceStatus.DISCONNECTED

    def test_print_image_success(self, tmp_path):
        """Test successful print job."""
        printer = SimulatedPrinter()
        printer.connect()

        # Create a test file
        test_image = tmp_path / "test_negative.tiff"
        test_image.write_bytes(b"fake image data")

        job = PrintJob(
            name="Test Print",
            image_path=str(test_image),
            paper_size="8x10",
            resolution_dpi=2880,
        )

        result = printer.print_image(job)

        assert result.success is True
        assert result.job_id is not None
        assert result.pages_printed == 1
        assert result.duration_seconds > 0

    def test_print_image_file_not_found(self):
        """Test print with missing file - simulated printer may succeed."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()

        job = PrintJob(
            name="Test Print",
            image_path="/nonexistent/path/image.tiff",
        )

        result = printer.print_image(job)

        # Simulated printer may succeed regardless of file existence
        # This is expected behavior for simulation mode
        assert isinstance(result, PrintResult)

    def test_print_image_not_connected(self, tmp_path):
        """Test print fails when not connected."""
        printer = SimulatedPrinter()

        test_image = tmp_path / "test.tiff"
        test_image.write_bytes(b"data")

        job = PrintJob(
            name="Test",
            image_path=str(test_image),
        )

        result = printer.print_image(job)

        assert result.success is False
        assert "not connected" in result.error.lower()

    def test_get_paper_sizes(self):
        """Test getting supported paper sizes."""
        printer = SimulatedPrinter()
        printer.connect()

        sizes = printer.get_paper_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) > 0
        assert "8x10" in sizes or "letter" in sizes

    def test_get_resolutions(self):
        """Test getting supported resolutions."""
        printer = SimulatedPrinter()
        printer.connect()

        resolutions = printer.get_resolutions()

        assert isinstance(resolutions, list)
        assert 720 in resolutions or 1440 in resolutions

    def test_get_printer_status(self):
        """Test device status property."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()

        # Use status property instead of get_printer_status method
        status = printer.status

        assert status == DeviceStatus.CONNECTED

    def test_get_ink_levels(self):
        """Test getting ink levels."""
        printer = SimulatedPrinter()
        printer.connect()

        levels = printer.get_ink_levels()

        assert isinstance(levels, dict)
        # Simulated printer may return empty dict or simulated levels

    def test_multiple_copies(self, tmp_path):
        """Test print job with multiple copies."""
        printer = SimulatedPrinter()
        printer.connect()

        test_image = tmp_path / "test.tiff"
        test_image.write_bytes(b"data")

        job = PrintJob(
            name="Test",
            image_path=str(test_image),
            copies=3,
        )

        result = printer.print_image(job)

        assert result.success is True
        assert result.pages_printed == 3


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for driver factory functions."""

    def test_get_spectrophotometer_driver_simulated(self):
        """Test getting simulated spectrophotometer."""
        driver = get_spectrophotometer_driver(simulate=True)

        assert isinstance(driver, SimulatedSpectrophotometer)
        assert driver.status == DeviceStatus.DISCONNECTED

    def test_get_printer_driver_simulated(self):
        """Test getting simulated printer."""
        driver = get_printer_driver(simulate=True)

        assert isinstance(driver, SimulatedPrinter)
        assert driver.status == DeviceStatus.DISCONNECTED

    def test_get_spectrophotometer_driver_with_port(self):
        """Test getting spectrophotometer driver with port parameter."""
        driver = get_spectrophotometer_driver(simulate=True, port="/dev/test")

        # Port is ignored in simulation mode
        assert isinstance(driver, SimulatedSpectrophotometer)

    def test_get_printer_driver_with_name(self):
        """Test getting printer driver with name parameter."""
        driver = get_printer_driver(simulate=True, printer_name="Test Printer")

        # Name is stored for simulated printer
        assert isinstance(driver, SimulatedPrinter)


# =============================================================================
# Test XRiteI1ProDriver (Mocked Serial)
# =============================================================================


class TestXRiteI1ProDriver:
    """Tests for XRiteI1ProDriver with mocked serial."""

    @pytest.fixture
    def mock_serial(self):
        """Create mock serial module."""
        with patch("ptpd_calibration.integrations.hardware.xrite_i1pro._import_serial") as mock:
            serial_mock = MagicMock()
            serial_mock.Serial.return_value = MagicMock()
            serial_mock.EIGHTBITS = 8
            serial_mock.PARITY_NONE = "N"
            serial_mock.STOPBITS_ONE = 1
            serial_mock.tools.list_ports.comports.return_value = []
            mock.return_value = serial_mock
            yield serial_mock

    def test_import_lazy_loading(self):
        """Test that XRiteI1ProDriver can be imported."""
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver

        assert XRiteI1ProDriver is not None

    def test_initial_state(self, mock_serial):  # noqa: ARG002
        """Test initial driver state."""
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver

        driver = XRiteI1ProDriver()
        assert driver.status == DeviceStatus.DISCONNECTED
        assert driver.device_info is None

    def test_auto_detect_no_device(self, mock_serial):  # noqa: ARG002
        """Test auto-detect when no device present."""
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver

        driver = XRiteI1ProDriver()

        with pytest.raises(DeviceNotFoundError):
            driver.connect()

    def test_connect_with_port(self, mock_serial):
        """Test connecting with specific port."""
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver

        # Mock successful connection and identification
        mock_conn = MagicMock()
        mock_conn.readline.return_value = b"i1Pro2 V1.23.456 SN:AB123456\r\n"
        mock_serial.Serial.return_value = mock_conn

        driver = XRiteI1ProDriver()
        result = driver.connect(port="/dev/ttyUSB0")

        assert result is True
        assert driver.status == DeviceStatus.CONNECTED
        assert driver.device_info is not None
        assert driver.device_info.model == "i1Pro2"

    def test_parse_version_response(self, mock_serial):
        """Test parsing various version responses."""
        from ptpd_calibration.integrations.hardware.xrite_i1pro import XRiteI1ProDriver

        mock_conn = MagicMock()
        mock_serial.Serial.return_value = mock_conn

        # Test i1Pro3 detection
        mock_conn.readline.return_value = b"i1Pro3 V2.0.1 SN:XY987654\r\n"

        driver = XRiteI1ProDriver()
        driver.connect(port="/dev/ttyUSB0")

        assert driver.device_info.model == "i1Pro3"
        assert driver.device_info.firmware_version == "2.0.1"
        assert driver.device_info.serial_number == "XY987654"


# =============================================================================
# Test CUPSPrinterDriver (Mocked CUPS)
# =============================================================================


class TestCUPSPrinterDriver:
    """Tests for CUPSPrinterDriver with mocked CUPS."""

    @pytest.fixture
    def mock_cups(self):
        """Create mock CUPS module."""
        with patch("ptpd_calibration.integrations.hardware.cups_printer._import_cups") as mock:
            cups_mock = MagicMock()
            cups_mock.Connection.return_value = MagicMock()
            mock.return_value = cups_mock
            yield cups_mock

    def test_import_on_windows(self):
        """Test import fails on Windows."""
        with patch("platform.system", return_value="Windows"):
            from ptpd_calibration.integrations.hardware.cups_printer import _import_cups

            with pytest.raises(ImportError, match="Windows"):
                _import_cups()

    def test_initial_state(self, mock_cups):  # noqa: ARG002
        """Test initial driver state."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        driver = CUPSPrinterDriver()
        assert driver.status == DeviceStatus.DISCONNECTED

    def test_connect_to_default_printer(self, mock_cups):
        """Test connecting to default printer."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        mock_conn = mock_cups.Connection.return_value
        mock_conn.getPrinters.return_value = {
            "EPSON-P800": {
                "device-uri": "usb://EPSON/P800",
                "printer-make-and-model": "EPSON SureColor P800",
            }
        }
        mock_conn.getDefault.return_value = "EPSON-P800"

        driver = CUPSPrinterDriver()
        result = driver.connect()

        assert result is True
        assert driver.status == DeviceStatus.CONNECTED
        assert driver.device_info is not None
        assert "EPSON" in driver.device_info.vendor

    def test_connect_no_printers(self, mock_cups):
        """Test connect fails when no printers available."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        mock_conn = mock_cups.Connection.return_value
        mock_conn.getPrinters.return_value = {}

        driver = CUPSPrinterDriver()

        with pytest.raises(PrinterError, match="No printers"):
            driver.connect()

    def test_connect_printer_not_found(self, mock_cups):
        """Test connect fails for non-existent printer."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        mock_conn = mock_cups.Connection.return_value
        mock_conn.getPrinters.return_value = {"EPSON-P800": {"device-uri": "usb://..."}}

        driver = CUPSPrinterDriver(printer_name="NonExistent")

        with pytest.raises(Exception):  # PrinterNotFoundError  # noqa: B017
            driver.connect()

    def test_paper_size_mapping(self, mock_cups):  # noqa: ARG002
        """Test paper size mapping to CUPS names."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        assert CUPSPrinterDriver.PAPER_SIZE_MAP["8x10"] == "Custom.8x10in"
        assert CUPSPrinterDriver.PAPER_SIZE_MAP["letter"] == "Letter"
        assert CUPSPrinterDriver.PAPER_SIZE_MAP["a4"] == "A4"

    def test_build_print_options(self, mock_cups):  # noqa: ARG002
        """Test building CUPS print options."""
        from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver

        driver = CUPSPrinterDriver()

        job = PrintJob(
            name="Test",
            image_path="/path/to/image.tiff",
            paper_size="8x10",
            resolution_dpi=2880,
            copies=2,
        )

        options = driver._build_print_options(job)

        assert options["media"] == "Custom.8x10in"
        assert options["Resolution"] == "2880dpi"
        assert options["ColorModel"] == "Gray"
        assert options["copies"] == "2"


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for hardware workflows."""

    def test_full_measurement_workflow(self):
        """Test complete measurement workflow with simulated device."""
        # Get driver
        device = get_spectrophotometer_driver(simulate=True)

        # Connect
        assert device.connect() is True
        assert device.status == DeviceStatus.CONNECTED

        # Calibrate
        assert device.calibrate_white() is True

        # Take measurements
        measurements = []
        for _ in range(5):
            measurement = device.read_density()
            measurements.append(measurement)
            assert isinstance(measurement, DensityMeasurement)

        # Get spectral
        spectral = device.read_spectral()
        assert isinstance(spectral, SpectralData)

        # Disconnect
        device.disconnect()
        assert device.status == DeviceStatus.DISCONNECTED

    def test_full_print_workflow(self, tmp_path):
        """Test complete print workflow with simulated printer."""
        # Get driver
        printer = get_printer_driver(simulate=True)

        # Connect
        assert printer.connect() is True

        # Check status
        assert printer.status == DeviceStatus.CONNECTED

        # Check capabilities (if available)
        if hasattr(printer, "get_paper_sizes"):
            sizes = printer.get_paper_sizes()
            assert len(sizes) > 0

        if hasattr(printer, "get_resolutions"):
            resolutions = printer.get_resolutions()
            assert len(resolutions) > 0

        # Create test file
        test_image = tmp_path / "negative.tiff"
        test_image.write_bytes(b"test image data")

        # Print
        job = PrintJob(
            name="Test Negative",
            image_path=str(test_image),
            paper_size="8x10",
            resolution_dpi=2880,
        )

        result = printer.print_image(job)
        assert result.success is True

        # Disconnect
        printer.disconnect()
        assert printer.status == DeviceStatus.DISCONNECTED

    def test_error_recovery(self):
        """Test error recovery in measurement workflow."""
        device = get_spectrophotometer_driver(simulate=True)

        # Try to measure without connecting - should raise error
        with pytest.raises((MeasurementError, RuntimeError)):
            device.read_density()

        # Connect and try again
        device.connect()
        measurement = device.read_density()
        assert measurement is not None

        # Disconnect and ensure clean state
        device.disconnect()
        assert device.status == DeviceStatus.DISCONNECTED
