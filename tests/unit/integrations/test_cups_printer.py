"""Unit tests for CUPS printer driver.

Tests verify the CUPSPrinterDriver functionality with mocked CUPS connection.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.integrations.hardware.cups_printer import CUPSPrinterDriver
from ptpd_calibration.integrations.hardware.exceptions import (
    PrinterError,
    PrinterNotFoundError,
    PrintJobError,
)
from ptpd_calibration.integrations.protocols import DeviceStatus, PrintJob


@pytest.fixture
def mock_cups_module():
    """Mock the pycups module."""
    with patch(
        "ptpd_calibration.integrations.hardware.cups_printer._import_cups"
    ) as mock:
        cups_mock = MagicMock()
        cups_mock.Connection = MagicMock()
        mock.return_value = cups_mock
        yield cups_mock


@pytest.fixture
def mock_cups_connection():
    """Create a mocked CUPS connection object."""
    conn = MagicMock()
    conn.getPrinters.return_value = {
        "TestPrinter": {
            "device-uri": "usb://test",
            "printer-make-and-model": "Epson SureColor P800",
            "color-supported": True,
            "sides-supported": ["one-sided", "two-sided"],
        },
        "SecondPrinter": {
            "device-uri": "usb://test2",
            "printer-make-and-model": "Canon Pro-1000",
            "color-supported": True,
        },
    }
    conn.getDefault.return_value = "TestPrinter"
    conn.getPrinterAttributes.return_value = {
        "printer-state": 3,
        "printer-state-message": "Ready",
        "printer-is-accepting-jobs": True,
        "queued-job-count": 0,
        "printer-supply": [
            "type=ink;name=black;level=75",
            "type=ink;name=cyan;level=50",
            "type=ink;name=magenta;level=20",
        ],
    }
    conn.printFile.return_value = 12345
    conn.getPPD.return_value = None
    return conn


@pytest.fixture
def driver(mock_cups_module):  # noqa: ARG001
    """Create a CUPSPrinterDriver instance."""
    return CUPSPrinterDriver()


@pytest.fixture
def connected_driver(driver, mock_cups_module, mock_cups_connection):
    """Create a connected CUPSPrinterDriver instance."""
    mock_cups_module.Connection.return_value = mock_cups_connection
    driver.connect()
    return driver


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary image file."""
    image_file = tmp_path / "test_negative.tiff"
    image_file.write_bytes(b"fake TIFF data")
    return image_file


class TestCUPSPrinterDriverInit:
    """Test CUPSPrinterDriver initialization."""

    def test_initial_status_is_disconnected(self, driver: CUPSPrinterDriver) -> None:
        """Test that initial status is DISCONNECTED."""
        assert driver.status == DeviceStatus.DISCONNECTED

    def test_initial_device_info_is_none(self, driver: CUPSPrinterDriver) -> None:
        """Test that initial device info is None."""
        assert driver.device_info is None


class TestCUPSPrinterDriverConnect:
    """Test CUPSPrinterDriver connect method."""

    def test_connect_success(
        self,
        driver: CUPSPrinterDriver,
        mock_cups_module: MagicMock,
        mock_cups_connection: MagicMock,
    ) -> None:
        """Test successful connection."""
        mock_cups_module.Connection.return_value = mock_cups_connection

        result = driver.connect()

        assert result is True
        assert driver.status == DeviceStatus.CONNECTED
        assert driver.device_info is not None

    def test_connect_with_specific_printer(
        self,
        driver: CUPSPrinterDriver,
        mock_cups_module: MagicMock,
        mock_cups_connection: MagicMock,
    ) -> None:
        """Test connection to specific printer by name."""
        mock_cups_module.Connection.return_value = mock_cups_connection

        result = driver.connect(printer_name="SecondPrinter")

        assert result is True
        assert driver._printer_name == "SecondPrinter"

    def test_connect_printer_not_found(
        self,
        driver: CUPSPrinterDriver,
        mock_cups_module: MagicMock,
        mock_cups_connection: MagicMock,
    ) -> None:
        """Test connection failure when printer not found."""
        mock_cups_module.Connection.return_value = mock_cups_connection

        with pytest.raises(PrinterNotFoundError) as exc_info:
            driver.connect(printer_name="NonExistentPrinter")

        assert "NonExistentPrinter" in str(exc_info.value)

    def test_connect_no_printers_available(
        self,
        driver: CUPSPrinterDriver,
        mock_cups_module: MagicMock,
        mock_cups_connection: MagicMock,
    ) -> None:
        """Test connection failure when no printers available."""
        mock_cups_connection.getPrinters.return_value = {}
        mock_cups_module.Connection.return_value = mock_cups_connection

        with pytest.raises(PrinterError) as exc_info:
            driver.connect()

        assert "No printers available" in str(exc_info.value)

    def test_connect_cups_server_failure(
        self,
        driver: CUPSPrinterDriver,
        mock_cups_module: MagicMock,
    ) -> None:
        """Test connection failure when CUPS server unavailable."""
        mock_cups_module.Connection.side_effect = Exception("CUPS server error")

        with pytest.raises(PrinterError) as exc_info:
            driver.connect()

        assert "CUPS server" in str(exc_info.value)


class TestCUPSPrinterDriverDisconnect:
    """Test CUPSPrinterDriver disconnect method."""

    def test_disconnect_clears_state(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test that disconnect clears all state."""
        connected_driver.disconnect()

        assert connected_driver.status == DeviceStatus.DISCONNECTED
        assert connected_driver.device_info is None


class TestCUPSPrinterDriverPrintImage:
    """Test CUPSPrinterDriver print_image method."""

    def test_print_image_success(
        self,
        connected_driver: CUPSPrinterDriver,
        temp_image_file: Path,
    ) -> None:
        """Test successful print job submission."""
        job = PrintJob(
            name="Test Print",
            image_path=str(temp_image_file),
            paper_size="8x10",
            resolution_dpi=2880,
            copies=1,
        )

        result = connected_driver.print_image(job)

        assert result.success is True
        assert result.job_id == "12345"
        assert result.pages_printed == 1

    def test_print_image_not_connected(
        self,
        driver: CUPSPrinterDriver,
        temp_image_file: Path,
    ) -> None:
        """Test print fails when not connected."""
        job = PrintJob(
            name="Test Print",
            image_path=str(temp_image_file),
            paper_size="8x10",
            resolution_dpi=2880,
        )

        result = driver.print_image(job)

        assert result.success is False
        assert "not connected" in result.error.lower()

    def test_print_image_file_not_found(
        self,
        connected_driver: CUPSPrinterDriver,
    ) -> None:
        """Test print fails when image file not found."""
        job = PrintJob(
            name="Test Print",
            image_path="/nonexistent/path/image.tiff",
            paper_size="8x10",
            resolution_dpi=2880,
        )

        result = connected_driver.print_image(job)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_print_image_multiple_copies(
        self,
        connected_driver: CUPSPrinterDriver,
        temp_image_file: Path,
    ) -> None:
        """Test print with multiple copies."""
        job = PrintJob(
            name="Multi-Copy Print",
            image_path=str(temp_image_file),
            paper_size="letter",
            resolution_dpi=1440,
            copies=3,
        )

        result = connected_driver.print_image(job)

        assert result.success is True
        assert result.pages_printed == 3


class TestCUPSPrinterDriverBuildPrintOptions:
    """Test CUPSPrinterDriver _build_print_options method."""

    def test_build_options_standard_paper_size(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test building options with standard paper size."""
        job = PrintJob(
            name="Test",
            image_path="/tmp/test.tiff",
            paper_size="8x10",
            resolution_dpi=2880,
        )

        options = connected_driver._build_print_options(job)

        assert options["media"] == "Custom.8x10in"
        assert options["Resolution"] == "2880dpi"
        assert options["ColorModel"] == "Gray"

    def test_build_options_custom_paper_size(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test building options with custom paper size."""
        job = PrintJob(
            name="Test",
            image_path="/tmp/test.tiff",
            paper_size="CustomSize123",
            resolution_dpi=1440,
        )

        options = connected_driver._build_print_options(job)

        assert options["media"] == "CustomSize123"

    def test_build_options_invalid_paper_size_rejected(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test that invalid paper size is rejected."""
        job = PrintJob(
            name="Test",
            image_path="/tmp/test.tiff",
            paper_size="8x10; DROP TABLE",
            resolution_dpi=1440,
        )

        with pytest.raises(PrintJobError):
            connected_driver._build_print_options(job)


class TestCUPSPrinterDriverGetPaperSizes:
    """Test CUPSPrinterDriver get_paper_sizes method."""

    def test_get_paper_sizes_returns_defaults_when_disconnected(
        self, driver: CUPSPrinterDriver
    ) -> None:
        """Test that default sizes returned when not connected."""
        sizes = driver.get_paper_sizes()

        assert len(sizes) > 0
        assert "8x10" in sizes

    def test_get_paper_sizes_returns_list(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test that paper sizes returns a list."""
        sizes = connected_driver.get_paper_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) > 0


class TestCUPSPrinterDriverGetResolutions:
    """Test CUPSPrinterDriver get_resolutions method."""

    def test_get_resolutions_returns_standard_values(
        self, driver: CUPSPrinterDriver
    ) -> None:
        """Test that standard resolutions are returned."""
        resolutions = driver.get_resolutions()

        assert 360 in resolutions
        assert 720 in resolutions
        assert 1440 in resolutions
        assert 2880 in resolutions


class TestCUPSPrinterDriverGetPrinterStatus:
    """Test CUPSPrinterDriver get_printer_status method."""

    def test_get_printer_status_disconnected(
        self, driver: CUPSPrinterDriver
    ) -> None:
        """Test status when disconnected."""
        status = driver.get_printer_status()

        assert status["status"] == "disconnected"

    def test_get_printer_status_connected(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test status when connected."""
        status = connected_driver.get_printer_status()

        assert "status" in status
        assert "accepting_jobs" in status


class TestCUPSPrinterDriverGetInkLevels:
    """Test CUPSPrinterDriver get_ink_levels method."""

    def test_get_ink_levels_disconnected(
        self, driver: CUPSPrinterDriver
    ) -> None:
        """Test ink levels when disconnected."""
        levels = driver.get_ink_levels()

        assert levels == {}

    def test_get_ink_levels_connected(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test ink levels when connected."""
        levels = connected_driver.get_ink_levels()

        assert "black" in levels
        assert levels["black"]["level"] == 75
        assert levels["black"]["status"] == "ok"

    def test_get_ink_levels_low_threshold(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test that low ink is detected correctly."""
        levels = connected_driver.get_ink_levels()

        # magenta is at 20%, should be "low"
        assert levels["magenta"]["status"] == "low"


class TestCUPSPrinterDriverParseSupplyString:
    """Test CUPSPrinterDriver _parse_supply_string method."""

    def test_parse_supply_string_valid(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test parsing valid supply string."""
        result = connected_driver._parse_supply_string(
            "type=ink;name=black;level=75"
        )

        assert result is not None
        assert result["name"] == "black"
        assert result["level"] == 75

    def test_parse_supply_string_missing_name(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test parsing supply string without name."""
        result = connected_driver._parse_supply_string("type=ink;level=75")

        assert result is None

    def test_parse_supply_string_missing_level(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test parsing supply string without level."""
        result = connected_driver._parse_supply_string("type=ink;name=black")

        assert result is None

    def test_parse_supply_string_invalid_level(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test parsing supply string with invalid level."""
        result = connected_driver._parse_supply_string(
            "type=ink;name=black;level=invalid"
        )

        assert result is not None
        assert result["level"] == 0

    def test_parse_supply_string_malformed(
        self, connected_driver: CUPSPrinterDriver
    ) -> None:
        """Test parsing malformed supply string."""
        result = connected_driver._parse_supply_string("garbage data")

        assert result is None
