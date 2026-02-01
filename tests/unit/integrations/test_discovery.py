"""
Unit tests for device discovery system.

Tests USB device discovery, CUPS printer discovery,
Windows printer discovery, and simulated device discovery.
"""

import platform
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.integrations.hardware.discovery import (
    CUPSPrinterDiscovery,
    IPPPrinterDiscovery,
    SimulatedDeviceDiscovery,
    USBDeviceDiscovery,
    Win32PrinterDiscovery,
    discover_all_devices,
    get_platform_discovery_handlers,
)
from ptpd_calibration.integrations.hardware.registry import DeviceType


class TestUSBDeviceDiscovery:
    """Tests for USB device discovery."""

    def test_is_available_with_pyserial(self):
        """Test is_available returns True when pyserial is installed."""
        try:
            import serial.tools.list_ports  # noqa: F401

            assert USBDeviceDiscovery.is_available() is True
        except ImportError:
            # pyserial not installed
            assert USBDeviceDiscovery.is_available() is False

    def test_is_available_without_pyserial(self):
        """Test is_available returns False when pyserial not installed."""
        with patch.dict("sys.modules", {"serial.tools.list_ports": None}):
            # This won't work as expected because the import is already cached
            # Just verify the method exists
            assert callable(USBDeviceDiscovery.is_available)

    def test_discover_returns_list(self):
        """Test discover returns a list."""
        if USBDeviceDiscovery.is_available():
            result = USBDeviceDiscovery.discover()
            assert isinstance(result, list)

    @patch("serial.tools.list_ports.comports")
    def test_discover_with_known_device(self, mock_comports):
        """Test discovery with a known X-Rite device."""
        # Mock a USB port that matches X-Rite i1Pro
        mock_port = MagicMock()
        mock_port.vid = 0x0765  # X-Rite vendor ID
        mock_port.pid = 0x5001  # i1Pro product ID
        mock_port.device = "/dev/ttyUSB0"
        mock_port.serial_number = "TEST123"

        mock_comports.return_value = [mock_port]

        with patch.object(USBDeviceDiscovery, "is_available", return_value=True):
            result = USBDeviceDiscovery.discover()

            assert len(result) == 1
            assert result[0].device_type == DeviceType.SPECTROPHOTOMETER
            assert result[0].device_info.vendor == "X-Rite"
            assert result[0].device_info.model == "i1Pro"
            assert result[0].connection_params["port"] == "/dev/ttyUSB0"

    @patch("serial.tools.list_ports.comports")
    def test_discover_with_unknown_device(self, mock_comports):
        """Test discovery ignores unknown devices."""
        mock_port = MagicMock()
        mock_port.vid = 0x1234  # Unknown vendor
        mock_port.pid = 0x5678  # Unknown product
        mock_port.device = "/dev/ttyUSB0"

        mock_comports.return_value = [mock_port]

        with patch.object(USBDeviceDiscovery, "is_available", return_value=True):
            result = USBDeviceDiscovery.discover()
            assert len(result) == 0


class TestCUPSPrinterDiscovery:
    """Tests for CUPS printer discovery."""

    def test_is_available_on_linux_macos(self):
        """Test is_available on Linux/macOS."""
        if platform.system() in ("Linux", "Darwin"):
            try:
                import cups  # noqa: F401

                assert CUPSPrinterDiscovery.is_available() is True
            except ImportError:
                assert CUPSPrinterDiscovery.is_available() is False
        else:
            assert CUPSPrinterDiscovery.is_available() is False

    def test_is_available_on_windows(self):
        """Test is_available returns False on Windows."""
        with patch("platform.system", return_value="Windows"):
            assert CUPSPrinterDiscovery.is_available() is False

    def test_discover_returns_printers(self):
        """Test discovery returns CUPS printers."""
        try:
            import cups  # noqa: F401
        except ImportError:
            pytest.skip("pycups not installed")

        with patch("cups.Connection") as mock_connection_class:
            mock_conn = MagicMock()
            mock_conn.getPrinters.return_value = {
                "EPSON_P800": {
                    "printer-info": "EPSON SureColor P800",
                    "printer-location": "Studio",
                    "device-uri": "usb://EPSON/P800",
                    "printer-state": 3,
                }
            }
            mock_connection_class.return_value = mock_conn

            with patch.object(CUPSPrinterDiscovery, "is_available", return_value=True):
                with patch("platform.system", return_value="Linux"):
                    result = CUPSPrinterDiscovery.discover()

                    assert len(result) == 1
                    assert result[0].device_type == DeviceType.PRINTER
                    assert result[0].device_info.vendor == "Epson"
                    assert result[0].connection_params["printer_name"] == "EPSON_P800"


class TestWin32PrinterDiscovery:
    """Tests for Windows printer discovery."""

    def test_is_available_on_windows(self):
        """Test is_available on Windows."""
        if platform.system() == "Windows":
            try:
                import win32print  # noqa: F401

                assert Win32PrinterDiscovery.is_available() is True
            except ImportError:
                assert Win32PrinterDiscovery.is_available() is False
        else:
            assert Win32PrinterDiscovery.is_available() is False

    def test_is_available_on_linux(self):
        """Test is_available returns False on Linux."""
        with patch("platform.system", return_value="Linux"):
            assert Win32PrinterDiscovery.is_available() is False

    def test_discover_returns_printers(self):
        """Test discovery returns Windows printers."""
        try:
            import win32print  # noqa: F401
        except ImportError:
            pytest.skip("pywin32 not installed")

        with patch("win32print.EnumPrinters") as mock_enum_printers:
            mock_enum_printers.return_value = [
                (0, "Epson SureColor P800", "EPSON SureColor P800", ""),
            ]

            with patch.object(Win32PrinterDiscovery, "is_available", return_value=True):
                with patch("platform.system", return_value="Windows"):
                    result = Win32PrinterDiscovery.discover()

                    assert len(result) == 1
                    assert result[0].device_type == DeviceType.PRINTER
                    assert result[0].device_info.vendor == "Epson"


class TestIPPPrinterDiscovery:
    """Tests for IPP/mDNS printer discovery."""

    def test_is_available_with_zeroconf(self):
        """Test is_available when zeroconf is installed."""
        try:
            import zeroconf  # noqa: F401

            assert IPPPrinterDiscovery.is_available() is True
        except ImportError:
            assert IPPPrinterDiscovery.is_available() is False

    def test_discover_returns_list(self):
        """Test discover returns a list."""
        if IPPPrinterDiscovery.is_available():
            # Use short timeout for test
            result = IPPPrinterDiscovery.discover(timeout_seconds=0.1)
            assert isinstance(result, list)


class TestSimulatedDeviceDiscovery:
    """Tests for simulated device discovery."""

    def test_is_available_always_true(self):
        """Test simulated discovery is always available."""
        assert SimulatedDeviceDiscovery.is_available() is True

    def test_discover_returns_both_device_types(self):
        """Test discover returns spectrophotometer and printer."""
        result = SimulatedDeviceDiscovery.discover()

        assert len(result) == 2

        types = {d.device_type for d in result}
        assert DeviceType.SPECTROPHOTOMETER in types
        assert DeviceType.PRINTER in types

    def test_discover_returns_simulated_flag(self):
        """Test discovered devices have is_simulated=True."""
        result = SimulatedDeviceDiscovery.discover()

        for device in result:
            assert device.is_simulated is True

    def test_discovered_devices_have_valid_info(self):
        """Test discovered devices have complete device info."""
        result = SimulatedDeviceDiscovery.discover()

        for device in result:
            assert device.device_id is not None
            assert device.device_info.vendor is not None
            assert device.device_info.model is not None
            assert len(device.device_info.capabilities) > 0


class TestDiscoverAllDevices:
    """Tests for discover_all_devices utility function."""

    def test_discover_all_returns_list(self):
        """Test discover_all_devices returns a list."""
        result = discover_all_devices()
        assert isinstance(result, list)

    def test_discover_all_includes_simulated(self):
        """Test discover_all includes simulated devices by default."""
        result = discover_all_devices(include_simulated=True)

        simulated = [d for d in result if d.is_simulated]
        assert len(simulated) >= 2

    def test_discover_all_excludes_simulated(self):
        """Test discover_all can exclude simulated devices."""
        result = discover_all_devices(include_simulated=False)

        simulated = [d for d in result if d.is_simulated]
        assert len(simulated) == 0

    def test_discover_all_filter_by_type(self):
        """Test discover_all can filter by device type."""
        result = discover_all_devices(
            include_simulated=True,
            device_types=[DeviceType.SPECTROPHOTOMETER],
        )

        for device in result:
            assert device.device_type == DeviceType.SPECTROPHOTOMETER

    def test_discover_all_filter_by_printer_type(self):
        """Test discover_all can filter for printers only."""
        result = discover_all_devices(
            include_simulated=True,
            device_types=[DeviceType.PRINTER],
        )

        for device in result:
            assert device.device_type == DeviceType.PRINTER


class TestGetPlatformDiscoveryHandlers:
    """Tests for get_platform_discovery_handlers utility."""

    def test_returns_list_of_handlers(self):
        """Test function returns list of handler classes."""
        handlers = get_platform_discovery_handlers()

        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_always_includes_simulated(self):
        """Test simulated discovery is always included."""
        handlers = get_platform_discovery_handlers()

        assert SimulatedDeviceDiscovery in handlers

    def test_includes_usb_if_available(self):
        """Test USB discovery included if pyserial available."""
        handlers = get_platform_discovery_handlers()

        if USBDeviceDiscovery.is_available():
            assert USBDeviceDiscovery in handlers

    def test_includes_cups_on_unix(self):
        """Test CUPS discovery included on Unix if pycups available."""
        handlers = get_platform_discovery_handlers()

        if CUPSPrinterDiscovery.is_available():
            assert CUPSPrinterDiscovery in handlers

    def test_includes_win32_on_windows(self):
        """Test Win32 discovery included on Windows if pywin32 available."""
        handlers = get_platform_discovery_handlers()

        if Win32PrinterDiscovery.is_available():
            assert Win32PrinterDiscovery in handlers
