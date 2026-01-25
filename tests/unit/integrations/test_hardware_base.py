"""Unit tests for hardware base module.

Tests verify the base class and utility functions work correctly.
"""

from typing import Any

from ptpd_calibration.integrations.hardware.base import (
    HardwareDeviceBase,
    parse_device_response,
)
from ptpd_calibration.integrations.protocols import DeviceInfo, DeviceStatus


class ConcreteDevice(HardwareDeviceBase):
    """Concrete implementation for testing."""

    def __init__(self, device_type: str = "test_device") -> None:
        super().__init__(device_type)
        self.connect_called = False
        self.disconnect_called = False

    def connect(self, **kwargs: Any) -> bool:  # noqa: ARG002
        self.connect_called = True
        self._set_status(DeviceStatus.CONNECTED, "Connected to test device")
        return True

    def disconnect(self) -> None:
        self.disconnect_called = True
        self._set_status(DeviceStatus.DISCONNECTED, "Disconnected from test device")
        self._clear_device_info()


class TestHardwareDeviceBase:
    """Test HardwareDeviceBase class."""

    def test_initial_status_is_disconnected(self) -> None:
        """Test that initial status is DISCONNECTED."""
        device = ConcreteDevice()
        assert device.status == DeviceStatus.DISCONNECTED

    def test_initial_device_info_is_none(self) -> None:
        """Test that initial device info is None."""
        device = ConcreteDevice()
        assert device.device_info is None

    def test_is_connected_false_initially(self) -> None:
        """Test that is_connected is False initially."""
        device = ConcreteDevice()
        assert device.is_connected is False

    def test_is_connected_true_after_connect(self) -> None:
        """Test that is_connected is True after connect."""
        device = ConcreteDevice()
        device.connect()
        assert device.is_connected is True

    def test_set_status_updates_status(self) -> None:
        """Test that _set_status updates status."""
        device = ConcreteDevice()
        device._set_status(DeviceStatus.CONNECTED)
        assert device.status == DeviceStatus.CONNECTED

    def test_set_status_with_message_logs(self) -> None:
        """Test that _set_status with message logs appropriately."""
        device = ConcreteDevice()
        # Should not raise
        device._set_status(DeviceStatus.CONNECTED, "Test message")
        assert device.status == DeviceStatus.CONNECTED

    def test_set_device_info(self) -> None:
        """Test that _set_device_info sets device info."""
        device = ConcreteDevice()
        info = DeviceInfo(
            vendor="Test",
            model="Model",
            serial_number="123",
            firmware_version="1.0",
            capabilities=["test"],
        )
        device._set_device_info(info)
        assert device.device_info == info

    def test_clear_device_info(self) -> None:
        """Test that _clear_device_info clears device info."""
        device = ConcreteDevice()
        info = DeviceInfo(
            vendor="Test",
            model="Model",
            serial_number="123",
            firmware_version="1.0",
            capabilities=["test"],
        )
        device._set_device_info(info)
        device._clear_device_info()
        assert device.device_info is None

    def test_connect_calls_abstract_method(self) -> None:
        """Test that connect calls the concrete implementation."""
        device = ConcreteDevice()
        device.connect()
        assert device.connect_called is True

    def test_disconnect_calls_abstract_method(self) -> None:
        """Test that disconnect calls the concrete implementation."""
        device = ConcreteDevice()
        device.connect()
        device.disconnect()
        assert device.disconnect_called is True

    def test_disconnect_clears_device_info(self) -> None:
        """Test that disconnect clears device info."""
        device = ConcreteDevice()
        device.connect()
        info = DeviceInfo(
            vendor="Test",
            model="Model",
            serial_number="123",
            firmware_version="1.0",
            capabilities=["test"],
        )
        device._set_device_info(info)
        device.disconnect()
        assert device.device_info is None


class TestParseDeviceResponse:
    """Test parse_device_response utility function."""

    def test_parse_simple_key_value(self) -> None:
        """Test parsing simple key=value pairs."""
        response = "D=1.234,L=50.12"
        result = parse_device_response(response)
        assert result["D"] == 1.234
        assert result["L"] == 50.12

    def test_parse_with_ok_prefix(self) -> None:
        """Test parsing with OK: prefix."""
        response = "OK:D=1.234,L=50.12"
        result = parse_device_response(response)
        assert result["D"] == 1.234
        assert result["L"] == 50.12

    def test_parse_with_custom_prefix(self) -> None:
        """Test parsing with custom prefix."""
        response = "RESULT:D=1.234,L=50.12"
        result = parse_device_response(response, remove_prefix="RESULT:")
        assert result["D"] == 1.234
        assert result["L"] == 50.12

    def test_parse_with_type_conversion(self) -> None:
        """Test parsing with type conversion via field_map."""
        response = "COUNT=42,NAME=test"
        field_map = {"COUNT": int, "NAME": str}
        result = parse_device_response(response, field_map=field_map)
        assert result["COUNT"] == 42
        assert isinstance(result["COUNT"], int)
        assert result["NAME"] == "test"

    def test_parse_empty_response(self) -> None:
        """Test parsing empty response."""
        response = ""
        result = parse_device_response(response)
        assert result == {}

    def test_parse_no_equals_sign(self) -> None:
        """Test parsing response without equals sign."""
        response = "garbage,data,here"
        result = parse_device_response(response)
        assert result == {}

    def test_parse_with_spaces(self) -> None:
        """Test parsing with spaces around values."""
        response = "D = 1.234 , L = 50.12"
        result = parse_device_response(response)
        assert result["D"] == 1.234
        assert result["L"] == 50.12

    def test_parse_keys_uppercased(self) -> None:
        """Test that keys are uppercased."""
        response = "density=1.234,lightness=50.12"
        result = parse_device_response(response)
        assert "DENSITY" in result
        assert "LIGHTNESS" in result

    def test_parse_string_values(self) -> None:
        """Test parsing string values (non-numeric)."""
        response = "STATUS=ready,MODE=reflection"
        result = parse_device_response(response)
        assert result["STATUS"] == "ready"
        assert result["MODE"] == "reflection"

    def test_parse_invalid_float_becomes_string(self) -> None:
        """Test that invalid float becomes string."""
        response = "VALUE=not_a_number"
        result = parse_device_response(response)
        assert result["VALUE"] == "not_a_number"

    def test_parse_with_custom_delimiters(self) -> None:
        """Test parsing with custom delimiters."""
        response = "D:1.234;L:50.12"
        result = parse_device_response(
            response,
            delimiter=";",
            key_value_separator=":",
            remove_prefix=""
        )
        assert result["D"] == 1.234
        assert result["L"] == 50.12

    def test_parse_handles_multiple_equals_signs(self) -> None:
        """Test parsing value containing equals sign."""
        response = "FORMULA=a=b+c"
        result = parse_device_response(response)
        assert result["FORMULA"] == "a=b+c"
