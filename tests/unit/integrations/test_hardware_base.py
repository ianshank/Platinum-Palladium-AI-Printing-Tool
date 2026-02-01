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


class TestHardwareDeviceBaseDebugIntegration:
    """Test HardwareDeviceBase debug/logging integration."""

    def test_protocol_logger_lazy_init(self) -> None:
        """Test protocol logger is lazily initialized."""
        device = ConcreteDevice()
        assert device._protocol_logger is None

        # Access property
        logger = device.protocol_logger
        assert logger is not None
        assert device._protocol_logger is logger

    def test_protocol_logger_same_instance(self) -> None:
        """Test protocol logger returns same instance."""
        device = ConcreteDevice()
        logger1 = device.protocol_logger
        logger2 = device.protocol_logger
        assert logger1 is logger2

    def test_log_command_returns_start_time(self) -> None:
        """Test _log_command logs and returns start time."""
        device = ConcreteDevice()
        count_before = len(device.protocol_logger.get_messages())

        start_time = device._log_command("TEST_CMD")

        assert isinstance(start_time, float)
        assert start_time > 0

        # Verify message was logged
        messages = device.protocol_logger.get_messages()
        assert len(messages) == count_before + 1
        assert messages[-1].command == "TEST_CMD"

    def test_log_response_with_latency(self) -> None:
        """Test _log_response logs response with latency."""
        device = ConcreteDevice()
        count_before = len(device.protocol_logger.get_messages())

        start_time = device._log_command("CMD_TEST")
        device._log_response("CMD_TEST", "OK_RESPONSE", start_time)

        messages = device.protocol_logger.get_messages()
        assert len(messages) == count_before + 2
        # Check the last receive message
        assert messages[-1].response == "OK_RESPONSE"
        assert messages[-1].latency_ms is not None

    def test_log_response_without_start_time(self) -> None:
        """Test _log_response works without start time."""
        device = ConcreteDevice()
        count_before = len(device.protocol_logger.get_messages())

        device._log_response("CMD_NO_TIME", "OK_NO_LATENCY", None)

        messages = device.protocol_logger.get_messages()
        assert len(messages) == count_before + 1
        assert messages[-1].latency_ms is None

    def test_log_error_with_latency(self) -> None:
        """Test _log_error logs errors with latency."""
        device = ConcreteDevice()
        count_before = len(device.protocol_logger.get_messages())

        start_time = device._log_command("FAIL_CMD_TIMED")
        device._log_error("FAIL_CMD_TIMED", "TimeoutError", start_time)

        messages = device.protocol_logger.get_messages()
        assert len(messages) == count_before + 2
        assert messages[-1].error == "TimeoutError"
        assert messages[-1].latency_ms is not None

    def test_log_error_without_start_time(self) -> None:
        """Test _log_error works without start time."""
        device = ConcreteDevice()
        count_before = len(device.protocol_logger.get_messages())

        device._log_error("FAIL_CMD_NO_TIME", "ErrorNoLatency", None)

        messages = device.protocol_logger.get_messages()
        assert len(messages) == count_before + 1
        assert messages[-1].latency_ms is None

    def test_track_operation_context_manager(self) -> None:
        """Test _track_operation initializes debugger."""
        device = ConcreteDevice()
        assert device._debugger is None

        with device._track_operation("test_op"):
            pass

        assert device._debugger is not None

    def test_track_operation_with_metadata(self) -> None:
        """Test _track_operation accepts metadata."""
        device = ConcreteDevice()

        # Should not raise
        with device._track_operation("test_op", key="value", num=42):
            pass

    def test_debugger_reused(self) -> None:
        """Test debugger instance is reused."""
        device = ConcreteDevice()

        with device._track_operation("op1"):
            pass
        debugger1 = device._debugger

        with device._track_operation("op2"):
            pass
        debugger2 = device._debugger

        assert debugger1 is debugger2

    def test_error_status_logged(self) -> None:
        """Test error status is logged appropriately."""
        device = ConcreteDevice()
        device._set_status(DeviceStatus.ERROR, "Device error")
        assert device.status == DeviceStatus.ERROR

    def test_disconnected_from_connected_logged(self) -> None:
        """Test disconnection from connected state is logged."""
        device = ConcreteDevice()
        device._set_status(DeviceStatus.CONNECTED, "Connected")
        device._set_status(DeviceStatus.DISCONNECTED, "Lost connection")
        assert device.status == DeviceStatus.DISCONNECTED
