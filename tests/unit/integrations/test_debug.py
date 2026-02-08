"""
Unit tests for hardware debugging infrastructure.

Tests ProtocolLogger, HardwareDebugger, and diagnostic utilities.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

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


class TestProtocolMessage:
    """Tests for ProtocolMessage dataclass."""

    def test_create_send_message(self):
        """Test creating a send message."""
        from datetime import datetime, timezone

        msg = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.SEND,
            device_type="spectrophotometer",
            command="*IDN?",
        )

        assert msg.direction == ProtocolDirection.SEND
        assert msg.command == "*IDN?"
        assert msg.response is None
        assert msg.error is None

    def test_create_receive_message_with_latency(self):
        """Test creating a receive message with latency."""
        from datetime import datetime, timezone

        msg = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.RECEIVE,
            device_type="spectrophotometer",
            command="*IDN?",
            response="X-Rite i1Pro",
            latency_ms=15.3,
        )

        assert msg.direction == ProtocolDirection.RECEIVE
        assert msg.latency_ms == 15.3

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from datetime import datetime, timezone

        msg = ProtocolMessage(
            timestamp=datetime.now(timezone.utc),
            direction=ProtocolDirection.SEND,
            device_type="printer",
            command="STATUS",
            raw_bytes=b"\x01\x02\x03",
        )

        data = msg.to_dict()

        assert "timestamp" in data
        assert data["direction"] == "send"
        assert data["raw_bytes"] == "010203"


class TestProtocolLogger:
    """Tests for ProtocolLogger."""

    def test_init(self):
        """Test logger initialization."""
        pl = ProtocolLogger(device_type="test_device")

        assert pl.device_type == "test_device"
        assert pl.max_messages == 1000
        assert len(pl.get_messages()) == 0

    def test_log_send(self):
        """Test logging outgoing commands."""
        pl = ProtocolLogger(device_type="spectrophotometer")

        pl.log_send("*IDN?")

        messages = pl.get_messages()
        assert len(messages) == 1
        assert messages[0].direction == ProtocolDirection.SEND
        assert messages[0].command == "*IDN?"

    def test_log_receive(self):
        """Test logging received responses."""
        pl = ProtocolLogger(device_type="spectrophotometer")

        pl.log_receive("*IDN?", "X-Rite i1Pro v1.2", latency_ms=12.5)

        messages = pl.get_messages()
        assert len(messages) == 1
        assert messages[0].direction == ProtocolDirection.RECEIVE
        assert messages[0].response == "X-Rite i1Pro v1.2"
        assert messages[0].latency_ms == 12.5

    def test_log_error(self):
        """Test logging communication errors."""
        pl = ProtocolLogger(device_type="spectrophotometer")

        pl.log_error("READ", "Timeout waiting for response", latency_ms=5000.0)

        messages = pl.get_messages()
        assert len(messages) == 1
        assert messages[0].error == "Timeout waiting for response"

    def test_message_limit(self):
        """Test message circular buffer limit."""
        pl = ProtocolLogger(device_type="test", max_messages=5)

        for i in range(10):
            pl.log_send(f"CMD{i}")

        messages = pl.get_messages()
        assert len(messages) == 5
        assert messages[0].command == "CMD5"
        assert messages[-1].command == "CMD9"

    def test_get_messages_with_limit(self):
        """Test getting limited number of messages."""
        pl = ProtocolLogger(device_type="test")

        for i in range(10):
            pl.log_send(f"CMD{i}")

        messages = pl.get_messages(limit=3)
        assert len(messages) == 3
        assert messages[0].command == "CMD7"

    def test_get_messages_with_direction_filter(self):
        """Test filtering messages by direction."""
        pl = ProtocolLogger(device_type="test")

        pl.log_send("CMD1")
        pl.log_receive("CMD1", "OK")
        pl.log_send("CMD2")
        pl.log_receive("CMD2", "OK")

        sends = pl.get_messages(direction=ProtocolDirection.SEND)
        assert len(sends) == 2
        assert all(m.direction == ProtocolDirection.SEND for m in sends)

    def test_get_statistics(self):
        """Test getting communication statistics."""
        pl = ProtocolLogger(device_type="test")

        pl.log_send("CMD1")
        pl.log_receive("CMD1", "OK", latency_ms=10.0)
        pl.log_send("CMD2")
        pl.log_receive("CMD2", "OK", latency_ms=20.0)
        pl.log_error("CMD3", "Timeout", latency_ms=1000.0)

        stats = pl.get_statistics()

        assert stats["total_messages"] == 5
        assert stats["sends"] == 2
        assert stats["receives"] == 3  # 2 ok + 1 error
        assert stats["errors"] == 1
        assert stats["avg_latency_ms"] == pytest.approx(343.33, rel=0.01)

    def test_get_statistics_empty(self):
        """Test statistics on empty logger."""
        pl = ProtocolLogger(device_type="test")

        stats = pl.get_statistics()

        assert stats["total_messages"] == 0
        assert stats["avg_latency_ms"] is None

    def test_clear(self):
        """Test clearing messages."""
        pl = ProtocolLogger(device_type="test")

        pl.log_send("CMD1")
        pl.log_send("CMD2")
        assert len(pl.get_messages()) == 2

        pl.clear()
        assert len(pl.get_messages()) == 0

    def test_export_to_file(self):
        """Test exporting messages to JSON file."""
        pl = ProtocolLogger(device_type="spectrophotometer")

        pl.log_send("*IDN?")
        pl.log_receive("*IDN?", "X-Rite i1Pro", latency_ms=15.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "protocol.json"
            pl.export_to_file(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)

            assert data["device_type"] == "spectrophotometer"
            assert len(data["messages"]) == 2
            assert "statistics" in data


class TestOperationMetrics:
    """Tests for OperationMetrics."""

    def test_duration_calculation(self):
        """Test duration is calculated correctly."""
        from datetime import datetime, timedelta, timezone

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=150)

        metrics = OperationMetrics(
            operation="test_op",
            start_time=start,
            end_time=end,
            success=True,
        )

        assert metrics.duration_ms == pytest.approx(150.0, rel=0.01)

    def test_duration_none_without_end_time(self):
        """Test duration is None when operation not finished."""
        from datetime import datetime, timezone

        metrics = OperationMetrics(
            operation="test_op",
            start_time=datetime.now(timezone.utc),
        )

        assert metrics.duration_ms is None


class TestHardwareDebugger:
    """Tests for HardwareDebugger singleton."""

    def test_singleton_pattern(self):
        """Test debugger is a singleton."""
        d1 = HardwareDebugger()
        d2 = HardwareDebugger()

        assert d1 is d2

    def test_enable_disable(self):
        """Test enabling and disabling debugging."""
        debugger = HardwareDebugger()

        debugger.disable()
        assert not debugger.enabled

        debugger.enable(DebugLevel.VERBOSE)
        assert debugger.enabled
        assert debugger.level == DebugLevel.VERBOSE

        debugger.disable()
        assert not debugger.enabled
        assert debugger.level == DebugLevel.OFF

    def test_get_protocol_logger(self):
        """Test getting protocol logger for device."""
        debugger = HardwareDebugger()

        pl1 = debugger.get_protocol_logger("spectro")
        pl2 = debugger.get_protocol_logger("spectro")
        pl3 = debugger.get_protocol_logger("printer")

        assert pl1 is pl2
        assert pl1 is not pl3

    def test_track_operation(self):
        """Test operation tracking."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.VERBOSE)
        debugger._operations.clear()

        with debugger.track_operation("test_operation", device="test") as metrics:
            time.sleep(0.01)

        assert metrics.success
        assert metrics.duration_ms is not None
        assert metrics.duration_ms >= 10

        ops = debugger.get_operations()
        assert len(ops) >= 1
        assert ops[-1].operation == "test_operation"

        debugger.disable()

    def test_track_operation_with_error(self):
        """Test operation tracking when error occurs."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.VERBOSE)
        debugger._operations.clear()

        with pytest.raises(ValueError), debugger.track_operation("failing_operation"):
            raise ValueError("Test error")

        ops = debugger.get_operations()
        assert len(ops) >= 1
        assert not ops[-1].success
        assert "Test error" in ops[-1].error

        debugger.disable()

    def test_get_operations_with_filter(self):
        """Test filtering operations by name."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.MINIMAL)
        debugger._operations.clear()

        with debugger.track_operation("calibrate"):
            pass
        with debugger.track_operation("measure"):
            pass
        with debugger.track_operation("calibrate_white"):
            pass

        calibrate_ops = debugger.get_operations(operation_filter="calibrate")
        assert len(calibrate_ops) == 2

        debugger.disable()

    def test_get_performance_report(self):
        """Test generating performance report."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.MINIMAL)
        debugger._operations.clear()

        for _ in range(3):
            with debugger.track_operation("fast_op"):
                pass
            with debugger.track_operation("slow_op"):
                time.sleep(0.01)

        report = debugger.get_performance_report()

        assert report["total_operations"] == 6
        assert report["success_rate"] == 1.0
        assert "fast_op" in report["operations"]
        assert "slow_op" in report["operations"]
        assert report["operations"]["fast_op"]["count"] == 3

        debugger.disable()

    def test_clear(self):
        """Test clearing all debug data."""
        debugger = HardwareDebugger()
        debugger.enable()

        with debugger.track_operation("test"):
            pass

        pl = debugger.get_protocol_logger("test")
        pl.log_send("CMD")

        debugger.clear()

        assert len(debugger.get_operations()) == 0
        assert len(pl.get_messages()) == 0

        debugger.disable()


class TestDebugDecorator:
    """Tests for debug_hardware_call decorator."""

    def test_decorator_when_disabled(self):
        """Test decorator does nothing when debug disabled."""
        debugger = HardwareDebugger()
        debugger.disable()

        @debug_hardware_call
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_decorator_when_enabled(self):
        """Test decorator tracks operations when enabled."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.VERBOSE)
        debugger._operations.clear()

        @debug_hardware_call
        def test_operation():
            return "result"

        result = test_operation()

        assert result == "result"
        ops = debugger.get_operations()
        assert len(ops) >= 1
        assert "test_operation" in ops[-1].operation

        debugger.disable()

    def test_decorator_with_class_method(self):
        """Test decorator works with class methods."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.VERBOSE)
        debugger._operations.clear()

        class TestDevice:
            @debug_hardware_call
            def read_value(self):
                return 42

        device = TestDevice()
        result = device.read_value()

        assert result == 42
        ops = debugger.get_operations()
        assert len(ops) >= 1
        assert "TestDevice.read_value" in ops[-1].operation

        debugger.disable()


class TestDebugModeContextManager:
    """Tests for debug_mode context manager."""

    def test_enables_debugging(self):
        """Test debug_mode enables debugging within context."""
        debugger = HardwareDebugger()
        debugger.disable()

        assert not debugger.enabled

        with debug_mode(DebugLevel.TRACE):
            assert debugger.enabled
            assert debugger.level == DebugLevel.TRACE

        assert not debugger.enabled

    def test_restores_previous_state(self):
        """Test debug_mode restores previous state on exit."""
        debugger = HardwareDebugger()
        debugger.enable(DebugLevel.MINIMAL)

        with debug_mode(DebugLevel.TRACE):
            assert debugger.level == DebugLevel.TRACE

        assert debugger.enabled
        assert debugger.level == DebugLevel.MINIMAL

        debugger.disable()


class TestDiagnosticReport:
    """Tests for DiagnosticReport model."""

    def test_create_empty_report(self):
        """Test creating empty diagnostic report."""
        report = DiagnosticReport()

        assert report.report_id is not None
        assert report.generated_at is not None
        assert len(report.devices) == 0
        assert len(report.warnings) == 0

    def test_summary(self):
        """Test generating report summary."""
        report = DiagnosticReport(
            devices=[{"device_id": "test"}],
            connections=[{"connected": True}, {"connected": False}],
            warnings=["Test warning"],
            errors=["Test error"],
        )

        summary = report.summary

        assert "Devices: 1" in summary
        assert "Active connections: 1" in summary
        assert "Warnings: 1" in summary
        assert "Errors: 1" in summary
        assert "Test warning" in summary
        assert "Test error" in summary

    def test_model_dump_json(self):
        """Test JSON serialization."""
        report = DiagnosticReport()

        json_str = report.model_dump_json()
        data = json.loads(json_str)

        assert "report_id" in data
        assert "generated_at" in data


class TestGetDiagnosticReport:
    """Tests for get_diagnostic_report function."""

    def test_returns_report(self):
        """Test function returns DiagnosticReport."""
        report = get_diagnostic_report()

        assert isinstance(report, DiagnosticReport)
        assert "platform" in report.system_info
        assert "python_version" in report.system_info

    def test_includes_recent_operations(self):
        """Test report includes recent operations."""
        debugger = HardwareDebugger()
        debugger.enable()
        debugger._operations.clear()

        with debugger.track_operation("test_for_report"):
            pass

        report = get_diagnostic_report()

        ops = [o for o in report.recent_operations if "test_for_report" in o.get("operation", "")]
        assert len(ops) >= 1

        debugger.disable()


class TestSaveDebugSession:
    """Tests for save_debug_session function."""

    def test_saves_to_file(self):
        """Test saving debug session to file."""
        debugger = HardwareDebugger()
        debugger.enable()

        with debugger.track_operation("save_test"):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_debug_session(Path(tmpdir) / "session")

            assert path.exists()
            assert path.suffix == ".json"

            with open(path) as f:
                data = json.load(f)

            assert "session_id" in data
            assert "diagnostic_report" in data
            assert "performance_report" in data

        debugger.disable()

    def test_creates_parent_directories(self):
        """Test function creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_debug_session(Path(tmpdir) / "nested" / "dir" / "session.json")

            assert path.exists()
            assert path.parent.exists()
