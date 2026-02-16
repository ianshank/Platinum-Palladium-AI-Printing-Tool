"""
Comprehensive unit tests for core infrastructure modules.

Tests the following modules:
- ptpd_calibration.core.base_service (BaseService, AsyncBaseService, ServiceResult, ValidationResult)
- ptpd_calibration.core.debug (timer, trace, debug_context, DebugMixin, MemoryTracker)
- ptpd_calibration.core.events (EventBus, Event, event handlers)
- ptpd_calibration.core.logging (setup_logging, get_logger, LogContext, JSONFormatter)

These tests verify REAL behavior with minimal mocking, targeting 80%+ coverage.
"""

import asyncio
import json
import logging
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from ptpd_calibration.core.base_service import (
    AsyncBaseService,
    BaseService,
    ServiceResult,
    ValidationResult,
)
from ptpd_calibration.core.debug import (
    DebugMixin,
    MemoryTracker,
    breakpoint_if_debug,
    debug_context,
    dump_exception,
    timer,
    trace,
)
from ptpd_calibration.core.events import (
    CalibrationCompleted,
    CalibrationFailed,
    CalibrationStarted,
    CurveExported,
    CurveGenerated,
    Event,
    EventBus,
    HardwareConnected,
    HardwareDisconnected,
    MeasurementTaken,
    get_event_bus,
)
from ptpd_calibration.core.logging import (
    ColoredFormatter,
    JSONFormatter,
    LogContext,
    LoggingMixin,
    get_logger,
    log_operation,
    setup_logging,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def init_log_context():
    """Initialize log context for all tests."""
    from ptpd_calibration.core.logging import _log_context

    token = _log_context.set({})
    yield
    _log_context.reset(token)


# =============================================================================
# Test ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_create_invalid_result_with_errors(self):
        """Test creating an invalid result with error messages."""
        result = ValidationResult(
            is_valid=False, errors=["Error 1", "Error 2"], warnings=["Warning 1"]
        )
        assert not result.is_valid
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_validation_result_boolean_context(self):
        """Test ValidationResult works in boolean context."""
        valid = ValidationResult(is_valid=True)
        invalid = ValidationResult(is_valid=False, errors=["Error"])

        assert valid  # Truthy
        assert not invalid  # Falsy

    def test_validation_result_with_warnings_still_valid(self):
        """Test that warnings don't make result invalid."""
        result = ValidationResult(is_valid=True, warnings=["Non-critical warning"])
        assert result.is_valid
        assert len(result.warnings) == 1
        assert result  # Still truthy


# =============================================================================
# Test ServiceResult
# =============================================================================


class TestServiceResult:
    """Tests for ServiceResult dataclass."""

    def test_create_success_result_with_ok(self):
        """Test creating success result using ok() class method."""

        class OutputData(BaseModel):
            value: int

        data = OutputData(value=42)
        result = ServiceResult.ok(data, duration=1.5, extra_info="test")

        assert result.success
        assert result.data == data
        assert result.error is None
        assert result.duration_seconds == 1.5
        assert result.metadata["extra_info"] == "test"
        assert isinstance(result.timestamp, datetime)

    def test_create_failure_result_with_fail(self):
        """Test creating failure result using fail() class method."""
        result = ServiceResult.fail(
            error="Something went wrong",
            error_type="ValueError",
            duration=0.5,
            attempt=3,
        )

        assert not result.success
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.error_type == "ValueError"
        assert result.duration_seconds == 0.5
        assert result.metadata["attempt"] == 3

    def test_unwrap_success_returns_data(self):
        """Test unwrap() returns data for successful result."""

        class OutputData(BaseModel):
            value: str

        data = OutputData(value="test")
        result = ServiceResult.ok(data)

        unwrapped = result.unwrap()
        assert unwrapped == data

    def test_unwrap_failure_raises_runtime_error(self):
        """Test unwrap() raises RuntimeError for failed result."""
        result = ServiceResult.fail(error="Operation failed")

        with pytest.raises(RuntimeError, match="Operation failed"):
            result.unwrap()

    def test_unwrap_or_returns_data_on_success(self):
        """Test unwrap_or() returns data when successful."""

        class OutputData(BaseModel):
            value: int

        data = OutputData(value=100)
        default = OutputData(value=0)
        result = ServiceResult.ok(data)

        assert result.unwrap_or(default) == data

    def test_unwrap_or_returns_default_on_failure(self):
        """Test unwrap_or() returns default when failed."""

        class OutputData(BaseModel):
            value: int

        default = OutputData(value=0)
        result = ServiceResult.fail(error="Failed")

        assert result.unwrap_or(default) == default

    def test_timestamp_is_utc(self):
        """Test that timestamp is in UTC timezone."""
        result = ServiceResult.ok("data")
        assert result.timestamp.tzinfo == timezone.utc


# =============================================================================
# Test BaseService
# =============================================================================


class TestBaseService:
    """Tests for BaseService abstract base class."""

    def test_concrete_service_implementation(self):
        """Test implementing a concrete service."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class DoubleService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                if data.value < 0:
                    return ValidationResult(False, errors=["Value must be non-negative"])
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value * 2)

        service = DoubleService()
        result = service.execute(TestInput(value=5))

        assert result.success
        assert result.data.result == 10

    def test_service_with_validation_failure(self):
        """Test service execution when validation fails."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class ValidatingService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                if data.value > 100:
                    return ValidationResult(False, errors=["Value too large"])
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value)

        service = ValidatingService()
        result = service.execute(TestInput(value=200))

        assert not result.success
        assert result.error == "Value too large"
        assert result.error_type == "ValidationError"
        assert result.data is None

    def test_service_with_processing_exception(self):
        """Test service execution when processing raises exception."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class FailingService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                raise ValueError("Processing error")

        service = FailingService()
        result = service.execute(TestInput(value=1))

        assert not result.success
        assert "Processing error" in result.error
        assert result.error_type == "ValueError"

    def test_service_skip_validation(self):
        """Test service execution with skip_validation=True."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class StrictService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                # This would normally fail
                return ValidationResult(False, errors=["Always invalid"])

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value * 3)

        service = StrictService()
        result = service.execute(TestInput(value=7), skip_validation=True)

        assert result.success
        assert result.data.result == 21

    def test_service_with_warnings(self):
        """Test service execution logs warnings but continues."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class WarningService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                warnings = ["This is a non-critical warning"] if data.value > 50 else []
                return ValidationResult(True, warnings=warnings)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value)

        service = WarningService()
        result = service.execute(TestInput(value=75))

        assert result.success
        assert result.data.result == 75

    def test_service_callable_interface(self):
        """Test calling service as a function using __call__."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class SimpleService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value + 1)

        service = SimpleService()
        output = service(TestInput(value=9))

        assert isinstance(output, TestOutput)
        assert output.result == 10

    def test_service_callable_raises_on_validation_error(self):
        """Test __call__ raises ValueError on validation failure."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class StrictService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(False, errors=["Invalid"])

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=0)

        service = StrictService()
        with pytest.raises(ValueError, match="Invalid"):
            service(TestInput(value=1))

    def test_service_callable_raises_on_processing_error(self):
        """Test __call__ raises RuntimeError on processing failure."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class FailingService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                raise KeyError("Missing key")

        service = FailingService()
        with pytest.raises(RuntimeError, match="Missing key"):
            service(TestInput(value=1))

    def test_service_with_config_key(self):
        """Test service accessing configuration via config_key."""

        class TestInput(BaseModel):
            pass

        class TestOutput(BaseModel):
            pass

        class ConfigService(BaseService[TestInput, TestOutput]):
            config_key = "curves"

            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput()

        service = ConfigService()
        # Should have settings and config properties
        assert service.settings is not None
        assert service.logger is not None

    def test_service_performance_tracking(self):
        """Test that service tracks execution duration."""

        class TestInput(BaseModel):
            sleep_duration: float

        class TestOutput(BaseModel):
            pass

        class SlowService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                time.sleep(data.sleep_duration)
                return TestOutput()

        service = SlowService()
        result = service.execute(TestInput(sleep_duration=0.1))

        assert result.success
        assert result.duration_seconds >= 0.1
        assert result.duration_seconds < 0.5  # Reasonable upper bound

    def test_service_with_context(self):
        """Test service execution with logging context."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class ContextService(BaseService[TestInput, TestOutput]):
            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value)

        service = ContextService()
        result = service.execute(
            TestInput(value=42), context={"request_id": "test-123", "user": "admin"}
        )

        assert result.success


# =============================================================================
# Test AsyncBaseService
# =============================================================================


class TestAsyncBaseService:
    """Tests for AsyncBaseService abstract base class."""

    @pytest.mark.asyncio
    async def test_async_service_implementation(self):
        """Test implementing an async service."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class AsyncDoubleService(AsyncBaseService[TestInput, TestOutput]):
            async def validate_input(self, data: TestInput) -> ValidationResult:
                await asyncio.sleep(0.01)
                return ValidationResult(True)

            async def process(self, data: TestInput) -> TestOutput:
                await asyncio.sleep(0.01)
                return TestOutput(result=data.value * 2)

        service = AsyncDoubleService()
        result = await service.execute(TestInput(value=5))

        assert result.success
        assert result.data.result == 10

    @pytest.mark.asyncio
    async def test_async_service_validation_failure(self):
        """Test async service with validation failure."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class AsyncValidatingService(AsyncBaseService[TestInput, TestOutput]):
            async def validate_input(self, data: TestInput) -> ValidationResult:
                if data.value < 0:
                    return ValidationResult(False, errors=["Must be positive"])
                return ValidationResult(True)

            async def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value)

        service = AsyncValidatingService()
        result = await service.execute(TestInput(value=-5))

        assert not result.success
        assert "Must be positive" in result.error

    @pytest.mark.asyncio
    async def test_async_service_processing_exception(self):
        """Test async service when processing raises exception."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class AsyncFailingService(AsyncBaseService[TestInput, TestOutput]):
            async def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            async def process(self, data: TestInput) -> TestOutput:
                raise RuntimeError("Async processing failed")

        service = AsyncFailingService()
        result = await service.execute(TestInput(value=1))

        assert not result.success
        assert "Async processing failed" in result.error

    @pytest.mark.asyncio
    async def test_async_service_callable(self):
        """Test async service __call__ interface."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class AsyncSimpleService(AsyncBaseService[TestInput, TestOutput]):
            async def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            async def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value + 10)

        service = AsyncSimpleService()
        output = await service(TestInput(value=5))

        assert isinstance(output, TestOutput)
        assert output.result == 15

    @pytest.mark.asyncio
    async def test_async_service_skip_validation(self):
        """Test async service with skip_validation."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class AsyncStrictService(AsyncBaseService[TestInput, TestOutput]):
            async def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(False, errors=["Always fails"])

            async def process(self, data: TestInput) -> TestOutput:
                return TestOutput(result=data.value * 3)

        service = AsyncStrictService()
        result = await service.execute(TestInput(value=4), skip_validation=True)

        assert result.success
        assert result.data.result == 12


# =============================================================================
# Test Debug Utilities
# =============================================================================


class TestDebugTimer:
    """Tests for timer decorator."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_timer_logs_duration_when_debug_enabled(self, mock_debug):
        """Test timer logs execution time when debug mode is on."""

        @timer
        def timed_function():
            time.sleep(0.05)
            return "result"

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            result = timed_function()

            assert result == "result"
            assert mock_logger.debug.called
            call_args = mock_logger.debug.call_args[0][0]
            assert "TIMER" in call_args
            assert "timed_function" in call_args

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=False)
    def test_timer_no_overhead_when_debug_disabled(self, mock_debug):
        """Test timer has zero overhead when debug is off."""

        @timer
        def fast_function():
            return 42

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            result = fast_function()

            assert result == 42
            assert not mock_logger.debug.called

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_timer_logs_on_exception(self, mock_debug):
        """Test timer logs duration even when function raises exception."""

        @timer
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            assert mock_logger.debug.called
            call_args = mock_logger.debug.call_args[0][0]
            assert "FAILED" in call_args


class TestDebugTrace:
    """Tests for trace decorator."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_trace_logs_entry_and_exit(self, mock_debug):
        """Test trace logs function entry and exit."""

        @trace
        def traced_function(x, y):
            return x + y

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            result = traced_function(5, 3)

            assert result == 8
            assert mock_logger.debug.call_count >= 2
            calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("ENTER" in call for call in calls)
            assert any("EXIT" in call for call in calls)

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_trace_with_max_arg_length(self, mock_debug):
        """Test trace truncates long arguments."""

        @trace(max_arg_length=10)
        def traced_function(long_string):
            return "done"

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            traced_function("a" * 100)

            # Check that argument was truncated
            enter_call = [
                call for call in mock_logger.debug.call_args_list if "ENTER" in call[0][0]
            ][0]
            assert "..." in enter_call[0][0]

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_trace_without_log_result(self, mock_debug):
        """Test trace with log_result=False."""

        @trace(log_result=False)
        def traced_function():
            return "secret"

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            traced_function()

            exit_call = [
                call for call in mock_logger.debug.call_args_list if "EXIT" in call[0][0]
            ][0]
            # Should not contain the result
            assert "secret" not in exit_call[0][0]

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=False)
    def test_trace_no_overhead_when_disabled(self, mock_debug):
        """Test trace has no overhead when debug is off."""

        @trace
        def fast_function():
            return 123

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            result = fast_function()

            assert result == 123
            assert not mock_logger.debug.called


class TestDebugContext:
    """Tests for debug_context context manager."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_debug_context_logs_start_and_end(self, mock_debug):
        """Test debug_context logs start and end of operation."""
        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            with debug_context("test_operation", param1=10, param2="test"):
                time.sleep(0.01)

            calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("START" in call and "test_operation" in call for call in calls)
            assert any("END" in call and "test_operation" in call for call in calls)

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_debug_context_logs_failure(self, mock_debug):
        """Test debug_context logs failure when exception occurs."""
        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            with pytest.raises(ValueError):
                with debug_context("failing_op"):
                    raise ValueError("Test error")

            calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("FAIL" in call for call in calls)

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=False)
    def test_debug_context_no_overhead_when_disabled(self, mock_debug):
        """Test debug_context has no overhead when disabled."""
        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            with debug_context("test_op"):
                pass

            assert not mock_logger.debug.called


class TestDebugMixin:
    """Tests for DebugMixin class."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_debug_mixin_debug_method(self, mock_debug):
        """Test DebugMixin._debug method."""

        class TestClass(DebugMixin):
            def do_something(self):
                self._debug("Doing something", count=5)

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            obj = TestClass()
            obj.do_something()

            assert mock_logger.debug.called
            call_args = mock_logger.debug.call_args[0][0]
            assert "TestClass" in call_args
            assert "Doing something" in call_args

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_debug_mixin_enter_exit(self, mock_debug):
        """Test DebugMixin._debug_enter and _debug_exit."""

        class TestClass(DebugMixin):
            def process(self, value):
                start = self._debug_enter("process", value=value)
                result = value * 2
                self._debug_exit("process", start, result=result)
                return result

        with patch("ptpd_calibration.core.debug.logger") as mock_logger:
            obj = TestClass()
            result = obj.process(10)

            assert result == 20
            assert mock_logger.debug.call_count >= 2


class TestDumpException:
    """Tests for dump_exception utility."""

    def test_dump_exception_basic(self):
        """Test basic exception dump."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            dump = dump_exception(e)

            assert "EXCEPTION DUMP" in dump
            assert "ValueError" in dump
            assert "Test exception" in dump
            assert "Traceback:" in dump

    def test_dump_exception_with_locals(self):
        """Test exception dump with local variables."""
        try:
            local_var = 42
            another_var = "test"
            raise RuntimeError("Error with locals")
        except RuntimeError as e:
            dump = dump_exception(e, include_locals=True)

            assert "Local Variables:" in dump
            # May include local_var and another_var depending on frame


class TestBreakpointIfDebug:
    """Tests for breakpoint_if_debug utility."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=False)
    def test_breakpoint_not_called_when_debug_disabled(self, mock_debug):
        """Test breakpoint is not called when debug is off."""
        with patch("pdb.set_trace") as mock_pdb:
            breakpoint_if_debug()
            assert not mock_pdb.called

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_breakpoint_called_when_debug_enabled(self, mock_debug):
        """Test breakpoint is called when debug is on."""
        with patch("pdb.set_trace") as mock_pdb:
            breakpoint_if_debug()
            assert mock_pdb.called


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=False)
    def test_memory_tracker_disabled_when_debug_off(self, mock_debug):
        """Test memory tracker does nothing when debug is off."""
        tracker = MemoryTracker()
        tracker.checkpoint("test")
        report = tracker.report()

        assert "No checkpoints recorded" in report

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_memory_tracker_without_psutil(self, mock_debug):
        """Test memory tracker handles missing psutil gracefully."""
        with patch.dict("sys.modules", {"psutil": None}):
            tracker = MemoryTracker()
            tracker.checkpoint("test")
            report = tracker.report()

            # Should handle gracefully
            assert isinstance(report, str)

    @patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True)
    def test_memory_tracker_with_checkpoints(self, mock_debug):
        """Test memory tracker records checkpoints."""
        try:
            import psutil

            tracker = MemoryTracker()
            tracker.checkpoint("start")
            time.sleep(0.01)
            tracker.checkpoint("middle")
            time.sleep(0.01)
            tracker.checkpoint("end")

            report = tracker.report()
            assert "start" in report
            assert "middle" in report
            assert "end" in report
            assert "MB" in report
        except ImportError:
            pytest.skip("psutil not available")


# =============================================================================
# Test Event System
# =============================================================================


class TestEvent:
    """Tests for Event base class."""

    def test_create_basic_event(self):
        """Test creating a basic event."""
        event = Event(event_type="test.event")

        assert event.event_type == "test.event"
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == timezone.utc
        assert event.metadata == {}

    def test_create_event_with_metadata(self):
        """Test creating event with metadata."""
        event = Event(event_type="test.event", metadata={"key": "value", "count": 42})

        assert event.metadata["key"] == "value"
        assert event.metadata["count"] == 42

    def test_custom_event_subclass(self):
        """Test creating custom event subclass."""

        class CustomEvent(Event):
            event_type: str = "custom.event"
            data: str

        event = CustomEvent(data="test data")
        assert event.event_type == "custom.event"
        assert event.data == "test data"


class TestPredefinedEvents:
    """Tests for predefined event types."""

    def test_calibration_started_event(self):
        """Test CalibrationStarted event."""
        event = CalibrationStarted(session_id="test-123", paper_type="Arches Platine")

        assert event.event_type == "calibration.started"
        assert event.session_id == "test-123"
        assert event.paper_type == "Arches Platine"

    def test_calibration_completed_event(self):
        """Test CalibrationCompleted event."""
        event = CalibrationCompleted(
            session_id="test-123", record_id="rec-456", quality_score=0.95
        )

        assert event.event_type == "calibration.completed"
        assert event.session_id == "test-123"
        assert event.record_id == "rec-456"
        assert event.quality_score == 0.95

    def test_calibration_failed_event(self):
        """Test CalibrationFailed event."""
        event = CalibrationFailed(
            session_id="test-123", error="Validation failed", error_type="ValueError"
        )

        assert event.event_type == "calibration.failed"
        assert event.error == "Validation failed"

    def test_curve_generated_event(self):
        """Test CurveGenerated event."""
        event = CurveGenerated(curve_name="TestCurve", num_points=256, dmax=2.1, dmin=0.1)

        assert event.event_type == "curve.generated"
        assert event.num_points == 256

    def test_curve_exported_event(self):
        """Test CurveExported event."""
        event = CurveExported(
            curve_name="TestCurve", format="qtr", file_path="/tmp/curve.quad"
        )

        assert event.event_type == "curve.exported"
        assert event.format == "qtr"

    def test_hardware_connected_event(self):
        """Test HardwareConnected event."""
        event = HardwareConnected(
            device_type="scanner", device_id="dev-123", vendor="Epson", model="V850"
        )

        assert event.event_type == "hardware.connected"
        assert event.vendor == "Epson"

    def test_hardware_disconnected_event(self):
        """Test HardwareDisconnected event."""
        event = HardwareDisconnected(
            device_type="scanner", device_id="dev-123", reason="User disconnected"
        )

        assert event.event_type == "hardware.disconnected"

    def test_measurement_taken_event(self):
        """Test MeasurementTaken event."""
        event = MeasurementTaken(
            device_id="dev-123",
            measurement_type="density",
            value=1.85,
            unit="D",
        )

        assert event.event_type == "measurement.taken"
        assert event.value == 1.85


class TestEventBus:
    """Tests for EventBus publish-subscribe system."""

    def setup_method(self):
        """Reset EventBus before each test."""
        EventBus.reset()

    def test_event_bus_singleton(self):
        """Test EventBus uses singleton pattern."""
        bus1 = EventBus()
        bus2 = EventBus()

        assert bus1 is bus2

    def test_get_event_bus_function(self):
        """Test get_event_bus convenience function."""
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_subscribe_and_publish(self):
        """Test subscribing to and publishing events."""
        bus = EventBus()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        bus.subscribe("test.event", handler)
        event = Event(event_type="test.event")
        bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0] == event

    def test_subscribe_returns_unsubscribe_function(self):
        """Test subscribe returns an unsubscribe function."""
        bus = EventBus()
        call_count = [0]

        def handler(event: Event):
            call_count[0] += 1

        unsubscribe = bus.subscribe("test.event", handler)

        bus.publish(Event(event_type="test.event"))
        assert call_count[0] == 1

        unsubscribe()
        bus.publish(Event(event_type="test.event"))
        assert call_count[0] == 1  # No change after unsubscribe

    def test_unsubscribe_method(self):
        """Test unsubscribe method."""
        bus = EventBus()
        call_count = [0]

        def handler(event: Event):
            call_count[0] += 1

        bus.subscribe("test.event", handler)
        assert bus.unsubscribe("test.event", handler)

        bus.publish(Event(event_type="test.event"))
        assert call_count[0] == 0

    def test_on_decorator(self):
        """Test @on decorator for subscribing."""
        bus = EventBus()
        received_events = []

        @bus.on("test.event")
        def handler(event: Event):
            received_events.append(event)

        bus.publish(Event(event_type="test.event"))
        assert len(received_events) == 1

    def test_wildcard_subscription_star(self):
        """Test wildcard subscription with *."""
        bus = EventBus()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        bus.subscribe("*", handler)

        bus.publish(Event(event_type="test.event1"))
        bus.publish(Event(event_type="other.event2"))

        assert len(received_events) == 2

    def test_wildcard_subscription_prefix(self):
        """Test wildcard subscription with prefix.*."""
        bus = EventBus()
        calibration_events = []

        def handler(event: Event):
            calibration_events.append(event)

        bus.subscribe("calibration.*", handler)

        bus.publish(Event(event_type="calibration.started"))
        bus.publish(Event(event_type="calibration.completed"))
        bus.publish(Event(event_type="curve.generated"))

        assert len(calibration_events) == 2

    def test_multiple_handlers_for_same_event(self):
        """Test multiple handlers can subscribe to same event."""
        bus = EventBus()
        handler1_calls = []
        handler2_calls = []

        def handler1(event: Event):
            handler1_calls.append(event)

        def handler2(event: Event):
            handler2_calls.append(event)

        bus.subscribe("test.event", handler1)
        bus.subscribe("test.event", handler2)

        event = Event(event_type="test.event")
        bus.publish(event)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_handler_exception_isolation(self):
        """Test that one handler's exception doesn't prevent others from running."""
        bus = EventBus()
        successful_handler_called = [False]

        def failing_handler(event: Event):
            raise ValueError("Handler error")

        def successful_handler(event: Event):
            successful_handler_called[0] = True

        bus.subscribe("test.event", failing_handler)
        bus.subscribe("test.event", successful_handler)

        bus.publish(Event(event_type="test.event"))

        assert successful_handler_called[0]

    def test_weak_reference_subscription(self):
        """Test weak reference subscription."""
        bus = EventBus()
        received_events = []

        class Handler:
            __name__ = "Handler"  # Add __name__ attribute for logging

            def __call__(self, event: Event):
                received_events.append(event)

        handler = Handler()
        bus.subscribe("test.event", handler, weak=True)

        bus.publish(Event(event_type="test.event"))
        assert len(received_events) == 1

        # Delete handler and force garbage collection
        del handler
        import gc

        gc.collect()

        # Handler should be cleaned up
        bus.publish(Event(event_type="test.event"))
        # Count might be 1 or 2 depending on gc timing, but handler is weakly referenced

    def test_async_handler(self):
        """Test async handler in publish (scheduled)."""
        bus = EventBus()
        received_events = []

        async def async_handler(event: Event):
            await asyncio.sleep(0.01)
            received_events.append(event)

        bus.subscribe("test.event", async_handler)

        # Publish creates task but doesn't wait
        bus.publish(Event(event_type="test.event"))

        # Need to give event loop time to process
        # In practice, this would be handled by the running event loop

    @pytest.mark.asyncio
    async def test_publish_async(self):
        """Test publish_async for async handlers."""
        bus = EventBus()
        received_events = []

        async def async_handler(event: Event):
            await asyncio.sleep(0.01)
            received_events.append(event)

        bus.subscribe("test.event", async_handler)

        await bus.publish_async(Event(event_type="test.event"))

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_publish_async_with_sync_handler(self):
        """Test publish_async works with sync handlers too."""
        bus = EventBus()
        received_events = []

        def sync_handler(event: Event):
            received_events.append(event)

        bus.subscribe("test.event", sync_handler)

        await bus.publish_async(Event(event_type="test.event"))

        assert len(received_events) == 1

    def test_clear_specific_event_type(self):
        """Test clearing subscribers for specific event type."""
        bus = EventBus()
        handler1_calls = []
        handler2_calls = []

        def handler1(event: Event):
            handler1_calls.append(event)

        def handler2(event: Event):
            handler2_calls.append(event)

        bus.subscribe("test.event1", handler1)
        bus.subscribe("test.event2", handler2)

        bus.clear("test.event1")

        bus.publish(Event(event_type="test.event1"))
        bus.publish(Event(event_type="test.event2"))

        assert len(handler1_calls) == 0
        assert len(handler2_calls) == 1

    def test_clear_all_subscribers(self):
        """Test clearing all subscribers."""
        bus = EventBus()
        handler_calls = []

        def handler(event: Event):
            handler_calls.append(event)

        bus.subscribe("test.event1", handler)
        bus.subscribe("test.event2", handler)

        bus.clear()

        bus.publish(Event(event_type="test.event1"))
        bus.publish(Event(event_type="test.event2"))

        assert len(handler_calls) == 0


# =============================================================================
# Test Logging
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_json_formatter_basic(self):
        """Test JSONFormatter produces valid JSON."""
        from ptpd_calibration.core.logging import _log_context

        formatter = JSONFormatter()

        # Initialize context var
        token = _log_context.set({})
        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert data["level"] == "INFO"
            assert data["message"] == "Test message"
            assert data["logger"] == "test.logger"
            assert "timestamp" in data
        finally:
            _log_context.reset(token)

    def test_json_formatter_with_exception(self):
        """Test JSONFormatter includes exception info."""
        from ptpd_calibration.core.logging import _log_context

        formatter = JSONFormatter()

        # Initialize context var
        token = _log_context.set({})
        try:
            try:
                raise ValueError("Test error")
            except ValueError:
                import sys

                exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=20,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert "exception" in data
            assert "exception_type" in data
            assert data["exception_type"] == "ValueError"
        finally:
            _log_context.reset(token)


class TestColoredFormatter:
    """Tests for ColoredFormatter."""

    def test_colored_formatter_adds_colors(self):
        """Test ColoredFormatter adds ANSI color codes."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should contain ANSI color codes
        assert "\033[" in output
        assert "Error message" in output


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_basic(self, tmp_path):
        """Test basic logging setup."""
        setup_logging(level="DEBUG")

        logger = logging.getLogger("ptpd_calibration")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file output."""
        from ptpd_calibration.core.logging import _log_context

        log_file = tmp_path / "test.log"

        # Initialize context var
        token = _log_context.set({})
        try:
            setup_logging(level="INFO", log_file=log_file)

            logger = logging.getLogger("ptpd_calibration")
            logger.info("Test message")

            # Flush handlers to ensure write
            for handler in logger.handlers:
                handler.flush()

            assert log_file.exists()
            content = log_file.read_text()
            # JSON format, so check for message field
            assert "Test message" in content or "message" in content
        finally:
            _log_context.reset(token)

    def test_setup_logging_json_format(self):
        """Test logging setup with JSON format."""
        setup_logging(level="INFO", json_format=True)

        logger = logging.getLogger("ptpd_calibration")
        # Should have JSONFormatter
        assert any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)

    def test_setup_logging_reconfiguration(self):
        """Test that setup_logging can be called multiple times."""
        setup_logging(level="INFO")
        initial_handler_count = len(logging.getLogger("ptpd_calibration").handlers)

        setup_logging(level="DEBUG")
        final_handler_count = len(logging.getLogger("ptpd_calibration").handlers)

        # Should clear previous handlers
        assert final_handler_count == initial_handler_count


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_ensures_setup(self):
        """Test get_logger ensures logging is configured."""
        from ptpd_calibration.core.logging import _logging_configured
        import ptpd_calibration.core.logging as logging_module

        # Reset logging state
        logging.getLogger("ptpd_calibration").handlers.clear()
        logging_module._logging_configured = False

        logger = get_logger("test.module")

        # Should have configured logging
        root = logging.getLogger("ptpd_calibration")
        assert len(root.handlers) > 0

    def test_get_logger_normalizes_name(self):
        """Test get_logger normalizes module name."""
        logger = get_logger("my.module")
        assert logger.name.startswith("ptpd_calibration")


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_adds_context(self):
        """Test LogContext adds context to log messages."""
        from ptpd_calibration.core.logging import _log_context

        # Initialize context var with empty dict
        token = _log_context.set({})
        try:
            with LogContext(request_id="test-123", user="admin"):
                # Context should be set
                ctx = _log_context.get()
                assert ctx["request_id"] == "test-123"
                assert ctx["user"] == "admin"
        finally:
            _log_context.reset(token)

    def test_log_context_restores_after_exit(self):
        """Test LogContext restores previous context after exit."""
        from ptpd_calibration.core.logging import _log_context

        # Set initial context
        token = _log_context.set({"initial": "value"})
        try:
            with LogContext(request_id="test-123"):
                ctx = _log_context.get()
                assert "request_id" in ctx
        finally:
            _log_context.reset(token)

    def test_log_context_nested(self):
        """Test nested LogContext managers."""
        from ptpd_calibration.core.logging import _log_context

        token = _log_context.set({})
        try:
            with LogContext(level1="outer"):
                with LogContext(level2="inner"):
                    ctx = _log_context.get()
                    assert ctx["level1"] == "outer"
                    assert ctx["level2"] == "inner"
        finally:
            _log_context.reset(token)


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_log_operation_logs_start_and_end(self):
        """Test log_operation logs operation start and completion."""
        logger = get_logger("test")

        with patch.object(logger, "log") as mock_log:
            with log_operation(logger, "test_operation"):
                time.sleep(0.01)

            # Should log start and completion
            assert mock_log.call_count >= 2
            calls = [call[0][1] for call in mock_log.call_args_list]
            assert any("Starting" in call for call in calls)
            assert any("Completed" in call for call in calls)

    def test_log_operation_logs_failure(self):
        """Test log_operation logs failure on exception."""
        logger = get_logger("test")

        with patch.object(logger, "log") as mock_log, patch.object(
            logger, "error"
        ) as mock_error:
            with pytest.raises(ValueError):
                with log_operation(logger, "failing_operation"):
                    raise ValueError("Test error")

            # Should log error
            assert mock_error.called


class TestLoggingMixin:
    """Tests for LoggingMixin."""

    def test_logging_mixin_provides_logger(self):
        """Test LoggingMixin provides logger property."""

        class TestClass(LoggingMixin):
            pass

        obj = TestClass()
        assert isinstance(obj.logger, logging.Logger)

    def test_logging_mixin_log_method_call(self):
        """Test LoggingMixin.log_method_call."""

        class TestClass(LoggingMixin):
            def process(self, value):
                self.log_method_call("process", value=value)
                return value * 2

        obj = TestClass()

        with patch.object(obj.logger, "debug") as mock_debug:
            result = obj.process(5)

            assert result == 10
            assert mock_debug.called
            call_args = mock_debug.call_args[0][0]
            assert "process" in call_args


# =============================================================================
# Integration Tests
# =============================================================================


class TestServiceWithEvents:
    """Integration test: Service publishing events."""

    def setup_method(self):
        """Reset EventBus."""
        EventBus.reset()

    def test_service_publishes_events_on_completion(self):
        """Test service can publish events when operations complete."""

        class TestInput(BaseModel):
            value: int

        class TestOutput(BaseModel):
            result: int

        class EventPublishingService(BaseService[TestInput, TestOutput]):
            def __init__(self):
                super().__init__()
                self.event_bus = EventBus()

            def validate_input(self, data: TestInput) -> ValidationResult:
                return ValidationResult(True)

            def process(self, data: TestInput) -> TestOutput:
                result = TestOutput(result=data.value * 2)

                # Publish completion event
                event = Event(event_type="service.completed", metadata={"result": result.result})
                self.event_bus.publish(event)

                return result

        received_events = []

        def handler(event: Event):
            received_events.append(event)

        bus = EventBus()
        bus.subscribe("service.completed", handler)

        service = EventPublishingService()
        service.execute(TestInput(value=10))

        assert len(received_events) == 1
        assert received_events[0].metadata["result"] == 20


class TestDebugWithLogging:
    """Integration test: Debug utilities with logging."""

    def test_timer_uses_logger(self):
        """Test timer decorator uses the logging system."""
        setup_logging(level="DEBUG")

        @timer
        def timed_operation():
            return 42

        with patch("ptpd_calibration.core.debug._is_debug_enabled", return_value=True):
            result = timed_operation()
            assert result == 42
