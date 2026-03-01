"""
Tests for error handling system.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ptpd_calibration.template.errors import (
    ConfigurationError,
    ErrorBoundary,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    NetworkError,
    NotFoundError,
    ResourceError,
    TemplateError,
    TimeoutError,
    ValidationError,
    create_gradio_error_wrapper,
    error_handler,
    retry_on_error,
)


class TestTemplateError:
    """Tests for TemplateError base class."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = TemplateError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.error_code == "TEMPLATE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_error_with_details(self) -> None:
        """Test error with additional details."""
        error = TemplateError(
            "Processing failed",
            error_code="PROC_ERROR",
            details={"item_id": 123, "reason": "invalid format"},
        )
        assert error.error_code == "PROC_ERROR"
        assert error.details["item_id"] == 123

    def test_error_with_cause(self) -> None:
        """Test error with original cause."""
        original = ValueError("Invalid value")
        error = TemplateError("Wrapper error", cause=original)
        assert error.cause is original

    def test_error_with_context(self) -> None:
        """Test error with context information."""
        context = ErrorContext(
            operation="data_processing",
            component="analyzer",
        )
        error = TemplateError("Error occurred", context=context)
        assert error.context.operation == "data_processing"
        assert error.context.component == "analyzer"

    def test_to_dict(self) -> None:
        """Test error serialization to dictionary."""
        error = TemplateError(
            "Test error",
            error_code="TEST",
            details={"key": "value"},
        )
        result = error.to_dict()
        assert result["error_code"] == "TEST"
        assert result["message"] == "Test error"
        assert result["details"]["key"] == "value"
        assert "timestamp" in result


class TestSpecializedErrors:
    """Tests for specialized error types."""

    def test_validation_error(self) -> None:
        """Test validation error."""
        error = ValidationError(
            "Invalid email format",
            field="email",
            value="not-an-email",
        )
        assert error.error_code == "VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.LOW
        assert error.http_status == 400
        assert error.details["field"] == "email"

    def test_configuration_error(self) -> None:
        """Test configuration error."""
        error = ConfigurationError("Missing required setting")
        assert error.error_code == "CONFIG_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = TimeoutError(
            "Request timed out",
            operation="api_call",
            timeout_seconds=30.0,
        )
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.http_status == 504
        assert error.details["timeout_seconds"] == 30.0
        assert error.recoverable is True

    def test_resource_error(self) -> None:
        """Test resource error."""
        error = ResourceError(
            "Memory limit exceeded",
            resource_type="memory",
            limit="4GB",
            current="4.5GB",
        )
        assert error.error_code == "RESOURCE_ERROR"
        assert error.details["resource_type"] == "memory"

    def test_network_error(self) -> None:
        """Test network error."""
        error = NetworkError("Connection refused")
        assert error.error_code == "NETWORK_ERROR"
        assert error.category == ErrorCategory.NETWORK

    def test_not_found_error(self) -> None:
        """Test not found error."""
        error = NotFoundError("Item not found")
        assert error.error_code == "NOT_FOUND"
        assert error.http_status == 404


class TestErrorBoundary:
    """Tests for ErrorBoundary."""

    def test_protect_success(self, error_boundary: ErrorBoundary) -> None:
        """Test protection with successful execution."""
        result = None
        with error_boundary.protect("test_operation"):
            result = 42
        assert result == 42

    def test_protect_handles_error(self, error_boundary: ErrorBoundary) -> None:
        """Test protection handles errors."""
        with error_boundary.protect("test_operation"):
            raise ValueError("Test error")
        # Should not raise

    def test_protect_reraise(self, strict_error_boundary: ErrorBoundary) -> None:
        """Test protection with reraise."""
        with pytest.raises(TemplateError):
            with strict_error_boundary.protect("test_operation"):
                raise ValueError("Test error")

    def test_wrap_function(self, error_boundary: ErrorBoundary) -> None:
        """Test wrapping a function."""

        @error_boundary.wrap
        def divide(a: int, b: int) -> float:
            return a / b

        # Success case
        assert divide(10, 2) == 5.0

        # Error case - returns default
        result = divide(10, 0)
        assert result is None

    def test_wrap_async_function(self, error_boundary: ErrorBoundary) -> None:
        """Test wrapping an async function."""

        @error_boundary.wrap
        async def async_divide(a: int, b: int) -> float:
            return a / b

        # Success case
        result = asyncio.get_event_loop().run_until_complete(async_divide(10, 2))
        assert result == 5.0

    def test_on_error_callback(self) -> None:
        """Test error callback."""
        errors_caught: list[TemplateError] = []

        boundary = ErrorBoundary(
            component="test",
            on_error=lambda e: errors_caught.append(e),
        )

        with boundary.protect("operation"):
            raise ValueError("Test")

        assert len(errors_caught) == 1
        assert "Test" in str(errors_caught[0])


class TestErrorHandler:
    """Tests for error_handler decorator."""

    def test_error_handler_success(self) -> None:
        """Test error handler with successful function."""

        @error_handler(component="test")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    def test_error_handler_with_error(self) -> None:
        """Test error handler catching error."""

        @error_handler(component="test", default_return=-1)
        def divide(a: int, b: int) -> int:
            return a // b

        assert divide(10, 2) == 5
        assert divide(10, 0) == -1


class TestRetryOnError:
    """Tests for retry_on_error decorator."""

    def test_retry_success_first_try(self) -> None:
        """Test retry with success on first attempt."""
        call_count = 0

        @retry_on_error(max_retries=3)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self) -> None:
        """Test retry with success after failures."""
        call_count = 0

        @retry_on_error(max_retries=3, retry_delay=0.01)
        def succeed_on_third() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TemplateError("Temporary failure")
            return "success"

        result = succeed_on_third()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self) -> None:
        """Test retry with all attempts exhausted."""

        @retry_on_error(max_retries=2, retry_delay=0.01)
        def always_fail() -> str:
            raise TemplateError("Always fails")

        with pytest.raises(TemplateError):
            always_fail()

    def test_retry_async(self) -> None:
        """Test retry with async function."""
        call_count = 0

        @retry_on_error(max_retries=3, retry_delay=0.01)
        async def async_succeed_on_second() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TemplateError("Temporary failure")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(async_succeed_on_second())
        assert result == "success"
        assert call_count == 2

    def test_retry_on_callback(self) -> None:
        """Test retry callback is called."""
        retries: list[tuple[Exception, int]] = []

        @retry_on_error(
            max_retries=2,
            retry_delay=0.01,
            on_retry=lambda e, n: retries.append((e, n)),
        )
        def fail_twice() -> str:
            if len(retries) < 2:
                raise TemplateError("Failure")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert len(retries) == 2


class TestGradioErrorWrapper:
    """Tests for Gradio error wrapper."""

    def test_gradio_wrapper_success(self) -> None:
        """Test wrapper with successful function."""

        @create_gradio_error_wrapper("test_component")
        def process(data: str) -> str:
            return f"processed: {data}"

        result = process("input")
        assert result == "processed: input"

    def test_gradio_wrapper_error(self) -> None:
        """Test wrapper with error returns user-friendly message."""

        @create_gradio_error_wrapper("test_component")
        def fail(data: str) -> str:
            raise TemplateError("Processing failed", user_message="Please try again")

        result = fail("input")
        assert result == (None, "Error: Please try again")

    def test_gradio_wrapper_unexpected_error(self) -> None:
        """Test wrapper with unexpected error."""

        @create_gradio_error_wrapper("test_component")
        def crash(data: str) -> str:
            raise RuntimeError("Unexpected")

        result = crash("input")
        assert result == (None, "An unexpected error occurred. Please try again.")
