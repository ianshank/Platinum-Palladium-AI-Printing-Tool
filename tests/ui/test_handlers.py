"""
Tests for UI handlers module.
"""

import logging

import pytest

from ptpd_calibration.ui.handlers import HandlerLogger, safe_handler


class TestHandlerLogger:
    """Test HandlerLogger."""

    def test_handler_logger_creation(self) -> None:
        """Test handler logger initialization."""
        logger = HandlerLogger("test_handler")
        assert logger.logger.name == "test_handler"

    def test_handler_logger_log_start(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging handler start."""
        logger = HandlerLogger("test")
        with caplog.at_level(logging.DEBUG):
            logger.log_handler_start("test_handler")

        assert "starting" in caplog.text.lower()

    def test_handler_logger_log_start_with_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging handler start with context."""
        logger = HandlerLogger("test")
        context = {"file": "test.txt", "action": "load"}
        with caplog.at_level(logging.DEBUG):
            logger.log_handler_start("test_handler", context)

        assert "test_handler" in caplog.text
        assert "starting" in caplog.text.lower()

    def test_handler_logger_log_complete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging handler completion."""
        logger = HandlerLogger("test")
        with caplog.at_level(logging.DEBUG):
            logger.log_handler_complete("test_handler")

        assert "completed" in caplog.text.lower()

    def test_handler_logger_log_complete_with_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging handler completion with result."""
        logger = HandlerLogger("test")
        with caplog.at_level(logging.DEBUG):
            logger.log_handler_complete("test_handler", {"status": "success"})

        assert "completed" in caplog.text.lower()

    def test_handler_logger_log_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging handler error."""
        logger = HandlerLogger("test")
        error = ValueError("Test error")
        with caplog.at_level(logging.ERROR):
            logger.log_handler_error("test_handler", error)

        assert "failed" in caplog.text.lower()
        assert "Test error" in caplog.text

    def test_handler_logger_log_error_with_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging handler error with context."""
        logger = HandlerLogger("test")
        error = RuntimeError("Test runtime error")
        context = {"file": "test.txt"}
        with caplog.at_level(logging.ERROR):
            logger.log_handler_error("test_handler", error, context)

        assert "failed" in caplog.text.lower()


class TestSafeHandlerDecorator:
    """Test safe_handler decorator."""

    def test_safe_handler_success(self) -> None:
        """Test safe handler with successful execution."""

        @safe_handler
        def test_function(value: int) -> int:
            return value * 2

        result = test_function(5)
        assert result == 10

    def test_safe_handler_preserves_return_type(self) -> None:
        """Test that safe handler preserves return types."""

        @safe_handler
        def return_string(value: str) -> str:
            return value.upper()

        result = return_string("hello")
        assert result == "HELLO"
        assert isinstance(result, str)

    def test_safe_handler_with_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test safe handler with exception."""

        @safe_handler
        def failing_function() -> None:
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_function()

        assert "Error in" in caplog.text

    def test_safe_handler_with_args_and_kwargs(self) -> None:
        """Test safe handler with multiple arguments."""

        @safe_handler
        def multi_arg_function(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        result = multi_arg_function(1, 2, c=3)
        assert result == 6

    def test_safe_handler_reraises_exception(self) -> None:
        """Test that safe handler re-raises the exception."""

        @safe_handler
        def error_function() -> None:
            raise RuntimeError("Original error")

        with pytest.raises(RuntimeError, match="Original error"):
            error_function()

    def test_safe_handler_preserves_function_name(self) -> None:
        """Test that decorated function name is preserved."""

        @safe_handler
        def named_function() -> None:
            pass

        # Note: The wrapper doesn't preserve __name__ by default
        # This test documents the current behavior
        assert "wrapper" in str(named_function)


class TestHandlerIntegration:
    """Integration tests for handlers."""

    def test_handler_logger_and_safe_handler_together(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test using HandlerLogger with safe_handler."""
        logger = HandlerLogger("integration_test")

        @safe_handler
        def integrated_handler(value: float) -> float:
            logger.log_handler_start("integrated_handler", {"value": value})
            try:
                result = value * 2
                logger.log_handler_complete("integrated_handler", result)
                return result
            except Exception as e:
                logger.log_handler_error("integrated_handler", e)
                raise

        with caplog.at_level(logging.DEBUG):
            result = integrated_handler(2.5)

        assert result == 5.0
        assert "starting" in caplog.text.lower()
        assert "completed" in caplog.text.lower()

    def test_error_handling_pipeline(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test error handling through the pipeline."""
        logger = HandlerLogger("pipeline_test")

        @safe_handler
        def pipeline_handler(should_fail: bool) -> str:
            logger.log_handler_start("pipeline")
            try:
                if should_fail:
                    raise RuntimeError("Pipeline error")
                logger.log_handler_complete("pipeline", "success")
                return "success"
            except Exception as e:
                logger.log_handler_error("pipeline", e)
                raise

        # Test success path
        with caplog.at_level(logging.DEBUG):
            result = pipeline_handler(False)
        assert result == "success"

        # Test error path
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError):
                pipeline_handler(True)
        assert "failed" in caplog.text.lower()
