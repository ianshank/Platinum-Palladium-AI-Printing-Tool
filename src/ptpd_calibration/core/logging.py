"""
Centralized logging configuration for PTPD Calibration System.

Provides structured logging with JSON formatting support, context managers
for adding context to log messages, and lazy configuration.

Usage:
    from ptpd_calibration.core.logging import get_logger, setup_logging

    # Setup logging (typically at application startup)
    setup_logging(level="DEBUG", json_format=True)

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"record_id": 123})
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Context variable for request/operation tracking
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs logs as JSON objects with consistent fields for easy parsing
    by log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted string.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context variables
        ctx = _log_context.get()
        if ctx:
            log_data["context"] = ctx

        # Add extra fields from record
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)

        # Check for common extra attributes
        for attr in ("request_id", "user_id", "operation", "duration"):
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            log_data["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development.

    Adds ANSI color codes to log output for better readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors.

        Args:
            record: The log record to format.

        Returns:
            Colored formatted string.
        """
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


_logging_configured = False


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
    json_format: bool = False,
    colored: bool = True,
) -> None:
    """Configure application logging.

    Should be called once at application startup. Subsequent calls
    will update the configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config value.
        log_file: Optional file path for log output.
        json_format: Use JSON formatting for structured logs.
        colored: Use colored output in console (ignored if json_format=True).
    """
    global _logging_configured

    # Import here to avoid circular imports
    from ptpd_calibration.config import get_settings

    settings = get_settings()
    level = level or settings.log_level

    # Root logger for our package
    root_logger = logging.getLogger("ptpd_calibration")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers on reconfiguration
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    elif colored and sys.stdout.isatty():
        console_handler.setFormatter(
            ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    _logging_configured = True
    root_logger.debug(f"Logging configured: level={level}, file={log_file}, json={json_format}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Automatically ensures logging is configured before returning.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Operation completed", extra={"duration": 1.5})
    """
    global _logging_configured

    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()

    # Normalize name to be under our package
    if not name.startswith("ptpd_calibration"):
        name = f"ptpd_calibration.{name}"

    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to log messages.

    Useful for adding request IDs, user IDs, or other contextual
    information to all log messages within a scope.

    Example:
        with LogContext(request_id="abc-123", user="john"):
            logger.info("Processing request")  # Includes context
    """

    def __init__(self, **context: Any):
        """Initialize with context key-value pairs.

        Args:
            **context: Key-value pairs to add to log context.
        """
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        """Enter context, adding to context variable."""
        current = _log_context.get()
        new_context = {**current, **self.context}
        self._token = _log_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context, restoring previous state."""
        if self._token is not None:
            _log_context.reset(self._token)


def log_operation(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
):
    """Context manager for logging operation start/end with timing.

    Args:
        logger: Logger to use.
        operation: Operation name for logging.
        level: Log level for messages.

    Example:
        with log_operation(logger, "curve_generation"):
            # ... do work ...
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def _log_op():
        start = time.perf_counter()
        logger.log(level, f"Starting: {operation}")
        try:
            yield
            elapsed = time.perf_counter() - start
            logger.log(
                level, f"Completed: {operation}", extra={"duration_seconds": round(elapsed, 4)}
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(
                f"Failed: {operation}",
                extra={
                    "duration_seconds": round(elapsed, 4),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    return _log_op()


class LoggingMixin:
    """Mixin class to add logging capabilities to any class.

    Provides a `logger` property that returns a logger named
    after the class.

    Example:
        class MyService(LoggingMixin):
            def process(self):
                self.logger.info("Processing...")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__)

    def log_method_call(
        self,
        method_name: str,
        **params: Any,
    ) -> None:
        """Log a method call with parameters.

        Args:
            method_name: Name of the method being called.
            **params: Parameters to log.
        """
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        self.logger.debug(f"{self.__class__.__name__}.{method_name}({params_str})")
