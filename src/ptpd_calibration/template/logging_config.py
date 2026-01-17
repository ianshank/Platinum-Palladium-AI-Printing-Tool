"""
Centralized Logging Infrastructure

Provides structured logging with:
- JSON and text formats
- File rotation
- Context management
- Request tracing
- Performance logging
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Generator, Optional, TypeVar

from pydantic import BaseModel

# Context variables for request tracing
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_operation: ContextVar[Optional[str]] = ContextVar("operation", default=None)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class LogRecord(BaseModel):
    """Structured log record."""

    timestamp: str
    level: str
    logger: str
    message: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: dict[str, Any] = {}
    exception: Optional[str] = None
    traceback: Optional[str] = None


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record = LogRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            request_id=_request_id.get(),
            session_id=_session_id.get(),
            user_id=_user_id.get(),
            operation=_operation.get(),
        )

        # Add extra fields
        if hasattr(record, "duration_ms"):
            log_record.duration_ms = record.duration_ms
        if hasattr(record, "extra_data"):
            log_record.extra = record.extra_data

        # Add exception info
        if record.exc_info:
            log_record.exception = str(record.exc_info[1])
            log_record.traceback = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_record.model_dump(exclude_none=True))


class ConsoleFormatter(logging.Formatter):
    """Colorized console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Build context string
        context_parts = []
        if _request_id.get():
            context_parts.append(f"req={_request_id.get()[:8]}")
        if _operation.get():
            context_parts.append(f"op={_operation.get()}")

        context = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format duration if present
        duration = ""
        if hasattr(record, "duration_ms"):
            duration = f" ({record.duration_ms:.2f}ms)"

        formatted = (
            f"{color}{record.levelname:8}{reset} | "
            f"{record.name:30} | "
            f"{record.getMessage()}{context}{duration}"
        )

        if record.exc_info:
            formatted += "\n" + "".join(
                traceback.format_exception(*record.exc_info)
            )

        return formatted


class StructuredLogger:
    """
    Enhanced logger with structured logging support.

    Usage:
        logger = StructuredLogger("my_module")
        logger.info("Processing started", item_count=42)
        with logger.timed("expensive_operation"):
            do_something()
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """Initialize structured logger."""
        self._logger = logger or logging.getLogger(name)
        self.name = name

    # Reserved parameter names that should not be passed through as kwargs
    _RESERVED_PARAMS = frozenset({"level", "exc_info", "msg", "message"})

    def _filter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter out reserved parameter names from kwargs."""
        return {k: v for k, v in kwargs.items() if k not in self._RESERVED_PARAMS}

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        extra_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Internal log method with extra data."""
        extra = {"extra_data": extra_data} if extra_data else {}
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_data=self._filter_kwargs(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra_data=self._filter_kwargs(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, extra_data=self._filter_kwargs(kwargs))

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, extra_data=self._filter_kwargs(kwargs))

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, extra_data=self._filter_kwargs(kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, extra_data=self._filter_kwargs(kwargs))

    @contextmanager
    def timed(
        self,
        operation: str,
        log_start: bool = True,
        log_end: bool = True,
        level: int = logging.INFO,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager for timing operations.

        Args:
            operation: Name of the operation being timed
            log_start: Whether to log when operation starts
            log_end: Whether to log when operation ends
            level: Log level for messages

        Yields:
            Dictionary to store operation metadata

        Usage:
            with logger.timed("database_query") as ctx:
                result = db.query(...)
                ctx["row_count"] = len(result)
        """
        context: dict[str, Any] = {"operation": operation}
        start_time = time.perf_counter()

        if log_start:
            self._log(level, f"Starting: {operation}")

        old_operation = _operation.get()
        _operation.set(operation)

        try:
            yield context
            context["success"] = True
        except Exception as e:
            context["success"] = False
            context["error"] = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            context["duration_ms"] = duration_ms
            _operation.set(old_operation)

            if log_end:
                status = "Completed" if context.get("success", False) else "Failed"
                record = logging.LogRecord(
                    name=self.name,
                    level=level,
                    pathname="",
                    lineno=0,
                    msg=f"{status}: {operation}",
                    args=(),
                    exc_info=None,
                )
                record.duration_ms = duration_ms
                self._logger.handle(record)


class LogContext:
    """
    Context manager for setting log context variables.

    Usage:
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")  # Includes context
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        auto_request_id: bool = False,
    ):
        """Initialize log context."""
        self.request_id = request_id or (str(uuid.uuid4()) if auto_request_id else None)
        self.session_id = session_id
        self.user_id = user_id
        self.operation = operation
        self._tokens: list[Any] = []

    def __enter__(self) -> "LogContext":
        """Enter context and set variables."""
        if self.request_id:
            self._tokens.append(_request_id.set(self.request_id))
        if self.session_id:
            self._tokens.append(_session_id.set(self.session_id))
        if self.user_id:
            self._tokens.append(_user_id.set(self.user_id))
        if self.operation:
            self._tokens.append(_operation.set(self.operation))
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and reset variables."""
        for token in reversed(self._tokens):
            if hasattr(token, "var"):
                token.var.reset(token)


def logged(
    level: int = logging.INFO,
    log_args: bool = True,
    log_result: bool = False,
    include_timing: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for automatic function logging.

    Args:
        level: Log level for messages
        log_args: Whether to log function arguments
        log_result: Whether to log return value
        include_timing: Whether to log execution time

    Usage:
        @logged(level=logging.DEBUG, log_result=True)
        def process_data(data: list) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        logger = StructuredLogger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__qualname__

            # Build log data
            log_data: dict[str, Any] = {"function": func_name}
            if log_args:
                # Truncate large args for logging
                log_data["args_count"] = len(args)
                log_data["kwargs_keys"] = list(kwargs.keys())

            if include_timing:
                with logger.timed(func_name, log_start=True, log_end=True, level=level) as ctx:
                    result = func(*args, **kwargs)
                    if log_result:
                        ctx["has_result"] = result is not None
                    return result
            else:
                logger._log(level, f"Calling: {func_name}", **log_data)
                result = func(*args, **kwargs)
                if log_result:
                    logger._log(level, f"Returned: {func_name}", has_result=result is not None)
                return result

        return wrapper  # type: ignore

    return decorator


# Logger registry
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    json_format: bool = False,
    file_enabled: bool = False,
    file_path: Optional[Path] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console_enabled: bool = True,
    root_logger: bool = True,
) -> None:
    """
    Configure logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (ignored if json_format=True)
        json_format: Use JSON structured logging
        file_enabled: Enable file logging
        file_path: Path to log file
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        console_enabled: Enable console output
        root_logger: Also configure root logger
    """
    # Convert level string to int
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create handlers
    handlers: list[logging.Handler] = []

    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())

        handlers.append(console_handler)

    if file_enabled and file_path:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)

        # Always use JSON format for file logs
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)

    # Configure root logger
    if root_logger:
        root = logging.getLogger()
        root.setLevel(log_level)

        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Add new handlers
        for handler in handlers:
            root.addHandler(handler)

    # Log configuration
    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        level=level,
        json_format=json_format,
        file_enabled=file_enabled,
        console_enabled=console_enabled,
    )


def setup_logging_from_config() -> None:
    """Configure logging from template configuration."""
    try:
        from ptpd_calibration.template.config import get_template_config

        config = get_template_config()
        setup_logging(
            level=config.logging.level,
            format_string=config.logging.format,
            json_format=config.logging.json_format,
            file_enabled=config.logging.file_enabled,
            file_path=config.logging.file_path,
            max_bytes=config.logging.max_bytes,
            backup_count=config.logging.backup_count,
            console_enabled=config.logging.console_enabled,
        )
    except ImportError:
        # Fallback to basic setup
        setup_logging()
