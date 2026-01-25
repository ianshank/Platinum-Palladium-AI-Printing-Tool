"""
Error Handling System

Provides centralized error handling with:
- Custom exception hierarchy
- Error boundaries for Gradio/FastAPI
- Automatic error logging
- User-friendly error messages
- Retry logic
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"  # User can continue
    MEDIUM = "medium"  # Feature degraded
    HIGH = "high"  # Feature unavailable
    CRITICAL = "critical"  # Application error


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INTERNAL = "internal"
    EXTERNAL = "external"
    USER_INPUT = "user_input"


@dataclass
class ErrorContext:
    """Context information for errors."""

    operation: str
    component: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TemplateError(Exception):
    """
    Base exception for template errors.

    Provides structured error information with:
    - Error code for programmatic handling
    - User-friendly message
    - Severity and category classification
    - Context for debugging
    """

    error_code: str = "TEMPLATE_ERROR"
    default_message: str = "An error occurred"
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.INTERNAL
    http_status: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
        context: ErrorContext | None = None,
        user_message: str | None = None,
        recoverable: bool = True,
    ):
        """
        Initialize template error.

        Args:
            message: Technical error message
            error_code: Override default error code
            details: Additional error details
            cause: Original exception that caused this error
            context: Error context information
            user_message: User-friendly message (defaults to message)
            recoverable: Whether the operation can be retried
        """
        self.message = message or self.default_message
        self.error_code = error_code or self.__class__.error_code
        self.details = details or {}
        self.cause = cause
        self.context = context
        self.user_message = user_message or self.message
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        result = {
            "error_code": self.error_code,
            "message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.details:
            result["details"] = self.details

        if self.context:
            result["context"] = {
                "operation": self.context.operation,
                "component": self.context.component,
            }

        return result

    def log(self, include_traceback: bool = True) -> None:
        """Log this error with appropriate level."""
        log_method = {
            ErrorSeverity.LOW: logger.warning,
            ErrorSeverity.MEDIUM: logger.error,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
        }.get(self.severity, logger.error)

        log_method(
            f"{self.error_code}: {self.message}",
            exc_info=include_traceback and self.cause is not None,
            details=self.details,
            context=self.context.operation if self.context else None,
        )


class ConfigurationError(TemplateError):
    """Configuration-related errors."""

    error_code = "CONFIG_ERROR"
    default_message = "Configuration error"
    severity = ErrorSeverity.HIGH
    category = ErrorCategory.CONFIGURATION
    http_status = 500


class ValidationError(TemplateError):
    """Input validation errors."""

    error_code = "VALIDATION_ERROR"
    default_message = "Validation failed"
    severity = ErrorSeverity.LOW
    category = ErrorCategory.VALIDATION
    http_status = 400

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: Any | None = None,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize validation error."""
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate for safety
        if constraints:
            details["constraints"] = constraints

        super().__init__(message, details=details, **kwargs)


class TimeoutError(TemplateError):
    """Operation timeout errors."""

    error_code = "TIMEOUT_ERROR"
    default_message = "Operation timed out"
    severity = ErrorSeverity.MEDIUM
    category = ErrorCategory.TIMEOUT
    http_status = 504

    def __init__(
        self,
        message: str | None = None,
        *,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ):
        """Initialize timeout error."""
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(message, details=details, recoverable=True, **kwargs)


class ResourceError(TemplateError):
    """Resource-related errors (memory, disk, etc.)."""

    error_code = "RESOURCE_ERROR"
    default_message = "Resource error"
    severity = ErrorSeverity.HIGH
    category = ErrorCategory.RESOURCE
    http_status = 503

    def __init__(
        self,
        message: str | None = None,
        *,
        resource_type: str | None = None,
        limit: Any | None = None,
        current: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize resource error."""
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if limit is not None:
            details["limit"] = str(limit)
        if current is not None:
            details["current"] = str(current)

        super().__init__(message, details=details, **kwargs)


class NetworkError(TemplateError):
    """Network-related errors."""

    error_code = "NETWORK_ERROR"
    default_message = "Network error"
    severity = ErrorSeverity.MEDIUM
    category = ErrorCategory.NETWORK
    http_status = 502


class NotFoundError(TemplateError):
    """Resource not found errors."""

    error_code = "NOT_FOUND"
    default_message = "Resource not found"
    severity = ErrorSeverity.LOW
    category = ErrorCategory.NOT_FOUND
    http_status = 404


class AuthenticationError(TemplateError):
    """Authentication errors."""

    error_code = "AUTH_ERROR"
    default_message = "Authentication failed"
    severity = ErrorSeverity.MEDIUM
    category = ErrorCategory.AUTHENTICATION
    http_status = 401


class ExternalServiceError(TemplateError):
    """External service errors (API calls, etc.)."""

    error_code = "EXTERNAL_ERROR"
    default_message = "External service error"
    severity = ErrorSeverity.MEDIUM
    category = ErrorCategory.EXTERNAL
    http_status = 502


class ErrorResponse(BaseModel):
    """Standardized error response for API."""

    error_code: str
    message: str
    severity: str
    category: str
    recoverable: bool
    timestamp: str
    details: dict[str, Any] | None = None
    trace_id: str | None = None


class ErrorBoundary:
    """
    Error boundary for catching and handling errors.

    Usage:
        boundary = ErrorBoundary(
            component="image_processor",
            on_error=lambda e: notify_user(e)
        )

        with boundary.protect():
            process_image(data)

        # Or as decorator
        @boundary.wrap
        def process_image(data):
            ...
    """

    def __init__(
        self,
        component: str,
        *,
        default_return: Any = None,
        reraise: bool = False,
        on_error: Callable[[TemplateError], None] | None = None,
        suppress_exceptions: list[type[Exception]] | None = None,
        convert_exceptions: bool = True,
        log_errors: bool = True,
    ):
        """
        Initialize error boundary.

        Args:
            component: Component name for context
            default_return: Value to return on error (if not reraising)
            reraise: Whether to reraise errors after handling
            on_error: Callback for error handling
            suppress_exceptions: Exception types to suppress
            convert_exceptions: Convert non-TemplateError to TemplateError
            log_errors: Whether to log errors
        """
        self.component = component
        self.default_return = default_return
        self.reraise = reraise
        self.on_error = on_error
        self.suppress_exceptions = suppress_exceptions or []
        self.convert_exceptions = convert_exceptions
        self.log_errors = log_errors

    def _handle_error(
        self,
        error: Exception,
        operation: str = "unknown",
    ) -> TemplateError:
        """Handle and optionally convert an error."""
        # Already a TemplateError
        if isinstance(error, TemplateError):
            template_error = error
        # Convert standard exceptions
        elif self.convert_exceptions:
            template_error = TemplateError(
                str(error),
                cause=error,
                context=ErrorContext(operation=operation, component=self.component),
            )
        else:
            raise error

        # Add context if not present
        if not template_error.context:
            template_error.context = ErrorContext(
                operation=operation,
                component=self.component,
            )

        # Log error
        if self.log_errors:
            template_error.log()

        # Call error handler
        if self.on_error:
            try:
                self.on_error(template_error)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")

        return template_error

    @contextmanager
    def protect(
        self,
        operation: str = "operation",
    ) -> Generator[None, None, None]:
        """
        Context manager for protected code execution.

        Args:
            operation: Name of the operation being protected

        Yields:
            None

        Raises:
            TemplateError: If reraise=True and an error occurs
        """
        try:
            yield
        except Exception as e:
            # Check if exception should be suppressed
            if any(isinstance(e, exc_type) for exc_type in self.suppress_exceptions):
                return

            template_error = self._handle_error(e, operation)

            if self.reraise:
                raise template_error from e

    def wrap(self, func: F) -> F:
        """
        Decorator for wrapping functions in error boundary.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if any(isinstance(e, exc_type) for exc_type in self.suppress_exceptions):
                    return self.default_return

                template_error = self._handle_error(e, func.__qualname__)

                if self.reraise:
                    raise template_error from e

                return self.default_return

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if any(isinstance(e, exc_type) for exc_type in self.suppress_exceptions):
                    return self.default_return

                template_error = self._handle_error(e, func.__qualname__)

                if self.reraise:
                    raise template_error from e

                return self.default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


def error_handler(
    *,
    component: str = "unknown",
    default_return: Any = None,
    reraise: bool = False,
    log: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for error handling.

    Args:
        component: Component name for context
        default_return: Value to return on error
        reraise: Whether to reraise after handling
        log: Whether to log errors

    Usage:
        @error_handler(component="processor", default_return=[])
        def process_items(items: list) -> list:
            ...
    """
    boundary = ErrorBoundary(
        component=component,
        default_return=default_return,
        reraise=reraise,
        log_errors=log,
    )
    return boundary.wrap


def retry_on_error(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    retry_exceptions: list[type[Exception]] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for automatic retry on errors.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        exponential_backoff: Use exponential backoff
        retry_exceptions: Exception types to retry (default: all TemplateError)
        on_retry: Callback on retry (receives exception and attempt number)

    Usage:
        @retry_on_error(max_retries=3, exponential_backoff=True)
        def fetch_data() -> dict:
            ...
    """
    retry_types = tuple(retry_exceptions or [TemplateError, ConnectionError, TimeoutError])

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_types as e:
                    last_error = e

                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)

                        if on_retry:
                            on_retry(e, attempt + 1)
                        else:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} after error: {e}",
                                delay_seconds=delay,
                            )

                        import time
                        time.sleep(delay)
                    else:
                        raise

            raise last_error  # type: ignore

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_types as e:
                    last_error = e

                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)

                        if on_retry:
                            on_retry(e, attempt + 1)
                        else:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} after error: {e}",
                                delay_seconds=delay,
                            )

                        await asyncio.sleep(delay)
                    else:
                        raise

            raise last_error  # type: ignore

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# FastAPI exception handler integration
def create_fastapi_exception_handlers() -> dict[type[Exception], Callable]:
    """
    Create FastAPI exception handlers.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()

        for exc_type, handler in create_fastapi_exception_handlers().items():
            app.add_exception_handler(exc_type, handler)
    """
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
    except ImportError:
        return {}

    async def template_error_handler(
        request: Request,
        exc: TemplateError,
    ) -> JSONResponse:
        """Handle TemplateError exceptions."""
        exc.log()

        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    async def generic_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle generic exceptions."""
        logger.exception(f"Unhandled exception: {exc}")

        error = TemplateError(
            "An unexpected error occurred",
            cause=exc,
            user_message="An unexpected error occurred. Please try again.",
        )

        return JSONResponse(
            status_code=500,
            content=error.to_dict(),
        )

    return {
        TemplateError: template_error_handler,
        Exception: generic_error_handler,
    }


# Gradio error handling
def create_gradio_error_wrapper(
    component: str,
    user_friendly: bool = True,
) -> Callable[[F], F]:
    """
    Create error wrapper for Gradio event handlers.

    Args:
        component: Component name for context
        user_friendly: Return user-friendly error messages

    Usage:
        @create_gradio_error_wrapper("image_upload")
        def handle_upload(file):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except TemplateError as e:
                e.log()
                if user_friendly:
                    # Return tuple with error message for Gradio
                    return (None, f"Error: {e.user_message}")
                raise
            except Exception as e:
                logger.exception(f"Error in {component}: {e}")
                if user_friendly:
                    return (None, "An unexpected error occurred. Please try again.")
                raise TemplateError(
                    str(e),
                    cause=e,
                    context=ErrorContext(
                        operation=func.__qualname__,
                        component=component,
                    ),
                )

        return wrapper  # type: ignore

    return decorator
