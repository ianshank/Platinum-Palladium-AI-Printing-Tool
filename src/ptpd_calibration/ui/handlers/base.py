"""
Base handler utilities for UI event handling.

Provides common patterns for logging, error handling,
and state management in Gradio event handlers.
"""

import logging
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class HandlerLogger:
    """Structured logging for event handlers."""

    def __init__(self, name: str) -> None:
        """Initialize handler logger.

        Args:
            name: Logger name.
        """
        self.logger = logging.getLogger(name)

    def log_handler_start(
        self, handler_name: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log handler start.

        Args:
            handler_name: Name of the handler.
            context: Optional context dictionary.
        """
        ctx_str = f" with {context}" if context else ""
        self.logger.debug(f"Handler '{handler_name}' starting{ctx_str}")

    def log_handler_complete(
        self, handler_name: str, result: Optional[Any] = None
    ) -> None:
        """Log handler completion.

        Args:
            handler_name: Name of the handler.
            result: Optional result value.
        """
        res_str = f": {result}" if result else ""
        self.logger.debug(f"Handler '{handler_name}' completed{res_str}")

    def log_handler_error(
        self, handler_name: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log handler error.

        Args:
            handler_name: Name of the handler.
            error: The exception that occurred.
            context: Optional context dictionary.
        """
        ctx_str = f" with {context}" if context else ""
        self.logger.error(
            f"Handler '{handler_name}' failed{ctx_str}: {str(error)}", exc_info=True
        )


def safe_handler(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for safe handler execution with logging.

    Args:
        func: The handler function to wrap.

    Returns:
        Wrapped handler with error handling.
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise

    return wrapper


__all__ = ["HandlerLogger", "safe_handler"]
