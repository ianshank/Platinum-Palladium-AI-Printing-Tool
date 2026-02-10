"""
Debugging utilities for PTPD Calibration System.

Provides decorators and utilities for debugging, profiling, and
tracing code execution. All utilities respect the debug setting
and have zero overhead when disabled.

Usage:
    from ptpd_calibration.core.debug import timer, trace, debug_context

    @timer
    def expensive_operation():
        # ... code ...

    @trace
    def complex_function(x, y):
        # Entry and exit will be logged with arguments
        return x + y

    with debug_context("curve_fitting", points=256):
        # ... operations logged as a group ...
"""

import functools
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _is_debug_enabled() -> bool:
    """Check if debug mode is enabled in settings."""
    try:
        from ptpd_calibration.config import get_settings

        return get_settings().debug
    except Exception:
        return False


def timer(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to measure and log function execution time.

    Only logs when debug mode is enabled, otherwise has zero overhead.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with timing.

    Example:
        @timer
        def process_image(image):
            # ... processing ...
            return result
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if not _is_debug_enabled():
            return func(*args, **kwargs)

        func_name = f"{func.__module__}.{func.__name__}"
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(
                f"TIMER | {func_name} | {elapsed:.4f}s",
                extra={"function": func_name, "duration": elapsed},
            )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.debug(
                f"TIMER | {func_name} | {elapsed:.4f}s | FAILED: {e}",
                extra={"function": func_name, "duration": elapsed, "error": str(e)},
            )
            raise

    return wrapper


def trace(
    func: Callable[P, T] | None = None,
    *,
    max_arg_length: int = 100,
    log_result: bool = True,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log function entry/exit with arguments.

    Only logs when debug mode is enabled.

    Args:
        func: Function to wrap (when used without arguments).
        max_arg_length: Maximum length for argument string representation.
        log_result: Whether to log the return value.

    Returns:
        Wrapped function with tracing.

    Example:
        @trace
        def process(data):
            return data * 2

        @trace(max_arg_length=50, log_result=False)
        def sensitive_operation(password):
            # password won't be logged
            pass
    """

    def _truncate(val: Any, max_len: int) -> str:
        """Truncate string representation for readability."""
        s = repr(val)
        return s if len(s) <= max_len else s[:max_len] + "..."

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not _is_debug_enabled():
                return fn(*args, **kwargs)

            func_name = f"{fn.__module__}.{fn.__name__}"

            # Format arguments
            args_repr = [_truncate(a, max_arg_length) for a in args]
            kwargs_repr = [f"{k}={_truncate(v, max_arg_length)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logger.debug(f"ENTER | {func_name}({signature})")

            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start

                if log_result:
                    result_str = _truncate(result, max_arg_length)
                    logger.debug(f"EXIT  | {func_name} | {elapsed:.4f}s | -> {result_str}")
                else:
                    logger.debug(f"EXIT  | {func_name} | {elapsed:.4f}s")

                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.debug(f"RAISE | {func_name} | {elapsed:.4f}s | {type(e).__name__}: {e}")
                raise

        return wrapper

    # Support both @trace and @trace(...)
    if func is not None:
        return decorator(func)
    return decorator


@contextmanager
def debug_context(name: str, **extra: Any) -> Any:
    """Context manager for debug logging with timing.

    Groups related operations under a named context for easier
    log analysis. Only logs when debug mode is enabled.

    Args:
        name: Name of the operation/context.
        **extra: Additional context to include in logs.

    Example:
        with debug_context("batch_processing", batch_size=100):
            for item in batch:
                process(item)
    """
    if not _is_debug_enabled():
        yield
        return

    extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
    context_str = f"{name}" + (f" ({extra_str})" if extra_str else "")

    logger.debug(f"START | {context_str}")
    start = time.perf_counter()

    try:
        yield
        elapsed = time.perf_counter() - start
        logger.debug(f"END   | {context_str} | {elapsed:.4f}s")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.debug(f"FAIL  | {context_str} | {elapsed:.4f}s | {type(e).__name__}: {e}")
        raise


class DebugMixin:
    """Mixin class to add debug logging capabilities to any class.

    Provides methods for logging method entry/exit with timing.
    All debug methods are no-ops when debug mode is disabled.

    Example:
        class MyProcessor(DebugMixin):
            def process(self, data):
                start = self._debug_enter("process", data_size=len(data))
                try:
                    result = self._do_process(data)
                    self._debug_exit("process", start, success=True)
                    return result
                except Exception as e:
                    self._debug_exit("process", start, error=str(e))
                    raise
    """

    @property
    def _debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return _is_debug_enabled()

    def _debug(self, message: str, **extra: Any) -> None:
        """Log a debug message if debug mode is enabled.

        Args:
            message: Message to log.
            **extra: Additional context.
        """
        if self._debug_enabled:
            logger.debug(f"[{self.__class__.__name__}] {message}", extra=extra)

    def _debug_enter(self, method: str, **params: Any) -> float:
        """Log method entry and return start time.

        Args:
            method: Method name.
            **params: Method parameters to log.

        Returns:
            Start time for use with _debug_exit.
        """
        if self._debug_enabled:
            params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            logger.debug(f"[{self.__class__.__name__}.{method}] ENTER ({params_str})")
        return time.perf_counter()

    def _debug_exit(
        self,
        method: str,
        start_time: float,
        result: Any = None,
        **extra: Any,
    ) -> None:
        """Log method exit with timing.

        Args:
            method: Method name.
            start_time: Start time from _debug_enter.
            result: Optional return value to log.
            **extra: Additional context (e.g., error=...).
        """
        if self._debug_enabled:
            elapsed = time.perf_counter() - start_time
            result_str = f" -> {result!r}" if result is not None else ""
            extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
            extra_suffix = f" ({extra_str})" if extra_str else ""

            logger.debug(
                f"[{self.__class__.__name__}.{method}] "
                f"EXIT ({elapsed:.4f}s){result_str}{extra_suffix}"
            )


def dump_exception(exc: Exception, include_locals: bool = False) -> str:
    """Create a detailed exception dump for debugging.

    Args:
        exc: Exception to dump.
        include_locals: Include local variables (may expose sensitive data).

    Returns:
        Formatted exception information string.
    """
    lines = [
        "=" * 60,
        "EXCEPTION DUMP",
        "=" * 60,
        f"Type: {type(exc).__name__}",
        f"Message: {exc}",
        "",
        "Traceback:",
    ]

    # Get traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    lines.extend(tb)

    # Optionally include locals
    if include_locals and exc.__traceback__:
        lines.append("")
        lines.append("Local Variables:")
        lines.append("-" * 40)

        frame = exc.__traceback__.tb_frame
        while frame:
            lines.append(f"Frame: {frame.f_code.co_name}")
            for key, value in frame.f_locals.items():
                if not key.startswith("__"):
                    try:
                        value_str = repr(value)[:200]
                    except Exception:
                        value_str = "<unrepresentable>"
                    lines.append(f"  {key} = {value_str}")
            next_frame = frame.f_back
            if next_frame is None:
                break
            frame = next_frame

    lines.append("=" * 60)
    return "\n".join(lines)


def breakpoint_if_debug() -> None:
    """Enter debugger if debug mode is enabled.

    Use this as a conditional breakpoint that only triggers
    in debug mode.

    Example:
        def complex_calculation(data):
            intermediate = transform(data)
            breakpoint_if_debug()  # Pause here only in debug mode
            return finalize(intermediate)
    """
    if _is_debug_enabled():
        import pdb

        pdb.set_trace()


class MemoryTracker:
    """Track memory usage for debugging memory issues.

    Only active when debug mode is enabled.

    Example:
        tracker = MemoryTracker()
        tracker.checkpoint("before_load")
        data = load_large_file()
        tracker.checkpoint("after_load")
        print(tracker.report())
    """

    def __init__(self) -> None:
        """Initialize memory tracker."""
        self.checkpoints: list[tuple[str, float, float]] = []
        self._enabled = _is_debug_enabled()

    def checkpoint(self, name: str) -> None:
        """Record a memory checkpoint.

        Args:
            name: Name for this checkpoint.
        """
        if not self._enabled:
            return

        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            self.checkpoints.append(
                (
                    name,
                    time.perf_counter(),
                    mem_info.rss / 1024 / 1024,  # MB
                )
            )
        except ImportError:
            # psutil not available
            pass

    def report(self) -> str:
        """Generate memory usage report.

        Returns:
            Formatted report string.
        """
        if not self.checkpoints:
            return "No checkpoints recorded (debug mode disabled or psutil not available)"

        lines = ["Memory Usage Report", "=" * 40]

        prev_mem = 0.0
        prev_time = self.checkpoints[0][1]

        for name, timestamp, mem_mb in self.checkpoints:
            delta_mem = mem_mb - prev_mem
            delta_time = timestamp - prev_time
            delta_str = f"+{delta_mem:.1f}" if delta_mem >= 0 else f"{delta_mem:.1f}"

            lines.append(f"{name:30} | {mem_mb:8.1f} MB | {delta_str:8} MB | {delta_time:.3f}s")

            prev_mem = mem_mb
            prev_time = timestamp

        return "\n".join(lines)
