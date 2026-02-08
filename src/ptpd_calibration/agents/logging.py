"""
Structured logging system for agentic operations.

Provides JSON-formatted logging with trace IDs for debugging and observability.
"""

import json
import logging
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from uuid import uuid4

from ptpd_calibration.config import get_settings


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Types of events that can be logged."""

    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    # Subagent events
    SUBAGENT_SPAWNED = "subagent.spawned"
    SUBAGENT_COMPLETED = "subagent.completed"
    SUBAGENT_FAILED = "subagent.failed"

    # Tool events
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Planning events
    PLAN_CREATED = "plan.created"
    PLAN_STEP_STARTED = "plan.step.started"
    PLAN_STEP_COMPLETED = "plan.step.completed"
    PLAN_STEP_FAILED = "plan.step.failed"
    PLAN_ADAPTED = "plan.adapted"

    # Memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_PRUNED = "memory.pruned"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"

    # Communication events
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"

    # System events
    HEALTH_CHECK = "health.check"
    METRIC_RECORDED = "metric.recorded"
    ERROR = "error"


@dataclass
class LogContext:
    """Context for structured logging."""

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4())[:8])
    parent_span_id: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    session_id: str | None = None

    def child_span(self) -> "LogContext":
        """Create a child span context."""
        return LogContext(
            trace_id=self.trace_id,
            span_id=str(uuid4())[:8],
            parent_span_id=self.span_id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            session_id=self.session_id,
        )


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: str
    level: str
    event_type: str
    message: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    duration_ms: float | None = None
    data: dict = field(default_factory=dict)
    error: str | None = None
    stack_trace: str | None = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        entry_dict = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(entry_dict, default=str)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_data["span_id"] = record.span_id
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, "agent_type"):
            log_data["agent_type"] = record.agent_type
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "data"):
            log_data["data"] = record.data

        # Add exception info if present
        if record.exc_info:
            log_data["error"] = str(record.exc_info[1])
            log_data["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class AgentLogger:
    """
    Structured logger for agent operations.

    Provides JSON-formatted logging with automatic context tracking.
    """

    def __init__(
        self,
        name: str = "ptpd.agents",
        level: str | None = None,
        json_output: bool = True,
    ):
        """
        Initialize the agent logger.

        Args:
            name: Logger name.
            level: Log level (defaults to settings).
            json_output: Whether to use JSON formatting.
        """
        self.logger = logging.getLogger(name)
        self._context: LogContext | None = None
        self._json_output = json_output

        # Set level from settings or parameter
        settings = get_settings()
        log_level = level or settings.log_level
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Configure handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if json_output:
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    )
                )
            self.logger.addHandler(handler)

    def set_context(self, context: LogContext) -> None:
        """Set the current logging context."""
        self._context = context

    def get_context(self) -> LogContext:
        """Get the current logging context, creating one if needed."""
        if self._context is None:
            self._context = LogContext()
        return self._context

    @contextmanager
    def span(self, name: str, event_type: EventType | None = None):
        """
        Create a logging span for tracking nested operations.

        Args:
            name: Span name.
            event_type: Optional event type for span start/end.

        Yields:
            Child LogContext for the span.
        """
        parent_context = self.get_context()
        child_context = parent_context.child_span()
        old_context = self._context
        self._context = child_context

        start_time = time.time()
        try:
            if event_type:
                self.info(f"Starting: {name}", event_type=event_type)
            yield child_context
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(
                f"Failed: {name}",
                event_type=EventType.ERROR,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise
        finally:
            self._context = old_context

    def _log(
        self,
        level: int,
        message: str,
        event_type: EventType | str | None = None,
        duration_ms: float | None = None,
        data: dict | None = None,
        error: str | None = None,
        **kwargs,
    ) -> None:
        """
        Internal logging method.

        Args:
            level: Log level.
            message: Log message.
            event_type: Type of event.
            duration_ms: Duration in milliseconds.
            data: Additional data to log.
            error: Error message if applicable.
            **kwargs: Additional fields.
        """
        context = self.get_context()

        extra = {
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "agent_id": context.agent_id,
            "agent_type": context.agent_type,
        }

        if event_type:
            extra["event_type"] = event_type.value if isinstance(event_type, EventType) else event_type
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        if data:
            extra["data"] = data
        if error:
            extra["error"] = error

        extra.update(kwargs)

        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    # Convenience methods for common events

    def log_agent_started(
        self,
        agent_id: str,
        agent_type: str,
        task: str,
        **kwargs,
    ) -> None:
        """Log agent start event."""
        self.info(
            f"Agent started: {agent_type}",
            event_type=EventType.AGENT_STARTED,
            data={"agent_id": agent_id, "agent_type": agent_type, "task": task[:200], **kwargs},
        )

    def log_agent_completed(
        self,
        agent_id: str,
        duration_ms: float,
        result_summary: str,
        **kwargs,
    ) -> None:
        """Log agent completion event."""
        self.info(
            "Agent completed",
            event_type=EventType.AGENT_COMPLETED,
            duration_ms=duration_ms,
            data={"agent_id": agent_id, "result_summary": result_summary[:200], **kwargs},
        )

    def log_agent_failed(
        self,
        agent_id: str,
        error: str,
        duration_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Log agent failure event."""
        self.error(
            f"Agent failed: {error}",
            event_type=EventType.AGENT_FAILED,
            error=error,
            duration_ms=duration_ms,
            data={"agent_id": agent_id, **kwargs},
        )

    def log_tool_called(
        self,
        tool_name: str,
        args: dict,
        **kwargs,
    ) -> None:
        """Log tool invocation."""
        # Truncate large args
        truncated_args = {k: str(v)[:100] for k, v in args.items()}
        self.debug(
            f"Tool called: {tool_name}",
            event_type=EventType.TOOL_CALLED,
            data={"tool_name": tool_name, "args": truncated_args, **kwargs},
        )

    def log_tool_completed(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        **kwargs,
    ) -> None:
        """Log tool completion."""
        level = logging.DEBUG if success else logging.WARNING
        self._log(
            level,
            f"Tool completed: {tool_name} (success={success})",
            event_type=EventType.TOOL_COMPLETED if success else EventType.TOOL_FAILED,
            duration_ms=duration_ms,
            data={"tool_name": tool_name, "success": success, **kwargs},
        )

    def log_plan_created(
        self,
        goal: str,
        num_steps: int,
        **kwargs,
    ) -> None:
        """Log plan creation."""
        self.info(
            f"Plan created: {goal[:100]}",
            event_type=EventType.PLAN_CREATED,
            data={"goal": goal, "num_steps": num_steps, **kwargs},
        )

    def log_llm_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        **kwargs,
    ) -> None:
        """Log LLM request."""
        self.debug(
            f"LLM request: {provider}/{model}",
            event_type=EventType.LLM_REQUEST,
            data={"provider": provider, "model": model, "prompt_tokens": prompt_tokens, **kwargs},
        )

    def log_llm_response(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        completion_tokens: int | None = None,
        **kwargs,
    ) -> None:
        """Log LLM response."""
        self.debug(
            f"LLM response: {provider}/{model}",
            event_type=EventType.LLM_RESPONSE,
            duration_ms=duration_ms,
            data={"provider": provider, "model": model, "completion_tokens": completion_tokens, **kwargs},
        )

    def log_message_sent(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        **kwargs,
    ) -> None:
        """Log inter-agent message."""
        self.debug(
            f"Message: {from_agent} -> {to_agent}",
            event_type=EventType.MESSAGE_SENT,
            data={"from_agent": from_agent, "to_agent": to_agent, "message_type": message_type, **kwargs},
        )


def timed_operation(logger: AgentLogger, event_type: EventType | None = None):
    """
    Decorator for timing and logging operations.

    Args:
        logger: Logger instance.
        event_type: Event type to log.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Completed: {func.__name__}",
                    event_type=event_type,
                    duration_ms=duration_ms,
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed: {func.__name__}",
                    event_type=EventType.ERROR,
                    error=str(e),
                    duration_ms=duration_ms,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Completed: {func.__name__}",
                    event_type=event_type,
                    duration_ms=duration_ms,
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed: {func.__name__}",
                    event_type=EventType.ERROR,
                    error=str(e),
                    duration_ms=duration_ms,
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Global logger instance
_agent_logger: AgentLogger | None = None


def get_agent_logger() -> AgentLogger:
    """Get the global agent logger instance."""
    global _agent_logger
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    return _agent_logger


def configure_agent_logging(
    level: str = "INFO",
    json_output: bool = True,
) -> AgentLogger:
    """
    Configure agent logging.

    Args:
        level: Log level.
        json_output: Whether to use JSON formatting.

    Returns:
        Configured AgentLogger instance.
    """
    global _agent_logger
    _agent_logger = AgentLogger(level=level, json_output=json_output)
    return _agent_logger
