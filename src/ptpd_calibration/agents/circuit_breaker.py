"""
Circuit breaker pattern for graceful degradation.

Provides protection against cascading failures when external services
(like LLM APIs) become unavailable.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.agents.logging import EventType, get_agent_logger

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerSettings(BaseSettings):
    """Settings for circuit breaker behavior."""

    model_config = SettingsConfigDict(env_prefix="PTPD_CIRCUIT_")

    # Failure thresholds
    failure_threshold: int = Field(
        default=3, ge=1, le=100, description="Consecutive failures before opening circuit"
    )
    success_threshold: int = Field(
        default=2, ge=1, le=100, description="Consecutive successes to close circuit from half-open"
    )

    # Timing
    cooldown_seconds: float = Field(
        default=30.0, ge=5.0, le=600.0, description="Time to wait before testing recovery"
    )
    half_open_timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=120.0, description="Timeout for half-open test requests"
    )

    # Fallback behavior
    enable_fallback: bool = Field(
        default=True, description="Enable fallback responses when circuit is open"
    )
    fallback_cache_ttl_seconds: float = Field(
        default=300.0, ge=60.0, le=3600.0, description="TTL for cached fallback responses"
    )


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        service_name: str | None = None,
        retry_after: float | None = None,
    ):
        """
        Initialize the error.

        Args:
            message: Error message.
            service_name: Name of the protected service.
            retry_after: Seconds until retry is allowed.
        """
        super().__init__(message)
        self.service_name = service_name
        self.retry_after = retry_after


@dataclass
class CircuitBreakerState:
    """Tracks the state of a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0
    last_error: str | None = None


@dataclass
class FallbackCacheEntry:
    """Cache entry for fallback responses."""

    value: Any
    timestamp: float
    key: str


class CircuitBreaker:
    """
    Circuit breaker for protecting against service failures.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests blocked (use fallback if available)
    - HALF_OPEN: Testing if service recovered

    Example:
        ```python
        breaker = CircuitBreaker("llm_service")

        try:
            result = await breaker.call(llm_client.complete, prompt="Hello")
        except CircuitOpenError:
            # Handle circuit open state
            result = "Service temporarily unavailable"
        ```
    """

    def __init__(
        self,
        name: str,
        settings: CircuitBreakerSettings | None = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Name identifying this circuit breaker.
            settings: Configuration settings.
        """
        self.name = name
        self.settings = settings or CircuitBreakerSettings()
        self._state = CircuitBreakerState()
        self._logger = get_agent_logger()
        self._lock = asyncio.Lock()

        # Fallback cache
        self._fallback_cache: dict[str, FallbackCacheEntry] = {}

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state.state == CircuitState.HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._state.state != CircuitState.OPEN:
            return False
        elapsed = time.time() - self._state.last_failure_time
        return elapsed >= self.settings.cooldown_seconds

    def _on_success(self) -> None:
        """Handle successful request."""
        self._state.last_success_time = time.time()
        self._state.total_successes += 1

        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            if self._state.success_count >= self.settings.success_threshold:
                self._transition_to_closed()
        else:
            # Reset failure count on success in closed state
            self._state.failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed request."""
        self._state.last_failure_time = time.time()
        self._state.total_failures += 1
        self._state.failure_count += 1
        self._state.last_error = str(error)

        if self._state.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to_open()
        elif (
            self._state.state == CircuitState.CLOSED
            and self._state.failure_count >= self.settings.failure_threshold
        ):
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        old_state = self._state.state
        self._state.state = CircuitState.OPEN
        self._state.success_count = 0

        self._logger.warning(
            f"Circuit breaker OPEN: {self.name}",
            event_type=EventType.ERROR,
            data={
                "circuit": self.name,
                "old_state": old_state.value,
                "new_state": CircuitState.OPEN.value,
                "failure_count": self._state.failure_count,
                "last_error": self._state.last_error,
            },
        )

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        old_state = self._state.state
        self._state.state = CircuitState.HALF_OPEN
        self._state.success_count = 0
        self._state.failure_count = 0

        self._logger.info(
            f"Circuit breaker HALF_OPEN: {self.name}",
            data={
                "circuit": self.name,
                "old_state": old_state.value,
                "new_state": CircuitState.HALF_OPEN.value,
            },
        )

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        old_state = self._state.state
        self._state.state = CircuitState.CLOSED
        self._state.failure_count = 0
        self._state.success_count = 0

        self._logger.info(
            f"Circuit breaker CLOSED: {self.name}",
            data={
                "circuit": self.name,
                "old_state": old_state.value,
                "new_state": CircuitState.CLOSED.value,
            },
        )

    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for fallback."""
        # Simple key based on function name and string repr of args
        return f"{func.__name__}:{hash((str(args), str(sorted(kwargs.items()))))}"

    def _get_fallback(self, cache_key: str) -> Any | None:
        """Get cached fallback value if available and not expired."""
        if cache_key not in self._fallback_cache:
            return None

        entry = self._fallback_cache[cache_key]
        age = time.time() - entry.timestamp

        if age > self.settings.fallback_cache_ttl_seconds:
            del self._fallback_cache[cache_key]
            return None

        return entry.value

    def _store_fallback(self, cache_key: str, value: Any) -> None:
        """Store successful result for potential fallback use."""
        self._fallback_cache[cache_key] = FallbackCacheEntry(
            value=value,
            timestamp=time.time(),
            key=cache_key,
        )

        # Clean up old entries
        current_time = time.time()
        expired_keys = [
            k
            for k, v in self._fallback_cache.items()
            if current_time - v.timestamp > self.settings.fallback_cache_ttl_seconds
        ]
        for k in expired_keys:
            del self._fallback_cache[k]

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: T | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to call.
            *args: Positional arguments for the function.
            fallback: Optional fallback value if circuit is open.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the function call.

        Raises:
            CircuitOpenError: If circuit is open and no fallback available.
        """
        async with self._lock:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                self._transition_to_half_open()

        # Generate cache key for fallback
        cache_key = self._get_cache_key(func, args, kwargs)

        # Check circuit state
        if self._state.state == CircuitState.OPEN:
            retry_after = (
                self.settings.cooldown_seconds
                - (time.time() - self._state.last_failure_time)
            )

            # Try fallback
            if self.settings.enable_fallback:
                cached = self._get_fallback(cache_key)
                if cached is not None:
                    self._logger.debug(
                        f"Circuit open, using cached fallback: {self.name}",
                        data={"circuit": self.name},
                    )
                    return cast(T, cached)

            if fallback is not None:
                return fallback

            raise CircuitOpenError(
                message=f"Circuit breaker {self.name} is open",
                service_name=self.name,
                retry_after=max(0, retry_after),
            )

        # Execute the function
        try:
            # Set timeout for half-open state
            result: T
            if self._state.state == CircuitState.HALF_OPEN:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.settings.half_open_timeout_seconds,
                )
            else:
                result = await func(*args, **kwargs)

            # Success handling
            async with self._lock:
                self._on_success()

            # Cache for fallback
            if self.settings.enable_fallback:
                self._store_fallback(cache_key, result)

            return result

        except asyncio.TimeoutError as e:
            async with self._lock:
                self._on_failure(e)
            raise

        except Exception as e:
            async with self._lock:
                self._on_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitBreakerState()
        self._logger.info(
            f"Circuit breaker manually reset: {self.name}",
            data={"circuit": self.name},
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats.
        """
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "total_failures": self._state.total_failures,
            "total_successes": self._state.total_successes,
            "last_failure_time": self._state.last_failure_time,
            "last_success_time": self._state.last_success_time,
            "last_error": self._state.last_error,
            "cache_size": len(self._fallback_cache),
        }


# Registry of circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    settings: CircuitBreakerSettings | None = None,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Args:
        name: Name of the circuit breaker.
        settings: Optional settings for new circuit breakers.

    Returns:
        CircuitBreaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, settings)
    return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    for breaker in _circuit_breakers.values():
        breaker.reset()


def get_all_circuit_breaker_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for all circuit breakers.

    Returns:
        Dictionary mapping breaker names to their stats.
    """
    return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}
