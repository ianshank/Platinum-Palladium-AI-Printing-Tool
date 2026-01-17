"""
Middleware Components

Provides middleware patterns for:
- Request/Response processing
- Logging
- Timeout enforcement
- Rate limiting
- Error handling
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel

from ptpd_calibration.template.errors import TemplateError, TimeoutError
from ptpd_calibration.template.logging_config import LogContext, get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


@dataclass
class RequestContext:
    """Context for a request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.perf_counter)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MiddlewareBase(ABC, Generic[RequestT, ResponseT]):
    """
    Base class for middleware.

    Middleware can process requests before handling and responses after.
    """

    @abstractmethod
    async def process_request(
        self,
        request: RequestT,
        context: RequestContext,
    ) -> Optional[RequestT]:
        """
        Process incoming request.

        Return None to short-circuit the chain.
        """
        pass

    @abstractmethod
    async def process_response(
        self,
        response: ResponseT,
        context: RequestContext,
    ) -> ResponseT:
        """Process outgoing response."""
        pass

    async def process_error(
        self,
        error: Exception,
        context: RequestContext,
    ) -> Optional[ResponseT]:
        """
        Process an error.

        Return a response to handle the error, or None to propagate.
        """
        return None


class RequestMiddleware(MiddlewareBase[Any, Any]):
    """Request-only middleware (no response processing)."""

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        return response


class ResponseMiddleware(MiddlewareBase[Any, Any]):
    """Response-only middleware (no request processing)."""

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        return request


class MiddlewareChain:
    """
    Chain of middleware for processing requests.

    Usage:
        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.add(TimeoutMiddleware(30))
        chain.add(RateLimitMiddleware(100, 60))

        # Process request through chain
        response = await chain.process(request, handler)
    """

    def __init__(self):
        """Initialize empty middleware chain."""
        self._middleware: list[MiddlewareBase] = []

    def add(self, middleware: MiddlewareBase) -> "MiddlewareChain":
        """Add middleware to the chain."""
        self._middleware.append(middleware)
        return self

    def remove(self, middleware: MiddlewareBase) -> bool:
        """Remove middleware from the chain."""
        try:
            self._middleware.remove(middleware)
            return True
        except ValueError:
            return False

    async def process(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """
        Process a request through the middleware chain.

        Args:
            request: The request to process
            handler: The final handler function

        Returns:
            The response
        """
        context = RequestContext()

        # Process request through middleware
        current_request = request
        for middleware in self._middleware:
            try:
                result = await middleware.process_request(current_request, context)
                if result is None:
                    # Middleware short-circuited the chain
                    return None
                current_request = result
            except Exception as e:
                response = await middleware.process_error(e, context)
                if response is not None:
                    return response
                raise

        # Call the handler
        try:
            if asyncio.iscoroutinefunction(handler):
                response = await handler(current_request)
            else:
                response = handler(current_request)
        except Exception as e:
            # Try to handle error in middleware
            for middleware in reversed(self._middleware):
                error_response = await middleware.process_error(e, context)
                if error_response is not None:
                    return error_response
            raise

        # Process response through middleware (in reverse order)
        current_response = response
        for middleware in reversed(self._middleware):
            current_response = await middleware.process_response(
                current_response, context
            )

        return current_response

    def wrap(self, handler: Callable) -> Callable:
        """
        Create a wrapped handler that uses this middleware chain.

        Args:
            handler: The handler to wrap

        Returns:
            Wrapped handler
        """

        @wraps(handler)
        async def wrapped(request: Any) -> Any:
            return await self.process(request, handler)

        return wrapped


class LoggingMiddleware(MiddlewareBase[Any, Any]):
    """
    Middleware for request/response logging.

    Logs:
    - Request start with ID
    - Response status and duration
    - Errors with context
    """

    def __init__(
        self,
        log_request_body: bool = False,
        log_response_body: bool = False,
        slow_request_threshold_ms: float = 1000,
    ):
        """
        Initialize logging middleware.

        Args:
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            slow_request_threshold_ms: Threshold for slow request warnings
        """
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        """Log incoming request."""
        log_data: Dict[str, Any] = {
            "request_id": context.request_id,
            "type": type(request).__name__,
        }

        if self.log_request_body and hasattr(request, "__dict__"):
            log_data["body"] = str(request)[:500]

        logger.info("Request started", **log_data)
        return request

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        """Log outgoing response."""
        duration_ms = (time.perf_counter() - context.start_time) * 1000

        log_data: Dict[str, Any] = {
            "request_id": context.request_id,
            "duration_ms": duration_ms,
            "type": type(response).__name__,
        }

        if self.log_response_body and hasattr(response, "__dict__"):
            log_data["body"] = str(response)[:500]

        if duration_ms > self.slow_request_threshold_ms:
            logger.warning("Slow request completed", **log_data)
        else:
            logger.info("Request completed", **log_data)

        return response

    async def process_error(
        self,
        error: Exception,
        context: RequestContext,
    ) -> None:
        """Log error."""
        duration_ms = (time.perf_counter() - context.start_time) * 1000

        logger.error(
            "Request error",
            request_id=context.request_id,
            duration_ms=duration_ms,
            error=str(error),
            error_type=type(error).__name__,
        )

        return None


class TimeoutMiddleware(MiddlewareBase[Any, Any]):
    """
    Middleware for enforcing request timeouts.

    Raises TimeoutError if request exceeds the configured timeout.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize timeout middleware.

        Args:
            timeout_seconds: Request timeout in seconds
        """
        self.timeout_seconds = timeout_seconds
        self._tasks: Dict[str, asyncio.Task] = {}

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        """Start timeout tracking."""
        context.metadata["timeout_seconds"] = self.timeout_seconds
        return request

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        """Clear timeout tracking."""
        return response

    def wrap_with_timeout(self, handler: Callable) -> Callable:
        """Wrap a handler with timeout enforcement."""

        @wraps(handler)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    handler(*args, **kwargs),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Request timed out after {self.timeout_seconds}s",
                    timeout_seconds=self.timeout_seconds,
                )

        return wrapped


class RateLimitMiddleware(MiddlewareBase[Any, Any]):
    """
    Middleware for rate limiting.

    Uses sliding window algorithm for accurate rate limiting.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_func: Optional[Callable[[Any, RequestContext], str]] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            key_func: Function to extract rate limit key (default: by IP)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_func = key_func or self._default_key
        self._requests: Dict[str, list[float]] = defaultdict(list)

    def _default_key(self, request: Any, context: RequestContext) -> str:
        """Default key function using request ID prefix."""
        return context.user_id or context.session_id or "global"

    def _clean_old_requests(self, key: str, now: float) -> None:
        """Remove requests outside the window."""
        cutoff = now - self.window_seconds
        self._requests[key] = [
            t for t in self._requests[key] if t > cutoff
        ]

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        """Check rate limit."""
        key = self.key_func(request, context)
        now = time.time()

        # Clean old requests
        self._clean_old_requests(key, now)

        # Check limit
        if len(self._requests[key]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded",
                key=key,
                limit=self.max_requests,
                window_seconds=self.window_seconds,
            )
            raise TemplateError(
                f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s",
                error_code="RATE_LIMIT_EXCEEDED",
            )

        # Record request
        self._requests[key].append(now)

        # Add rate limit info to context
        context.metadata["rate_limit"] = {
            "remaining": self.max_requests - len(self._requests[key]),
            "limit": self.max_requests,
            "reset": int(now + self.window_seconds),
        }

        return request

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        """Add rate limit headers (if response supports it)."""
        return response

    def get_status(self, key: str) -> Dict[str, Any]:
        """Get rate limit status for a key."""
        now = time.time()
        self._clean_old_requests(key, now)

        return {
            "current": len(self._requests[key]),
            "remaining": self.max_requests - len(self._requests[key]),
            "limit": self.max_requests,
            "window_seconds": self.window_seconds,
        }


class AuthenticationMiddleware(MiddlewareBase[Any, Any]):
    """
    Middleware for authentication.

    Validates API keys or tokens.
    """

    def __init__(
        self,
        api_keys: Optional[set[str]] = None,
        key_header: str = "X-API-Key",
        key_extractor: Optional[Callable[[Any], Optional[str]]] = None,
        allow_anonymous: bool = False,
    ):
        """
        Initialize authentication middleware.

        Args:
            api_keys: Set of valid API keys
            key_header: Header name for API key
            key_extractor: Function to extract key from request
            allow_anonymous: Allow requests without key
        """
        self.api_keys = api_keys or set()
        self.key_header = key_header
        self.key_extractor = key_extractor
        self.allow_anonymous = allow_anonymous

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        """Validate authentication."""
        # Extract API key
        api_key = None

        if self.key_extractor:
            api_key = self.key_extractor(request)
        elif hasattr(request, "headers"):
            api_key = request.headers.get(self.key_header)

        # Validate
        if api_key:
            if api_key not in self.api_keys:
                raise TemplateError(
                    "Invalid API key",
                    error_code="INVALID_API_KEY",
                )
            context.metadata["authenticated"] = True
            context.user_id = api_key[:8]  # Use key prefix as user ID

        elif not self.allow_anonymous:
            raise TemplateError(
                "Authentication required",
                error_code="AUTH_REQUIRED",
            )

        return request

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        return response

    def add_api_key(self, key: str) -> None:
        """Add a valid API key."""
        self.api_keys.add(key)

    def remove_api_key(self, key: str) -> None:
        """Remove an API key."""
        self.api_keys.discard(key)


class CachingMiddleware(MiddlewareBase[Any, Any]):
    """
    Middleware for response caching.

    Caches responses based on request characteristics.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_size: int = 1000,
        key_func: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize caching middleware.

        Args:
            ttl_seconds: Cache TTL in seconds
            max_size: Maximum cache entries
            key_func: Function to generate cache key
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.key_func = key_func or (lambda r: str(hash(str(r))))
        self._cache: Dict[str, tuple[Any, float]] = {}

    async def process_request(
        self,
        request: Any,
        context: RequestContext,
    ) -> Any:
        """Check cache for response."""
        key = self.key_func(request)
        now = time.time()

        if key in self._cache:
            response, expires = self._cache[key]
            if now < expires:
                context.metadata["cache_hit"] = True
                logger.debug("Cache hit", cache_key=key[:16])
                # Return cached response by setting it in context
                context.metadata["cached_response"] = response
            else:
                # Expired
                del self._cache[key]

        context.metadata["cache_key"] = key
        return request

    async def process_response(
        self,
        response: Any,
        context: RequestContext,
    ) -> Any:
        """Cache the response."""
        # Check if we already have a cached response
        if "cached_response" in context.metadata:
            return context.metadata["cached_response"]

        # Cache new response
        key = context.metadata.get("cache_key")
        if key:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (response, time.time() + self.ttl_seconds)
            logger.debug("Cached response", cache_key=key[:16])

        return response

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid_entries = sum(1 for _, (_, exp) in self._cache.items() if exp > now)

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }
