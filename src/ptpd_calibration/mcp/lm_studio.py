"""
LM Studio client implementation for local LLM inference.

LM Studio provides an OpenAI-compatible API, making integration straightforward
while supporting various local LLM models.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from ptpd_calibration.mcp.config import LMStudioSettings, get_mcp_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: str = "local"
    permissions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from API response."""
        return cls(
            id=data.get("id", "unknown"),
            object=data.get("object", "model"),
            created=data.get("created"),
            owned_by=data.get("owned_by", "local"),
            permissions=data.get("permissions", []),
        )


@dataclass
class CompletionUsage:
    """Token usage information from completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "CompletionUsage":
        """Create CompletionUsage from API response."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class CompletionResult:
    """Result from a chat completion request."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[CompletionUsage] = None
    created: Optional[datetime] = None
    id: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "CompletionResult":
        """Create CompletionResult from API response."""
        choices = data.get("choices", [])
        content = ""
        finish_reason = None

        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            finish_reason = choices[0].get("finish_reason")

        usage = None
        if "usage" in data:
            usage = CompletionUsage.from_api_response(data["usage"])

        created = None
        if "created" in data:
            created = datetime.fromtimestamp(data["created"], tz=timezone.utc)

        return cls(
            content=content,
            model=data.get("model", "unknown"),
            finish_reason=finish_reason,
            usage=usage,
            created=created,
            id=data.get("id"),
        )


class LMStudioError(Exception):
    """Base exception for LM Studio errors."""

    pass


class LMStudioConnectionError(LMStudioError):
    """Connection error to LM Studio server."""

    pass


class LMStudioAPIError(LMStudioError):
    """API error from LM Studio server."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LMStudioTimeoutError(LMStudioError):
    """Timeout error for LM Studio requests."""

    pass


class BaseLMStudioClient(ABC):
    """Abstract base class for LM Studio clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        pass


class LMStudioClient(BaseLMStudioClient):
    """
    Client for LM Studio's OpenAI-compatible API.

    This client provides async methods for interacting with a local
    LM Studio instance for LLM inference.

    Example:
        ```python
        client = LMStudioClient()

        # Check connection
        if await client.health_check():
            # Generate completion
            result = await client.complete([
                {"role": "user", "content": "Hello!"}
            ])
            print(result.content)
        ```
    """

    def __init__(self, settings: Optional[LMStudioSettings] = None):
        """
        Initialize LM Studio client.

        Args:
            settings: LM Studio connection settings. Uses default settings if not provided.
        """
        self.settings = settings or get_mcp_settings().lm_studio
        self._http_client: Optional[Any] = None

    async def _get_http_client(self) -> Any:
        """Get or create async HTTP client."""
        if self._http_client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx package required for LM Studio client. "
                    "Install with: pip install ptpd-calibration[mcp]"
                )

            self._http_client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(self.settings.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "LMStudioClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Make HTTP request with retry logic."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package required. Install with: pip install ptpd-calibration[mcp]"
            )

        client = await self._get_http_client()
        last_exception: Optional[Exception] = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = await getattr(client, method)(endpoint, **kwargs)

                if response.status_code >= 400:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("error", {}).get("message", error_detail)
                    except Exception:
                        pass
                    raise LMStudioAPIError(
                        f"API error: {error_detail}",
                        status_code=response.status_code,
                    )

                return response.json()

            except httpx.ConnectError as e:
                last_exception = LMStudioConnectionError(
                    f"Failed to connect to LM Studio at {self.settings.base_url}: {e}"
                )
            except httpx.TimeoutException as e:
                last_exception = LMStudioTimeoutError(
                    f"Request timed out after {self.settings.timeout_seconds}s: {e}"
                )
            except LMStudioAPIError:
                raise
            except Exception as e:
                last_exception = LMStudioError(f"Unexpected error: {e}")

            if attempt < self.settings.max_retries:
                delay = self.settings.retry_delay_seconds * (2**attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.settings.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {last_exception}"
                )
                await asyncio.sleep(delay)

        raise last_exception or LMStudioError("Unknown error occurred")

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            **kwargs: Additional parameters passed to the API.

        Returns:
            CompletionResult with the generated content.

        Raises:
            LMStudioConnectionError: If unable to connect to server.
            LMStudioAPIError: If the API returns an error.
            LMStudioTimeoutError: If the request times out.
        """
        # Build messages list with optional system prompt
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # Build request payload
        payload: dict[str, Any] = {
            "messages": all_messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "top_p": kwargs.pop("top_p", self.settings.top_p),
            "stream": False,
        }

        # Add model if specified
        if self.settings.model:
            payload["model"] = self.settings.model

        # Add any extra parameters
        payload.update(kwargs)

        response = await self._make_request_with_retry(
            "post",
            "/chat/completions",
            json=payload,
        )

        return CompletionResult.from_api_response(response)

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            **kwargs: Additional parameters passed to the API.

        Yields:
            String chunks of the generated content.

        Raises:
            LMStudioConnectionError: If unable to connect to server.
            LMStudioAPIError: If the API returns an error.
            LMStudioTimeoutError: If the request times out.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package required. Install with: pip install ptpd-calibration[mcp]"
            )

        # Build messages list with optional system prompt
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # Build request payload
        payload: dict[str, Any] = {
            "messages": all_messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "top_p": kwargs.pop("top_p", self.settings.top_p),
            "stream": True,
        }

        # Add model if specified
        if self.settings.model:
            payload["model"] = self.settings.model

        # Add any extra parameters
        payload.update(kwargs)

        client = await self._get_http_client()

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    content = await response.aread()
                    raise LMStudioAPIError(
                        f"API error: {content.decode()}",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            import json

                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except Exception as e:
                            logger.debug(f"Failed to parse stream chunk: {e}")
                            continue

        except httpx.ConnectError as e:
            raise LMStudioConnectionError(
                f"Failed to connect to LM Studio at {self.settings.base_url}: {e}"
            )
        except httpx.TimeoutException as e:
            raise LMStudioTimeoutError(
                f"Stream request timed out after {self.settings.timeout_seconds}s: {e}"
            )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models.

        Returns:
            List of ModelInfo objects for available models.

        Raises:
            LMStudioConnectionError: If unable to connect to server.
            LMStudioAPIError: If the API returns an error.
        """
        response = await self._make_request_with_retry("get", "/models")

        models = []
        for model_data in response.get("data", []):
            models.append(ModelInfo.from_api_response(model_data))

        return models

    async def health_check(self) -> bool:
        """
        Check if the LM Studio server is healthy.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            await self.list_models()
            return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo if found, None otherwise.
        """
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None


class MockLMStudioClient(BaseLMStudioClient):
    """
    Mock LM Studio client for testing.

    This client provides predictable responses without requiring
    an actual LM Studio instance.
    """

    def __init__(
        self,
        default_response: str = "This is a mock response from LM Studio.",
        models: Optional[list[str]] = None,
    ):
        """
        Initialize mock client.

        Args:
            default_response: Default response content.
            models: List of mock model IDs.
        """
        self.default_response = default_response
        self.models = models or ["mock-model-1", "mock-model-2"]
        self.call_history: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a mock completion."""
        self.call_history.append(
            {
                "method": "complete",
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        return CompletionResult(
            content=self.default_response,
            model="mock-model",
            finish_reason="stop",
            usage=CompletionUsage(
                prompt_tokens=len(str(messages)),
                completion_tokens=len(self.default_response),
                total_tokens=len(str(messages)) + len(self.default_response),
            ),
            created=datetime.now(timezone.utc),
            id="mock-completion-id",
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a mock completion."""
        self.call_history.append(
            {
                "method": "stream",
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        # Yield response word by word
        words = self.default_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)  # Simulate streaming delay

    async def list_models(self) -> list[ModelInfo]:
        """List mock models."""
        return [
            ModelInfo(
                id=model_id,
                object="model",
                created=int(datetime.now(timezone.utc).timestamp()),
                owned_by="mock",
            )
            for model_id in self.models
        ]

    async def health_check(self) -> bool:
        """Always return True for mock client."""
        return True


def create_lm_studio_client(
    settings: Optional[LMStudioSettings] = None,
    use_mock: bool = False,
    mock_response: Optional[str] = None,
) -> BaseLMStudioClient:
    """
    Create an LM Studio client.

    Args:
        settings: LM Studio settings. Uses default settings if not provided.
        use_mock: If True, create a mock client for testing.
        mock_response: Default response for mock client.

    Returns:
        BaseLMStudioClient instance.

    Example:
        ```python
        # Production client
        client = create_lm_studio_client()

        # Mock client for testing
        mock_client = create_lm_studio_client(use_mock=True)
        ```
    """
    if use_mock:
        return MockLMStudioClient(
            default_response=mock_response or "Mock response for testing."
        )

    return LMStudioClient(settings=settings)
