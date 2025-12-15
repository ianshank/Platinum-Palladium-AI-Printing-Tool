"""
LLM client implementations for different providers.

Supports:
- Anthropic Claude API
- OpenAI API
- LM Studio (local, OpenAI-compatible)
- Ollama (local, OpenAI-compatible)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ptpd_calibration.config import LLMProvider, LLMSettings, get_settings


logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name for logging."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used."""
        pass

    def is_available(self) -> bool:
        """Check if the client is available (server is running, API key is valid, etc.)."""
        return True


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initialize Anthropic client.

        Args:
            settings: LLM settings with API key.
        """
        self.settings = settings or get_settings().llm

        self.api_key = self.settings.anthropic_api_key or self.settings.api_key
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set PTPD_LLM_ANTHROPIC_API_KEY or PTPD_LLM_API_KEY environment variable."
            )

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    @property
    def model_name(self) -> str:
        return self.settings.anthropic_model

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate completion using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install ptpd-calibration[llm]"
            )

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            system=system or "",
            messages=messages,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )

        return response.content[0].text

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install ptpd-calibration[llm]"
            )

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        async with client.messages.stream(
            model=self.settings.anthropic_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            system=system or "",
            messages=messages,
            temperature=temperature if temperature is not None else self.settings.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initialize OpenAI client.

        Args:
            settings: LLM settings with API key.
        """
        self.settings = settings or get_settings().llm

        self.api_key = self.settings.openai_api_key or self.settings.api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set PTPD_LLM_OPENAI_API_KEY or PTPD_LLM_API_KEY environment variable."
            )

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_name(self) -> str:
        return self.settings.openai_model

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate completion using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install ptpd-calibration[llm]"
            )

        client = AsyncOpenAI(api_key=self.api_key)

        # Prepend system message
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            messages=all_messages,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )

        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install ptpd-calibration[llm]"
            )

        client = AsyncOpenAI(api_key=self.api_key)

        # Prepend system message
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        stream = await client.chat.completions.create(
            model=self.settings.openai_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            messages=all_messages,
            temperature=temperature if temperature is not None else self.settings.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalLLMClient(LLMClient):
    """Base client for local LLM servers with OpenAI-compatible API.

    This client works with:
    - LM Studio
    - Ollama
    - text-generation-webui
    - LocalAI
    - Any other server with OpenAI-compatible API
    """

    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize local LLM client.

        Args:
            settings: LLM settings.
            base_url: Override base URL from settings.
            model: Override model name from settings.
            timeout: Override timeout from settings.
        """
        self.settings = settings or get_settings().llm
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._is_available: Optional[bool] = None

    @property
    def base_url(self) -> str:
        """Get the base URL for the local server."""
        return self._base_url or self.settings.get_base_url() or "http://localhost:1234/v1"

    @property
    def provider_name(self) -> str:
        return "Local LLM"

    @property
    def model_name(self) -> str:
        return self._model or self.settings.get_model_name()

    @property
    def timeout(self) -> int:
        """Get the timeout for requests."""
        return self._timeout or self.settings.get_timeout()

    def is_available(self) -> bool:
        """Check if the local server is available."""
        if self._is_available is not None:
            return self._is_available

        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                # Try to reach the models endpoint
                response = client.get(f"{self.base_url}/models")
                self._is_available = response.status_code == 200
        except Exception as e:
            logger.debug(f"Local LLM server not available: {e}")
            self._is_available = False

        return self._is_available

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate completion using local OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="not-needed",  # Local servers don't require API keys
            timeout=float(self.timeout),
        )

        # Prepend system message
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens or self.settings.max_tokens,
                messages=all_messages,
                temperature=temperature if temperature is not None else self.settings.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Local LLM completion failed: {e}")
            raise

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream completion using local OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="not-needed",
            timeout=float(self.timeout),
        )

        # Prepend system message
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens or self.settings.max_tokens,
                messages=all_messages,
                temperature=temperature if temperature is not None else self.settings.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Local LLM streaming failed: {e}")
            raise


class LMStudioClient(LocalLLMClient):
    """Client for LM Studio local inference server.

    LM Studio provides a local server with OpenAI-compatible API on port 1234.
    Start LM Studio and enable the local server before using this client.
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initialize LM Studio client.

        Args:
            settings: LLM settings with LM Studio configuration.
        """
        settings = settings or get_settings().llm
        super().__init__(
            settings=settings,
            base_url=settings.lm_studio_base_url,
            model=settings.lm_studio_model,
            timeout=settings.lm_studio_timeout,
        )

    @property
    def provider_name(self) -> str:
        return "LM Studio"

    @property
    def model_name(self) -> str:
        return self._model or self.settings.lm_studio_model


class OllamaClient(LocalLLMClient):
    """Client for Ollama local inference server.

    Ollama provides a local server with OpenAI-compatible API on port 11434.
    Install Ollama and pull a model before using this client:
        ollama pull llama3.2
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initialize Ollama client.

        Args:
            settings: LLM settings with Ollama configuration.
        """
        settings = settings or get_settings().llm
        super().__init__(
            settings=settings,
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout,
        )

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def model_name(self) -> str:
        return self._model or self.settings.ollama_model

    async def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            List of model names available in Ollama.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Use Ollama's native API for listing models
                ollama_api_url = self.base_url.replace("/v1", "")
                response = await client.get(f"{ollama_api_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Failed to list Ollama models: {e}")
        return []


def create_client(settings: Optional[LLMSettings] = None) -> LLMClient:
    """
    Create an LLM client based on settings.

    Args:
        settings: LLM settings. Uses global settings if not provided.

    Returns:
        LLMClient instance for the configured provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    settings = settings or get_settings().llm

    if settings.provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(settings)
    elif settings.provider == LLMProvider.OPENAI:
        return OpenAIClient(settings)
    elif settings.provider == LLMProvider.LM_STUDIO:
        return LMStudioClient(settings)
    elif settings.provider == LLMProvider.OLLAMA:
        return OllamaClient(settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.provider}")


def get_available_providers() -> list[tuple[LLMProvider, bool]]:
    """
    Check which LLM providers are available.

    Returns:
        List of (provider, is_available) tuples.
    """
    results = []

    for provider in LLMProvider:
        try:
            settings = LLMSettings(provider=provider)

            if provider in (LLMProvider.LM_STUDIO, LLMProvider.OLLAMA):
                # Check if local server is available
                client = create_client(settings)
                is_available = client.is_available()
            elif provider == LLMProvider.ANTHROPIC:
                # Check if API key is configured
                is_available = bool(settings.anthropic_api_key or settings.api_key)
            elif provider == LLMProvider.OPENAI:
                # Check if API key is configured
                is_available = bool(settings.openai_api_key or settings.api_key)
            else:
                is_available = False

            results.append((provider, is_available))
        except Exception as e:
            logger.debug(f"Error checking provider {provider}: {e}")
            results.append((provider, False))

    return results
