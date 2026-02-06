"""
LLM client implementations for different providers.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ptpd_calibration.config import LLMProvider, LLMSettings, get_settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        pass


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, settings: LLMSettings | None = None):
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

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate completion using Anthropic API."""
        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "anthropic package required. Install with: pip install ptpd-calibration[llm]"
            ) from err

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
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "anthropic package required. Install with: pip install ptpd-calibration[llm]"
            ) from err

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

    def __init__(self, settings: LLMSettings | None = None):
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

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate completion using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise ImportError(
                "openai package required. Install with: pip install ptpd-calibration[llm]"
            ) from err

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
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise ImportError(
                "openai package required. Install with: pip install ptpd-calibration[llm]"
            ) from err

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


class VertexAIClient(LLMClient):
    """Client for Google Vertex AI (Gemini) API."""

    def __init__(self, settings: LLMSettings | None = None):
        """
        Initialize Vertex AI client.

        Uses Google Cloud Application Default Credentials (ADC).
        Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`.

        Args:
            settings: LLM settings with Vertex AI project/location config.
        """
        self.settings = settings or get_settings().llm

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate completion using Vertex AI Gemini API."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as err:
            raise ImportError(
                "google-genai package required. Install with: pip install ptpd-calibration[vertex]"
            ) from err

        logger.debug(
            "VertexAI complete: model=%s, project=%s, msgs=%d",
            self.settings.vertex_model,
            self.settings.vertex_project,
            len(messages),
        )

        client = genai.Client(
            vertexai=True,
            project=self.settings.vertex_project,
            location=self.settings.vertex_location,
        )

        contents = _convert_messages_to_gemini(messages, types)

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )
        if system:
            gen_config.system_instruction = system

        response = client.models.generate_content(
            model=self.settings.vertex_model,
            contents=contents,
            config=gen_config,
        )

        return response.text or ""

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using Vertex AI Gemini API."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as err:
            raise ImportError(
                "google-genai package required. Install with: pip install ptpd-calibration[vertex]"
            ) from err

        client = genai.Client(
            vertexai=True,
            project=self.settings.vertex_project,
            location=self.settings.vertex_location,
        )

        contents = _convert_messages_to_gemini(messages, types)

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )
        if system:
            gen_config.system_instruction = system

        for chunk in client.models.generate_content_stream(
            model=self.settings.vertex_model,
            contents=contents,
            config=gen_config,
        ):
            if chunk.text:
                yield chunk.text


def _convert_messages_to_gemini(messages: list[dict], types: Any) -> list:  # type: ignore[type-arg]
    """Convert OpenAI-style messages to Gemini Content format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        types: The google.genai.types module.

    Returns:
        List of Gemini Content objects.
    """
    contents = []
    for msg in messages:
        role = msg["role"]
        # Map roles: user->user, assistant->model
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(msg["content"])],
            )
        )
    return contents


def create_client(settings: LLMSettings | None = None) -> LLMClient:
    """
    Create an LLM client based on settings.

    Args:
        settings: LLM settings. Uses global settings if not provided.

    Returns:
        LLMClient instance for the configured provider.
    """
    settings = settings or get_settings().llm

    if settings.provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(settings)
    elif settings.provider == LLMProvider.OPENAI:
        return OpenAIClient(settings)
    elif settings.provider == LLMProvider.VERTEX_AI:
        return VertexAIClient(settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.provider}")
