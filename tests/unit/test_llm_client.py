"""
Unit tests for LLM client implementations and factory.
"""
from unittest.mock import MagicMock, patch

import pytest

from ptpd_calibration.config import LLMProvider, LLMSettings
from ptpd_calibration.llm.client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    create_client,
)


def _make_settings(
    provider: LLMProvider = LLMProvider.ANTHROPIC,
    api_key: str = "test-key",
) -> LLMSettings:
    """Helper to build LLMSettings with sensible test defaults."""
    return LLMSettings(
        provider=provider,
        api_key=api_key,
        max_tokens=256,
        temperature=0.7,
    )


# ── AnthropicClient ─────────────────────────────────────────────

class TestAnthropicClient:
    """Tests for the Anthropic LLM client."""

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_without_api_key(self) -> None:
        settings = LLMSettings(provider=LLMProvider.ANTHROPIC)
        with pytest.raises(ValueError, match="Anthropic API key"):
            AnthropicClient(settings)

    def test_init_with_api_key(self) -> None:
        settings = _make_settings(LLMProvider.ANTHROPIC, "sk-ant-test")
        client = AnthropicClient(settings)
        assert isinstance(client, LLMClient)
        # api_key should be set (could be overridden by env, so just verify it's non-empty)
        assert client.api_key

    def test_prefers_anthropic_specific_key(self) -> None:
        settings = LLMSettings(
            provider=LLMProvider.ANTHROPIC,
            api_key="generic-key",
            anthropic_api_key="specific-key",
        )
        client = AnthropicClient(settings)
        assert client.api_key == "specific-key"


# ── OpenAIClient ────────────────────────────────────────────────

class TestOpenAIClient:
    """Tests for the OpenAI LLM client."""

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_without_api_key(self) -> None:
        settings = LLMSettings(provider=LLMProvider.OPENAI)
        with pytest.raises(ValueError, match="OpenAI API key"):
            OpenAIClient(settings)

    def test_init_with_api_key(self) -> None:
        settings = _make_settings(LLMProvider.OPENAI, "sk-openai-test")
        client = OpenAIClient(settings)
        assert isinstance(client, LLMClient)
        assert client.api_key

    def test_prefers_openai_specific_key(self) -> None:
        settings = LLMSettings(
            provider=LLMProvider.OPENAI,
            api_key="generic-key",
            openai_api_key="specific-key",
        )
        client = OpenAIClient(settings)
        assert client.api_key == "specific-key"


# ── Factory ─────────────────────────────────────────────────────

class TestCreateClient:
    """Tests for the create_client factory function."""

    def test_creates_anthropic_client(self) -> None:
        settings = _make_settings(LLMProvider.ANTHROPIC)
        client = create_client(settings)
        assert isinstance(client, AnthropicClient)

    def test_creates_openai_client(self) -> None:
        settings = _make_settings(LLMProvider.OPENAI)
        client = create_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_raises_for_unsupported_provider(self) -> None:
        settings = _make_settings(LLMProvider.ANTHROPIC)
        settings.provider = "unsupported_provider"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_client(settings)
