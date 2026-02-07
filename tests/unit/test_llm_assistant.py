"""
Unit tests for the CalibrationAssistant.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ptpd_calibration.config import LLMProvider, LLMSettings
from ptpd_calibration.llm.assistant import CalibrationAssistant, create_assistant
from ptpd_calibration.llm.client import AnthropicClient, LLMClient


def _make_settings() -> LLMSettings:
    return LLMSettings(
        provider=LLMProvider.ANTHROPIC,
        api_key="test-key",
    )


def _make_mock_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value="Mock response")
    return client


class TestCalibrationAssistant:
    """Tests for the CalibrationAssistant class."""

    def test_init_with_explicit_client(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)
        assert assistant.client is mock_client

    def test_history_starts_empty(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)
        assert assistant.get_history() == []

    def test_clear_history(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)
        # Use the public attribute name
        assistant.conversation_history.append({"role": "user", "content": "hi"})
        assistant.clear_history()
        assert assistant.get_history() == []

    @pytest.mark.asyncio
    async def test_chat_appends_to_history(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)

        response = await assistant.chat("What is Pt/Pd?")

        assert response == "Mock response"
        history = assistant.get_history()
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_passes_system_prompt(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)

        await assistant.chat("test")

        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert "system" in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_without_history(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)

        # First message with history
        await assistant.chat("first")
        # Second message without history — no prior history included
        await assistant.chat("second", include_history=False)

        # The second call should only include the latest user message
        second_call_kwargs = mock_client.complete.call_args.kwargs
        messages = second_call_kwargs.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["content"] == "second"

    @pytest.mark.asyncio
    async def test_troubleshoot(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)

        result = await assistant.troubleshoot("Bronzing in shadows")
        assert isinstance(result, str)
        mock_client.complete.assert_called()

    @pytest.mark.asyncio
    async def test_explain_concept(self) -> None:
        mock_client = _make_mock_client()
        assistant = CalibrationAssistant(client=mock_client)

        result = await assistant.explain_concept("contrast agents")
        assert isinstance(result, str)
        mock_client.complete.assert_called()


class TestCreateAssistant:
    """Tests for the create_assistant factory."""

    def test_creates_with_api_key(self) -> None:
        assistant = create_assistant(api_key="test-key", provider="anthropic")
        assert isinstance(assistant, CalibrationAssistant)
        assert isinstance(assistant.client, AnthropicClient)

    def test_creates_with_openai_provider(self) -> None:
        from ptpd_calibration.llm.client import OpenAIClient

        assistant = create_assistant(api_key="test-key", provider="openai")
        assert isinstance(assistant.client, OpenAIClient)

    def test_raises_for_unsupported_provider(self) -> None:
        with pytest.raises(ValueError):
            create_assistant(api_key="k", provider="invalid")
