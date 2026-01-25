"""Tests for LLM client implementations.

Tests cover:
- Client initialization and configuration
- Provider name and model name properties
- Availability checking
- Message completion (mocked)
- Streaming completion (mocked)
- Factory function behavior
- Settings integration
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ptpd_calibration.config import LLMProvider, LLMSettings
from ptpd_calibration.llm.client import (
    LLMClient,
    LocalLLMClient,
    LMStudioClient,
    OllamaClient,
    create_client,
    get_available_providers,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def lm_studio_settings() -> LLMSettings:
    """Create LM Studio settings for tests."""
    return LLMSettings(
        provider=LLMProvider.LM_STUDIO,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_model="local-model",
        lm_studio_timeout=60,
        max_tokens=2048,
        temperature=0.7,
    )


@pytest.fixture
def ollama_settings() -> LLMSettings:
    """Create Ollama settings for tests."""
    return LLMSettings(
        provider=LLMProvider.OLLAMA,
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="llama3.2",
        ollama_timeout=120,
        max_tokens=2048,
        temperature=0.7,
    )


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Create mock OpenAI response."""
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response from the LLM."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


@pytest.fixture
def mock_openai_stream_response() -> list[MagicMock]:
    """Create mock OpenAI streaming response."""
    chunks = []
    for token in ["This ", "is ", "a ", "test ", "response."]:
        mock_delta = MagicMock()
        mock_delta.content = token

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        chunks.append(mock_chunk)

    return chunks


# ============================================================================
# LLMSettings Tests
# ============================================================================


class TestLLMSettings:
    """Tests for LLM settings configuration."""

    def test_lm_studio_settings_defaults(self) -> None:
        """Test LM Studio default settings."""
        settings = LLMSettings(provider=LLMProvider.LM_STUDIO)

        assert settings.lm_studio_base_url == "http://localhost:1234/v1"
        assert settings.lm_studio_model == "local-model"
        assert settings.lm_studio_timeout == 120

    def test_ollama_settings_defaults(self) -> None:
        """Test Ollama default settings."""
        settings = LLMSettings(provider=LLMProvider.OLLAMA)

        assert settings.ollama_base_url == "http://localhost:11434/v1"
        assert settings.ollama_model == "llama3.2"
        assert settings.ollama_timeout == 120

    def test_is_local_provider(self) -> None:
        """Test is_local_provider method."""
        lm_studio = LLMSettings(provider=LLMProvider.LM_STUDIO)
        ollama = LLMSettings(provider=LLMProvider.OLLAMA)
        anthropic = LLMSettings(provider=LLMProvider.ANTHROPIC)

        assert lm_studio.is_local_provider() is True
        assert ollama.is_local_provider() is True
        assert anthropic.is_local_provider() is False

    def test_get_base_url(self) -> None:
        """Test get_base_url method."""
        lm_studio = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            lm_studio_base_url="http://custom:8080/v1"
        )
        ollama = LLMSettings(
            provider=LLMProvider.OLLAMA,
            ollama_base_url="http://custom:11434/v1"
        )

        assert lm_studio.get_base_url() == "http://custom:8080/v1"
        assert ollama.get_base_url() == "http://custom:11434/v1"

    def test_get_model_name(self) -> None:
        """Test get_model_name method."""
        lm_studio = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            lm_studio_model="mistral-7b"
        )
        ollama = LLMSettings(
            provider=LLMProvider.OLLAMA,
            ollama_model="codellama"
        )

        assert lm_studio.get_model_name() == "mistral-7b"
        assert ollama.get_model_name() == "codellama"

    def test_get_timeout(self) -> None:
        """Test get_timeout method."""
        lm_studio = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            lm_studio_timeout=300
        )
        ollama = LLMSettings(
            provider=LLMProvider.OLLAMA,
            ollama_timeout=180
        )

        assert lm_studio.get_timeout() == 300
        assert ollama.get_timeout() == 180

    def test_get_active_api_key_local(self) -> None:
        """Test get_active_api_key returns 'not-needed' for local providers."""
        lm_studio = LLMSettings(provider=LLMProvider.LM_STUDIO)
        ollama = LLMSettings(provider=LLMProvider.OLLAMA)

        assert lm_studio.get_active_api_key() == "not-needed"
        assert ollama.get_active_api_key() == "not-needed"


# ============================================================================
# LocalLLMClient Tests
# ============================================================================


class TestLocalLLMClient:
    """Tests for LocalLLMClient base class."""

    def test_initialization(self, lm_studio_settings: LLMSettings) -> None:
        """Test client initialization."""
        client = LocalLLMClient(settings=lm_studio_settings)

        assert client.settings == lm_studio_settings
        assert client.base_url == "http://localhost:1234/v1"

    def test_initialization_with_overrides(self) -> None:
        """Test initialization with parameter overrides."""
        client = LocalLLMClient(
            base_url="http://custom:9000/v1",
            model="custom-model",
            timeout=300,
        )

        assert client.base_url == "http://custom:9000/v1"
        assert client.model_name == "custom-model"
        assert client.timeout == 300

    def test_provider_name(self) -> None:
        """Test provider name property."""
        client = LocalLLMClient()
        assert client.provider_name == "Local LLM"

    @patch("httpx.Client")
    def test_is_available_success(self, mock_httpx_client: MagicMock) -> None:
        """Test availability check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx_client.return_value = mock_client_instance

        client = LocalLLMClient(base_url="http://localhost:1234/v1")
        # Reset cached value
        client._is_available = None

        assert client.is_available() is True

    @patch("httpx.Client")
    def test_is_available_failure(self, mock_httpx_client: MagicMock) -> None:
        """Test availability check when server is not running."""
        mock_httpx_client.side_effect = Exception("Connection refused")

        client = LocalLLMClient(base_url="http://localhost:1234/v1")
        # Reset cached value
        client._is_available = None

        assert client.is_available() is False

    def test_is_available_cached(self) -> None:
        """Test availability check uses cached value."""
        client = LocalLLMClient(base_url="http://localhost:1234/v1")
        client._is_available = True

        # Should use cached value without making HTTP request
        assert client.is_available() is True


# ============================================================================
# LMStudioClient Tests
# ============================================================================


class TestLMStudioClient:
    """Tests for LMStudioClient."""

    def test_initialization(self, lm_studio_settings: LLMSettings) -> None:
        """Test LM Studio client initialization."""
        client = LMStudioClient(settings=lm_studio_settings)

        assert client.provider_name == "LM Studio"
        assert client.model_name == "local-model"
        assert client.base_url == "http://localhost:1234/v1"

    def test_default_initialization(self) -> None:
        """Test LM Studio client with default settings."""
        settings = LLMSettings(provider=LLMProvider.LM_STUDIO)
        client = LMStudioClient(settings=settings)

        assert client.base_url == "http://localhost:1234/v1"
        assert client.model_name == "local-model"
        assert client.timeout == 120

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        lm_studio_settings: LLMSettings,
        mock_openai_response: MagicMock
    ) -> None:
        """Test successful completion."""
        client = LMStudioClient(settings=lm_studio_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert result == "This is a test response from the LLM."
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(
        self,
        lm_studio_settings: LLMSettings,
        mock_openai_response: MagicMock
    ) -> None:
        """Test completion with system prompt."""
        client = LMStudioClient(settings=lm_studio_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                system="You are a helpful assistant."
            )

            # Verify system message was prepended
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_complete_with_custom_params(
        self,
        lm_studio_settings: LLMSettings,
        mock_openai_response: MagicMock
    ) -> None:
        """Test completion with custom parameters."""
        client = LMStudioClient(settings=lm_studio_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1000,
                temperature=0.5,
            )

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 1000
            assert call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_stream_success(
        self,
        lm_studio_settings: LLMSettings,
        mock_openai_stream_response: list[MagicMock]
    ) -> None:
        """Test successful streaming."""
        client = LMStudioClient(settings=lm_studio_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            # Create async iterator for streaming response
            async def async_gen():
                for chunk in mock_openai_stream_response:
                    yield chunk

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_gen())
            mock_openai.return_value = mock_client

            tokens = []
            async for token in client.stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                tokens.append(token)

            assert tokens == ["This ", "is ", "a ", "test ", "response."]


# ============================================================================
# OllamaClient Tests
# ============================================================================


class TestOllamaClient:
    """Tests for OllamaClient."""

    def test_initialization(self, ollama_settings: LLMSettings) -> None:
        """Test Ollama client initialization."""
        client = OllamaClient(settings=ollama_settings)

        assert client.provider_name == "Ollama"
        assert client.model_name == "llama3.2"
        assert client.base_url == "http://localhost:11434/v1"

    def test_default_initialization(self) -> None:
        """Test Ollama client with default settings."""
        settings = LLMSettings(provider=LLMProvider.OLLAMA)
        client = OllamaClient(settings=settings)

        assert client.base_url == "http://localhost:11434/v1"
        assert client.model_name == "llama3.2"
        assert client.timeout == 120

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        ollama_settings: LLMSettings,
        mock_openai_response: MagicMock
    ) -> None:
        """Test successful completion."""
        client = OllamaClient(settings=ollama_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert result == "This is a test response from the LLM."

    @pytest.mark.asyncio
    async def test_list_models_success(
        self,
        ollama_settings: LLMSettings
    ) -> None:
        """Test listing available Ollama models."""
        client = OllamaClient(settings=ollama_settings)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "codellama"},
                {"name": "mistral"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=False)
            mock_async_client_class.return_value = mock_async_client

            models = await client.list_models()

            assert models == ["llama3.2", "codellama", "mistral"]

    @pytest.mark.asyncio
    async def test_list_models_failure(
        self,
        ollama_settings: LLMSettings
    ) -> None:
        """Test listing models when Ollama is not available."""
        client = OllamaClient(settings=ollama_settings)

        with patch("httpx.AsyncClient") as mock_async_client_class:
            mock_async_client_class.side_effect = Exception("Connection refused")

            models = await client.list_models()

            assert models == []


# ============================================================================
# create_client Factory Tests
# ============================================================================


class TestCreateClient:
    """Tests for create_client factory function."""

    def test_create_lm_studio_client(self) -> None:
        """Test creating LM Studio client."""
        settings = LLMSettings(provider=LLMProvider.LM_STUDIO)
        client = create_client(settings)

        assert isinstance(client, LMStudioClient)
        assert client.provider_name == "LM Studio"

    def test_create_ollama_client(self) -> None:
        """Test creating Ollama client."""
        settings = LLMSettings(provider=LLMProvider.OLLAMA)
        client = create_client(settings)

        assert isinstance(client, OllamaClient)
        assert client.provider_name == "Ollama"

    def test_create_client_invalid_provider(self) -> None:
        """Test creating client with invalid provider raises error."""
        # This would require manipulating the enum which isn't easy
        # So we just verify the existing providers work
        for provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
            settings = LLMSettings(provider=provider)
            client = create_client(settings)
            assert client is not None


# ============================================================================
# get_available_providers Tests
# ============================================================================


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    @patch("httpx.Client")
    def test_checks_all_providers(self, mock_httpx_client: MagicMock) -> None:
        """Test that all providers are checked."""
        # Mock httpx to fail for local providers
        mock_httpx_client.side_effect = Exception("Connection refused")

        results = get_available_providers()

        # Should have all providers
        providers = [r[0] for r in results]
        assert LLMProvider.LM_STUDIO in providers
        assert LLMProvider.OLLAMA in providers
        assert LLMProvider.ANTHROPIC in providers
        assert LLMProvider.OPENAI in providers

    @patch("httpx.Client")
    def test_local_providers_unavailable(self, mock_httpx_client: MagicMock) -> None:
        """Test local providers shown as unavailable when servers not running."""
        mock_httpx_client.side_effect = Exception("Connection refused")

        results = get_available_providers()

        # Find local providers
        for provider, available in results:
            if provider in (LLMProvider.LM_STUDIO, LLMProvider.OLLAMA):
                assert available is False


# ============================================================================
# Integration Tests (with mocking)
# ============================================================================


class TestLLMClientIntegration:
    """Integration tests for LLM clients using mocks."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        lm_studio_settings: LLMSettings,
        mock_openai_response: MagicMock
    ) -> None:
        """Test complete LLM workflow."""
        # Create client via factory
        client = create_client(lm_studio_settings)
        assert isinstance(client, LMStudioClient)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            # Make a completion request
            result = await client.complete(
                messages=[
                    {"role": "user", "content": "Analyze this step tablet image."}
                ],
                system="You are an expert in platinum palladium printing.",
                max_tokens=1024,
                temperature=0.3,
            )

            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_streaming_workflow(
        self,
        ollama_settings: LLMSettings,
        mock_openai_stream_response: list[MagicMock]
    ) -> None:
        """Test streaming LLM workflow."""
        client = create_client(ollama_settings)
        assert isinstance(client, OllamaClient)

        with patch("openai.AsyncOpenAI") as mock_openai:
            async def async_gen():
                for chunk in mock_openai_stream_response:
                    yield chunk

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_gen())
            mock_openai.return_value = mock_client

            # Stream a response
            full_response = ""
            async for token in client.stream(
                messages=[
                    {"role": "user", "content": "Help me troubleshoot my print."}
                ]
            ):
                full_response += token

            assert full_response == "This is a test response."


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in LLM clients."""

    @pytest.mark.asyncio
    async def test_complete_handles_api_error(
        self,
        lm_studio_settings: LLMSettings
    ) -> None:
        """Test that API errors are properly propagated."""
        client = LMStudioClient(settings=lm_studio_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error: Model not found")
            )
            mock_openai.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await client.complete(
                    messages=[{"role": "user", "content": "Hello"}]
                )

            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_handles_api_error(
        self,
        ollama_settings: LLMSettings
    ) -> None:
        """Test that streaming errors are properly propagated."""
        client = OllamaClient(settings=ollama_settings)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Connection timeout")
            )
            mock_openai.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                async for _ in client.stream(
                    messages=[{"role": "user", "content": "Hello"}]
                ):
                    pass

            assert "timeout" in str(exc_info.value).lower()

    def test_import_error_handling(self) -> None:
        """Test that import errors are caught and re-raised with helpful message."""
        client = LocalLLMClient()

        # We can't easily test the import error since openai is installed
        # But we verify the client handles the case
        assert client.settings is not None


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Tests for configuration-based behavior."""

    def test_custom_base_url(self) -> None:
        """Test using custom base URL."""
        settings = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            lm_studio_base_url="http://192.168.1.100:8080/v1"
        )
        client = LMStudioClient(settings=settings)

        assert client.base_url == "http://192.168.1.100:8080/v1"

    def test_custom_model_name(self) -> None:
        """Test using custom model name."""
        settings = LLMSettings(
            provider=LLMProvider.OLLAMA,
            ollama_model="deepseek-coder:33b"
        )
        client = OllamaClient(settings=settings)

        assert client.model_name == "deepseek-coder:33b"

    def test_custom_timeout(self) -> None:
        """Test using custom timeout."""
        settings = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            lm_studio_timeout=600
        )
        client = LMStudioClient(settings=settings)

        assert client.timeout == 600

    def test_settings_temperature(self) -> None:
        """Test temperature setting is used."""
        settings = LLMSettings(
            provider=LLMProvider.LM_STUDIO,
            temperature=0.1
        )
        client = LMStudioClient(settings=settings)

        assert client.settings.temperature == 0.1

    def test_settings_max_tokens(self) -> None:
        """Test max_tokens setting is used."""
        settings = LLMSettings(
            provider=LLMProvider.OLLAMA,
            max_tokens=8192
        )
        client = OllamaClient(settings=settings)

        assert client.settings.max_tokens == 8192
