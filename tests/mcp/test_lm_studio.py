"""Tests for LM Studio client."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_from_api_response(self):
        """Test creating ModelInfo from API response."""
        from ptpd_calibration.mcp.lm_studio import ModelInfo

        data = {
            "id": "llama-3.2-3b",
            "object": "model",
            "created": 1234567890,
            "owned_by": "local",
        }

        info = ModelInfo.from_api_response(data)

        assert info.id == "llama-3.2-3b"
        assert info.object == "model"
        assert info.created == 1234567890
        assert info.owned_by == "local"

    def test_from_api_response_minimal(self):
        """Test creating ModelInfo with minimal data."""
        from ptpd_calibration.mcp.lm_studio import ModelInfo

        data = {"id": "test-model"}

        info = ModelInfo.from_api_response(data)

        assert info.id == "test-model"
        assert info.object == "model"


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_from_api_response(self):
        """Test creating CompletionResult from API response."""
        from ptpd_calibration.mcp.lm_studio import CompletionResult

        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

        result = CompletionResult.from_api_response(data)

        assert result.content == "Hello! How can I help?"
        assert result.model == "test-model"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.total_tokens == 18
        assert result.id == "chatcmpl-123"

    def test_from_api_response_empty_choices(self):
        """Test handling empty choices array."""
        from ptpd_calibration.mcp.lm_studio import CompletionResult

        data = {
            "model": "test-model",
            "choices": [],
        }

        result = CompletionResult.from_api_response(data)

        assert result.content == ""
        assert result.model == "test-model"


class TestMockLMStudioClient:
    """Tests for mock LM Studio client."""

    @pytest.mark.asyncio
    async def test_complete(self, mock_lm_studio_client):
        """Test mock completion."""
        messages = [{"role": "user", "content": "Hello!"}]

        result = await mock_lm_studio_client.complete(messages)

        assert result.content == "This is a test response from mock LM Studio."
        assert result.model == "mock-model"
        assert result.finish_reason == "stop"
        assert len(mock_lm_studio_client.call_history) == 1

    @pytest.mark.asyncio
    async def test_stream(self, mock_lm_studio_client):
        """Test mock streaming."""
        messages = [{"role": "user", "content": "Hello!"}]

        chunks = []
        async for chunk in mock_lm_studio_client.stream(messages):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert full_response == "This is a test response from mock LM Studio."

    @pytest.mark.asyncio
    async def test_list_models(self, mock_lm_studio_client):
        """Test mock model listing."""
        models = await mock_lm_studio_client.list_models()

        assert len(models) == 2
        assert models[0].id == "test-model-1"
        assert models[1].id == "test-model-2"

    @pytest.mark.asyncio
    async def test_health_check(self, mock_lm_studio_client):
        """Test mock health check."""
        is_healthy = await mock_lm_studio_client.health_check()

        assert is_healthy is True


class TestLMStudioClient:
    """Tests for LM Studio client."""

    @pytest.mark.asyncio
    async def test_client_creation(self, lm_studio_settings):
        """Test client creation with settings."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        client = LMStudioClient(settings=lm_studio_settings)

        assert client.settings.host == "localhost"
        assert client.settings.port == 1234
        assert client._http_client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, lm_studio_settings):
        """Test async context manager."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        async with LMStudioClient(settings=lm_studio_settings) as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_complete_with_mock_httpx(self, lm_studio_settings, mock_httpx_client):
        """Test completion with mocked httpx."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        client = LMStudioClient(settings=lm_studio_settings)
        client._http_client = mock_httpx_client

        messages = [{"role": "user", "content": "Hello!"}]
        result = await client.complete(messages)

        assert result.content == "Test response content"
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, lm_studio_settings, mock_httpx_client):
        """Test completion with system prompt."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        client = LMStudioClient(settings=lm_studio_settings)
        client._http_client = mock_httpx_client

        messages = [{"role": "user", "content": "Hello!"}]
        result = await client.complete(
            messages,
            system="You are a helpful assistant.",
        )

        # Verify system prompt was included in request
        call_args = mock_httpx_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload.get("messages", [])) == 2
        assert payload["messages"][0]["role"] == "system"


class TestLMStudioErrors:
    """Tests for LM Studio error classes."""

    def test_lm_studio_error(self):
        """Test base error class."""
        from ptpd_calibration.mcp.lm_studio import LMStudioError

        error = LMStudioError("Test error")
        assert str(error) == "Test error"

    def test_connection_error(self):
        """Test connection error class."""
        from ptpd_calibration.mcp.lm_studio import LMStudioConnectionError

        error = LMStudioConnectionError("Cannot connect")
        assert str(error) == "Cannot connect"

    def test_api_error(self):
        """Test API error class with status code."""
        from ptpd_calibration.mcp.lm_studio import LMStudioAPIError

        error = LMStudioAPIError("Bad request", status_code=400)
        assert str(error) == "Bad request"
        assert error.status_code == 400

    def test_timeout_error(self):
        """Test timeout error class."""
        from ptpd_calibration.mcp.lm_studio import LMStudioTimeoutError

        error = LMStudioTimeoutError("Request timed out")
        assert str(error) == "Request timed out"


class TestCreateLMStudioClient:
    """Tests for client factory function."""

    def test_create_real_client(self, lm_studio_settings):
        """Test creating real client."""
        from ptpd_calibration.mcp.lm_studio import create_lm_studio_client, LMStudioClient

        client = create_lm_studio_client(settings=lm_studio_settings)

        assert isinstance(client, LMStudioClient)

    def test_create_mock_client(self):
        """Test creating mock client."""
        from ptpd_calibration.mcp.lm_studio import create_lm_studio_client, MockLMStudioClient

        client = create_lm_studio_client(use_mock=True)

        assert isinstance(client, MockLMStudioClient)

    def test_create_mock_client_with_response(self):
        """Test creating mock client with custom response."""
        from ptpd_calibration.mcp.lm_studio import create_lm_studio_client, MockLMStudioClient

        client = create_lm_studio_client(
            use_mock=True,
            mock_response="Custom test response",
        )

        assert isinstance(client, MockLMStudioClient)
        assert client.default_response == "Custom test response"
