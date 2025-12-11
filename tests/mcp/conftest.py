"""Pytest fixtures for MCP tests."""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mcp_settings():
    """Create test MCP settings."""
    from ptpd_calibration.mcp.config import MCPSettings, MCPServerSettings, LMStudioSettings

    return MCPSettings(
        lm_studio=LMStudioSettings(
            host="localhost",
            port=1234,
            api_key="test-key",
            timeout_seconds=30,
        ),
        server=MCPServerSettings(
            name="test-mcp-server",
            version="1.0.0-test",
            debug_mode=True,
            log_level="DEBUG",
        ),
    )


@pytest.fixture
def lm_studio_settings():
    """Create test LM Studio settings."""
    from ptpd_calibration.mcp.config import LMStudioSettings

    return LMStudioSettings(
        host="localhost",
        port=1234,
        api_key="test-key",
        model="test-model",
        max_tokens=1024,
        temperature=0.5,
        timeout_seconds=30,
        max_retries=2,
    )


@pytest.fixture
def mock_lm_studio_client():
    """Create a mock LM Studio client."""
    from ptpd_calibration.mcp.lm_studio import MockLMStudioClient

    return MockLMStudioClient(
        default_response="This is a test response from mock LM Studio.",
        models=["test-model-1", "test-model-2"],
    )


@pytest.fixture
def sample_densities():
    """Sample density measurements for testing."""
    return [
        0.10, 0.18, 0.28, 0.40, 0.52,
        0.65, 0.78, 0.92, 1.05, 1.18,
        1.30, 1.42, 1.53, 1.64, 1.74,
        1.83, 1.91, 1.98, 2.04, 2.09, 2.12
    ]


@pytest.fixture
def sample_non_monotonic_densities():
    """Sample non-monotonic density measurements for testing."""
    return [
        0.10, 0.18, 0.28, 0.25, 0.52,  # 0.25 is non-monotonic
        0.65, 0.78, 0.92, 1.05, 1.18,
        1.30, 1.42, 1.53, 1.64, 1.74,
        1.83, 1.91, 1.98, 2.04, 2.09, 2.12
    ]


@pytest.fixture
def sample_curve_data():
    """Sample curve data for testing."""
    import numpy as np
    return {
        "input_values": list(range(256)),
        "output_values": [int(v) for v in np.clip(np.linspace(0, 255, 256) * 0.9 + 10, 0, 255)],
    }


@pytest.fixture
def tool_registry():
    """Create a test tool registry."""
    from ptpd_calibration.mcp.tools import create_mcp_tools

    return create_mcp_tools()


@pytest.fixture
def resource_registry():
    """Create a test resource registry."""
    from ptpd_calibration.mcp.resources import create_calibration_resources

    return create_calibration_resources()


@pytest.fixture
def mcp_server(mcp_settings, tool_registry, resource_registry):
    """Create a test MCP server."""
    from ptpd_calibration.mcp.server import MCPServer

    return MCPServer(
        settings=mcp_settings.server,
        tool_registry=tool_registry,
        resource_registry=resource_registry,
    )


@pytest.fixture
def mock_database():
    """Create a mock calibration database."""
    mock = MagicMock()
    mock.query.return_value = []
    mock.get_record.return_value = None
    mock.add_record.return_value = None
    return mock


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for testing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response content",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"data": [{"id": "model-1"}, {"id": "model-2"}]},
    )

    return mock_client
