"""Tests for MCP configuration."""

import os
import pytest
from unittest.mock import patch


class TestLMStudioSettings:
    """Tests for LM Studio settings."""

    def test_default_settings(self):
        """Test default settings values."""
        from ptpd_calibration.mcp.config import LMStudioSettings

        settings = LMStudioSettings()

        assert settings.host == "localhost"
        assert settings.port == 1234
        assert settings.api_key == "lm-studio"
        assert settings.use_https is False
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
        assert settings.timeout_seconds == 120
        assert settings.max_retries == 3

    def test_base_url_http(self):
        """Test HTTP base URL generation."""
        from ptpd_calibration.mcp.config import LMStudioSettings

        settings = LMStudioSettings(host="myhost", port=5000, use_https=False)

        assert settings.base_url == "http://myhost:5000/v1"
        assert settings.chat_completions_url == "http://myhost:5000/v1/chat/completions"
        assert settings.models_url == "http://myhost:5000/v1/models"

    def test_base_url_https(self):
        """Test HTTPS base URL generation."""
        from ptpd_calibration.mcp.config import LMStudioSettings

        settings = LMStudioSettings(host="secure.host", port=443, use_https=True)

        assert settings.base_url == "https://secure.host:443/v1"

    def test_environment_override(self):
        """Test settings can be overridden via environment variables."""
        from ptpd_calibration.mcp.config import LMStudioSettings

        with patch.dict(os.environ, {
            "PTPD_MCP_LMSTUDIO_HOST": "custom-host",
            "PTPD_MCP_LMSTUDIO_PORT": "9999",
            "PTPD_MCP_LMSTUDIO_MAX_TOKENS": "2048",
        }):
            settings = LMStudioSettings()

            assert settings.host == "custom-host"
            assert settings.port == 9999
            assert settings.max_tokens == 2048

    def test_port_validation(self):
        """Test port number validation."""
        from ptpd_calibration.mcp.config import LMStudioSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LMStudioSettings(port=0)

        with pytest.raises(ValidationError):
            LMStudioSettings(port=70000)

    def test_temperature_validation(self):
        """Test temperature range validation."""
        from ptpd_calibration.mcp.config import LMStudioSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LMStudioSettings(temperature=-0.1)

        with pytest.raises(ValidationError):
            LMStudioSettings(temperature=2.1)


class TestMCPServerSettings:
    """Tests for MCP server settings."""

    def test_default_settings(self):
        """Test default settings values."""
        from ptpd_calibration.mcp.config import MCPServerSettings, TransportType

        settings = MCPServerSettings()

        assert settings.name == "ptpd-calibration-mcp"
        assert settings.version == "1.0.0"
        assert settings.transport == TransportType.STDIO
        assert settings.host == "127.0.0.1"
        assert settings.port == 8765
        assert settings.enable_tools is True
        assert settings.enable_resources is True

    def test_log_level_validation(self):
        """Test log level validation."""
        from ptpd_calibration.mcp.config import MCPServerSettings
        from pydantic import ValidationError

        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = MCPServerSettings(log_level=level)
            assert settings.log_level == level

        # Invalid log level
        with pytest.raises(ValidationError):
            MCPServerSettings(log_level="INVALID")

    def test_auth_validation(self):
        """Test authentication settings validation."""
        from ptpd_calibration.mcp.config import MCPServerSettings
        from pydantic import ValidationError

        # Valid: auth disabled
        settings = MCPServerSettings(require_authentication=False)
        assert settings.require_authentication is False

        # Valid: auth enabled with token
        settings = MCPServerSettings(
            require_authentication=True,
            auth_token="my-secret-token"
        )
        assert settings.require_authentication is True
        assert settings.auth_token == "my-secret-token"

        # Invalid: auth enabled without token
        with pytest.raises(ValidationError):
            MCPServerSettings(require_authentication=True, auth_token=None)


class TestMCPSettings:
    """Tests for combined MCP settings."""

    def test_default_settings(self):
        """Test default combined settings."""
        from ptpd_calibration.mcp.config import MCPSettings

        settings = MCPSettings()

        assert settings.enabled is True
        assert settings.lm_studio is not None
        assert settings.server is not None

    def test_get_mcp_settings_singleton(self):
        """Test settings singleton behavior."""
        from ptpd_calibration.mcp.config import get_mcp_settings, reset_mcp_settings

        reset_mcp_settings()

        settings1 = get_mcp_settings()
        settings2 = get_mcp_settings()

        assert settings1 is settings2

    def test_force_reload(self):
        """Test force reload creates new instance."""
        from ptpd_calibration.mcp.config import get_mcp_settings, reset_mcp_settings

        reset_mcp_settings()

        settings1 = get_mcp_settings()
        settings2 = get_mcp_settings(force_reload=True)

        assert settings1 is not settings2


class TestTransportType:
    """Tests for transport type enum."""

    def test_transport_types(self):
        """Test all transport types are defined."""
        from ptpd_calibration.mcp.config import TransportType

        assert TransportType.STDIO.value == "stdio"
        assert TransportType.SSE.value == "sse"
        assert TransportType.WEBSOCKET.value == "websocket"
