"""
Configuration settings for MCP server and LM Studio integration.

Uses pydantic-settings for environment-based configuration with validation.
All settings can be overridden via environment variables with PTPD_MCP_ prefix.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TransportType(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class LMStudioSettings(BaseSettings):
    """Settings for LM Studio connection.

    LM Studio provides an OpenAI-compatible API endpoint for local LLM inference.
    These settings configure the connection to a running LM Studio instance.

    Environment variables:
        PTPD_MCP_LMSTUDIO_HOST: Server hostname (default: localhost)
        PTPD_MCP_LMSTUDIO_PORT: Server port (default: 1234)
        PTPD_MCP_LMSTUDIO_API_KEY: API key if required (default: lm-studio)
        PTPD_MCP_LMSTUDIO_MODEL: Model identifier to use
        PTPD_MCP_LMSTUDIO_TIMEOUT_SECONDS: Request timeout in seconds
        PTPD_MCP_LMSTUDIO_MAX_RETRIES: Maximum retry attempts
    """

    model_config = SettingsConfigDict(
        env_prefix="PTPD_MCP_LMSTUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Connection settings
    host: str = Field(
        default="localhost",
        description="LM Studio server hostname",
    )
    port: int = Field(
        default=1234,
        ge=1,
        le=65535,
        description="LM Studio server port",
    )
    api_key: str = Field(
        default="lm-studio",
        description="API key for LM Studio (often not required for local)",
    )
    use_https: bool = Field(
        default=False,
        description="Use HTTPS for connection",
    )

    # Model settings
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use (None = use LM Studio default)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling",
    )

    # Request settings
    timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Initial delay between retries (exponential backoff)",
    )

    # Health check settings
    health_check_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Interval for health checks",
    )
    health_check_enabled: bool = Field(
        default=True,
        description="Enable periodic health checks",
    )

    @property
    def base_url(self) -> str:
        """Get the base URL for LM Studio API."""
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.host}:{self.port}/v1"

    @property
    def chat_completions_url(self) -> str:
        """Get the chat completions endpoint URL."""
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        """Get the models listing endpoint URL."""
        return f"{self.base_url}/models"


class MCPServerSettings(BaseSettings):
    """Settings for MCP server configuration.

    Environment variables:
        PTPD_MCP_SERVER_NAME: Server name for identification
        PTPD_MCP_SERVER_VERSION: Server version string
        PTPD_MCP_SERVER_TRANSPORT: Transport type (stdio, sse, websocket)
        PTPD_MCP_SERVER_HOST: Server bind address for network transports
        PTPD_MCP_SERVER_PORT: Server port for network transports
    """

    model_config = SettingsConfigDict(
        env_prefix="PTPD_MCP_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server identification
    name: str = Field(
        default="ptpd-calibration-mcp",
        description="MCP server name",
    )
    version: str = Field(
        default="1.0.0",
        description="MCP server version",
    )
    description: str = Field(
        default="Platinum/Palladium calibration tools and resources via MCP",
        description="Server description",
    )

    # Transport settings
    transport: TransportType = Field(
        default=TransportType.STDIO,
        description="MCP transport type",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Server bind address (for sse/websocket)",
    )
    port: int = Field(
        default=8765,
        ge=1,
        le=65535,
        description="Server port (for sse/websocket)",
    )

    # Capability settings
    enable_tools: bool = Field(
        default=True,
        description="Enable MCP tools capability",
    )
    enable_resources: bool = Field(
        default=True,
        description="Enable MCP resources capability",
    )
    enable_prompts: bool = Field(
        default=True,
        description="Enable MCP prompts capability",
    )

    # Logging and debugging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path (None = stderr only)",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )

    # Security settings
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed origins for CORS (network transports)",
    )
    require_authentication: bool = Field(
        default=False,
        description="Require authentication for connections",
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token (if required)",
    )

    # Performance settings
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent request handling",
    )
    request_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Request processing timeout",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @model_validator(mode="after")
    def validate_auth_settings(self) -> "MCPServerSettings":
        """Validate authentication settings are consistent."""
        if self.require_authentication and not self.auth_token:
            raise ValueError("auth_token must be set when require_authentication is True")
        return self


class MCPSettings(BaseSettings):
    """Combined MCP settings aggregating server and LM Studio configuration.

    This is the main settings class for MCP functionality.

    Environment variables: See LMStudioSettings and MCPServerSettings.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTPD_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    lm_studio: LMStudioSettings = Field(
        default_factory=LMStudioSettings,
        description="LM Studio connection settings",
    )
    server: MCPServerSettings = Field(
        default_factory=MCPServerSettings,
        description="MCP server settings",
    )

    # Feature flags
    enabled: bool = Field(
        default=True,
        description="Enable MCP functionality",
    )
    auto_connect_lm_studio: bool = Field(
        default=False,
        description="Automatically connect to LM Studio on startup",
    )


# Singleton instance cache
_settings: Optional[MCPSettings] = None


def get_mcp_settings(force_reload: bool = False) -> MCPSettings:
    """
    Get MCP settings singleton.

    Args:
        force_reload: If True, reload settings from environment.

    Returns:
        MCPSettings instance.
    """
    global _settings
    if _settings is None or force_reload:
        _settings = MCPSettings()
    return _settings


def reset_mcp_settings() -> None:
    """Reset settings singleton (useful for testing)."""
    global _settings
    _settings = None
