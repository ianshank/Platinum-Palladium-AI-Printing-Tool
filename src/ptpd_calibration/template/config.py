"""
Template Configuration System

Environment-based configuration with no hardcoded values.
Supports multiple deployment environments including Huggingface Spaces.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentType(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HUGGINGFACE = "huggingface"
    TESTING = "testing"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        description="Log format string"
    )
    json_format: bool = Field(default=False, description="Use JSON structured logging")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    max_bytes: int = Field(default=10_485_760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup files")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    include_trace: bool = Field(default=False, description="Include stack traces")


class TimeoutConfig(BaseModel):
    """Timeout configuration for operations."""

    default_seconds: float = Field(default=30.0, description="Default timeout")
    ui_operation_seconds: float = Field(default=60.0, description="UI operation timeout")
    api_request_seconds: float = Field(default=120.0, description="API request timeout")
    agent_task_seconds: float = Field(default=300.0, description="Agent task timeout")
    model_inference_seconds: float = Field(default=180.0, description="Model inference timeout")
    file_upload_seconds: float = Field(default=60.0, description="File upload timeout")


class ResourceLimits(BaseModel):
    """Resource limit configuration."""

    max_memory_mb: int = Field(default=4096, description="Max memory usage in MB")
    max_file_size_mb: int = Field(default=100, description="Max file size in MB")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")
    max_batch_size: int = Field(default=50, description="Max batch processing size")
    max_agent_iterations: int = Field(default=10, description="Max agent iterations")
    cache_size_mb: int = Field(default=512, description="Cache size in MB")


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key_required: bool = Field(default=False, description="Require API key")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    validate_inputs: bool = Field(default=True, description="Validate all inputs")
    sanitize_outputs: bool = Field(default=True, description="Sanitize outputs")


class FeatureFlags(BaseModel):
    """Feature flag configuration for graceful degradation."""

    enable_deep_learning: bool = Field(default=True, description="Enable DL features")
    enable_llm_features: bool = Field(default=True, description="Enable LLM features")
    enable_agent_system: bool = Field(default=True, description="Enable agent system")
    enable_file_persistence: bool = Field(default=True, description="Enable file persistence")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_advanced_features: bool = Field(default=True, description="Enable advanced features")

    @classmethod
    def minimal(cls) -> "FeatureFlags":
        """Create minimal feature set for constrained environments."""
        return cls(
            enable_deep_learning=False,
            enable_llm_features=False,
            enable_agent_system=False,
            enable_file_persistence=False,
            enable_monitoring=False,
            enable_caching=True,
            enable_advanced_features=False,
        )


class HuggingfaceConfig(BaseModel):
    """Huggingface Spaces specific configuration."""

    space_id: Optional[str] = Field(default=None, description="HF Space ID")
    is_space: bool = Field(default=False, description="Running in HF Space")
    persistent_storage_path: Optional[Path] = Field(
        default=None,
        description="Persistent storage path (if enabled)"
    )
    use_zero_gpu: bool = Field(default=False, description="Use ZeroGPU")

    @model_validator(mode="after")
    def detect_huggingface_environment(self) -> "HuggingfaceConfig":
        """Auto-detect Huggingface environment."""
        if os.environ.get("SPACE_ID"):
            self.space_id = os.environ.get("SPACE_ID")
            self.is_space = True
        if os.environ.get("SPACE_AUTHOR_NAME"):
            self.is_space = True
        return self


class SubAgentConfig(BaseModel):
    """Sub-agent configuration."""

    name: str = Field(..., description="Agent name")
    enabled: bool = Field(default=True, description="Agent enabled")
    timeout_seconds: float = Field(default=60.0, description="Agent timeout")
    max_retries: int = Field(default=3, description="Max retry attempts")
    priority: int = Field(default=5, description="Agent priority (1-10)")
    tools: list[str] = Field(default_factory=list, description="Available tools")

    class Config:
        extra = "allow"


class AgentSystemConfig(BaseModel):
    """Agent system configuration."""

    coordinator_enabled: bool = Field(default=True, description="Enable coordinator")
    parallel_execution: bool = Field(default=True, description="Allow parallel execution")
    max_parallel_agents: int = Field(default=3, description="Max parallel agents")
    memory_window_size: int = Field(default=10, description="Memory window size")
    planning_enabled: bool = Field(default=True, description="Enable planning")
    reflection_enabled: bool = Field(default=True, description="Enable reflection")
    sub_agents: list[SubAgentConfig] = Field(
        default_factory=list,
        description="Sub-agent configurations"
    )


class UIConfig(BaseModel):
    """UI configuration."""

    title: str = Field(default="Application", description="Application title")
    description: str = Field(default="", description="Application description")
    theme: str = Field(default="default", description="UI theme")
    show_api_info: bool = Field(default=False, description="Show API info")
    share: bool = Field(default=False, description="Enable sharing")
    server_port: int = Field(default=7860, description="Server port")
    server_name: str = Field(default="0.0.0.0", description="Server name")
    max_file_size_mb: int = Field(default=100, description="Max upload size")
    analytics_enabled: bool = Field(default=False, description="Enable analytics")


class APIConfig(BaseModel):
    """API configuration."""

    enabled: bool = Field(default=True, description="Enable API")
    prefix: str = Field(default="/api/v1", description="API prefix")
    docs_enabled: bool = Field(default=True, description="Enable docs")
    docs_url: str = Field(default="/docs", description="Docs URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI URL")


class TestingConfig(BaseModel):
    """Testing configuration."""

    coverage_threshold: float = Field(default=80.0, description="Coverage threshold %")
    parallel_workers: int = Field(default=4, description="Parallel test workers")
    timeout_seconds: float = Field(default=300.0, description="Test timeout")
    markers: list[str] = Field(
        default=["unit", "integration", "e2e", "api", "slow"],
        description="Test markers"
    )
    fixtures_path: Path = Field(default=Path("tests/fixtures"), description="Fixtures path")


class TemplateConfig(BaseSettings):
    """
    Main template configuration.

    All values can be overridden via environment variables with TEMPLATE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEMPLATE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Core settings
    app_name: str = Field(default="Application", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # Nested configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    huggingface: HuggingfaceConfig = Field(default_factory=HuggingfaceConfig)
    agents: AgentSystemConfig = Field(default_factory=AgentSystemConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)

    # Data paths
    data_dir: Path = Field(
        default=Path.home() / ".app_data",
        description="Data directory"
    )
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory")
    temp_dir: Optional[Path] = Field(default=None, description="Temp directory")

    @field_validator("environment", mode="before")
    @classmethod
    def parse_environment(cls, v: Any) -> EnvironmentType:
        """Parse environment from string."""
        if isinstance(v, str):
            return EnvironmentType(v.lower())
        return v

    @model_validator(mode="after")
    def configure_for_environment(self) -> "TemplateConfig":
        """Apply environment-specific defaults."""
        if self.environment == EnvironmentType.HUGGINGFACE or self.huggingface.is_space:
            self._configure_huggingface()
        elif self.environment == EnvironmentType.PRODUCTION:
            self._configure_production()
        elif self.environment == EnvironmentType.TESTING:
            self._configure_testing()
        return self

    def _configure_huggingface(self) -> None:
        """Configure for Huggingface Spaces."""
        self.environment = EnvironmentType.HUGGINGFACE
        self.huggingface.is_space = True

        # Use /tmp for ephemeral storage
        self.data_dir = Path("/tmp/app_data")
        self.cache_dir = Path("/tmp/cache")
        self.temp_dir = Path("/tmp")

        # Adjust logging for HF
        self.logging.file_enabled = True
        self.logging.file_path = Path("/tmp/app.log")
        self.logging.max_bytes = 5_242_880  # 5MB

        # Reduce resource limits
        self.resources.max_memory_mb = 2048
        self.resources.max_concurrent_requests = 5
        self.resources.cache_size_mb = 256

        # Security for public space
        self.security.cors_origins = ["*"]
        self.security.rate_limit_enabled = True

        # UI settings
        self.ui.share = False
        self.ui.server_name = "0.0.0.0"

    def _configure_production(self) -> None:
        """Configure for production."""
        self.debug = False
        self.logging.level = "WARNING"
        self.logging.json_format = True
        self.security.validate_inputs = True
        self.security.sanitize_outputs = True
        self.ui.show_api_info = False

    def _configure_testing(self) -> None:
        """Configure for testing."""
        self.debug = True
        self.logging.level = "DEBUG"
        self.logging.console_enabled = True
        self.resources.max_memory_mb = 1024
        self.timeouts.default_seconds = 10.0
        self.data_dir = Path("/tmp/test_data")

    def get_data_path(self, *parts: str) -> Path:
        """Get path within data directory, creating if needed."""
        path = self.data_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return getattr(self.features, f"enable_{feature}", False)

    def to_safe_dict(self) -> dict[str, Any]:
        """Export config without sensitive values."""
        data = self.model_dump()
        # Mask sensitive values
        sensitive_keys = {"api_key", "secret", "token", "password", "credential"}

        def mask_sensitive(d: dict) -> dict:
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = mask_sensitive(v)
                elif any(s in k.lower() for s in sensitive_keys):
                    d[k] = "***MASKED***"
            return d

        return mask_sensitive(data)


# Global configuration instance
_config: Optional[TemplateConfig] = None


@lru_cache(maxsize=1)
def get_template_config() -> TemplateConfig:
    """Get the global template configuration."""
    global _config
    if _config is None:
        _config = TemplateConfig()
    return _config


def configure_template(**kwargs: Any) -> TemplateConfig:
    """Configure template with custom settings."""
    global _config
    get_template_config.cache_clear()
    _config = TemplateConfig(**kwargs)
    return _config


def reset_config() -> None:
    """Reset configuration to defaults (for testing)."""
    global _config
    get_template_config.cache_clear()
    _config = None
