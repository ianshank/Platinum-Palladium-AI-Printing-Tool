"""
Tests for template configuration system.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ptpd_calibration.template.config import (
    EnvironmentType,
    FeatureFlags,
    LoggingConfig,
    ResourceLimits,
    SecurityConfig,
    TemplateConfig,
    TimeoutConfig,
    configure_template,
    get_template_config,
    reset_config,
)


class TestEnvironmentType:
    """Tests for EnvironmentType enum."""

    def test_all_environments_defined(self) -> None:
        """Test that all expected environments are defined."""
        assert EnvironmentType.DEVELOPMENT == "development"
        assert EnvironmentType.STAGING == "staging"
        assert EnvironmentType.PRODUCTION == "production"
        assert EnvironmentType.HUGGINGFACE == "huggingface"
        assert EnvironmentType.TESTING == "testing"


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.json_format is False
        assert config.file_enabled is True
        assert config.console_enabled is True
        assert config.max_bytes == 10_485_760

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LoggingConfig(
            level="DEBUG",
            json_format=True,
            file_path=Path("/tmp/test.log"),
        )
        assert config.level == "DEBUG"
        assert config.json_format is True
        assert config.file_path == Path("/tmp/test.log")


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_default_timeouts(self) -> None:
        """Test default timeout values."""
        config = TimeoutConfig()
        assert config.default_seconds == 30.0
        assert config.ui_operation_seconds == 60.0
        assert config.api_request_seconds == 120.0

    def test_custom_timeouts(self) -> None:
        """Test custom timeout values."""
        config = TimeoutConfig(
            default_seconds=10.0,
            agent_task_seconds=600.0,
        )
        assert config.default_seconds == 10.0
        assert config.agent_task_seconds == 600.0


class TestResourceLimits:
    """Tests for ResourceLimits."""

    def test_default_limits(self) -> None:
        """Test default resource limits."""
        config = ResourceLimits()
        assert config.max_memory_mb == 4096
        assert config.max_file_size_mb == 100
        assert config.max_concurrent_requests == 10

    def test_custom_limits(self) -> None:
        """Test custom resource limits."""
        config = ResourceLimits(
            max_memory_mb=2048,
            max_batch_size=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_batch_size == 100


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_security(self) -> None:
        """Test default security configuration (localhost-only CORS for security)."""
        config = SecurityConfig()
        assert config.api_key_required is False
        assert config.rate_limit_enabled is True
        # Secure default: localhost only instead of wildcard "*"
        assert config.cors_origins == ["http://localhost:7860", "http://127.0.0.1:7860"]

    def test_strict_security(self) -> None:
        """Test strict security configuration."""
        config = SecurityConfig(
            api_key_required=True,
            cors_origins=["https://example.com"],
        )
        assert config.api_key_required is True
        assert config.cors_origins == ["https://example.com"]


class TestFeatureFlags:
    """Tests for FeatureFlags."""

    def test_default_features(self) -> None:
        """Test that all features are enabled by default."""
        config = FeatureFlags()
        assert config.enable_deep_learning is True
        assert config.enable_llm_features is True
        assert config.enable_agent_system is True

    def test_minimal_features(self) -> None:
        """Test minimal feature set."""
        config = FeatureFlags.minimal()
        assert config.enable_deep_learning is False
        assert config.enable_llm_features is False
        assert config.enable_agent_system is False
        assert config.enable_caching is True  # Caching stays enabled


class TestTemplateConfig:
    """Tests for TemplateConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        reset_config()
        config = TemplateConfig()
        assert config.app_name == "Application"
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.debug is False

    def test_environment_string_parsing(self) -> None:
        """Test that environment can be parsed from string."""
        config = TemplateConfig(environment="production")
        assert config.environment == EnvironmentType.PRODUCTION

    def test_testing_environment_config(self, test_config: TemplateConfig) -> None:
        """Test testing environment configuration."""
        assert test_config.environment == EnvironmentType.TESTING
        assert test_config.debug is True
        assert test_config.logging.level == "DEBUG"

    def test_production_environment_config(self, production_config: TemplateConfig) -> None:
        """Test production environment configuration."""
        assert production_config.environment == EnvironmentType.PRODUCTION
        assert production_config.debug is False
        assert production_config.logging.level == "WARNING"
        assert production_config.logging.json_format is True

    def test_huggingface_environment_config(self, huggingface_config: TemplateConfig) -> None:
        """Test Huggingface environment configuration."""
        assert huggingface_config.environment == EnvironmentType.HUGGINGFACE
        assert huggingface_config.huggingface.is_space is True
        assert huggingface_config.data_dir == Path("/tmp/app_data")
        assert huggingface_config.resources.max_memory_mb == 2048

    def test_get_data_path(self, test_config: TemplateConfig) -> None:
        """Test data path generation."""
        path = test_config.get_data_path("subdir", "file.txt")
        assert path.parent.exists()
        assert str(path).endswith("subdir/file.txt")

    def test_is_feature_enabled(self, test_config: TemplateConfig) -> None:
        """Test feature flag checking."""
        assert test_config.is_feature_enabled("deep_learning") is True
        assert test_config.is_feature_enabled("nonexistent") is False

    def test_to_safe_dict(self, test_config: TemplateConfig) -> None:
        """Test safe dictionary export."""
        safe_dict = test_config.to_safe_dict()
        assert isinstance(safe_dict, dict)
        assert "app_name" in safe_dict


class TestConfigFunctions:
    """Tests for configuration functions."""

    def test_get_template_config_singleton(self) -> None:
        """Test that get_template_config returns same instance."""
        reset_config()
        config1 = get_template_config()
        config2 = get_template_config()
        assert config1 is config2

    def test_configure_template(self, temp_dir: Path) -> None:
        """Test configure_template function."""
        reset_config()
        config = configure_template(
            app_name="TestApp",
            version="1.0.0",
            data_dir=temp_dir,
        )
        assert config.app_name == "TestApp"
        assert config.version == "1.0.0"
        assert config.data_dir == temp_dir

    def test_reset_config(self) -> None:
        """Test reset_config function."""
        configure_template(app_name="CustomApp")
        reset_config()
        config = get_template_config()
        assert config.app_name == "Application"  # Back to default

    @patch.dict(os.environ, {"SPACE_ID": "user/test-space"})
    def test_huggingface_auto_detection(self) -> None:
        """Test automatic Huggingface environment detection."""
        reset_config()
        config = TemplateConfig()
        # Note: This tests the auto-detection in HuggingfaceConfig
        assert config.huggingface.space_id == "user/test-space"
        assert config.huggingface.is_space is True
