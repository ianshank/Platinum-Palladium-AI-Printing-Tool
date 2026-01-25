"""
Integration tests for template system.

Tests that the template system properly integrates with the application,
including bootstrap initialization, health checks, and error handling.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ============================================================================
# Bootstrap Integration Tests
# ============================================================================


class TestBootstrapIntegration:
    """Tests for bootstrap module integration."""

    def test_bootstrap_initializes_correctly(self, temp_dir: Path) -> None:
        """Test that bootstrap creates a valid AppContext."""
        # Reset any existing context
        from ptpd_calibration.bootstrap import reset_app_context, initialize_app
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            assert ctx.initialized is True
            assert ctx.settings is not None
            assert ctx.template_config is not None
            assert ctx.logger is not None
            assert ctx.health is not None
            assert ctx.error_boundary is not None

        reset_app_context()

    def test_bootstrap_detects_environment(self) -> None:
        """Test that bootstrap correctly detects deployment environment."""
        from ptpd_calibration.bootstrap import _detect_environment
        from ptpd_calibration.template.config import EnvironmentType

        # Save original env vars
        orig_env = {k: os.environ.get(k) for k in ["TESTING", "PRODUCTION", "SPACE_ID", "CI"]}

        def clear_env_vars():
            """Clear environment variables that affect detection."""
            for key in ["TESTING", "PRODUCTION", "SPACE_ID", "CI"]:
                os.environ.pop(key, None)

        try:
            # Test testing environment detection (pytest is in sys.modules, so this always wins)
            clear_env_vars()
            os.environ["TESTING"] = "1"
            assert _detect_environment() == EnvironmentType.TESTING

            # Note: Production detection cannot be tested within pytest because
            # "pytest" in sys.modules returns True, which causes TESTING to be detected.
            # The detection order is: Huggingface -> Testing (pytest check) -> Production
            # So we can only test Huggingface (which has higher priority)

            # Test Huggingface environment detection (highest priority)
            clear_env_vars()
            os.environ["SPACE_ID"] = "test-space"
            assert _detect_environment() == EnvironmentType.HUGGINGFACE

            # Verify pytest detection takes precedence over env vars
            clear_env_vars()
            # Even without TESTING env var, pytest in sys.modules triggers TESTING
            result = _detect_environment()
            assert result == EnvironmentType.TESTING

        finally:
            # Restore original env vars
            clear_env_vars()
            for key, value in orig_env.items():
                if value is not None:
                    os.environ[key] = value

    def test_bootstrap_bridges_settings(self, temp_dir: Path) -> None:
        """Test that bootstrap bridges app settings to template config."""
        from ptpd_calibration.bootstrap import _bridge_settings, reset_app_context
        from ptpd_calibration.config import Settings
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        settings = Settings(
            app_name="Test App",
            debug=True,
            data_dir=temp_dir,
            log_level="DEBUG",
        )

        config = _bridge_settings(settings, EnvironmentType.TESTING)

        assert config.app_name == "Test App"
        assert config.debug is True
        assert config.logging.level == "DEBUG"
        assert config.environment == EnvironmentType.TESTING

    def test_get_app_context_lazy_initialization(self) -> None:
        """Test that get_app_context initializes on first access."""
        from ptpd_calibration.bootstrap import get_app_context, reset_app_context

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = get_app_context()
            assert ctx.initialized is True

            # Second call should return same context
            ctx2 = get_app_context()
            assert ctx is ctx2

        reset_app_context()


# ============================================================================
# Health Check Integration Tests
# ============================================================================


class TestHealthCheckIntegration:
    """Tests for health check system integration."""

    @pytest.mark.asyncio
    async def test_health_checks_registered(self, temp_dir: Path) -> None:
        """Test that all expected health checks are registered."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            # Run health checks
            report = await ctx.health.check_all()

            # Verify expected checks are present
            check_names = [c.name for c in report.components]
            assert "database" in check_names
            assert "data_directory" in check_names
            assert "deep_learning" in check_names
            assert "llm" in check_names
            assert "memory" in check_names

        reset_app_context()

    @pytest.mark.asyncio
    async def test_health_check_report_format(self, temp_dir: Path) -> None:
        """Test that health check report has correct format."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType
        from ptpd_calibration.template.health import HealthStatus

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            report = await ctx.health.check_all()

            # Verify report structure
            assert report.status in [
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED,
                HealthStatus.UNHEALTHY,
            ]
            assert len(report.components) > 0
            # HealthReport has version and environment, not app_name
            assert report.version is not None
            assert report.environment is not None
            assert report.uptime_seconds >= 0

            # Each component should have required fields
            for component in report.components:
                assert component.name is not None
                assert component.status in [
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY,
                ]

        reset_app_context()


# ============================================================================
# Error Handling Integration Tests
# ============================================================================


class TestErrorHandlingIntegration:
    """Tests for error handling integration."""

    def test_error_boundary_catches_exceptions(self) -> None:
        """Test that error boundaries properly catch and log exceptions."""
        from ptpd_calibration.bootstrap import create_component_boundary

        boundary = create_component_boundary("test_component", default_return="default")

        # Test that boundary returns default on error
        result = None
        with boundary.protect(operation="test_op"):
            raise ValueError("Test error")

        # After error, the context should complete without raising
        # The default_return is applied via the wrap decorator, not protect context

    def test_error_boundary_wrap_function(self) -> None:
        """Test error boundary wrap decorator."""
        from ptpd_calibration.bootstrap import create_component_boundary

        boundary = create_component_boundary("test_component", default_return=-1)

        @boundary.wrap
        def failing_function(x: int) -> int:
            if x < 0:
                raise ValueError("Negative input")
            return x * 2

        # Success case
        assert failing_function(5) == 10

        # Error case - should return default
        assert failing_function(-1) == -1

    def test_request_context_tracking(self) -> None:
        """Test that request context is properly tracked."""
        from ptpd_calibration.bootstrap import with_request_context, reset_app_context

        reset_app_context()

        with with_request_context(
            request_id="test-123",
            user_id="user-456",
            operation="test_operation",
        ) as ctx:
            # Context should be active
            assert ctx is not None

        reset_app_context()


# ============================================================================
# Logging Integration Tests
# ============================================================================


class TestLoggingIntegration:
    """Tests for logging system integration."""

    def test_logger_creation(self) -> None:
        """Test that loggers are created correctly."""
        from ptpd_calibration.bootstrap import get_app_logger, reset_app_context

        reset_app_context()

        logger = get_app_logger("test.module")
        assert logger is not None

        # Logger should have structured logging methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")

        reset_app_context()

    def test_structured_logging(self, capsys: pytest.CaptureFixture) -> None:
        """Test that structured logging works."""
        from ptpd_calibration.bootstrap import (
            get_app_logger,
            reset_app_context,
            initialize_app,
        )
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            initialize_app(environment=EnvironmentType.TESTING)
            logger = get_app_logger("test.structured")

            # Log with structured data
            logger.info("Test message", key1="value1", key2=42)

            # Note: In testing environment, output may go to various handlers
            # This test just verifies no exceptions are raised

        reset_app_context()


# ============================================================================
# Feature Flag Integration Tests
# ============================================================================


class TestFeatureFlagIntegration:
    """Tests for feature flag integration."""

    def test_feature_flags_detection(self) -> None:
        """Test that feature flags are detected based on environment."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            # Features should be boolean flags
            assert isinstance(ctx.features.enable_deep_learning, bool)
            assert isinstance(ctx.features.enable_llm_features, bool)
            assert isinstance(ctx.features.enable_agent_system, bool)

        reset_app_context()

    def test_is_feature_enabled(self) -> None:
        """Test feature enabled check."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            # Should work for existing features
            result = ctx.is_feature_enabled("agent_system")
            assert isinstance(result, bool)

            # Should return False for non-existent features
            result = ctx.is_feature_enabled("nonexistent_feature")
            assert result is False

        reset_app_context()


# ============================================================================
# Configuration Integration Tests
# ============================================================================


class TestConfigurationIntegration:
    """Tests for configuration system integration."""

    def test_environment_specific_config(self) -> None:
        """Test that configuration varies by environment."""
        from ptpd_calibration.bootstrap import _bridge_settings, reset_app_context
        from ptpd_calibration.config import Settings
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        settings = Settings(app_name="Test App")

        # Test development config
        dev_config = _bridge_settings(settings, EnvironmentType.DEVELOPMENT)
        assert dev_config.logging.json_format is False

        # Test production config
        prod_config = _bridge_settings(settings, EnvironmentType.PRODUCTION)
        assert prod_config.logging.json_format is True

        # Test Huggingface config
        hf_config = _bridge_settings(settings, EnvironmentType.HUGGINGFACE)
        assert hf_config.logging.json_format is True
        assert hf_config.resources.max_memory_mb == 2048  # Reduced for HF

    def test_data_path_creation(self, temp_dir: Path) -> None:
        """Test that data paths are created correctly."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.config import Settings
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        settings = Settings(app_name="Test App", data_dir=temp_dir)

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(settings=settings, environment=EnvironmentType.TESTING)

            # Test path creation
            test_path = ctx.get_data_path("subdir", "file.txt")
            assert test_path.parent.exists()
            assert str(test_path).endswith("subdir/file.txt")

        reset_app_context()


# ============================================================================
# Async Integration Tests
# ============================================================================


class TestAsyncIntegration:
    """Tests for async functionality integration."""

    @pytest.mark.asyncio
    async def test_async_health_check(self) -> None:
        """Test async health check execution."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType
        from ptpd_calibration.template.health import HealthStatus

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            # Health check should be async
            report = await ctx.health.check_all()
            assert report is not None
            assert report.status in [
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED,
                HealthStatus.UNHEALTHY,
            ]

        reset_app_context()

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self) -> None:
        """Test that multiple health checks can run concurrently."""
        from ptpd_calibration.bootstrap import initialize_app, reset_app_context
        from ptpd_calibration.template.config import EnvironmentType

        reset_app_context()

        with patch.dict(os.environ, {"TESTING": "1"}):
            ctx = initialize_app(environment=EnvironmentType.TESTING)

            # Run multiple health checks concurrently
            reports = await asyncio.gather(
                ctx.health.check_all(),
                ctx.health.check_all(),
                ctx.health.check_all(),
            )

            assert len(reports) == 3
            for report in reports:
                assert report is not None

        reset_app_context()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
