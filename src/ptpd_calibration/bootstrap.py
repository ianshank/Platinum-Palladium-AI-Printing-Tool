"""
Application Bootstrap Module

Initializes the application with template system integration:
- Configuration bridging (template + application settings)
- Logging setup with structured output
- Error handling with boundaries
- Health check registration
- Feature flag management

Usage:
    from ptpd_calibration.bootstrap import initialize_app, get_app_context

    # Initialize at application startup
    ctx = initialize_app()

    # Access initialized components
    logger = ctx.logger
    health = ctx.health
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ptpd_calibration.config import Settings, get_settings
from ptpd_calibration.template.config import (
    EnvironmentType,
    FeatureFlags,
    TemplateConfig,
    configure_template,
)
from ptpd_calibration.template.errors import (
    ErrorBoundary,
)
from ptpd_calibration.template.health import HealthCheck, HealthStatus
from ptpd_calibration.template.logging_config import (
    LogContext,
    StructuredLogger,
    get_logger,
    setup_logging,
)


@dataclass
class AppContext:
    """
    Application context holding initialized components.

    Provides access to:
    - Configuration (both app settings and template config)
    - Logger
    - Health check system
    - Error boundaries
    - Feature flags
    """

    settings: Settings
    template_config: TemplateConfig
    logger: StructuredLogger
    health: HealthCheck
    error_boundary: ErrorBoundary
    features: FeatureFlags
    initialized: bool = False
    _health_checks_registered: bool = field(default=False, repr=False)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return getattr(self.features, f"enable_{feature}", False)

    def get_data_path(self, *parts: str) -> Path:
        """Get a path within the data directory."""
        path = self.settings.data_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Global application context with thread-safe access
_app_context: AppContext | None = None
_app_context_lock = threading.Lock()


def _detect_environment() -> EnvironmentType:
    """Detect the deployment environment."""
    # Huggingface Spaces detection
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_AUTHOR_NAME"):
        return EnvironmentType.HUGGINGFACE

    # Testing detection
    if "pytest" in sys.modules or os.environ.get("TESTING"):
        return EnvironmentType.TESTING

    # Production detection
    if os.environ.get("PRODUCTION") or os.environ.get("ENV") == "production":
        return EnvironmentType.PRODUCTION

    # Staging detection
    if os.environ.get("STAGING") or os.environ.get("ENV") == "staging":
        return EnvironmentType.STAGING

    # Default to development
    return EnvironmentType.DEVELOPMENT


def _bridge_settings(settings: Settings, environment: EnvironmentType) -> TemplateConfig:
    """
    Bridge application settings to template configuration.

    Maps existing Settings values to TemplateConfig for consistency.
    """
    # Determine log level
    log_level = settings.log_level.upper()
    if settings.debug:
        log_level = "DEBUG"

    # Determine JSON logging based on environment
    json_format = environment in (EnvironmentType.PRODUCTION, EnvironmentType.HUGGINGFACE)

    # Determine data directory based on environment
    if environment == EnvironmentType.HUGGINGFACE:
        data_dir = Path("/tmp/ptpd_data")
    else:
        data_dir = settings.data_dir

    # Feature flags based on optional imports
    features = FeatureFlags()

    # Check deep learning availability
    try:
        import torch
        features.enable_deep_learning = True
    except ImportError:
        features.enable_deep_learning = False

    # Check LLM availability
    if settings.llm.get_active_api_key():
        features.enable_llm_features = True
    else:
        features.enable_llm_features = False

    # Create template config
    return configure_template(
        app_name=settings.app_name,
        version="1.1.0",  # From package
        environment=environment,
        debug=settings.debug,
        data_dir=data_dir,
        logging={
            "level": log_level,
            "json_format": json_format,
            "console_enabled": True,
            "file_enabled": environment != EnvironmentType.TESTING,
            "file_path": data_dir / "app.log" if environment != EnvironmentType.TESTING else None,
        },
        timeouts={
            "default_seconds": 30.0,
            "ui_operation_seconds": float(settings.agent.tool_timeout_seconds),
            "api_request_seconds": float(settings.llm.timeout_seconds),
            "agent_task_seconds": float(settings.agent.planning_timeout_seconds),
        },
        resources={
            "max_memory_mb": 4096 if environment != EnvironmentType.HUGGINGFACE else 2048,
            "max_file_size_mb": settings.api.max_upload_size_mb,
            "max_concurrent_requests": 10 if environment != EnvironmentType.HUGGINGFACE else 5,
            "max_agent_iterations": settings.agent.max_iterations,
        },
        security={
            "cors_origins": settings.api.cors_origins,
            "rate_limit_enabled": True,
            "rate_limit_requests": settings.api.rate_limit_per_minute,
        },
        features=features.model_dump(),
        ui={
            "title": settings.app_name,
            "server_port": 7860,
            "server_name": "0.0.0.0",
            "max_file_size_mb": settings.api.max_upload_size_mb,
        },
        api={
            "enabled": True,
            "prefix": "/api/v1",
            "docs_enabled": settings.debug,
        },
    )


def _register_health_checks(health: HealthCheck, settings: Settings) -> None:
    """Register health checks for application components."""

    # Database health check
    def check_database() -> tuple[HealthStatus, str]:
        try:
            db_path = settings.data_management.database_path or settings.data_dir / "ptpd.db"
            if db_path.exists():
                return HealthStatus.HEALTHY, f"Database accessible: {db_path.name}"
            return HealthStatus.DEGRADED, "Database not initialized"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database error: {e}"

    health.register("database", check_database, critical=False)

    # Data directory health check
    def check_data_directory() -> tuple[HealthStatus, str]:
        try:
            data_dir = settings.data_dir
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = data_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            return HealthStatus.HEALTHY, f"Data directory accessible: {data_dir}"
        except Exception as e:
            return HealthStatus.DEGRADED, f"Data directory issue: {e}"

    health.register("data_directory", check_data_directory, critical=False)

    # Deep learning health check
    def check_deep_learning() -> tuple[HealthStatus, str]:
        try:
            import torch

            if torch.cuda.is_available():
                device = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "MPS (Apple Silicon)"
            else:
                device = "CPU"

            return HealthStatus.HEALTHY, f"PyTorch available: {device}"
        except ImportError:
            return HealthStatus.DEGRADED, "PyTorch not installed"
        except Exception as e:
            return HealthStatus.DEGRADED, f"PyTorch error: {e}"

    health.register("deep_learning", check_deep_learning, critical=False)

    # LLM health check
    def check_llm() -> tuple[HealthStatus, str]:
        api_key = settings.llm.get_active_api_key()
        if api_key:
            provider = settings.llm.provider.value
            return HealthStatus.HEALTHY, f"LLM configured: {provider}"
        return HealthStatus.DEGRADED, "LLM API key not configured"

    health.register("llm", check_llm, critical=False)

    # Memory health check
    def check_memory() -> tuple[HealthStatus, str]:
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return HealthStatus.UNHEALTHY, f"Memory critical: {memory.percent}%"
            elif memory.percent > 85:
                return HealthStatus.DEGRADED, f"Memory high: {memory.percent}%"
            return HealthStatus.HEALTHY, f"Memory usage: {memory.percent}%"
        except ImportError:
            return HealthStatus.HEALTHY, "Memory check unavailable (psutil not installed)"

    health.register("memory", check_memory, critical=True)


def initialize_app(
    settings: Settings | None = None,
    environment: EnvironmentType | None = None,
) -> AppContext:
    """
    Initialize the application with all template system components.

    Args:
        settings: Optional application settings (uses get_settings() if None)
        environment: Optional environment override (auto-detected if None)

    Returns:
        AppContext with all initialized components
    """
    global _app_context

    # Get settings
    if settings is None:
        settings = get_settings()

    # Detect environment
    if environment is None:
        environment = _detect_environment()

    # Bridge settings to template config
    template_config = _bridge_settings(settings, environment)

    # Setup logging
    setup_logging(
        level=template_config.logging.level,
        json_format=template_config.logging.json_format,
        file_enabled=template_config.logging.file_enabled,
        file_path=template_config.logging.file_path,
        max_bytes=template_config.logging.max_bytes,
        backup_count=template_config.logging.backup_count,
        console_enabled=template_config.logging.console_enabled,
    )

    # Get logger
    logger = get_logger("ptpd_calibration")
    logger.info(
        "Application initializing",
        app_name=settings.app_name,
        environment=environment.value,
        debug=settings.debug,
    )

    # Create health check system
    health = HealthCheck.configure(
        app_name=settings.app_name,
        version="1.1.0",
        environment=environment.value,
    )

    # Register health checks
    _register_health_checks(health, settings)

    # Create error boundary for the application
    error_boundary = ErrorBoundary(
        component="application",
        reraise=False,
        log_errors=True,
    )

    # Create app context
    _app_context = AppContext(
        settings=settings,
        template_config=template_config,
        logger=logger,
        health=health,
        error_boundary=error_boundary,
        features=template_config.features,
        initialized=True,
    )

    # Ensure directories exist
    settings.ensure_directories()

    logger.info(
        "Application initialized",
        data_dir=str(settings.data_dir),
        features={
            "deep_learning": _app_context.features.enable_deep_learning,
            "llm": _app_context.features.enable_llm_features,
            "agents": _app_context.features.enable_agent_system,
        },
    )

    return _app_context


def get_app_context() -> AppContext:
    """
    Get the application context, initializing if necessary.

    Thread-safe implementation using double-checked locking pattern.

    Returns:
        AppContext with all initialized components
    """
    global _app_context

    # Fast path: context already initialized
    if _app_context is not None and _app_context.initialized:
        return _app_context

    # Slow path: acquire lock and initialize
    with _app_context_lock:
        # Double-check after acquiring lock
        if _app_context is None or not _app_context.initialized:
            return initialize_app()
        return _app_context


def reset_app_context() -> None:
    """Reset the application context (for testing)."""
    global _app_context
    with _app_context_lock:
        _app_context = None


# Convenience functions for common operations


def get_app_logger(name: str) -> StructuredLogger:
    """Get a logger for a module."""
    return get_logger(name)


def get_app_health() -> HealthCheck:
    """Get the health check system."""
    return get_app_context().health


def create_component_boundary(
    component: str,
    default_return: Any = None,
) -> ErrorBoundary:
    """
    Create an error boundary for a component.

    Args:
        component: Component name
        default_return: Value to return on error

    Returns:
        Configured ErrorBoundary
    """
    return ErrorBoundary(
        component=component,
        default_return=default_return,
        reraise=False,
        log_errors=True,
    )


def with_request_context(
    request_id: str | None = None,
    user_id: str | None = None,
    operation: str | None = None,
) -> LogContext:
    """
    Create a log context for a request.

    Args:
        request_id: Optional request ID (auto-generated if None)
        user_id: Optional user ID
        operation: Optional operation name

    Returns:
        LogContext for use with 'with' statement
    """
    return LogContext(
        request_id=request_id,
        user_id=user_id,
        operation=operation,
        auto_request_id=request_id is None,
    )
