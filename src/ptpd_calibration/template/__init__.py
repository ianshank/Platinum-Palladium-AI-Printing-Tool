"""
Agentic Coding Template System

A production-ready template for building AI-powered applications with:
- Dynamic configuration management
- Centralized logging and monitoring
- Sub-agent coordination
- Error handling boundaries
- Comprehensive testing infrastructure

This module provides reusable components, factories, and patterns
for building applications deployable on Huggingface Spaces.
"""

from ptpd_calibration.template.config import (
    EnvironmentType,
    TemplateConfig,
    configure_template,
    get_template_config,
)
from ptpd_calibration.template.errors import (
    ConfigurationError,
    ErrorBoundary,
    ResourceError,
    TemplateError,
    TimeoutError,
    ValidationError,
    error_handler,
)
from ptpd_calibration.template.health import (
    ComponentHealth,
    HealthCheck,
    HealthStatus,
    health_check_endpoint,
)
from ptpd_calibration.template.logging_config import (
    LogContext,
    StructuredLogger,
    get_logger,
    setup_logging,
)

__all__ = [
    # Configuration
    "TemplateConfig",
    "EnvironmentType",
    "get_template_config",
    "configure_template",
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "StructuredLogger",
    # Errors
    "TemplateError",
    "ConfigurationError",
    "ValidationError",
    "TimeoutError",
    "ResourceError",
    "ErrorBoundary",
    "error_handler",
    # Health
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
    "health_check_endpoint",
]
