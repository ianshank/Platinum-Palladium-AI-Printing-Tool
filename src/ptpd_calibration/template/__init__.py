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
    TemplateConfig,
    EnvironmentType,
    get_template_config,
    configure_template,
)
from ptpd_calibration.template.logging_config import (
    setup_logging,
    get_logger,
    LogContext,
    StructuredLogger,
)
from ptpd_calibration.template.errors import (
    TemplateError,
    ConfigurationError,
    ValidationError,
    TimeoutError,
    ResourceError,
    ErrorBoundary,
    error_handler,
)
from ptpd_calibration.template.health import (
    HealthCheck,
    HealthStatus,
    ComponentHealth,
    health_check_endpoint,
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
