"""
Reusable Component Library

Provides factory patterns and reusable components for:
- UI component builders
- API endpoint factories
- Service composition
- Middleware chains
"""

from ptpd_calibration.template.components.factories import (
    ComponentFactory,
    EndpointFactory,
    ServiceFactory,
)
from ptpd_calibration.template.components.middleware import (
    LoggingMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
    RequestMiddleware,
    ResponseMiddleware,
    TimeoutMiddleware,
)
from ptpd_calibration.template.components.ui import (
    GradioAppBuilder,
    TabBuilder,
    UIComponentBuilder,
)

__all__ = [
    # Factories
    "ComponentFactory",
    "ServiceFactory",
    "EndpointFactory",
    # UI
    "UIComponentBuilder",
    "GradioAppBuilder",
    "TabBuilder",
    # Middleware
    "MiddlewareChain",
    "RequestMiddleware",
    "ResponseMiddleware",
    "LoggingMiddleware",
    "TimeoutMiddleware",
    "RateLimitMiddleware",
]
