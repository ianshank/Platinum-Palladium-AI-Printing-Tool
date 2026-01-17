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
    ServiceFactory,
    EndpointFactory,
)
from ptpd_calibration.template.components.ui import (
    UIComponentBuilder,
    GradioAppBuilder,
    TabBuilder,
)
from ptpd_calibration.template.components.middleware import (
    MiddlewareChain,
    RequestMiddleware,
    ResponseMiddleware,
    LoggingMiddleware,
    TimeoutMiddleware,
    RateLimitMiddleware,
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
