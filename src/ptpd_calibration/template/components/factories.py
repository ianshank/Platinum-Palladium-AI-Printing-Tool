"""
Factory Pattern Implementations

Provides generic factory patterns for creating components dynamically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ComponentConfig(BaseModel):
    """Base configuration for components."""

    name: str
    enabled: bool = True
    metadata: dict[str, Any] = {}


class ComponentFactory(Generic[T]):
    """
    Generic factory for creating components.

    Supports:
    - Registration of creators
    - Singleton and prototype patterns
    - Lazy initialization
    - Configuration-based creation

    Usage:
        factory = ComponentFactory[MyComponent]()

        # Register creators
        factory.register("type_a", TypeAComponent)
        factory.register("type_b", lambda cfg: TypeBComponent(cfg))

        # Create instances
        component = factory.create("type_a", config)
    """

    def __init__(self, singleton: bool = False):
        """
        Initialize factory.

        Args:
            singleton: If True, cache and reuse instances
        """
        self._creators: Dict[str, Callable[..., T]] = {}
        self._singleton = singleton
        self._instances: Dict[str, T] = {}
        self._configs: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        creator: Callable[..., T] | Type[T],
        config: Optional[Any] = None,
    ) -> "ComponentFactory[T]":
        """
        Register a component creator.

        Args:
            name: Component type name
            creator: Factory function or class
            config: Optional default configuration

        Returns:
            Self for chaining
        """
        self._creators[name] = creator
        if config is not None:
            self._configs[name] = config

        logger.debug(f"Registered component: {name}")
        return self

    def unregister(self, name: str) -> Optional[Callable[..., T]]:
        """Unregister a component creator."""
        creator = self._creators.pop(name, None)
        self._configs.pop(name, None)
        self._instances.pop(name, None)
        return creator

    def create(
        self,
        name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> T:
        """
        Create a component instance.

        Args:
            name: Component type name
            config: Configuration (overrides default)
            **kwargs: Additional arguments for creator

        Returns:
            Component instance

        Raises:
            KeyError: If component type not registered
        """
        if name not in self._creators:
            raise KeyError(f"Component not registered: {name}")

        # Check singleton cache
        if self._singleton and name in self._instances:
            return self._instances[name]

        # Get config
        final_config = config or self._configs.get(name)

        # Create instance
        creator = self._creators[name]

        if final_config is not None:
            instance = creator(final_config, **kwargs)
        elif kwargs:
            instance = creator(**kwargs)
        else:
            instance = creator()

        # Cache if singleton
        if self._singleton:
            self._instances[name] = instance

        logger.debug(f"Created component: {name}")
        return instance

    def get_or_create(
        self,
        name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> T:
        """Get existing instance or create new one."""
        if name in self._instances:
            return self._instances[name]
        return self.create(name, config, **kwargs)

    def list_types(self) -> list[str]:
        """List registered component types."""
        return list(self._creators.keys())

    def has(self, name: str) -> bool:
        """Check if component type is registered."""
        return name in self._creators

    def clear_cache(self) -> None:
        """Clear singleton cache."""
        self._instances.clear()


@dataclass
class ServiceDefinition:
    """Definition of a service for the factory."""

    name: str
    service_class: Type[Any]
    config_class: Optional[Type[BaseModel]] = None
    dependencies: list[str] = field(default_factory=list)
    singleton: bool = True
    lazy: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ServiceFactory:
    """
    Factory for creating and managing services with dependencies.

    Provides:
    - Dependency injection
    - Lifecycle management
    - Configuration validation
    - Lazy initialization

    Usage:
        factory = ServiceFactory()

        # Register services
        factory.register(ServiceDefinition(
            name="database",
            service_class=DatabaseService,
            config_class=DatabaseConfig,
        ))

        factory.register(ServiceDefinition(
            name="cache",
            service_class=CacheService,
            dependencies=["database"],
        ))

        # Get service (creates with dependencies)
        cache = factory.get("cache")
    """

    def __init__(self):
        """Initialize service factory."""
        self._definitions: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._configs: Dict[str, BaseModel] = {}
        self._initializing: set[str] = set()

    def register(
        self,
        definition: ServiceDefinition,
        config: Optional[BaseModel] = None,
    ) -> "ServiceFactory":
        """
        Register a service definition.

        Args:
            definition: Service definition
            config: Optional configuration

        Returns:
            Self for chaining
        """
        self._definitions[definition.name] = definition

        if config is not None:
            self._configs[definition.name] = config

        logger.debug(
            f"Registered service: {definition.name}",
            dependencies=definition.dependencies,
        )
        return self

    def configure(self, name: str, config: BaseModel) -> "ServiceFactory":
        """Configure a registered service."""
        if name not in self._definitions:
            raise KeyError(f"Service not registered: {name}")

        definition = self._definitions[name]
        if definition.config_class and not isinstance(config, definition.config_class):
            raise TypeError(
                f"Config must be {definition.config_class.__name__}, "
                f"got {type(config).__name__}"
            )

        self._configs[name] = config

        # Clear instance if already created
        self._instances.pop(name, None)

        return self

    def get(self, name: str) -> Any:
        """
        Get a service instance, creating if needed.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
            RuntimeError: If circular dependency detected
        """
        if name not in self._definitions:
            raise KeyError(f"Service not registered: {name}")

        definition = self._definitions[name]

        # Return cached instance if singleton
        if definition.singleton and name in self._instances:
            return self._instances[name]

        # Check for circular dependencies
        if name in self._initializing:
            raise RuntimeError(f"Circular dependency detected: {name}")

        self._initializing.add(name)

        try:
            # Create dependencies first
            deps: Dict[str, Any] = {}
            for dep_name in definition.dependencies:
                deps[dep_name] = self.get(dep_name)

            # Get config
            config = self._configs.get(name)

            # Create service
            if config and deps:
                instance = definition.service_class(config, **deps)
            elif config:
                instance = definition.service_class(config)
            elif deps:
                instance = definition.service_class(**deps)
            else:
                instance = definition.service_class()

            # Cache if singleton
            if definition.singleton:
                self._instances[name] = instance

            logger.debug(f"Created service: {name}")
            return instance

        finally:
            self._initializing.discard(name)

    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._definitions

    def is_initialized(self, name: str) -> bool:
        """Check if service is initialized."""
        return name in self._instances

    def list_services(self) -> list[str]:
        """List registered services."""
        return list(self._definitions.keys())

    def get_dependency_graph(self) -> Dict[str, list[str]]:
        """Get service dependency graph."""
        return {
            name: definition.dependencies
            for name, definition in self._definitions.items()
        }

    async def initialize_all(self) -> None:
        """Initialize all non-lazy services."""
        for name, definition in self._definitions.items():
            if not definition.lazy:
                self.get(name)

    async def shutdown_all(self) -> None:
        """Shutdown all services."""
        # Shutdown in reverse dependency order
        order = self._topological_sort()

        for name in reversed(order):
            if name in self._instances:
                instance = self._instances[name]
                if hasattr(instance, "close"):
                    try:
                        if hasattr(instance.close, "__call__"):
                            result = instance.close()
                            if hasattr(result, "__await__"):
                                await result
                    except Exception as e:
                        logger.error(f"Error closing service {name}: {e}")

        self._instances.clear()

    def _topological_sort(self) -> list[str]:
        """Sort services by dependency order."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)

            definition = self._definitions.get(name)
            if definition:
                for dep in definition.dependencies:
                    visit(dep)

            order.append(name)

        for name in self._definitions:
            visit(name)

        return order


class EndpointConfig(BaseModel):
    """Configuration for an API endpoint."""

    path: str
    methods: list[str] = ["GET"]
    name: Optional[str] = None
    description: str = ""
    tags: list[str] = []
    deprecated: bool = False
    include_in_schema: bool = True


class EndpointFactory:
    """
    Factory for creating API endpoints.

    Provides:
    - Endpoint registration
    - Route generation
    - OpenAPI documentation
    - Middleware application

    Usage:
        factory = EndpointFactory()

        @factory.endpoint(EndpointConfig(
            path="/items/{id}",
            methods=["GET"],
            tags=["items"],
        ))
        async def get_item(id: int) -> dict:
            ...

        # Create router
        router = factory.create_router()
    """

    def __init__(self, prefix: str = ""):
        """Initialize endpoint factory."""
        self._prefix = prefix
        self._endpoints: list[tuple[EndpointConfig, Callable]] = []
        self._middleware: list[Callable] = []

    def endpoint(
        self,
        config: EndpointConfig,
    ) -> Callable[[Callable], Callable]:
        """
        Decorator to register an endpoint.

        Args:
            config: Endpoint configuration

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._endpoints.append((config, func))
            return func

        return decorator

    def add_middleware(self, middleware: Callable) -> "EndpointFactory":
        """Add middleware to all endpoints."""
        self._middleware.append(middleware)
        return self

    def create_router(self) -> Any:
        """
        Create a FastAPI router with all endpoints.

        Returns:
            FastAPI APIRouter
        """
        try:
            from fastapi import APIRouter
        except ImportError:
            raise ImportError("FastAPI required for router creation")

        router = APIRouter(prefix=self._prefix)

        for config, handler in self._endpoints:
            # Apply middleware
            wrapped = handler
            for middleware in reversed(self._middleware):
                wrapped = middleware(wrapped)

            # Add route for each method
            for method in config.methods:
                router.add_api_route(
                    path=config.path,
                    endpoint=wrapped,
                    methods=[method],
                    name=config.name,
                    description=config.description,
                    tags=config.tags,
                    deprecated=config.deprecated,
                    include_in_schema=config.include_in_schema,
                )

        return router

    def list_endpoints(self) -> list[dict[str, Any]]:
        """List registered endpoints."""
        return [
            {
                "path": f"{self._prefix}{config.path}",
                "methods": config.methods,
                "name": config.name or handler.__name__,
                "tags": config.tags,
            }
            for config, handler in self._endpoints
        ]
