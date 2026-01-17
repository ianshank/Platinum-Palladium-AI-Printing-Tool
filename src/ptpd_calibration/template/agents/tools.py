"""
Tool Registry System

Provides a framework for defining and managing tools that agents can use:
- Tool definition with schema
- Parameter validation
- Execution tracking
- Error handling
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from ptpd_calibration.template.errors import ValidationError
from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    QUERY = "query"  # Information retrieval
    ANALYZE = "analyze"  # Data analysis
    TRANSFORM = "transform"  # Data transformation
    GENERATE = "generate"  # Content generation
    EXECUTE = "execute"  # Action execution
    MEMORY = "memory"  # Memory operations
    UTILITY = "utility"  # Utility functions


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str  # Python type as string
    description: str
    required: bool = True
    default: Any | None = None
    enum: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None


class ToolSchema(BaseModel):
    """Complete schema for a tool."""

    name: str
    description: str
    category: ToolCategory = ToolCategory.UTILITY
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "Any"
    returns_description: str = ""
    examples: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ToolResult(BaseModel, Generic[T]):
    """Result from tool execution."""

    success: bool
    output: T | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success_result(cls, output: T, **metadata: Any) -> ToolResult[T]:
        """Create successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, **metadata: Any) -> ToolResult[T]:
        """Create error result."""
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class Tool:
    """
    Tool definition with handler function.

    Attributes:
        schema: Tool schema with name, description, parameters
        handler: Function to execute the tool
        async_handler: Whether handler is async
        validate_params: Whether to validate parameters
        track_usage: Whether to track usage statistics
    """

    schema: ToolSchema
    handler: Callable[..., Any]
    async_handler: bool = False
    validate_params: bool = True
    track_usage: bool = True

    # Usage tracking
    _call_count: int = field(default=0, repr=False)
    _total_time_ms: float = field(default=0.0, repr=False)
    _error_count: int = field(default=0, repr=False)
    _last_called: datetime | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.schema.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.schema.description

    def _validate_parameters(self, **kwargs: Any) -> dict[str, Any]:
        """Validate and coerce parameters."""
        validated = {}

        for param in self.schema.parameters:
            value = kwargs.get(param.name)

            # Check required
            if value is None:
                if param.required and param.default is None:
                    raise ValidationError(
                        f"Missing required parameter: {param.name}",
                        field=param.name,
                    )
                value = param.default

            # Check enum
            if param.enum and value is not None and value not in param.enum:
                raise ValidationError(
                    f"Invalid value for {param.name}: must be one of {param.enum}",
                    field=param.name,
                    value=value,
                )

            # Check numeric bounds
            if value is not None and isinstance(value, (int, float)):
                if param.min_value is not None and value < param.min_value:
                    raise ValidationError(
                        f"Value for {param.name} must be >= {param.min_value}",
                        field=param.name,
                        value=value,
                    )
                if param.max_value is not None and value > param.max_value:
                    raise ValidationError(
                        f"Value for {param.name} must be <= {param.max_value}",
                        field=param.name,
                        value=value,
                    )

            validated[param.name] = value

        return validated

    async def execute(self, **kwargs: Any) -> ToolResult[Any]:
        """
        Execute the tool with parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with output or error
        """
        start_time = time.perf_counter()

        try:
            # Validate parameters
            if self.validate_params:
                kwargs = self._validate_parameters(**kwargs)

            # Execute handler
            if self.async_handler:
                result = await self.handler(**kwargs)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.handler(**kwargs),
                )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Track usage
            if self.track_usage:
                self._call_count += 1
                self._total_time_ms += execution_time_ms
                self._last_called = datetime.utcnow()

            logger.debug(
                f"Tool executed: {self.name}",
                execution_time_ms=execution_time_ms,
            )

            return ToolResult.success_result(
                output=result,
                execution_time_ms=execution_time_ms,
            )

        except ValidationError:
            raise

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            if self.track_usage:
                self._error_count += 1

            logger.error(f"Tool error: {self.name}", error=str(e))

            return ToolResult.error_result(
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

    def execute_sync(self, **kwargs: Any) -> ToolResult[Any]:
        """Synchronous execution wrapper."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute(**kwargs))

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": (
                self._total_time_ms / self._call_count
                if self._call_count > 0
                else 0
            ),
            "last_called": (
                self._last_called.isoformat()
                if self._last_called
                else None
            ),
        }

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for LLM function calling."""
        properties = {}
        required = []

        for param in self.schema.parameters:
            prop: dict[str, Any] = {
                "description": param.description,
            }

            # Map Python types to JSON schema types
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            prop["type"] = type_map.get(param.type, "string")

            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def tool(
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.UTILITY,
    tags: list[str] | None = None,
    validate: bool = True,
    track: bool = True,
) -> Callable[[F], Tool]:
    """
    Decorator to create a tool from a function.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category
        tags: Optional tags for organization
        validate: Whether to validate parameters
        track: Whether to track usage

    Usage:
        @tool(name="search", category=ToolCategory.QUERY)
        def search_documents(query: str, limit: int = 10) -> list[dict]:
            '''Search documents by query.'''
            ...
    """

    def decorator(func: F) -> Tool:
        # Extract function signature
        sig = inspect.signature(func)

        # Build parameters from signature
        parameters: list[ToolParameter] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            param_type = "str"
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)

            # Check if required
            required = param.default == inspect.Parameter.empty

            # Get default value
            default = None if required else param.default

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter: {param_name}",  # Could extract from docstring
                    required=required,
                    default=default,
                )
            )

        # Build schema
        schema = ToolSchema(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip(),
            category=category,
            parameters=parameters,
            returns=str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else "Any",
            tags=tags or [],
        )

        return Tool(
            schema=schema,
            handler=func,
            async_handler=asyncio.iscoroutinefunction(func),
            validate_params=validate,
            track_usage=track,
        )

    return decorator


class ToolRegistry:
    """
    Registry for managing tools.

    Provides:
    - Tool registration and lookup
    - Category-based organization
    - Bulk operations
    - Statistics

    Usage:
        registry = ToolRegistry()

        @registry.register(category=ToolCategory.QUERY)
        def search(query: str) -> list:
            ...

        result = await registry.execute("search", query="hello")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}
        self._by_category: dict[ToolCategory, list[str]] = {
            cat: [] for cat in ToolCategory
        }
        self._logger = get_logger("tools.registry")

    def add(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool
        self._by_category[tool.schema.category].append(tool.name)
        self._logger.debug(f"Registered tool: {tool.name}")

    def remove(self, name: str) -> Tool | None:
        """Remove a tool from the registry."""
        tool = self._tools.pop(name, None)
        if tool:
            self._by_category[tool.schema.category].remove(name)
            self._logger.debug(f"Removed tool: {name}")
        return tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def list_tools(
        self,
        category: ToolCategory | None = None,
        tags: list[str] | None = None,
    ) -> list[Tool]:
        """List tools, optionally filtered by category or tags."""
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.schema.category == category]

        if tags:
            tools = [
                t for t in tools
                if any(tag in t.schema.tags for tag in tags)
            ]

        return tools

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory = ToolCategory.UTILITY,
        tags: list[str] | None = None,
    ) -> Callable[[F], F]:
        """
        Decorator to register a function as a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description
            category: Tool category
            tags: Optional tags

        Usage:
            registry = ToolRegistry()

            @registry.register(category=ToolCategory.QUERY)
            def search(query: str) -> list:
                ...
        """

        def decorator(func: F) -> F:
            t = tool(
                name=name,
                description=description,
                category=category,
                tags=tags,
            )(func)
            self.add(t)
            return func

        return decorator

    async def execute(self, name: str, **kwargs: Any) -> ToolResult[Any]:
        """Execute a tool by name."""
        t = self.get(name)
        if t is None:
            return ToolResult.error_result(f"Tool not found: {name}")
        return await t.execute(**kwargs)

    def execute_sync(self, name: str, **kwargs: Any) -> ToolResult[Any]:
        """Synchronous tool execution."""
        t = self.get(name)
        if t is None:
            return ToolResult.error_result(f"Tool not found: {name}")
        return t.execute_sync(**kwargs)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [t.to_json_schema() for t in self._tools.values()]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all tools."""
        return {
            "total_tools": len(self._tools),
            "by_category": {
                cat.value: len(tools)
                for cat, tools in self._by_category.items()
            },
            "tools": {
                name: tool.get_stats()
                for name, tool in self._tools.items()
            },
        }

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._by_category = {cat: [] for cat in ToolCategory}


# Global registry instance
_global_registry: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_global_tool(
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.UTILITY,
    tags: list[str] | None = None,
) -> Callable[[F], F]:
    """Register a tool in the global registry."""
    return get_global_registry().register(
        name=name,
        description=description,
        category=category,
        tags=tags,
    )
