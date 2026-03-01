"""
Pytest fixtures for template tests.

Provides reusable fixtures for testing:
- Configuration
- Logging
- Agents
- Components
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from ptpd_calibration.template.config import (
    EnvironmentType,
    TemplateConfig,
    configure_template,
    reset_config,
)
from ptpd_calibration.template.logging_config import (
    StructuredLogger,
    setup_logging,
)
from ptpd_calibration.template.errors import (
    ErrorBoundary,
    TemplateError,
)
from ptpd_calibration.template.agents.base import (
    AgentBase,
    AgentConfig,
    AgentContext,
    AgentResult,
)
from ptpd_calibration.template.agents.tools import (
    Tool,
    ToolRegistry,
    ToolSchema,
    ToolCategory,
    ToolParameter,
)
from ptpd_calibration.template.agents.memory import (
    AgentMemory,
    MemoryType,
)
from ptpd_calibration.template.agents.coordinator import (
    AgentCoordinator,
    Task,
    TaskPriority,
    ExecutionPlan,
)
from ptpd_calibration.template.health import (
    HealthCheck,
    HealthStatus,
)
from ptpd_calibration.template.components.factories import (
    ComponentFactory,
    ServiceFactory,
    ServiceDefinition,
)
from ptpd_calibration.template.components.middleware import (
    MiddlewareChain,
    LoggingMiddleware,
    TimeoutMiddleware,
    RateLimitMiddleware,
    RequestContext,
)


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Generator[TemplateConfig, None, None]:
    """Create a test configuration."""
    config = configure_template(
        app_name="TestApp",
        version="0.0.1",
        environment=EnvironmentType.TESTING,
        debug=True,
        data_dir=temp_dir / "data",
    )
    yield config
    reset_config()


@pytest.fixture
def production_config(temp_dir: Path) -> Generator[TemplateConfig, None, None]:
    """Create a production-like configuration."""
    config = configure_template(
        app_name="ProdApp",
        version="1.0.0",
        environment=EnvironmentType.PRODUCTION,
        debug=False,
        data_dir=temp_dir / "data",
    )
    yield config
    reset_config()


@pytest.fixture
def huggingface_config(temp_dir: Path) -> Generator[TemplateConfig, None, None]:
    """Create a Huggingface Spaces configuration."""
    config = configure_template(
        app_name="HFApp",
        version="1.0.0",
        environment=EnvironmentType.HUGGINGFACE,
        data_dir=temp_dir / "data",
    )
    yield config
    reset_config()


# ============================================================================
# Logging Fixtures
# ============================================================================


@pytest.fixture
def test_logger() -> StructuredLogger:
    """Create a test logger."""
    setup_logging(level="DEBUG", console_enabled=True)
    return StructuredLogger("test")


@pytest.fixture
def json_logger(temp_dir: Path) -> StructuredLogger:
    """Create a JSON file logger."""
    log_file = temp_dir / "test.log"
    setup_logging(
        level="DEBUG",
        json_format=True,
        file_enabled=True,
        file_path=log_file,
    )
    return StructuredLogger("test.json")


# ============================================================================
# Error Handling Fixtures
# ============================================================================


@pytest.fixture
def error_boundary() -> ErrorBoundary:
    """Create a test error boundary."""
    return ErrorBoundary(
        component="test",
        default_return=None,
        reraise=False,
        log_errors=True,
    )


@pytest.fixture
def strict_error_boundary() -> ErrorBoundary:
    """Create a strict error boundary that reraises."""
    return ErrorBoundary(
        component="test_strict",
        reraise=True,
        log_errors=True,
    )


# ============================================================================
# Agent Fixtures
# ============================================================================


class MockAgent(AgentBase[dict, dict]):
    """Mock agent for testing."""

    async def _execute(
        self,
        input_data: dict,
        context: AgentContext,
    ) -> AgentResult[dict]:
        """Execute mock processing."""
        context.add_thought("Processing input")
        context.add_action("process", "mock_tool", data=input_data)

        result = {"processed": True, "input": input_data}

        context.add_observation(f"Processed: {result}")

        return AgentResult.success_result(
            output=result,
            iterations_used=context.iteration,
        )


class FailingAgent(AgentBase[dict, dict]):
    """Agent that always fails."""

    async def _execute(
        self,
        input_data: dict,
        context: AgentContext,
    ) -> AgentResult[dict]:
        """Fail with an error."""
        raise TemplateError("Test failure", error_code="TEST_ERROR")


class SlowAgent(AgentBase[dict, dict]):
    """Agent that takes a long time."""

    async def _execute(
        self,
        input_data: dict,
        context: AgentContext,
    ) -> AgentResult[dict]:
        """Sleep and then return."""
        delay = input_data.get("delay", 10)
        await asyncio.sleep(delay)
        return AgentResult.success_result(output={"delayed": delay})


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Create a mock agent configuration."""
    return AgentConfig(
        name="mock_agent",
        description="A mock agent for testing",
        timeout_seconds=5.0,
        max_iterations=10,
    )


@pytest.fixture
def mock_agent(mock_agent_config: AgentConfig) -> MockAgent:
    """Create a mock agent instance."""
    return MockAgent(mock_agent_config)


@pytest.fixture
def failing_agent() -> FailingAgent:
    """Create a failing agent instance."""
    return FailingAgent(AgentConfig(name="failing_agent"))


@pytest.fixture
def slow_agent() -> SlowAgent:
    """Create a slow agent instance."""
    return SlowAgent(AgentConfig(name="slow_agent", timeout_seconds=1.0))


@pytest.fixture
def agent_context() -> AgentContext:
    """Create an agent context."""
    return AgentContext(
        task_id="test-task-123",
        task_type="test",
        timeout_seconds=10.0,
        max_iterations=5,
    )


# ============================================================================
# Tool Fixtures
# ============================================================================


@pytest.fixture
def simple_tool() -> Tool:
    """Create a simple tool."""
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return Tool(
        schema=ToolSchema(
            name="add_numbers",
            description="Add two numbers together",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(name="a", type="int", description="First number"),
                ToolParameter(name="b", type="int", description="Second number"),
            ],
            returns="int",
        ),
        handler=add_numbers,
    )


@pytest.fixture
def async_tool() -> Tool:
    """Create an async tool."""
    async def fetch_data(url: str) -> dict:
        """Fetch data from URL."""
        await asyncio.sleep(0.1)
        return {"url": url, "status": "ok"}

    return Tool(
        schema=ToolSchema(
            name="fetch_data",
            description="Fetch data from a URL",
            category=ToolCategory.QUERY,
            parameters=[
                ToolParameter(name="url", type="str", description="URL to fetch"),
            ],
        ),
        handler=fetch_data,
        async_handler=True,
    )


@pytest.fixture
def tool_registry(simple_tool: Tool, async_tool: Tool) -> ToolRegistry:
    """Create a tool registry with sample tools."""
    registry = ToolRegistry()
    registry.add(simple_tool)
    registry.add(async_tool)
    return registry


# ============================================================================
# Memory Fixtures
# ============================================================================


@pytest.fixture
def agent_memory(temp_dir: Path) -> AgentMemory:
    """Create an agent memory instance."""
    return AgentMemory(
        working_size=10,
        long_term_path=temp_dir / "memory.json",
    )


@pytest.fixture
def populated_memory(agent_memory: AgentMemory) -> AgentMemory:
    """Create a memory instance with sample data."""
    agent_memory.add("User requested data analysis", MemoryType.USER_INPUT)
    agent_memory.add("Loaded dataset with 1000 rows", MemoryType.OBSERVATION)
    agent_memory.add("Running statistical analysis", MemoryType.ACTION)
    agent_memory.add("Mean value is 42.5", MemoryType.FACT, importance=0.8)
    agent_memory.add("Analysis may be skewed by outliers", MemoryType.THOUGHT)
    return agent_memory


# ============================================================================
# Coordinator Fixtures
# ============================================================================


@pytest.fixture
def coordinator() -> AgentCoordinator:
    """Create an agent coordinator."""
    return AgentCoordinator(max_parallel_agents=3)


@pytest.fixture
def populated_coordinator(
    coordinator: AgentCoordinator,
    mock_agent: MockAgent,
) -> AgentCoordinator:
    """Create a coordinator with registered agents."""
    coordinator.register_agent("mock", mock_agent)
    return coordinator


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task."""
    return Task(
        name="Test Task",
        task_type="mock",
        input_data={"key": "value"},
        priority=TaskPriority.NORMAL,
    )


@pytest.fixture
def sample_plan() -> ExecutionPlan:
    """Create a sample execution plan."""
    plan = ExecutionPlan(name="Test Plan")
    plan.add_step("Step 1", "mock", input_data={"step": 1})
    step2 = plan.add_step("Step 2", "mock", input_data={"step": 2})
    plan.add_step(
        "Step 3",
        "mock",
        input_data={"step": 3},
        depends_on=[step2.id],
    )
    return plan


# ============================================================================
# Health Check Fixtures
# ============================================================================


@pytest.fixture
def health_check() -> HealthCheck:
    """Create a health check instance."""
    return HealthCheck(
        app_name="TestApp",
        version="0.0.1",
        environment="testing",
    )


@pytest.fixture
def health_check_with_components(health_check: HealthCheck) -> HealthCheck:
    """Create a health check with custom components."""

    def check_database() -> tuple[HealthStatus, str]:
        return HealthStatus.HEALTHY, "Connected"

    def check_cache() -> tuple[HealthStatus, str]:
        return HealthStatus.DEGRADED, "High latency"

    health_check.register("database", check_database, critical=True)
    health_check.register("cache", check_cache, critical=False)

    return health_check


# ============================================================================
# Factory Fixtures
# ============================================================================


@pytest.fixture
def component_factory() -> ComponentFactory:
    """Create a component factory."""
    factory: ComponentFactory[Any] = ComponentFactory()

    # Register some test components
    factory.register("simple", lambda: {"type": "simple"})
    factory.register("configured", lambda cfg: {"type": "configured", "config": cfg})

    return factory


@pytest.fixture
def service_factory() -> ServiceFactory:
    """Create a service factory."""
    factory = ServiceFactory()

    # Register test services
    class DatabaseService:
        def __init__(self):
            self.connected = True

    class CacheService:
        def __init__(self, database: Any = None):
            self.database = database

    factory.register(ServiceDefinition(
        name="database",
        service_class=DatabaseService,
    ))

    factory.register(ServiceDefinition(
        name="cache",
        service_class=CacheService,
        dependencies=["database"],
    ))

    return factory


# ============================================================================
# Middleware Fixtures
# ============================================================================


@pytest.fixture
def middleware_chain() -> MiddlewareChain:
    """Create a middleware chain."""
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware(slow_request_threshold_ms=100))
    return chain


@pytest.fixture
def rate_limited_chain() -> MiddlewareChain:
    """Create a rate-limited middleware chain."""
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    chain.add(RateLimitMiddleware(max_requests=5, window_seconds=1))
    return chain


@pytest.fixture
def request_context() -> RequestContext:
    """Create a request context."""
    return RequestContext(user_id="test-user", session_id="test-session")


# ============================================================================
# Event Loop Fixture
# ============================================================================


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
