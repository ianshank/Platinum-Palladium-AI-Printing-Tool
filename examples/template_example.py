#!/usr/bin/env python3
"""
Example: Using the Agentic Coding Template System

This example demonstrates how to use all components of the template system
to build a production-ready application deployable on Huggingface Spaces.

Run with:
    PYTHONPATH=src python examples/template_example.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

# ============================================================================
# 1. Configuration Setup
# ============================================================================

from ptpd_calibration.template.config import (
    EnvironmentType,
    FeatureFlags,
    LoggingConfig,
    ResourceLimits,
    configure_template,
    get_template_config,
)

# Configure the application
config = configure_template(
    app_name="Example Application",
    version="1.0.0",
    environment=EnvironmentType.DEVELOPMENT,  # Change to HUGGINGFACE for deployment
    debug=True,
    logging=LoggingConfig(
        level="DEBUG",
        console_enabled=True,
        json_format=False,  # Use JSON in production
    ),
    resources=ResourceLimits(
        max_memory_mb=2048,
        max_concurrent_requests=5,
    ),
    features=FeatureFlags(
        enable_deep_learning=True,
        enable_llm_features=True,
        enable_agent_system=True,
    ),
)

print(f"Configured: {config.app_name} v{config.version}")
print(f"Environment: {config.environment.value}")

# ============================================================================
# 2. Logging Setup
# ============================================================================

from ptpd_calibration.template.logging_config import (
    LogContext,
    get_logger,
    logged,
    setup_logging,
)

# Setup logging based on config
setup_logging(
    level=config.logging.level,
    json_format=config.logging.json_format,
    console_enabled=config.logging.console_enabled,
)

# Get a logger
logger = get_logger(__name__)
logger.info("Application starting", environment=config.environment.value)

# ============================================================================
# 3. Error Handling
# ============================================================================

from ptpd_calibration.template.errors import (
    ErrorBoundary,
    TemplateError,
    ValidationError,
    error_handler,
    retry_on_error,
)

# Create error boundary for a component
processor_boundary = ErrorBoundary(
    component="data_processor",
    default_return=None,
    reraise=False,
    log_errors=True,
)


@error_handler(component="validator", default_return=False)
def validate_input(data: dict) -> bool:
    """Validate input data with automatic error handling."""
    if "name" not in data:
        raise ValidationError("Missing required field", field="name")
    return True


@retry_on_error(max_retries=3, retry_delay=0.5, exponential_backoff=True)
async def fetch_data(url: str) -> dict:
    """Fetch data with automatic retry on failure."""
    logger.info("Fetching data", url=url)
    # Simulate API call
    if not hasattr(fetch_data, "_attempt"):
        fetch_data._attempt = 0
    fetch_data._attempt += 1

    if fetch_data._attempt < 2:
        raise TemplateError("Temporary network error")

    return {"data": "success", "url": url}


# ============================================================================
# 4. Agent System
# ============================================================================

from ptpd_calibration.template.agents import (
    AgentBase,
    AgentConfig,
    AgentContext,
    AgentCoordinator,
    AgentMemory,
    AgentResult,
    ExecutionPlan,
    MemoryType,
    Task,
    TaskPriority,
    Tool,
    ToolCategory,
    ToolRegistry,
    tool,
)


# Define a custom agent
class DataProcessingAgent(AgentBase[dict, dict]):
    """Agent for processing data."""

    async def _execute(
        self,
        input_data: dict,
        context: AgentContext,
    ) -> AgentResult[dict]:
        context.add_thought("Analyzing input data structure")

        # Validate
        if "data" not in input_data:
            return AgentResult.failure_result(
                error="Missing 'data' field",
                error_code="INVALID_INPUT",
            )

        context.add_action("process", "data_processor", items=len(input_data["data"]))

        # Process
        processed = [item.upper() if isinstance(item, str) else item for item in input_data["data"]]

        context.add_observation(f"Processed {len(processed)} items")

        return AgentResult.success_result(
            output={"processed": processed, "count": len(processed)},
            iterations_used=context.iteration,
        )


# Define tools
@tool(name="analyze", category=ToolCategory.ANALYZE)
def analyze_data(data: list) -> dict:
    """Analyze data and return statistics."""
    return {
        "count": len(data),
        "has_strings": any(isinstance(x, str) for x in data),
        "has_numbers": any(isinstance(x, (int, float)) for x in data),
    }


@tool(name="transform", category=ToolCategory.TRANSFORM)
def transform_data(data: list, operation: str = "uppercase") -> list:
    """Transform data based on operation."""
    if operation == "uppercase":
        return [str(x).upper() for x in data]
    elif operation == "lowercase":
        return [str(x).lower() for x in data]
    return data


# ============================================================================
# 5. Health Checks
# ============================================================================

from ptpd_calibration.template.health import (
    HealthCheck,
    HealthStatus,
)

# Configure health checks
health = HealthCheck.configure(
    app_name=config.app_name,
    version=config.version,
    environment=config.environment.value,
)


def check_data_service() -> tuple[HealthStatus, str]:
    """Check data service availability."""
    # Simulate service check
    return HealthStatus.HEALTHY, "Data service operational"


def check_cache() -> tuple[HealthStatus, str]:
    """Check cache availability."""
    return HealthStatus.DEGRADED, "Cache latency high"


health.register("data_service", check_data_service, critical=True)
health.register("cache", check_cache, critical=False)

# ============================================================================
# 6. Component Factories
# ============================================================================

from ptpd_calibration.template.components import (
    ComponentFactory,
    ServiceDefinition,
    ServiceFactory,
)


# Example services
class DatabaseService:
    def __init__(self):
        self.connected = True
        logger.info("Database service initialized")


class CacheService:
    def __init__(self, database: DatabaseService):
        self.database = database
        logger.info("Cache service initialized with database")


# Setup service factory
service_factory = ServiceFactory()
service_factory.register(
    ServiceDefinition(name="database", service_class=DatabaseService)
)
service_factory.register(
    ServiceDefinition(name="cache", service_class=CacheService, dependencies=["database"])
)

# ============================================================================
# 7. Main Application
# ============================================================================


async def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("Starting Example Application")
    logger.info("=" * 60)

    # Use log context for request tracing
    with LogContext(request_id="main-001", operation="main"):
        # Test error handling
        logger.info("Testing error handling...")
        is_valid = validate_input({"name": "test"})
        logger.info(f"Validation result: {is_valid}")

        # Test retry logic
        logger.info("Testing retry logic...")
        try:
            data = await fetch_data("https://api.example.com")
            logger.info(f"Fetch result: {data}")
        except TemplateError as e:
            logger.error(f"Fetch failed: {e}")

        # Test agent system
        logger.info("Testing agent system...")

        # Create agent
        agent_config = AgentConfig(
            name="data_processor",
            timeout_seconds=30.0,
            max_iterations=5,
        )
        agent = DataProcessingAgent(agent_config)

        # Create coordinator
        coordinator = AgentCoordinator(max_parallel_agents=3)
        coordinator.register_agent("processor", agent)

        # Run a task
        task = Task(
            name="Process Sample Data",
            task_type="processor",
            input_data={"data": ["hello", "world", "test"]},
            priority=TaskPriority.HIGH,
        )

        result = await coordinator.run_task(task)
        logger.info(
            "Agent task completed",
            success=result.success,
            output=result.output,
        )

        # Test tools
        logger.info("Testing tools...")
        registry = ToolRegistry()
        registry.add(analyze_data)
        registry.add(transform_data)

        analysis = await registry.execute("analyze", data=[1, 2, "three", "four"])
        logger.info(f"Analysis result: {analysis.output}")

        transformation = await registry.execute(
            "transform", data=["hello", "world"], operation="uppercase"
        )
        logger.info(f"Transform result: {transformation.output}")

        # Test health checks
        logger.info("Testing health checks...")
        report = await health.check_all()
        logger.info(
            "Health check completed",
            status=report.status.value,
            components=[c.name for c in report.components],
        )

        # Test service factory
        logger.info("Testing service factory...")
        cache = service_factory.get("cache")
        logger.info(f"Cache service ready: {cache is not None}")

        # Test agent memory
        logger.info("Testing agent memory...")
        memory = AgentMemory(working_size=10)
        memory.add_observation("Processed 100 records")
        memory.add_fact("Average processing time: 50ms", importance=0.8)
        memory.add_thought("Consider batching for better performance")

        context_str = memory.get_context(max_entries=5)
        logger.info(f"Memory context:\n{context_str}")

        # Create and execute a plan
        logger.info("Testing execution plan...")
        plan = coordinator.create_plan("Sample Pipeline")
        step1 = plan.add_step(
            "Validate Input", "processor", input_data={"data": ["a", "b", "c"]}
        )
        step2 = plan.add_step(
            "Transform Data",
            "processor",
            input_data={"data": ["x", "y", "z"]},
            parallel_with=[step1.id],  # Can run in parallel with step1
        )

        results = await coordinator.execute_plan(plan)
        logger.info(f"Plan completed with {len(results)} results")

    logger.info("=" * 60)
    logger.info("Example Application Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
