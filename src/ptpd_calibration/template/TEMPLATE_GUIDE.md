# Agentic Coding Template System Guide

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

## Quick Start

```python
from ptpd_calibration.template import (
    # Configuration
    TemplateConfig, configure_template, EnvironmentType,
    # Logging
    setup_logging, get_logger, LogContext,
    # Errors
    ErrorBoundary, error_handler, retry_on_error,
    # Health
    HealthCheck, health_check_endpoint,
)

# 1. Configure the application
config = configure_template(
    app_name="MyApp",
    version="1.0.0",
    environment=EnvironmentType.PRODUCTION,
)

# 2. Setup logging
setup_logging(
    level=config.logging.level,
    json_format=config.logging.json_format,
)

# 3. Get a logger
logger = get_logger(__name__)
logger.info("Application starting", version=config.version)
```

---

## SECTION 1: CONFIGURATION SYSTEM

### 1.1 Environment-Based Configuration

The template supports multiple deployment environments with automatic configuration:

```python
from ptpd_calibration.template.config import (
    TemplateConfig,
    configure_template,
    get_template_config,
    EnvironmentType,
)

# Development (default)
config = configure_template(environment=EnvironmentType.DEVELOPMENT)
# - debug=True, console logging, file logging enabled

# Testing
config = configure_template(environment=EnvironmentType.TESTING)
# - debug=True, DEBUG log level, shorter timeouts

# Production
config = configure_template(environment=EnvironmentType.PRODUCTION)
# - debug=False, WARNING level, JSON logging, strict security

# Huggingface Spaces
config = configure_template(environment=EnvironmentType.HUGGINGFACE)
# - /tmp paths, reduced memory limits, rate limiting enabled
```

### 1.2 Configuration Options

```python
from ptpd_calibration.template.config import (
    LoggingConfig,
    TimeoutConfig,
    ResourceLimits,
    SecurityConfig,
    FeatureFlags,
)

config = configure_template(
    # Core settings
    app_name="MyApp",
    version="1.0.0",
    environment=EnvironmentType.PRODUCTION,
    debug=False,

    # Nested configurations
    logging=LoggingConfig(
        level="INFO",
        json_format=True,
        file_enabled=True,
        file_path=Path("/var/log/app.log"),
    ),
    timeouts=TimeoutConfig(
        default_seconds=30.0,
        api_request_seconds=120.0,
        agent_task_seconds=300.0,
    ),
    resources=ResourceLimits(
        max_memory_mb=4096,
        max_concurrent_requests=10,
    ),
    security=SecurityConfig(
        api_key_required=True,
        rate_limit_enabled=True,
    ),
    features=FeatureFlags(
        enable_deep_learning=True,
        enable_llm_features=True,
    ),
)
```

### 1.3 Environment Variables

All settings can be overridden via environment variables with `TEMPLATE_` prefix:

```bash
export TEMPLATE_APP_NAME="MyApp"
export TEMPLATE_ENVIRONMENT="production"
export TEMPLATE_DEBUG="false"
export TEMPLATE_LOGGING__LEVEL="WARNING"
export TEMPLATE_LOGGING__JSON_FORMAT="true"
export TEMPLATE_RESOURCES__MAX_MEMORY_MB="2048"
```

---

## SECTION 2: LOGGING INFRASTRUCTURE

### 2.1 Structured Logging

```python
from ptpd_calibration.template.logging_config import (
    setup_logging,
    get_logger,
    LogContext,
    logged,
)

# Setup logging
setup_logging(
    level="INFO",
    json_format=True,  # For production
    file_enabled=True,
    file_path=Path("/var/log/app.log"),
    max_bytes=10_485_760,  # 10MB
    backup_count=5,
)

# Get a logger
logger = get_logger(__name__)

# Log with structured data
logger.info("Processing started", item_count=42, batch_id="abc123")

# Log with timing
with logger.timed("database_query") as ctx:
    result = db.query(...)
    ctx["row_count"] = len(result)
# Logs: "Completed: database_query (150.32ms)"
```

### 2.2 Request Context

```python
from ptpd_calibration.template.logging_config import LogContext

# All logs within this context include the request/user info
with LogContext(request_id="req-123", user_id="user-456"):
    logger.info("Processing request")  # Includes context
    do_work()
    logger.info("Request complete")
```

### 2.3 Function Logging Decorator

```python
from ptpd_calibration.template.logging_config import logged

@logged(level=logging.DEBUG, log_result=True, include_timing=True)
def process_data(data: list) -> dict:
    """Automatically logs function entry, exit, and timing."""
    return {"processed": len(data)}
```

---

## SECTION 3: ERROR HANDLING

### 3.1 Error Hierarchy

```python
from ptpd_calibration.template.errors import (
    TemplateError,        # Base error
    ValidationError,      # Input validation (400)
    ConfigurationError,   # Config issues (500)
    TimeoutError,         # Operation timeout (504)
    ResourceError,        # Resource limits (503)
    NetworkError,         # Network issues (502)
    NotFoundError,        # Not found (404)
    AuthenticationError,  # Auth failure (401)
)

# Raise with context
raise ValidationError(
    "Invalid email format",
    field="email",
    value=email_input,
    user_message="Please enter a valid email address",
)

# Errors are automatically serializable
error.to_dict()  # For API responses
```

### 3.2 Error Boundaries

```python
from ptpd_calibration.template.errors import ErrorBoundary

# Create boundary for a component
boundary = ErrorBoundary(
    component="image_processor",
    default_return=None,  # Return on error
    reraise=False,        # Don't propagate
    log_errors=True,
)

# Use as context manager
with boundary.protect(operation="resize_image"):
    result = resize(image, size)

# Use as decorator
@boundary.wrap
def process_image(image):
    ...
```

### 3.3 Automatic Retry

```python
from ptpd_calibration.template.errors import retry_on_error

@retry_on_error(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True,
)
async def fetch_external_data():
    """Automatically retries on TemplateError or ConnectionError."""
    ...
```

---

## SECTION 4: AGENT SYSTEM

### 4.1 Creating Custom Agents

```python
from ptpd_calibration.template.agents import (
    AgentBase,
    AgentConfig,
    AgentContext,
    AgentResult,
)

class DataAnalysisAgent(AgentBase[dict, dict]):
    """Agent for analyzing data."""

    async def _execute(
        self,
        input_data: dict,
        context: AgentContext,
    ) -> AgentResult[dict]:
        # Add reasoning trace
        context.add_thought("Analyzing input data structure")

        # Perform analysis
        result = await self.analyze(input_data)

        context.add_observation(f"Found {len(result)} patterns")

        return AgentResult.success_result(
            output=result,
            iterations_used=context.iteration,
        )

# Create and use agent
config = AgentConfig(
    name="data_analyzer",
    timeout_seconds=60.0,
    max_iterations=10,
)

agent = DataAnalysisAgent(config)
result = await agent.run({"data": [1, 2, 3]})
```

### 4.2 Tool Registration

```python
from ptpd_calibration.template.agents import (
    Tool,
    ToolRegistry,
    ToolCategory,
    tool,
)

# Using decorator
@tool(name="search", category=ToolCategory.QUERY)
def search_documents(query: str, limit: int = 10) -> list[dict]:
    """Search documents by query."""
    return db.search(query, limit=limit)

# Register in registry
registry = ToolRegistry()
registry.add(search_documents)

# Use decorator registration
@registry.register(category=ToolCategory.ANALYZE)
def analyze_sentiment(text: str) -> float:
    """Analyze text sentiment."""
    return sentiment_model.predict(text)

# Execute tools
result = await registry.execute("search", query="hello", limit=5)
```

### 4.3 Agent Coordination

```python
from ptpd_calibration.template.agents import (
    AgentCoordinator,
    Task,
    TaskPriority,
    ExecutionPlan,
)

# Create coordinator
coordinator = AgentCoordinator(max_parallel_agents=3)

# Register agents
coordinator.register_agent("analyzer", DataAnalysisAgent(config))
coordinator.register_agent("processor", DataProcessorAgent(config))

# Run single task
task = Task(
    name="Analyze Dataset",
    task_type="analyzer",
    input_data={"dataset": "sales.csv"},
    priority=TaskPriority.HIGH,
)
result = await coordinator.run_task(task)

# Run parallel tasks
tasks = [Task(task_type="processor", input_data=d) for d in data_items]
results = await coordinator.run_parallel(tasks)

# Execute a plan with dependencies
plan = coordinator.create_plan("ETL Pipeline")
step1 = plan.add_step("Extract", "extractor", input_data=source)
step2 = plan.add_step("Transform", "transformer", depends_on=[step1.id])
step3 = plan.add_step("Load", "loader", depends_on=[step2.id])

results = await coordinator.execute_plan(plan)
```

### 4.4 Agent Memory

```python
from ptpd_calibration.template.agents import AgentMemory, MemoryType

memory = AgentMemory(working_size=20)

# Add memories
memory.add_observation("Dataset has 1000 rows")
memory.add_fact("Primary key is 'user_id'", importance=0.9)
memory.add_thought("Consider normalizing the data")

# Search memories
relevant = memory.search("user_id", limit=5)

# Get context for prompt
context_str = memory.get_context(max_entries=10, max_chars=2000)
```

---

## SECTION 5: COMPONENT FACTORIES

### 5.1 Service Factory

```python
from ptpd_calibration.template.components import (
    ServiceFactory,
    ServiceDefinition,
)

factory = ServiceFactory()

# Register services with dependencies
factory.register(ServiceDefinition(
    name="database",
    service_class=DatabaseService,
    config_class=DatabaseConfig,
    singleton=True,
))

factory.register(ServiceDefinition(
    name="cache",
    service_class=CacheService,
    dependencies=["database"],  # Injected automatically
))

# Get service (creates with dependencies)
cache = factory.get("cache")

# Shutdown all services
await factory.shutdown_all()
```

### 5.2 UI Component Builder

```python
from ptpd_calibration.template.components import (
    UIComponentBuilder,
    GradioAppBuilder,
    TabBuilder,
    TabConfig,
    AppConfig,
)

# Create a custom tab
class SettingsTab(TabBuilder):
    def build_components(self) -> None:
        self.api_key = self.builder.text_input(
            "API Key",
            placeholder="Enter your API key",
        )
        self.save_btn = self.builder.button("Save", variant="primary")

    def setup_events(self) -> None:
        self.save_btn.click(
            self.wrap_handler(self.on_save),
            inputs=[self.api_key],
            outputs=[...],
        )

    def on_save(self, api_key: str) -> str:
        # Handler with automatic error handling
        save_config(api_key)
        return "Saved!"

# Build complete app
builder = GradioAppBuilder(AppConfig(
    title="My Application",
    server_port=7860,
))

builder.add_tab(SettingsTab(TabConfig(id="settings", label="Settings")))
builder.add_tab(ProcessingTab(TabConfig(id="process", label="Process")))

app = builder.build()
app.launch()
```

---

## SECTION 6: MIDDLEWARE

### 6.1 Middleware Chain

```python
from ptpd_calibration.template.components import (
    MiddlewareChain,
    LoggingMiddleware,
    TimeoutMiddleware,
    RateLimitMiddleware,
)

chain = MiddlewareChain()
chain.add(LoggingMiddleware(slow_request_threshold_ms=1000))
chain.add(TimeoutMiddleware(timeout_seconds=30))
chain.add(RateLimitMiddleware(max_requests=100, window_seconds=60))

# Process request through chain
response = await chain.process(request, handler)

# Or wrap a handler
@chain.wrap
async def my_handler(request):
    ...
```

### 6.2 FastAPI Integration

```python
from fastapi import FastAPI
from ptpd_calibration.template.errors import create_fastapi_exception_handlers
from ptpd_calibration.template.health import health_check_endpoint

app = FastAPI()

# Add exception handlers
for exc_type, handler in create_fastapi_exception_handlers().items():
    app.add_exception_handler(exc_type, handler)

# Add health endpoints
app.include_router(health_check_endpoint())
```

---

## SECTION 7: HEALTH CHECKS

### 7.1 Health Check System

```python
from ptpd_calibration.template.health import (
    HealthCheck,
    HealthStatus,
    health_check_endpoint,
)

# Configure health checks
health = HealthCheck.configure(
    app_name="MyApp",
    version="1.0.0",
    environment="production",
)

# Register custom checks
health.register(
    "database",
    check_database_connection,
    critical=True,  # App is unhealthy if this fails
)

health.register(
    "external_api",
    check_external_api,
    critical=False,  # App is degraded if this fails
    timeout_seconds=5.0,
)

# Get health report
report = await health.check_all()
print(f"Status: {report.status}")  # HEALTHY, DEGRADED, or UNHEALTHY
```

---

## SECTION 8: TESTING

### 8.1 Test Fixtures

```python
import pytest
from ptpd_calibration.template.config import reset_config

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    from ptpd_calibration.template.config import configure_template, EnvironmentType

    config = configure_template(
        environment=EnvironmentType.TESTING,
        data_dir=temp_dir / "data",
    )
    yield config
    reset_config()

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    from ptpd_calibration.template.agents import AgentConfig

    config = AgentConfig(
        name="test_agent",
        timeout_seconds=5.0,
    )
    return MyTestAgent(config)
```

### 8.2 Testing Agents

```python
@pytest.mark.asyncio
async def test_agent_execution(mock_agent, agent_context):
    result = await mock_agent.run({"input": "test"}, agent_context)

    assert result.success is True
    assert result.output is not None
    assert "test" in str(result.output)

@pytest.mark.asyncio
async def test_agent_timeout(slow_agent):
    result = await slow_agent.run({"delay": 10})

    assert result.success is False
    assert "timed out" in result.error.lower()
```

---

## SECTION 9: HUGGINGFACE DEPLOYMENT

### 9.1 Configuration for Spaces

```python
# Auto-detected when SPACE_ID environment variable is set
config = configure_template(environment=EnvironmentType.HUGGINGFACE)

# Or explicit configuration
config = configure_template(
    environment=EnvironmentType.HUGGINGFACE,
    # Paths adjusted for ephemeral storage
    data_dir=Path("/tmp/app_data"),
    # Reduced resource limits
    resources=ResourceLimits(
        max_memory_mb=2048,
        max_concurrent_requests=5,
    ),
)
```

### 9.2 Graceful Degradation

```python
from ptpd_calibration.template.config import get_template_config

config = get_template_config()

# Check feature availability
if config.is_feature_enabled("deep_learning"):
    from ptpd_calibration.deep_learning import NeuralModel
    model = NeuralModel()
else:
    # Use fallback
    model = SimpleModel()
```

---

## SECTION 10: BEST PRACTICES

### 10.1 Error Handling

```python
# DO: Use specific error types
raise ValidationError("Invalid email", field="email")

# DON'T: Use bare exceptions
raise Exception("Something went wrong")

# DO: Provide user-friendly messages
raise TemplateError(
    "Database connection failed",  # Technical message
    user_message="Service temporarily unavailable",  # User message
)
```

### 10.2 Logging

```python
# DO: Use structured logging
logger.info("User registered", user_id=user.id, email=user.email)

# DON'T: Use string formatting
logger.info(f"User {user.id} registered with email {user.email}")

# DO: Use timing for performance-critical operations
with logger.timed("api_call"):
    response = await api.call()
```

### 10.3 Testing

```python
# DO: Test with proper fixtures
def test_feature(test_config, mock_agent):
    ...

# DO: Test error cases
def test_validation_error():
    with pytest.raises(ValidationError) as exc:
        validate(invalid_data)
    assert exc.value.details["field"] == "email"

# DO: Test timeouts
@pytest.mark.asyncio
async def test_timeout():
    result = await slow_operation()
    assert result.success is False
```

---

## Template Instantiation Checklist

When starting a new project:

- [ ] Configure environment (`DEVELOPMENT`, `TESTING`, `PRODUCTION`, `HUGGINGFACE`)
- [ ] Set up logging with appropriate level and format
- [ ] Define error boundaries for each major component
- [ ] Register health checks for all dependencies
- [ ] Create agents for complex tasks
- [ ] Register tools in the tool registry
- [ ] Set up middleware chain for API
- [ ] Write comprehensive tests
- [ ] Document configuration options
- [ ] Test Huggingface deployment with `HUGGINGFACE` environment
