"""
Tests for agent system.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ptpd_calibration.template.agents.base import (
    AgentBase,
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
)
from ptpd_calibration.template.agents.tools import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    tool,
)
from ptpd_calibration.template.agents.memory import (
    AgentMemory,
    MemoryEntry,
    MemoryType,
)
from ptpd_calibration.template.agents.coordinator import (
    AgentCoordinator,
    ExecutionPlan,
    Task,
    TaskPriority,
    TaskRouter,
    TaskStatus,
)


# ============================================================================
# Agent Base Tests
# ============================================================================


class TestAgentContext:
    """Tests for AgentContext."""

    def test_context_creation(self, agent_context: AgentContext) -> None:
        """Test context creation with defaults."""
        assert agent_context.task_id == "test-task-123"
        assert agent_context.task_type == "test"
        assert agent_context.iteration == 0

    def test_add_thought(self, agent_context: AgentContext) -> None:
        """Test adding thoughts to reasoning trace."""
        agent_context.add_thought("This is my first thought")
        assert len(agent_context.reasoning_trace) == 1
        assert agent_context.reasoning_trace[0]["type"] == "thought"
        assert agent_context.reasoning_trace[0]["content"] == "This is my first thought"

    def test_add_action(self, agent_context: AgentContext) -> None:
        """Test adding actions to reasoning trace."""
        agent_context.add_action("search", "search_tool", query="test")
        assert len(agent_context.reasoning_trace) == 1
        assert agent_context.reasoning_trace[0]["type"] == "action"
        assert agent_context.reasoning_trace[0]["tool"] == "search_tool"

    def test_add_observation(self, agent_context: AgentContext) -> None:
        """Test adding observations to reasoning trace."""
        agent_context.add_observation("Found 5 results")
        assert len(agent_context.reasoning_trace) == 1
        assert agent_context.reasoning_trace[0]["type"] == "observation"


class TestAgentResult:
    """Tests for AgentResult."""

    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = AgentResult.success_result(
            output={"data": "test"},
            iterations_used=3,
        )
        assert result.success is True
        assert result.state == AgentState.COMPLETED
        assert result.output == {"data": "test"}

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        result = AgentResult.failure_result(
            error="Something went wrong",
            error_code="TEST_ERROR",
        )
        assert result.success is False
        assert result.state == AgentState.FAILED
        assert result.error == "Something went wrong"


class TestAgentBase:
    """Tests for AgentBase."""

    @pytest.mark.asyncio
    async def test_agent_run_success(
        self,
        mock_agent: Any,
        agent_context: AgentContext,
    ) -> None:
        """Test successful agent execution."""
        result = await mock_agent.run({"input": "test"}, agent_context)
        assert result.success is True
        assert result.output["processed"] is True

    @pytest.mark.asyncio
    async def test_agent_run_failure(self, failing_agent: Any) -> None:
        """Test agent failure handling."""
        result = await failing_agent.run({})
        assert result.success is False
        assert result.error_code == "TEST_ERROR"

    @pytest.mark.asyncio
    async def test_agent_timeout(self, slow_agent: Any) -> None:
        """Test agent timeout handling."""
        result = await slow_agent.run({"delay": 10})
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_agent_state_transitions(self, mock_agent: Any) -> None:
        """Test agent state transitions during execution."""
        assert mock_agent.state == AgentState.IDLE
        await mock_agent.run({})
        # After completion, should be in COMPLETED state
        assert mock_agent.state == AgentState.COMPLETED


# ============================================================================
# Tool Tests
# ============================================================================


class TestTool:
    """Tests for Tool."""

    @pytest.mark.asyncio
    async def test_tool_execute_sync(self, simple_tool: Tool) -> None:
        """Test executing a sync tool."""
        result = await simple_tool.execute(a=5, b=3)
        assert result.success is True
        assert result.output == 8

    @pytest.mark.asyncio
    async def test_tool_execute_async(self, async_tool: Tool) -> None:
        """Test executing an async tool."""
        result = await async_tool.execute(url="https://example.com")
        assert result.success is True
        assert result.output["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_tool_validation_error(self, simple_tool: Tool) -> None:
        """Test tool parameter validation."""
        from ptpd_calibration.template.errors import ValidationError

        with pytest.raises(ValidationError):
            await simple_tool.execute(a=5)  # Missing required 'b'

    def test_tool_to_json_schema(self, simple_tool: Tool) -> None:
        """Test converting tool to JSON schema."""
        schema = simple_tool.to_json_schema()
        assert schema["name"] == "add_numbers"
        assert "parameters" in schema
        assert "a" in schema["parameters"]["properties"]

    def test_tool_get_stats(self, simple_tool: Tool) -> None:
        """Test tool usage statistics."""
        simple_tool.execute_sync(a=1, b=2)
        simple_tool.execute_sync(a=3, b=4)
        stats = simple_tool.get_stats()
        assert stats["call_count"] == 2
        assert stats["error_count"] == 0


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_basic(self) -> None:
        """Test basic tool decorator usage."""

        @tool(name="multiply", category=ToolCategory.UTILITY)
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        assert isinstance(multiply, Tool)
        assert multiply.name == "multiply"
        assert multiply.schema.category == ToolCategory.UTILITY

    def test_tool_decorator_infers_name(self) -> None:
        """Test tool decorator infers name from function."""

        @tool()
        def my_function(x: str) -> str:
            return x

        assert my_function.name == "my_function"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_registry_add_tool(self, tool_registry: ToolRegistry) -> None:
        """Test adding tools to registry."""
        assert tool_registry.has("add_numbers")
        assert tool_registry.has("fetch_data")

    def test_registry_get_tool(self, tool_registry: ToolRegistry) -> None:
        """Test getting tool from registry."""
        tool = tool_registry.get("add_numbers")
        assert tool is not None
        assert tool.name == "add_numbers"

    def test_registry_list_tools(self, tool_registry: ToolRegistry) -> None:
        """Test listing tools by category."""
        utility_tools = tool_registry.list_tools(category=ToolCategory.UTILITY)
        assert len(utility_tools) >= 1

    @pytest.mark.asyncio
    async def test_registry_execute(self, tool_registry: ToolRegistry) -> None:
        """Test executing tool through registry."""
        result = await tool_registry.execute("add_numbers", a=10, b=20)
        assert result.success is True
        assert result.output == 30

    def test_registry_register_decorator(self) -> None:
        """Test registry register decorator."""
        registry = ToolRegistry()

        @registry.register(category=ToolCategory.ANALYZE)
        def analyze(data: list) -> dict:
            return {"count": len(data)}

        assert registry.has("analyze")


# ============================================================================
# Memory Tests
# ============================================================================


class TestAgentMemory:
    """Tests for AgentMemory."""

    def test_add_memory(self, agent_memory: AgentMemory) -> None:
        """Test adding memory entries."""
        entry = agent_memory.add("Test observation", MemoryType.OBSERVATION)
        assert entry.type == MemoryType.OBSERVATION
        assert entry.content == "Test observation"

    def test_memory_shortcuts(self, agent_memory: AgentMemory) -> None:
        """Test memory shortcut methods."""
        agent_memory.add_observation("Saw something")
        agent_memory.add_action("Did something")
        agent_memory.add_thought("Thinking...")
        agent_memory.add_fact("Earth is round", importance=0.9)

        assert len(agent_memory) == 4

    def test_get_recent(self, populated_memory: AgentMemory) -> None:
        """Test getting recent memories."""
        recent = populated_memory.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_by_type(self, populated_memory: AgentMemory) -> None:
        """Test getting memories by type."""
        facts = populated_memory.get_by_type(MemoryType.FACT)
        assert len(facts) >= 1
        assert all(m.type == MemoryType.FACT for m in facts)

    def test_get_important(self, populated_memory: AgentMemory) -> None:
        """Test getting important memories."""
        important = populated_memory.get_important(threshold=0.7)
        assert all(m.importance >= 0.7 for m in important)

    def test_search(self, populated_memory: AgentMemory) -> None:
        """Test searching memories."""
        results = populated_memory.search("analysis")
        assert len(results) > 0

    def test_get_context(self, populated_memory: AgentMemory) -> None:
        """Test getting formatted context."""
        context = populated_memory.get_context(max_entries=3)
        assert isinstance(context, str)
        assert len(context) > 0

    def test_memory_stats(self, populated_memory: AgentMemory) -> None:
        """Test memory statistics."""
        stats = populated_memory.get_stats()
        assert stats["working_count"] == 5
        assert "by_type" in stats


# ============================================================================
# Coordinator Tests
# ============================================================================


class TestTaskRouter:
    """Tests for TaskRouter."""

    def test_add_route(self) -> None:
        """Test adding a direct route."""
        router = TaskRouter()
        router.add_route("analysis", "analyzer_agent")

        task = Task(task_type="analysis")
        agent = router.route(task, ["analyzer_agent", "other_agent"])
        assert agent == "analyzer_agent"

    def test_load_balancing(self) -> None:
        """Test load balancing among agents."""
        router = TaskRouter()

        task = Task(task_type="generic")
        agents = ["agent_a", "agent_b", "agent_c"]

        # All agents should have equal load initially
        agent = router.route(task, agents)
        assert agent in agents


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_add_step(self, sample_plan: ExecutionPlan) -> None:
        """Test adding steps to plan."""
        assert len(sample_plan.steps) == 3

    def test_get_ready_steps(self, sample_plan: ExecutionPlan) -> None:
        """Test getting ready steps."""
        ready = sample_plan.get_ready_steps()
        # Only steps without dependencies should be ready
        assert len(ready) == 2  # Step 1 and Step 2

    def test_is_complete(self, sample_plan: ExecutionPlan) -> None:
        """Test plan completion check."""
        assert sample_plan.is_complete() is False

        # Mark all as completed
        from ptpd_calibration.template.agents.coordinator import PlanStepStatus
        for step in sample_plan.steps:
            step.status = PlanStepStatus.COMPLETED

        assert sample_plan.is_complete() is True


class TestAgentCoordinator:
    """Tests for AgentCoordinator."""

    def test_register_agent(self, coordinator: AgentCoordinator, mock_agent: Any) -> None:
        """Test registering an agent."""
        coordinator.register_agent("mock", mock_agent)
        assert "mock" in coordinator.list_agents()

    def test_get_agent(
        self,
        populated_coordinator: AgentCoordinator,
    ) -> None:
        """Test getting a registered agent."""
        agent = populated_coordinator.get_agent("mock")
        assert agent is not None

    @pytest.mark.asyncio
    async def test_run_task(
        self,
        populated_coordinator: AgentCoordinator,
        sample_task: Task,
    ) -> None:
        """Test running a single task."""
        result = await populated_coordinator.run_task(sample_task)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_parallel(
        self,
        populated_coordinator: AgentCoordinator,
    ) -> None:
        """Test running multiple tasks in parallel."""
        tasks = [
            Task(name=f"Task {i}", task_type="mock", input_data={"i": i})
            for i in range(3)
        ]

        results = await populated_coordinator.run_parallel(tasks)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_plan(
        self,
        populated_coordinator: AgentCoordinator,
        sample_plan: ExecutionPlan,
    ) -> None:
        """Test executing a plan."""
        results = await populated_coordinator.execute_plan(sample_plan)
        assert len(results) == 3
        assert sample_plan.status == TaskStatus.COMPLETED

    def test_get_stats(self, populated_coordinator: AgentCoordinator) -> None:
        """Test getting coordinator statistics."""
        stats = populated_coordinator.get_stats()
        assert "agents" in stats
        assert "tasks" in stats
        assert "memory" in stats
