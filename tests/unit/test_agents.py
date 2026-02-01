"""
Comprehensive unit tests for the agentic system.

Tests for CalibrationAgent, ToolRegistry, AgentMemory, and Planning system.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from ptpd_calibration.agents.agent import (
    AgentConfig,
    CalibrationAgent,
    ReasoningStep,
    create_agent,
)
from ptpd_calibration.agents.memory import AgentMemory, MemoryItem
from ptpd_calibration.agents.planning import Plan, Planner, PlanStatus, PlanStep
from ptpd_calibration.agents.tools import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_calibration_tools,
)
from ptpd_calibration.config import AgentSettings, LLMSettings

# =============================================================================
# ToolResult Tests
# =============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_successful_result_to_string(self):
        """Test string conversion for successful result."""
        result = ToolResult(success=True, data={"key": "value"})
        output = result.to_string()
        assert "key" in output
        assert "value" in output

    def test_error_result_to_string(self):
        """Test string conversion for error result."""
        result = ToolResult(success=False, error="Something went wrong")
        output = result.to_string()
        assert "Error:" in output
        assert "Something went wrong" in output

    def test_list_data_to_string(self):
        """Test string conversion for list data."""
        result = ToolResult(success=True, data=[1, 2, 3])
        output = result.to_string()
        assert "1" in output
        assert "2" in output
        assert "3" in output

    def test_string_data_to_string(self):
        """Test string conversion for string data."""
        result = ToolResult(success=True, data="plain text")
        assert result.to_string() == "plain text"

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        metadata = {"execution_time": 0.5, "source": "test"}
        result = ToolResult(success=True, data="test", metadata=metadata)
        assert result.metadata == metadata


# =============================================================================
# ToolParameter Tests
# =============================================================================


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_required_parameter(self):
        """Test required parameter creation."""
        param = ToolParameter(
            name="input",
            type="string",
            description="Input value",
            required=True,
        )
        assert param.required is True
        assert param.default is None

    def test_optional_parameter_with_default(self):
        """Test optional parameter with default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            enum=["json", "csv", "xml"],
        )
        assert param.enum == ["json", "csv", "xml"]


# =============================================================================
# Tool Tests
# =============================================================================


class TestTool:
    """Tests for Tool class."""

    @pytest.fixture
    def simple_tool(self):
        """Create a simple test tool."""

        def handler(value: str) -> ToolResult:
            return ToolResult(success=True, data=f"processed: {value}")

        return Tool(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="value",
                    type="string",
                    description="Input value",
                )
            ],
            handler=handler,
            category=ToolCategory.ANALYSIS,
        )

    def test_tool_creation(self, simple_tool):
        """Test tool creation with all attributes."""
        assert simple_tool.name == "test_tool"
        assert simple_tool.description == "A test tool"
        assert len(simple_tool.parameters) == 1
        assert simple_tool.category == ToolCategory.ANALYSIS

    @pytest.mark.asyncio
    async def test_tool_execution_success(self, simple_tool):
        """Test successful tool execution."""
        result = await simple_tool.execute(value="hello")
        assert result.success is True
        assert "processed: hello" in result.data

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution with error."""

        def failing_handler(**_kwargs):
            raise ValueError("Test error")

        tool = Tool(
            name="failing_tool",
            description="A tool that fails",
            parameters=[],
            handler=failing_handler,
        )
        result = await tool.execute()
        assert result.success is False
        assert "Test error" in result.error

    def test_to_anthropic_format(self, simple_tool):
        """Test conversion to Anthropic tool format."""
        formatted = simple_tool.to_anthropic_format()
        assert formatted["name"] == "test_tool"
        assert formatted["description"] == "A test tool"
        assert "input_schema" in formatted
        assert formatted["input_schema"]["type"] == "object"
        assert "value" in formatted["input_schema"]["properties"]

    def test_to_anthropic_format_with_optional_params(self):
        """Test Anthropic format with optional parameters."""

        def handler(**_kwargs):
            return ToolResult(success=True, data="ok")

        tool = Tool(
            name="multi_param_tool",
            description="Tool with multiple params",
            parameters=[
                ToolParameter(name="required_param", type="string", description="Required", required=True),
                ToolParameter(name="optional_param", type="integer", description="Optional", required=False, default=5),
            ],
            handler=handler,
        )
        formatted = tool.to_anthropic_format()
        assert "required_param" in formatted["input_schema"]["required"]
        assert "optional_param" not in formatted["input_schema"]["required"]


# =============================================================================
# ToolRegistry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create an empty registry."""
        return ToolRegistry()

    @pytest.fixture
    def analysis_tool(self):
        """Create an analysis category tool."""
        return Tool(
            name="analyze",
            description="Analyze data",
            parameters=[],
            handler=lambda: ToolResult(success=True, data="analyzed"),
            category=ToolCategory.ANALYSIS,
        )

    @pytest.fixture
    def database_tool(self):
        """Create a database category tool."""
        return Tool(
            name="query_db",
            description="Query database",
            parameters=[],
            handler=lambda: ToolResult(success=True, data="queried"),
            category=ToolCategory.DATABASE,
        )

    def test_register_tool(self, registry, analysis_tool):
        """Test registering a tool."""
        registry.register(analysis_tool)
        assert registry.get("analyze") is not None

    def test_get_nonexistent_tool(self, registry):
        """Test getting a tool that doesn't exist."""
        result = registry.get("nonexistent")
        assert result is None

    def test_list_all_tools(self, registry, analysis_tool, database_tool):
        """Test listing all tools."""
        registry.register(analysis_tool)
        registry.register(database_tool)
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_tools_by_category(self, registry, analysis_tool, database_tool):
        """Test listing tools filtered by category."""
        registry.register(analysis_tool)
        registry.register(database_tool)

        analysis_tools = registry.list_tools(category=ToolCategory.ANALYSIS)
        assert len(analysis_tools) == 1
        assert analysis_tools[0].name == "analyze"

        db_tools = registry.list_tools(category=ToolCategory.DATABASE)
        assert len(db_tools) == 1
        assert db_tools[0].name == "query_db"

    def test_to_anthropic_format(self, registry, analysis_tool):
        """Test converting entire registry to Anthropic format."""
        registry.register(analysis_tool)
        formatted = registry.to_anthropic_format()
        assert len(formatted) == 1
        assert formatted[0]["name"] == "analyze"


class TestCreateCalibrationTools:
    """Tests for create_calibration_tools factory function."""

    def test_creates_basic_tools_without_database(self):
        """Test tool creation without database."""
        registry = create_calibration_tools(database=None, predictor=None)
        tools = registry.list_tools()
        # Should have basic tools: analyze_densities, compare_calibrations, generate_curve, suggest_parameters, create_test_plan
        tool_names = [t.name for t in tools]
        assert "analyze_densities" in tool_names
        assert "generate_curve" in tool_names
        assert "suggest_parameters" in tool_names
        assert "create_test_plan" in tool_names

    def test_creates_database_tools_with_database(self, populated_database):
        """Test tool creation with database."""
        registry = create_calibration_tools(database=populated_database, predictor=None)
        tool_names = [t.name for t in registry.list_tools()]
        assert "search_calibrations" in tool_names
        assert "get_calibration" in tool_names
        assert "save_calibration" in tool_names

    def test_creates_prediction_tools_with_predictor(self):
        """Test tool creation with predictor."""
        mock_predictor = MagicMock()
        registry = create_calibration_tools(database=None, predictor=mock_predictor)
        tool_names = [t.name for t in registry.list_tools()]
        assert "predict_response" in tool_names


# =============================================================================
# MemoryItem Tests
# =============================================================================


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        item = MemoryItem(key="test", content="test content")
        assert item.category == "general"
        assert item.importance == 0.5
        assert item.access_count == 0
        assert isinstance(item.id, UUID)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        item = MemoryItem(
            key="test_key",
            content="test content",
            category="fact",
            importance=0.8,
            metadata={"source": "test"},
        )
        data = item.to_dict()
        assert data["key"] == "test_key"
        assert data["content"] == "test content"
        assert data["category"] == "fact"
        assert data["importance"] == 0.8
        assert data["metadata"]["source"] == "test"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": str(uuid4()),
            "key": "restored_key",
            "content": "restored content",
            "category": "insight",
            "importance": 0.9,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 5,
            "metadata": {"restored": True},
        }
        item = MemoryItem.from_dict(data)
        assert item.key == "restored_key"
        assert item.content == "restored content"
        assert item.category == "insight"
        assert item.importance == 0.9
        assert item.access_count == 5

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = MemoryItem(
            key="roundtrip",
            content="roundtrip content",
            category="preference",
            importance=0.7,
            metadata={"round": "trip"},
        )
        data = original.to_dict()
        restored = MemoryItem.from_dict(data)
        assert restored.key == original.key
        assert restored.content == original.content
        assert restored.category == original.category
        assert restored.importance == original.importance


# =============================================================================
# AgentMemory Tests
# =============================================================================


class TestAgentMemory:
    """Tests for AgentMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a fresh AgentMemory instance."""
        return AgentMemory(max_items=100, working_memory_size=5)

    def test_remember_and_get(self, memory):
        """Test storing and retrieving a memory."""
        item = memory.remember(
            key="test_memory",
            content="This is a test",
            category="fact",
            importance=0.8,
        )
        assert item.key == "test_memory"

        retrieved = memory.get("test_memory")
        assert retrieved is not None
        assert retrieved.content == "This is a test"

    def test_remember_updates_existing(self, memory):
        """Test that remember updates existing items."""
        memory.remember(key="updating", content="original")
        memory.remember(key="updating", content="updated")

        retrieved = memory.get("updating")
        assert retrieved.content == "updated"

    def test_recall_by_query(self, memory):
        """Test recall with query matching."""
        memory.remember("paper_pref", "I prefer Arches Platine paper", category="preference")
        memory.remember("chem_fact", "Na2 increases contrast", category="fact")
        memory.remember("paper_fact", "Paper should be sized", category="fact")

        results = memory.recall("paper", limit=10)
        assert len(results) >= 2
        # Results should be scored by relevance
        keys = [r.key for r in results]
        assert "paper_pref" in keys
        assert "paper_fact" in keys

    def test_recall_by_category(self, memory):
        """Test recall filtered by category."""
        memory.remember("fact1", "Fact one", category="fact")
        memory.remember("pref1", "Preference one", category="preference")

        facts = memory.recall("one", category="fact")
        assert all(r.category == "fact" for r in facts)

    def test_forget(self, memory):
        """Test forgetting a memory."""
        memory.remember("to_forget", "will be forgotten")
        assert memory.get("to_forget") is not None

        result = memory.forget("to_forget")
        assert result is True
        assert memory.get("to_forget") is None

    def test_forget_nonexistent(self, memory):
        """Test forgetting a memory that doesn't exist."""
        result = memory.forget("nonexistent")
        assert result is False

    def test_working_memory(self, memory):
        """Test working memory operations."""
        memory.add_to_working_memory("item1")
        memory.add_to_working_memory("item2")
        memory.add_to_working_memory("item3")

        working = memory.get_working_memory()
        assert len(working) == 3
        assert "item1" in working
        assert "item2" in working
        assert "item3" in working

    def test_working_memory_size_limit(self, memory):
        """Test that working memory respects size limit."""
        # Working memory size is 5
        for i in range(10):
            memory.add_to_working_memory(f"item{i}")

        working = memory.get_working_memory()
        assert len(working) == 5
        # Should contain the most recent items
        assert "item9" in working
        assert "item5" in working
        assert "item0" not in working

    def test_clear_working_memory(self, memory):
        """Test clearing working memory."""
        memory.add_to_working_memory("item1")
        memory.add_to_working_memory("item2")
        memory.clear_working_memory()
        assert memory.get_working_memory() == []

    def test_get_categories(self, memory):
        """Test getting all categories."""
        memory.remember("f1", "fact", category="fact")
        memory.remember("p1", "pref", category="preference")
        memory.remember("i1", "insight", category="insight")

        categories = memory.get_categories()
        assert "fact" in categories
        assert "preference" in categories
        assert "insight" in categories

    def test_get_by_category(self, memory):
        """Test getting all memories in a category."""
        memory.remember("f1", "fact one", category="fact")
        memory.remember("f2", "fact two", category="fact")
        memory.remember("p1", "pref one", category="preference")

        facts = memory.get_by_category("fact")
        assert len(facts) == 2
        assert all(f.category == "fact" for f in facts)

    def test_summary(self, memory):
        """Test memory summary."""
        memory.remember("f1", "fact", category="fact")
        memory.add_to_working_memory("working")

        summary = memory.summary()
        assert "Memory:" in summary
        assert "fact:1" in summary
        assert "working: 1" in summary

    def test_auto_prune_on_max_items(self):
        """Test that memory auto-prunes when max items exceeded."""
        memory = AgentMemory(max_items=5, working_memory_size=3)

        for i in range(10):
            memory.remember(
                f"item{i}",
                f"content {i}",
                importance=i / 10,  # Lower items have lower importance
            )

        # Should have pruned to max_items
        total = len(memory._long_term)
        assert total <= 5

    def test_persistence(self, tmp_path):
        """Test saving and loading memory."""
        storage_path = tmp_path / "memory.json"

        # Create and populate memory
        memory1 = AgentMemory(storage_path=storage_path)
        memory1.remember("persistent", "will be saved", importance=0.9)
        memory1.add_to_working_memory("working item")
        # Force save
        memory1._save()

        # Create new memory from same path
        memory2 = AgentMemory(storage_path=storage_path)
        retrieved = memory2.get("persistent")
        assert retrieved is not None
        assert retrieved.content == "will be saved"

        # Note: Working memory persistence depends on implementation
        # The memory system may or may not persist working memory
        # Just verify long-term memory works


# =============================================================================
# PlanStep Tests
# =============================================================================


class TestPlanStep:
    """Tests for PlanStep class."""

    def test_default_status(self):
        """Test default status is pending."""
        step = PlanStep(description="Do something")
        assert step.status == PlanStatus.PENDING

    def test_mark_started(self):
        """Test marking step as started."""
        step = PlanStep(description="Do something")
        step.mark_started()
        assert step.status == PlanStatus.IN_PROGRESS
        assert step.started_at is not None

    def test_mark_completed(self):
        """Test marking step as completed."""
        step = PlanStep(description="Do something")
        step.mark_completed("Success result")
        assert step.status == PlanStatus.COMPLETED
        assert step.result == "Success result"
        assert step.completed_at is not None

    def test_mark_failed(self):
        """Test marking step as failed."""
        step = PlanStep(description="Do something")
        step.mark_failed("Error occurred")
        assert step.status == PlanStatus.FAILED
        assert step.error == "Error occurred"
        assert step.completed_at is not None

    def test_can_execute_no_dependencies(self):
        """Test can_execute with no dependencies."""
        step = PlanStep(description="Independent step")
        assert step.can_execute(set()) is True

    def test_can_execute_with_dependencies(self):
        """Test can_execute with dependencies."""
        dep_id = uuid4()
        step = PlanStep(description="Dependent step", dependencies=[dep_id])

        # Cannot execute without dependency completed
        assert step.can_execute(set()) is False

        # Can execute after dependency completed
        assert step.can_execute({dep_id}) is True


# =============================================================================
# Plan Tests
# =============================================================================


class TestPlan:
    """Tests for Plan class."""

    @pytest.fixture
    def simple_plan(self):
        """Create a simple plan with 3 steps."""
        return Plan(
            goal="Complete test task",
            steps=[
                PlanStep(description="Step 1"),
                PlanStep(description="Step 2"),
                PlanStep(description="Step 3"),
            ],
        )

    def test_current_step_index(self, simple_plan):
        """Test getting current step index."""
        assert simple_plan.current_step_index == 0

        simple_plan.steps[0].mark_completed("done")
        assert simple_plan.current_step_index == 1

    def test_current_step(self, simple_plan):
        """Test getting current step."""
        current = simple_plan.current_step
        assert current is not None
        assert current.description == "Step 1"

    def test_progress(self, simple_plan):
        """Test progress calculation."""
        assert simple_plan.progress == 0.0

        simple_plan.steps[0].mark_completed("done")
        assert abs(simple_plan.progress - 1 / 3) < 0.01

        simple_plan.steps[1].mark_completed("done")
        assert abs(simple_plan.progress - 2 / 3) < 0.01

        simple_plan.steps[2].mark_completed("done")
        assert simple_plan.progress == 1.0

    def test_is_complete(self, simple_plan):
        """Test is_complete property."""
        assert simple_plan.is_complete is False

        for step in simple_plan.steps:
            step.mark_completed("done")

        assert simple_plan.is_complete is True

    def test_has_failed(self, simple_plan):
        """Test has_failed property."""
        assert simple_plan.has_failed is False

        simple_plan.steps[1].mark_failed("error")
        assert simple_plan.has_failed is True

    def test_next_step(self, simple_plan):
        """Test getting next step."""
        next_step = simple_plan.next_step()
        assert next_step.description == "Step 1"

        simple_plan.steps[0].mark_completed("done")
        next_step = simple_plan.next_step()
        assert next_step.description == "Step 2"

    def test_next_step_with_dependencies(self):
        """Test next_step respects dependencies."""
        step1 = PlanStep(description="Step 1")
        step2 = PlanStep(description="Step 2", dependencies=[step1.id])

        plan = Plan(goal="Dependent plan", steps=[step1, step2])

        # Step 1 should be next (no dependencies)
        assert plan.next_step().description == "Step 1"

        # After step 1 completes, step 2 should be available
        step1.mark_completed("done")
        assert plan.next_step().description == "Step 2"

    def test_advance(self, simple_plan):
        """Test advancing to next step."""
        result = simple_plan.advance()
        assert result is True
        assert simple_plan.steps[0].status == PlanStatus.IN_PROGRESS

    def test_adapt(self, simple_plan):
        """Test plan adaptation."""
        simple_plan.adapt("Need additional steps", ["New step A", "New step B"])

        assert simple_plan.adapted_count == 1
        assert len(simple_plan.steps) == 5
        assert simple_plan.status == PlanStatus.ADAPTED
        assert len(simple_plan.adaptation_history) == 1

    def test_summary(self, simple_plan):
        """Test plan summary."""
        summary = simple_plan.summary()
        assert "Complete test task" in summary
        assert "0/3" in summary
        assert "0%" in summary


# =============================================================================
# Planner Tests
# =============================================================================


class TestPlanner:
    """Tests for Planner class."""

    @pytest.fixture
    def planner(self):
        """Create a planner instance."""
        return Planner(max_steps=10)

    def test_create_plan_from_template(self, planner):
        """Test plan creation using template."""
        plan = planner.create_plan("calibrate the step tablet")
        assert plan.goal == "calibrate the step tablet"
        assert len(plan.steps) > 0
        # Should use calibrate template
        step_descriptions = [s.description for s in plan.steps]
        assert any("scan" in d.lower() or "step tablet" in d.lower() for d in step_descriptions)

    def test_create_generic_plan(self, planner):
        """Test generic plan creation when no template matches."""
        plan = planner.create_plan("do something unusual")
        assert plan.goal == "do something unusual"
        assert len(plan.steps) > 0
        # Should use generic plan
        step_descriptions = [s.description for s in plan.steps]
        assert any("analyze" in d.lower() for d in step_descriptions)

    def test_template_matching(self, planner):
        """Test that templates are matched correctly."""
        # Test various template triggers
        templates = {
            "analyze": "analyze the calibration data",
            "troubleshoot": "troubleshoot the bronzing issue",
            "recipe": "create a recipe for warm tones",
            "predict": "predict the density response",
        }

        for template_key, goal in templates.items():
            plan = planner.create_plan(goal)
            assert len(plan.steps) > 0, f"Template {template_key} should create steps"

    def test_suggest_adaptation_for_error(self, planner):
        """Test adaptation suggestion for errors."""
        plan = Plan(goal="test", steps=[PlanStep(description="test")])
        suggestions = planner.suggest_adaptation(plan, "An error occurred during processing")
        assert suggestions is not None
        assert any("investigate" in s.lower() or "error" in s.lower() for s in suggestions)

    def test_suggest_adaptation_for_unexpected(self, planner):
        """Test adaptation suggestion for unexpected results."""
        plan = Plan(goal="test", steps=[PlanStep(description="test")])
        suggestions = planner.suggest_adaptation(plan, "Found surprising density values")
        assert suggestions is not None
        assert any("unexpected" in s.lower() or "analyze" in s.lower() for s in suggestions)

    def test_suggest_adaptation_returns_none(self, planner):
        """Test that normal observations return no adaptation."""
        plan = Plan(goal="test", steps=[PlanStep(description="test")])
        suggestions = planner.suggest_adaptation(plan, "Everything looks good")
        assert suggestions is None


# =============================================================================
# ReasoningStep Tests
# =============================================================================


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_thought_step(self):
        """Test creating a thought step."""
        step = ReasoningStep(step_type="thought", content="I should analyze the data")
        assert step.step_type == "thought"
        assert step.content == "I should analyze the data"
        assert step.tool_name is None

    def test_action_step(self):
        """Test creating an action step."""
        step = ReasoningStep(
            step_type="action",
            content="Calling analyze_densities",
            tool_name="analyze_densities",
            tool_args={"densities": [0.1, 0.5, 1.0]},
        )
        assert step.step_type == "action"
        assert step.tool_name == "analyze_densities"
        assert step.tool_args["densities"] == [0.1, 0.5, 1.0]

    def test_observation_step(self):
        """Test creating an observation step."""
        tool_result = ToolResult(success=True, data={"dmax": 2.1})
        step = ReasoningStep(
            step_type="observation",
            content="Dmax is 2.1",
            tool_result=tool_result,
        )
        assert step.step_type == "observation"
        assert step.tool_result.success is True


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        assert config.llm_settings is None
        assert config.agent_settings is None
        assert config.memory_path is None
        assert config.database is None
        assert config.predictor is None

    def test_config_with_settings(self):
        """Test configuration with custom settings."""
        llm_settings = LLMSettings(api_key="test-key")
        agent_settings = AgentSettings(max_iterations=50)

        config = AgentConfig(
            llm_settings=llm_settings,
            agent_settings=agent_settings,
        )
        assert config.llm_settings.api_key == "test-key"
        assert config.agent_settings.max_iterations == 50


# =============================================================================
# CalibrationAgent Tests
# =============================================================================


class TestCalibrationAgent:
    """Tests for CalibrationAgent class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock(
            return_value="THOUGHT: I should analyze the data\nACTION: FINISH"
        )
        return client

    @pytest.fixture
    def agent(self, mock_llm_client):
        """Create an agent with mocked LLM client."""
        config = AgentConfig(
            agent_settings=AgentSettings(max_iterations=5),
        )
        agent = CalibrationAgent(config)
        agent.client = mock_llm_client
        return agent

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig()
        agent = CalibrationAgent(config)

        assert agent.client is None  # Lazy initialization
        assert agent.tools is not None
        assert agent.memory is not None
        assert agent.planner is not None

    @pytest.mark.asyncio
    async def test_agent_run_simple(self, agent):
        """Test running agent on a simple task."""
        result = await agent.run("Analyze my calibration")
        assert result is not None
        assert len(agent._reasoning_trace) > 0

    @pytest.mark.asyncio
    async def test_agent_run_with_tool_call(self, agent, mock_llm_client):
        """Test agent run with tool execution."""
        # First call returns action, second returns final response
        mock_llm_client.complete = AsyncMock(
            side_effect=[
                'THOUGHT: I need to analyze the densities\nACTION: {"tool": "analyze_densities", "args": {"densities": [0.1, 0.5, 1.0]}}',
                "THOUGHT: Analysis complete\nACTION: FINISH",
                "The analysis shows good linearity.",
            ]
        )

        result = await agent.run("Analyze these densities: 0.1, 0.5, 1.0")
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_max_iterations(self, agent, mock_llm_client):
        """Test that agent stops at max iterations."""
        # Always return a non-FINISH action
        mock_llm_client.complete = AsyncMock(
            return_value='THOUGHT: Keep going\nACTION: {"tool": "analyze_densities", "args": {"densities": [0.1]}}'
        )

        await agent.run("Keep analyzing forever")
        # Should have stopped at max_iterations (5)
        assert agent._iteration_count <= agent.settings.max_iterations

    def test_get_reasoning_trace(self, agent):
        """Test getting reasoning trace."""
        agent._reasoning_trace = [
            ReasoningStep(step_type="thought", content="thinking"),
            ReasoningStep(step_type="action", content="acting", tool_name="test"),
        ]

        trace = agent.get_reasoning_trace()
        assert len(trace) == 2
        assert trace[0]["type"] == "thought"
        assert trace[1]["type"] == "action"

    def test_get_plan_status(self, agent):
        """Test getting plan status."""
        agent._current_plan = Plan(
            goal="Test goal",
            steps=[
                PlanStep(description="Step 1"),
                PlanStep(description="Step 2"),
            ],
        )
        agent._current_plan.steps[0].mark_completed("done")

        status = agent.get_plan_status()
        assert status["goal"] == "Test goal"
        assert status["progress"] == 0.5
        assert status["total_steps"] == 2

    def test_get_plan_status_none(self, agent):
        """Test plan status when no plan exists."""
        agent._current_plan = None
        status = agent.get_plan_status()
        assert status is None


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_agent_minimal(self):
        """Test creating agent with minimal config."""
        agent = create_agent()
        assert agent is not None
        assert isinstance(agent, CalibrationAgent)

    def test_create_agent_with_api_key(self):
        """Test creating agent with API key."""
        agent = create_agent(api_key="test-key")
        assert agent.config.llm_settings.api_key == "test-key"

    def test_create_agent_with_database(self, populated_database):
        """Test creating agent with database."""
        agent = create_agent(database=populated_database)
        # Should have database-related tools
        tool_names = [t.name for t in agent.tools.list_tools()]
        assert "search_calibrations" in tool_names

    def test_create_agent_with_memory_path(self, tmp_path):
        """Test creating agent with memory path."""
        memory_path = tmp_path / "agent_memory.json"
        agent = create_agent(memory_path=memory_path)
        assert agent.memory.storage_path == memory_path


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Integration tests for agent components working together."""

    @pytest.mark.asyncio
    async def test_agent_memory_persistence_across_runs(self, tmp_path):
        """Test that agent memory persists across runs."""
        memory_path = tmp_path / "persistent_memory.json"

        # First agent run
        agent1 = create_agent(memory_path=memory_path)
        agent1.memory.remember("calibration_insight", "Na2 increases contrast", importance=0.9)

        # Second agent (new instance, same memory path)
        agent2 = create_agent(memory_path=memory_path)
        recalled = agent2.memory.recall("contrast")
        assert len(recalled) > 0
        assert any("Na2" in r.content for r in recalled)

    def test_planner_and_tools_integration(self):
        """Test that planner creates executable plans with available tools."""
        agent = create_agent()
        plan = agent.planner.create_plan("analyze density measurements")

        # Plan should have steps
        assert len(plan.steps) > 0

        # Tools should be available for analysis
        tool_names = [t.name for t in agent.tools.list_tools()]
        assert "analyze_densities" in tool_names
