"""
Unit tests for agents/tools.py module.

Tests tool definitions, parameters, results, and registry.
"""

import pytest

from ptpd_calibration.agents.tools import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_calibration_tools,
)


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert ToolCategory.ANALYSIS.value == "analysis"
        assert ToolCategory.DATABASE.value == "database"
        assert ToolCategory.PREDICTION.value == "prediction"
        assert ToolCategory.CURVES.value == "curves"
        assert ToolCategory.PLANNING.value == "planning"
        assert ToolCategory.MEMORY.value == "memory"


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_create_parameter(self):
        """Test creating a tool parameter."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
        )
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is True
        assert param.enum is None
        assert param.default is None

    def test_optional_parameter(self):
        """Test creating optional parameter."""
        param = ToolParameter(
            name="optional",
            type="integer",
            description="Optional param",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="choice",
            type="string",
            description="Choice param",
            enum=["option1", "option2", "option3"],
        )
        assert param.enum == ["option1", "option2", "option3"]


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_successful_result(self):
        """Test creating successful result."""
        result = ToolResult(
            success=True,
            data={"value": 42},
        )
        assert result.success is True
        assert result.data == {"value": 42}
        assert result.error is None

    def test_failed_result(self):
        """Test creating failed result."""
        result = ToolResult(
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = ToolResult(
            success=True,
            data="test",
            metadata={"duration_ms": 100},
        )
        assert result.metadata == {"duration_ms": 100}

    def test_to_string_success_dict(self):
        """Test to_string with dict data."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
        )
        output = result.to_string()
        assert "key" in output
        assert "value" in output

    def test_to_string_success_list(self):
        """Test to_string with list data."""
        result = ToolResult(
            success=True,
            data=[1, 2, 3],
        )
        output = result.to_string()
        assert "1" in output
        assert "2" in output
        assert "3" in output

    def test_to_string_success_string(self):
        """Test to_string with string data."""
        result = ToolResult(
            success=True,
            data="plain text",
        )
        assert result.to_string() == "plain text"

    def test_to_string_error(self):
        """Test to_string with error."""
        result = ToolResult(
            success=False,
            error="Test error",
        )
        assert result.to_string() == "Error: Test error"


class TestTool:
    """Tests for Tool dataclass."""

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for testing."""
        return Tool(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="input",
                    type="string",
                    description="Input value",
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Count value",
                    required=False,
                    default=1,
                ),
            ],
            handler=lambda input, count=1: {"result": input * count},
        )

    def test_create_tool(self, sample_tool):
        """Test creating a tool."""
        assert sample_tool.name == "test_tool"
        assert sample_tool.description == "A test tool"
        assert len(sample_tool.parameters) == 2
        assert sample_tool.category == ToolCategory.ANALYSIS

    def test_tool_with_category(self):
        """Test creating tool with specific category."""
        tool = Tool(
            name="db_tool",
            description="Database tool",
            parameters=[],
            handler=lambda: None,
            category=ToolCategory.DATABASE,
        )
        assert tool.category == ToolCategory.DATABASE

    def test_to_anthropic_format(self, sample_tool):
        """Test conversion to Anthropic format."""
        anthropic = sample_tool.to_anthropic_format()

        assert anthropic["name"] == "test_tool"
        assert anthropic["description"] == "A test tool"
        assert "input_schema" in anthropic

        schema = anthropic["input_schema"]
        assert schema["type"] == "object"
        assert "input" in schema["properties"]
        assert "count" in schema["properties"]
        assert "input" in schema["required"]
        assert "count" not in schema["required"]

    def test_to_anthropic_format_with_enum(self):
        """Test Anthropic format with enum parameter."""
        tool = Tool(
            name="enum_tool",
            description="Tool with enum",
            parameters=[
                ToolParameter(
                    name="choice",
                    type="string",
                    description="Pick one",
                    enum=["a", "b", "c"],
                ),
            ],
            handler=lambda choice: choice,
        )
        anthropic = tool.to_anthropic_format()
        schema = anthropic["input_schema"]
        assert schema["properties"]["choice"]["enum"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_execute_success(self, sample_tool):
        """Test executing tool successfully."""
        result = await sample_tool.execute(input="test", count=3)
        assert result.success is True
        assert result.data == {"result": "testtesttest"}

    @pytest.mark.asyncio
    async def test_execute_returns_toolresult(self):
        """Test executing tool that returns ToolResult."""
        tool = Tool(
            name="result_tool",
            description="Returns ToolResult",
            parameters=[],
            handler=lambda: ToolResult(success=True, data="custom"),
        )
        result = await tool.execute()
        assert result.success is True
        assert result.data == "custom"

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test executing tool with error."""
        def error_handler():
            raise ValueError("Test error")

        tool = Tool(
            name="error_tool",
            description="Raises error",
            parameters=[],
            handler=error_handler,
        )
        result = await tool.execute()
        assert result.success is False
        assert "Test error" in result.error


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return ToolRegistry()

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool."""
        return Tool(
            name="sample",
            description="Sample tool",
            parameters=[],
            handler=lambda: "result",
        )

    def test_create_registry(self, registry):
        """Test creating a registry."""
        assert registry._tools == {}

    def test_register_tool(self, registry, sample_tool):
        """Test registering a tool."""
        registry.register(sample_tool)
        assert "sample" in registry._tools

    def test_get_tool(self, registry, sample_tool):
        """Test getting a tool by name."""
        registry.register(sample_tool)
        tool = registry.get("sample")
        assert tool is not None
        assert tool.name == "sample"

    def test_get_nonexistent_tool(self, registry):
        """Test getting nonexistent tool."""
        tool = registry.get("nonexistent")
        assert tool is None

    def test_list_tools(self, registry):
        """Test listing all tools."""
        tool1 = Tool(
            name="tool1",
            description="First",
            parameters=[],
            handler=lambda: None,
            category=ToolCategory.ANALYSIS,
        )
        tool2 = Tool(
            name="tool2",
            description="Second",
            parameters=[],
            handler=lambda: None,
            category=ToolCategory.DATABASE,
        )
        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_tools_by_category(self, registry):
        """Test listing tools filtered by category."""
        tool1 = Tool(
            name="analysis_tool",
            description="Analysis",
            parameters=[],
            handler=lambda: None,
            category=ToolCategory.ANALYSIS,
        )
        tool2 = Tool(
            name="db_tool",
            description="Database",
            parameters=[],
            handler=lambda: None,
            category=ToolCategory.DATABASE,
        )
        registry.register(tool1)
        registry.register(tool2)

        analysis_tools = registry.list_tools(category=ToolCategory.ANALYSIS)
        assert len(analysis_tools) == 1
        assert analysis_tools[0].name == "analysis_tool"

    def test_to_anthropic_format(self, registry, sample_tool):
        """Test converting registry to Anthropic format."""
        registry.register(sample_tool)
        formats = registry.to_anthropic_format()
        assert len(formats) == 1
        assert formats[0]["name"] == "sample"


class TestCreateCalibrationTools:
    """Tests for create_calibration_tools function."""

    def test_create_without_database(self):
        """Test creating tools without database."""
        registry = create_calibration_tools()
        tools = registry.list_tools()
        # Should have analysis, curve, and planning tools but not database
        tool_names = [t.name for t in tools]
        assert "analyze_densities" in tool_names
        assert "generate_curve" in tool_names
        assert "suggest_parameters" in tool_names
        assert "create_test_plan" in tool_names
        # Database tools should not be present
        assert "search_calibrations" not in tool_names
        assert "get_calibration" not in tool_names
        assert "save_calibration" not in tool_names

    def test_analysis_tool_present(self):
        """Test analyze_densities tool is present."""
        registry = create_calibration_tools()
        tool = registry.get("analyze_densities")
        assert tool is not None
        assert tool.category == ToolCategory.ANALYSIS

    def test_curve_tool_present(self):
        """Test generate_curve tool is present."""
        registry = create_calibration_tools()
        tool = registry.get("generate_curve")
        assert tool is not None
        assert tool.category == ToolCategory.CURVES

    def test_planning_tools_present(self):
        """Test planning tools are present."""
        registry = create_calibration_tools()
        suggest = registry.get("suggest_parameters")
        plan = registry.get("create_test_plan")
        assert suggest is not None
        assert plan is not None
        assert suggest.category == ToolCategory.PLANNING
        assert plan.category == ToolCategory.PLANNING
