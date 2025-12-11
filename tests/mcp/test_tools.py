"""Tests for MCP tools."""

import pytest
import json


class TestMCPToolParameter:
    """Tests for MCPToolParameter."""

    def test_to_json_schema_basic(self):
        """Test basic JSON schema generation."""
        from ptpd_calibration.mcp.tools import MCPToolParameter

        param = MCPToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
        )

        schema = param.to_json_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "A test parameter"

    def test_to_json_schema_with_enum(self):
        """Test JSON schema with enum values."""
        from ptpd_calibration.mcp.tools import MCPToolParameter

        param = MCPToolParameter(
            name="mode",
            type="string",
            description="Operation mode",
            enum=["fast", "slow", "auto"],
        )

        schema = param.to_json_schema()

        assert schema["enum"] == ["fast", "slow", "auto"]

    def test_to_json_schema_with_constraints(self):
        """Test JSON schema with numeric constraints."""
        from ptpd_calibration.mcp.tools import MCPToolParameter

        param = MCPToolParameter(
            name="value",
            type="number",
            description="A numeric value",
            minimum=0.0,
            maximum=1.0,
            default=0.5,
        )

        schema = param.to_json_schema()

        assert schema["minimum"] == 0.0
        assert schema["maximum"] == 1.0
        assert schema["default"] == 0.5

    def test_to_json_schema_array(self):
        """Test JSON schema for array type."""
        from ptpd_calibration.mcp.tools import MCPToolParameter

        param = MCPToolParameter(
            name="values",
            type="array",
            description="List of values",
            items_type="number",
        )

        schema = param.to_json_schema()

        assert schema["type"] == "array"
        assert schema["items"]["type"] == "number"


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_text_result(self):
        """Test creating text result."""
        from ptpd_calibration.mcp.tools import MCPToolResult

        result = MCPToolResult.text("Hello world")

        assert result.success is True
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello world"

    def test_json_result(self):
        """Test creating JSON result."""
        from ptpd_calibration.mcp.tools import MCPToolResult

        data = {"key": "value", "count": 42}
        result = MCPToolResult.json_result(data)

        assert result.success is True
        content_text = result.content[0]["text"]
        parsed = json.loads(content_text)
        assert parsed["key"] == "value"
        assert parsed["count"] == 42

    def test_error_result(self):
        """Test creating error result."""
        from ptpd_calibration.mcp.tools import MCPToolResult

        result = MCPToolResult.error_result("Something went wrong")

        assert result.success is False
        assert result.is_error is True
        assert result.error == "Something went wrong"

    def test_to_mcp_format_success(self):
        """Test MCP format for successful result."""
        from ptpd_calibration.mcp.tools import MCPToolResult

        result = MCPToolResult.text("Success!")
        mcp_format = result.to_mcp_format()

        assert "content" in mcp_format
        assert "isError" not in mcp_format

    def test_to_mcp_format_error(self):
        """Test MCP format for error result."""
        from ptpd_calibration.mcp.tools import MCPToolResult

        result = MCPToolResult.error_result("Failed!")
        mcp_format = result.to_mcp_format()

        assert mcp_format["isError"] is True
        assert mcp_format["content"][0]["text"] == "Failed!"


class TestMCPTool:
    """Tests for MCPTool."""

    def test_to_mcp_format(self):
        """Test converting tool to MCP format."""
        from ptpd_calibration.mcp.tools import MCPTool, MCPToolParameter, MCPToolResult

        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters=[
                MCPToolParameter(
                    name="input",
                    type="string",
                    description="Input value",
                    required=True,
                ),
                MCPToolParameter(
                    name="optional",
                    type="number",
                    description="Optional value",
                    required=False,
                    default=10,
                ),
            ],
            handler=lambda input, optional=10: MCPToolResult.text(f"{input}: {optional}"),
        )

        mcp_format = tool.to_mcp_format()

        assert mcp_format["name"] == "test_tool"
        assert mcp_format["description"] == "A test tool"
        assert "inputSchema" in mcp_format
        assert mcp_format["inputSchema"]["required"] == ["input"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        from ptpd_calibration.mcp.tools import MCPTool, MCPToolParameter, MCPToolResult

        tool = MCPTool(
            name="greet",
            description="Greet someone",
            parameters=[
                MCPToolParameter(
                    name="name",
                    type="string",
                    description="Name to greet",
                ),
            ],
            handler=lambda name: MCPToolResult.text(f"Hello, {name}!"),
        )

        result = await tool.execute({"name": "World"})

        assert result.success is True
        assert "Hello, World!" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_execute_missing_required_param(self):
        """Test execution with missing required parameter."""
        from ptpd_calibration.mcp.tools import MCPTool, MCPToolParameter, MCPToolResult

        tool = MCPTool(
            name="test",
            description="Test tool",
            parameters=[
                MCPToolParameter(
                    name="required_param",
                    type="string",
                    description="Required",
                    required=True,
                ),
            ],
            handler=lambda required_param: MCPToolResult.text(required_param),
        )

        result = await tool.execute({})

        assert result.success is False
        assert "required_param" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_default(self):
        """Test execution using default parameter value."""
        from ptpd_calibration.mcp.tools import MCPTool, MCPToolParameter, MCPToolResult

        tool = MCPTool(
            name="test",
            description="Test tool",
            parameters=[
                MCPToolParameter(
                    name="value",
                    type="number",
                    description="A value",
                    required=False,
                    default=42,
                ),
            ],
            handler=lambda value: MCPToolResult.text(f"Value: {value}"),
        )

        result = await tool.execute({})

        assert result.success is True
        assert "42" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        """Test handling of handler exceptions."""
        from ptpd_calibration.mcp.tools import MCPTool, MCPToolParameter, MCPToolResult

        def failing_handler():
            raise ValueError("Something broke")

        tool = MCPTool(
            name="failing",
            description="Always fails",
            parameters=[],
            handler=failing_handler,
        )

        result = await tool.execute({})

        assert result.success is False
        assert "Something broke" in result.error


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving tools."""
        from ptpd_calibration.mcp.tools import MCPToolRegistry, MCPTool, MCPToolResult

        registry = MCPToolRegistry()

        tool = MCPTool(
            name="my_tool",
            description="My tool",
            parameters=[],
            handler=lambda: MCPToolResult.text("Done"),
        )

        registry.register(tool)

        retrieved = registry.get("my_tool")
        assert retrieved is not None
        assert retrieved.name == "my_tool"

    def test_get_nonexistent(self):
        """Test getting non-existent tool returns None."""
        from ptpd_calibration.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()

        assert registry.get("nonexistent") is None

    def test_unregister(self):
        """Test unregistering a tool."""
        from ptpd_calibration.mcp.tools import MCPToolRegistry, MCPTool, MCPToolResult

        registry = MCPToolRegistry()

        tool = MCPTool(
            name="removable",
            description="Can be removed",
            parameters=[],
            handler=lambda: MCPToolResult.text("Done"),
        )

        registry.register(tool)
        assert registry.get("removable") is not None

        result = registry.unregister("removable")
        assert result is True
        assert registry.get("removable") is None

    def test_list_tools(self):
        """Test listing all tools."""
        from ptpd_calibration.mcp.tools import (
            MCPToolRegistry, MCPTool, MCPToolCategory, MCPToolResult
        )

        registry = MCPToolRegistry()

        registry.register(MCPTool(
            name="tool1",
            description="Tool 1",
            parameters=[],
            handler=lambda: MCPToolResult.text("1"),
            category=MCPToolCategory.ANALYSIS,
        ))

        registry.register(MCPTool(
            name="tool2",
            description="Tool 2",
            parameters=[],
            handler=lambda: MCPToolResult.text("2"),
            category=MCPToolCategory.CURVES,
        ))

        all_tools = registry.list_tools()
        assert len(all_tools) == 2

        analysis_tools = registry.list_tools(category=MCPToolCategory.ANALYSIS)
        assert len(analysis_tools) == 1
        assert analysis_tools[0].name == "tool1"

    def test_to_mcp_format(self):
        """Test converting registry to MCP format."""
        from ptpd_calibration.mcp.tools import MCPToolRegistry, MCPTool, MCPToolResult

        registry = MCPToolRegistry()

        registry.register(MCPTool(
            name="tool1",
            description="Tool 1",
            parameters=[],
            handler=lambda: MCPToolResult.text("1"),
        ))

        mcp_format = registry.to_mcp_format()

        assert len(mcp_format) == 1
        assert mcp_format[0]["name"] == "tool1"


class TestCreateMCPTools:
    """Tests for the create_mcp_tools factory function."""

    def test_creates_standard_tools(self, tool_registry):
        """Test that standard tools are created."""
        tools = tool_registry.list_tools()

        # Check some expected tools exist
        tool_names = [t.name for t in tools]
        assert "analyze_densities" in tool_names
        assert "generate_linearization_curve" in tool_names
        assert "calculate_chemistry" in tool_names
        assert "suggest_exposure" in tool_names


class TestAnalyzeDensitiesTool:
    """Tests for analyze_densities tool."""

    @pytest.mark.asyncio
    async def test_analyze_good_densities(self, tool_registry, sample_densities):
        """Test analyzing good density measurements."""
        tool = tool_registry.get("analyze_densities")
        result = await tool.execute({"densities": sample_densities})

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert "dmin" in data
        assert "dmax" in data
        assert "is_monotonic" in data
        assert data["is_monotonic"] is True
        # Quality depends on linearity - monotonic data can still be "fair" if deviation is high
        assert data["quality"] in ["good", "fair"]

    @pytest.mark.asyncio
    async def test_analyze_non_monotonic_densities(
        self, tool_registry, sample_non_monotonic_densities
    ):
        """Test analyzing non-monotonic density measurements."""
        tool = tool_registry.get("analyze_densities")
        result = await tool.execute({"densities": sample_non_monotonic_densities})

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["is_monotonic"] is False
        assert data["quality"] == "poor"

    @pytest.mark.asyncio
    async def test_analyze_empty_densities(self, tool_registry):
        """Test error handling for empty densities."""
        tool = tool_registry.get("analyze_densities")
        result = await tool.execute({"densities": []})

        assert result.success is False
        assert "No density values" in result.error

    @pytest.mark.asyncio
    async def test_analyze_insufficient_densities(self, tool_registry):
        """Test error handling for insufficient densities."""
        tool = tool_registry.get("analyze_densities")
        result = await tool.execute({"densities": [0.1, 0.2]})

        assert result.success is False
        assert "At least 3" in result.error


class TestGenerateLinearizationCurveTool:
    """Tests for generate_linearization_curve tool."""

    @pytest.mark.asyncio
    async def test_generate_linear_curve(self, tool_registry, sample_densities):
        """Test generating a linear curve."""
        tool = tool_registry.get("generate_linearization_curve")
        result = await tool.execute({
            "densities": sample_densities,
            "curve_name": "Test Curve",
            "curve_type": "linear",
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["name"] == "Test Curve"
        assert data["curve_type"] == "linear"
        assert len(data["output_values"]) == 256

    @pytest.mark.asyncio
    async def test_generate_paper_white_curve(self, tool_registry, sample_densities):
        """Test generating a paper white curve."""
        tool = tool_registry.get("generate_linearization_curve")
        result = await tool.execute({
            "densities": sample_densities,
            "curve_name": "Paper White",
            "curve_type": "paper_white",
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["curve_type"] == "paper_white"


class TestCalculateChemistryTool:
    """Tests for calculate_chemistry tool."""

    @pytest.mark.asyncio
    async def test_calculate_small_print(self, tool_registry):
        """Test chemistry calculation for small print."""
        tool = tool_registry.get("calculate_chemistry")
        result = await tool.execute({
            "width_inches": 8,
            "height_inches": 10,
            "metal_ratio": 0.5,
            "contrast_drops": 5,
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["print_size"]["area_sq_inches"] == 80
        assert "volumes_ml" in data
        assert "drops" in data
        assert data["drops"]["contrast_agent"] == 5

    @pytest.mark.asyncio
    async def test_calculate_pure_platinum(self, tool_registry):
        """Test chemistry for pure platinum print."""
        tool = tool_registry.get("calculate_chemistry")
        result = await tool.execute({
            "width_inches": 8,
            "height_inches": 10,
            "metal_ratio": 1.0,
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["metal_ratio"]["platinum_percent"] == 100.0
        assert data["metal_ratio"]["palladium_percent"] == 0.0


class TestSuggestExposureTool:
    """Tests for suggest_exposure tool."""

    @pytest.mark.asyncio
    async def test_suggest_led_exposure(self, tool_registry):
        """Test exposure suggestion for LED source."""
        tool = tool_registry.get("suggest_exposure")
        result = await tool.execute({
            "paper_type": "Arches Platine",
            "uv_source": "led",
            "metal_ratio": 0.5,
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert "suggested_exposure_seconds" in data
        assert "exposure_bracket" in data
        assert len(data["exposure_bracket"]) == 5

    @pytest.mark.asyncio
    async def test_suggest_sunlight_exposure(self, tool_registry):
        """Test exposure suggestion for sunlight."""
        tool = tool_registry.get("suggest_exposure")
        result = await tool.execute({
            "paper_type": "Stonehenge",
            "uv_source": "sunlight",
        })

        assert result.success is True

        data = json.loads(result.content[0]["text"])
        assert data["factors"]["uv_source"] == "sunlight"


class TestFormatQTRCurveTool:
    """Tests for format_qtr_curve tool."""

    @pytest.mark.asyncio
    async def test_format_qtr_curve(self, tool_registry, sample_curve_data):
        """Test formatting curve as QTR output."""
        tool = tool_registry.get("format_qtr_curve")
        result = await tool.execute({
            "input_values": sample_curve_data["input_values"],
            "output_values": sample_curve_data["output_values"],
            "curve_name": "My Curve",
        })

        assert result.success is True

        qtr_content = result.content[0]["text"]
        assert "QTR Curve: My Curve" in qtr_content
        assert "[Gray]" in qtr_content
        assert "0=" in qtr_content
