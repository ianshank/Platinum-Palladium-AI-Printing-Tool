"""Tests for MCP server."""

import json
import pytest


class TestMCPError:
    """Tests for MCPError."""

    def test_to_dict(self):
        """Test converting error to dict."""
        from ptpd_calibration.mcp.server import MCPError

        error = MCPError(code=-32600, message="Invalid request")

        result = error.to_dict()

        assert result["code"] == -32600
        assert result["message"] == "Invalid request"

    def test_to_dict_with_data(self):
        """Test converting error with data to dict."""
        from ptpd_calibration.mcp.server import MCPError

        error = MCPError(
            code=-32602,
            message="Invalid params",
            data={"param": "missing_field"},
        )

        result = error.to_dict()

        assert result["data"]["param"] == "missing_field"


class TestMCPRequest:
    """Tests for MCPRequest."""

    def test_from_dict(self):
        """Test creating request from dict."""
        from ptpd_calibration.mcp.server import MCPRequest

        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"category": "analysis"},
        }

        request = MCPRequest.from_dict(data)

        assert request.jsonrpc == "2.0"
        assert request.id == 1
        assert request.method == "tools/list"
        assert request.params["category"] == "analysis"

    def test_from_dict_minimal(self):
        """Test creating request with minimal data."""
        from ptpd_calibration.mcp.server import MCPRequest

        data = {"method": "initialize"}

        request = MCPRequest.from_dict(data)

        assert request.method == "initialize"
        assert request.id is None
        assert request.params == {}


class TestMCPResponse:
    """Tests for MCPResponse."""

    def test_success_response(self):
        """Test successful response."""
        from ptpd_calibration.mcp.server import MCPResponse

        response = MCPResponse(
            id=1,
            result={"tools": []},
        )

        result = response.to_dict()

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"]["tools"] == []
        assert "error" not in result

    def test_error_response(self):
        """Test error response."""
        from ptpd_calibration.mcp.server import MCPResponse, MCPError

        response = MCPResponse(
            id=1,
            error=MCPError(code=-32600, message="Invalid request"),
        )

        result = response.to_dict()

        assert result["id"] == 1
        assert result["error"]["code"] == -32600
        assert "result" not in result

    def test_to_json(self):
        """Test JSON serialization."""
        from ptpd_calibration.mcp.server import MCPResponse

        response = MCPResponse(id=1, result={"status": "ok"})

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == 1
        assert parsed["result"]["status"] == "ok"


class TestServerCapabilities:
    """Tests for ServerCapabilities."""

    def test_to_dict_all_enabled(self):
        """Test capabilities dict with all enabled."""
        from ptpd_calibration.mcp.server import ServerCapabilities

        caps = ServerCapabilities(tools=True, resources=True, prompts=True)

        result = caps.to_dict()

        assert "tools" in result
        assert "resources" in result
        assert "prompts" in result

    def test_to_dict_partial(self):
        """Test capabilities dict with partial enabled."""
        from ptpd_calibration.mcp.server import ServerCapabilities

        caps = ServerCapabilities(tools=True, resources=False, prompts=False)

        result = caps.to_dict()

        assert "tools" in result
        assert "resources" not in result
        assert "prompts" not in result


class TestMCPServer:
    """Tests for MCPServer."""

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, mcp_server):
        """Test handling invalid JSON."""
        response = await mcp_server.handle_message("not valid json")

        result = json.loads(response)
        assert result["error"]["code"] == -32700  # Parse error

    @pytest.mark.asyncio
    async def test_handle_method_not_found(self, mcp_server):
        """Test handling unknown method."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert result["error"]["code"] == -32601  # Method not found

    @pytest.mark.asyncio
    async def test_handle_initialize(self, mcp_server):
        """Test initialize handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0",
                },
            },
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "result" in result
        assert result["result"]["serverInfo"]["name"] == "test-mcp-server"
        assert "capabilities" in result["result"]

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, mcp_server):
        """Test tools/list handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "tools" in result["result"]
        assert len(result["result"]["tools"]) > 0

    @pytest.mark.asyncio
    async def test_handle_call_tool(self, mcp_server, sample_densities):
        """Test tools/call handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_densities",
                "arguments": {
                    "densities": sample_densities,
                },
            },
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "content" in result["result"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_not_found(self, mcp_server):
        """Test calling non-existent tool."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "error" in result
        assert "not found" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_handle_list_resources(self, mcp_server):
        """Test resources/list handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list",
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "resources" in result["result"]
        assert len(result["result"]["resources"]) > 0

    @pytest.mark.asyncio
    async def test_handle_read_resource(self, mcp_server):
        """Test resources/read handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {
                "uri": "ptpd://system/info",
            },
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "contents" in result["result"]
        assert len(result["result"]["contents"]) > 0

    @pytest.mark.asyncio
    async def test_handle_read_resource_not_found(self, mcp_server):
        """Test reading non-existent resource."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/read",
            "params": {
                "uri": "ptpd://nonexistent",
            },
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "error" in result
        assert "not found" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_handle_shutdown(self, mcp_server):
        """Test shutdown handler."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "shutdown",
        })

        response = await mcp_server.handle_message(message)

        result = json.loads(response)
        assert "result" in result
        assert mcp_server._running is False

    @pytest.mark.asyncio
    async def test_notification_no_response(self, mcp_server):
        """Test that notifications (no id) don't get responses."""
        message = json.dumps({
            "jsonrpc": "2.0",
            "method": "unknown/notification",
            # No id = notification
        })

        response = await mcp_server.handle_message(message)

        assert response is None


class TestCreateMCPServer:
    """Tests for create_mcp_server factory."""

    def test_create_with_defaults(self):
        """Test creating server with default settings."""
        from ptpd_calibration.mcp.server import create_mcp_server

        server = create_mcp_server()

        assert server is not None
        assert server.tool_registry is not None
        assert server.resource_registry is not None

    def test_create_with_custom_settings(self, mcp_settings):
        """Test creating server with custom settings."""
        from ptpd_calibration.mcp.server import create_mcp_server

        server = create_mcp_server(settings=mcp_settings)

        assert server.settings.name == "test-mcp-server"

    def test_create_with_database(self, mock_database):
        """Test creating server with database."""
        from ptpd_calibration.mcp.server import create_mcp_server

        server = create_mcp_server(database=mock_database)

        # Server should have database-related tools
        assert server.tool_registry.get("search_calibrations") is not None


class TestMCPMessageType:
    """Tests for MCPMessageType enum."""

    def test_message_types(self):
        """Test all message types are defined."""
        from ptpd_calibration.mcp.server import MCPMessageType

        assert MCPMessageType.INITIALIZE.value == "initialize"
        assert MCPMessageType.LIST_TOOLS.value == "tools/list"
        assert MCPMessageType.CALL_TOOL.value == "tools/call"
        assert MCPMessageType.LIST_RESOURCES.value == "resources/list"
        assert MCPMessageType.READ_RESOURCE.value == "resources/read"


class TestMCPErrorCode:
    """Tests for MCPErrorCode constants."""

    def test_standard_error_codes(self):
        """Test standard JSON-RPC error codes."""
        from ptpd_calibration.mcp.server import MCPErrorCode

        assert MCPErrorCode.PARSE_ERROR == -32700
        assert MCPErrorCode.INVALID_REQUEST == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND == -32601
        assert MCPErrorCode.INVALID_PARAMS == -32602
        assert MCPErrorCode.INTERNAL_ERROR == -32603

    def test_custom_error_codes(self):
        """Test custom error codes."""
        from ptpd_calibration.mcp.server import MCPErrorCode

        assert MCPErrorCode.RESOURCE_NOT_FOUND == -32001
        assert MCPErrorCode.TOOL_NOT_FOUND == -32002
        assert MCPErrorCode.TOOL_EXECUTION_ERROR == -32003
