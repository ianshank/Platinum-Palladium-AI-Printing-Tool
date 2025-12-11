"""
MCP server implementation for PTPD Calibration.

This module provides an MCP (Model Context Protocol) server that exposes
calibration tools and resources for integration with LM Studio and other
MCP-compatible clients.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from ptpd_calibration.mcp.config import (
    MCPServerSettings,
    MCPSettings,
    TransportType,
    get_mcp_settings,
)
from ptpd_calibration.mcp.resources import ResourceRegistry, create_calibration_resources
from ptpd_calibration.mcp.tools import MCPToolRegistry, create_mcp_tools

logger = logging.getLogger(__name__)


class MCPMessageType(str, Enum):
    """MCP protocol message types."""

    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"

    # Capabilities
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"

    # Notifications
    PROGRESS = "notifications/progress"
    LOG = "notifications/log"

    # Errors
    ERROR = "error"


@dataclass
class MCPError:
    """MCP error representation."""

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


class MCPErrorCode:
    """Standard MCP error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes
    RESOURCE_NOT_FOUND = -32001
    TOOL_NOT_FOUND = -32002
    TOOL_EXECUTION_ERROR = -32003


@dataclass
class ServerCapabilities:
    """Server capabilities declaration."""

    tools: bool = True
    resources: bool = True
    prompts: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP capabilities format."""
        caps: dict[str, Any] = {}

        if self.tools:
            caps["tools"] = {}
        if self.resources:
            caps["resources"] = {}
        if self.prompts:
            caps["prompts"] = {}

        return caps


@dataclass
class MCPRequest:
    """Parsed MCP request."""

    jsonrpc: str
    method: str
    id: Optional[str | int] = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPRequest":
        """Parse request from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            id=data.get("id"),
            params=data.get("params", {}),
        )


@dataclass
class MCPResponse:
    """MCP response."""

    jsonrpc: str = "2.0"
    id: Optional[str | int] = None
    result: Optional[Any] = None
    error: Optional[MCPError] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        response: dict[str, Any] = {"jsonrpc": self.jsonrpc}

        if self.id is not None:
            response["id"] = self.id

        if self.error is not None:
            response["error"] = self.error.to_dict()
        elif self.result is not None:
            response["result"] = self.result

        return response

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class MCPServer:
    """
    MCP server for PTPD Calibration.

    This server implements the Model Context Protocol to expose calibration
    tools and resources for use by LM Studio and other MCP clients.

    Example:
        ```python
        # Create and run server
        server = create_mcp_server()
        await server.run()
        ```

    Environment variables:
        PTPD_MCP_SERVER_NAME: Server name
        PTPD_MCP_SERVER_TRANSPORT: Transport type (stdio, sse, websocket)
        PTPD_MCP_SERVER_PORT: Port for network transports
    """

    def __init__(
        self,
        settings: Optional[MCPServerSettings] = None,
        tool_registry: Optional[MCPToolRegistry] = None,
        resource_registry: Optional[ResourceRegistry] = None,
        database: Optional[Any] = None,
    ):
        """
        Initialize MCP server.

        Args:
            settings: Server settings. Uses defaults from environment if not provided.
            tool_registry: Custom tool registry. Creates default if not provided.
            resource_registry: Custom resource registry. Creates default if not provided.
            database: CalibrationDatabase for data access.
        """
        self.settings = settings or get_mcp_settings().server
        self._setup_logging()

        # Initialize registries
        self.tool_registry = tool_registry or create_mcp_tools(database=database)
        self.resource_registry = resource_registry or create_calibration_resources(
            database=database
        )

        # Server state
        self._initialized = False
        self._running = False
        self._client_info: Optional[dict[str, Any]] = None

        # Capabilities
        self.capabilities = ServerCapabilities(
            tools=self.settings.enable_tools,
            resources=self.settings.enable_resources,
            prompts=self.settings.enable_prompts,
        )

        logger.info(
            f"MCP Server initialized: {self.settings.name} v{self.settings.version}"
        )

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.settings.log_level)
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if self.settings.debug_mode:
            logger.setLevel(logging.DEBUG)

    async def handle_message(self, message: str) -> Optional[str]:
        """
        Handle an incoming MCP message.

        Args:
            message: JSON-RPC message string.

        Returns:
            Response JSON string, or None for notifications.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return MCPResponse(
                error=MCPError(MCPErrorCode.PARSE_ERROR, f"Invalid JSON: {e}")
            ).to_json()

        request = MCPRequest.from_dict(data)

        # Dispatch to handler
        handler = self._get_handler(request.method)
        if handler is None:
            if request.id is not None:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        MCPErrorCode.METHOD_NOT_FOUND,
                        f"Method not found: {request.method}",
                    ),
                ).to_json()
            return None

        try:
            result = await handler(request)

            if request.id is not None:
                return MCPResponse(id=request.id, result=result).to_json()
            return None

        except Exception as e:
            logger.exception(f"Error handling {request.method}")
            if request.id is not None:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(MCPErrorCode.INTERNAL_ERROR, str(e)),
                ).to_json()
            return None

    def _get_handler(self, method: str) -> Optional[Any]:
        """Get handler for a method."""
        handlers = {
            MCPMessageType.INITIALIZE.value: self._handle_initialize,
            MCPMessageType.SHUTDOWN.value: self._handle_shutdown,
            MCPMessageType.LIST_TOOLS.value: self._handle_list_tools,
            MCPMessageType.CALL_TOOL.value: self._handle_call_tool,
            MCPMessageType.LIST_RESOURCES.value: self._handle_list_resources,
            MCPMessageType.READ_RESOURCE.value: self._handle_read_resource,
            MCPMessageType.LIST_PROMPTS.value: self._handle_list_prompts,
            MCPMessageType.GET_PROMPT.value: self._handle_get_prompt,
        }
        return handlers.get(method)

    async def _handle_initialize(self, request: MCPRequest) -> dict[str, Any]:
        """Handle initialize request."""
        self._client_info = request.params.get("clientInfo")
        self._initialized = True

        logger.info(f"Client initialized: {self._client_info}")

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": {
                "name": self.settings.name,
                "version": self.settings.version,
            },
        }

    async def _handle_shutdown(self, request: MCPRequest) -> dict[str, Any]:
        """Handle shutdown request."""
        self._running = False
        self._initialized = False
        logger.info("Server shutdown requested")
        return {}

    async def _handle_list_tools(self, request: MCPRequest) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self.settings.enable_tools:
            return {"tools": []}

        tools = self.tool_registry.to_mcp_format()
        return {"tools": tools}

    async def _handle_call_tool(self, request: MCPRequest) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self.settings.enable_tools:
            raise ValueError("Tools capability not enabled")

        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")

        logger.debug(f"Calling tool: {tool_name} with args: {arguments}")

        result = await tool.execute(arguments)
        return result.to_mcp_format()

    async def _handle_list_resources(self, request: MCPRequest) -> dict[str, Any]:
        """Handle resources/list request."""
        if not self.settings.enable_resources:
            return {"resources": []}

        resources = self.resource_registry.to_mcp_format()
        return {"resources": resources}

    async def _handle_read_resource(self, request: MCPRequest) -> dict[str, Any]:
        """Handle resources/read request."""
        if not self.settings.enable_resources:
            raise ValueError("Resources capability not enabled")

        uri = request.params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")

        resource = self.resource_registry.get(uri)
        if resource is None:
            raise ValueError(f"Resource not found: {uri}")

        logger.debug(f"Reading resource: {uri}")

        content = await resource.read()
        return {"contents": [content.to_mcp_format()]}

    async def _handle_list_prompts(self, request: MCPRequest) -> dict[str, Any]:
        """Handle prompts/list request."""
        # Prompts not implemented yet
        return {"prompts": []}

    async def _handle_get_prompt(self, request: MCPRequest) -> dict[str, Any]:
        """Handle prompts/get request."""
        raise ValueError("Prompt not found")

    async def run_stdio(self) -> None:
        """
        Run server using stdio transport.

        This reads JSON-RPC messages from stdin and writes responses to stdout.
        """
        logger.info("Starting MCP server with stdio transport")
        self._running = True

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        try:
            while self._running:
                try:
                    # Read content length header
                    header_line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=1.0,
                    )

                    if not header_line:
                        continue

                    header = header_line.decode().strip()
                    if not header.startswith("Content-Length:"):
                        continue

                    content_length = int(header.split(":")[1].strip())

                    # Read blank line
                    await reader.readline()

                    # Read content
                    content = await reader.read(content_length)
                    message = content.decode()

                    logger.debug(f"Received: {message[:200]}...")

                    # Handle message
                    response = await self.handle_message(message)

                    if response:
                        # Write response with header
                        response_bytes = response.encode()
                        output = f"Content-Length: {len(response_bytes)}\r\n\r\n".encode()
                        output += response_bytes

                        writer.write(output)
                        await writer.drain()

                        logger.debug(f"Sent: {response[:200]}...")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in message loop: {e}")
                    if not self._running:
                        break

        finally:
            writer.close()
            logger.info("MCP server stopped")

    async def run(self) -> None:
        """
        Run the MCP server with configured transport.

        This is the main entry point for starting the server.
        """
        if self.settings.transport == TransportType.STDIO:
            await self.run_stdio()
        elif self.settings.transport == TransportType.SSE:
            await self.run_sse()
        elif self.settings.transport == TransportType.WEBSOCKET:
            await self.run_websocket()
        else:
            raise ValueError(f"Unsupported transport: {self.settings.transport}")

    async def run_sse(self) -> None:
        """Run server using SSE transport."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import StreamingResponse
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
        except ImportError:
            raise ImportError(
                "FastAPI and uvicorn required for SSE transport. "
                "Install with: pip install ptpd-calibration[mcp]"
            )

        app = FastAPI(
            title=self.settings.name,
            version=self.settings.version,
            description=self.settings.description,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/sse")
        async def sse_endpoint(request: Request) -> StreamingResponse:
            async def event_generator():
                self._running = True
                while self._running:
                    # Wait for messages (simplified - real implementation would use queues)
                    await asyncio.sleep(0.1)
                    yield "data: ping\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        @app.post("/message")
        async def message_endpoint(request: Request) -> dict:
            body = await request.body()
            response = await self.handle_message(body.decode())
            if response:
                return json.loads(response)
            return {}

        logger.info(f"Starting MCP server with SSE transport on {self.settings.host}:{self.settings.port}")

        config = uvicorn.Config(
            app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_websocket(self) -> None:
        """Run server using WebSocket transport."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required for WebSocket transport. "
                "Install with: pip install ptpd-calibration[mcp]"
            )

        async def handle_connection(websocket: Any) -> None:
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            self._running = True

            try:
                async for message in websocket:
                    response = await self.handle_message(message)
                    if response:
                        await websocket.send(response)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                logger.info("WebSocket client disconnected")

        logger.info(
            f"Starting MCP server with WebSocket transport on "
            f"ws://{self.settings.host}:{self.settings.port}"
        )

        async with websockets.serve(
            handle_connection,
            self.settings.host,
            self.settings.port,
        ):
            await asyncio.Future()  # Run forever


def create_mcp_server(
    settings: Optional[MCPSettings] = None,
    database: Optional[Any] = None,
) -> MCPServer:
    """
    Create an MCP server instance.

    Args:
        settings: MCP settings. Uses environment defaults if not provided.
        database: Optional CalibrationDatabase for data access.

    Returns:
        Configured MCPServer instance.

    Example:
        ```python
        # Create with defaults
        server = create_mcp_server()

        # Create with custom settings
        from ptpd_calibration.mcp.config import MCPSettings
        settings = MCPSettings()
        settings.server.transport = TransportType.WEBSOCKET
        server = create_mcp_server(settings=settings)

        # Run the server
        import asyncio
        asyncio.run(server.run())
        ```
    """
    mcp_settings = settings or get_mcp_settings()
    return MCPServer(
        settings=mcp_settings.server,
        database=database,
    )


def main() -> None:
    """
    Main entry point for running the MCP server.

    This function is exposed as the 'ptpd-mcp' command.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="PTPD Calibration MCP Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "websocket"],
        default="stdio",
        help="Transport type to use",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (for sse/websocket)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (for sse/websocket)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Update settings from args
    settings = get_mcp_settings()
    settings.server.transport = TransportType(args.transport)
    settings.server.host = args.host
    settings.server.port = args.port
    settings.server.debug_mode = args.debug

    if args.debug:
        settings.server.log_level = "DEBUG"

    # Create and run server
    server = create_mcp_server(settings=settings)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
