"""
MCP (Model Context Protocol) server for LM Studio integration.

This module provides an MCP server that exposes calibration tools and resources
to local LLM instances running in LM Studio.
"""

from ptpd_calibration.mcp.config import LMStudioSettings, MCPServerSettings
from ptpd_calibration.mcp.lm_studio import LMStudioClient, create_lm_studio_client
from ptpd_calibration.mcp.resources import (
    MCPResource,
    ResourceRegistry,
    create_calibration_resources,
)
from ptpd_calibration.mcp.server import MCPServer, create_mcp_server
from ptpd_calibration.mcp.tools import MCPTool, MCPToolRegistry, create_mcp_tools

__all__ = [
    # Configuration
    "LMStudioSettings",
    "MCPServerSettings",
    # Client
    "LMStudioClient",
    "create_lm_studio_client",
    # Resources
    "MCPResource",
    "ResourceRegistry",
    "create_calibration_resources",
    # Tools
    "MCPTool",
    "MCPToolRegistry",
    "create_mcp_tools",
    # Server
    "MCPServer",
    "create_mcp_server",
]
