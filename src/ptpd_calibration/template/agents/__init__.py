"""
Sub-Agent Coordination System

Provides a framework for building and coordinating AI agents with:
- Dynamic task routing
- Tool registration
- Memory management
- Planning and execution
- Parallel agent execution
"""

from ptpd_calibration.template.agents.base import (
    AgentBase,
    AgentContext,
    AgentResult,
    AgentState,
)
from ptpd_calibration.template.agents.coordinator import (
    AgentCoordinator,
    ExecutionPlan,
    TaskRouter,
)
from ptpd_calibration.template.agents.memory import (
    AgentMemory,
    MemoryEntry,
    MemoryType,
)
from ptpd_calibration.template.agents.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    tool,
)

__all__ = [
    # Base
    "AgentBase",
    "AgentContext",
    "AgentResult",
    "AgentState",
    # Coordinator
    "AgentCoordinator",
    "TaskRouter",
    "ExecutionPlan",
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "tool",
    # Memory
    "AgentMemory",
    "MemoryEntry",
    "MemoryType",
]
