"""
Agentic system for autonomous calibration assistance.

This module provides:
- CalibrationAgent: Main ReAct-style agent for calibration tasks
- Subagents: Specialized agents (Planner, SQE, Coder, Reviewer)
- Orchestrator: Multi-agent workflow coordination
- Communication: Inter-agent messaging
- Logging: Structured JSON logging for observability
"""

from ptpd_calibration.agents.agent import (
    AgentConfig,
    CalibrationAgent,
    ReasoningStep,
    create_agent,
)
from ptpd_calibration.agents.communication import (
    AgentMessage,
    ConversationContext,
    ConversationManager,
    MessageBus,
    MessageHandler,
    MessagePriority,
    MessageType,
    get_message_bus,
    start_message_bus,
    stop_message_bus,
)
from ptpd_calibration.agents.logging import (
    AgentLogger,
    EventType,
    LogContext,
    LogEntry,
    LogLevel,
    configure_agent_logging,
    get_agent_logger,
    timed_operation,
)
from ptpd_calibration.agents.memory import (
    AgentMemory,
    MemoryItem,
)
from ptpd_calibration.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorConfig,
    TaskStatus,
    Workflow,
    WorkflowStatus,
    WorkflowTask,
    orchestrate_development,
)
from ptpd_calibration.agents.planning import (
    Plan,
    Planner,
    PlanStatus,
    PlanStep,
)
from ptpd_calibration.agents.subagents import (
    BaseSubagent,
    CoderAgent,
    PlannerAgent,
    ReviewerAgent,
    SQEAgent,
    SubagentCapability,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
)
from ptpd_calibration.agents.subagents.base import (
    get_subagent_registry,
    register_subagent,
)
from ptpd_calibration.agents.tools import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_calibration_tools,
)
from ptpd_calibration.agents.utils import (
    extract_classes,
    extract_code_block,
    extract_functions,
    extract_imports,
    format_bullet_list,
    parse_json_response,
    sanitize_identifier,
    truncate_text,
)

__all__ = [
    # Core Agent
    "CalibrationAgent",
    "AgentConfig",
    "ReasoningStep",
    "create_agent",
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolCategory",
    "ToolParameter",
    "create_calibration_tools",
    # Memory
    "AgentMemory",
    "MemoryItem",
    # Planning
    "Plan",
    "PlanStep",
    "PlanStatus",
    "Planner",
    # Subagents
    "BaseSubagent",
    "SubagentConfig",
    "SubagentCapability",
    "SubagentStatus",
    "SubagentResult",
    "SubagentRegistry",
    "get_subagent_registry",
    "register_subagent",
    "PlannerAgent",
    "SQEAgent",
    "CoderAgent",
    "ReviewerAgent",
    # Orchestrator
    "OrchestratorAgent",
    "OrchestratorConfig",
    "Workflow",
    "WorkflowTask",
    "WorkflowStatus",
    "TaskStatus",
    "orchestrate_development",
    # Communication
    "MessageBus",
    "MessageHandler",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "ConversationContext",
    "ConversationManager",
    "get_message_bus",
    "start_message_bus",
    "stop_message_bus",
    # Logging
    "AgentLogger",
    "LogContext",
    "LogEntry",
    "LogLevel",
    "EventType",
    "get_agent_logger",
    "configure_agent_logging",
    "timed_operation",
    # Utils
    "extract_classes",
    "extract_code_block",
    "extract_functions",
    "extract_imports",
    "format_bullet_list",
    "parse_json_response",
    "sanitize_identifier",
    "truncate_text",
]
