"""
Agentic system for autonomous calibration assistance.

This module provides:
- Task routing for intelligent classification of incoming tasks
- ReAct-style agent for autonomous task completion
- Planning system for task decomposition
- Memory system for context persistence
- Tool registry for capability management
- Skill modules for domain-specific expertise
"""

from ptpd_calibration.agents.tools import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_calibration_tools,
    register_composite_tools,
)
from ptpd_calibration.agents.agent import (
    CalibrationAgent,
    AgentConfig,
    ReasoningStep,
    create_agent,
)
from ptpd_calibration.agents.memory import (
    AgentMemory,
    MemoryItem,
)
from ptpd_calibration.agents.planning import (
    Plan,
    PlanStep,
    PlanStatus,
    Planner,
)
from ptpd_calibration.agents.router import (
    TaskRouter,
    TaskComplexity,
    TaskCategory as RouteCategory,
    RoutingPattern,
    RoutingResult,
    PatternRegistry,
    create_router,
)
from ptpd_calibration.agents.skills import (
    Skill,
    SkillResult,
    SkillContext,
    SkillRegistry,
    CalibrationSkill,
    ChemistrySkill,
    QualitySkill,
    TroubleshootingSkill,
    create_default_skills,
)

__all__ = [
    # Tools
    "Tool",
    "ToolCategory",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "create_calibration_tools",
    "register_composite_tools",
    # Agent
    "CalibrationAgent",
    "AgentConfig",
    "ReasoningStep",
    "create_agent",
    # Memory
    "AgentMemory",
    "MemoryItem",
    # Planning
    "Plan",
    "PlanStep",
    "PlanStatus",
    "Planner",
    # Router
    "TaskRouter",
    "TaskComplexity",
    "RouteCategory",
    "RoutingPattern",
    "RoutingResult",
    "PatternRegistry",
    "create_router",
    # Skills
    "Skill",
    "SkillResult",
    "SkillContext",
    "SkillRegistry",
    "CalibrationSkill",
    "ChemistrySkill",
    "QualitySkill",
    "TroubleshootingSkill",
    "create_default_skills",
]
