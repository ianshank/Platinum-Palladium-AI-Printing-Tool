"""
Subagent system for specialized task delegation.

Provides specialized agents: Planner, SQE, Coder, Reviewer.
"""

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
)
from ptpd_calibration.agents.subagents.coder import CoderAgent
from ptpd_calibration.agents.subagents.planner import PlannerAgent
from ptpd_calibration.agents.subagents.reviewer import ReviewerAgent
from ptpd_calibration.agents.subagents.sqa import SQEAgent

__all__ = [
    # Base
    "BaseSubagent",
    "SubagentConfig",
    "SubagentCapability",
    "SubagentStatus",
    "SubagentResult",
    "SubagentRegistry",
    # Implementations
    "PlannerAgent",
    "SQEAgent",
    "CoderAgent",
    "ReviewerAgent",
]
