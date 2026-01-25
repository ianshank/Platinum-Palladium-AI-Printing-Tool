"""
Skill modules for domain-specific agent capabilities.

Skills encapsulate domain expertise and provide focused functionality
that the agent can use to accomplish specific tasks. Each skill:
- Has one primary responsibility
- Uses existing module functions (no duplication)
- Returns structured results
- Can be tested independently
"""

from ptpd_calibration.agents.skills.base import (
    Skill,
    SkillResult,
    SkillContext,
    SkillRegistry,
)
from ptpd_calibration.agents.skills.calibration import CalibrationSkill
from ptpd_calibration.agents.skills.chemistry import ChemistrySkill
from ptpd_calibration.agents.skills.quality import QualitySkill
from ptpd_calibration.agents.skills.troubleshooting import TroubleshootingSkill


def create_default_skills() -> SkillRegistry:
    """
    Create a skill registry with all default skills registered.

    Returns:
        SkillRegistry with all standard skills.
    """
    registry = SkillRegistry()
    registry.register(CalibrationSkill())
    registry.register(ChemistrySkill())
    registry.register(QualitySkill())
    registry.register(TroubleshootingSkill())
    return registry


__all__ = [
    # Base classes
    "Skill",
    "SkillResult",
    "SkillContext",
    "SkillRegistry",
    # Skills
    "CalibrationSkill",
    "ChemistrySkill",
    "QualitySkill",
    "TroubleshootingSkill",
    # Factory
    "create_default_skills",
]
