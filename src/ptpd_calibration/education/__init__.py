"""
Educational components for Pt/Pd printing.

This module provides comprehensive educational resources including:
- Interactive tutorials for all skill levels
- Comprehensive glossary of Pt/Pd terminology
- Contextual tips and best practices
"""

from ptpd_calibration.education.glossary import (
    Glossary,
    GlossaryTerm,
    TermCategory,
)
from ptpd_calibration.education.tips import (
    Tip,
    TipCategory,
    TipDifficulty,
    TipsManager,
)
from ptpd_calibration.education.tutorials import (
    ActionType,
    Tutorial,
    TutorialDifficulty,
    TutorialManager,
    TutorialStep,
    UserProgress,
)

__all__ = [
    # Tutorial components
    "Tutorial",
    "TutorialStep",
    "TutorialDifficulty",
    "TutorialManager",
    "UserProgress",
    "ActionType",
    # Glossary components
    "Glossary",
    "GlossaryTerm",
    "TermCategory",
    # Tips components
    "Tip",
    "TipCategory",
    "TipDifficulty",
    "TipsManager",
]
