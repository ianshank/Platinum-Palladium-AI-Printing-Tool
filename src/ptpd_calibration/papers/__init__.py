"""
Paper profiles database for storing and retrieving paper-specific settings.

Common papers for platinum/palladium printing with recommended settings.
"""

from ptpd_calibration.papers.profiles import (
    CoatingBehavior,
    PaperCharacteristics,
    PaperDatabase,
    PaperProfile,
)

__all__ = [
    "PaperProfile",
    "PaperDatabase",
    "PaperCharacteristics",
    "CoatingBehavior",
]
