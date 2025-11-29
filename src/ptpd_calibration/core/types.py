"""
Domain-specific types and enumerations for Pt/Pd printing.
"""

from enum import Enum


class ChemistryType(str, Enum):
    """Metal chemistry types for Pt/Pd printing."""

    PURE_PLATINUM = "pure_platinum"
    PURE_PALLADIUM = "pure_palladium"
    PLATINUM_PALLADIUM = "platinum_palladium"
    ZIATYPE = "ziatype"
    NA2_PROCESS = "na2_process"


class ContrastAgent(str, Enum):
    """Contrast control agents used in Pt/Pd printing."""

    NONE = "none"
    NA2 = "na2"  # Sodium chloroplatinate
    POTASSIUM_CHLORATE = "potassium_chlorate"
    HYDROGEN_PEROXIDE = "hydrogen_peroxide"
    DICHROMATE = "dichromate"


class DeveloperType(str, Enum):
    """Developer types for Pt/Pd printing."""

    POTASSIUM_OXALATE = "potassium_oxalate"
    AMMONIUM_CITRATE = "ammonium_citrate"
    SODIUM_CITRATE = "sodium_citrate"
    EDTA = "edta"


class PaperSizing(str, Enum):
    """Paper sizing types."""

    NONE = "none"
    INTERNAL = "internal"
    GELATIN = "gelatin"
    STARCH = "starch"
    ARROWROOT = "arrowroot"


class CurveType(str, Enum):
    """Types of linearization curves."""

    LINEAR = "linear"
    PAPER_WHITE = "paper_white"
    AESTHETIC = "aesthetic"
    CUSTOM = "custom"


class MeasurementUnit(str, Enum):
    """Units for density measurement."""

    VISUAL_DENSITY = "visual_density"
    STATUS_A = "status_a"
    STATUS_M = "status_m"
    LAB = "lab"
    RGB = "rgb"
