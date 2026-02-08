"""
Domain-specific types and enumerations for Pt/Pd printing.
"""

from enum import Enum


class ChemistryType(str, Enum):
    """Chemistry types for alternative photographic printing processes."""

    # Platinum/Palladium processes
    PURE_PLATINUM = "pure_platinum"
    PURE_PALLADIUM = "pure_palladium"
    PLATINUM_PALLADIUM = "platinum_palladium"
    ZIATYPE = "ziatype"
    NA2_PROCESS = "na2_process"

    # Iron-based processes
    CYANOTYPE = "cyanotype"
    CYANOTYPE_NEW = "cyanotype_new"  # Modern "new" cyanotype formula
    VAN_DYKE = "van_dyke"
    KALLITYPE = "kallitype"

    # Silver-based processes
    SILVER_GELATIN = "silver_gelatin"
    SILVER_GELATIN_RC = "silver_gelatin_rc"  # Resin-coated paper
    SILVER_GELATIN_FB = "silver_gelatin_fb"  # Fiber-based paper
    ARGYROTYPE = "argyrotype"
    SALT_PRINT = "salt_print"

    # Other alternative processes
    GUM_BICHROMATE = "gum_bichromate"
    CARBON_TRANSFER = "carbon_transfer"
    ALBUMEN = "albumen"


class ContrastAgent(str, Enum):
    """Contrast control agents used in Pt/Pd printing."""

    NONE = "none"
    NA2 = "na2"  # Sodium chloroplatinate
    POTASSIUM_CHLORATE = "potassium_chlorate"
    HYDROGEN_PEROXIDE = "hydrogen_peroxide"
    DICHROMATE = "dichromate"


class DeveloperType(str, Enum):
    """Developer types for alternative photographic processes."""

    # Platinum/Palladium developers
    POTASSIUM_OXALATE = "potassium_oxalate"
    AMMONIUM_CITRATE = "ammonium_citrate"
    SODIUM_CITRATE = "sodium_citrate"
    EDTA = "edta"

    # Silver gelatin paper developers
    DEKTOL = "dektol"  # Kodak D-72 type
    D_72 = "d_72"  # Standard paper developer
    D_76 = "d_76"  # Fine grain film developer (can be used for paper)
    SELECTOL = "selectol"  # Warm tone developer
    SELECTOL_SOFT = "selectol_soft"  # Low contrast
    ILFORD_MULTIGRADE = "ilford_multigrade"
    ILFORD_PQ_UNIVERSAL = "ilford_pq_universal"
    SPRINT_QUICKSILVER = "sprint_quicksilver"
    ETHOL_LPD = "ethol_lpd"  # Variable contrast developer
    ANSCO_130 = "ansco_130"  # Warm tone
    AMIDOL = "amidol"  # Cold tone, archival
    PYROGALLOL = "pyrogallol"  # Historic developer

    # Cyanotype developers (water-based)
    WATER = "water"  # Standard cyanotype development
    HYDROGEN_PEROXIDE_DEV = "hydrogen_peroxide_dev"  # Intensification
    CITRIC_ACID = "citric_acid"  # Acid bath for better blues

    # Van Dyke / Kallitype developers
    BORAX = "borax"
    SODIUM_CARBONATE = "sodium_carbonate"
    ROCHELLE_SALT = "rochelle_salt"


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


class FixerType(str, Enum):
    """Fixer types for silver-based processes."""

    SODIUM_THIOSULFATE = "sodium_thiosulfate"  # Standard fixer
    AMMONIUM_THIOSULFATE = "ammonium_thiosulfate"  # Rapid fixer
    KODAK_RAPID_FIX = "kodak_rapid_fix"
    ILFORD_RAPID_FIX = "ilford_rapid_fix"
    TF_4 = "tf_4"  # Alkaline archival fixer
    TF_5 = "tf_5"  # Neutral archival fixer


class PaperGrade(str, Enum):
    """Contrast grades for silver gelatin papers."""

    GRADE_0 = "grade_0"  # Very low contrast
    GRADE_1 = "grade_1"  # Low contrast
    GRADE_2 = "grade_2"  # Normal contrast
    GRADE_3 = "grade_3"  # High contrast
    GRADE_4 = "grade_4"  # Very high contrast
    GRADE_5 = "grade_5"  # Extremely high contrast
    VARIABLE = "variable"  # Multigrade/variable contrast paper


class PaperSurface(str, Enum):
    """Paper surface types for silver gelatin papers."""

    GLOSSY = "glossy"
    SEMI_GLOSSY = "semi_glossy"
    PEARL = "pearl"
    LUSTRE = "lustre"
    SATIN = "satin"
    MATTE = "matte"
    SEMI_MATTE = "semi_matte"


class PaperBase(str, Enum):
    """Paper base types for silver gelatin papers."""

    FIBER = "fiber"  # FB - Fiber-based (archival)
    RESIN_COATED = "resin_coated"  # RC - Resin-coated (faster processing)
    BARYTA = "baryta"  # Traditional fiber with barium sulfate coating


class SafelightFilter(str, Enum):
    """Safelight filter colors for darkroom work."""

    OC = "oc"  # Light amber (general purpose B&W paper)
    OA = "oa"  # Greenish yellow (orthochromatic)
    RED_1 = "red_1"  # Red (panchromatic film handling)
    RED_1A = "red_1a"  # Dark red
    AMBER_13 = "amber_13"  # Amber (slow orthochromatic)


class CyanotypeFormula(str, Enum):
    """Cyanotype formula variants."""

    CLASSIC = "classic"  # Traditional Sir John Herschel formula
    NEW = "new"  # Mike Ware's "New Cyanotype" formula
    WARE = "ware"  # Mike Ware variant
    REX = "rex"  # Rex cyanotype formula


class ToneColor(str, Enum):
    """Print tone color characteristics."""

    NEUTRAL_BLACK = "neutral_black"
    WARM_BLACK = "warm_black"
    COOL_BLACK = "cool_black"
    BROWN = "brown"
    SEPIA = "sepia"
    BLUE = "blue"  # Cyanotype
    PRUSSIAN_BLUE = "prussian_blue"  # Deep cyanotype
    PURPLE_BROWN = "purple_brown"  # Van Dyke
    OLIVE = "olive"
