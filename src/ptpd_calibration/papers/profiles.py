"""
Paper profiles database for platinum/palladium printing.

Contains profiles for common papers with recommended settings and characteristics.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4


class CoatingBehavior(str, Enum):
    """How paper behaves during coating."""

    ABSORBS_QUICKLY = "absorbs_quickly"
    ABSORBS_SLOWLY = "absorbs_slowly"
    POOLS = "pools"
    SPREADS_EVENLY = "spreads_evenly"


@dataclass
class PaperCharacteristics:
    """Physical characteristics of a paper."""

    # Surface properties
    surface: str = "smooth"  # smooth, slightly_textured, textured, rough
    weight_gsm: int = 300  # grams per square meter
    thickness_mm: float = 0.5

    # Sizing
    sizing: str = "internal"  # internal, external, unsized
    sizing_strength: str = "medium"  # none, light, medium, heavy

    # Color
    paper_white: str = "natural"  # bright_white, natural, cream, warm
    paper_white_hex: str = "#FAF8F5"

    # Print characteristics
    typical_dmax: float = 1.5
    typical_dmin: float = 0.08
    contrast_tendency: str = "neutral"  # low, neutral, high

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "surface": self.surface,
            "weight_gsm": self.weight_gsm,
            "thickness_mm": self.thickness_mm,
            "sizing": self.sizing,
            "sizing_strength": self.sizing_strength,
            "paper_white": self.paper_white,
            "paper_white_hex": self.paper_white_hex,
            "typical_dmax": self.typical_dmax,
            "typical_dmin": self.typical_dmin,
            "contrast_tendency": self.contrast_tendency,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperCharacteristics":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PaperProfile:
    """Complete profile for a paper type."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    manufacturer: str = ""

    # Characteristics
    characteristics: PaperCharacteristics = field(default_factory=PaperCharacteristics)

    # Coating recommendations
    absorbency: str = "medium"  # low, medium, high
    coating_behavior: CoatingBehavior = CoatingBehavior.SPREADS_EVENLY
    drops_per_square_inch: float = 0.465
    recommended_coating_method: str = "brush"

    # Exposure recommendations
    base_exposure_minutes: float = 10.0
    exposure_notes: str = ""

    # Chemistry recommendations
    recommended_pt_ratio: float = 0.0  # 0 = all Pd, 1 = all Pt
    recommended_na2_ratio: float = 0.25
    chemistry_notes: str = ""

    # Developer recommendations
    recommended_developer: str = "Potassium Oxalate"
    developer_temperature_f: float = 68.0
    development_time_minutes: float = 2.0

    # General notes
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # User-defined flag
    is_custom: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "manufacturer": self.manufacturer,
            "characteristics": self.characteristics.to_dict(),
            "absorbency": self.absorbency,
            "coating_behavior": self.coating_behavior.value,
            "drops_per_square_inch": self.drops_per_square_inch,
            "recommended_coating_method": self.recommended_coating_method,
            "base_exposure_minutes": self.base_exposure_minutes,
            "exposure_notes": self.exposure_notes,
            "recommended_pt_ratio": self.recommended_pt_ratio,
            "recommended_na2_ratio": self.recommended_na2_ratio,
            "chemistry_notes": self.chemistry_notes,
            "recommended_developer": self.recommended_developer,
            "developer_temperature_f": self.developer_temperature_f,
            "development_time_minutes": self.development_time_minutes,
            "notes": self.notes,
            "tags": self.tags,
            "is_custom": self.is_custom,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperProfile":
        """Create from dictionary."""
        chars_data = data.pop("characteristics", {})
        coating_behavior = data.pop("coating_behavior", "spreads_evenly")
        profile_id = data.pop("id", None)

        profile = cls(
            id=UUID(profile_id) if profile_id else uuid4(),
            characteristics=PaperCharacteristics.from_dict(chars_data),
            coating_behavior=CoatingBehavior(coating_behavior),
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )
        return profile


# Built-in paper profiles
BUILTIN_PAPERS = {
    "arches_platine": PaperProfile(
        name="Arches Platine",
        manufacturer="Arches",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=310,
            thickness_mm=0.56,
            sizing="internal",
            sizing_strength="medium",
            paper_white="natural",
            paper_white_hex="#FAF6EE",
            typical_dmax=1.6,
            typical_dmin=0.07,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.45,
        recommended_coating_method="brush",
        base_exposure_minutes=10.0,
        exposure_notes="Very consistent results. Good starting point for calibration.",
        recommended_pt_ratio=0.0,
        recommended_na2_ratio=0.25,
        chemistry_notes="Works well with all metal ratios.",
        notes="Industry standard for Pt/Pd. Excellent for beginners.",
        tags=["recommended", "consistent", "smooth"],
    ),
    "bergger_cot320": PaperProfile(
        name="Bergger COT 320",
        manufacturer="Bergger",
        characteristics=PaperCharacteristics(
            surface="slightly_textured",
            weight_gsm=320,
            thickness_mm=0.58,
            sizing="internal",
            sizing_strength="heavy",
            paper_white="cream",
            paper_white_hex="#F5F0E5",
            typical_dmax=1.55,
            typical_dmin=0.08,
            contrast_tendency="neutral",
        ),
        absorbency="low",
        coating_behavior=CoatingBehavior.ABSORBS_SLOWLY,
        drops_per_square_inch=0.40,
        recommended_coating_method="brush",
        base_exposure_minutes=12.0,
        exposure_notes="Slightly longer exposure than Platine due to heavier sizing.",
        recommended_pt_ratio=0.25,
        recommended_na2_ratio=0.20,
        chemistry_notes="Benefits from some platinum for deeper blacks.",
        notes="Popular alternative to Arches. Cream base adds warmth.",
        tags=["warm_tone", "heavy_sizing"],
    ),
    "hahnemuhle_platinum_rag": PaperProfile(
        name="Hahnemuhle Platinum Rag",
        manufacturer="Hahnemuhle",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=300,
            thickness_mm=0.52,
            sizing="internal",
            sizing_strength="medium",
            paper_white="bright_white",
            paper_white_hex="#FAFAFA",
            typical_dmax=1.65,
            typical_dmin=0.06,
            contrast_tendency="high",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.45,
        recommended_coating_method="brush",
        base_exposure_minutes=9.0,
        exposure_notes="Faster paper. Watch for overexposure in shadows.",
        recommended_pt_ratio=0.0,
        recommended_na2_ratio=0.30,
        chemistry_notes="Higher contrast tendency, reduce Na2 for lower contrast negs.",
        notes="Excellent Dmax. Good for high-key images.",
        tags=["bright_white", "high_dmax"],
    ),
    "revere_platinum": PaperProfile(
        name="Revere Platinum",
        manufacturer="Legion Paper",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=320,
            thickness_mm=0.55,
            sizing="internal",
            sizing_strength="medium",
            paper_white="natural",
            paper_white_hex="#F8F4EA",
            typical_dmax=1.5,
            typical_dmin=0.09,
            contrast_tendency="low",
        ),
        absorbency="high",
        coating_behavior=CoatingBehavior.ABSORBS_QUICKLY,
        drops_per_square_inch=0.50,
        recommended_coating_method="brush",
        base_exposure_minutes=11.0,
        exposure_notes="More absorbent, may need more chemistry.",
        recommended_pt_ratio=0.0,
        recommended_na2_ratio=0.25,
        chemistry_notes="Lower contrast, good for high contrast negatives.",
        notes="American-made alternative. Natural warmth.",
        tags=["american_made", "warm_tone"],
    ),
    "stonehenge": PaperProfile(
        name="Stonehenge",
        manufacturer="Legion Paper",
        characteristics=PaperCharacteristics(
            surface="slightly_textured",
            weight_gsm=245,
            thickness_mm=0.42,
            sizing="internal",
            sizing_strength="light",
            paper_white="warm",
            paper_white_hex="#F5EEE0",
            typical_dmax=1.4,
            typical_dmin=0.10,
            contrast_tendency="low",
        ),
        absorbency="high",
        coating_behavior=CoatingBehavior.ABSORBS_QUICKLY,
        drops_per_square_inch=0.55,
        recommended_coating_method="brush",
        base_exposure_minutes=8.0,
        exposure_notes="Lighter weight, faster exposure. May curl when wet.",
        recommended_pt_ratio=0.0,
        recommended_na2_ratio=0.20,
        chemistry_notes="Low contrast paper, use with normal negatives.",
        notes="Budget option. Good for testing and practice.",
        tags=["budget", "practice", "light_weight"],
    ),
    "weston_diploma_parchment": PaperProfile(
        name="Weston Diploma Parchment",
        manufacturer="Weston",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=280,
            thickness_mm=0.48,
            sizing="external",
            sizing_strength="medium",
            paper_white="cream",
            paper_white_hex="#F8F2E6",
            typical_dmax=1.45,
            typical_dmin=0.09,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.45,
        recommended_coating_method="brush",
        base_exposure_minutes=10.0,
        notes="Classic paper. Beautiful cream tone.",
        tags=["classic", "cream_base"],
    ),
}


# Cyanotype-specific paper profiles
CYANOTYPE_PAPERS = {
    "arches_platine_cyanotype": PaperProfile(
        name="Arches Platine (Cyanotype)",
        manufacturer="Arches",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=310,
            thickness_mm=0.56,
            sizing="internal",
            sizing_strength="medium",
            paper_white="natural",
            paper_white_hex="#FAF6EE",
            typical_dmax=1.9,  # Cyanotypes have good Dmax
            typical_dmin=0.12,
            contrast_tendency="high",  # Cyanotype is inherently high contrast
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.30,  # Less sensitizer for cyanotype
        recommended_coating_method="brush",
        base_exposure_minutes=15.0,  # Sunlight exposure
        exposure_notes="15-20 min in direct sunlight. Look for bronze-olive color shift.",
        recommended_developer="Water",
        development_time_minutes=10.0,  # Wash time
        notes="Excellent for cyanotype with consistent results.",
        tags=["cyanotype", "recommended", "smooth"],
    ),
    "stonehenge_cyanotype": PaperProfile(
        name="Stonehenge (Cyanotype)",
        manufacturer="Legion Paper",
        characteristics=PaperCharacteristics(
            surface="slightly_textured",
            weight_gsm=245,
            thickness_mm=0.42,
            sizing="internal",
            sizing_strength="light",
            paper_white="warm",
            paper_white_hex="#F5EEE0",
            typical_dmax=1.8,
            typical_dmin=0.15,
            contrast_tendency="high",
        ),
        absorbency="high",
        coating_behavior=CoatingBehavior.ABSORBS_QUICKLY,
        drops_per_square_inch=0.35,
        recommended_coating_method="brush",
        base_exposure_minutes=12.0,
        exposure_notes="High absorbency works well with cyanotype. May need less solution.",
        recommended_developer="Water",
        development_time_minutes=8.0,
        notes="Budget-friendly option for cyanotype. Good for practice.",
        tags=["cyanotype", "budget", "practice"],
    ),
    "fabriano_artistico_cyanotype": PaperProfile(
        name="Fabriano Artistico (Cyanotype)",
        manufacturer="Fabriano",
        characteristics=PaperCharacteristics(
            surface="textured",
            weight_gsm=300,
            thickness_mm=0.55,
            sizing="internal",
            sizing_strength="medium",
            paper_white="natural",
            paper_white_hex="#FAF8F2",
            typical_dmax=1.85,
            typical_dmin=0.13,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.32,
        recommended_coating_method="brush",
        base_exposure_minutes=18.0,
        exposure_notes="Cold press texture shows in final print. Hot press for smoother look.",
        recommended_developer="Water",
        development_time_minutes=10.0,
        notes="Beautiful texture for artistic cyanotypes.",
        tags=["cyanotype", "textured", "artistic"],
    ),
    "rives_bfk_cyanotype": PaperProfile(
        name="Rives BFK (Cyanotype)",
        manufacturer="Rives",
        characteristics=PaperCharacteristics(
            surface="smooth",
            weight_gsm=280,
            thickness_mm=0.50,
            sizing="internal",
            sizing_strength="medium",
            paper_white="cream",
            paper_white_hex="#F8F4E8",
            typical_dmax=1.75,
            typical_dmin=0.14,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.30,
        recommended_coating_method="brush",
        base_exposure_minutes=16.0,
        exposure_notes="Classic printmaking paper. Excellent for cyanotype.",
        recommended_developer="Water",
        development_time_minutes=10.0,
        notes="Printmaking standard. Cream base adds warmth to blues.",
        tags=["cyanotype", "printmaking", "classic"],
    ),
    "cotton_fabric_cyanotype": PaperProfile(
        name="Cotton Muslin (Cyanotype)",
        manufacturer="Various",
        characteristics=PaperCharacteristics(
            surface="textured",
            weight_gsm=150,
            thickness_mm=0.30,
            sizing="unsized",
            sizing_strength="none",
            paper_white="bright_white",
            paper_white_hex="#FFFFFF",
            typical_dmax=1.7,
            typical_dmin=0.18,
            contrast_tendency="high",
        ),
        absorbency="high",
        coating_behavior=CoatingBehavior.ABSORBS_QUICKLY,
        drops_per_square_inch=0.50,  # Fabric needs more
        recommended_coating_method="brush",
        base_exposure_minutes=20.0,
        exposure_notes="Pre-wash fabric to remove sizing. Expose until bronze color.",
        recommended_developer="Water",
        development_time_minutes=15.0,
        notes="For wearable art and home decor. Pre-wash and iron before coating.",
        tags=["cyanotype", "fabric", "wearable"],
    ),
}


# Silver Gelatin paper profiles
SILVER_GELATIN_PAPERS = {
    "ilford_mgiv_rc": PaperProfile(
        name="Ilford Multigrade IV RC",
        manufacturer="Ilford",
        characteristics=PaperCharacteristics(
            surface="pearl",
            weight_gsm=190,
            thickness_mm=0.28,
            sizing="resin_coated",
            sizing_strength="none",
            paper_white="bright_white",
            paper_white_hex="#FAFAFA",
            typical_dmax=2.0,
            typical_dmin=0.05,
            contrast_tendency="variable",
        ),
        absorbency="low",
        coating_behavior=CoatingBehavior.ABSORBS_SLOWLY,
        drops_per_square_inch=0.0,  # Pre-sensitized
        recommended_coating_method="none",
        base_exposure_minutes=0.5,  # Seconds to minutes under enlarger
        exposure_notes="Variable contrast. Use multigrade filters for contrast control.",
        recommended_developer="Ilford Multigrade Developer",
        developer_temperature_f=68.0,
        development_time_minutes=1.5,
        notes="Industry standard RC paper. Fast processing, excellent results.",
        tags=["silver_gelatin", "rc", "variable_contrast", "recommended"],
    ),
    "ilford_mgiv_fb": PaperProfile(
        name="Ilford Multigrade IV FB",
        manufacturer="Ilford",
        characteristics=PaperCharacteristics(
            surface="glossy",
            weight_gsm=255,
            thickness_mm=0.35,
            sizing="baryta",
            sizing_strength="medium",
            paper_white="bright_white",
            paper_white_hex="#FAFAFA",
            typical_dmax=2.1,
            typical_dmin=0.04,
            contrast_tendency="variable",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.ABSORBS_SLOWLY,
        drops_per_square_inch=0.0,
        recommended_coating_method="none",
        base_exposure_minutes=0.5,
        exposure_notes="Variable contrast fiber base. Archival quality.",
        recommended_developer="Dektol",
        developer_temperature_f=68.0,
        development_time_minutes=2.0,
        notes="Archival fiber base. Longer wash required. Beautiful tones.",
        tags=["silver_gelatin", "fb", "variable_contrast", "archival"],
    ),
    "ilford_mgfb_warmtone": PaperProfile(
        name="Ilford MGFB Warmtone",
        manufacturer="Ilford",
        characteristics=PaperCharacteristics(
            surface="semi_glossy",
            weight_gsm=255,
            thickness_mm=0.35,
            sizing="baryta",
            sizing_strength="medium",
            paper_white="cream",
            paper_white_hex="#F8F4E8",
            typical_dmax=2.0,
            typical_dmin=0.05,
            contrast_tendency="variable",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.ABSORBS_SLOWLY,
        drops_per_square_inch=0.0,
        recommended_coating_method="none",
        base_exposure_minutes=0.5,
        exposure_notes="Warm image tone. Can be further warmed with selenium toning.",
        recommended_developer="Selectol",
        developer_temperature_f=68.0,
        development_time_minutes=2.5,
        notes="Beautiful warm brown-black tones. Excellent for portraits.",
        tags=["silver_gelatin", "fb", "warm_tone", "portrait"],
    ),
    "foma_fomabrom_fb": PaperProfile(
        name="Foma Fomabrom FB",
        manufacturer="Foma",
        characteristics=PaperCharacteristics(
            surface="glossy",
            weight_gsm=240,
            thickness_mm=0.32,
            sizing="baryta",
            sizing_strength="medium",
            paper_white="natural",
            paper_white_hex="#FAF6EE",
            typical_dmax=2.0,
            typical_dmin=0.05,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.0,
        recommended_coating_method="none",
        base_exposure_minutes=0.5,
        exposure_notes="Graded paper available in grades 0-5. Pick grade for negative.",
        recommended_developer="Dektol",
        developer_temperature_f=68.0,
        development_time_minutes=2.0,
        notes="Budget-friendly Czech paper. Good value, excellent quality.",
        tags=["silver_gelatin", "fb", "graded", "budget"],
    ),
    "bergger_prestige_cb": PaperProfile(
        name="Bergger Prestige CB",
        manufacturer="Bergger",
        characteristics=PaperCharacteristics(
            surface="semi_glossy",
            weight_gsm=300,
            thickness_mm=0.42,
            sizing="baryta",
            sizing_strength="heavy",
            paper_white="natural",
            paper_white_hex="#FAF4E6",
            typical_dmax=2.1,
            typical_dmin=0.04,
            contrast_tendency="neutral",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.ABSORBS_SLOWLY,
        drops_per_square_inch=0.0,
        recommended_coating_method="none",
        base_exposure_minutes=0.5,
        exposure_notes="Neutral to warm tone. Responds well to selenium toning.",
        recommended_developer="Dektol",
        developer_temperature_f=68.0,
        development_time_minutes=2.0,
        notes="French premium paper. Rich blacks, beautiful highlight separation.",
        tags=["silver_gelatin", "fb", "premium", "archival"],
    ),
    "oriental_seagull_fb": PaperProfile(
        name="Oriental Seagull FB",
        manufacturer="Oriental",
        characteristics=PaperCharacteristics(
            surface="glossy",
            weight_gsm=250,
            thickness_mm=0.35,
            sizing="baryta",
            sizing_strength="medium",
            paper_white="bright_white",
            paper_white_hex="#FAFAFA",
            typical_dmax=2.15,
            typical_dmin=0.03,
            contrast_tendency="variable",
        ),
        absorbency="medium",
        coating_behavior=CoatingBehavior.SPREADS_EVENLY,
        drops_per_square_inch=0.0,
        recommended_coating_method="none",
        base_exposure_minutes=0.5,
        exposure_notes="Cold tone paper. Very high Dmax possible.",
        recommended_developer="Dektol",
        developer_temperature_f=68.0,
        development_time_minutes=2.0,
        notes="Favored for maximum black. Cold/neutral tone.",
        tags=["silver_gelatin", "fb", "cold_tone", "high_dmax"],
    ),
}


# Combined papers dictionary for all processes
ALL_PAPERS = {
    **BUILTIN_PAPERS,
    **CYANOTYPE_PAPERS,
    **SILVER_GELATIN_PAPERS,
}


class PaperDatabase:
    """Database for managing paper profiles.

    Combines built-in profiles with user-defined custom papers.
    """

    def __init__(self, custom_papers_file: Path | None = None):
        """Initialize paper database.

        Args:
            custom_papers_file: Path to custom papers JSON file.
                               Defaults to ~/.ptpd/papers.json
        """
        self.custom_papers_file = custom_papers_file or Path.home() / ".ptpd" / "papers.json"
        self._custom_papers: dict[str, PaperProfile] = {}
        self._load_custom_papers()

    def get_paper(self, name: str) -> PaperProfile | None:
        """Get a paper profile by name.

        Args:
            name: Paper name or key

        Returns:
            PaperProfile or None if not found
        """
        # Check built-in first
        key = name.lower().replace(" ", "_")
        if key in BUILTIN_PAPERS:
            return BUILTIN_PAPERS[key]

        # Check by full name
        for paper in BUILTIN_PAPERS.values():
            if paper.name.lower() == name.lower():
                return paper

        # Check custom papers
        if key in self._custom_papers:
            return self._custom_papers[key]

        for paper in self._custom_papers.values():
            if paper.name.lower() == name.lower():
                return paper

        return None

    def list_papers(self) -> list[PaperProfile]:
        """Get all available papers.

        Returns:
            List of all paper profiles (built-in + custom)
        """
        papers = list(BUILTIN_PAPERS.values())
        papers.extend(self._custom_papers.values())
        return sorted(papers, key=lambda p: p.name)

    def list_paper_names(self) -> list[str]:
        """Get names of all available papers.

        Returns:
            List of paper names
        """
        return [p.name for p in self.list_papers()]

    # Compatibility helpers for earlier UI code
    def get_all(self) -> list[PaperProfile]:
        """Alias for list_papers."""
        return self.list_papers()

    def get_by_name(self, name: str) -> PaperProfile | None:
        """Alias for get_paper."""
        return self.get_paper(name)

    def add_profile(self, profile: PaperProfile) -> None:
        """Alias for add_custom_paper."""
        self.add_custom_paper(profile)

    def add_custom_paper(self, profile: PaperProfile) -> None:
        """Add a custom paper profile.

        Args:
            profile: Paper profile to add
        """
        profile.is_custom = True
        key = profile.name.lower().replace(" ", "_")
        self._custom_papers[key] = profile
        self._save_custom_papers()

    def remove_custom_paper(self, name: str) -> bool:
        """Remove a custom paper profile.

        Args:
            name: Paper name to remove

        Returns:
            True if removed, False if not found
        """
        key = name.lower().replace(" ", "_")
        if key in self._custom_papers:
            del self._custom_papers[key]
            self._save_custom_papers()
            return True
        return False

    def search_papers(
        self,
        surface: str | None = None,
        absorbency: str | None = None,
        manufacturer: str | None = None,
        tags: list[str] | None = None,
    ) -> list[PaperProfile]:
        """Search papers by criteria.

        Args:
            surface: Filter by surface type
            absorbency: Filter by absorbency
            manufacturer: Filter by manufacturer
            tags: Filter by tags (any match)

        Returns:
            List of matching papers
        """
        results = []

        for paper in self.list_papers():
            if surface and paper.characteristics.surface != surface:
                continue
            if absorbency and paper.absorbency != absorbency:
                continue
            if manufacturer and paper.manufacturer.lower() != manufacturer.lower():
                continue
            if tags and not any(t in paper.tags for t in tags):
                continue

            results.append(paper)

        return results

    def get_manufacturers(self) -> list[str]:
        """Get list of all manufacturers.

        Returns:
            List of manufacturer names
        """
        manufacturers = set()
        for paper in self.list_papers():
            if paper.manufacturer:
                manufacturers.add(paper.manufacturer)
        return sorted(manufacturers)

    def _load_custom_papers(self) -> None:
        """Load custom papers from file."""
        if not self.custom_papers_file.exists():
            return

        try:
            with open(self.custom_papers_file) as f:
                data = json.load(f)

            for key, paper_data in data.items():
                self._custom_papers[key] = PaperProfile.from_dict(paper_data)
        except Exception as e:
            import logging

            logging.warning(f"Failed to load custom papers: {e}")
            pass

    def _save_custom_papers(self) -> None:
        """Save custom papers to file."""
        self.custom_papers_file.parent.mkdir(parents=True, exist_ok=True)

        data = {key: paper.to_dict() for key, paper in self._custom_papers.items()}

        with open(self.custom_papers_file, "w") as f:
            json.dump(data, f, indent=2)
