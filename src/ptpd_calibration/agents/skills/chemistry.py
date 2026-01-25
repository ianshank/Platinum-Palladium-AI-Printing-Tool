"""
Chemistry skill for coating solution calculations.

This skill wraps the existing chemistry calculators and provides
a focused interface for chemistry-related tasks.
"""

from typing import Any, Optional

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ptpd_calibration.agents.skills.base import (
    Skill,
    SkillCategory,
    SkillContext,
    SkillResult,
    SkillSettings,
)


class ChemistrySkillSettings(SkillSettings):
    """Settings for the chemistry skill."""

    model_config = SettingsConfigDict(env_prefix="PTPD_SKILL_CHEMISTRY_")

    # Default chemistry parameters
    default_metal_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Default platinum ratio (0=all Pd, 1=all Pt)"
    )
    default_contrast_drops: int = Field(
        default=5, ge=0, le=15, description="Default Na2 contrast agent drops"
    )
    default_coating_factor: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Default coating factor"
    )

    # Paper absorbency factors
    low_absorbency_factor: float = Field(
        default=0.8, description="Factor for low absorbency papers"
    )
    high_absorbency_factor: float = Field(
        default=1.3, description="Factor for high absorbency papers"
    )

    # Cost estimation
    platinum_cost_per_ml: float = Field(
        default=2.50, description="Cost per ml of platinum solution"
    )
    palladium_cost_per_ml: float = Field(
        default=1.50, description="Cost per ml of palladium solution"
    )


class ChemistrySkill(Skill[ChemistrySkillSettings]):
    """
    Skill for chemistry-related tasks.

    Provides capabilities for:
    - Coating solution calculations (Pt/Pd, Cyanotype, Silver Gelatin)
    - Metal ratio recommendations for desired tones
    - Contrast agent calculations
    - Cost estimation
    """

    @property
    def name(self) -> str:
        return "chemistry"

    @property
    def description(self) -> str:
        return (
            "Handles coating solution calculations, metal ratio optimization, "
            "and chemistry recommendations for alternative photographic processes"
        )

    @property
    def category(self) -> SkillCategory:
        return SkillCategory.CHEMISTRY

    def _default_settings(self) -> ChemistrySkillSettings:
        return ChemistrySkillSettings()

    def get_capabilities(self) -> list[str]:
        return [
            "Calculate coating solution drops for Pt/Pd printing",
            "Recommend metal ratios for desired tones (warm/neutral/cool)",
            "Calculate contrast agent amounts",
            "Estimate chemistry costs",
            "Support multiple processes (Pt/Pd, Cyanotype, Silver Gelatin)",
            "Adjust calculations for paper absorbency",
        ]

    def can_handle(self, task: str, context: Optional[SkillContext] = None) -> float:
        """Determine if this skill can handle the chemistry task."""
        task_lower = task.lower()

        # High confidence keywords
        high_confidence_keywords = [
            "chemistry",
            "coating",
            "drops",
            "solution",
            "formula",
            "ferric oxalate",
            "na2",
            "contrast agent",
            "platinum ratio",
            "palladium ratio",
            "metal ratio",
        ]

        # Medium confidence keywords
        medium_confidence_keywords = [
            "calculate",
            "how many",
            "mix",
            "prepare",
            "sensitizer",
            "warm tone",
            "cool tone",
            "cost",
        ]

        # Check for high confidence matches
        if any(kw in task_lower for kw in high_confidence_keywords):
            return 0.9

        # Check for medium confidence matches
        if any(kw in task_lower for kw in medium_confidence_keywords):
            return 0.5

        return 0.0

    def execute(
        self,
        task: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute chemistry task."""
        task_lower = task.lower()

        # Determine the specific operation
        if "calculate" in task_lower or "drops" in task_lower or "coating" in task_lower:
            return self._calculate_coating(context, **kwargs)
        elif "ratio" in task_lower or "warm" in task_lower or "cool" in task_lower:
            return self._recommend_ratio(context, **kwargs)
        elif "contrast" in task_lower or "na2" in task_lower:
            return self._calculate_contrast(context, **kwargs)
        elif "cost" in task_lower:
            return self._estimate_cost(context, **kwargs)
        else:
            # Default to coating calculation
            return self._calculate_coating(context, **kwargs)

    def _calculate_coating(
        self,
        context: Optional[SkillContext],
        print_width: float = 8.0,
        print_height: float = 10.0,
        metal_ratio: Optional[float] = None,
        paper_absorbency: str = "normal",
        **kwargs: Any,
    ) -> SkillResult:
        """Calculate coating solution for given print size."""
        # Get metal ratio from context or use default
        if metal_ratio is None:
            metal_ratio = (
                context.metal_ratio
                if context and context.metal_ratio is not None
                else self.settings.default_metal_ratio
            )

        # Calculate coating area
        area_sq_inches = print_width * print_height

        # Base drops per square inch (Bostick & Sullivan formula approximation)
        base_drops_per_sq_inch = 0.25

        # Apply absorbency factor
        absorbency_factors = {
            "low": self.settings.low_absorbency_factor,
            "normal": 1.0,
            "high": self.settings.high_absorbency_factor,
        }
        absorbency_factor = absorbency_factors.get(paper_absorbency, 1.0)

        total_drops = area_sq_inches * base_drops_per_sq_inch * absorbency_factor

        # Calculate individual components
        fo_drops = round(total_drops)  # Ferric Oxalate
        pt_drops = round(total_drops * metal_ratio)  # Platinum
        pd_drops = round(total_drops * (1 - metal_ratio))  # Palladium
        na2_drops = self.settings.default_contrast_drops  # Na2 contrast

        # Total sensitizer
        total_sensitizer = fo_drops + pt_drops + pd_drops

        # Estimate ml
        drops_per_ml = 20  # Approximate
        total_ml = total_sensitizer / drops_per_ml

        return SkillResult.success_result(
            data={
                "print_size": f"{print_width} x {print_height} inches",
                "area_sq_inches": round(area_sq_inches, 1),
                "components": {
                    "ferric_oxalate": fo_drops,
                    "platinum": pt_drops,
                    "palladium": pd_drops,
                    "na2_contrast": na2_drops,
                },
                "total_drops": total_sensitizer + na2_drops,
                "total_ml": round(total_ml, 2),
                "metal_ratio": metal_ratio,
                "metal_description": self._describe_ratio(metal_ratio),
                "paper_absorbency": paper_absorbency,
            },
            message=(
                f"Coating for {print_width}x{print_height}\": "
                f"{fo_drops} FO + {pt_drops} Pt + {pd_drops} Pd + {na2_drops} Na2"
            ),
            confidence=0.95,
            next_actions=[
                "Mix solution in amber bottle",
                "Apply coating with brush or glass rod",
                "Dry print in dark for 15-20 minutes",
            ],
            suggestions=[
                "Use glass dropper for accurate measurement",
                "Mix solution immediately before use for best results",
            ],
        )

    def _recommend_ratio(
        self,
        context: Optional[SkillContext],
        target_tone: str = "neutral",
        target_contrast: str = "normal",
        **kwargs: Any,
    ) -> SkillResult:
        """Recommend metal ratio for desired characteristics."""
        task = kwargs.get("task", "")
        task_lower = task.lower() if task else target_tone.lower()

        # Determine tone from task or parameter
        if "warm" in task_lower:
            target_tone = "warm"
        elif "cool" in task_lower:
            target_tone = "cool"
        elif "neutral" in task_lower:
            target_tone = "neutral"

        # Tone to ratio mapping
        tone_ratios = {
            "warm": {
                "ratio": 0.2,
                "description": "20% Platinum / 80% Palladium",
                "characteristics": "Warm brown tones, softer highlights",
            },
            "neutral": {
                "ratio": 0.5,
                "description": "50% Platinum / 50% Palladium",
                "characteristics": "Balanced tones, versatile",
            },
            "cool": {
                "ratio": 0.8,
                "description": "80% Platinum / 20% Palladium",
                "characteristics": "Cool silver-gray tones, crisp highlights",
            },
        }

        recommendation = tone_ratios.get(target_tone, tone_ratios["neutral"])

        # Contrast recommendations
        contrast_na2 = {
            "low": {"drops": 0, "description": "No contrast agent - softer look"},
            "normal": {"drops": 5, "description": "Standard contrast"},
            "high": {"drops": 10, "description": "Enhanced contrast - punchier look"},
        }

        contrast_rec = contrast_na2.get(target_contrast, contrast_na2["normal"])

        return SkillResult.success_result(
            data={
                "recommended_ratio": recommendation["ratio"],
                "ratio_description": recommendation["description"],
                "characteristics": recommendation["characteristics"],
                "target_tone": target_tone,
                "contrast_na2_drops": contrast_rec["drops"],
                "contrast_description": contrast_rec["description"],
            },
            message=(
                f"For {target_tone} tones: {recommendation['description']} "
                f"with {contrast_rec['drops']} drops Na2"
            ),
            confidence=0.9,
            suggestions=[
                "Run test strips to verify desired look",
                "Adjust ratio +/- 10% based on paper characteristics",
            ],
        )

    def _calculate_contrast(
        self,
        context: Optional[SkillContext],
        target_contrast: str = "normal",
        subject_type: str = "general",
        **kwargs: Any,
    ) -> SkillResult:
        """Calculate contrast agent amount."""
        # Contrast recommendations by subject type
        subject_recommendations = {
            "portrait": {
                "base_drops": 3,
                "description": "Softer contrast for skin tones",
            },
            "landscape": {
                "base_drops": 6,
                "description": "Medium contrast for natural scenes",
            },
            "architecture": {
                "base_drops": 8,
                "description": "Higher contrast for structure definition",
            },
            "general": {
                "base_drops": 5,
                "description": "Balanced for varied subjects",
            },
        }

        subject_rec = subject_recommendations.get(subject_type, subject_recommendations["general"])

        # Contrast level adjustment
        contrast_multipliers = {
            "low": 0.0,
            "normal": 1.0,
            "high": 1.6,
        }

        multiplier = contrast_multipliers.get(target_contrast, 1.0)
        na2_drops = round(subject_rec["base_drops"] * multiplier)

        return SkillResult.success_result(
            data={
                "na2_drops": na2_drops,
                "target_contrast": target_contrast,
                "subject_type": subject_type,
                "base_recommendation": subject_rec["base_drops"],
                "description": subject_rec["description"],
            },
            message=f"Recommended {na2_drops} drops Na2 for {target_contrast} contrast ({subject_type})",
            confidence=0.85,
            suggestions=[
                "Start with recommended amount and adjust based on results",
                "Less Na2 gives softer gradations, more gives punchier shadows",
            ],
        )

    def _estimate_cost(
        self,
        context: Optional[SkillContext],
        print_width: float = 8.0,
        print_height: float = 10.0,
        metal_ratio: Optional[float] = None,
        num_prints: int = 1,
        **kwargs: Any,
    ) -> SkillResult:
        """Estimate chemistry cost for prints."""
        if metal_ratio is None:
            metal_ratio = (
                context.metal_ratio
                if context and context.metal_ratio is not None
                else self.settings.default_metal_ratio
            )

        # Calculate based on coating
        area_sq_inches = print_width * print_height
        total_drops = area_sq_inches * 0.25  # Approximate

        pt_drops = total_drops * metal_ratio
        pd_drops = total_drops * (1 - metal_ratio)

        # Convert to ml
        drops_per_ml = 20
        pt_ml = pt_drops / drops_per_ml
        pd_ml = pd_drops / drops_per_ml

        # Calculate costs
        pt_cost = pt_ml * self.settings.platinum_cost_per_ml
        pd_cost = pd_ml * self.settings.palladium_cost_per_ml
        total_cost_per_print = pt_cost + pd_cost

        return SkillResult.success_result(
            data={
                "cost_per_print": round(total_cost_per_print, 2),
                "total_cost": round(total_cost_per_print * num_prints, 2),
                "breakdown": {
                    "platinum_cost": round(pt_cost, 2),
                    "palladium_cost": round(pd_cost, 2),
                },
                "print_size": f"{print_width} x {print_height}",
                "num_prints": num_prints,
                "metal_ratio": metal_ratio,
            },
            message=f"Estimated ${total_cost_per_print:.2f} per print ({num_prints} prints = ${total_cost_per_print * num_prints:.2f})",
            confidence=0.8,
            warnings=[
                "Costs are estimates based on configured prices",
                "Actual costs vary by supplier and concentration",
            ],
        )

    def _describe_ratio(self, ratio: float) -> str:
        """Convert ratio to human-readable description."""
        pt_percent = int(ratio * 100)
        pd_percent = 100 - pt_percent

        if ratio >= 0.8:
            tone = "Cool"
        elif ratio <= 0.2:
            tone = "Warm"
        else:
            tone = "Neutral"

        return f"{tone} ({pt_percent}% Pt / {pd_percent}% Pd)"
