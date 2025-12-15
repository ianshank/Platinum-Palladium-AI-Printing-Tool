"""
Troubleshooting skill for diagnosing and resolving print issues.

This skill provides problem diagnosis capabilities based on symptoms,
measurements, and historical data.
"""

from dataclasses import dataclass, field
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


@dataclass
class DiagnosisEntry:
    """A single diagnosis with cause and solution."""

    symptom: str
    likely_cause: str
    solution: str
    confidence: float = 0.8
    category: str = "general"


class TroubleshootingSkillSettings(SkillSettings):
    """Settings for the troubleshooting skill."""

    model_config = SettingsConfigDict(env_prefix="PTPD_SKILL_TROUBLESHOOT_")

    # Diagnosis parameters
    max_diagnoses: int = Field(
        default=5, ge=1, le=10, description="Maximum diagnoses to return"
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum confidence for diagnoses"
    )

    # Problem thresholds
    high_dmin_threshold: float = Field(
        default=0.15, description="Threshold for high Dmin warning"
    )
    low_dmax_threshold: float = Field(
        default=1.6, description="Threshold for low Dmax warning"
    )


class TroubleshootingSkill(Skill[TroubleshootingSkillSettings]):
    """
    Skill for troubleshooting print issues.

    Provides capabilities for:
    - Symptom-based diagnosis
    - Data-driven problem identification
    - Solution recommendations
    - Verification suggestions
    """

    def __init__(self, settings: Optional[TroubleshootingSkillSettings] = None) -> None:
        super().__init__(settings)
        self._knowledge_base = self._build_knowledge_base()

    @property
    def name(self) -> str:
        return "troubleshooting"

    @property
    def description(self) -> str:
        return (
            "Diagnoses print problems based on symptoms and measurements, "
            "providing targeted solutions and verification steps"
        )

    @property
    def category(self) -> SkillCategory:
        return SkillCategory.TROUBLESHOOTING

    def _default_settings(self) -> TroubleshootingSkillSettings:
        return TroubleshootingSkillSettings()

    def get_capabilities(self) -> list[str]:
        return [
            "Diagnose print problems from visual symptoms",
            "Analyze density data for issues",
            "Provide targeted solutions",
            "Suggest verification steps",
            "Rank causes by likelihood",
            "Learn from historical problems",
        ]

    def can_handle(self, task: str, context: Optional[SkillContext] = None) -> float:
        """Determine if this skill can handle the troubleshooting task."""
        task_lower = task.lower()

        # High confidence keywords
        high_confidence_keywords = [
            "troubleshoot",
            "problem",
            "issue",
            "wrong",
            "not working",
            "fix",
            "diagnose",
            "why",
        ]

        # Symptom keywords
        symptom_keywords = [
            "muddy",
            "faded",
            "blocked",
            "flat",
            "uneven",
            "staining",
            "bronzing",
            "fog",
            "dark",
            "light",
        ]

        # Check for high confidence matches
        if any(kw in task_lower for kw in high_confidence_keywords):
            return 0.9

        # Check for symptom keywords
        if any(kw in task_lower for kw in symptom_keywords):
            return 0.8

        return 0.0

    def execute(
        self,
        task: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute troubleshooting task."""
        # Extract symptoms from task and kwargs
        symptoms = kwargs.pop("symptoms", [])
        if not symptoms:
            symptoms = self._extract_symptoms_from_task(task)

        # Get additional data
        densities = kwargs.pop("densities", None) or (context.densities if context else None)

        # Run diagnosis
        return self._diagnose(
            symptoms=symptoms,
            densities=densities,
            context=context,
            **kwargs,
        )

    def _diagnose(
        self,
        symptoms: list[str],
        densities: Optional[list[float]] = None,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Run comprehensive diagnosis."""
        all_diagnoses: list[DiagnosisEntry] = []

        # Symptom-based diagnosis
        for symptom in symptoms:
            diagnoses = self._diagnose_symptom(symptom)
            all_diagnoses.extend(diagnoses)

        # Data-based diagnosis
        if densities:
            density_diagnoses = self._diagnose_from_densities(densities)
            all_diagnoses.extend(density_diagnoses)

        # Context-based diagnosis
        if context:
            context_diagnoses = self._diagnose_from_context(context)
            all_diagnoses.extend(context_diagnoses)

        # Deduplicate and rank
        ranked_diagnoses = self._rank_diagnoses(all_diagnoses)

        # Limit results
        top_diagnoses = ranked_diagnoses[: self.settings.max_diagnoses]

        # Generate priority actions
        priority_actions = []
        seen_solutions = set()
        for diag in top_diagnoses:
            if diag.solution not in seen_solutions:
                priority_actions.append(diag.solution)
                seen_solutions.add(diag.solution)
                if len(priority_actions) >= 3:
                    break

        # Calculate overall confidence
        overall_confidence = (
            sum(d.confidence for d in top_diagnoses) / len(top_diagnoses)
            if top_diagnoses
            else 0.0
        )

        return SkillResult.success_result(
            data={
                "diagnoses": [
                    {
                        "symptom": d.symptom,
                        "likely_cause": d.likely_cause,
                        "solution": d.solution,
                        "confidence": round(d.confidence, 2),
                        "category": d.category,
                    }
                    for d in top_diagnoses
                ],
                "priority_actions": priority_actions,
                "total_issues_found": len(all_diagnoses),
                "symptoms_analyzed": symptoms,
                "density_analysis": self._summarize_densities(densities) if densities else None,
            },
            message=(
                f"Found {len(top_diagnoses)} potential issues"
                if top_diagnoses
                else "No specific issues identified"
            ),
            confidence=overall_confidence,
            suggestions=priority_actions,
            next_actions=["Implement suggested fixes and retest", "Document results for future reference"],
        )

    def _diagnose_symptom(self, symptom: str) -> list[DiagnosisEntry]:
        """Diagnose based on a symptom."""
        symptom_lower = symptom.lower()
        diagnoses = []

        for keyword, entries in self._knowledge_base.items():
            if keyword in symptom_lower:
                for entry in entries:
                    diagnoses.append(
                        DiagnosisEntry(
                            symptom=symptom,
                            likely_cause=entry["cause"],
                            solution=entry["solution"],
                            confidence=entry.get("confidence", 0.8),
                            category=entry.get("category", "general"),
                        )
                    )

        return diagnoses

    def _diagnose_from_densities(self, densities: list[float]) -> list[DiagnosisEntry]:
        """Diagnose based on density measurements."""
        diagnoses = []
        dmin = min(densities)
        dmax = max(densities)
        density_range = dmax - dmin

        if dmin > self.settings.high_dmin_threshold:
            diagnoses.append(
                DiagnosisEntry(
                    symptom=f"High Dmin ({dmin:.3f})",
                    likely_cause="Fog or staining in highlights",
                    solution="Check paper humidity, reduce exposure, or extend clearing time",
                    confidence=0.85,
                    category="density",
                )
            )

        if dmax < self.settings.low_dmax_threshold:
            diagnoses.append(
                DiagnosisEntry(
                    symptom=f"Low Dmax ({dmax:.3f})",
                    likely_cause="Insufficient shadow density",
                    solution="Increase exposure time or check chemistry freshness",
                    confidence=0.85,
                    category="density",
                )
            )

        if density_range < 1.5:
            diagnoses.append(
                DiagnosisEntry(
                    symptom=f"Narrow density range ({density_range:.3f})",
                    likely_cause="Compressed tonal range",
                    solution="Review chemistry formula and exposure settings",
                    confidence=0.8,
                    category="density",
                )
            )

        # Check monotonicity
        non_monotonic_count = sum(
            1 for i in range(len(densities) - 1) if densities[i] > densities[i + 1]
        )
        if non_monotonic_count > 0:
            diagnoses.append(
                DiagnosisEntry(
                    symptom=f"Non-monotonic response ({non_monotonic_count} reversals)",
                    likely_cause="Measurement error or process instability",
                    solution="Check measurements and rescan, or apply monotonicity correction",
                    confidence=0.75,
                    category="density",
                )
            )

        return diagnoses

    def _diagnose_from_context(self, context: SkillContext) -> list[DiagnosisEntry]:
        """Diagnose based on context information."""
        diagnoses = []

        # Check humidity
        if context.humidity is not None:
            if context.humidity < 40:
                diagnoses.append(
                    DiagnosisEntry(
                        symptom=f"Low humidity ({context.humidity}%)",
                        likely_cause="Paper may absorb sensitizer unevenly",
                        solution="Condition paper in higher humidity or use humidifying tray",
                        confidence=0.7,
                        category="environment",
                    )
                )
            elif context.humidity > 60:
                diagnoses.append(
                    DiagnosisEntry(
                        symptom=f"High humidity ({context.humidity}%)",
                        likely_cause="Paper may be too damp, causing spreading or fog",
                        solution="Reduce room humidity or use dehumidifier",
                        confidence=0.7,
                        category="environment",
                    )
                )

        # Check temperature
        if context.temperature is not None:
            if context.temperature < 18:
                diagnoses.append(
                    DiagnosisEntry(
                        symptom=f"Low temperature ({context.temperature}C)",
                        likely_cause="Chemistry may react slower than expected",
                        solution="Warm the workspace to 20-22C",
                        confidence=0.6,
                        category="environment",
                    )
                )

        return diagnoses

    def _rank_diagnoses(self, diagnoses: list[DiagnosisEntry]) -> list[DiagnosisEntry]:
        """Rank diagnoses by confidence and deduplicate."""
        # Deduplicate by cause
        seen_causes: dict[str, DiagnosisEntry] = {}
        for diag in diagnoses:
            key = diag.likely_cause.lower()
            if key not in seen_causes or diag.confidence > seen_causes[key].confidence:
                seen_causes[key] = diag

        # Sort by confidence
        ranked = sorted(seen_causes.values(), key=lambda d: d.confidence, reverse=True)

        # Filter by minimum confidence
        return [d for d in ranked if d.confidence >= self.settings.min_confidence]

    def _extract_symptoms_from_task(self, task: str) -> list[str]:
        """Extract symptoms from task description."""
        task_lower = task.lower()
        symptoms = []

        symptom_keywords = [
            "muddy",
            "faded",
            "blocked",
            "flat",
            "uneven",
            "staining",
            "bronzing",
            "fog",
            "fogged",
            "dark",
            "light",
            "weak",
            "strong",
            "contrasty",
            "flat",
        ]

        for keyword in symptom_keywords:
            if keyword in task_lower:
                symptoms.append(keyword)

        return symptoms if symptoms else [task]  # Return full task if no keywords found

    def _summarize_densities(self, densities: list[float]) -> dict[str, Any]:
        """Summarize density measurements."""
        return {
            "dmin": round(min(densities), 3),
            "dmax": round(max(densities), 3),
            "range": round(max(densities) - min(densities), 3),
            "num_steps": len(densities),
        }

    def _build_knowledge_base(self) -> dict[str, list[dict[str, Any]]]:
        """Build the troubleshooting knowledge base."""
        return {
            "muddy": [
                {
                    "cause": "Overexposure",
                    "solution": "Reduce exposure time by 20-30%",
                    "confidence": 0.85,
                    "category": "exposure",
                },
                {
                    "cause": "Paper humidity too high",
                    "solution": "Condition paper in lower humidity environment",
                    "confidence": 0.75,
                    "category": "environment",
                },
                {
                    "cause": "Developer too active",
                    "solution": "Dilute developer or reduce development time",
                    "confidence": 0.7,
                    "category": "chemistry",
                },
            ],
            "faded": [
                {
                    "cause": "Underexposure",
                    "solution": "Increase exposure time by 30-50%",
                    "confidence": 0.9,
                    "category": "exposure",
                },
                {
                    "cause": "Chemistry exhausted",
                    "solution": "Prepare fresh coating solution",
                    "confidence": 0.8,
                    "category": "chemistry",
                },
                {
                    "cause": "Paper humidity too low",
                    "solution": "Humidify paper before coating",
                    "confidence": 0.7,
                    "category": "environment",
                },
            ],
            "blocked": [
                {
                    "cause": "Shadow blocking from overexposure",
                    "solution": "Reduce exposure or increase negative density range",
                    "confidence": 0.85,
                    "category": "exposure",
                },
                {
                    "cause": "Paper too absorbent",
                    "solution": "Consider sizing or different paper",
                    "confidence": 0.7,
                    "category": "paper",
                },
                {
                    "cause": "Too much contrast agent",
                    "solution": "Reduce Na2 by 2-3 drops",
                    "confidence": 0.75,
                    "category": "chemistry",
                },
            ],
            "flat": [
                {
                    "cause": "Insufficient exposure",
                    "solution": "Increase exposure time",
                    "confidence": 0.8,
                    "category": "exposure",
                },
                {
                    "cause": "Low negative density range",
                    "solution": "Adjust digital negative curve for more contrast",
                    "confidence": 0.75,
                    "category": "curve",
                },
                {
                    "cause": "Developer exhausted",
                    "solution": "Use fresh developer",
                    "confidence": 0.7,
                    "category": "chemistry",
                },
            ],
            "uneven": [
                {
                    "cause": "Uneven coating application",
                    "solution": "Use glass rod or improve brush technique",
                    "confidence": 0.9,
                    "category": "technique",
                },
                {
                    "cause": "Paper not flat during exposure",
                    "solution": "Use vacuum frame or glass pressure",
                    "confidence": 0.8,
                    "category": "technique",
                },
                {
                    "cause": "UV light source not uniform",
                    "solution": "Check light source uniformity with test strip",
                    "confidence": 0.7,
                    "category": "equipment",
                },
            ],
            "staining": [
                {
                    "cause": "Incomplete clearing",
                    "solution": "Extend clearing bath time or use fresh clearing solution",
                    "confidence": 0.9,
                    "category": "processing",
                },
                {
                    "cause": "Paper residue",
                    "solution": "Use distilled water for final rinse",
                    "confidence": 0.75,
                    "category": "processing",
                },
                {
                    "cause": "Old or contaminated chemistry",
                    "solution": "Replace clearing agents and use fresh solutions",
                    "confidence": 0.7,
                    "category": "chemistry",
                },
            ],
            "bronzing": [
                {
                    "cause": "Over-coating with sensitizer",
                    "solution": "Reduce sensitizer volume",
                    "confidence": 0.85,
                    "category": "technique",
                },
                {
                    "cause": "Too much platinum in mixture",
                    "solution": "Increase palladium ratio",
                    "confidence": 0.75,
                    "category": "chemistry",
                },
                {
                    "cause": "Incomplete wash after development",
                    "solution": "Extend final wash time",
                    "confidence": 0.7,
                    "category": "processing",
                },
            ],
            "fog": [
                {
                    "cause": "Paper exposed to light before coating",
                    "solution": "Store paper in dark and coat under safe light",
                    "confidence": 0.8,
                    "category": "handling",
                },
                {
                    "cause": "Old or degraded chemistry",
                    "solution": "Use fresh sensitizer solution",
                    "confidence": 0.8,
                    "category": "chemistry",
                },
                {
                    "cause": "Paper humidity too high",
                    "solution": "Condition paper at 45-50% humidity",
                    "confidence": 0.7,
                    "category": "environment",
                },
            ],
        }
