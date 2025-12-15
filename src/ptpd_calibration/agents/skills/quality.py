"""
Quality skill for calibration and print quality assessment.

This skill provides quality assurance capabilities including
pre-print validation, post-print analysis, and quality grading.
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


class QualitySkillSettings(SkillSettings):
    """Settings for the quality skill."""

    model_config = SettingsConfigDict(env_prefix="PTPD_SKILL_QUALITY_")

    # Density thresholds
    min_dmax: float = Field(
        default=1.8, ge=1.0, le=3.0, description="Minimum acceptable Dmax"
    )
    max_dmin: float = Field(
        default=0.15, ge=0.0, le=0.5, description="Maximum acceptable Dmin"
    )
    min_density_range: float = Field(
        default=1.5, ge=0.5, le=3.0, description="Minimum acceptable density range"
    )

    # Environmental thresholds
    min_humidity: float = Field(
        default=40.0, description="Minimum recommended humidity %"
    )
    max_humidity: float = Field(
        default=60.0, description="Maximum recommended humidity %"
    )
    min_temperature: float = Field(
        default=18.0, description="Minimum recommended temperature C"
    )
    max_temperature: float = Field(
        default=24.0, description="Maximum recommended temperature C"
    )

    # Chemistry thresholds
    chemistry_max_age_hours: int = Field(
        default=24, description="Maximum chemistry age in hours"
    )


class QualitySkill(Skill[QualitySkillSettings]):
    """
    Skill for quality assessment tasks.

    Provides capabilities for:
    - Pre-print condition validation
    - Post-print quality analysis
    - Density range assessment
    - Environmental condition checking
    - Chemistry freshness validation
    """

    @property
    def name(self) -> str:
        return "quality"

    @property
    def description(self) -> str:
        return (
            "Handles quality assessment for calibrations and prints, "
            "including pre-print validation and post-print analysis"
        )

    @property
    def category(self) -> SkillCategory:
        return SkillCategory.QUALITY

    def _default_settings(self) -> QualitySkillSettings:
        return QualitySkillSettings()

    def get_capabilities(self) -> list[str]:
        return [
            "Validate pre-print conditions (environment, chemistry, paper)",
            "Assess calibration quality (density range, linearity, monotonicity)",
            "Grade print quality (A-F scale)",
            "Check environmental conditions (humidity, temperature)",
            "Validate chemistry freshness",
            "Generate go/no-go recommendations for printing",
        ]

    def can_handle(self, task: str, context: Optional[SkillContext] = None) -> float:
        """Determine if this skill can handle the quality task."""
        task_lower = task.lower()

        # High confidence keywords
        high_confidence_keywords = [
            "quality",
            "qa",
            "check",
            "validate",
            "verify",
            "grade",
            "assess",
            "pre-print",
            "post-print",
        ]

        # Medium confidence keywords
        medium_confidence_keywords = [
            "dmax",
            "dmin",
            "density range",
            "humidity",
            "temperature",
            "condition",
            "ready",
            "pass",
            "fail",
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
        """Execute quality assessment task."""
        task_lower = task.lower()

        # Determine the specific operation
        if "pre-print" in task_lower or "ready" in task_lower or "before" in task_lower:
            return self._pre_print_check(context, **kwargs)
        elif "post-print" in task_lower or "after" in task_lower or "result" in task_lower:
            return self._post_print_analysis(context, **kwargs)
        elif "environment" in task_lower or "humidity" in task_lower or "temperature" in task_lower:
            return self._check_environment(context, **kwargs)
        elif "chemistry" in task_lower or "fresh" in task_lower:
            return self._check_chemistry(context, **kwargs)
        elif "grade" in task_lower or "score" in task_lower:
            return self._grade_calibration(context, **kwargs)
        else:
            # Default to comprehensive quality check
            return self._comprehensive_check(context, **kwargs)

    def _pre_print_check(
        self,
        context: Optional[SkillContext],
        **kwargs: Any,
    ) -> SkillResult:
        """Check if conditions are ready for printing."""
        checks = {
            "environment": {"passed": True, "issues": []},
            "chemistry": {"passed": True, "issues": []},
            "paper": {"passed": True, "issues": []},
            "calibration": {"passed": True, "issues": []},
        }

        # Check environment if available
        if context:
            if context.humidity is not None:
                if context.humidity < self.settings.min_humidity:
                    checks["environment"]["passed"] = False
                    checks["environment"]["issues"].append(
                        f"Humidity too low ({context.humidity}% < {self.settings.min_humidity}%)"
                    )
                elif context.humidity > self.settings.max_humidity:
                    checks["environment"]["passed"] = False
                    checks["environment"]["issues"].append(
                        f"Humidity too high ({context.humidity}% > {self.settings.max_humidity}%)"
                    )

            if context.temperature is not None:
                if context.temperature < self.settings.min_temperature:
                    checks["environment"]["passed"] = False
                    checks["environment"]["issues"].append(
                        f"Temperature too low ({context.temperature}C < {self.settings.min_temperature}C)"
                    )
                elif context.temperature > self.settings.max_temperature:
                    checks["environment"]["passed"] = False
                    checks["environment"]["issues"].append(
                        f"Temperature too high ({context.temperature}C > {self.settings.max_temperature}C)"
                    )

            # Check if calibration data is available
            if not context.densities:
                checks["calibration"]["passed"] = False
                checks["calibration"]["issues"].append("No calibration data available")

        # Calculate overall pass/fail
        all_passed = all(check["passed"] for check in checks.values())
        total_issues = sum(len(check["issues"]) for check in checks.values())

        if all_passed:
            message = "All pre-print checks passed - ready to print"
            recommendation = "go"
        elif total_issues <= 1:
            message = "Minor issues detected - proceed with caution"
            recommendation = "caution"
        else:
            message = f"{total_issues} issues detected - address before printing"
            recommendation = "no-go"

        return SkillResult.success_result(
            data={
                "checks": checks,
                "all_passed": all_passed,
                "total_issues": total_issues,
                "recommendation": recommendation,
            },
            message=message,
            confidence=0.9 if all_passed else 0.7,
            warnings=[
                issue for check in checks.values() for issue in check["issues"]
            ],
            next_actions=self._get_remediation_actions(checks) if not all_passed else [],
        )

    def _post_print_analysis(
        self,
        context: Optional[SkillContext],
        densities: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Analyze print quality after printing."""
        if densities is None and context and context.densities:
            densities = context.densities

        if not densities:
            return SkillResult.failure_result(
                message="No density data available for analysis",
                suggestions=["Scan the print and extract density measurements"],
            )

        dmin = min(densities)
        dmax = max(densities)
        density_range = dmax - dmin

        # Calculate step uniformity
        step_sizes = [densities[i + 1] - densities[i] for i in range(len(densities) - 1)]
        avg_step = sum(step_sizes) / len(step_sizes) if step_sizes else 0
        uniformity = (
            1.0 - (max(step_sizes) - min(step_sizes)) / avg_step
            if avg_step > 0
            else 0
        )

        # Quality assessment
        quality_issues = []
        quality_score = 100

        if dmin > self.settings.max_dmin:
            quality_issues.append(f"High Dmin ({dmin:.3f}) - possible fog or staining")
            quality_score -= 15

        if dmax < self.settings.min_dmax:
            quality_issues.append(f"Low Dmax ({dmax:.3f}) - insufficient shadow density")
            quality_score -= 20

        if density_range < self.settings.min_density_range:
            quality_issues.append(f"Narrow density range ({density_range:.3f})")
            quality_score -= 25

        if uniformity < 0.7:
            quality_issues.append(f"Poor step uniformity ({uniformity:.2f})")
            quality_score -= 15

        # Check monotonicity
        is_monotonic = all(
            densities[i] <= densities[i + 1] for i in range(len(densities) - 1)
        )
        if not is_monotonic:
            quality_issues.append("Non-monotonic response detected")
            quality_score -= 10

        quality_score = max(0, quality_score)
        grade = self._score_to_grade(quality_score)

        return SkillResult.success_result(
            data={
                "dmin": round(dmin, 3),
                "dmax": round(dmax, 3),
                "density_range": round(density_range, 3),
                "step_uniformity": round(uniformity, 3),
                "is_monotonic": is_monotonic,
                "quality_score": quality_score,
                "quality_grade": grade,
                "issues": quality_issues,
            },
            message=f"Print quality: Grade {grade} ({quality_score}/100)",
            confidence=0.9,
            warnings=quality_issues,
            suggestions=self._get_quality_suggestions(quality_issues),
        )

    def _check_environment(
        self,
        context: Optional[SkillContext],
        humidity: Optional[float] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Check environmental conditions."""
        # Use provided values or context
        humidity = humidity or (context.humidity if context else None)
        temperature = temperature or (context.temperature if context else None)

        issues = []
        recommendations = []

        if humidity is not None:
            if humidity < self.settings.min_humidity:
                issues.append(f"Humidity too low: {humidity}%")
                recommendations.append("Use a humidifier or humidity tray")
            elif humidity > self.settings.max_humidity:
                issues.append(f"Humidity too high: {humidity}%")
                recommendations.append("Use a dehumidifier or air conditioning")
            else:
                recommendations.append(f"Humidity ({humidity}%) is in optimal range")

        if temperature is not None:
            if temperature < self.settings.min_temperature:
                issues.append(f"Temperature too low: {temperature}C")
                recommendations.append("Increase room temperature")
            elif temperature > self.settings.max_temperature:
                issues.append(f"Temperature too high: {temperature}C")
                recommendations.append("Cool the workspace")
            else:
                recommendations.append(f"Temperature ({temperature}C) is in optimal range")

        passed = len(issues) == 0

        return SkillResult.success_result(
            data={
                "humidity": humidity,
                "temperature": temperature,
                "optimal_humidity_range": [self.settings.min_humidity, self.settings.max_humidity],
                "optimal_temperature_range": [self.settings.min_temperature, self.settings.max_temperature],
                "passed": passed,
                "issues": issues,
            },
            message=(
                "Environmental conditions are optimal"
                if passed
                else f"{len(issues)} environmental issue(s) detected"
            ),
            confidence=0.9,
            warnings=issues,
            suggestions=recommendations,
        )

    def _check_chemistry(
        self,
        context: Optional[SkillContext],
        chemistry_age_hours: Optional[int] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Check chemistry freshness."""
        if chemistry_age_hours is None:
            # Assume fresh if not specified
            chemistry_age_hours = 0

        is_fresh = chemistry_age_hours < self.settings.chemistry_max_age_hours

        if is_fresh:
            message = f"Chemistry is fresh ({chemistry_age_hours} hours old)"
            suggestions = []
        else:
            message = f"Chemistry may be degraded ({chemistry_age_hours} hours old)"
            suggestions = [
                "Prepare fresh sensitizer solution",
                "Old chemistry can result in lower Dmax and increased fog",
            ]

        return SkillResult.success_result(
            data={
                "chemistry_age_hours": chemistry_age_hours,
                "max_age_hours": self.settings.chemistry_max_age_hours,
                "is_fresh": is_fresh,
            },
            message=message,
            confidence=0.9 if chemistry_age_hours == 0 else 0.8,
            warnings=[] if is_fresh else ["Chemistry should be replaced"],
            suggestions=suggestions,
        )

    def _grade_calibration(
        self,
        context: Optional[SkillContext],
        densities: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Grade a calibration on A-F scale."""
        if densities is None and context and context.densities:
            densities = context.densities

        if not densities:
            return SkillResult.failure_result(
                message="No density data available for grading",
            )

        # Delegate to post-print analysis
        return self._post_print_analysis(context, densities)

    def _comprehensive_check(
        self,
        context: Optional[SkillContext],
        **kwargs: Any,
    ) -> SkillResult:
        """Run comprehensive quality checks."""
        results = {
            "pre_print": self._pre_print_check(context, **kwargs).data,
            "environment": self._check_environment(context, **kwargs).data,
        }

        if context and context.densities:
            results["calibration"] = self._post_print_analysis(context, **kwargs).data

        all_passed = (
            results["pre_print"].get("all_passed", False)
            and results["environment"].get("passed", False)
        )

        return SkillResult.success_result(
            data=results,
            message=(
                "All quality checks passed"
                if all_passed
                else "Some quality checks need attention"
            ),
            confidence=0.9 if all_passed else 0.7,
        )

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _get_remediation_actions(self, checks: dict) -> list[str]:
        """Get remediation actions for failed checks."""
        actions = []
        for category, check in checks.items():
            if not check["passed"]:
                if category == "environment":
                    actions.append("Adjust environmental conditions before printing")
                elif category == "chemistry":
                    actions.append("Prepare fresh chemistry solution")
                elif category == "paper":
                    actions.append("Condition paper to proper humidity")
                elif category == "calibration":
                    actions.append("Run calibration workflow first")
        return actions

    def _get_quality_suggestions(self, issues: list[str]) -> list[str]:
        """Get suggestions based on quality issues."""
        suggestions = []
        issue_text = " ".join(issues).lower()

        if "dmin" in issue_text:
            suggestions.append("Reduce exposure time or check paper humidity")
        if "dmax" in issue_text:
            suggestions.append("Increase exposure time or check chemistry strength")
        if "range" in issue_text:
            suggestions.append("Review chemistry formula and exposure settings")
        if "uniformity" in issue_text:
            suggestions.append("Consider curve smoothing or interpolation adjustment")
        if "monotonic" in issue_text:
            suggestions.append("Check for measurement errors or apply monotonicity enforcement")

        return suggestions
