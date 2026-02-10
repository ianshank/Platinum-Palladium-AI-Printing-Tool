"""
AI-powered curve enhancement and optimization.

Uses LLM analysis to provide intelligent suggestions and
automatic enhancements for calibration curves.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.analysis import CurveAnalyzer
from ptpd_calibration.curves.modifier import CurveModifier, SmoothingMethod


class EnhancementGoal(str, Enum):
    """Goals for AI curve enhancement."""

    LINEARIZATION = "linearization"
    MAXIMIZE_RANGE = "maximize_range"
    SMOOTH_GRADATION = "smooth_gradation"
    HIGHLIGHT_DETAIL = "highlight_detail"
    SHADOW_DETAIL = "shadow_detail"
    PORTRAIT_AESTHETIC = "portrait_aesthetic"
    LANDSCAPE_AESTHETIC = "landscape_aesthetic"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"


@dataclass
class EnhancementResult:
    """Result of AI curve enhancement."""

    original_curve: CurveData
    enhanced_curve: CurveData
    adjustments_applied: list[str]
    analysis: dict
    suggestions: list[str]
    confidence: float


@dataclass
class CurveIssue:
    """An identified issue with a curve."""

    issue_type: str
    severity: str  # "low", "medium", "high"
    location: str  # "highlights", "midtones", "shadows", "overall"
    description: str
    suggested_fix: str


class CurveAIEnhancer:
    """
    AI-powered curve enhancement system.

    Analyzes curves, identifies issues, and applies intelligent
    enhancements based on the specified goal.
    """

    def __init__(self) -> None:
        """Initialize the AI enhancer."""
        self.modifier = CurveModifier()
        self.analyzer = CurveAnalyzer()

    async def analyze_and_enhance(
        self,
        curve: CurveData,
        goal: EnhancementGoal = EnhancementGoal.LINEARIZATION,
        auto_apply: bool = True,
        target_densities: list[float] | None = None,
    ) -> EnhancementResult:
        """
        Analyze a curve and optionally apply enhancements.

        Args:
            curve: Input curve to analyze.
            goal: Enhancement goal.
            auto_apply: Whether to automatically apply enhancements.
            target_densities: Optional target density values.

        Returns:
            EnhancementResult with analysis and enhanced curve.
        """
        # Analyze the curve
        analysis = self._analyze_curve(curve, target_densities)

        # Identify issues
        issues = self._identify_issues(curve, analysis, goal)

        # Generate suggestions
        suggestions = self._generate_suggestions(issues, goal)

        # Apply enhancements if requested
        if auto_apply and issues:
            enhanced_curve, adjustments = self._apply_enhancements(curve, issues, goal)
        else:
            enhanced_curve = curve
            adjustments = []

        # Calculate confidence
        confidence = self._calculate_confidence(issues, adjustments)

        return EnhancementResult(
            original_curve=curve,
            enhanced_curve=enhanced_curve,
            adjustments_applied=adjustments,
            analysis=analysis,
            suggestions=suggestions,
            confidence=confidence,
        )

    def analyze_for_llm(
        self,
        curve: CurveData,
        include_recommendations: bool = True,
    ) -> str:
        """
        Generate a detailed analysis formatted for LLM consumption.

        Args:
            curve: Curve to analyze.
            include_recommendations: Whether to include recommendations.

        Returns:
            Formatted analysis string.
        """
        analysis = self._analyze_curve(curve, None)
        issues = self._identify_issues(curve, analysis, EnhancementGoal.LINEARIZATION)

        lines = [
            f"## Curve Analysis: {curve.name}",
            "",
            "### Basic Statistics",
            f"- Number of points: {analysis['num_points']}",
            f"- Output range: {analysis['min_output']:.3f} to {analysis['max_output']:.3f}",
            f"- Monotonic: {'Yes' if analysis['is_monotonic'] else 'No'}",
            "",
            "### Linearity Analysis",
            f"- Max deviation from linear: {analysis['max_linearity_error']:.3f}",
            f"- RMS error: {analysis['rms_linearity_error']:.3f}",
            f"- Curve shape: {analysis['shape']}",
            "",
            "### Tonal Distribution",
            f"- Highlight response: {analysis['highlight_response']}",
            f"- Midtone response: {analysis['midtone_response']}",
            f"- Shadow response: {analysis['shadow_response']}",
            "",
        ]

        if issues:
            lines.extend(
                [
                    "### Identified Issues",
                ]
            )
            for issue in issues:
                lines.append(
                    f"- **{issue.severity.upper()}** ({issue.location}): {issue.description}"
                )
            lines.append("")

        if include_recommendations:
            recommendations = self._generate_suggestions(issues, EnhancementGoal.LINEARIZATION)
            if recommendations:
                lines.extend(
                    [
                        "### Recommendations",
                    ]
                )
                for rec in recommendations:
                    lines.append(f"- {rec}")

        return "\n".join(lines)

    def get_enhancement_prompt(
        self,
        curve: CurveData,
        goal: EnhancementGoal,
        user_requirements: str | None = None,
    ) -> str:
        """
        Generate a prompt for LLM-based enhancement suggestions.

        Args:
            curve: Curve to enhance.
            goal: Enhancement goal.
            user_requirements: Optional user-specified requirements.

        Returns:
            Formatted prompt for LLM.
        """
        analysis_text = self.analyze_for_llm(curve, include_recommendations=False)

        prompt = f"""{analysis_text}

### Enhancement Goal
{goal.value.replace("_", " ").title()}

"""
        if user_requirements:
            prompt += f"""### User Requirements
{user_requirements}

"""

        prompt += """### Task
Based on the curve analysis above, please provide:

1. **Assessment**: A brief assessment of the curve's current state
2. **Specific Adjustments**: Recommended adjustments with specific values:
   - Brightness (-1.0 to 1.0)
   - Contrast (-1.0 to 1.0)
   - Gamma (0.1 to 10.0, where 1.0 is no change)
   - Highlights (-1.0 to 1.0)
   - Shadows (-1.0 to 1.0)
   - Midtones (-1.0 to 1.0)
   - Smoothing (0.0 to 1.0)
3. **Expected Outcome**: What the adjustments will achieve
4. **Warnings**: Any potential issues to watch for

Please respond in a structured format that can be parsed."""

        return prompt

    async def enhance_with_llm(
        self,
        curve: CurveData,
        goal: EnhancementGoal,
        user_requirements: str | None = None,
    ) -> EnhancementResult:
        """
        Enhance curve using LLM suggestions.

        Args:
            curve: Curve to enhance.
            goal: Enhancement goal.
            user_requirements: Optional user requirements.

        Returns:
            EnhancementResult with LLM-guided enhancements.
        """
        try:
            from ptpd_calibration.llm import create_assistant

            # Get LLM suggestions
            assistant = create_assistant()
            prompt = self.get_enhancement_prompt(curve, goal, user_requirements)
            response = await assistant.chat(prompt, include_history=False)

            # Parse LLM response for adjustments
            adjustments = self._parse_llm_response(response)

            # Apply adjustments
            enhanced = curve
            applied = []

            if "brightness" in adjustments and adjustments["brightness"] != 0:
                enhanced = self.modifier.adjust_brightness(enhanced, adjustments["brightness"])
                applied.append(f"brightness: {adjustments['brightness']:+.2f}")

            if "contrast" in adjustments and adjustments["contrast"] != 0:
                enhanced = self.modifier.adjust_contrast(enhanced, adjustments["contrast"])
                applied.append(f"contrast: {adjustments['contrast']:+.2f}")

            if "gamma" in adjustments and adjustments["gamma"] != 1.0:
                enhanced = self.modifier.adjust_gamma(enhanced, adjustments["gamma"])
                applied.append(f"gamma: {adjustments['gamma']:.2f}")

            if "highlights" in adjustments and adjustments["highlights"] != 0:
                enhanced = self.modifier.adjust_highlights(enhanced, adjustments["highlights"])
                applied.append(f"highlights: {adjustments['highlights']:+.2f}")

            if "shadows" in adjustments and adjustments["shadows"] != 0:
                enhanced = self.modifier.adjust_shadows(enhanced, adjustments["shadows"])
                applied.append(f"shadows: {adjustments['shadows']:+.2f}")

            if "midtones" in adjustments and adjustments["midtones"] != 0:
                enhanced = self.modifier.adjust_midtones(enhanced, adjustments["midtones"])
                applied.append(f"midtones: {adjustments['midtones']:+.2f}")

            if "smoothing" in adjustments and adjustments["smoothing"] > 0:
                enhanced = self.modifier.smooth(
                    enhanced, SmoothingMethod.GAUSSIAN, adjustments["smoothing"]
                )
                applied.append(f"smoothing: {adjustments['smoothing']:.2f}")

            analysis = self._analyze_curve(curve, None)

            return EnhancementResult(
                original_curve=curve,
                enhanced_curve=enhanced,
                adjustments_applied=applied,
                analysis=analysis,
                suggestions=[response[:500]],  # Truncate for storage
                confidence=0.8,
            )

        except ImportError:
            # Fall back to rule-based enhancement
            return await self.analyze_and_enhance(curve, goal, auto_apply=True)
        except Exception:
            # Fall back on error
            return await self.analyze_and_enhance(curve, goal, auto_apply=True)

    def _analyze_curve(
        self,
        curve: CurveData,
        target_densities: list[float] | None,
    ) -> dict:
        """Perform detailed curve analysis."""
        outputs = np.array(curve.output_values)
        inputs = np.array(curve.input_values)
        n = len(outputs)

        # Basic statistics
        analysis: dict[str, Any] = {
            "num_points": n,
            "min_output": float(np.min(outputs)),
            "max_output": float(np.max(outputs)),
            "mean_output": float(np.mean(outputs)),
        }

        # Monotonicity
        diffs = np.diff(outputs)
        analysis["is_monotonic"] = bool(np.all(diffs >= -0.001))
        analysis["num_reversals"] = int(np.sum(diffs < -0.001))

        # Linearity analysis
        linear = inputs
        errors = outputs - linear
        analysis["max_linearity_error"] = float(np.max(np.abs(errors)))
        analysis["rms_linearity_error"] = float(np.sqrt(np.mean(errors**2)))
        analysis["mean_linearity_error"] = float(np.mean(errors))

        # Shape characterization
        mid_idx = n // 2
        mid_error = outputs[mid_idx] - inputs[mid_idx]
        if mid_error > 0.1:
            analysis["shape"] = "convex (lifts midtones)"
        elif mid_error < -0.1:
            analysis["shape"] = "concave (darkens midtones)"
        else:
            analysis["shape"] = "approximately linear"

        # Tonal distribution
        highlight_idx = n // 5
        midtone_idx = n // 2
        shadow_idx = 4 * n // 5

        h_response = outputs[highlight_idx] / max(inputs[highlight_idx], 0.001)
        m_response = outputs[midtone_idx] / max(inputs[midtone_idx], 0.001)
        s_response = outputs[shadow_idx] / max(inputs[shadow_idx], 0.001)

        analysis["highlight_response"] = (
            "compressed" if h_response < 0.9 else "expanded" if h_response > 1.1 else "normal"
        )
        analysis["midtone_response"] = (
            "compressed" if m_response < 0.9 else "expanded" if m_response > 1.1 else "normal"
        )
        analysis["shadow_response"] = (
            "compressed" if s_response < 0.9 else "expanded" if s_response > 1.1 else "normal"
        )

        # Smoothness
        second_diff = np.diff(diffs)
        analysis["roughness"] = float(np.std(second_diff))

        # Slope analysis
        if len(diffs) > 0:
            analysis["mean_slope"] = float(np.mean(diffs) * n)
            analysis["max_slope"] = float(np.max(diffs) * n)
            analysis["min_slope"] = float(np.min(diffs) * n)

        return analysis

    def _identify_issues(
        self,
        curve: CurveData,
        analysis: dict,
        goal: EnhancementGoal,
    ) -> list[CurveIssue]:
        """Identify issues with the curve based on the goal."""
        issues = []

        # Check monotonicity
        if not analysis.get("is_monotonic", True):
            issues.append(
                CurveIssue(
                    issue_type="non_monotonic",
                    severity="high",
                    location="overall",
                    description=f"Curve has {analysis.get('num_reversals', 0)} reversals",
                    suggested_fix="Apply monotonicity enforcement",
                )
            )

        # Check roughness
        roughness = analysis.get("roughness", 0)
        if roughness > 0.01:
            issues.append(
                CurveIssue(
                    issue_type="rough",
                    severity="medium" if roughness < 0.03 else "high",
                    location="overall",
                    description=f"Curve is rough (std: {roughness:.4f})",
                    suggested_fix="Apply smoothing filter",
                )
            )

        # Goal-specific checks
        if goal == EnhancementGoal.LINEARIZATION:
            if analysis["max_linearity_error"] > 0.1:
                issues.append(
                    CurveIssue(
                        issue_type="non_linear",
                        severity="high" if analysis["max_linearity_error"] > 0.2 else "medium",
                        location="overall",
                        description=f"Max linearity error: {analysis['max_linearity_error']:.3f}",
                        suggested_fix="Apply linearization correction",
                    )
                )

        elif goal in (EnhancementGoal.HIGHLIGHT_DETAIL, EnhancementGoal.HIGH_KEY):
            if analysis["highlight_response"] == "compressed":
                issues.append(
                    CurveIssue(
                        issue_type="compressed_highlights",
                        severity="medium",
                        location="highlights",
                        description="Highlights are compressed",
                        suggested_fix="Boost highlight values",
                    )
                )

        elif goal in (EnhancementGoal.SHADOW_DETAIL, EnhancementGoal.LOW_KEY):
            if analysis["shadow_response"] == "compressed":
                issues.append(
                    CurveIssue(
                        issue_type="compressed_shadows",
                        severity="medium",
                        location="shadows",
                        description="Shadows are compressed",
                        suggested_fix="Open up shadow values",
                    )
                )

        elif goal == EnhancementGoal.SMOOTH_GRADATION and roughness > 0.005:
            issues.append(
                CurveIssue(
                    issue_type="rough_gradation",
                    severity="medium",
                    location="overall",
                    description="Gradation is not smooth enough",
                    suggested_fix="Apply heavy smoothing",
                )
            )

        return issues

    def _generate_suggestions(
        self,
        issues: list[CurveIssue],
        goal: EnhancementGoal,
    ) -> list[str]:
        """Generate human-readable suggestions."""
        suggestions = []

        for issue in issues:
            suggestions.append(f"{issue.description}. {issue.suggested_fix}.")

        # Goal-specific suggestions
        goal_suggestions = {
            EnhancementGoal.LINEARIZATION: "For digital negatives, aim for a linear response where input matches output.",
            EnhancementGoal.MAXIMIZE_RANGE: "Consider adjusting black and white points to maximize density range.",
            EnhancementGoal.PORTRAIT_AESTHETIC: "A slight S-curve can add pleasing contrast for portraits.",
            EnhancementGoal.LANDSCAPE_AESTHETIC: "Consider boosting midtone contrast for landscape impact.",
            EnhancementGoal.HIGHLIGHT_DETAIL: "Use a shoulder in the highlights to preserve detail.",
            EnhancementGoal.SHADOW_DETAIL: "Use a toe in the shadows to open up dark areas.",
        }

        if goal in goal_suggestions:
            suggestions.append(goal_suggestions[goal])

        return suggestions

    def _apply_enhancements(
        self,
        curve: CurveData,
        issues: list[CurveIssue],
        goal: EnhancementGoal,
    ) -> tuple[CurveData, list[str]]:
        """Apply automatic enhancements based on issues."""
        enhanced = curve
        applied = []

        for issue in issues:
            if issue.issue_type == "non_monotonic":
                enhanced = self.modifier.enforce_monotonicity(enhanced)
                applied.append("Enforced monotonicity")

            elif issue.issue_type == "rough":
                strength = 0.3 if issue.severity == "medium" else 0.5
                enhanced = self.modifier.smooth(enhanced, SmoothingMethod.GAUSSIAN, strength)
                applied.append(f"Applied smoothing ({strength:.1f})")

            elif issue.issue_type == "compressed_highlights":
                enhanced = self.modifier.adjust_highlights(enhanced, 0.15)
                applied.append("Boosted highlights (+0.15)")

            elif issue.issue_type == "compressed_shadows":
                enhanced = self.modifier.adjust_shadows(enhanced, 0.15)
                applied.append("Opened shadows (+0.15)")

        # Goal-specific enhancements
        if goal == EnhancementGoal.PORTRAIT_AESTHETIC and "contrast" not in str(applied):
            enhanced = self.modifier.adjust_contrast(enhanced, 0.1)
            applied.append("Added slight contrast for portraits (+0.1)")

        elif goal == EnhancementGoal.SMOOTH_GRADATION:
            enhanced = self.modifier.smooth(enhanced, SmoothingMethod.SAVGOL, 0.4)
            applied.append("Applied Savitzky-Golay smoothing (0.4)")

        return enhanced, applied

    def _calculate_confidence(
        self,
        issues: list[CurveIssue],
        adjustments: list[str],
    ) -> float:
        """Calculate confidence score for the enhancement."""
        # Start with high confidence
        confidence = 1.0

        # Reduce for high-severity issues
        high_severity = sum(1 for i in issues if i.severity == "high")
        confidence -= high_severity * 0.15

        # Reduce for many adjustments (more complex = less certain)
        confidence -= len(adjustments) * 0.05

        return max(0.3, min(1.0, confidence))

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response for adjustment values."""
        adjustments = {
            "brightness": 0.0,
            "contrast": 0.0,
            "gamma": 1.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "midtones": 0.0,
            "smoothing": 0.0,
        }

        # Simple keyword-based parsing
        response_lower = response.lower()

        for key in adjustments:
            # Look for patterns like "brightness: 0.1" or "brightness = -0.2"
            import re

            patterns = [
                rf"{key}[:\s=]+([+-]?\d*\.?\d+)",
                rf"{key}\s*\(([+-]?\d*\.?\d+)\)",
            ]

            for pattern in patterns:
                match = re.search(pattern, response_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        # Clamp to reasonable ranges
                        if key == "gamma":
                            value = max(0.1, min(10.0, value))
                        elif key == "smoothing":
                            value = max(0.0, min(1.0, value))
                        else:
                            value = max(-1.0, min(1.0, value))
                        adjustments[key] = value
                        break
                    except ValueError:
                        pass

        return adjustments


# Convenience function


async def enhance_curve(
    curve: CurveData,
    goal: str = "linearization",
    use_llm: bool = False,
    user_requirements: str | None = None,
) -> EnhancementResult:
    """
    Convenience function to enhance a curve.

    Args:
        curve: Input curve.
        goal: Enhancement goal name.
        use_llm: Whether to use LLM for enhancement.
        user_requirements: Optional user requirements.

    Returns:
        EnhancementResult with enhanced curve.
    """
    enhancer = CurveAIEnhancer()
    goal_enum = EnhancementGoal(goal)

    if use_llm:
        return await enhancer.enhance_with_llm(curve, goal_enum, user_requirements)
    else:
        return await enhancer.analyze_and_enhance(curve, goal_enum, auto_apply=True)
