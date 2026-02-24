"""
Active learning for efficient calibration.

Suggests optimal experiments to maximize learning with minimal prints.
"""

import numpy as np

from ptpd_calibration.config import MLSettings, get_settings
from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.ml.predictor import CurvePredictor


class ActiveLearner:
    """
    Active learning system for calibration optimization.

    Suggests the most informative experiments based on model uncertainty
    and parameter space coverage.
    """

    def __init__(
        self,
        predictor: CurvePredictor | None = None,
        settings: MLSettings | None = None,
    ):
        """
        Initialize the active learner.

        Args:
            predictor: Trained CurvePredictor for uncertainty estimation.
            settings: ML settings.
        """
        self.predictor = predictor
        self.settings = settings or get_settings().ml

    def suggest_next_experiment(
        self,
        current_setup: CalibrationRecord,
        variations: list[dict],
        strategy: str = "uncertainty",
    ) -> dict:
        """
        Suggest the most informative next experiment.

        Args:
            current_setup: Current calibration setup.
            variations: List of parameter variations to consider.
            strategy: Selection strategy ("uncertainty", "diversity", "combined").

        Returns:
            Dictionary with suggested variation and rationale.
        """
        if not variations:
            return {
                "variation": {},
                "rationale": "No variations to consider",
                "score": 0.0,
            }

        scores = []

        for variation in variations:
            # Create modified record
            modified = self._apply_variation(current_setup, variation)

            # Calculate score based on strategy
            if strategy == "uncertainty" and self.predictor and self.predictor.is_trained:
                _, uncertainty = self.predictor.predict(modified, return_uncertainty=True)
                score = uncertainty or 0.0
            elif strategy == "diversity":
                score = self._calculate_diversity_score(variation)
            else:
                # Combined: uncertainty + diversity
                uncertainty_score = 0.0
                if self.predictor and self.predictor.is_trained:
                    _, uncertainty = self.predictor.predict(modified, return_uncertainty=True)
                    uncertainty_score = uncertainty or 0.0

                diversity_score = self._calculate_diversity_score(variation)
                exploration_weight = self.settings.exploration_weight

                score = (
                    1 - exploration_weight
                ) * uncertainty_score + exploration_weight * diversity_score

            scores.append(score)

        # Select best
        best_idx = np.argmax(scores)
        best_variation = variations[best_idx]
        best_score = scores[best_idx]

        # Generate rationale
        rationale = self._generate_rationale(best_variation, best_score, strategy)

        return {
            "variation": best_variation,
            "rationale": rationale,
            "score": best_score,
            "all_scores": list(zip(variations, scores, strict=True)),
        }

    def suggest_exposure_bracket(
        self,
        setup: CalibrationRecord,
        bracket_stops: float = 0.5,
        num_brackets: int = 5,
    ) -> list[float]:
        """
        Suggest exposure bracket for testing.

        Args:
            setup: Base calibration setup.
            bracket_stops: Stop increment between brackets.
            num_brackets: Number of brackets.

        Returns:
            List of exposure times in seconds.
        """
        base_exposure = setup.exposure_time
        half_range = (num_brackets - 1) / 2 * bracket_stops

        brackets = []
        for i in range(num_brackets):
            stop_offset = -half_range + i * bracket_stops
            exposure = base_exposure * (2**stop_offset)
            brackets.append(round(exposure, 1))

        return brackets

    def suggest_metal_ratio_series(
        self,
        setup: CalibrationRecord,
        num_steps: int = 5,
    ) -> list[float]:
        """
        Suggest metal ratio series for testing.

        Args:
            setup: Base calibration setup.
            num_steps: Number of ratios to test.

        Returns:
            List of Pt:Pd ratios (0 = pure Pd, 1 = pure Pt).
        """
        # Common ratios used in Pt/Pd printing
        if num_steps <= 3:
            return [0.0, 0.5, 1.0]
        elif num_steps == 5:
            return [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            return list(np.linspace(0, 1, num_steps))

    def suggest_contrast_series(
        self,
        setup: CalibrationRecord,
        agent_type: str = "na2",
        num_steps: int = 5,
    ) -> list[float]:
        """
        Suggest contrast agent series for testing.

        Args:
            setup: Base calibration setup.
            agent_type: Type of contrast agent.
            num_steps: Number of amounts to test.

        Returns:
            List of contrast agent amounts (drops or %).
        """
        # Typical Na2 range is 0-15 drops
        if agent_type == "na2":
            if num_steps <= 3:
                return [0.0, 5.0, 10.0]
            elif num_steps == 5:
                return [0.0, 3.0, 6.0, 9.0, 12.0]
            else:
                return list(np.linspace(0, 12, num_steps))
        else:
            # Generic range
            return list(np.linspace(0, 10, num_steps))

    def evaluate_calibration_quality(
        self,
        record: CalibrationRecord,
        target_curve: list[float] | None = None,
    ) -> dict:
        """
        Evaluate quality of a completed calibration.

        Args:
            record: Completed calibration record.
            target_curve: Optional target density values.

        Returns:
            Dictionary with quality metrics and recommendations.
        """
        densities = record.measured_densities
        if not densities:
            return {
                "quality_score": 0.0,
                "metrics": {},
                "recommendations": ["No density measurements available"],
            }

        metrics = {}
        recommendations = []

        # Basic metrics
        dmin = min(densities)
        dmax = max(densities)
        density_range = dmax - dmin

        metrics["dmin"] = dmin
        metrics["dmax"] = dmax
        metrics["density_range"] = density_range

        # Quality score components
        score = 0.0

        # Density range (target: 1.8-2.2)
        if 1.8 <= density_range <= 2.2:
            score += 0.3
        elif 1.5 <= density_range <= 2.5:
            score += 0.2
        else:
            score += 0.1
            if density_range < 1.5:
                recommendations.append(
                    f"Low density range ({density_range:.2f}). Consider longer exposure."
                )
            else:
                recommendations.append(
                    f"Very high density range ({density_range:.2f}). Consider shorter exposure."
                )

        # Dmin (target: < 0.12)
        if dmin < 0.12:
            score += 0.2
        elif dmin < 0.18:
            score += 0.1
        else:
            recommendations.append(f"High Dmin ({dmin:.2f}). Check clearing or paper quality.")

        # Monotonicity
        diffs = np.diff(densities)
        reversals = np.sum(diffs < -0.02)
        metrics["reversals"] = int(reversals)

        if reversals == 0:
            score += 0.2
        elif reversals <= 2:
            score += 0.1
            recommendations.append("Minor density reversals detected.")
        else:
            recommendations.append(
                f"{reversals} density reversals detected. Possible solarization."
            )

        # Smoothness
        second_diff = np.diff(diffs)
        roughness = np.std(second_diff)
        metrics["roughness"] = float(roughness)

        if roughness < 0.05:
            score += 0.15
        elif roughness < 0.1:
            score += 0.1
        else:
            recommendations.append("Curve is rough. Consider longer exposure brackets.")

        # Compare to target if provided
        if target_curve and len(target_curve) == len(densities):
            errors = np.abs(np.array(densities) - np.array(target_curve))
            max_error = float(np.max(errors))
            mean_error = float(np.mean(errors))

            metrics["max_target_error"] = max_error
            metrics["mean_target_error"] = mean_error

            if mean_error < 0.1:
                score += 0.15
            elif mean_error < 0.2:
                score += 0.1

        if not recommendations:
            recommendations.append("Calibration quality is excellent!")

        return {
            "quality_score": min(1.0, score),
            "metrics": metrics,
            "recommendations": recommendations,
        }

    def _apply_variation(self, record: CalibrationRecord, variation: dict) -> CalibrationRecord:
        """Apply parameter variation to record."""
        data = record.model_dump()
        data.update(variation)
        return CalibrationRecord(**data)

    def _calculate_diversity_score(self, variation: dict) -> float:
        """Calculate diversity score for a variation."""
        # Higher score for larger parameter changes
        score = 0.0

        if "metal_ratio" in variation:
            score += abs(variation["metal_ratio"] - 0.5) * 0.5

        if "exposure_time" in variation:
            score += 0.3

        if "contrast_amount" in variation:
            score += min(variation["contrast_amount"] / 10, 0.3)

        return score

    def _generate_rationale(self, variation: dict, score: float, strategy: str) -> str:
        """Generate human-readable rationale for suggestion."""
        parts = []

        if "metal_ratio" in variation:
            ratio = variation["metal_ratio"]
            if ratio < 0.3:
                parts.append("testing with higher Pd content for warmer tones")
            elif ratio > 0.7:
                parts.append("testing with higher Pt content for cooler tones and higher Dmax")
            else:
                parts.append("testing balanced Pt/Pd ratio")

        if "exposure_time" in variation:
            parts.append(f"testing exposure time of {variation['exposure_time']:.0f}s")

        if "contrast_amount" in variation:
            amount = variation["contrast_amount"]
            if amount == 0:
                parts.append("testing without contrast agent")
            else:
                parts.append(f"testing with {amount:.0f} drops of contrast agent")

        if not parts:
            parts.append("exploring new parameter combination")

        base = f"This variation is suggested for {', '.join(parts)}."

        if strategy == "uncertainty":
            base += f" Model uncertainty: {score:.2f}."
        elif strategy == "diversity":
            base += f" Diversity score: {score:.2f}."
        else:
            base += f" Combined score: {score:.2f}."

        return base
