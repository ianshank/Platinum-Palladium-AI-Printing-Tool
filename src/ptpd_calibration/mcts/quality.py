"""
Quality scoring for MCTS calibration optimization.

Computes multi-objective quality scores combining linearity, dmax target matching,
smoothness, and material cost efficiency.
"""

import logging

import numpy as np

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings
from ptpd_calibration.mcts.types import SimulationResult

logger = logging.getLogger(__name__)


class QualityScorer:
    """Computes quality scores for calibration curves.

    Combines multiple quality metrics (linearity, dmax, smoothness, cost)
    with configurable weights to produce an overall quality score in [0, 1].
    """

    def __init__(self, settings: MCTSSettings | None = None):
        """Initialize QualityScorer.

        Args:
            settings: MCTS settings with quality metric weights and targets.
                     If None, uses defaults.
        """
        self.settings = settings or MCTSSettings()

        logger.debug(
            f"Initialized QualityScorer with weights: "
            f"linearity={self.settings.linearity_weight:.2f}, "
            f"dmax={self.settings.dmax_weight:.2f}, "
            f"smoothness={self.settings.smoothness_weight:.2f}, "
            f"cost={self.settings.cost_weight:.2f}"
        )

    def score(
        self,
        result: SimulationResult,
        target_curve: list[float] | None = None,
    ) -> float:
        """Compute overall quality score in [0, 1].

        Args:
            result: SimulationResult to score
            target_curve: Optional target curve for comparison. If provided,
                         linearity score compares to this instead of ideal linear.

        Returns:
            Overall quality score (weighted combination of metrics)
        """
        # Compute individual scores
        scores = {
            "linearity": self._linearity_score(result.density_curve, target_curve),
            "dmax": self._dmax_score(result.dmax),
            "smoothness": self._smoothness_score(result.density_curve),
            "cost": self._cost_score(result.parameters),
        }

        # Get weights
        weights = {
            "linearity": self.settings.linearity_weight,
            "dmax": self.settings.dmax_weight,
            "smoothness": self.settings.smoothness_weight,
            "cost": self.settings.cost_weight,
        }

        # Compute weighted sum
        total_weight = sum(weights.values())
        if total_weight == 0:
            logger.warning("All quality weights are zero, returning score of 0.0")
            return 0.0

        weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight

        # Ensure result is in [0, 1]
        final_score = np.clip(weighted_score, 0.0, 1.0)

        logger.debug(
            f"Quality scores: linearity={scores['linearity']:.3f}, "
            f"dmax={scores['dmax']:.3f}, smoothness={scores['smoothness']:.3f}, "
            f"cost={scores['cost']:.3f}, weighted={final_score:.3f}"
        )

        return float(final_score)

    def _linearity_score(
        self,
        curve: list[float],
        target_curve: list[float] | None = None,
    ) -> float:
        """Score how linear the curve is (ideal is straight line from dmin to dmax).

        Args:
            curve: Density curve to score
            target_curve: Optional target curve for comparison

        Returns:
            Linearity score in [0, 1], where 1 is perfectly linear
        """
        if len(curve) < 2:
            logger.warning("Curve too short for linearity scoring")
            return 0.0

        curve_array = np.array(curve)

        if target_curve is not None:
            # Compare to provided target curve
            if len(target_curve) != len(curve):
                logger.warning(
                    f"Target curve length {len(target_curve)} != curve length {len(curve)}, "
                    "using ideal linear instead"
                )
                target_array = None
            else:
                target_array = np.array(target_curve)
        else:
            target_array = None

        # If no target, use ideal linear ramp
        if target_array is None:
            dmin = curve_array.min()
            dmax = curve_array.max()
            ideal = np.linspace(dmin, dmax, len(curve))
        else:
            ideal = target_array

        # Compute RMSE between curve and ideal
        mse = np.mean((curve_array - ideal) ** 2)
        rmse = np.sqrt(mse)

        # Normalize by range
        density_range = curve_array.max() - curve_array.min()
        if density_range < 1e-6:
            logger.warning("Density range too small for linearity scoring")
            return 0.0

        normalized_rmse = rmse / density_range

        # Convert to score: lower RMSE = higher score
        # Use exponential decay: score = exp(-k * rmse)
        # With k=5, rmse=0.2 gives score ~0.37, rmse=0.1 gives ~0.61
        score = np.exp(-5.0 * normalized_rmse)

        return float(np.clip(score, 0.0, 1.0))

    def _dmax_score(self, dmax: float) -> float:
        """Score how close dmax is to target.

        Args:
            dmax: Maximum density achieved

        Returns:
            Dmax score in [0, 1], where 1 is exactly on target
        """
        target = self.settings.target_dmax

        # Gaussian-shaped score centered on target
        # score = exp(-((dmax - target) / sigma)^2)
        # sigma controls width; use 0.5 so that deviation of ±0.5 gives score ~0.14
        sigma = 0.5
        deviation = abs(dmax - target)
        score = np.exp(-((deviation / sigma) ** 2))

        return float(np.clip(score, 0.0, 1.0))

    def _smoothness_score(self, curve: list[float]) -> float:
        """Score smoothness (penalize second-derivative magnitude).

        Args:
            curve: Density curve to score

        Returns:
            Smoothness score in [0, 1], where 1 is perfectly smooth
        """
        if len(curve) < 3:
            logger.warning("Curve too short for smoothness scoring")
            return 1.0  # No way to compute second derivative

        curve_array = np.array(curve)

        # Compute first derivative (forward differences)
        first_deriv = np.diff(curve_array)

        # Compute second derivative (forward differences of first deriv)
        second_deriv = np.diff(first_deriv)

        # Measure of smoothness: lower second derivative magnitude = smoother
        # Use mean absolute second derivative
        mean_abs_second_deriv = np.mean(np.abs(second_deriv))

        # Normalize by range
        density_range = curve_array.max() - curve_array.min()
        if density_range < 1e-6:
            return 1.0  # Flat curve is perfectly smooth

        normalized_second_deriv = mean_abs_second_deriv / density_range

        # Convert to score: lower second derivative = higher score
        # Use exponential decay: score = exp(-k * second_deriv)
        # With k=20, normalized_second_deriv=0.1 gives score ~0.14
        score = np.exp(-20.0 * normalized_second_deriv)

        return float(np.clip(score, 0.0, 1.0))

    def _cost_score(self, parameters: dict[str, float]) -> float:
        """Score material cost efficiency.

        Args:
            parameters: Calibration parameters dictionary

        Returns:
            Cost score in [0, 1], where 1 is most cost-efficient
        """
        # Extract relevant cost parameters
        metal_ratio = parameters.get("metal_ratio", 0.5)  # Pt fraction
        coating_weight = parameters.get("coating_weight", 1.5)  # ml/sq-inch

        # Cost model:
        # - Platinum is ~2x palladium cost (approximate)
        # - Lower metal_ratio = more Pd = lower cost
        # - Lower coating_weight = less material = lower cost

        # Metal cost score: favor palladium (lower metal_ratio)
        # metal_ratio 0.0 (pure Pd) -> score 1.0
        # metal_ratio 1.0 (pure Pt) -> score 0.0
        metal_cost_score = 1.0 - metal_ratio

        # Coating cost score: favor lower coating weight
        # Normalize by range from configuration
        coating_range = DEFAULT_PARAMETER_RANGES.get("coating_weight")
        min_coating = coating_range.min_value if coating_range else 0.5
        max_coating = coating_range.max_value if coating_range else 3.0
        coating_normalized = (coating_weight - min_coating) / (max_coating - min_coating)
        coating_normalized = np.clip(coating_normalized, 0.0, 1.0)
        coating_cost_score = 1.0 - coating_normalized

        # Combine with equal weight
        # (Could make this configurable if needed)
        overall_cost_score = 0.6 * metal_cost_score + 0.4 * coating_cost_score

        return float(np.clip(overall_cost_score, 0.0, 1.0))
