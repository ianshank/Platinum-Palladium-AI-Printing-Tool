"""
Curve generation for Pt/Pd calibration.

Generates linearization and correction curves from step tablet measurements.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from ptpd_calibration.config import CurveSettings, InterpolationMethod, get_settings
from ptpd_calibration.core.models import CurveData, ExtractionResult
from ptpd_calibration.core.types import CurveType


@dataclass
class TargetCurve:
    """Target curve definition for linearization."""

    input_values: list[float]
    output_values: list[float]
    name: str = "Target"

    @classmethod
    def linear(cls, num_points: int) -> "TargetCurve":
        """Create a linear target curve."""
        values = list(np.linspace(0, 1, num_points))
        return cls(input_values=values, output_values=values, name="Linear")

    @classmethod
    def paper_white_preserve(cls, num_points: int, highlight_hold: float = 0.05) -> "TargetCurve":
        """
        Create target curve that preserves paper white in highlights.

        Args:
            num_points: Number of points in curve.
            highlight_hold: Fraction of highlights to hold at paper white.
        """
        inputs = np.linspace(0, 1, num_points)
        outputs = np.zeros(num_points)

        # Hold highlights at zero
        hold_idx = int(num_points * highlight_hold)
        outputs[:hold_idx] = 0

        # Linear ramp for rest
        if hold_idx < num_points:
            outputs[hold_idx:] = np.linspace(0, 1, num_points - hold_idx)

        return cls(
            input_values=list(inputs),
            output_values=list(outputs),
            name="Paper White Preserve",
        )

    @classmethod
    def aesthetic(cls, num_points: int, shadow_boost: float = 0.1) -> "TargetCurve":
        """
        Create aesthetic curve with slight S-curve characteristic.

        Args:
            num_points: Number of points in curve.
            shadow_boost: Amount of shadow boost (0-0.3).
        """
        inputs = np.linspace(0, 1, num_points)

        # S-curve using smoothstep
        t = inputs
        outputs = t * t * (3 - 2 * t)

        # Add shadow boost
        outputs = outputs * (1 - shadow_boost) + inputs * shadow_boost

        return cls(
            input_values=list(inputs),
            output_values=list(outputs),
            name="Aesthetic",
        )


class CurveGenerator:
    """
    Generates calibration curves from step tablet measurements.

    Supports multiple interpolation methods and target curves for
    different printing requirements.
    """

    def __init__(self, settings: CurveSettings | None = None):
        """
        Initialize the curve generator.

        Args:
            settings: Curve generation settings.
        """
        self.settings = settings or get_settings().curves

    def generate(
        self,
        measured_densities: list[float],
        curve_type: CurveType = CurveType.LINEAR,
        target_curve: TargetCurve | None = None,
        name: str = "Calibration Curve",
        paper_type: str | None = None,
        chemistry: str | None = None,
    ) -> CurveData:
        """
        Generate a calibration curve from measured densities.

        Args:
            measured_densities: List of measured densities from step tablet.
            curve_type: Type of curve to generate.
            target_curve: Optional custom target curve.
            name: Curve name.
            paper_type: Paper type for metadata.
            chemistry: Chemistry description for metadata.

        Returns:
            CurveData with correction values.
        """
        num_steps = len(measured_densities)
        if num_steps < 2:
            raise ValueError("At least 2 measurements required")

        # Normalize densities
        densities = np.array(measured_densities)
        dmin = np.min(densities)
        dmax = np.max(densities)

        if dmax - dmin < 0.01:
            raise ValueError("Density range too small for calibration")

        normalized = (densities - dmin) / (dmax - dmin)

        # Create input values (step positions)
        input_steps = np.linspace(0, 1, num_steps)

        # Get target curve
        if target_curve is None:
            if curve_type == CurveType.LINEAR:
                target_curve = TargetCurve.linear(num_steps)
            elif curve_type == CurveType.PAPER_WHITE:
                target_curve = TargetCurve.paper_white_preserve(
                    num_steps, self.settings.highlight_hold_point
                )
            elif curve_type == CurveType.AESTHETIC:
                target_curve = TargetCurve.aesthetic(num_steps)
            else:
                target_curve = TargetCurve.linear(num_steps)

        # Calculate correction curve
        correction = self._calculate_correction(input_steps, normalized, target_curve)

        # Interpolate to output resolution
        output_values = self._interpolate_curve(correction, self.settings.num_output_points)

        # Apply smoothing if requested
        if self.settings.smoothing_factor > 0:
            output_values = self._smooth_curve(output_values, self.settings.smoothing_factor)

        # Enforce monotonicity if requested
        if self.settings.monotonicity_enforcement:
            output_values = self._enforce_monotonicity(output_values)

        # Create output input values
        input_values = list(np.linspace(0, 1, len(output_values)))

        return CurveData(
            name=name,
            curve_type=curve_type,
            paper_type=paper_type,
            chemistry=chemistry,
            input_values=input_values,
            output_values=list(output_values),
            target_curve_type=target_curve.name,
        )

    def generate_from_extraction(
        self,
        extraction: ExtractionResult,
        curve_type: CurveType = CurveType.LINEAR,
        target_curve: TargetCurve | None = None,
        name: str = "Calibration Curve",
        paper_type: str | None = None,
        chemistry: str | None = None,
    ) -> CurveData:
        """
        Generate curve directly from extraction result.

        Args:
            extraction: Step tablet extraction result.
            curve_type: Type of curve to generate.
            target_curve: Optional custom target curve.
            name: Curve name.
            paper_type: Paper type for metadata.
            chemistry: Chemistry description for metadata.

        Returns:
            CurveData with correction values.
        """
        densities = extraction.get_densities()
        if not densities:
            raise ValueError("No density measurements in extraction")

        curve = self.generate(
            densities,
            curve_type=curve_type,
            target_curve=target_curve,
            name=name,
            paper_type=paper_type,
            chemistry=chemistry,
        )

        # Link to source extraction
        curve.source_extraction_id = extraction.id

        return curve

    def _calculate_correction(
        self,
        input_steps: np.ndarray,
        measured: np.ndarray,
        target: TargetCurve,
    ) -> np.ndarray:
        """
        Calculate correction curve to map measured to target.

        The correction curve is the inverse of the measured response,
        composed with the target curve.
        """
        # Interpolate target to match input steps
        target_interp = np.interp(input_steps, target.input_values, target.output_values)

        # Create inverse of measured curve
        # For each target value, find what input is needed
        correction = np.zeros(len(input_steps))

        for i, target_val in enumerate(target_interp):
            # Find input that produces this target value in measured curve
            # This requires inverting the measured->target mapping
            if i == 0:
                correction[i] = 0.0
            elif i == len(input_steps) - 1:
                correction[i] = 1.0
            else:
                # Binary search for the correction value
                correction[i] = self._inverse_lookup(measured, target_val)

        return correction

    def _inverse_lookup(self, measured: np.ndarray, target_value: float) -> float:
        """Find input value that produces target output in measured curve."""
        # Handle edge cases
        if target_value <= measured[0]:
            return 0.0
        if target_value >= measured[-1]:
            return 1.0

        # Find bracketing indices
        idx = np.searchsorted(measured, target_value)
        if idx == 0:
            return 0.0
        if idx >= len(measured):
            return 1.0

        # Linear interpolation
        x0, x1 = (idx - 1) / (len(measured) - 1), idx / (len(measured) - 1)
        y0, y1 = measured[idx - 1], measured[idx]

        if y1 == y0:
            return x0

        t = (target_value - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    def _interpolate_curve(self, values: np.ndarray, num_points: int) -> np.ndarray:
        """Interpolate curve to specified number of points."""
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, num_points)

        method = self.settings.default_interpolation

        if method == InterpolationMethod.LINEAR:
            interp = interp1d(x_old, values, kind="linear")
        elif method == InterpolationMethod.CUBIC:
            interp = interp1d(x_old, values, kind="cubic")
        elif method in (InterpolationMethod.MONOTONIC, InterpolationMethod.PCHIP):
            interp = PchipInterpolator(x_old, values)
        else:
            interp = interp1d(x_old, values, kind="linear")

        return np.clip(interp(x_new), 0, 1)

    def _smooth_curve(self, values: np.ndarray, factor: float) -> np.ndarray:
        """Apply smoothing to curve."""
        from scipy.ndimage import gaussian_filter1d

        sigma = factor * len(values) / 10
        return gaussian_filter1d(values, sigma)

    def _enforce_monotonicity(self, values: np.ndarray) -> np.ndarray:
        """Ensure curve is monotonically increasing."""
        result = values.copy()
        for i in range(1, len(result)):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
        return result


def generate_linearization_curve(
    measured_densities: list[float],
    name: str = "Linearization",
    paper_type: str | None = None,
    chemistry: str | None = None,
) -> CurveData:
    """
    Convenience function to generate a linearization curve.

    Args:
        measured_densities: Measured densities from step tablet.
        name: Curve name.
        paper_type: Paper type for metadata.
        chemistry: Chemistry description for metadata.

    Returns:
        CurveData for linearization.
    """
    generator = CurveGenerator()
    return generator.generate(
        measured_densities,
        curve_type=CurveType.LINEAR,
        name=name,
        paper_type=paper_type,
        chemistry=chemistry,
    )
