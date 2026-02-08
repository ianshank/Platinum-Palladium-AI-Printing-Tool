"""
Auto-linearization module for automatic curve generation.

Provides algorithms for creating linearization curves from step wedge measurements,
with various target curve options and optimization methods.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import interpolate

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType


class LinearizationMethod(str, Enum):
    """Methods for computing linearization curves."""

    DIRECT_INVERSION = "direct_inversion"  # Simple inverse mapping
    SPLINE_FIT = "spline_fit"  # Smooth spline interpolation
    POLYNOMIAL_FIT = "polynomial_fit"  # Polynomial regression
    ITERATIVE = "iterative"  # Iterative refinement
    HYBRID = "hybrid"  # Combination of methods


class TargetResponse(str, Enum):
    """Target response curve types."""

    LINEAR = "linear"  # Straight line (gamma 1.0)
    GAMMA_18 = "gamma_18"  # Monitor gamma 1.8
    GAMMA_22 = "gamma_22"  # sRGB gamma 2.2
    PAPER_WHITE = "paper_white"  # Preserve paper white
    PERCEPTUAL = "perceptual"  # Perceptually uniform
    CUSTOM = "custom"  # User-defined target


@dataclass
class LinearizationConfig:
    """Configuration for linearization."""

    method: LinearizationMethod = LinearizationMethod.SPLINE_FIT
    target: TargetResponse = TargetResponse.LINEAR
    output_points: int = 256  # Number of points in output curve
    smoothing: float = 0.1  # Smoothing factor (0-1)
    iterations: int = 3  # For iterative method
    polynomial_degree: int = 5  # For polynomial method
    preserve_endpoints: bool = True  # Keep 0->0 and 1->1


@dataclass
class LinearizationResult:
    """Result of linearization process."""

    curve: CurveData
    method_used: LinearizationMethod
    target_response: TargetResponse
    measured_densities: list[float]
    target_densities: list[float]
    residual_error: float  # RMS error from target
    max_deviation: float  # Maximum deviation from target
    iterations_used: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "curve_name": self.curve.name,
            "curve_points": len(self.curve.input_values),
            "method": self.method_used.value,
            "target": self.target_response.value,
            "residual_error": round(self.residual_error, 4),
            "max_deviation": round(self.max_deviation, 4),
            "iterations": self.iterations_used,
            "notes": self.notes,
        }


class AutoLinearizer:
    """Automatically generate linearization curves from step wedge measurements.

    Takes measured densities from a step wedge and computes the correction
    curve needed to achieve a target response (typically linear).
    """

    def __init__(self, config: LinearizationConfig | None = None):
        """Initialize auto-linearizer.

        Args:
            config: Linearization configuration. If None, uses defaults.
        """
        self.config = config or LinearizationConfig()

    def linearize(
        self,
        measured_densities: list[float],
        curve_name: str | None = None,
        target: TargetResponse | None = None,
        method: LinearizationMethod | None = None,
    ) -> LinearizationResult:
        """Generate linearization curve from measured densities.

        Args:
            measured_densities: List of measured density values from step wedge
            curve_name: Name for the output curve
            target: Target response type (overrides config)
            method: Linearization method (overrides config)

        Returns:
            LinearizationResult with computed curve
        """
        target = target or self.config.target
        method = method or self.config.method
        curve_name = curve_name or f"Linearized ({target.value})"

        # Normalize densities
        densities = np.array(measured_densities)
        num_steps = len(densities)

        # Create input positions (evenly spaced)
        input_positions = np.linspace(0, 1, num_steps)

        # Compute target densities
        target_densities = self._compute_target(target, num_steps)

        # Compute linearization based on method
        if method == LinearizationMethod.DIRECT_INVERSION:
            curve, notes = self._direct_inversion(input_positions, densities, target_densities)
        elif method == LinearizationMethod.SPLINE_FIT:
            curve, notes = self._spline_fit(input_positions, densities, target_densities)
        elif method == LinearizationMethod.POLYNOMIAL_FIT:
            curve, notes = self._polynomial_fit(input_positions, densities, target_densities)
        elif method == LinearizationMethod.ITERATIVE:
            curve, notes = self._iterative_linearize(input_positions, densities, target_densities)
        else:  # HYBRID
            curve, notes = self._hybrid_linearize(input_positions, densities, target_densities)

        # Compute error metrics
        residual_error, max_deviation = self._compute_error(
            curve, input_positions, densities, target_densities
        )

        # Create CurveData
        curve_data = CurveData(
            name=curve_name,
            input_values=curve[0].tolist(),
            output_values=curve[1].tolist(),
            curve_type=CurveType.LINEAR,
        )

        return LinearizationResult(
            curve=curve_data,
            method_used=method,
            target_response=target,
            measured_densities=measured_densities,
            target_densities=target_densities.tolist(),
            residual_error=residual_error,
            max_deviation=max_deviation,
            notes=notes,
        )

    def refine_curve(
        self,
        existing_curve: CurveData,
        new_measurements: list[float],
    ) -> LinearizationResult:
        """Refine an existing linearization curve with new measurements.

        Args:
            existing_curve: Current linearization curve
            new_measurements: New density measurements taken with current curve

        Returns:
            Refined LinearizationResult
        """
        # Analyze the new measurements to see how far off we are
        num_steps = len(new_measurements)
        input_positions = np.linspace(0, 1, num_steps)
        target_densities = self._compute_target(self.config.target, num_steps)

        # Compute correction needed
        measured = np.array(new_measurements)
        error = target_densities - measured

        # Get current curve as interpolator
        existing_inputs = np.array(existing_curve.input_values)
        existing_outputs = np.array(existing_curve.output_values)
        current_interp = interpolate.interp1d(
            existing_inputs, existing_outputs, kind="cubic", fill_value="extrapolate"
        )

        # Apply correction
        correction_factor = 0.5  # Dampening to avoid overcorrection
        corrected_outputs = current_interp(input_positions) + error * correction_factor

        # Ensure monotonicity and bounds
        corrected_outputs = np.clip(corrected_outputs, 0, 1)
        corrected_outputs = self._enforce_monotonicity(corrected_outputs)

        # Create refined curve
        output_x = np.linspace(0, 1, self.config.output_points)
        refined_interp = interpolate.interp1d(
            input_positions, corrected_outputs, kind="cubic", fill_value="extrapolate"
        )
        output_y = np.clip(refined_interp(output_x), 0, 1)

        # Compute error metrics
        residual_error = float(np.sqrt(np.mean(error**2)))
        max_deviation = float(np.max(np.abs(error)))

        notes = [
            f"Refined from existing curve: {existing_curve.name}",
            f"Correction factor: {correction_factor}",
        ]

        curve_data = CurveData(
            name=f"{existing_curve.name} (refined)",
            input_values=output_x.tolist(),
            output_values=output_y.tolist(),
            curve_type=CurveType.LINEAR,
        )

        return LinearizationResult(
            curve=curve_data,
            method_used=LinearizationMethod.ITERATIVE,
            target_response=self.config.target,
            measured_densities=new_measurements,
            target_densities=target_densities.tolist(),
            residual_error=residual_error,
            max_deviation=max_deviation,
            iterations_used=1,
            notes=notes,
        )

    def _compute_target(self, target: TargetResponse, num_points: int) -> np.ndarray:
        """Compute target density values.

        Args:
            target: Target response type
            num_points: Number of points

        Returns:
            Array of target density values
        """
        x = np.linspace(0, 1, num_points)

        if target == TargetResponse.LINEAR:
            return x

        elif target == TargetResponse.GAMMA_18:
            return x ** (1 / 1.8)

        elif target == TargetResponse.GAMMA_22:
            return x ** (1 / 2.2)

        elif target == TargetResponse.PAPER_WHITE:
            # Slightly lift blacks, compress highlights
            return np.clip(x * 0.95 + 0.02, 0, 1)

        elif target == TargetResponse.PERCEPTUAL:
            # Use L* (lightness) curve for perceptual uniformity
            # Simplified approximation
            return (
                np.where(x <= 0.008856, x * 903.3 / 100, 1.16 * np.power(x, 1 / 3) - 0.16) / 1.0
            )  # Normalize

        else:  # CUSTOM or default
            return x

    def _direct_inversion(
        self,
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[str]]:
        """Simple direct inversion method.

        Args:
            input_positions: Input step positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (curve data, notes)
        """
        notes = ["Direct inversion method"]

        # Normalize measured to 0-1 range
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)

        # Create inverse mapping
        # For each target value, find what input gives that measured value
        inverse_interp = interpolate.interp1d(
            measured_norm, input_positions, kind="linear", fill_value="extrapolate"
        )

        # Generate output curve
        output_x = np.linspace(0, 1, self.config.output_points)
        output_y = np.clip(inverse_interp(output_x), 0, 1)

        if self.config.preserve_endpoints:
            output_y[0] = 0.0
            output_y[-1] = 1.0

        return (output_x, output_y), notes

    def _spline_fit(
        self,
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[str]]:
        """Spline-based linearization.

        Args:
            input_positions: Input step positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (curve data, notes)
        """
        notes = ["Cubic spline interpolation"]

        # Normalize measured
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)

        # Create smoothing spline
        smoothing = self.config.smoothing * len(measured)

        try:
            # Need at least 4 points for cubic spline
            if len(measured_norm) >= 4:
                spline = interpolate.UnivariateSpline(
                    measured_norm, input_positions, s=smoothing, k=3
                )
                notes.append(f"Smoothing factor: {smoothing:.2f}")
            else:
                # Use linear for few points
                spline = interpolate.interp1d(
                    measured_norm, input_positions, kind="linear", fill_value="extrapolate"
                )
                notes.append("Linear interpolation (few data points)")
        except Exception as e:
            # Fall back to linear interpolation
            # Log the exception for debugging
            import logging

            logging.warning(f"Spline fit failed, falling back to linear: {e}")
            spline = interpolate.interp1d(
                measured_norm, input_positions, kind="linear", fill_value="extrapolate"
            )
            notes.append("Fallback to linear interpolation")

        # Generate output curve
        output_x = np.linspace(0, 1, self.config.output_points)
        output_y = np.clip(spline(output_x), 0, 1)

        # Enforce monotonicity
        output_y = self._enforce_monotonicity(output_y)

        if self.config.preserve_endpoints:
            output_y[0] = 0.0
            output_y[-1] = 1.0

        return (output_x, output_y), notes

    def _polynomial_fit(
        self,
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[str]]:
        """Polynomial regression linearization.

        Args:
            input_positions: Input step positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (curve data, notes)
        """
        degree = self.config.polynomial_degree
        notes = [f"Polynomial fit (degree {degree})"]

        # Normalize measured
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)

        # Fit polynomial
        coeffs = np.polyfit(measured_norm, input_positions, degree)
        poly = np.poly1d(coeffs)

        # Generate output curve
        output_x = np.linspace(0, 1, self.config.output_points)
        output_y = np.clip(poly(output_x), 0, 1)

        # Enforce monotonicity
        output_y = self._enforce_monotonicity(output_y)

        if self.config.preserve_endpoints:
            output_y[0] = 0.0
            output_y[-1] = 1.0

        return (output_x, output_y), notes

    def _iterative_linearize(
        self,
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[str]]:
        """Iterative refinement linearization.

        Args:
            input_positions: Input step positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (curve data, notes)
        """
        notes = [f"Iterative refinement ({self.config.iterations} iterations)"]

        # Start with direct inversion
        curve, _ = self._direct_inversion(input_positions, measured, target)
        current_y = curve[1].copy()

        # Iteratively refine
        for i in range(self.config.iterations):
            # Compute error at measurement points
            measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)
            error = target - measured_norm

            # Adjust curve based on error
            adjustment = 0.5 / (i + 1)  # Decreasing adjustment
            current_y = np.clip(
                current_y + np.interp(curve[0], input_positions, error) * adjustment, 0, 1
            )

            # Enforce monotonicity
            current_y = self._enforce_monotonicity(current_y)

        if self.config.preserve_endpoints:
            current_y[0] = 0.0
            current_y[-1] = 1.0

        return (curve[0], current_y), notes

    def _hybrid_linearize(
        self,
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[str]]:
        """Hybrid method combining spline and iterative refinement.

        Args:
            input_positions: Input step positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (curve data, notes)
        """
        notes = ["Hybrid method (spline + iterative)"]

        # Start with spline fit
        curve, spline_notes = self._spline_fit(input_positions, measured, target)
        notes.extend(spline_notes)

        # Refine iteratively
        current_y = curve[1].copy()

        for i in range(2):  # Fewer iterations for hybrid
            # interp = interpolate.interp1d(
            #     curve[0], current_y,
            #     kind="cubic", fill_value="extrapolate"
            # )

            measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)
            error = target - measured_norm

            adjustment = 0.3 / (i + 1)
            current_y = np.clip(
                current_y + np.interp(curve[0], input_positions, error) * adjustment, 0, 1
            )
            current_y = self._enforce_monotonicity(current_y)

        if self.config.preserve_endpoints:
            current_y[0] = 0.0
            current_y[-1] = 1.0

        notes.append("Applied 2 refinement iterations")

        return (curve[0], current_y), notes

    def _enforce_monotonicity(self, values: np.ndarray) -> np.ndarray:
        """Ensure curve values are monotonically increasing.

        Args:
            values: Array of curve values

        Returns:
            Monotonically increasing array
        """
        result = values.copy()
        for i in range(1, len(result)):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
        return result

    def _compute_error(
        self,
        curve: tuple[np.ndarray, np.ndarray],
        input_positions: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, float]:
        """Compute error metrics for the linearization.

        Args:
            curve: Generated curve (x, y)
            input_positions: Original input positions
            measured: Measured densities
            target: Target densities

        Returns:
            Tuple of (RMS error, max deviation)
        """
        # Interpolate curve at input positions
        interp = interpolate.interp1d(curve[0], curve[1], kind="linear", fill_value="extrapolate")

        # Normalize measured
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-10)

        # Compute predicted output
        predicted = interp(measured_norm)

        # Compute error vs target
        error = target - predicted

        rms_error = float(np.sqrt(np.mean(error**2)))
        max_dev = float(np.max(np.abs(error)))

        return rms_error, max_dev

    @staticmethod
    def get_methods() -> list[tuple[str, str]]:
        """Get list of linearization methods with descriptions.

        Returns:
            List of (value, description) tuples
        """
        return [
            (
                LinearizationMethod.DIRECT_INVERSION.value,
                "Direct Inversion - Simple inverse mapping",
            ),
            (LinearizationMethod.SPLINE_FIT.value, "Spline Fit - Smooth cubic spline"),
            (LinearizationMethod.POLYNOMIAL_FIT.value, "Polynomial - Polynomial regression"),
            (LinearizationMethod.ITERATIVE.value, "Iterative - Refinement iterations"),
            (LinearizationMethod.HYBRID.value, "Hybrid - Spline + iterative"),
        ]

    @staticmethod
    def get_targets() -> list[tuple[str, str]]:
        """Get list of target responses with descriptions.

        Returns:
            List of (value, description) tuples
        """
        return [
            (TargetResponse.LINEAR.value, "Linear - Straight 45° line"),
            (TargetResponse.GAMMA_18.value, "Gamma 1.8 - Mac display"),
            (TargetResponse.GAMMA_22.value, "Gamma 2.2 - sRGB/Windows"),
            (TargetResponse.PAPER_WHITE.value, "Paper White - Preserve highlights"),
            (TargetResponse.PERCEPTUAL.value, "Perceptual - L* lightness"),
        ]
