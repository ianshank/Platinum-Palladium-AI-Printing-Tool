"""
Curve modification utilities for editing and enhancing calibration curves.

Provides tools for adjusting, smoothing, blending, and transforming curves.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType


class AdjustmentType(str, Enum):
    """Types of curve adjustments."""

    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAMMA = "gamma"
    LEVELS = "levels"
    HIGHLIGHT = "highlight"
    SHADOW = "shadow"
    MIDTONE = "midtone"


class SmoothingMethod(str, Enum):
    """Smoothing methods for curves."""

    GAUSSIAN = "gaussian"
    SAVGOL = "savgol"
    MOVING_AVERAGE = "moving_average"
    SPLINE = "spline"


class BlendMode(str, Enum):
    """Blend modes for combining curves."""

    AVERAGE = "average"
    WEIGHTED = "weighted"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    MIN = "min"
    MAX = "max"


@dataclass
class CurveAdjustment:
    """Specification for a curve adjustment."""

    adjustment_type: AdjustmentType
    amount: float  # -1.0 to 1.0 typically
    parameters: dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class CurveModifier:
    """
    Modifier for editing and enhancing calibration curves.

    Provides non-destructive operations that can be applied to curves
    for fine-tuning and optimization.
    """

    def __init__(self, preserve_endpoints: bool = True):
        """
        Initialize the curve modifier.

        Args:
            preserve_endpoints: Whether to preserve 0 and 1 endpoints.
        """
        self.preserve_endpoints = preserve_endpoints

    def adjust_brightness(
        self,
        curve: CurveData,
        amount: float,
        preserve_black: bool = True,
    ) -> CurveData:
        """
        Adjust overall brightness of the curve.

        Args:
            curve: Input curve.
            amount: Brightness adjustment (-1.0 to 1.0).
            preserve_black: Whether to preserve black point.

        Returns:
            Modified CurveData.
        """
        outputs = np.array(curve.output_values)

        if preserve_black:
            # Shift midtones and highlights, preserve shadows
            shift_factor = np.linspace(0, 1, len(outputs)) ** 0.5
            outputs = outputs + amount * 0.3 * shift_factor
        else:
            outputs = outputs + amount * 0.3

        outputs = np.clip(outputs, 0, 1)

        if self.preserve_endpoints:
            outputs[0] = curve.output_values[0]
            outputs[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, outputs, f"brightness({amount:+.2f})")

    def adjust_contrast(
        self,
        curve: CurveData,
        amount: float,
        pivot: float = 0.5,
    ) -> CurveData:
        """
        Adjust contrast around a pivot point.

        Args:
            curve: Input curve.
            amount: Contrast adjustment (-1.0 to 1.0).
            pivot: Pivot point for contrast (0-1).

        Returns:
            Modified CurveData.
        """
        outputs = np.array(curve.output_values)

        # Calculate contrast factor
        factor = 1.0 + amount

        # Apply contrast around pivot
        outputs = pivot + (outputs - pivot) * factor
        outputs = np.clip(outputs, 0, 1)

        if self.preserve_endpoints:
            outputs[0] = curve.output_values[0]
            outputs[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, outputs, f"contrast({amount:+.2f})")

    def adjust_gamma(
        self,
        curve: CurveData,
        gamma: float,
    ) -> CurveData:
        """
        Apply gamma correction to the curve.

        Args:
            curve: Input curve.
            gamma: Gamma value (0.1-10.0, 1.0 = no change).

        Returns:
            Modified CurveData.
        """
        gamma = max(0.1, min(10.0, gamma))
        outputs = np.array(curve.output_values)

        # Apply gamma
        outputs = np.power(outputs, gamma)

        if self.preserve_endpoints:
            outputs[0] = curve.output_values[0]
            outputs[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, outputs, f"gamma({gamma:.2f})")

    def adjust_levels(
        self,
        curve: CurveData,
        black_point: float = 0.0,
        white_point: float = 1.0,
        midpoint: float = 0.5,
    ) -> CurveData:
        """
        Apply levels adjustment to the curve.

        Args:
            curve: Input curve.
            black_point: New black point (0-1).
            white_point: New white point (0-1).
            midpoint: Midpoint gamma adjustment (0-1).

        Returns:
            Modified CurveData.
        """
        outputs = np.array(curve.output_values)

        # Map to new range
        range_size = white_point - black_point
        if range_size > 0:
            outputs = (outputs - black_point) / range_size
            outputs = np.clip(outputs, 0, 1)

            # Apply midpoint gamma
            gamma = np.log(0.5) / np.log(midpoint) if midpoint > 0 and midpoint < 1 else 1.0
            outputs = np.power(outputs, gamma)

        if self.preserve_endpoints:
            outputs[0] = 0.0
            outputs[-1] = 1.0

        return self._create_modified_curve(curve, outputs, f"levels({black_point:.2f},{white_point:.2f})")

    def adjust_highlights(
        self,
        curve: CurveData,
        amount: float,
        threshold: float = 0.6,
    ) -> CurveData:
        """
        Adjust highlight region of the curve.

        Args:
            curve: Input curve.
            amount: Adjustment amount (-1.0 to 1.0).
            threshold: Threshold for highlight region (0-1).

        Returns:
            Modified CurveData.
        """
        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)

        # Create smooth transition mask
        mask = np.clip((inputs - threshold) / (1 - threshold), 0, 1) ** 2

        # Apply adjustment
        adjustment = amount * 0.3 * mask
        outputs = outputs + adjustment
        outputs = np.clip(outputs, 0, 1)

        if self.preserve_endpoints:
            outputs[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, outputs, f"highlights({amount:+.2f})")

    def adjust_shadows(
        self,
        curve: CurveData,
        amount: float,
        threshold: float = 0.4,
    ) -> CurveData:
        """
        Adjust shadow region of the curve.

        Args:
            curve: Input curve.
            amount: Adjustment amount (-1.0 to 1.0).
            threshold: Threshold for shadow region (0-1).

        Returns:
            Modified CurveData.
        """
        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)

        # Create smooth transition mask (inverted)
        mask = np.clip((threshold - inputs) / threshold, 0, 1) ** 2

        # Apply adjustment
        adjustment = amount * 0.3 * mask
        outputs = outputs + adjustment
        outputs = np.clip(outputs, 0, 1)

        if self.preserve_endpoints:
            outputs[0] = curve.output_values[0]

        return self._create_modified_curve(curve, outputs, f"shadows({amount:+.2f})")

    def adjust_midtones(
        self,
        curve: CurveData,
        amount: float,
        center: float = 0.5,
        width: float = 0.3,
    ) -> CurveData:
        """
        Adjust midtone region of the curve.

        Args:
            curve: Input curve.
            amount: Adjustment amount (-1.0 to 1.0).
            center: Center of midtone region (0-1).
            width: Width of midtone region.

        Returns:
            Modified CurveData.
        """
        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)

        # Create gaussian mask centered on midtones
        mask = np.exp(-((inputs - center) ** 2) / (2 * width ** 2))

        # Apply adjustment
        adjustment = amount * 0.3 * mask
        outputs = outputs + adjustment
        outputs = np.clip(outputs, 0, 1)

        if self.preserve_endpoints:
            outputs[0] = curve.output_values[0]
            outputs[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, outputs, f"midtones({amount:+.2f})")

    def smooth(
        self,
        curve: CurveData,
        method: SmoothingMethod = SmoothingMethod.GAUSSIAN,
        strength: float = 0.5,
    ) -> CurveData:
        """
        Smooth the curve to reduce noise.

        Args:
            curve: Input curve.
            method: Smoothing method.
            strength: Smoothing strength (0-1).

        Returns:
            Smoothed CurveData.
        """
        outputs = np.array(curve.output_values)
        n = len(outputs)

        if method == SmoothingMethod.GAUSSIAN:
            sigma = max(1, int(strength * n / 20))
            smoothed = gaussian_filter1d(outputs, sigma)

        elif method == SmoothingMethod.SAVGOL:
            window = max(5, int(strength * n / 10) | 1)  # Must be odd
            window = min(window, n - 1)
            if window % 2 == 0:
                window += 1
            polyorder = min(3, window - 1)
            smoothed = savgol_filter(outputs, window, polyorder)

        elif method == SmoothingMethod.MOVING_AVERAGE:
            window = max(3, int(strength * n / 10))
            kernel = np.ones(window) / window
            smoothed = np.convolve(outputs, kernel, mode="same")

        elif method == SmoothingMethod.SPLINE:
            inputs = np.array(curve.input_values)
            # Subsample and interpolate
            subsample = max(10, int(n * (1 - strength * 0.9)))
            indices = np.linspace(0, n - 1, subsample).astype(int)
            interp = PchipInterpolator(inputs[indices], outputs[indices])
            smoothed = interp(inputs)

        else:
            smoothed = outputs

        smoothed = np.clip(smoothed, 0, 1)

        if self.preserve_endpoints:
            smoothed[0] = curve.output_values[0]
            smoothed[-1] = curve.output_values[-1]

        return self._create_modified_curve(curve, smoothed, f"smooth({method.value},{strength:.2f})")

    def enforce_monotonicity(
        self,
        curve: CurveData,
        direction: str = "increasing",
    ) -> CurveData:
        """
        Enforce monotonicity in the curve.

        Args:
            curve: Input curve.
            direction: "increasing" or "decreasing".

        Returns:
            Monotonic CurveData.
        """
        outputs = np.array(curve.output_values)

        if direction == "increasing":
            # Make monotonically increasing
            for i in range(1, len(outputs)):
                if outputs[i] < outputs[i - 1]:
                    outputs[i] = outputs[i - 1]
        else:
            # Make monotonically decreasing
            for i in range(1, len(outputs)):
                if outputs[i] > outputs[i - 1]:
                    outputs[i] = outputs[i - 1]

        return self._create_modified_curve(curve, outputs, f"monotonic({direction})")

    def blend(
        self,
        curve1: CurveData,
        curve2: CurveData,
        mode: BlendMode = BlendMode.AVERAGE,
        weight: float = 0.5,
    ) -> CurveData:
        """
        Blend two curves together.

        Args:
            curve1: First curve.
            curve2: Second curve.
            mode: Blend mode.
            weight: Weight for weighted blend (0 = curve1, 1 = curve2).

        Returns:
            Blended CurveData.
        """
        # Resample to common resolution
        n = max(len(curve1.output_values), len(curve2.output_values))
        x = np.linspace(0, 1, n)

        y1 = np.interp(x, curve1.input_values, curve1.output_values)
        y2 = np.interp(x, curve2.input_values, curve2.output_values)

        if mode == BlendMode.AVERAGE:
            blended = (y1 + y2) / 2

        elif mode == BlendMode.WEIGHTED:
            blended = y1 * (1 - weight) + y2 * weight

        elif mode == BlendMode.MULTIPLY:
            blended = y1 * y2

        elif mode == BlendMode.SCREEN:
            blended = 1 - (1 - y1) * (1 - y2)

        elif mode == BlendMode.OVERLAY:
            # Overlay blend mode
            mask = y1 < 0.5
            blended = np.where(mask, 2 * y1 * y2, 1 - 2 * (1 - y1) * (1 - y2))

        elif mode == BlendMode.MIN:
            blended = np.minimum(y1, y2)

        elif mode == BlendMode.MAX:
            blended = np.maximum(y1, y2)

        else:
            blended = (y1 + y2) / 2

        blended = np.clip(blended, 0, 1)

        return CurveData(
            name=f"{curve1.name} + {curve2.name} ({mode.value})",
            input_values=list(x),
            output_values=list(blended),
            curve_type=CurveType.CUSTOM,
        )

    def resample(
        self,
        curve: CurveData,
        num_points: int = 256,
        method: str = "pchip",
    ) -> CurveData:
        """
        Resample curve to a different number of points.

        Args:
            curve: Input curve.
            num_points: Number of output points.
            method: Interpolation method ("linear", "cubic", "pchip").

        Returns:
            Resampled CurveData.
        """
        x_old = np.array(curve.input_values)
        y_old = np.array(curve.output_values)
        x_new = np.linspace(0, 1, num_points)

        if method == "pchip":
            interp = PchipInterpolator(x_old, y_old)
            y_new = interp(x_new)
        elif method == "cubic":
            interp = interp1d(x_old, y_old, kind="cubic", fill_value="extrapolate")
            y_new = interp(x_new)
        else:
            y_new = np.interp(x_new, x_old, y_old)

        y_new = np.clip(y_new, 0, 1)

        return CurveData(
            name=curve.name,
            input_values=list(x_new),
            output_values=list(y_new),
            curve_type=curve.curve_type,
            paper_type=curve.paper_type,
            chemistry=curve.chemistry,
            notes=curve.notes,
        )

    def invert(self, curve: CurveData) -> CurveData:
        """
        Invert the curve (flip vertically).

        Args:
            curve: Input curve.

        Returns:
            Inverted CurveData.
        """
        outputs = 1.0 - np.array(curve.output_values)

        return self._create_modified_curve(curve, outputs, "inverted")

    def reverse(self, curve: CurveData) -> CurveData:
        """
        Reverse the curve (flip horizontally).

        Args:
            curve: Input curve.

        Returns:
            Reversed CurveData.
        """
        outputs = np.array(curve.output_values)[::-1]

        return self._create_modified_curve(curve, outputs, "reversed")

    def add_point_adjustment(
        self,
        curve: CurveData,
        input_value: float,
        output_value: float,
        influence: float = 0.1,
    ) -> CurveData:
        """
        Add a point adjustment to the curve.

        Args:
            curve: Input curve.
            input_value: X position of adjustment (0-1).
            output_value: Target Y value (0-1).
            influence: Radius of influence (0-1).

        Returns:
            Adjusted CurveData.
        """
        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)

        # Current value at input position
        current = np.interp(input_value, inputs, outputs)

        # Calculate adjustment needed
        delta = output_value - current

        # Create gaussian influence mask
        mask = np.exp(-((inputs - input_value) ** 2) / (2 * influence ** 2))

        # Apply adjustment
        outputs = outputs + delta * mask
        outputs = np.clip(outputs, 0, 1)

        return self._create_modified_curve(
            curve, outputs, f"point({input_value:.2f},{output_value:.2f})"
        )

    def apply_adjustments(
        self,
        curve: CurveData,
        adjustments: list[CurveAdjustment],
    ) -> CurveData:
        """
        Apply multiple adjustments in sequence.

        Args:
            curve: Input curve.
            adjustments: List of adjustments to apply.

        Returns:
            Modified CurveData with all adjustments applied.
        """
        result = curve

        for adj in adjustments:
            if adj.adjustment_type == AdjustmentType.BRIGHTNESS:
                result = self.adjust_brightness(result, adj.amount, **adj.parameters)
            elif adj.adjustment_type == AdjustmentType.CONTRAST:
                result = self.adjust_contrast(result, adj.amount, **adj.parameters)
            elif adj.adjustment_type == AdjustmentType.GAMMA:
                result = self.adjust_gamma(result, adj.amount)
            elif adj.adjustment_type == AdjustmentType.LEVELS:
                result = self.adjust_levels(result, **adj.parameters)
            elif adj.adjustment_type == AdjustmentType.HIGHLIGHT:
                result = self.adjust_highlights(result, adj.amount, **adj.parameters)
            elif adj.adjustment_type == AdjustmentType.SHADOW:
                result = self.adjust_shadows(result, adj.amount, **adj.parameters)
            elif adj.adjustment_type == AdjustmentType.MIDTONE:
                result = self.adjust_midtones(result, adj.amount, **adj.parameters)

        return result

    def _create_modified_curve(
        self,
        original: CurveData,
        new_outputs: np.ndarray,
        modification: str,
    ) -> CurveData:
        """Create a new CurveData with modified outputs."""
        notes = original.notes or ""
        if notes:
            notes += f"; {modification}"
        else:
            notes = modification

        return CurveData(
            name=original.name,
            input_values=original.input_values.copy(),
            output_values=list(new_outputs),
            curve_type=CurveType.CUSTOM,
            paper_type=original.paper_type,
            chemistry=original.chemistry,
            notes=notes,
        )


# Convenience functions


def adjust_curve(
    curve: CurveData,
    brightness: float = 0.0,
    contrast: float = 0.0,
    gamma: float = 1.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    midtones: float = 0.0,
) -> CurveData:
    """
    Apply multiple adjustments to a curve in one call.

    Args:
        curve: Input curve.
        brightness: Brightness adjustment (-1 to 1).
        contrast: Contrast adjustment (-1 to 1).
        gamma: Gamma value (0.1-10, 1=no change).
        highlights: Highlight adjustment (-1 to 1).
        shadows: Shadow adjustment (-1 to 1).
        midtones: Midtone adjustment (-1 to 1).

    Returns:
        Modified CurveData.
    """
    modifier = CurveModifier()
    result = curve

    if brightness != 0:
        result = modifier.adjust_brightness(result, brightness)
    if contrast != 0:
        result = modifier.adjust_contrast(result, contrast)
    if gamma != 1.0:
        result = modifier.adjust_gamma(result, gamma)
    if highlights != 0:
        result = modifier.adjust_highlights(result, highlights)
    if shadows != 0:
        result = modifier.adjust_shadows(result, shadows)
    if midtones != 0:
        result = modifier.adjust_midtones(result, midtones)

    return result


def smooth_curve(
    curve: CurveData,
    strength: float = 0.5,
    method: str = "gaussian",
) -> CurveData:
    """
    Convenience function to smooth a curve.

    Args:
        curve: Input curve.
        strength: Smoothing strength (0-1).
        method: Smoothing method name.

    Returns:
        Smoothed CurveData.
    """
    modifier = CurveModifier()
    return modifier.smooth(curve, SmoothingMethod(method), strength)


def blend_curves(
    curve1: CurveData,
    curve2: CurveData,
    weight: float = 0.5,
    mode: str = "weighted",
) -> CurveData:
    """
    Convenience function to blend two curves.

    Args:
        curve1: First curve.
        curve2: Second curve.
        weight: Blend weight (0=curve1, 1=curve2).
        mode: Blend mode name.

    Returns:
        Blended CurveData.
    """
    modifier = CurveModifier()
    return modifier.blend(curve1, curve2, BlendMode(mode), weight)
