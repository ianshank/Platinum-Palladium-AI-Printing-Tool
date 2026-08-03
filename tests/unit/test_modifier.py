"""
Tests for the curve modifier utilities.
"""

import numpy as np
import pytest

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.modifier import (
    AdjustmentType,
    BlendMode,
    CurveAdjustment,
    CurveModifier,
    SmoothingMethod,
    adjust_curve,
    blend_curves,
    smooth_curve,
)


@pytest.fixture
def linear_curve():
    """Create a linear curve for testing."""
    inputs = list(np.linspace(0, 1, 256))
    outputs = list(np.linspace(0, 1, 256))
    return CurveData(
        name="Linear Test",
        input_values=inputs,
        output_values=outputs,
    )


@pytest.fixture
def nonlinear_curve():
    """Create a non-linear (gamma 2.2) curve for testing."""
    inputs = list(np.linspace(0, 1, 256))
    outputs = list(np.power(np.linspace(0, 1, 256), 2.2))
    return CurveData(
        name="Gamma 2.2",
        input_values=inputs,
        output_values=outputs,
    )


@pytest.fixture
def noisy_curve():
    """Create a noisy curve for testing."""
    inputs = list(np.linspace(0, 1, 256))
    base = np.linspace(0, 1, 256)
    noise = np.random.normal(0, 0.03, 256)
    outputs = list(np.clip(base + noise, 0, 1))
    return CurveData(
        name="Noisy Curve",
        input_values=inputs,
        output_values=outputs,
    )


class TestCurveModifier:
    """Tests for CurveModifier class."""

    def test_modifier_initialization(self):
        """Test modifier initialization."""
        modifier = CurveModifier()
        assert modifier.preserve_endpoints is True

        modifier_no_preserve = CurveModifier(preserve_endpoints=False)
        assert modifier_no_preserve.preserve_endpoints is False

    def test_adjust_brightness_positive(self, linear_curve):
        """Test positive brightness adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_brightness(linear_curve, 0.5)

        assert result.name == linear_curve.name
        # Midtones should be brighter
        mid_idx = len(result.output_values) // 2
        assert result.output_values[mid_idx] > linear_curve.output_values[mid_idx]

    def test_adjust_brightness_negative(self, linear_curve):
        """Test negative brightness adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_brightness(linear_curve, -0.5)

        mid_idx = len(result.output_values) // 2
        assert result.output_values[mid_idx] < linear_curve.output_values[mid_idx]

    def test_adjust_brightness_preserves_endpoints(self, linear_curve):
        """Test that endpoints are preserved."""
        modifier = CurveModifier(preserve_endpoints=True)
        result = modifier.adjust_brightness(linear_curve, 0.5)

        assert result.output_values[0] == linear_curve.output_values[0]
        assert result.output_values[-1] == linear_curve.output_values[-1]

    def test_adjust_contrast_positive(self, linear_curve):
        """Test positive contrast adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_contrast(linear_curve, 0.5)

        # Values below pivot should decrease, above should increase
        low_idx = len(result.output_values) // 4
        high_idx = 3 * len(result.output_values) // 4

        assert result.output_values[low_idx] < linear_curve.output_values[low_idx]
        assert result.output_values[high_idx] > linear_curve.output_values[high_idx]

    def test_adjust_contrast_negative(self, linear_curve):
        """Test negative contrast (compression)."""
        modifier = CurveModifier()
        result = modifier.adjust_contrast(linear_curve, -0.5)

        # Values should move toward pivot
        low_idx = len(result.output_values) // 4
        high_idx = 3 * len(result.output_values) // 4

        assert result.output_values[low_idx] > linear_curve.output_values[low_idx]
        assert result.output_values[high_idx] < linear_curve.output_values[high_idx]

    def test_adjust_gamma(self, linear_curve):
        """Test gamma adjustment."""
        modifier = CurveModifier()

        # Gamma < 1 brightens midtones
        result_bright = modifier.adjust_gamma(linear_curve, 0.5)
        mid_idx = len(result_bright.output_values) // 2
        assert result_bright.output_values[mid_idx] > linear_curve.output_values[mid_idx]

        # Gamma > 1 darkens midtones
        result_dark = modifier.adjust_gamma(linear_curve, 2.0)
        assert result_dark.output_values[mid_idx] < linear_curve.output_values[mid_idx]

    def test_adjust_gamma_clamped(self, linear_curve):
        """Test gamma is clamped to valid range."""
        modifier = CurveModifier()

        # Extreme values should be clamped
        result_low = modifier.adjust_gamma(linear_curve, 0.01)  # Should clamp to 0.1
        result_high = modifier.adjust_gamma(linear_curve, 100.0)  # Should clamp to 10.0

        # Results should still be valid
        assert all(0 <= v <= 1 for v in result_low.output_values)
        assert all(0 <= v <= 1 for v in result_high.output_values)

    def test_adjust_levels(self, linear_curve):
        """Test levels adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_levels(linear_curve, black_point=0.1, white_point=0.9)

        # Output should be rescaled
        assert result.output_values[0] == 0.0  # Endpoint preserved
        assert result.output_values[-1] == 1.0  # Endpoint preserved

    def test_adjust_highlights(self, linear_curve):
        """Test highlight adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_highlights(linear_curve, 0.5)

        # High values should be affected more than low values
        high_idx = int(len(result.output_values) * 0.9)
        low_idx = int(len(result.output_values) * 0.1)

        high_delta = abs(result.output_values[high_idx] - linear_curve.output_values[high_idx])
        low_delta = abs(result.output_values[low_idx] - linear_curve.output_values[low_idx])

        assert high_delta > low_delta

    def test_adjust_shadows(self, linear_curve):
        """Test shadow adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_shadows(linear_curve, 0.5)

        # Low values should be affected more than high values
        high_idx = int(len(result.output_values) * 0.9)
        low_idx = int(len(result.output_values) * 0.1)

        high_delta = abs(result.output_values[high_idx] - linear_curve.output_values[high_idx])
        low_delta = abs(result.output_values[low_idx] - linear_curve.output_values[low_idx])

        assert low_delta > high_delta

    def test_adjust_midtones(self, linear_curve):
        """Test midtone adjustment."""
        modifier = CurveModifier()
        result = modifier.adjust_midtones(linear_curve, 0.5)

        # Midtones should be affected most
        mid_idx = len(result.output_values) // 2
        quarter_idx = len(result.output_values) // 4

        mid_delta = abs(result.output_values[mid_idx] - linear_curve.output_values[mid_idx])
        quarter_delta = abs(
            result.output_values[quarter_idx] - linear_curve.output_values[quarter_idx]
        )

        assert mid_delta > quarter_delta


class TestSmoothing:
    """Tests for smoothing methods."""

    def test_smooth_gaussian(self, noisy_curve):
        """Test Gaussian smoothing."""
        modifier = CurveModifier()
        result = modifier.smooth(noisy_curve, SmoothingMethod.GAUSSIAN, 0.5)

        # Calculate roughness (std of second derivative)
        orig_rough = np.std(np.diff(np.diff(noisy_curve.output_values)))
        new_rough = np.std(np.diff(np.diff(result.output_values)))

        assert new_rough < orig_rough

    def test_smooth_savgol(self, noisy_curve):
        """Test Savitzky-Golay smoothing."""
        modifier = CurveModifier()
        result = modifier.smooth(noisy_curve, SmoothingMethod.SAVGOL, 0.5)

        # Result should be valid
        assert all(0 <= v <= 1 for v in result.output_values)
        assert len(result.output_values) == len(noisy_curve.output_values)

    def test_smooth_moving_average(self, noisy_curve):
        """Test moving average smoothing."""
        modifier = CurveModifier()
        result = modifier.smooth(noisy_curve, SmoothingMethod.MOVING_AVERAGE, 0.5)

        assert all(0 <= v <= 1 for v in result.output_values)

    def test_smooth_spline(self, noisy_curve):
        """Test spline smoothing."""
        modifier = CurveModifier()
        result = modifier.smooth(noisy_curve, SmoothingMethod.SPLINE, 0.5)

        assert all(0 <= v <= 1 for v in result.output_values)

    def test_smooth_preserves_endpoints(self, noisy_curve):
        """Test that smoothing preserves endpoints."""
        modifier = CurveModifier(preserve_endpoints=True)
        result = modifier.smooth(noisy_curve, SmoothingMethod.GAUSSIAN, 0.5)

        assert result.output_values[0] == noisy_curve.output_values[0]
        assert result.output_values[-1] == noisy_curve.output_values[-1]


class TestBlending:
    """Tests for curve blending."""

    def test_blend_average(self, linear_curve, nonlinear_curve):
        """Test average blending."""
        modifier = CurveModifier()
        result = modifier.blend(linear_curve, nonlinear_curve, BlendMode.AVERAGE)

        # Result should be between the two curves
        mid_idx = len(result.output_values) // 2
        linear_val = linear_curve.output_values[mid_idx]
        nonlinear_val = np.interp(
            result.input_values[mid_idx],
            nonlinear_curve.input_values,
            nonlinear_curve.output_values,
        )

        assert (
            min(linear_val, nonlinear_val)
            <= result.output_values[mid_idx]
            <= max(linear_val, nonlinear_val)
        )

    def test_blend_weighted(self, linear_curve, nonlinear_curve):
        """Test weighted blending."""
        modifier = CurveModifier()

        # Weight 0 should give curve1
        result_0 = modifier.blend(linear_curve, nonlinear_curve, BlendMode.WEIGHTED, weight=0.0)
        assert np.allclose(result_0.output_values, linear_curve.output_values, atol=0.01)

        # Weight 1 should give curve2
        result_1 = modifier.blend(linear_curve, nonlinear_curve, BlendMode.WEIGHTED, weight=1.0)
        expected = np.interp(
            result_1.input_values, nonlinear_curve.input_values, nonlinear_curve.output_values
        )
        assert np.allclose(result_1.output_values, expected, atol=0.01)

    def test_blend_multiply(self, linear_curve):
        """Test multiply blending."""
        modifier = CurveModifier()
        result = modifier.blend(linear_curve, linear_curve, BlendMode.MULTIPLY)

        # x * x = x^2
        expected = np.array(linear_curve.output_values) ** 2
        assert np.allclose(result.output_values, expected, atol=0.01)

    def test_blend_screen(self, linear_curve):
        """Test screen blending."""
        modifier = CurveModifier()
        result = modifier.blend(linear_curve, linear_curve, BlendMode.SCREEN)

        # Screen with itself: 1 - (1-x)^2 = 2x - x^2
        x = np.array(linear_curve.output_values)
        expected = 1 - (1 - x) * (1 - x)
        assert np.allclose(result.output_values, expected, atol=0.01)

    def test_blend_min(self, linear_curve, nonlinear_curve):
        """Test min blending."""
        modifier = CurveModifier()
        result = modifier.blend(linear_curve, nonlinear_curve, BlendMode.MIN)

        # All values should be <= both inputs
        assert all(v <= 1.0 for v in result.output_values)

    def test_blend_max(self, linear_curve, nonlinear_curve):
        """Test max blending."""
        modifier = CurveModifier()
        result = modifier.blend(linear_curve, nonlinear_curve, BlendMode.MAX)

        # All values should be >= 0
        assert all(v >= 0.0 for v in result.output_values)


class TestMonotonicity:
    """Tests for monotonicity enforcement."""

    def test_enforce_monotonicity_increasing(self):
        """Test enforcing increasing monotonicity."""
        inputs = list(np.linspace(0, 1, 10))
        outputs = [0.0, 0.2, 0.15, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 1.0]  # Non-monotonic
        curve = CurveData(name="Non-mono", input_values=inputs, output_values=outputs)

        modifier = CurveModifier()
        result = modifier.enforce_monotonicity(curve, direction="increasing")

        # Check monotonicity
        diffs = np.diff(result.output_values)
        assert all(d >= 0 for d in diffs)

    def test_enforce_monotonicity_decreasing(self):
        """Test enforcing decreasing monotonicity."""
        inputs = list(np.linspace(0, 1, 10))
        outputs = [1.0, 0.8, 0.85, 0.6, 0.65, 0.4, 0.3, 0.2, 0.1, 0.0]  # Non-monotonic
        curve = CurveData(name="Non-mono", input_values=inputs, output_values=outputs)

        modifier = CurveModifier()
        result = modifier.enforce_monotonicity(curve, direction="decreasing")

        # Check monotonicity
        diffs = np.diff(result.output_values)
        assert all(d <= 0 for d in diffs)


class TestResample:
    """Tests for curve resampling."""

    def test_resample_upsample(self, linear_curve):  # noqa: ARG002
        """Test upsampling a curve."""
        # Create a smaller curve first
        inputs = list(np.linspace(0, 1, 10))
        outputs = list(np.linspace(0, 1, 10))
        small_curve = CurveData(name="Small", input_values=inputs, output_values=outputs)

        modifier = CurveModifier()
        result = modifier.resample(small_curve, num_points=256)

        assert len(result.output_values) == 256
        assert result.output_values[0] == pytest.approx(0.0, abs=0.01)
        assert result.output_values[-1] == pytest.approx(1.0, abs=0.01)

    def test_resample_downsample(self, linear_curve):
        """Test downsampling a curve."""
        modifier = CurveModifier()
        result = modifier.resample(linear_curve, num_points=32)

        assert len(result.output_values) == 32


class TestInvertAndReverse:
    """Tests for invert and reverse operations."""

    def test_invert(self, linear_curve):
        """Test curve inversion."""
        modifier = CurveModifier()
        result = modifier.invert(linear_curve)

        # Inverted linear should be 1-x
        expected = [1.0 - v for v in linear_curve.output_values]
        assert np.allclose(result.output_values, expected, atol=0.001)

    def test_reverse(self, linear_curve):
        """Test curve reversal."""
        modifier = CurveModifier()
        result = modifier.reverse(linear_curve)

        # Reversed should be flipped horizontally
        expected = linear_curve.output_values[::-1]
        assert np.allclose(result.output_values, expected, atol=0.001)


class TestPointAdjustment:
    """Tests for point adjustments."""

    def test_add_point_adjustment(self, linear_curve):
        """Test adding a point adjustment."""
        modifier = CurveModifier()
        result = modifier.add_point_adjustment(
            linear_curve,
            input_value=0.5,
            output_value=0.7,
            influence=0.1,
        )

        # Value at 0.5 should be closer to 0.7
        mid_idx = len(result.output_values) // 2
        assert result.output_values[mid_idx] > linear_curve.output_values[mid_idx]


class TestApplyAdjustments:
    """Tests for applying multiple adjustments."""

    def test_apply_adjustments_sequence(self, linear_curve):
        """Test applying multiple adjustments in sequence."""
        modifier = CurveModifier()
        adjustments = [
            CurveAdjustment(AdjustmentType.BRIGHTNESS, 0.1),
            CurveAdjustment(AdjustmentType.CONTRAST, 0.2),
            CurveAdjustment(AdjustmentType.GAMMA, 1.2),
        ]

        result = modifier.apply_adjustments(linear_curve, adjustments)

        # Result should be different from original
        assert not np.allclose(result.output_values, linear_curve.output_values)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_adjust_curve(self, linear_curve):
        """Test adjust_curve convenience function."""
        result = adjust_curve(
            linear_curve,
            brightness=0.1,
            contrast=0.1,
            gamma=1.1,
        )

        assert not np.allclose(result.output_values, linear_curve.output_values)

    def test_smooth_curve(self, noisy_curve):
        """Test smooth_curve convenience function."""
        result = smooth_curve(noisy_curve, strength=0.5, method="gaussian")

        orig_rough = np.std(np.diff(np.diff(noisy_curve.output_values)))
        new_rough = np.std(np.diff(np.diff(result.output_values)))
        assert new_rough < orig_rough

    def test_blend_curves(self, linear_curve, nonlinear_curve):
        """Test blend_curves convenience function."""
        result = blend_curves(linear_curve, nonlinear_curve, weight=0.5, mode="weighted")

        assert len(result.output_values) > 0
        assert all(0 <= v <= 1 for v in result.output_values)
