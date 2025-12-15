"""
Curve manipulation unit tests.
"""

import pytest
import numpy as np
from typing import Any


class TestCurveGeneration:
    """Tests for curve generation functions."""

    def test_linear_curve_generation(self) -> None:
        """Test generation of linear (identity) curve."""
        steps = 21
        input_values = [i * 5 for i in range(steps)]
        output_values = input_values.copy()  # Identity curve

        assert len(input_values) == steps
        assert len(output_values) == steps
        assert input_values == output_values

    def test_contrast_curve_generation(self) -> None:
        """Test generation of contrast curve with S-curve shape."""
        steps = 21
        input_values = np.linspace(0, 100, steps)

        # Simple S-curve using tanh
        midpoint = 50
        slope = 0.1
        output_values = 50 + 50 * np.tanh(slope * (input_values - midpoint))

        # Verify S-curve properties
        assert output_values[0] < 10  # Shadow compression
        assert output_values[-1] > 90  # Highlight compression
        assert abs(output_values[steps // 2] - 50) < 5  # Midpoint near 50

    def test_linearization_curve_generation(
        self, sample_scan_measurements: list[dict[str, Any]]
    ) -> None:
        """Test generation of linearization curve from measurements."""
        measurements = sample_scan_measurements
        target_densities = [m["target_density"] for m in measurements]
        measured_densities = [m["measured_density"] for m in measurements]

        # Linearization inverts the measured response
        # Simple approximation for testing
        correction_factors = [
            t / m if m > 0 else 1 for t, m in zip(target_densities, measured_densities)
        ]

        assert len(correction_factors) == len(measurements)

    def test_curve_steps_configuration(self, mock_config: dict[str, Any]) -> None:
        """Test curve uses configured number of steps."""
        steps = mock_config["curves"]["default_steps"]
        input_values = list(range(0, 101, 100 // (steps - 1)))

        assert len(input_values) == steps


class TestCurveModification:
    """Tests for curve modification operations."""

    def test_curve_smoothing(self, sample_curve_data: dict[str, Any]) -> None:
        """Test curve smoothing operation."""
        output_values = np.array(sample_curve_data["output_values"], dtype=float)

        # Add some noise
        noisy_values = output_values + np.random.normal(0, 2, len(output_values))

        # Simple moving average smoothing
        window_size = 3
        smoothed = np.convolve(
            noisy_values, np.ones(window_size) / window_size, mode="same"
        )

        # Smoothed values should be less noisy
        original_variance = np.var(np.diff(noisy_values))
        smoothed_variance = np.var(np.diff(smoothed))

        assert smoothed_variance <= original_variance

    def test_curve_inversion(self, sample_curve_data: dict[str, Any]) -> None:
        """Test curve inversion operation."""
        output_values = np.array(sample_curve_data["output_values"])
        max_value = 100

        inverted = max_value - output_values

        assert inverted[0] == max_value - output_values[0]
        assert inverted[-1] == max_value - output_values[-1]
        assert np.all(inverted >= 0)

    def test_curve_contrast_adjustment(self, sample_curve_data: dict[str, Any]) -> None:
        """Test contrast adjustment of curve."""
        output_values = np.array(sample_curve_data["output_values"], dtype=float)
        contrast_factor = 1.5
        midpoint = 50

        # Contrast adjustment formula
        adjusted = midpoint + (output_values - midpoint) * contrast_factor
        adjusted = np.clip(adjusted, 0, 100)

        # Higher contrast means more separation from midpoint
        original_range = np.max(output_values) - np.min(output_values)
        adjusted_range = np.max(adjusted) - np.min(adjusted)

        assert adjusted_range >= original_range * 0.9  # Clipping may reduce range

    def test_curve_brightness_adjustment(
        self, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test brightness adjustment of curve."""
        output_values = np.array(sample_curve_data["output_values"], dtype=float)
        brightness_offset = 10

        adjusted = np.clip(output_values + brightness_offset, 0, 100)

        # Average should increase (or be clipped at max)
        original_mean = np.mean(output_values)
        adjusted_mean = np.mean(adjusted)

        assert adjusted_mean >= original_mean

    def test_curve_gamma_adjustment(self, sample_curve_data: dict[str, Any]) -> None:
        """Test gamma adjustment of curve."""
        output_values = np.array(sample_curve_data["output_values"], dtype=float)
        gamma = 2.2

        # Normalize to 0-1, apply gamma, scale back
        normalized = output_values / 100
        gamma_adjusted = np.power(normalized, 1 / gamma)
        adjusted = gamma_adjusted * 100

        # Gamma < 1 brightens midtones
        assert np.mean(adjusted) > np.mean(output_values)


class TestCurveBlending:
    """Tests for curve blending operations."""

    def test_equal_weight_blending(self) -> None:
        """Test blending two curves with equal weights."""
        curve1 = np.array([0, 25, 50, 75, 100], dtype=float)
        curve2 = np.array([0, 30, 60, 80, 100], dtype=float)
        weights = [0.5, 0.5]

        blended = curve1 * weights[0] + curve2 * weights[1]

        expected = np.array([0, 27.5, 55, 77.5, 100])
        np.testing.assert_array_almost_equal(blended, expected)

    def test_weighted_blending(self) -> None:
        """Test blending with unequal weights."""
        curve1 = np.array([0, 25, 50, 75, 100], dtype=float)
        curve2 = np.array([0, 35, 70, 85, 100], dtype=float)
        weights = [0.7, 0.3]

        blended = curve1 * weights[0] + curve2 * weights[1]

        # Result should be closer to curve1
        assert np.mean(np.abs(blended - curve1)) < np.mean(np.abs(blended - curve2))

    def test_blend_multiple_curves(self) -> None:
        """Test blending more than two curves."""
        curves = [
            np.array([0, 20, 40, 60, 100], dtype=float),
            np.array([0, 30, 50, 70, 100], dtype=float),
            np.array([0, 25, 55, 80, 100], dtype=float),
        ]
        weights = [0.33, 0.33, 0.34]

        blended = sum(c * w for c, w in zip(curves, weights))

        # Should be somewhere in the middle
        assert blended[2] > 40 and blended[2] < 60

    def test_blend_weights_sum_to_one(self) -> None:
        """Test that blend weights should sum to 1."""
        weights = [0.3, 0.5, 0.2]
        assert abs(sum(weights) - 1.0) < 0.001


class TestCurveValidation:
    """Tests for curve validation."""

    def test_monotonically_increasing_curve(
        self, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test curve is monotonically increasing."""
        output_values = sample_curve_data["output_values"]

        for i in range(1, len(output_values)):
            assert (
                output_values[i] >= output_values[i - 1]
            ), f"Curve not monotonic at index {i}"

    def test_curve_range_validation(self, sample_curve_data: dict[str, Any]) -> None:
        """Test curve values are within valid range."""
        output_values = sample_curve_data["output_values"]

        assert all(0 <= v <= 100 for v in output_values)

    def test_curve_length_matches_input(
        self, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test output length matches input length."""
        input_values = sample_curve_data["input_values"]
        output_values = sample_curve_data["output_values"]

        assert len(input_values) == len(output_values)

    def test_curve_metadata_present(self, sample_curve_data: dict[str, Any]) -> None:
        """Test curve has required metadata."""
        assert "metadata" in sample_curve_data
        assert "paper_type" in sample_curve_data["metadata"]


class TestCurveInterpolation:
    """Tests for curve interpolation."""

    def test_linear_interpolation(self) -> None:
        """Test linear interpolation between points."""
        x = np.array([0, 50, 100])
        y = np.array([0, 40, 100])

        # Interpolate at x=25
        interp_x = 25
        idx = np.searchsorted(x, interp_x)
        t = (interp_x - x[idx - 1]) / (x[idx] - x[idx - 1])
        interp_y = y[idx - 1] + t * (y[idx] - y[idx - 1])

        assert interp_y == 20  # Linear interpolation between 0 and 40

    def test_cubic_interpolation_smoothness(self) -> None:
        """Test cubic interpolation produces smooth results."""
        from scipy.interpolate import CubicSpline

        x = np.array([0, 25, 50, 75, 100])
        y = np.array([0, 20, 50, 80, 100])

        cs = CubicSpline(x, y)

        # Generate many points
        x_dense = np.linspace(0, 100, 100)
        y_dense = cs(x_dense)

        # Check smoothness via second derivative
        d2 = np.diff(np.diff(y_dense))
        assert np.max(np.abs(d2)) < 1  # Second derivative should be small
