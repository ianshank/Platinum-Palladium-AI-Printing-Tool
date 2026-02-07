"""Tests for auto-linearization module."""

import numpy as np
import pytest

from ptpd_calibration.curves import (
    AutoLinearizer,
    LinearizationConfig,
    LinearizationMethod,
    LinearizationResult,
    TargetResponse,
)


@pytest.fixture
def linearizer():
    """Create default linearizer."""
    return AutoLinearizer()


@pytest.fixture
def typical_densities():
    """Typical measured densities from a step wedge."""
    return [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]


@pytest.fixture
def linear_densities():
    """Perfectly linear densities."""
    return [i / 10.0 for i in range(11)]


@pytest.fixture
def nonlinear_densities():
    """Non-linear (S-curve) densities."""
    x = np.linspace(0, 1, 11)
    # S-curve response
    return (3 * x**2 - 2 * x**3).tolist()


class TestAutoLinearizer:
    """Test auto-linearizer functionality."""

    def test_linearize_basic(self, linearizer, typical_densities):
        """Test basic linearization."""
        result = linearizer.linearize(typical_densities)

        assert isinstance(result, LinearizationResult)
        assert result.curve is not None
        assert len(result.curve.input_values) > 0
        assert len(result.curve.output_values) > 0

    def test_linearize_with_name(self, linearizer, typical_densities):
        """Test linearization with custom name."""
        result = linearizer.linearize(typical_densities, curve_name="My Test Curve")

        assert result.curve.name == "My Test Curve"

    def test_linearize_direct_inversion(self, linearizer, typical_densities):
        """Test direct inversion method."""
        result = linearizer.linearize(
            typical_densities,
            method=LinearizationMethod.DIRECT_INVERSION,
        )

        assert result.method_used == LinearizationMethod.DIRECT_INVERSION
        assert "Direct inversion" in " ".join(result.notes)

    def test_linearize_spline_fit(self, linearizer, typical_densities):
        """Test spline fit method."""
        result = linearizer.linearize(
            typical_densities,
            method=LinearizationMethod.SPLINE_FIT,
        )

        assert result.method_used == LinearizationMethod.SPLINE_FIT

    def test_linearize_polynomial_fit(self, linearizer, typical_densities):
        """Test polynomial fit method."""
        result = linearizer.linearize(
            typical_densities,
            method=LinearizationMethod.POLYNOMIAL_FIT,
        )

        assert result.method_used == LinearizationMethod.POLYNOMIAL_FIT

    def test_linearize_iterative(self, linearizer, typical_densities):
        """Test iterative method."""
        result = linearizer.linearize(
            typical_densities,
            method=LinearizationMethod.ITERATIVE,
        )

        assert result.method_used == LinearizationMethod.ITERATIVE

    def test_linearize_hybrid(self, linearizer, typical_densities):
        """Test hybrid method."""
        result = linearizer.linearize(
            typical_densities,
            method=LinearizationMethod.HYBRID,
        )

        assert result.method_used == LinearizationMethod.HYBRID

    def test_target_linear(self, linearizer, typical_densities):
        """Test linear target response."""
        result = linearizer.linearize(
            typical_densities,
            target=TargetResponse.LINEAR,
        )

        assert result.target_response == TargetResponse.LINEAR

    def test_target_gamma_18(self, linearizer, typical_densities):
        """Test gamma 1.8 target response."""
        result = linearizer.linearize(
            typical_densities,
            target=TargetResponse.GAMMA_18,
        )

        assert result.target_response == TargetResponse.GAMMA_18

    def test_target_gamma_22(self, linearizer, typical_densities):
        """Test gamma 2.2 target response."""
        result = linearizer.linearize(
            typical_densities,
            target=TargetResponse.GAMMA_22,
        )

        assert result.target_response == TargetResponse.GAMMA_22

    def test_target_paper_white(self, linearizer, typical_densities):
        """Test paper white target response."""
        result = linearizer.linearize(
            typical_densities,
            target=TargetResponse.PAPER_WHITE,
        )

        assert result.target_response == TargetResponse.PAPER_WHITE

    def test_target_perceptual(self, linearizer, typical_densities):
        """Test perceptual target response."""
        result = linearizer.linearize(
            typical_densities,
            target=TargetResponse.PERCEPTUAL,
        )

        assert result.target_response == TargetResponse.PERCEPTUAL

    def test_curve_monotonicity(self, linearizer, typical_densities):
        """Test that output curve is monotonic."""
        result = linearizer.linearize(typical_densities)

        outputs = result.curve.output_values
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1], "Curve should be monotonically increasing"

    def test_curve_bounds(self, linearizer, typical_densities):
        """Test that curve values are bounded 0-1."""
        result = linearizer.linearize(typical_densities)

        for val in result.curve.output_values:
            assert 0 <= val <= 1, "All values should be in 0-1 range"

    def test_endpoint_preservation(self, linearizer, typical_densities):
        """Test that endpoints are preserved."""
        config = LinearizationConfig(preserve_endpoints=True)
        linearizer = AutoLinearizer(config)
        result = linearizer.linearize(typical_densities)

        assert result.curve.output_values[0] == 0.0
        assert result.curve.output_values[-1] == 1.0

    def test_error_metrics(self, linearizer, typical_densities):
        """Test error metrics are computed."""
        result = linearizer.linearize(typical_densities)

        assert result.residual_error >= 0
        assert result.max_deviation >= 0

    def test_refine_curve(self, linearizer, typical_densities):
        """Test curve refinement."""
        # First linearization
        result1 = linearizer.linearize(typical_densities)

        # Simulate new measurements (slightly off)
        new_measurements = [d * 1.05 for d in typical_densities]

        # Refine
        result2 = linearizer.refine_curve(result1.curve, new_measurements)

        assert result2.curve is not None
        assert "refined" in result2.curve.name.lower()

    def test_custom_config(self):
        """Test custom configuration."""
        config = LinearizationConfig(
            method=LinearizationMethod.POLYNOMIAL_FIT,
            target=TargetResponse.GAMMA_22,
            output_points=128,
            smoothing=0.2,
            polynomial_degree=7,
        )
        linearizer = AutoLinearizer(config)

        densities = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        result = linearizer.linearize(densities)

        assert result.method_used == LinearizationMethod.POLYNOMIAL_FIT

    def test_to_dict(self, linearizer, typical_densities):
        """Test result conversion to dictionary."""
        result = linearizer.linearize(typical_densities)
        d = result.to_dict()

        assert "curve_name" in d
        assert "method" in d
        assert "target" in d
        assert "residual_error" in d

    def test_get_methods(self):
        """Test getting available methods."""
        methods = AutoLinearizer.get_methods()

        assert len(methods) >= 5
        assert any("spline" in m[0].lower() for m in methods)

    def test_get_targets(self):
        """Test getting available targets."""
        targets = AutoLinearizer.get_targets()

        assert len(targets) >= 4
        assert any("linear" in t[0].lower() for t in targets)

    def test_measured_densities_stored(self, linearizer, typical_densities):
        """Test that measured densities are stored in result."""
        result = linearizer.linearize(typical_densities)

        assert result.measured_densities == typical_densities

    def test_target_densities_computed(self, linearizer, typical_densities):
        """Test that target densities are computed."""
        result = linearizer.linearize(typical_densities)

        assert len(result.target_densities) == len(typical_densities)

    def test_few_points(self, linearizer):
        """Test with minimal data points."""
        densities = [0.1, 0.5, 0.9]
        result = linearizer.linearize(densities)

        assert result.curve is not None

    def test_many_points(self, linearizer):
        """Test with many data points."""
        densities = [i / 40.0 for i in range(41)]
        result = linearizer.linearize(densities)

        assert result.curve is not None
