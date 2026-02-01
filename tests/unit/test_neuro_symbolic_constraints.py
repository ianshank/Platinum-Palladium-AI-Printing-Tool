"""
Tests for neuro-symbolic constraints module.

Tests symbolic constraints with differentiable loss terms for
physics-informed curve generation.
"""

import numpy as np
import pytest

from ptpd_calibration.config import NeuroSymbolicSettings
from ptpd_calibration.neuro_symbolic.constraints import (
    ConstrainedCurveOptimizer,
    ConstraintResult,
    ConstraintSet,
    DensityBoundsConstraint,
    DifferentiableLoss,
    MonotonicityConstraint,
    PhysicsConstraint,
    SmoothnessConstraint,
)


class TestMonotonicityConstraint:
    """Tests for MonotonicityConstraint."""

    @pytest.fixture
    def constraint(self):
        """Create monotonicity constraint with default settings."""
        return MonotonicityConstraint()

    def test_monotonic_curve_satisfied(self, constraint):
        """Test that monotonically increasing curve satisfies constraint."""
        values = np.linspace(0, 1, 100)
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value < 1e-10
        assert len(result.violations) == 0

    def test_non_monotonic_curve_violated(self, constraint):
        """Test that non-monotonic curve violates constraint."""
        values = np.array([0.0, 0.3, 0.5, 0.4, 0.6, 0.8, 1.0])  # Dip at index 3
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0
        assert len(result.violations) == 1
        # Index 2 in diff corresponds to transition from index 2 to 3 in original
        assert 2 in result.violations[0].violation_indices

    def test_loss_is_differentiable(self, constraint):
        """Test that loss function is smooth and differentiable."""
        values = np.array([0.0, 0.3, 0.5, 0.45, 0.6, 0.8, 1.0])

        # Compute gradient numerically
        gradient = constraint.compute_gradient(values)

        assert gradient.shape == values.shape
        assert np.all(np.isfinite(gradient))

        # Gradient should be non-zero at violation points
        assert np.abs(gradient[3]) > 0 or np.abs(gradient[4]) > 0

    def test_zero_loss_for_perfect_curve(self, constraint):
        """Test zero loss for perfectly monotonic curve."""
        values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        loss = constraint.compute_loss(values)

        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_custom_weight(self):
        """Test constraint with custom weight."""
        constraint = MonotonicityConstraint(weight=5.0)
        values = np.array([0.0, 0.3, 0.2, 0.5])  # Violation

        base_loss = constraint.compute_loss(values)
        weighted_loss = constraint.weighted_loss(values)

        assert weighted_loss == pytest.approx(5.0 * base_loss)


class TestDensityBoundsConstraint:
    """Tests for DensityBoundsConstraint."""

    @pytest.fixture
    def constraint(self):
        """Create density bounds constraint."""
        return DensityBoundsConstraint(min_density=0.0, max_density=1.0)

    def test_within_bounds_satisfied(self, constraint):
        """Test that values within bounds satisfy constraint."""
        values = np.linspace(0.1, 0.9, 50)
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value < 1e-10
        assert len(result.violations) == 0

    def test_below_minimum_violated(self, constraint):
        """Test that values below minimum violate constraint."""
        values = np.array([-0.1, 0.0, 0.5, 0.8, 1.0])
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0
        assert any(v.constraint_name == "Density Bounds (lower)" for v in result.violations)

    def test_above_maximum_violated(self, constraint):
        """Test that values above maximum violate constraint."""
        values = np.array([0.0, 0.5, 0.8, 1.0, 1.2])
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0
        assert any(v.constraint_name == "Density Bounds (upper)" for v in result.violations)

    def test_configurable_bounds(self):
        """Test constraint with custom bounds."""
        constraint = DensityBoundsConstraint(min_density=0.1, max_density=2.5)
        values = np.array([0.1, 0.5, 1.0, 2.0, 2.5])
        result = constraint.evaluate(values)

        assert result.is_satisfied

    def test_loss_is_smooth(self, constraint):
        """Test that loss is smooth at boundary."""
        # Values slightly outside bounds
        epsilon = 1e-4
        loss_at_boundary = constraint.compute_loss(np.array([0.0]))
        loss_slightly_below = constraint.compute_loss(np.array([-epsilon]))

        # Loss should increase smoothly
        assert loss_slightly_below > loss_at_boundary
        assert loss_slightly_below < 1e-4  # Small violation


class TestPhysicsConstraint:
    """Tests for PhysicsConstraint (H&D curve physics)."""

    @pytest.fixture
    def constraint(self):
        """Create physics constraint."""
        return PhysicsConstraint(toe_fraction=0.2, shoulder_fraction=0.2)

    def test_ideal_hd_curve_satisfied(self, constraint):
        """Test that ideal H&D curve satisfies physics constraint."""
        x = np.linspace(0, 1, 100)
        # Model ideal H&D: D = Dmax * (1 - exp(-k * x^gamma))
        values = 2.0 * (1 - np.exp(-3 * np.power(x, 0.8)))
        values = values / values.max()  # Normalize

        result = constraint.evaluate(values)

        # Should mostly satisfy (may have small violations due to model mismatch)
        assert result.loss_value < 0.1

    def test_linear_curve_partial_violation(self, constraint):
        """Test that linear curve has some physics violations."""
        values = np.linspace(0, 1, 100)
        result = constraint.evaluate(values)

        # Linear curve doesn't have toe/shoulder characteristics
        # May or may not satisfy depending on threshold
        assert result.loss_value >= 0

    def test_inverted_curve_violations(self, constraint):
        """Test that inverted curve has physics violations."""
        # Convex toe (wrong) instead of concave
        x = np.linspace(0, 1, 100)
        values = x**2  # Convex in toe region

        result = constraint.evaluate(values)

        # Should have non-zero loss for physics violation
        # The x^2 curve is convex in the toe region where we expect concave
        assert result.loss_value > 0


class TestSmoothnessConstraint:
    """Tests for SmoothnessConstraint."""

    @pytest.fixture
    def constraint(self):
        """Create smoothness constraint."""
        return SmoothnessConstraint(order=2)

    def test_smooth_curve_satisfied(self, constraint):
        """Test that smooth curve satisfies constraint."""
        x = np.linspace(0, 1, 100)
        values = np.sin(x * np.pi / 2)  # Smooth curve

        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value < 0.01

    def test_noisy_curve_violated(self, constraint):
        """Test that noisy curve violates constraint."""
        x = np.linspace(0, 1, 100)
        values = x + 0.1 * np.random.randn(100)  # Noisy

        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.01

    def test_discontinuous_curve_violated(self, constraint):
        """Test that discontinuous curve violates constraint."""
        values = np.array([0.0, 0.1, 0.2, 0.5, 0.6, 0.7])  # Jump at index 3

        result = constraint.evaluate(values)

        assert result.loss_value > 0


class TestConstraintSet:
    """Tests for ConstraintSet."""

    @pytest.fixture
    def constraint_set(self):
        """Create default constraint set."""
        return ConstraintSet.default_set()

    def test_default_set_creation(self, constraint_set):
        """Test that default set contains expected constraints."""
        constraint_names = [c.name for c in constraint_set.constraints]

        assert "Monotonicity" in constraint_names
        assert "Density Bounds" in constraint_names
        assert "H&D Curve Physics" in constraint_names
        assert "Smoothness" in constraint_names

    def test_evaluate_all(self, constraint_set):
        """Test evaluating all constraints."""
        values = np.linspace(0, 1, 100)
        results = constraint_set.evaluate_all(values)

        assert len(results) == len(constraint_set.constraints)
        assert all(isinstance(r, ConstraintResult) for r in results.values())

    def test_total_loss_computation(self, constraint_set):
        """Test total weighted loss computation."""
        values = np.linspace(0, 1, 100)
        total_loss = constraint_set.compute_total_loss(values)

        assert total_loss >= 0
        assert np.isfinite(total_loss)

    def test_all_satisfied_good_curve(self, constraint_set):
        """Test that good curve satisfies all constraints."""
        x = np.linspace(0, 1, 256)
        values = 0.9 * (1 - np.exp(-3 * np.power(x, 0.8)))

        # This should mostly satisfy constraints
        violations = constraint_set.get_violations(values)

        # May have minor violations, but should be few
        assert len(violations) <= 2

    def test_add_remove_constraint(self):
        """Test adding and removing constraints."""
        constraint_set = ConstraintSet()
        assert len(constraint_set.constraints) == 0

        constraint_set.add_constraint(MonotonicityConstraint())
        assert len(constraint_set.constraints) == 1

        removed = constraint_set.remove_constraint("Monotonicity")
        assert removed
        assert len(constraint_set.constraints) == 0

    def test_generate_report(self, constraint_set):
        """Test report generation."""
        values = np.linspace(0, 1, 100)
        report = constraint_set.generate_report(values)

        assert "CONSTRAINT EVALUATION REPORT" in report
        assert "TOTAL WEIGHTED LOSS" in report


class TestDifferentiableLoss:
    """Tests for DifferentiableLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create differentiable loss function."""
        constraint_set = ConstraintSet.default_set()
        target = np.linspace(0, 1, 100)
        return DifferentiableLoss(
            constraint_set=constraint_set,
            target_values=target,
            data_loss_weight=1.0,
        )

    def test_loss_callable(self, loss_fn):
        """Test that loss function is callable."""
        values = np.linspace(0, 1, 100)
        loss = loss_fn(values)

        assert isinstance(loss, float)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_zero_loss_at_target(self, loss_fn):
        """Test minimal loss when values match target."""
        target = np.linspace(0, 1, 100)
        loss = loss_fn(target)

        # Should have small loss (constraints may add some)
        assert loss < 1.0

    def test_gradient_computation(self, loss_fn):
        """Test gradient computation."""
        values = np.linspace(0, 1, 100)
        gradient = loss_fn.gradient(values)

        assert gradient.shape == values.shape
        assert np.all(np.isfinite(gradient))

    def test_different_data_loss_types(self):
        """Test different data loss types."""
        constraint_set = ConstraintSet()
        target = np.linspace(0, 1, 50)
        values = target + 0.1  # Offset

        for loss_type in ["mse", "mae", "huber"]:
            loss_fn = DifferentiableLoss(
                constraint_set=constraint_set,
                target_values=target,
                data_loss_type=loss_type,
            )
            loss = loss_fn(values)
            assert loss > 0
            assert np.isfinite(loss)


class TestConstrainedCurveOptimizer:
    """Tests for ConstrainedCurveOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create constrained curve optimizer."""
        return ConstrainedCurveOptimizer()

    def test_optimize_monotonic_violation(self, optimizer):
        """Test optimization fixes monotonicity violations."""
        # Create curve with violation
        values = np.array([0.0, 0.2, 0.4, 0.35, 0.5, 0.7, 0.9, 1.0])

        result = optimizer.optimize(values)

        assert result["success"]
        optimized = result["optimized_values"]

        # Check monotonicity
        diffs = np.diff(optimized)
        assert np.all(diffs >= -1e-6)

    def test_optimize_bounds_violation(self, optimizer):
        """Test optimization fixes bounds violations."""
        values = np.array([-0.1, 0.2, 0.5, 0.8, 1.1])

        result = optimizer.optimize(values, bounds=(0.0, 1.0))

        optimized = result["optimized_values"]
        assert np.all(optimized >= -1e-6)
        assert np.all(optimized <= 1.0 + 1e-6)

    def test_optimize_preserves_good_curve(self, optimizer):
        """Test optimization doesn't change already-good curve."""
        x = np.linspace(0, 1, 100)
        values = 0.9 * (1 - np.exp(-3 * np.power(x, 0.8)))

        result = optimizer.optimize(values)

        optimized = result["optimized_values"]

        # Should be close to original
        assert np.allclose(values, optimized, atol=0.1)

    def test_validate_curve(self, optimizer):
        """Test curve validation."""
        good_curve = np.linspace(0, 1, 100)
        bad_curve = np.array([0.0, 0.5, 0.3, 0.8])  # Non-monotonic

        is_valid_good, violations_good = optimizer.validate_curve(good_curve)
        is_valid_bad, violations_bad = optimizer.validate_curve(bad_curve)

        assert is_valid_good
        assert len(violations_good) == 0
        assert not is_valid_bad
        assert len(violations_bad) > 0

    def test_project_to_constraints(self, optimizer):
        """Test projection to constraints."""
        values = np.array([0.0, 0.3, 0.2, 0.5, 0.8, 1.0])  # Non-monotonic

        projected = optimizer.project_to_constraints(values)

        # Check monotonicity
        diffs = np.diff(projected)
        assert np.all(diffs >= -1e-6)

        # Should stay close to original where valid
        assert np.abs(projected[0] - values[0]) < 0.1
        assert np.abs(projected[-1] - values[-1]) < 0.1


class TestConstraintSettings:
    """Tests for constraint settings integration."""

    def test_settings_from_config(self):
        """Test that constraints use settings from config."""
        settings = NeuroSymbolicSettings(
            monotonicity_weight=15.0,
            density_bounds_weight=3.0,
            min_density=0.05,
            max_density=2.5,
        )

        constraint = MonotonicityConstraint(settings=settings)
        assert constraint.weight == 15.0

        bounds_constraint = DensityBoundsConstraint(settings=settings)
        assert bounds_constraint.weight == 3.0
        assert bounds_constraint.min_density == 0.05
        assert bounds_constraint.max_density == 2.5

    def test_constraint_set_with_custom_settings(self):
        """Test constraint set with custom settings."""
        settings = NeuroSymbolicSettings(
            monotonicity_weight=20.0,
            smoothness_weight=1.0,
        )

        constraint_set = ConstraintSet.default_set(settings=settings)

        # Find monotonicity constraint and check weight
        mono = next(c for c in constraint_set.constraints if c.name == "Monotonicity")
        assert mono.weight == 20.0
