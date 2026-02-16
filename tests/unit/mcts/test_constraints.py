"""
Tests for MCTS neuro-symbolic constraints.

Tests cover:
- Individual constraint evaluation (bounds, soft penalties)
- ActionPruner filtering and scoring
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from ptpd_calibration.mcts.config import (
    DEFAULT_PARAMETER_RANGES,
    MCTSSettings,
    PhysicsConstants,
)
from ptpd_calibration.mcts.constraints import (
    ActionPruner,
    CoatingWeightConstraint,
    DeveloperTemperatureConstraint,
    ExposureTimeConstraint,
    FOConcentrationConstraint,
    HumidityConstraint,
    MetalRatioConstraint,
    ParameterBoundsConstraint,
)
from ptpd_calibration.mcts.types import CalibrationAction, CalibrationState


class TestParameterBoundsConstraint:
    """Test hard bounds constraint for parameters."""

    @pytest.fixture
    def metal_ratio_constraint(self) -> ParameterBoundsConstraint:
        """Create metal ratio bounds constraint."""
        return ParameterBoundsConstraint("metal_ratio")

    def test_in_bounds_satisfied(self, metal_ratio_constraint: ParameterBoundsConstraint) -> None:
        """Test that in-bounds values satisfy constraint."""
        values = np.array([0.5])
        result = metal_ratio_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0
        assert len(result.violations) == 0
        assert "within bounds" in result.explanation

    def test_below_min_violated(self, metal_ratio_constraint: ParameterBoundsConstraint) -> None:
        """Test that values below minimum violate constraint."""
        values = np.array([-0.1])
        result = metal_ratio_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert len(result.violations) == 1
        assert "below minimum" in result.violations[0].description
        assert result.violations[0].violation_magnitude > 0

    def test_above_max_violated(self, metal_ratio_constraint: ParameterBoundsConstraint) -> None:
        """Test that values above maximum violate constraint."""
        values = np.array([1.5])
        result = metal_ratio_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert len(result.violations) == 1
        assert "above maximum" in result.violations[0].description

    def test_edge_cases(self, metal_ratio_constraint: ParameterBoundsConstraint) -> None:
        """Test edge case values (exactly at bounds)."""
        param_range = DEFAULT_PARAMETER_RANGES["metal_ratio"]

        # Exactly at minimum
        result_min = metal_ratio_constraint.evaluate(np.array([param_range.min_value]))
        assert result_min.is_satisfied
        assert result_min.loss_value == 0.0

        # Exactly at maximum
        result_max = metal_ratio_constraint.evaluate(np.array([param_range.max_value]))
        assert result_max.is_satisfied
        assert result_max.loss_value == 0.0

    def test_loss_increases_with_violation(
        self, metal_ratio_constraint: ParameterBoundsConstraint
    ) -> None:
        """Test that loss increases quadratically with violation magnitude."""
        # Small violation
        loss_small = metal_ratio_constraint.compute_loss(np.array([-0.05]))

        # Large violation
        loss_large = metal_ratio_constraint.compute_loss(np.array([-0.2]))

        assert loss_large > loss_small
        # Quadratic: (0.2)^2 / (0.05)^2 = 16
        assert loss_large / loss_small == pytest.approx(16.0, rel=0.01)

    def test_all_parameters_have_ranges(self) -> None:
        """Test that all standard parameters have defined ranges."""
        expected_params = [
            "metal_ratio",
            "coating_weight",
            "ferric_oxalate_pct",
            "exposure_time",
            "developer_temp",
            "humidity",
        ]

        for param in expected_params:
            constraint = ParameterBoundsConstraint(param)
            assert constraint.range is not None
            assert constraint.range.min_value < constraint.range.max_value

    def test_invalid_input_length(self, metal_ratio_constraint: ParameterBoundsConstraint) -> None:
        """Test handling of invalid input array length."""
        # Empty array
        result_empty = metal_ratio_constraint.evaluate(np.array([]))
        assert not result_empty.is_satisfied
        assert result_empty.loss_value > 0

        # Multiple values
        result_multi = metal_ratio_constraint.evaluate(np.array([0.5, 0.6]))
        assert not result_multi.is_satisfied


class TestFOConcentrationConstraint:
    """Test ferric oxalate concentration sweet spot constraint."""

    @pytest.fixture
    def fo_constraint(self) -> FOConcentrationConstraint:
        """Create FO concentration constraint."""
        return FOConcentrationConstraint()

    def test_in_sweet_spot_satisfied(self, fo_constraint: FOConcentrationConstraint) -> None:
        """Test that sweet spot values satisfy constraint."""
        values = np.array([20.0])  # Middle of sweet spot
        result = fo_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0
        assert len(result.violations) == 0

    def test_below_sweet_spot_soft_penalty(
        self, fo_constraint: FOConcentrationConstraint
    ) -> None:
        """Test soft penalty for values below sweet spot."""
        values = np.array([16.0])  # Below sweet spot, above hard minimum
        result = fo_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "low sensitivity" in result.violations[0].description

    def test_above_sweet_spot_soft_penalty(
        self, fo_constraint: FOConcentrationConstraint
    ) -> None:
        """Test soft penalty for values above sweet spot."""
        values = np.array([26.0])  # Above sweet spot, below hard maximum
        result = fo_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "coating issues" in result.violations[0].description

    def test_penalty_increases_with_deviation(
        self, fo_constraint: FOConcentrationConstraint
    ) -> None:
        """Test that penalty increases with distance from sweet spot."""
        # Small deviation
        loss_small = fo_constraint.compute_loss(np.array([17.5]))

        # Large deviation
        loss_large = fo_constraint.compute_loss(np.array([15.0]))

        assert loss_large > loss_small

    def test_sweet_spot_edges(self, fo_constraint: FOConcentrationConstraint) -> None:
        """Test values exactly at sweet spot boundaries."""
        # Lower edge
        result_low = fo_constraint.evaluate(np.array([fo_constraint.sweet_spot_min]))
        assert result_low.is_satisfied

        # Upper edge
        result_high = fo_constraint.evaluate(np.array([fo_constraint.sweet_spot_max]))
        assert result_high.is_satisfied


class TestMetalRatioConstraint:
    """Test metal ratio blending constraint."""

    @pytest.fixture
    def metal_constraint(self) -> MetalRatioConstraint:
        """Create metal ratio constraint."""
        return MetalRatioConstraint()

    def test_typical_blend_satisfied(self, metal_constraint: MetalRatioConstraint) -> None:
        """Test that typical blend ratios satisfy constraint."""
        values = np.array([0.5])  # 50/50 blend
        result = metal_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_pd_heavy_soft_penalty(self, metal_constraint: MetalRatioConstraint) -> None:
        """Test soft penalty for Pd-heavy (low ratio) blends."""
        values = np.array([0.1])  # Very Pd-heavy
        result = metal_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "Pd-heavy" in result.violations[0].description

    def test_pt_heavy_soft_penalty(self, metal_constraint: MetalRatioConstraint) -> None:
        """Test soft penalty for Pt-heavy (high ratio) blends."""
        values = np.array([0.9])  # Very Pt-heavy
        result = metal_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "Pt-heavy" in result.violations[0].description

    def test_pure_metals_have_penalty(self, metal_constraint: MetalRatioConstraint) -> None:
        """Test that pure metals (0.0 or 1.0) have soft penalties."""
        # Pure palladium
        loss_pd = metal_constraint.compute_loss(np.array([0.0]))
        assert loss_pd > 0.0

        # Pure platinum
        loss_pt = metal_constraint.compute_loss(np.array([1.0]))
        assert loss_pt > 0.0


class TestExposureTimeConstraint:
    """Test exposure time practicality constraint."""

    @pytest.fixture
    def exposure_constraint(self) -> ExposureTimeConstraint:
        """Create exposure time constraint."""
        return ExposureTimeConstraint()

    def test_practical_time_satisfied(self, exposure_constraint: ExposureTimeConstraint) -> None:
        """Test that practical exposure times satisfy constraint."""
        values = np.array([120.0])  # 2 minutes
        result = exposure_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_very_short_soft_penalty(self, exposure_constraint: ExposureTimeConstraint) -> None:
        """Test soft penalty for very short exposures."""
        values = np.array([40.0])  # Below practical minimum
        result = exposure_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "not reach full Dmax" in result.violations[0].description

    def test_very_long_soft_penalty(self, exposure_constraint: ExposureTimeConstraint) -> None:
        """Test soft penalty for very long exposures."""
        values = np.array([450.0])  # Above practical maximum
        result = exposure_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "diminishing returns" in result.violations[0].description

    def test_penalty_relative_to_range(
        self, exposure_constraint: ExposureTimeConstraint
    ) -> None:
        """Test that penalty is relative to practical range."""
        # 10% below minimum
        loss_below = exposure_constraint.compute_loss(np.array([54.0]))

        # 10% above maximum
        loss_above = exposure_constraint.compute_loss(np.array([330.0]))

        # Should have similar magnitude penalties for similar relative deviations
        assert abs(loss_below - loss_above) / max(loss_below, loss_above) < 0.5


class TestDeveloperTemperatureConstraint:
    """Test developer temperature constraint."""

    @pytest.fixture
    def temp_constraint(self) -> DeveloperTemperatureConstraint:
        """Create developer temperature constraint."""
        return DeveloperTemperatureConstraint()

    def test_reference_temp_satisfied(
        self, temp_constraint: DeveloperTemperatureConstraint
    ) -> None:
        """Test that reference temperature satisfies constraint."""
        values = np.array([temp_constraint.reference_temp])
        result = temp_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_within_tolerance_satisfied(
        self, temp_constraint: DeveloperTemperatureConstraint
    ) -> None:
        """Test that values within tolerance satisfy constraint."""
        # Just below reference
        result_low = temp_constraint.evaluate(
            np.array([temp_constraint.reference_temp - temp_constraint.tolerance + 0.1])
        )
        assert result_low.is_satisfied

        # Just above reference
        result_high = temp_constraint.evaluate(
            np.array([temp_constraint.reference_temp + temp_constraint.tolerance - 0.1])
        )
        assert result_high.is_satisfied

    def test_beyond_tolerance_soft_penalty(
        self, temp_constraint: DeveloperTemperatureConstraint
    ) -> None:
        """Test soft penalty for temperatures beyond tolerance."""
        # Well below reference
        values_low = np.array([temp_constraint.reference_temp - temp_constraint.tolerance - 5.0])
        result_low = temp_constraint.evaluate(values_low)

        assert not result_low.is_satisfied
        assert result_low.loss_value > 0.0
        assert "below reference" in result_low.violations[0].description

        # Well above reference
        values_high = np.array([temp_constraint.reference_temp + temp_constraint.tolerance + 5.0])
        result_high = temp_constraint.evaluate(values_high)

        assert not result_high.is_satisfied
        assert result_high.loss_value > 0.0
        assert "above reference" in result_high.violations[0].description

    def test_uses_physics_constants(self) -> None:
        """Test that constraint uses PhysicsConstants for reference temp."""
        custom_physics = PhysicsConstants(dev_temp_reference=30.0)
        constraint = DeveloperTemperatureConstraint(physics_constants=custom_physics)

        assert constraint.reference_temp == 30.0

        # Should be satisfied at custom reference
        result = constraint.evaluate(np.array([30.0]))
        assert result.is_satisfied


class TestCoatingWeightConstraint:
    """Test coating weight constraint."""

    @pytest.fixture
    def coating_constraint(self) -> CoatingWeightConstraint:
        """Create coating weight constraint."""
        return CoatingWeightConstraint()

    def test_normal_weight_satisfied(self, coating_constraint: CoatingWeightConstraint) -> None:
        """Test that normal coating weights satisfy constraint."""
        values = np.array([1.5])  # Default value
        result = coating_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_insufficient_weight_violated(
        self, coating_constraint: CoatingWeightConstraint
    ) -> None:
        """Test that insufficient coating weight violates constraint."""
        values = np.array([0.3])  # Below minimum
        result = coating_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "insufficient" in result.violations[0].description

    def test_excessive_weight_violated(
        self, coating_constraint: CoatingWeightConstraint
    ) -> None:
        """Test that excessive coating weight violates constraint."""
        values = np.array([4.0])  # Above maximum
        result = coating_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "excessive" in result.violations[0].description

    def test_boundary_values(self, coating_constraint: CoatingWeightConstraint) -> None:
        """Test values exactly at boundaries."""
        param_range = DEFAULT_PARAMETER_RANGES["coating_weight"]

        # Minimum
        result_min = coating_constraint.evaluate(np.array([param_range.min_value]))
        assert result_min.is_satisfied

        # Maximum
        result_max = coating_constraint.evaluate(np.array([param_range.max_value]))
        assert result_max.is_satisfied


class TestHumidityConstraint:
    """Test humidity constraint."""

    @pytest.fixture
    def humidity_constraint(self) -> HumidityConstraint:
        """Create humidity constraint."""
        return HumidityConstraint()

    def test_optimal_humidity_satisfied(self, humidity_constraint: HumidityConstraint) -> None:
        """Test that optimal humidity satisfies constraint."""
        values = np.array([humidity_constraint.optimal_humidity])
        result = humidity_constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_within_tolerance_satisfied(self, humidity_constraint: HumidityConstraint) -> None:
        """Test that values within tolerance satisfy constraint."""
        # Just below optimal
        result_low = humidity_constraint.evaluate(
            np.array([humidity_constraint.optimal_humidity - humidity_constraint.tolerance + 1])
        )
        assert result_low.is_satisfied

        # Just above optimal
        result_high = humidity_constraint.evaluate(
            np.array([humidity_constraint.optimal_humidity + humidity_constraint.tolerance - 1])
        )
        assert result_high.is_satisfied

    def test_low_humidity_soft_penalty(self, humidity_constraint: HumidityConstraint) -> None:
        """Test soft penalty for low humidity."""
        values = np.array([35.0])  # Below optimal
        result = humidity_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "fast drying" in result.violations[0].description

    def test_high_humidity_soft_penalty(self, humidity_constraint: HumidityConstraint) -> None:
        """Test soft penalty for high humidity."""
        values = np.array([70.0])  # Above optimal
        result = humidity_constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert "slow drying" in result.violations[0].description

    def test_uses_physics_constants(self) -> None:
        """Test that constraint uses PhysicsConstants for optimal humidity."""
        custom_physics = PhysicsConstants(humidity_optimal=55.0)
        constraint = HumidityConstraint(physics_constants=custom_physics)

        assert constraint.optimal_humidity == 55.0

        # Should be satisfied at custom optimal
        result = constraint.evaluate(np.array([55.0]))
        assert result.is_satisfied


class TestActionPruner:
    """Test action pruning and scoring."""

    @pytest.fixture
    def pruner(self) -> ActionPruner:
        """Create action pruner with default constraints."""
        return ActionPruner.create_default_pruner()

    @pytest.fixture
    def sample_state(self) -> CalibrationState:
        """Create sample calibration state."""
        return CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["coating_weight", "ferric_oxalate_pct"],
            depth=1,
        )

    def test_prune_invalid_actions(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that invalid actions are pruned."""
        actions = [
            CalibrationAction(dimension="coating_weight", value=1.5, bin_index=10),  # Valid
            CalibrationAction(dimension="coating_weight", value=5.0, bin_index=20),  # Invalid (too high)
            CalibrationAction(dimension="coating_weight", value=0.2, bin_index=2),  # Invalid (too low)
            CalibrationAction(dimension="ferric_oxalate_pct", value=20.0, bin_index=10),  # Valid
        ]

        pruned = pruner.prune_actions(sample_state, actions)

        assert len(pruned) == 2
        assert all(a.value >= 0.5 and a.value <= 3.0 for a in pruned if a.dimension == "coating_weight")

    def test_keep_valid_actions(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that all valid actions are kept."""
        actions = [
            CalibrationAction(dimension="coating_weight", value=0.8, bin_index=5),
            CalibrationAction(dimension="coating_weight", value=1.5, bin_index=10),
            CalibrationAction(dimension="coating_weight", value=2.5, bin_index=18),
            CalibrationAction(dimension="ferric_oxalate_pct", value=18.0, bin_index=6),
            CalibrationAction(dimension="ferric_oxalate_pct", value=22.0, bin_index=14),
        ]

        pruned = pruner.prune_actions(sample_state, actions)

        assert len(pruned) == len(actions)
        assert pruned == actions

    def test_empty_action_list(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test handling of empty action list."""
        pruned = pruner.prune_actions(sample_state, [])

        assert len(pruned) == 0

    def test_score_valid_action_high(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that valid actions get high scores."""
        # Ideal FO concentration (in sweet spot)
        action = CalibrationAction(dimension="ferric_oxalate_pct", value=20.0, bin_index=10)

        score = pruner.score_action(sample_state, action)

        assert 0.9 <= score <= 1.0  # Should be near perfect

    def test_score_soft_violation_lower(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that soft violations get lower scores than ideal values."""
        # FO outside sweet spot but within hard bounds
        action_soft_violation = CalibrationAction(dimension="ferric_oxalate_pct", value=16.0, bin_index=2)
        action_ideal = CalibrationAction(dimension="ferric_oxalate_pct", value=20.0, bin_index=10)

        score_soft = pruner.score_action(sample_state, action_soft_violation)
        score_ideal = pruner.score_action(sample_state, action_ideal)

        # Soft violation should have lower score than ideal
        assert score_soft < score_ideal
        # Score should still be reasonably high due to bounds constraint dominance
        assert 0.9 <= score_soft < 1.0

    def test_score_extreme_violation_very_low(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that extreme violations get lower scores than typical values."""
        # Metal ratio at extreme (pure Pd) vs typical blend
        state = CalibrationState(remaining_dimensions=["metal_ratio"])
        action_extreme = CalibrationAction(dimension="metal_ratio", value=0.01, bin_index=0)
        action_typical = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)

        score_extreme = pruner.score_action(state, action_extreme)
        score_typical = pruner.score_action(state, action_typical)

        # Extreme should have lower score than typical
        assert score_extreme < score_typical
        # Extreme still scores high due to bounds constraint dominance (low soft weight)
        assert 0.95 <= score_extreme < 1.0

    def test_score_multiple_dimensions(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test scoring works for all parameter dimensions."""
        dimensions = [
            "metal_ratio",
            "coating_weight",
            "ferric_oxalate_pct",
            "exposure_time",
            "developer_temp",
            "humidity",
        ]

        for dim in dimensions:
            state = CalibrationState(remaining_dimensions=[dim])
            param_range = DEFAULT_PARAMETER_RANGES[dim]
            action = CalibrationAction(
                dimension=dim,
                value=param_range.default_value,
                bin_index=10,
            )

            score = pruner.score_action(state, action)

            # Default values should score well
            assert score > 0.5, f"Default value for {dim} scored poorly: {score}"

    def test_default_pruner_has_all_constraints(self) -> None:
        """Test that default pruner includes all standard constraints."""
        pruner = ActionPruner.create_default_pruner()

        assert len(pruner.constraints) >= 11  # 6 bounds + 5 soft constraints

        # Check for bounds constraints
        bounds_constraints = [
            c for c in pruner.constraints if isinstance(c, ParameterBoundsConstraint)
        ]
        assert len(bounds_constraints) == 6

        # Check for soft constraints
        assert any(isinstance(c, FOConcentrationConstraint) for c in pruner.constraints)
        assert any(isinstance(c, MetalRatioConstraint) for c in pruner.constraints)
        assert any(isinstance(c, ExposureTimeConstraint) for c in pruner.constraints)
        assert any(isinstance(c, DeveloperTemperatureConstraint) for c in pruner.constraints)
        assert any(isinstance(c, HumidityConstraint) for c in pruner.constraints)

    def test_custom_constraints(self, sample_state: CalibrationState) -> None:
        """Test pruner with custom constraint set."""
        # Only bounds constraints
        custom_constraints = [
            ParameterBoundsConstraint("metal_ratio"),
            ParameterBoundsConstraint("coating_weight"),
        ]

        pruner = ActionPruner(constraints=custom_constraints)

        assert len(pruner.constraints) == 2

        # Should still prune invalid actions
        actions = [
            CalibrationAction(dimension="coating_weight", value=1.5, bin_index=10),  # Valid
            CalibrationAction(dimension="coating_weight", value=10.0, bin_index=20),  # Invalid
        ]

        pruned = pruner.prune_actions(sample_state, actions)
        assert len(pruned) == 1

    def test_score_bounds_normalization(
        self, pruner: ActionPruner, sample_state: CalibrationState
    ) -> None:
        """Test that scores are always in [0, 1] range."""
        # Test extreme values for all dimensions
        dimensions = list(DEFAULT_PARAMETER_RANGES.keys())

        for dim in dimensions:
            state = CalibrationState(remaining_dimensions=[dim])
            param_range = DEFAULT_PARAMETER_RANGES[dim]

            # Test minimum, middle, maximum
            for value in [param_range.min_value, param_range.default_value, param_range.max_value]:
                action = CalibrationAction(dimension=dim, value=value, bin_index=10)
                score = pruner.score_action(state, action)

                assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {dim}={value}"

    def test_pruner_with_mcts_settings(self) -> None:
        """Test pruner initialization with MCTS settings."""
        settings = MCTSSettings(action_bins=31)
        pruner = ActionPruner(settings=settings)

        assert pruner.settings.action_bins == 31
        assert len(pruner.constraints) > 0


@pytest.mark.parametrize(
    "param_name,valid_value,invalid_low,invalid_high",
    [
        ("metal_ratio", 0.5, -0.1, 1.5),
        ("coating_weight", 1.5, 0.2, 5.0),
        ("ferric_oxalate_pct", 20.0, 10.0, 35.0),
        ("exposure_time", 180.0, 10.0, 800.0),
        ("developer_temp", 25.0, 10.0, 60.0),
        ("humidity", 50.0, 10.0, 95.0),
    ],
)
def test_all_parameters_bounds_enforcement(
    param_name: str,
    valid_value: float,
    invalid_low: float,
    invalid_high: float,
) -> None:
    """Parametrized test for bounds enforcement across all parameters."""
    constraint = ParameterBoundsConstraint(param_name)

    # Valid value passes
    result_valid = constraint.evaluate(np.array([valid_value]))
    assert result_valid.is_satisfied

    # Invalid low fails
    result_low = constraint.evaluate(np.array([invalid_low]))
    assert not result_low.is_satisfied
    assert result_low.loss_value > 0.0

    # Invalid high fails
    result_high = constraint.evaluate(np.array([invalid_high]))
    assert not result_high.is_satisfied
    assert result_high.loss_value > 0.0


def test_constraint_gradient_computation() -> None:
    """Test that constraints can compute numerical gradients."""
    constraint = ParameterBoundsConstraint("metal_ratio")

    # Gradient should be zero for in-bounds values
    values = np.array([0.5])
    gradient = constraint.compute_gradient(values)

    assert gradient.shape == values.shape
    # Gradient is zero at valid point
    assert abs(gradient[0]) < 1e-4

    # Gradient should be non-zero for out-of-bounds values
    values_invalid = np.array([1.5])
    gradient_invalid = constraint.compute_gradient(values_invalid)

    # Should point toward valid region (negative, toward max of 1.0)
    assert gradient_invalid[0] != 0.0


def test_constraint_weighted_loss() -> None:
    """Test weighted loss computation."""
    constraint = ParameterBoundsConstraint("metal_ratio", weight=2.0)

    values = np.array([1.5])  # Violation
    base_loss = constraint.compute_loss(values)
    weighted = constraint.weighted_loss(values)

    assert weighted == base_loss * 2.0
