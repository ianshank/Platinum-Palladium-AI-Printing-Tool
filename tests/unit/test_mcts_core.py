"""
Comprehensive unit tests for MCTS calibration engine.

Tests all core components: config, types, simulator, quality scorer,
tree nodes, engine, constraints, and export functionality.
"""

import numpy as np
import pytest

# Config imports
from ptpd_calibration.mcts.config import (
    DEFAULT_PARAMETER_RANGES,
    MCTSSettings,
    ParameterRange,
    PhysicsConstants,
)

# Constraints imports
from ptpd_calibration.mcts.constraints import (
    ActionPruner,
    DeveloperTemperatureConstraint,
    ExposureTimeConstraint,
    FOConcentrationConstraint,
    HumidityConstraint,
    MetalRatioConstraint,
    ParameterBoundsConstraint,
)

# Engine imports
from ptpd_calibration.mcts.engine import MCTSEngine

# Export imports
from ptpd_calibration.mcts.export import MCTSResultExporter

# Quality scorer imports
from ptpd_calibration.mcts.quality import QualityScorer

# Simulator imports
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator, ProcessParameters

# Tree imports
from ptpd_calibration.mcts.tree import TreeNode

# Types imports
from ptpd_calibration.mcts.types import (
    CalibrationAction,
    CalibrationState,
    SearchResult,
    SimulationResult,
    TrainingExample,
    TrainingMetrics,
)

# =============================================================================
# 1. Config Tests
# =============================================================================


class TestParameterRange:
    """Tests for ParameterRange model."""

    def test_parameter_range_creation(self):
        """Test creating a ParameterRange."""
        param = ParameterRange(
            name="test_param",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            step=0.1,
            unit="units",
        )
        assert param.name == "test_param"
        assert param.min_value == 0.0
        assert param.max_value == 1.0
        assert param.default_value == 0.5
        assert param.step == 0.1
        assert param.unit == "units"

    def test_parameter_range_repr(self):
        """Test ParameterRange string representation."""
        param = ParameterRange(
            name="test_param",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            step=0.1,
            unit="units",
        )
        repr_str = repr(param)
        assert "test_param" in repr_str
        assert "[0.0, 1.0]" in repr_str
        assert "step=0.1" in repr_str
        assert "units" in repr_str

    def test_parameter_range_no_step(self):
        """Test ParameterRange without step."""
        param = ParameterRange(
            name="continuous",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        repr_str = repr(param)
        assert "step=" not in repr_str


class TestMCTSSettings:
    """Tests for MCTSSettings configuration."""

    def test_default_settings(self):
        """Test MCTSSettings with default values."""
        settings = MCTSSettings()
        assert settings.num_simulations == 800
        assert settings.c_puct == 1.4
        assert settings.dirichlet_alpha == 0.3
        assert settings.dirichlet_fraction == 0.25
        assert settings.temperature_initial == 1.0
        assert settings.temperature_final == 0.1
        assert settings.temperature_decay_steps == 30
        assert settings.action_bins == 21
        assert settings.target_dmax == 2.0
        assert settings.target_dmin == 0.05

    def test_quality_metric_weights(self):
        """Test quality metric weights are set correctly."""
        settings = MCTSSettings()
        assert settings.linearity_weight == 0.4
        assert settings.dmax_weight == 0.3
        assert settings.smoothness_weight == 0.2
        assert settings.cost_weight == 0.1

    def test_quality_scoring_parameters(self):
        """Test quality scoring parameters."""
        settings = MCTSSettings()
        assert settings.linearity_decay_rate == 5.0
        assert settings.dmax_scoring_sigma == 0.5
        assert settings.smoothness_decay_rate == 20.0
        assert settings.cost_metal_weight == 0.6

    def test_decision_order_validation(self):
        """Test decision_order is validated on initialization."""
        # Default decision order should be valid
        settings = MCTSSettings()
        assert len(settings.decision_order) == 6
        assert "metal_ratio" in settings.decision_order
        assert "coating_weight" in settings.decision_order
        assert "ferric_oxalate_pct" in settings.decision_order
        assert "exposure_time" in settings.decision_order
        assert "developer_temp" in settings.decision_order
        assert "humidity" in settings.decision_order

    def test_custom_decision_order(self):
        """Test custom decision order."""
        custom_order = ["humidity", "metal_ratio", "coating_weight"]
        settings = MCTSSettings(decision_order=custom_order)
        assert settings.decision_order == custom_order


class TestPhysicsConstants:
    """Tests for PhysicsConstants model."""

    def test_default_physics_constants(self):
        """Test PhysicsConstants with default values."""
        physics = PhysicsConstants()
        # Chemistry effects
        assert physics.pt_gamma_base == 1.6
        assert physics.pd_gamma_base == 2.2
        assert physics.fo_contrast_slope == 0.03
        assert physics.fo_contrast_center == 20.0

        # Exposure effects
        assert physics.exposure_dmax_rate == 0.005
        assert physics.exposure_dmax_ceiling == 2.5
        assert physics.exposure_dmax_halflife == 120.0

        # Environment effects
        assert physics.humidity_uniformity_slope == -0.005
        assert physics.humidity_optimal == 50.0
        assert physics.dev_temp_rate_slope == 0.02
        assert physics.dev_temp_reference == 25.0

        # Coating effects
        assert physics.coating_weight_dmax_slope == 0.3
        assert physics.coating_weight_dmax_ceiling == 2.8
        assert physics.paper_dmin_base == 0.08

    def test_new_physics_fields(self):
        """Test new physics constant fields are present."""
        physics = PhysicsConstants()
        # Shoulder and toe adjustments
        assert hasattr(physics, "shoulder_temp_sensitivity")
        assert physics.shoulder_temp_sensitivity == 0.01
        assert hasattr(physics, "shoulder_compression_factor")
        assert physics.shoulder_compression_factor == 0.5
        assert hasattr(physics, "toe_expansion_factor")
        assert physics.toe_expansion_factor == 0.3

        # Contrast clamping
        assert hasattr(physics, "contrast_min")
        assert physics.contrast_min == 0.5
        assert hasattr(physics, "contrast_max")
        assert physics.contrast_max == 2.0


class TestDefaultParameterRanges:
    """Tests for DEFAULT_PARAMETER_RANGES."""

    def test_all_six_parameters_present(self):
        """Test DEFAULT_PARAMETER_RANGES has all 6 parameters."""
        assert len(DEFAULT_PARAMETER_RANGES) == 6
        assert "metal_ratio" in DEFAULT_PARAMETER_RANGES
        assert "coating_weight" in DEFAULT_PARAMETER_RANGES
        assert "ferric_oxalate_pct" in DEFAULT_PARAMETER_RANGES
        assert "exposure_time" in DEFAULT_PARAMETER_RANGES
        assert "developer_temp" in DEFAULT_PARAMETER_RANGES
        assert "humidity" in DEFAULT_PARAMETER_RANGES

    def test_parameter_ranges_valid(self):
        """Test all parameter ranges have valid bounds."""
        for name, param_range in DEFAULT_PARAMETER_RANGES.items():
            assert param_range.min_value < param_range.max_value
            assert param_range.min_value <= param_range.default_value <= param_range.max_value


# =============================================================================
# 2. Types Tests
# =============================================================================


class TestCalibrationAction:
    """Tests for CalibrationAction."""

    def test_action_creation(self):
        """Test creating a CalibrationAction."""
        action = CalibrationAction(
            dimension="metal_ratio",
            value=0.5,
            bin_index=10,
        )
        assert action.dimension == "metal_ratio"
        assert action.value == 0.5
        assert action.bin_index == 10

    def test_action_repr(self):
        """Test CalibrationAction string representation."""
        action = CalibrationAction(
            dimension="metal_ratio",
            value=0.5,
            bin_index=10,
        )
        repr_str = repr(action)
        assert "metal_ratio" in repr_str
        assert "0.500" in repr_str
        assert "bin=10" in repr_str


class TestCalibrationState:
    """Tests for CalibrationState."""

    def test_initial_state(self):
        """Test creating an initial CalibrationState."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        assert state.depth == 0
        assert len(state.decided_parameters) == 0
        assert len(state.remaining_dimensions) == 2
        assert not state.is_terminal

    def test_terminal_state(self):
        """Test is_terminal property."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=[],
            depth=1,
        )
        assert state.is_terminal
        assert state.current_dimension is None

    def test_current_dimension(self):
        """Test current_dimension property."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        assert state.current_dimension == "metal_ratio"

    def test_apply_action(self):
        """Test applying an action to a state."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        action = CalibrationAction(
            dimension="metal_ratio",
            value=0.5,
            bin_index=10,
        )
        new_state = state.apply_action(action)

        # Check new state
        assert new_state.depth == 1
        assert new_state.decided_parameters == {"metal_ratio": 0.5}
        assert new_state.remaining_dimensions == ["coating_weight"]
        assert not new_state.is_terminal

        # Original state unchanged
        assert state.depth == 0
        assert len(state.decided_parameters) == 0

    def test_apply_action_to_terminal(self):
        """Test applying action to terminal state raises ValueError."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=[],
            depth=1,
        )
        action = CalibrationAction(
            dimension="coating_weight",
            value=1.5,
            bin_index=5,
        )
        with pytest.raises(ValueError, match="terminal"):
            state.apply_action(action)

    def test_apply_invalid_action(self):
        """Test applying action with dimension not in remaining_dimensions raises ValueError."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        action = CalibrationAction(
            dimension="coating_weight",
            value=1.5,
            bin_index=5,
        )
        with pytest.raises(ValueError, match="Invalid action dimension"):
            state.apply_action(action)


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_simulation_result_creation(self):
        """Test creating a SimulationResult."""
        result = SimulationResult(
            density_curve=[0.1, 0.5, 1.0, 1.5, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.85,
            parameters={"metal_ratio": 0.5},
            constraint_violations=[],
        )
        assert len(result.density_curve) == 5
        assert result.dmin == 0.1
        assert result.dmax == 2.0
        assert result.gamma == 1.8
        assert result.quality_score == 0.85

    def test_simulation_result_repr(self):
        """Test SimulationResult string representation."""
        result = SimulationResult(
            density_curve=[0.1, 1.0, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.85,
            parameters={},
        )
        repr_str = repr(result)
        assert "quality=0.850" in repr_str
        assert "dmax=2.00" in repr_str
        assert "gamma=1.80" in repr_str


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[0.1, 1.0, 2.0],
            quality_score=0.85,
            num_simulations=100,
            search_time_seconds=10.5,
        )
        assert result.quality_score == 0.85
        assert result.num_simulations == 100
        assert result.search_time_seconds == 10.5
        assert len(result.predicted_curve) == 3


class TestTrainingExample:
    """Tests for TrainingExample."""

    def test_training_example_creation(self):
        """Test creating a TrainingExample."""
        example = TrainingExample(
            state_features=[0.1, 0.2, 0.3],
            policy_target=[0.5, 0.3, 0.2],
            value_target=0.8,
        )
        assert len(example.state_features) == 3
        assert len(example.policy_target) == 3
        assert example.value_target == 0.8


class TestTrainingMetrics:
    """Tests for TrainingMetrics."""

    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            episode=10,
            value_loss=0.5,
            policy_loss=0.3,
            total_loss=0.8,
            best_quality=0.9,
            mean_quality=0.75,
            episodes_completed=10,
        )
        assert metrics.episode == 10
        assert metrics.total_loss == 0.8


# =============================================================================
# 3. Simulator Tests
# =============================================================================


class TestExtendedProcessSimulator:
    """Tests for ExtendedProcessSimulator."""

    def test_simulator_initialization(self):
        """Test initializing ExtendedProcessSimulator."""
        simulator = ExtendedProcessSimulator()
        assert simulator.physics is not None
        assert simulator.settings is not None

    def test_compute_process_parameters_defaults(self):
        """Test compute_process_parameters with defaults."""
        simulator = ExtendedProcessSimulator()
        params = {}
        process_params = simulator.compute_process_parameters(params)

        assert isinstance(process_params, ProcessParameters)
        assert 1.0 < process_params.gamma < 3.0
        assert 0.0 < process_params.dmin < 0.5
        # dmax depends on coating_weight and exposure_time defaults
        # With defaults, dmax should be reasonable (above dmin at least)
        assert process_params.dmax > process_params.dmin
        assert process_params.dmax < 3.5  # Below ceiling
        assert 0.5 < process_params.shoulder_position <= 1.0
        assert 0.0 <= process_params.toe_position < 0.5
        assert 0.5 <= process_params.contrast <= 2.0

    def test_compute_process_parameters_metal_ratio_gamma(self):
        """Test that metal_ratio affects gamma."""
        simulator = ExtendedProcessSimulator()

        # Pure Pt (metal_ratio=1.0)
        params_pt = {"metal_ratio": 1.0}
        process_pt = simulator.compute_process_parameters(params_pt)

        # Pure Pd (metal_ratio=0.0)
        params_pd = {"metal_ratio": 0.0}
        process_pd = simulator.compute_process_parameters(params_pd)

        # Pd should have higher gamma than Pt
        assert process_pd.gamma > process_pt.gamma

    def test_compute_process_parameters_coating_weight_dmax(self):
        """Test that coating_weight affects dmax."""
        simulator = ExtendedProcessSimulator()

        # Low coating weight
        params_low = {"coating_weight": 0.5}
        process_low = simulator.compute_process_parameters(params_low)

        # High coating weight
        params_high = {"coating_weight": 3.0}
        process_high = simulator.compute_process_parameters(params_high)

        # Higher coating should give higher dmax
        assert process_high.dmax > process_low.dmax

    def test_simulate_with_numpy_returns_valid_result(self):
        """Test simulate_with_numpy produces valid results."""
        simulator = ExtendedProcessSimulator()
        params = {
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        }
        result = simulator.simulate_with_numpy(params, num_steps=21)

        assert isinstance(result, SimulationResult)
        assert len(result.density_curve) == 21
        assert result.dmin >= 0.0
        assert result.dmax > result.dmin
        assert result.gamma > 0.0

    def test_simulate_returns_correct_length(self):
        """Test simulate returns density curve of correct length."""
        simulator = ExtendedProcessSimulator()
        params = {"metal_ratio": 0.5}

        for num_steps in [11, 21, 31]:
            result = simulator.simulate(params, num_steps=num_steps)
            assert len(result.density_curve) == num_steps

    def test_simulate_extreme_parameters_min(self):
        """Test simulate with all minimum parameters."""
        simulator = ExtendedProcessSimulator()
        params = {
            "metal_ratio": 0.0,
            "coating_weight": 0.5,
            "ferric_oxalate_pct": 15.0,
            "exposure_time": 30.0,
            "developer_temp": 20.0,
            "humidity": 30.0,
        }
        result = simulator.simulate(params)
        # Should still produce valid results
        assert result.dmin >= 0.0
        assert result.dmax > result.dmin

    def test_simulate_extreme_parameters_max(self):
        """Test simulate with all maximum parameters."""
        simulator = ExtendedProcessSimulator()
        params = {
            "metal_ratio": 1.0,
            "coating_weight": 3.0,
            "ferric_oxalate_pct": 27.0,
            "exposure_time": 600.0,
            "developer_temp": 50.0,
            "humidity": 80.0,
        }
        result = simulator.simulate(params)
        # Should still produce valid results
        assert result.dmin >= 0.0
        assert result.dmax > result.dmin


# =============================================================================
# 4. Quality Scorer Tests
# =============================================================================


class TestQualityScorer:
    """Tests for QualityScorer."""

    def test_scorer_initialization(self):
        """Test initializing QualityScorer."""
        scorer = QualityScorer()
        assert scorer.settings is not None

    def test_score_returns_value_in_range(self):
        """Test score returns value in [0, 1]."""
        scorer = QualityScorer()
        result = SimulationResult(
            density_curve=list(np.linspace(0.1, 2.0, 21)),
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.5, "coating_weight": 1.5},
        )
        score = scorer.score(result)
        assert 0.0 <= score <= 1.0

    def test_score_perfectly_linear_curve(self):
        """Test perfectly linear curve gets high linearity score."""
        scorer = QualityScorer()
        # Create perfectly linear curve
        linear_curve = list(np.linspace(0.1, 2.0, 21))
        result = SimulationResult(
            density_curve=linear_curve,
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.5, "coating_weight": 1.5},
        )
        score = scorer.score(result)
        # Should get high score for linearity
        assert score > 0.5

    def test_score_dmax_on_target(self):
        """Test dmax on target gets high dmax score."""
        settings = MCTSSettings(target_dmax=2.0)
        scorer = QualityScorer(settings=settings)
        result = SimulationResult(
            density_curve=list(np.linspace(0.1, 2.0, 21)),
            dmin=0.1,
            dmax=2.0,  # Exactly on target
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.5, "coating_weight": 1.5},
        )
        score = scorer.score(result)
        # Should get high score
        assert score > 0.5

    def test_score_smooth_curve(self):
        """Test smooth curve gets high smoothness score."""
        scorer = QualityScorer()
        # Linear is very smooth
        smooth_curve = list(np.linspace(0.1, 2.0, 21))
        result = SimulationResult(
            density_curve=smooth_curve,
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.5, "coating_weight": 1.5},
        )
        score = scorer.score(result)
        assert score > 0.5

    def test_cost_score_favors_palladium(self):
        """Test cost score favors palladium (metal_ratio=0)."""
        scorer = QualityScorer()

        # Pure Pd (should be cheaper)
        result_pd = SimulationResult(
            density_curve=list(np.linspace(0.1, 2.0, 21)),
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.0, "coating_weight": 1.5},
        )
        score_pd = scorer.score(result_pd)

        # Pure Pt (should be more expensive)
        result_pt = SimulationResult(
            density_curve=list(np.linspace(0.1, 2.0, 21)),
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 1.0, "coating_weight": 1.5},
        )
        score_pt = scorer.score(result_pt)

        # Pd should get higher score (lower cost)
        assert score_pd > score_pt

    def test_quality_weights_respected(self):
        """Test that quality weights from settings are respected."""
        # All weight on linearity
        settings_linearity = MCTSSettings(
            linearity_weight=1.0,
            dmax_weight=0.0,
            smoothness_weight=0.0,
            cost_weight=0.0,
        )
        scorer_linearity = QualityScorer(settings=settings_linearity)

        # Perfectly linear curve
        linear_curve = list(np.linspace(0.1, 2.0, 21))
        result = SimulationResult(
            density_curve=linear_curve,
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={"metal_ratio": 0.5, "coating_weight": 1.5},
        )

        score = scorer_linearity.score(result)
        # Should be very high (near 1.0) since only linearity matters
        assert score > 0.9


# =============================================================================
# 5. Tree Node Tests
# =============================================================================


class TestTreeNode:
    """Tests for TreeNode."""

    def test_node_creation(self):
        """Test creating a TreeNode."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        node = TreeNode(state=state)
        assert node.state == state
        assert node.parent is None
        assert node.visit_count == 0
        assert node.value_sum == 0.0

    def test_is_leaf(self):
        """Test is_leaf property."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        node = TreeNode(state=state)
        assert node.is_leaf

        # Add child
        action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
        child = node.expand(action)
        assert not node.is_leaf
        assert child.is_leaf

    def test_is_terminal(self):
        """Test is_terminal property."""
        terminal_state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=[],
            depth=1,
        )
        node = TreeNode(state=terminal_state)
        assert node.is_terminal

    def test_mean_value(self):
        """Test mean_value property."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        node = TreeNode(state=state)

        # Initially zero
        assert node.mean_value == 0.0

        # After backprop
        node.visit_count = 10
        node.value_sum = 8.0
        assert node.mean_value == 0.8

    def test_ucb_score(self):
        """Test ucb_score calculation."""
        parent_state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        parent = TreeNode(state=parent_state)
        parent.visit_count = 100

        child_state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["coating_weight"],
            depth=1,
        )
        child = TreeNode(state=child_state, parent=parent, prior=0.5)
        child.visit_count = 10
        child.value_sum = 7.0

        ucb = child.ucb_score(c_puct=1.4)
        # UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        # UCB = 0.7 + 1.4 * 0.5 * sqrt(100) / (1 + 10)
        # UCB = 0.7 + 1.4 * 0.5 * 10 / 11
        # UCB = 0.7 + 0.636... ≈ 1.336
        assert 1.3 < ucb < 1.4

    def test_expand_creates_child(self):
        """Test expand creates child node."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        node = TreeNode(state=state)

        action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
        child = node.expand(action, prior=0.25)

        assert len(node.children) == 1
        assert child.parent is node
        assert child.action == action
        assert child.prior == 0.25
        assert child.state.depth == 1

    def test_backpropagate_updates_values(self):
        """Test backpropagate updates values up to root."""
        # Create chain: root -> child1 -> child2
        root_state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight", "humidity"],
            depth=0,
        )
        root = TreeNode(state=root_state)

        action1 = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
        child1 = root.expand(action1)

        action2 = CalibrationAction(dimension="coating_weight", value=1.5, bin_index=10)
        child2 = child1.expand(action2)

        # Backpropagate from child2
        child2.backpropagate(0.8)

        # All nodes should be updated
        assert child2.visit_count == 1
        assert child2.value_sum == 0.8
        assert child1.visit_count == 1
        assert child1.value_sum == 0.8
        assert root.visit_count == 1
        assert root.value_sum == 0.8

    def test_select_child_picks_highest_ucb(self):
        """Test select_child picks child with highest UCB."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio", "coating_weight"],
            depth=0,
        )
        parent = TreeNode(state=state)
        parent.visit_count = 100

        # Create children with different visit counts
        action1 = CalibrationAction(dimension="metal_ratio", value=0.3, bin_index=5)
        child1 = parent.expand(action1, prior=0.5)
        child1.visit_count = 10
        child1.value_sum = 7.0

        action2 = CalibrationAction(dimension="metal_ratio", value=0.7, bin_index=15)
        child2 = parent.expand(action2, prior=0.5)
        child2.visit_count = 5  # Lower visit count -> higher UCB
        child2.value_sum = 4.0

        selected = parent.select_child(c_puct=1.4)
        # child2 should be selected (higher exploration bonus)
        assert selected == child2

    def test_get_visit_distribution(self):
        """Test get_visit_distribution."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        parent = TreeNode(state=state)

        action1 = CalibrationAction(dimension="metal_ratio", value=0.3, bin_index=5)
        child1 = parent.expand(action1)
        child1.visit_count = 10

        action2 = CalibrationAction(dimension="metal_ratio", value=0.7, bin_index=15)
        child2 = parent.expand(action2)
        child2.visit_count = 5

        dist = parent.get_visit_distribution()
        assert dist == {5: 10, 15: 5}

    def test_best_child_greedy(self):
        """Test best_child with greedy mode (temperature=0)."""
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        parent = TreeNode(state=state)

        action1 = CalibrationAction(dimension="metal_ratio", value=0.3, bin_index=5)
        child1 = parent.expand(action1)
        child1.visit_count = 10

        action2 = CalibrationAction(dimension="metal_ratio", value=0.7, bin_index=15)
        child2 = parent.expand(action2)
        child2.visit_count = 5

        best = parent.best_child(temperature=0.0)
        # Should pick child with most visits
        assert best == child1


# =============================================================================
# 6. Engine Tests
# =============================================================================


class TestMCTSEngine:
    """Tests for MCTSEngine."""

    def test_engine_initialization(self):
        """Test initializing MCTSEngine."""
        engine = MCTSEngine()
        assert engine.settings is not None
        assert engine.physics is not None
        assert engine.simulator is not None
        assert engine.scorer is not None

    def test_search_runs_without_error(self):
        """Test search runs without error (low num_simulations)."""
        settings = MCTSSettings(num_simulations=50)
        engine = MCTSEngine(settings=settings)
        result = engine.search()
        assert isinstance(result, SearchResult)

    def test_search_returns_valid_result(self):
        """Test search returns valid SearchResult."""
        settings = MCTSSettings(num_simulations=50)
        engine = MCTSEngine(settings=settings)
        result = engine.search()

        assert len(result.best_parameters) > 0
        assert len(result.predicted_curve) > 0
        assert 0.0 <= result.quality_score <= 1.0
        assert result.num_simulations == 50
        assert result.search_time_seconds > 0.0

    def test_search_respects_fixed_parameters(self):
        """Test search respects fixed_parameters."""
        settings = MCTSSettings(num_simulations=50)
        engine = MCTSEngine(settings=settings)

        fixed_params = {"metal_ratio": 0.75, "humidity": 55.0}
        result = engine.search(fixed_parameters=fixed_params)

        # Fixed parameters should be in best_parameters with exact values
        assert result.best_parameters["metal_ratio"] == 0.75
        assert result.best_parameters["humidity"] == 55.0

    def test_create_initial_state(self):
        """Test _create_initial_state."""
        engine = MCTSEngine()
        fixed_params = {"metal_ratio": 0.5}
        state = engine._create_initial_state(
            fixed_parameters=fixed_params,
            paper_type="Arches Platine",
            uv_source="LED",
        )

        assert state.decided_parameters == {"metal_ratio": 0.5}
        assert "metal_ratio" not in state.remaining_dimensions
        assert len(state.remaining_dimensions) == 5
        assert state.paper_type == "Arches Platine"
        assert state.uv_source == "LED"

    def test_rollout_fills_remaining_dimensions(self):
        """Test _rollout fills remaining dimensions."""
        engine = MCTSEngine()
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["coating_weight", "humidity"],
            depth=1,
        )
        params = engine._rollout(state)

        # Should have all 3 parameters
        assert "metal_ratio" in params
        assert "coating_weight" in params
        assert "humidity" in params
        assert params["metal_ratio"] == 0.5

    def test_get_temperature_linear_decay(self):
        """Test _get_temperature linear decay."""
        settings = MCTSSettings(
            temperature_initial=1.0,
            temperature_final=0.1,
            temperature_decay_steps=30,
        )
        engine = MCTSEngine(settings=settings)

        # At step 0
        assert engine._get_temperature(0) == 1.0

        # At step 15 (halfway)
        mid_temp = engine._get_temperature(15)
        assert 0.5 < mid_temp < 0.6

        # At step 30 (end)
        assert engine._get_temperature(30) == 0.1

        # After decay steps
        assert engine._get_temperature(100) == 0.1


# =============================================================================
# 7. Constraints Tests
# =============================================================================


class TestParameterBoundsConstraint:
    """Tests for ParameterBoundsConstraint."""

    def test_bounds_constraint_in_bounds(self):
        """Test parameter in bounds is satisfied."""
        constraint = ParameterBoundsConstraint("metal_ratio")
        values = np.array([0.5])
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_bounds_constraint_out_of_bounds_low(self):
        """Test parameter below minimum violates constraint."""
        constraint = ParameterBoundsConstraint("metal_ratio")
        values = np.array([-0.5])
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert len(result.violations) > 0

    def test_bounds_constraint_out_of_bounds_high(self):
        """Test parameter above maximum violates constraint."""
        constraint = ParameterBoundsConstraint("metal_ratio")
        values = np.array([1.5])
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0
        assert len(result.violations) > 0


class TestFOConcentrationConstraint:
    """Tests for FOConcentrationConstraint."""

    def test_fo_constraint_in_sweet_spot(self):
        """Test FO% in sweet spot is satisfied."""
        constraint = FOConcentrationConstraint()
        values = np.array([20.0])  # In sweet spot
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_fo_constraint_below_sweet_spot(self):
        """Test FO% below sweet spot has penalty."""
        constraint = FOConcentrationConstraint()
        values = np.array([16.0])  # Below sweet spot
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0

    def test_fo_constraint_above_sweet_spot(self):
        """Test FO% above sweet spot has penalty."""
        constraint = FOConcentrationConstraint()
        values = np.array([26.0])  # Above sweet spot
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0


class TestMetalRatioConstraint:
    """Tests for MetalRatioConstraint."""

    def test_metal_ratio_in_blend_range(self):
        """Test metal ratio in blend range is satisfied."""
        constraint = MetalRatioConstraint()
        values = np.array([0.5])  # In typical blend range
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_metal_ratio_extreme_low(self):
        """Test metal ratio near 0 (pure Pd) has penalty."""
        constraint = MetalRatioConstraint()
        values = np.array([0.1])  # Below blend range
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0

    def test_metal_ratio_extreme_high(self):
        """Test metal ratio near 1 (pure Pt) has penalty."""
        constraint = MetalRatioConstraint()
        values = np.array([0.9])  # Above blend range
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0


class TestExposureTimeConstraint:
    """Tests for ExposureTimeConstraint."""

    def test_exposure_time_in_practical_range(self):
        """Test exposure time in practical range is satisfied."""
        constraint = ExposureTimeConstraint()
        values = np.array([180.0])  # Typical exposure
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_exposure_time_too_short(self):
        """Test very short exposure has penalty."""
        constraint = ExposureTimeConstraint()
        values = np.array([30.0])  # Too short
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0

    def test_exposure_time_too_long(self):
        """Test very long exposure has penalty."""
        constraint = ExposureTimeConstraint()
        values = np.array([400.0])  # Too long
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0


class TestDeveloperTemperatureConstraint:
    """Tests for DeveloperTemperatureConstraint."""

    def test_dev_temp_near_reference(self):
        """Test developer temp near reference is satisfied."""
        constraint = DeveloperTemperatureConstraint()
        values = np.array([25.0])  # At reference
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_dev_temp_far_from_reference(self):
        """Test developer temp far from reference has penalty."""
        constraint = DeveloperTemperatureConstraint()
        values = np.array([40.0])  # 15°C above reference
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0


class TestHumidityConstraint:
    """Tests for HumidityConstraint."""

    def test_humidity_at_optimal(self):
        """Test humidity at optimal is satisfied."""
        constraint = HumidityConstraint()
        values = np.array([50.0])  # At optimal
        result = constraint.evaluate(values)

        assert result.is_satisfied
        assert result.loss_value == 0.0

    def test_humidity_far_from_optimal(self):
        """Test humidity far from optimal has penalty."""
        constraint = HumidityConstraint()
        values = np.array([70.0])  # 20% above optimal
        result = constraint.evaluate(values)

        assert not result.is_satisfied
        assert result.loss_value > 0.0


class TestActionPruner:
    """Tests for ActionPruner."""

    def test_pruner_initialization(self):
        """Test ActionPruner initialization."""
        pruner = ActionPruner()
        assert len(pruner.constraints) > 0

    def test_prune_actions_removes_invalid(self):
        """Test prune_actions removes invalid actions."""
        pruner = ActionPruner()
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )

        # Create actions, some invalid
        actions = [
            CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10),  # Valid
            CalibrationAction(
                dimension="metal_ratio", value=-0.5, bin_index=0
            ),  # Invalid (below min)
            CalibrationAction(
                dimension="metal_ratio", value=1.5, bin_index=20
            ),  # Invalid (above max)
        ]

        pruned = pruner.prune_actions(state, actions)
        # Only the valid action should remain
        assert len(pruned) == 1
        assert pruned[0].value == 0.5

    def test_score_action_returns_score(self):
        """Test score_action returns score in [0, 1]."""
        pruner = ActionPruner()
        state = CalibrationState(
            decided_parameters={},
            remaining_dimensions=["metal_ratio"],
            depth=0,
        )
        action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)

        score = pruner.score_action(state, action)
        assert 0.0 <= score <= 1.0


# =============================================================================
# 8. Export Tests
# =============================================================================


class TestMCTSResultExporter:
    """Tests for MCTSResultExporter."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SearchResult for testing."""
        return SearchResult(
            best_parameters={
                "metal_ratio": 0.5,
                "coating_weight": 1.5,
                "ferric_oxalate_pct": 20.0,
                "exposure_time": 180.0,
                "developer_temp": 25.0,
                "humidity": 50.0,
            },
            predicted_curve=list(np.linspace(0.1, 2.0, 21)),
            quality_score=0.85,
            num_simulations=100,
            search_time_seconds=10.5,
            paper_type="Arches Platine",
            uv_source="LED",
        )

    def test_exporter_initialization(self):
        """Test MCTSResultExporter initialization."""
        exporter = MCTSResultExporter()
        assert exporter.settings is not None

    def test_to_recipe_json_creates_valid_dict(self, sample_result):
        """Test to_recipe_json creates valid dictionary."""
        exporter = MCTSResultExporter()
        recipe = exporter.to_recipe_json(sample_result)

        assert "id" in recipe
        assert "timestamp" in recipe
        assert "parameters" in recipe
        assert "predicted_curve" in recipe
        assert "quality_score" in recipe
        assert "metadata" in recipe
        assert recipe["quality_score"] == 0.85
        assert len(recipe["predicted_curve"]["output_values"]) == 21

    def test_to_csv_creates_valid_csv(self, sample_result):
        """Test to_csv creates valid CSV."""
        exporter = MCTSResultExporter()
        csv_str = exporter.to_csv(sample_result)

        lines = csv_str.split("\n")
        assert lines[0] == "step,exposure,density"
        # Should have header + 21 data rows
        assert len(lines) == 22

    def test_to_qtr_curve_returns_256_values(self, sample_result):
        """Test to_qtr_curve returns 256 values in [0, 255]."""
        exporter = MCTSResultExporter()
        qtr_curve = exporter.to_qtr_curve(sample_result)

        assert len(qtr_curve) == 256
        assert all(0 <= val <= 255 for val in qtr_curve)
        assert all(isinstance(val, int) for val in qtr_curve)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
