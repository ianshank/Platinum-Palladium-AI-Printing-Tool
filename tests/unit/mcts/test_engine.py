"""
Unit tests for MCTS engine implementation.
"""

from __future__ import annotations

import pytest

from ptpd_calibration.mcts.config import MCTSSettings, PhysicsConstants
from ptpd_calibration.mcts.engine import MCTSEngine
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator
from ptpd_calibration.mcts.types import CalibrationState, SearchResult


@pytest.fixture
def fast_settings() -> MCTSSettings:
    """Create settings for fast testing."""
    return MCTSSettings(
        num_simulations=50,
        action_bins=11,
        temperature_decay_steps=10,
        decision_order=["metal_ratio", "coating_weight", "exposure_time"],
    )


@pytest.fixture
def engine(fast_settings: MCTSSettings) -> MCTSEngine:
    """Create MCTS engine for testing."""
    return MCTSEngine(settings=fast_settings)


def test_engine_initialization(fast_settings: MCTSSettings) -> None:
    """Test basic engine initialization."""
    physics = PhysicsConstants()
    simulator = ExtendedProcessSimulator(physics=physics)
    scorer = QualityScorer(settings=fast_settings)

    engine = MCTSEngine(
        settings=fast_settings,
        physics=physics,
        simulator=simulator,
        scorer=scorer,
    )

    assert engine.settings == fast_settings
    assert engine.physics == physics
    assert engine.simulator == simulator
    assert engine.scorer == scorer


def test_engine_default_initialization() -> None:
    """Test engine initialization with defaults."""
    engine = MCTSEngine()

    assert engine.settings is not None
    assert engine.physics is not None
    assert engine.simulator is not None
    assert engine.scorer is not None


def test_create_initial_state_no_fixed(engine: MCTSEngine) -> None:
    """Test initial state creation without fixed parameters."""
    state = engine._create_initial_state(
        fixed_parameters=None,
        paper_type="Arches Platine",
        uv_source="LED",
    )

    assert len(state.decided_parameters) == 0
    assert len(state.remaining_dimensions) == 3
    assert state.remaining_dimensions == ["metal_ratio", "coating_weight", "exposure_time"]
    assert state.depth == 0
    assert state.paper_type == "Arches Platine"
    assert state.uv_source == "LED"


def test_create_initial_state_with_fixed(engine: MCTSEngine) -> None:
    """Test initial state creation with fixed parameters."""
    fixed = {"metal_ratio": 0.7, "coating_weight": 1.8}
    state = engine._create_initial_state(
        fixed_parameters=fixed,
        paper_type=None,
        uv_source=None,
    )

    assert state.decided_parameters == fixed
    assert len(state.remaining_dimensions) == 1
    assert state.remaining_dimensions == ["exposure_time"]
    assert state.depth == 0


def test_should_expand_progressive_widening(engine: MCTSEngine) -> None:
    """Test progressive widening controls expansion."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight"],
    )
    node = TreeNode(state=state)

    # With 0 children and 0 visits, should not expand
    node.visit_count = 0
    assert engine._should_expand(node) is False

    # With 0 children and some visits, should expand
    node.visit_count = 5
    assert engine._should_expand(node) is True

    # Add children and check threshold
    for i in range(3):
        from ptpd_calibration.mcts.types import CalibrationAction

        action = CalibrationAction(dimension="metal_ratio", value=i * 0.3, bin_index=i)
        node.expand(action)

    # With k=3 children, threshold = c * k^alpha
    # Default: c=1.0, alpha=0.5
    # threshold = 1.0 * 3^0.5 ≈ 1.73
    # Need visits > threshold to expand
    node.visit_count = 1
    assert engine._should_expand(node) is False

    node.visit_count = 5
    assert engine._should_expand(node) is True


def test_should_expand_terminal_state(engine: MCTSEngine) -> None:
    """Test should_expand returns False for terminal state."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={"metal_ratio": 0.5, "coating_weight": 1.5, "exposure_time": 180.0},
        remaining_dimensions=[],
    )
    node = TreeNode(state=state)
    node.visit_count = 100

    assert engine._should_expand(node) is False


def test_should_expand_max_actions(engine: MCTSEngine) -> None:
    """Test should_expand respects max_actions_per_node."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio"],
    )
    node = TreeNode(state=state)
    node.visit_count = 1000

    # Add max number of children
    from ptpd_calibration.mcts.types import CalibrationAction

    for i in range(engine.settings.max_actions_per_node):
        action = CalibrationAction(dimension="metal_ratio", value=i * 0.05, bin_index=i)
        node.expand(action)

    # Should not expand beyond max
    assert engine._should_expand(node) is False


def test_generate_action(engine: MCTSEngine) -> None:
    """Test action generation from uniform sampling."""
    state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight"],
    )

    # Generate multiple actions and check they are valid
    for _ in range(10):
        action = engine._generate_action(state)

        assert action.dimension == "metal_ratio"
        assert 0.0 <= action.value <= 1.0
        assert 0 <= action.bin_index < engine.settings.action_bins


def test_generate_action_terminal_state(engine: MCTSEngine) -> None:
    """Test generate_action raises error for terminal state."""
    state = CalibrationState(
        decided_parameters={"metal_ratio": 0.5, "coating_weight": 1.5, "exposure_time": 180.0},
        remaining_dimensions=[],
    )

    with pytest.raises(ValueError, match="Cannot generate action for terminal state"):
        engine._generate_action(state)


def test_rollout_completes_state(engine: MCTSEngine) -> None:
    """Test rollout produces complete parameter set."""
    state = CalibrationState(
        decided_parameters={"metal_ratio": 0.6},
        remaining_dimensions=["coating_weight", "exposure_time"],
    )

    params = engine._rollout(state)

    assert "metal_ratio" in params
    assert params["metal_ratio"] == 0.6
    assert "coating_weight" in params
    assert "exposure_time" in params

    # Check values are in valid ranges
    from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

    for dim, value in params.items():
        param_range = DEFAULT_PARAMETER_RANGES[dim]
        assert param_range.min_value <= value <= param_range.max_value


def test_get_temperature_decay(engine: MCTSEngine) -> None:
    """Test temperature decay over steps."""
    initial = engine.settings.temperature_initial
    final = engine.settings.temperature_final
    decay_steps = engine.settings.temperature_decay_steps

    # At step 0, should be initial
    assert engine._get_temperature(0) == initial

    # At decay_steps, should be final
    assert engine._get_temperature(decay_steps) == final

    # Beyond decay_steps, should stay at final
    assert engine._get_temperature(decay_steps + 10) == final

    # Midpoint should be between initial and final
    mid_temp = engine._get_temperature(decay_steps // 2)
    assert final <= mid_temp <= initial


def test_search_completes(engine: MCTSEngine) -> None:
    """Test search with few simulations completes and returns valid result."""
    result = engine.search()

    assert isinstance(result, SearchResult)
    assert result.num_simulations == engine.settings.num_simulations
    assert result.search_time_seconds > 0
    assert 0.0 <= result.quality_score <= 1.0
    assert len(result.best_parameters) > 0
    assert len(result.predicted_curve) > 0


def test_search_with_fixed_parameters(engine: MCTSEngine) -> None:
    """Test search with fixed parameters preserves them in result."""
    fixed = {"metal_ratio": 0.65}

    result = engine.search(fixed_parameters=fixed)

    assert result.best_parameters["metal_ratio"] == 0.65
    assert "coating_weight" in result.best_parameters
    assert "exposure_time" in result.best_parameters


def test_search_with_target_curve(engine: MCTSEngine) -> None:
    """Test search with target curve."""
    # Create a target curve (21 points from 0.05 to 2.0)
    import numpy as np

    target = np.linspace(0.05, 2.0, 21).tolist()

    result = engine.search(target_curve=target)

    assert result.quality_score > 0.0
    assert len(result.predicted_curve) > 0


def test_search_respects_parameter_ranges(engine: MCTSEngine) -> None:
    """Test all parameters in result are within valid ranges."""
    from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

    result = engine.search()

    for dim, value in result.best_parameters.items():
        param_range = DEFAULT_PARAMETER_RANGES[dim]
        assert param_range.min_value <= value <= param_range.max_value


def test_search_with_all_fixed(fast_settings: MCTSSettings) -> None:
    """Test search with all parameters fixed."""
    # Override decision order to have only one parameter
    settings = MCTSSettings(
        num_simulations=50,  # Must be >= 50 per config validation
        decision_order=["metal_ratio"],
    )
    engine = MCTSEngine(settings=settings)

    fixed = {"metal_ratio": 0.5}
    result = engine.search(fixed_parameters=fixed)

    assert result.best_parameters == fixed
    assert result.quality_score >= 0.0


def test_search_produces_alternatives(engine: MCTSEngine) -> None:
    """Test search produces alternative parameter sets."""
    result = engine.search()

    # Should have some alternatives (up to 5)
    assert len(result.alternatives) <= 5

    # Each alternative should be a valid parameter set
    from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

    for alt_params in result.alternatives:
        for dim in engine.settings.decision_order:
            assert dim in alt_params
            param_range = DEFAULT_PARAMETER_RANGES[dim]
            assert param_range.min_value <= alt_params[dim] <= param_range.max_value


def test_search_with_paper_type_and_uv_source(engine: MCTSEngine) -> None:
    """Test search preserves paper type and UV source in result."""
    result = engine.search(
        paper_type="Bergger COT-320",
        uv_source="Metal Halide",
    )

    assert result.paper_type == "Bergger COT-320"
    assert result.uv_source == "Metal Halide"


def test_evaluate_terminal_node(engine: MCTSEngine) -> None:
    """Test evaluate on terminal node."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "exposure_time": 180.0,
        },
        remaining_dimensions=[],
    )
    node = TreeNode(state=state)

    quality = engine._evaluate(node, target_curve=None)

    assert 0.0 <= quality <= 1.0


def test_evaluate_non_terminal_node(engine: MCTSEngine) -> None:
    """Test evaluate on non-terminal node uses rollout."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={"metal_ratio": 0.6},
        remaining_dimensions=["coating_weight", "exposure_time"],
    )
    node = TreeNode(state=state)

    quality = engine._evaluate(node, target_curve=None)

    assert 0.0 <= quality <= 1.0


def test_select_traverses_to_leaf(engine: MCTSEngine) -> None:
    """Test select traverses tree using UCB to find leaf."""
    from ptpd_calibration.mcts.tree import TreeNode
    from ptpd_calibration.mcts.types import CalibrationAction

    # Create a small tree
    root_state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight"],
    )
    root = TreeNode(state=root_state)
    root.visit_count = 10

    # Add a child
    action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
    child = root.expand(action, prior=0.5)
    child.visit_count = 5

    # Select should return the child (only option)
    selected = engine._select(root)
    assert selected == child

    # If root is a leaf, should return root
    root2 = TreeNode(state=root_state)
    selected2 = engine._select(root2)
    assert selected2 == root2


def test_expand_adds_child(engine: MCTSEngine) -> None:
    """Test expand phase adds new child to node."""
    from ptpd_calibration.mcts.tree import TreeNode

    state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight"],
    )
    node = TreeNode(state=state)

    initial_children = len(node.children)
    child = engine._expand(node)

    assert len(node.children) == initial_children + 1
    assert child in node.children
    assert child.parent == node
    assert child.state.depth == node.state.depth + 1


def test_extract_result_visit_distribution(engine: MCTSEngine) -> None:
    """Test extract_result collects visit distribution."""
    from ptpd_calibration.mcts.tree import TreeNode
    from ptpd_calibration.mcts.types import CalibrationAction

    # Create root with two dimensions to explore (metal_ratio -> coating_weight)
    root_state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight", "exposure_time"],
    )
    root = TreeNode(state=root_state)

    # Add children for first level (metal_ratio) with different visit counts
    # Use bins within action_bins range (0-10 since action_bins=11)
    children_level1 = []
    for bin_idx in [2, 5, 8]:
        action = CalibrationAction(dimension="metal_ratio", value=bin_idx * 0.1, bin_index=bin_idx)
        child_state = root_state.apply_action(action)
        child = TreeNode(state=child_state, parent=root, action=action)
        child.visit_count = bin_idx * 2  # Different visit counts
        root.children.append(child)
        children_level1.append(child)

    # For the most visited child (bin_idx=8), add terminal grandchildren
    best_child = children_level1[-1]  # bin_idx=8
    for coating_bin in [3, 7]:
        coating_action = CalibrationAction(
            dimension="coating_weight",
            value=1.5,
            bin_index=coating_bin,
        )
        grandchild_state = best_child.state.apply_action(coating_action)
        grandchild = TreeNode(state=grandchild_state, parent=best_child, action=coating_action)
        grandchild.visit_count = coating_bin
        best_child.children.append(grandchild)

    # For the best grandchild, add terminal great-grandchildren
    best_grandchild = best_child.children[-1]  # bin_idx=7
    for exposure_bin in [6]:
        exposure_action = CalibrationAction(
            dimension="exposure_time",
            value=180.0,
            bin_index=exposure_bin,
        )
        terminal_state = best_grandchild.state.apply_action(exposure_action)
        terminal_node = TreeNode(
            state=terminal_state, parent=best_grandchild, action=exposure_action
        )
        terminal_node.visit_count = 1
        best_grandchild.children.append(terminal_node)

    result = engine._extract_result(
        root=root,
        target_curve=None,
        search_time=1.0,
        paper_type=None,
        uv_source=None,
    )

    assert "metal_ratio" in result.visit_distribution
    dist = result.visit_distribution["metal_ratio"]
    assert len(dist) == engine.settings.action_bins

    # Check that visit counts are reflected (normalized)
    total_visits = 4 + 10 + 16  # (2*2 + 5*2 + 8*2)
    assert dist[2] == pytest.approx(4 / total_visits)
    assert dist[5] == pytest.approx(10 / total_visits)
    assert dist[8] == pytest.approx(16 / total_visits)


def test_search_convergence(fast_settings: MCTSSettings) -> None:
    """Test search with more simulations produces reasonable parameters."""
    # Use more simulations for better convergence
    settings = MCTSSettings(
        num_simulations=100,
        decision_order=["metal_ratio", "coating_weight", "exposure_time"],
    )
    engine = MCTSEngine(settings=settings)

    result = engine.search()

    # Should produce valid parameters
    assert "metal_ratio" in result.best_parameters
    assert "coating_weight" in result.best_parameters
    assert "exposure_time" in result.best_parameters

    # Quality should be reasonable (>0.1 with random search)
    assert result.quality_score > 0.1

    # Should have produced a curve
    assert len(result.predicted_curve) > 0
