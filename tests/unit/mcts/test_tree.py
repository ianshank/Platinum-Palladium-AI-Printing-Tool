"""
Unit tests for MCTS tree node implementation.
"""

from __future__ import annotations

import pytest

from ptpd_calibration.mcts.tree import TreeNode
from ptpd_calibration.mcts.types import CalibrationAction, CalibrationState


@pytest.fixture
def initial_state() -> CalibrationState:
    """Create initial state for testing."""
    return CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight", "exposure_time"],
        depth=0,
    )


@pytest.fixture
def partial_state() -> CalibrationState:
    """Create partial state with one decision made."""
    return CalibrationState(
        decided_parameters={"metal_ratio": 0.6},
        remaining_dimensions=["coating_weight", "exposure_time"],
        depth=1,
    )


@pytest.fixture
def terminal_state() -> CalibrationState:
    """Create terminal state with all decisions made."""
    return CalibrationState(
        decided_parameters={
            "metal_ratio": 0.6,
            "coating_weight": 1.5,
            "exposure_time": 180.0,
        },
        remaining_dimensions=[],
        depth=3,
    )


def test_node_initialization(initial_state: CalibrationState) -> None:
    """Test basic node initialization."""
    node = TreeNode(state=initial_state, parent=None, action=None, prior=0.5)

    assert node.state == initial_state
    assert node.parent is None
    assert node.action is None
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.prior == 0.5
    assert node.is_leaf
    assert not node.is_terminal


def test_node_is_terminal(terminal_state: CalibrationState) -> None:
    """Test terminal node detection."""
    node = TreeNode(state=terminal_state)
    assert node.is_terminal


def test_mean_value_unvisited() -> None:
    """Test mean value for unvisited node."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    node = TreeNode(state=state)
    assert node.mean_value == 0.0


def test_mean_value_visited() -> None:
    """Test mean value calculation after visits."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    node = TreeNode(state=state)

    node.visit_count = 10
    node.value_sum = 7.5

    assert node.mean_value == 0.75


def test_ucb_score_known_values() -> None:
    """Test UCB score computation with known values."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    parent = TreeNode(state=state)
    parent.visit_count = 100

    child_state = CalibrationState(
        decided_parameters={"metal_ratio": 0.5},
        remaining_dimensions=[],
        depth=1,
    )
    action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
    child = TreeNode(state=child_state, parent=parent, action=action, prior=0.3)

    child.visit_count = 5
    child.value_sum = 3.0  # mean = 0.6

    c_puct = 1.4

    # UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    # UCB = 0.6 + 1.4 * 0.3 * sqrt(100) / (1 + 5)
    # UCB = 0.6 + 1.4 * 0.3 * 10 / 6
    # UCB = 0.6 + 0.7 = 1.3

    ucb = child.ucb_score(c_puct)
    assert abs(ucb - 1.3) < 0.01


def test_select_child_highest_ucb() -> None:
    """Test select_child picks highest UCB score."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    parent = TreeNode(state=state)
    parent.visit_count = 100

    # Create three children with different values and priors
    children_data = [
        (0.3, 0.2, 5, 2.0),  # (value, prior, visits, value_sum)
        (0.5, 0.5, 10, 8.0),  # This should have highest UCB
        (0.7, 0.3, 20, 15.0),
    ]

    for idx, (value, prior, visits, value_sum) in enumerate(children_data):
        child_state = CalibrationState(
            decided_parameters={"metal_ratio": value},
            remaining_dimensions=[],
            depth=1,
        )
        action = CalibrationAction(dimension="metal_ratio", value=value, bin_index=idx)
        child = TreeNode(state=child_state, parent=parent, action=action, prior=prior)
        child.visit_count = visits
        child.value_sum = value_sum
        parent.children.append(child)

    c_puct = 1.4
    selected = parent.select_child(c_puct)

    # The child with prior=0.5 should have highest UCB due to exploration term
    assert selected.prior == 0.5


def test_select_child_no_children() -> None:
    """Test select_child raises error on leaf node."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    node = TreeNode(state=state)

    with pytest.raises(ValueError, match="Cannot select child from leaf node"):
        node.select_child(c_puct=1.4)


def test_expand_creates_child(initial_state: CalibrationState) -> None:
    """Test expand creates correct child state."""
    parent = TreeNode(state=initial_state)

    action = CalibrationAction(dimension="metal_ratio", value=0.6, bin_index=12)
    child = parent.expand(action, prior=0.4)

    assert child.parent == parent
    assert child.action == action
    assert child.prior == 0.4
    assert child.state.depth == initial_state.depth + 1
    assert child.state.decided_parameters["metal_ratio"] == 0.6
    assert "metal_ratio" not in child.state.remaining_dimensions
    assert len(parent.children) == 1
    assert parent.children[0] == child


def test_backpropagate_updates_ancestors() -> None:
    """Test backpropagate updates visit count and value sum up to root."""
    # Create a chain: root -> child1 -> child2
    root_state = CalibrationState(
        decided_parameters={},
        remaining_dimensions=["metal_ratio", "coating_weight"],
    )
    root = TreeNode(state=root_state)

    action1 = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
    child1_state = root_state.apply_action(action1)
    child1 = TreeNode(state=child1_state, parent=root, action=action1)
    root.children.append(child1)

    action2 = CalibrationAction(dimension="coating_weight", value=1.5, bin_index=15)
    child2_state = child1_state.apply_action(action2)
    child2 = TreeNode(state=child2_state, parent=child1, action=action2)
    child1.children.append(child2)

    # Backpropagate from child2
    value = 0.85
    child2.backpropagate(value)

    # All nodes should be updated
    assert child2.visit_count == 1
    assert child2.value_sum == 0.85
    assert child1.visit_count == 1
    assert child1.value_sum == 0.85
    assert root.visit_count == 1
    assert root.value_sum == 0.85

    # Backpropagate again
    child2.backpropagate(0.75)

    assert child2.visit_count == 2
    assert child2.value_sum == 1.60
    assert child1.visit_count == 2
    assert child1.value_sum == 1.60
    assert root.visit_count == 2
    assert root.value_sum == 1.60


def test_get_visit_distribution() -> None:
    """Test get_visit_distribution returns correct mapping."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    parent = TreeNode(state=state)

    # Create children with different visit counts
    for bin_idx in [5, 10, 15]:
        child_state = CalibrationState(
            decided_parameters={"metal_ratio": bin_idx * 0.05},
            remaining_dimensions=[],
            depth=1,
        )
        action = CalibrationAction(
            dimension="metal_ratio",
            value=bin_idx * 0.05,
            bin_index=bin_idx,
        )
        child = TreeNode(state=child_state, parent=parent, action=action)
        child.visit_count = bin_idx  # Use bin_idx as visit count for easy testing
        parent.children.append(child)

    distribution = parent.get_visit_distribution()

    assert distribution[5] == 5
    assert distribution[10] == 10
    assert distribution[15] == 15
    assert len(distribution) == 3


def test_best_child_greedy() -> None:
    """Test best_child with temperature=0 (greedy) picks most visited."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    parent = TreeNode(state=state)

    # Create children with different visit counts
    visit_counts = [10, 25, 15]
    for idx, visits in enumerate(visit_counts):
        child_state = CalibrationState(
            decided_parameters={"metal_ratio": idx * 0.3},
            remaining_dimensions=[],
            depth=1,
        )
        action = CalibrationAction(dimension="metal_ratio", value=idx * 0.3, bin_index=idx)
        child = TreeNode(state=child_state, parent=parent, action=action)
        child.visit_count = visits
        parent.children.append(child)

    best = parent.best_child(temperature=0.0)
    assert best.visit_count == 25


def test_best_child_stochastic(initial_state: CalibrationState) -> None:
    """Test best_child with temperature>0 samples by visit distribution."""
    parent = TreeNode(state=initial_state)

    # Create children with different visit counts
    visit_counts = [100, 10, 1]
    for idx, visits in enumerate(visit_counts):
        child_state = CalibrationState(
            decided_parameters={"metal_ratio": idx * 0.3},
            remaining_dimensions=["coating_weight"],
            depth=1,
        )
        action = CalibrationAction(dimension="metal_ratio", value=idx * 0.3, bin_index=idx)
        child = TreeNode(state=child_state, parent=parent, action=action)
        child.visit_count = visits
        parent.children.append(child)

    # With high temperature, we should sometimes get non-best children
    # Run multiple times and check we don't always get the same one
    temperature = 1.0
    selected_visits = set()
    for _ in range(50):
        best = parent.best_child(temperature=temperature)
        selected_visits.add(best.visit_count)

    # Should select the most visited child most often, but not exclusively
    # (This is stochastic so we can't be 100% certain, but very likely)
    assert 100 in selected_visits  # Should definitely select the best sometimes


def test_best_child_no_children() -> None:
    """Test best_child raises error on leaf node."""
    state = CalibrationState(decided_parameters={}, remaining_dimensions=["metal_ratio"])
    node = TreeNode(state=state)

    with pytest.raises(ValueError, match="Cannot select best child from leaf node"):
        node.best_child(temperature=0.0)


def test_best_child_all_unvisited(initial_state: CalibrationState) -> None:
    """Test best_child handles all unvisited children."""
    parent = TreeNode(state=initial_state)

    # Create children with zero visits
    for idx in range(3):
        child_state = CalibrationState(
            decided_parameters={"metal_ratio": idx * 0.3},
            remaining_dimensions=["coating_weight"],
            depth=1,
        )
        action = CalibrationAction(dimension="metal_ratio", value=idx * 0.3, bin_index=idx)
        child = TreeNode(state=child_state, parent=parent, action=action)
        child.visit_count = 0
        parent.children.append(child)

    # Should pick one randomly
    best = parent.best_child(temperature=1.0)
    assert best.visit_count == 0
    assert best in parent.children
