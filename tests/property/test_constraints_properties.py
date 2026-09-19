"""Property-based tests for ``mcts/constraints.py`` (TST-08).

F4 note (``ParameterBoundsConstraint``): ``is_satisfied`` is the authoritative
verdict and is what ``ActionPruner.prune_actions`` uses. ``loss_value == 0``
does **not** imply ``is_satisfied``: for an out-of-range value whose distance
to the bound is sub-normal (e.g. ``-2.2e-175`` against a minimum of ``0``) the
squared penalty underflows to ``0.0`` while ``is_satisfied`` is correctly
``False``. Only the direction ``is_satisfied => loss == 0`` is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES
from ptpd_calibration.mcts.constraints import (
    ActionPruner,
    CoatingWeightConstraint,
    DeveloperTemperatureConstraint,
    ExposureTimeConstraint,
    FOConcentrationConstraint,
    HumidityConstraint,
    MetalRatioConstraint,
    ParameterBoundsConstraint,
    SymbolicConstraint,
)
from ptpd_calibration.mcts.types import CalibrationAction, CalibrationState

pytestmark = pytest.mark.property

PARAM_NAMES = sorted(DEFAULT_PARAMETER_RANGES)
BOUNDS_CONSTRAINTS = {name: ParameterBoundsConstraint(name) for name in PARAM_NAMES}
SOFT_CONSTRAINTS: list[SymbolicConstraint] = [
    FOConcentrationConstraint(),
    MetalRatioConstraint(),
    ExposureTimeConstraint(),
    DeveloperTemperatureConstraint(),
    CoatingWeightConstraint(),
    HumidityConstraint(),
]
SOFT_IDS = [type(c).__name__ for c in SOFT_CONSTRAINTS]
PRUNER = ActionPruner()

WIDE_MIN, WIDE_MAX = -1e6, 1e6
TINY_MAX = 1e-150
"""Magnitudes whose squared penalty underflows to zero (F4 regime)."""
VALUE_MIN, VALUE_MAX = -1e3, 1e3
EXACT_TOL = 1e-12
EDGE_EPS_MAX = 1e-3
BIN_INDEX_MAX = 50
MAX_ACTIONS = 30
MAX_WRONG_ARITY = 4
WRONG_ARITY_BOUNDS_LOSS = 1000.0
WRONG_ARITY_SOFT_LOSS = 100.0
"""Documented sentinel losses for inputs that are not a single value."""

bounds_values = st.one_of(
    st.floats(WIDE_MIN, WIDE_MAX, allow_nan=False, allow_infinity=False),
    st.floats(-TINY_MAX, TINY_MAX, allow_nan=False, allow_infinity=False),
)
soft_values = st.floats(VALUE_MIN, VALUE_MAX, allow_nan=False, allow_infinity=False)
actions = st.lists(
    st.builds(
        CalibrationAction,
        dimension=st.sampled_from(PARAM_NAMES),
        value=soft_values,
        bin_index=st.integers(0, BIN_INDEX_MAX),
    ),
    max_size=MAX_ACTIONS,
)
wrong_arity = st.lists(soft_values, max_size=MAX_WRONG_ARITY).filter(lambda v: len(v) != 1)


def _in_range(name: str, value: float) -> bool:
    rng = DEFAULT_PARAMETER_RANGES[name]
    return rng.min_value <= value <= rng.max_value


def _state() -> CalibrationState:
    return CalibrationState(remaining_dimensions=list(PARAM_NAMES))


@pytest.mark.parametrize("name", PARAM_NAMES)
@given(value=bounds_values)
def test_bounds_satisfied_iff_in_range(name: str, value: float) -> None:
    result = BOUNDS_CONSTRAINTS[name].evaluate(np.array([value]))

    assert result.is_satisfied == _in_range(name, value)
    assert result.loss_value >= 0.0
    if result.is_satisfied:
        assert result.loss_value == 0.0
    assert bool(result.violations) == (not result.is_satisfied)


@pytest.mark.parametrize("constraint", SOFT_CONSTRAINTS, ids=SOFT_IDS)
@given(value=soft_values)
def test_soft_constraint_loss_non_negative_and_consistent(
    constraint: SymbolicConstraint, value: float
) -> None:
    values = np.array([value])
    result = constraint.evaluate(values)

    assert result.loss_value >= 0.0
    assert result.loss_value == constraint.compute_loss(values)
    if result.is_satisfied:
        assert result.loss_value == 0.0
    else:
        assert result.violations


@given(eps=st.floats(0.0, EDGE_EPS_MAX))
def test_fo_penalty_is_continuous_and_symmetric_at_sweet_spot_edges(eps: float) -> None:
    constraint = FOConcentrationConstraint()
    below = constraint.compute_loss(np.array([constraint.sweet_spot_min - eps]))
    above = constraint.compute_loss(np.array([constraint.sweet_spot_max + eps]))

    assert constraint.compute_loss(np.array([constraint.sweet_spot_min])) == 0.0
    assert constraint.compute_loss(np.array([constraint.sweet_spot_max])) == 0.0
    assert below == pytest.approx(above, abs=EXACT_TOL)
    assert below <= eps  # quadratic penalty vanishes at least linearly


@pytest.mark.parametrize("name", PARAM_NAMES)
@given(values=wrong_arity)
def test_bounds_wrong_arity_uses_sentinel(name: str, values: list[float]) -> None:
    result = BOUNDS_CONSTRAINTS[name].evaluate(np.array(values))

    assert not result.is_satisfied
    assert result.loss_value == WRONG_ARITY_BOUNDS_LOSS


@pytest.mark.parametrize("constraint", SOFT_CONSTRAINTS, ids=SOFT_IDS)
@given(values=wrong_arity)
def test_soft_wrong_arity_uses_sentinel(
    constraint: SymbolicConstraint, values: list[float]
) -> None:
    result = constraint.evaluate(np.array(values))

    assert not result.is_satisfied
    assert result.loss_value == WRONG_ARITY_SOFT_LOSS


@given(candidates=actions)
def test_prune_actions_keeps_exactly_in_range_actions_in_order(
    candidates: list[CalibrationAction],
) -> None:
    pruned = PRUNER.prune_actions(_state(), candidates)
    expected = [a for a in candidates if _in_range(a.dimension, a.value)]

    assert pruned == expected
    assert all(kept is original for kept, original in zip(pruned, expected, strict=True))


@given(dimension=st.sampled_from(PARAM_NAMES), value=soft_values)
def test_score_action_in_unit_interval(dimension: str, value: float) -> None:
    score = PRUNER.score_action(
        _state(), CalibrationAction(dimension=dimension, value=value, bin_index=0)
    )

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("dimension", PARAM_NAMES)
def test_default_value_scores_one(dimension: str) -> None:
    default = DEFAULT_PARAMETER_RANGES[dimension].default_value
    score = PRUNER.score_action(
        _state(), CalibrationAction(dimension=dimension, value=default, bin_index=0)
    )

    assert score == pytest.approx(1.0, abs=EXACT_TOL)
