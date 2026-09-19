"""Property-based tests for ``mcts/quality.py`` (TST-08, MR-S12..S16)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings, PhysicsConstants
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.types import SimulationResult
from tests.property.strategies import param_values, sim_params

pytestmark = pytest.mark.property

SCORER = QualityScorer()
SETTINGS = SCORER.settings
PHYSICS = PhysicsConstants()

CURVE_MIN_LEN, CURVE_MAX_LEN = 2, 64
CURVE_MIN_LEN_FOR_SHAPE = 3
DENSITY_UPPER = 4.0
"""Upper clamp of the characteristic curve."""
MIN_RANGE = 0.01
"""Comfortably above the scorer's 1e-6 degenerate-range guard."""
PERFECT_SCORE_MIN = 0.999
EXACT_TOL = 1e-12
AFFINE_TOL = 1e-9
SCALE_MIN, SCALE_MAX = 0.1, 10.0
OFFSET_MIN, OFFSET_MAX = 0.0, 1.0
DEVIATION_MAX = 3.0
"""exp(-(3/sigma)^2) is still a positive double for every allowed sigma."""
STRICT_SEPARATION = 1e-6
METRICS = ("linearity", "dmax", "smoothness", "cost")
WEIGHT_FIELDS = {metric: f"{metric}_weight" for metric in METRICS}


def _field_upper_bound(name: str) -> float:
    return next(m.le for m in MCTSSettings.model_fields[name].metadata if hasattr(m, "le"))


WEIGHT_UPPER = min(_field_upper_bound(f) for f in WEIGHT_FIELDS.values())
WEIGHT_SCALE_MAX = WEIGHT_UPPER / max(getattr(SETTINGS, f) for f in WEIGHT_FIELDS.values())
WEIGHT_SCALE_MIN = 0.1

density_curves = st.lists(
    st.floats(0.0, DENSITY_UPPER, allow_nan=False, allow_infinity=False),
    min_size=CURVE_MIN_LEN,
    max_size=CURVE_MAX_LEN,
)
shape_curves = st.lists(
    st.floats(0.0, DENSITY_UPPER, allow_nan=False, allow_infinity=False),
    min_size=CURVE_MIN_LEN_FOR_SHAPE,
    max_size=CURVE_MAX_LEN,
).filter(lambda c: max(c) - min(c) >= MIN_RANGE)


def _result(curve: list[float], params: dict[str, float]) -> SimulationResult:
    return SimulationResult(
        density_curve=curve,
        dmin=min(curve),
        dmax=max(curve),
        density_range=max(curve) - min(curve),
        gamma=PHYSICS.pd_gamma_base,
        quality_score=0.0,
        parameters=params,
    )


def _sub_scores(scorer: QualityScorer, result: SimulationResult) -> dict[str, float]:
    return {
        "linearity": scorer._linearity_score(result.density_curve),
        "dmax": scorer._dmax_score(result.dmax),
        "smoothness": scorer._smoothness_score(result.density_curve),
        "cost": scorer._cost_score(result.parameters),
    }


@given(curve=density_curves, params=sim_params)
def test_score_and_sub_scores_in_unit_interval(
    curve: list[float], params: dict[str, float]
) -> None:
    result = _result(curve, params)

    assert 0.0 <= SCORER.score(result) <= 1.0
    for value in _sub_scores(SCORER, result).values():
        assert 0.0 <= value <= 1.0


@given(deviation=st.floats(0.0, DEVIATION_MAX))
def test_dmax_score_symmetric_and_maximal_at_target(deviation: float) -> None:
    target = SETTINGS.target_dmax

    assert SCORER._dmax_score(target) == 1.0
    assert SCORER._dmax_score(target + deviation) == pytest.approx(
        SCORER._dmax_score(target - deviation), abs=EXACT_TOL
    )
    assert SCORER._dmax_score(target + deviation) <= 1.0


@given(first=st.floats(0.0, DEVIATION_MAX), second=st.floats(0.0, DEVIATION_MAX))
def test_dmax_score_strictly_decreasing_in_deviation(first: float, second: float) -> None:
    lo, hi = sorted((first, second))
    assume(hi - lo >= STRICT_SEPARATION)
    target = SETTINGS.target_dmax

    assert SCORER._dmax_score(target + lo) > SCORER._dmax_score(target + hi)


@given(
    start=st.floats(0.0, DENSITY_UPPER - MIN_RANGE),
    span=st.floats(MIN_RANGE, 1.0),
    n=st.integers(CURVE_MIN_LEN_FOR_SHAPE, CURVE_MAX_LEN),
)
def test_perfect_ramp_scores_near_one(start: float, span: float, n: int) -> None:
    ramp = [float(v) for v in np.linspace(start, start + span, n)]

    assert SCORER._linearity_score(ramp) > PERFECT_SCORE_MIN
    assert SCORER._smoothness_score(ramp) > PERFECT_SCORE_MIN


@given(level=st.floats(0.0, DENSITY_UPPER), n=st.integers(CURVE_MIN_LEN_FOR_SHAPE, CURVE_MAX_LEN))
def test_flat_curve_has_zero_linearity_and_perfect_smoothness(level: float, n: int) -> None:
    flat = [level] * n

    assert SCORER._linearity_score(flat) == 0.0
    assert SCORER._smoothness_score(flat) == 1.0


@given(
    curve=shape_curves,
    scale=st.floats(SCALE_MIN, SCALE_MAX),
    offset=st.floats(OFFSET_MIN, OFFSET_MAX),
)
def test_shape_scores_are_affine_invariant(curve: list[float], scale: float, offset: float) -> None:
    """MR-S12: both shape scores are normalised by the density range."""
    transformed = [scale * v + offset for v in curve]

    assert SCORER._linearity_score(transformed) == pytest.approx(
        SCORER._linearity_score(curve), abs=AFFINE_TOL
    )
    assert SCORER._smoothness_score(transformed) == pytest.approx(
        SCORER._smoothness_score(curve), abs=AFFINE_TOL
    )


@given(curve=shape_curves)
def test_linearity_against_itself_is_perfect(curve: list[float]) -> None:
    """MR-S13: a curve compared with itself as target scores exactly 1."""
    assert SCORER._linearity_score(curve, target_curve=curve) == 1.0


@pytest.mark.parametrize("metric", METRICS)
@given(curve=density_curves, params=sim_params)
def test_single_weight_score_equals_sub_score(
    metric: str, curve: list[float], params: dict[str, float]
) -> None:
    weights = dict.fromkeys(WEIGHT_FIELDS.values(), 0.0)
    weights[WEIGHT_FIELDS[metric]] = WEIGHT_UPPER
    scorer = QualityScorer(MCTSSettings(**weights))
    result = _result(curve, params)

    assert scorer.score(result) == pytest.approx(_sub_scores(scorer, result)[metric], abs=EXACT_TOL)


@given(
    curve=density_curves, params=sim_params, factor=st.floats(WEIGHT_SCALE_MIN, WEIGHT_SCALE_MAX)
)
def test_score_invariant_under_weight_scaling(
    curve: list[float], params: dict[str, float], factor: float
) -> None:
    """MR-S16: multiplying all four weights by k > 0 leaves the score unchanged."""
    scaled = MCTSSettings(
        **{field: factor * getattr(SETTINGS, field) for field in WEIGHT_FIELDS.values()}
    )
    result = _result(curve, params)

    assert QualityScorer(scaled).score(result) == pytest.approx(SCORER.score(result), abs=EXACT_TOL)


@given(
    first=param_values("metal_ratio"),
    second=param_values("metal_ratio"),
    coating=param_values("coating_weight"),
)
def test_cost_score_non_increasing_in_metal_ratio(
    first: float, second: float, coating: float
) -> None:
    """MR-S15 (metal): more platinum never improves the cost score."""
    lo, hi = sorted((first, second))

    assert SCORER._cost_score({"metal_ratio": lo, "coating_weight": coating}) >= (
        SCORER._cost_score({"metal_ratio": hi, "coating_weight": coating}) - EXACT_TOL
    )


@given(
    first=param_values("coating_weight"),
    second=param_values("coating_weight"),
    metal_ratio=param_values("metal_ratio"),
)
def test_cost_score_non_increasing_in_coating_weight(
    first: float, second: float, metal_ratio: float
) -> None:
    """MR-S15 (coating): heavier coating never improves the cost score."""
    lo, hi = sorted((first, second))

    assert SCORER._cost_score({"metal_ratio": metal_ratio, "coating_weight": lo}) >= (
        SCORER._cost_score({"metal_ratio": metal_ratio, "coating_weight": hi}) - EXACT_TOL
    )


def test_cost_score_is_one_at_cheapest_corner() -> None:
    cheapest = {
        "metal_ratio": DEFAULT_PARAMETER_RANGES["metal_ratio"].min_value,
        "coating_weight": DEFAULT_PARAMETER_RANGES["coating_weight"].min_value,
    }

    assert SCORER._cost_score(cheapest) == pytest.approx(1.0, abs=EXACT_TOL)
