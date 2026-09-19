"""Property-based tests for ``ptpd_calibration.curves.modifier`` (TST-05).

Invariants follow docs/plans/2026-09-review/expert-testing.md section 5. The
F1 (spline knots) and F2 (endpoint pinning) defects are fixed in the modifier,
so the corresponding properties are asserted for all curves rather than being
marked ``xfail``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.modifier import BlendMode, CurveModifier, SmoothingMethod
from tests.property.strategies import CURVE_MAX_POINTS, curves, signed_unit, unit

pytestmark = pytest.mark.property

MODIFIER = CurveModifier()

IDENTITY_TOL = 1e-12
"""Neutral adjustments may differ from the input by float rounding only."""
EXACT_TOL = 1e-12
"""Tolerance for relations that hold exactly up to float rounding."""
NESTED_KNOT_TOL = 1e-9
"""Knot reproduction on nested grids (grid points may differ by an ulp)."""
NEUTRAL_AMOUNT = 0.0
NEUTRAL_GAMMA = 1.0
GAMMA_MIN, GAMMA_MAX = 0.1, 10.0
RESAMPLE_MIN_POINTS, RESAMPLE_MAX_POINTS = 3, 300
NESTED_MAX_FACTOR = 8
NESTED_MAX_SOURCE_POINTS = 64
LEVELS_BLACK_MAX = 0.4
LEVELS_WHITE_MIN = 0.6
LEVELS_MIDPOINT_MIN, LEVELS_MIDPOINT_MAX = 0.05, 0.95
SPLINE_SHORT_MAX_POINTS = 9
"""F1: curves shorter than ``MIN_SPLINE_KNOTS`` used to crash spline smoothing."""
CUBIC_MIN_POINTS = 4
"""``resample(method="cubic")`` uses a cubic spline, which scipy cannot build from
fewer than four knots (it raises ValueError); the method is therefore only
exercised from four points upwards."""

Adjustment = Callable[[CurveData, float], CurveData]


def _gamma_from_unit(amount: float) -> float:
    """Map an amount in [-1, 1] onto the documented gamma range."""
    return GAMMA_MIN + (amount + 1.0) / 2.0 * (GAMMA_MAX - GAMMA_MIN)


ADJUSTMENTS: dict[str, Adjustment] = {
    "brightness": lambda c, a: MODIFIER.adjust_brightness(c, a),
    "contrast": lambda c, a: MODIFIER.adjust_contrast(c, a),
    "gamma": lambda c, a: MODIFIER.adjust_gamma(c, _gamma_from_unit(a)),
    "highlights": lambda c, a: MODIFIER.adjust_highlights(c, a),
    "shadows": lambda c, a: MODIFIER.adjust_shadows(c, a),
    "midtones": lambda c, a: MODIFIER.adjust_midtones(c, a),
}
ADJUSTMENT_NAMES = sorted(ADJUSTMENTS)

NEUTRAL_ADJUSTMENTS: dict[str, Callable[[CurveData], CurveData]] = {
    "brightness": lambda c: MODIFIER.adjust_brightness(c, NEUTRAL_AMOUNT),
    "contrast": lambda c: MODIFIER.adjust_contrast(c, NEUTRAL_AMOUNT),
    "gamma": lambda c: MODIFIER.adjust_gamma(c, NEUTRAL_GAMMA),
    "highlights": lambda c: MODIFIER.adjust_highlights(c, NEUTRAL_AMOUNT),
    "shadows": lambda c: MODIFIER.adjust_shadows(c, NEUTRAL_AMOUNT),
    "midtones": lambda c: MODIFIER.adjust_midtones(c, NEUTRAL_AMOUNT),
}

# Which endpoints each adjustment pins when preserve_endpoints is set.
PINS_FIRST = {"brightness", "contrast", "gamma", "shadows", "midtones"}
PINS_LAST = {"brightness", "contrast", "gamma", "highlights", "midtones"}

SHAPE_PRESERVING_SMOOTHERS = [
    SmoothingMethod.GAUSSIAN,
    SmoothingMethod.MOVING_AVERAGE,
    SmoothingMethod.SPLINE,
]
COMMUTATIVE_BLEND_MODES = [
    BlendMode.AVERAGE,
    BlendMode.MULTIPLY,
    BlendMode.SCREEN,
    BlendMode.MIN,
    BlendMode.MAX,
]
SHAPE_PRESERVING_RESAMPLERS = ["pchip", "linear"]


def _arr(curve: CurveData) -> np.ndarray:
    return np.asarray(curve.output_values, dtype=float)


def _is_monotone(values: np.ndarray, tol: float = 0.0) -> bool:
    return bool(np.all(np.diff(values) >= -tol))


def _assert_bounded_unit(values: np.ndarray) -> None:
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


# --- adjust_* -----------------------------------------------------------------


@pytest.mark.parametrize("name", ADJUSTMENT_NAMES)
@given(curve=curves())
def test_neutral_amount_is_identity(name: str, curve: CurveData) -> None:
    """gamma=1 / amount=0 changes nothing beyond rounding and appends a note."""
    result = NEUTRAL_ADJUSTMENTS[name](curve)

    assert len(result.output_values) == len(curve.output_values)
    np.testing.assert_allclose(_arr(result), _arr(curve), atol=IDENTITY_TOL, rtol=0.0)
    assert result.notes is not None and name in result.notes


@pytest.mark.parametrize("name", ADJUSTMENT_NAMES)
@given(curve=curves(), amount=signed_unit)
def test_adjustment_keeps_bounds_and_length(name: str, curve: CurveData, amount: float) -> None:
    result = ADJUSTMENTS[name](curve, amount)

    assert len(result.output_values) == len(curve.output_values)
    assert result.input_values == curve.input_values
    _assert_bounded_unit(_arr(result))


@pytest.mark.parametrize("name", ADJUSTMENT_NAMES)
@given(curve=curves(monotone=True), amount=signed_unit)
def test_monotone_in_monotone_out(name: str, curve: CurveData, amount: float) -> None:
    """F2: holds for every monotone curve, not only those anchored at 0->1."""
    result = ADJUSTMENTS[name](curve, amount)
    out = _arr(result)

    assert _is_monotone(out)
    if name in PINS_FIRST:
        assert out[0] == curve.output_values[0]
    if name in PINS_LAST:
        assert out[-1] == curve.output_values[-1]


@pytest.mark.parametrize("name", ADJUSTMENT_NAMES)
@given(curve=curves(monotone=True, anchored=True), amount=signed_unit)
def test_anchored_curves_stay_anchored_and_monotone(
    name: str, curve: CurveData, amount: float
) -> None:
    result = ADJUSTMENTS[name](curve, amount)
    out = _arr(result)

    assert _is_monotone(out)
    if name in PINS_FIRST:
        assert out[0] == 0.0
    if name in PINS_LAST:
        assert out[-1] == 1.0


@given(curve=curves(), amount=unit)
def test_positive_brightness_never_darkens(curve: CurveData, amount: float) -> None:
    result = MODIFIER.adjust_brightness(curve, amount)

    assert np.all(_arr(result) >= _arr(curve) - EXACT_TOL)


@given(curve=curves(), amount=unit)
def test_negative_brightness_never_brightens(curve: CurveData, amount: float) -> None:
    result = MODIFIER.adjust_brightness(curve, -amount)

    assert np.all(_arr(result) <= _arr(curve) + EXACT_TOL)


@given(curve=curves())
def test_levels_default_is_identity_except_pinned_endpoints(curve: CurveData) -> None:
    """black=0, white=1, mid=0.5 is the identity; endpoints are pinned to 0.0/1.0."""
    result = MODIFIER.adjust_levels(curve)
    out = _arr(result)

    assert out[0] == 0.0
    assert out[-1] == 1.0
    np.testing.assert_allclose(out[1:-1], _arr(curve)[1:-1], atol=IDENTITY_TOL, rtol=0.0)


@given(
    curve=curves(monotone=True),
    black=st.floats(0.0, LEVELS_BLACK_MAX),
    white=st.floats(LEVELS_WHITE_MIN, 1.0),
    midpoint=st.floats(LEVELS_MIDPOINT_MIN, LEVELS_MIDPOINT_MAX),
)
def test_levels_keeps_monotonicity_and_bounds(
    curve: CurveData, black: float, white: float, midpoint: float
) -> None:
    result = MODIFIER.adjust_levels(curve, black, white, midpoint)
    out = _arr(result)

    assert len(out) == len(curve.output_values)
    _assert_bounded_unit(out)
    assert _is_monotone(out)
    assert out[0] == 0.0
    assert out[-1] == 1.0


# --- smooth -------------------------------------------------------------------


@pytest.mark.parametrize("method", list(SmoothingMethod))
@given(curve=curves(), strength=unit)
def test_smooth_keeps_length_bounds_and_endpoints(
    method: SmoothingMethod, curve: CurveData, strength: float
) -> None:
    result = MODIFIER.smooth(curve, method, strength)
    out = _arr(result)

    assert len(out) == len(curve.output_values)
    _assert_bounded_unit(out)
    assert out[0] == curve.output_values[0]
    assert out[-1] == curve.output_values[-1]


@pytest.mark.parametrize("method", SHAPE_PRESERVING_SMOOTHERS)
@given(curve=curves(monotone=True), strength=unit)
def test_smooth_preserves_monotonicity(
    method: SmoothingMethod, curve: CurveData, strength: float
) -> None:
    """Savitzky-Golay is excluded: it is a polynomial fit and may overshoot (F5)."""
    result = MODIFIER.smooth(curve, method, strength)

    assert _is_monotone(_arr(result))


@given(curve=curves(n_max=SPLINE_SHORT_MAX_POINTS), strength=unit)
def test_spline_smoothing_never_raises_for_short_curves(curve: CurveData, strength: float) -> None:
    """F1 regression: every n >= 3 must be accepted."""
    result = MODIFIER.smooth(curve, SmoothingMethod.SPLINE, strength)

    assert len(result.output_values) == len(curve.output_values)


# --- enforce_monotonicity -----------------------------------------------------


@given(curve=curves())
def test_enforce_monotonicity_is_idempotent_and_dominates_input(curve: CurveData) -> None:
    increasing = MODIFIER.enforce_monotonicity(curve)
    out = _arr(increasing)

    assert _is_monotone(out)
    assert np.all(out >= _arr(curve))
    assert out[0] == curve.output_values[0]
    assert np.array_equal(_arr(MODIFIER.enforce_monotonicity(increasing)), out)


@given(curve=curves())
def test_decreasing_enforcement_is_mirror_of_increasing(curve: CurveData) -> None:
    """enforce(invert(c), "decreasing") == invert(enforce(c, "increasing")) exactly."""
    via_invert = MODIFIER.invert(MODIFIER.enforce_monotonicity(curve, "increasing"))
    direct = MODIFIER.enforce_monotonicity(MODIFIER.invert(curve), "decreasing")

    assert np.array_equal(_arr(direct), _arr(via_invert))


# --- blend --------------------------------------------------------------------


@pytest.mark.parametrize("mode", COMMUTATIVE_BLEND_MODES)
@given(a=curves(), b=curves())
def test_blend_is_commutative(mode: BlendMode, a: CurveData, b: CurveData) -> None:
    ab = MODIFIER.blend(a, b, mode)
    ba = MODIFIER.blend(b, a, mode)

    assert len(ab.output_values) == max(len(a.output_values), len(b.output_values))
    np.testing.assert_allclose(_arr(ab), _arr(ba), atol=EXACT_TOL, rtol=0.0)
    _assert_bounded_unit(_arr(ab))


@given(a=curves(), b=curves())
def test_weighted_blend_extremes_select_one_curve(a: CurveData, b: CurveData) -> None:
    n = max(len(a.output_values), len(b.output_values))
    grid = np.linspace(0.0, 1.0, n)

    only_a = MODIFIER.blend(a, b, BlendMode.WEIGHTED, weight=0.0)
    only_b = MODIFIER.blend(a, b, BlendMode.WEIGHTED, weight=1.0)

    assert np.array_equal(_arr(only_a), np.interp(grid, a.input_values, a.output_values))
    assert np.array_equal(_arr(only_b), np.interp(grid, b.input_values, b.output_values))


@given(a=curves(), b=curves())
def test_multiply_below_min_and_screen_above_max(a: CurveData, b: CurveData) -> None:
    lo = _arr(MODIFIER.blend(a, b, BlendMode.MIN))
    hi = _arr(MODIFIER.blend(a, b, BlendMode.MAX))

    assert np.all(_arr(MODIFIER.blend(a, b, BlendMode.MULTIPLY)) <= lo + EXACT_TOL)
    assert np.all(_arr(MODIFIER.blend(a, b, BlendMode.SCREEN)) >= hi - EXACT_TOL)


@given(curve=curves())
def test_blending_a_curve_with_itself_is_resampling(curve: CurveData) -> None:
    blended = MODIFIER.blend(curve, curve, BlendMode.AVERAGE)
    resampled = MODIFIER.resample(curve, len(curve.output_values))

    np.testing.assert_allclose(_arr(blended), _arr(resampled), atol=EXACT_TOL, rtol=0.0)


# --- resample -----------------------------------------------------------------


@pytest.mark.parametrize("method", SHAPE_PRESERVING_RESAMPLERS)
@given(
    curve=curves(monotone=True),
    num_points=st.integers(RESAMPLE_MIN_POINTS, RESAMPLE_MAX_POINTS),
)
def test_resample_length_bounds_monotone(method: str, curve: CurveData, num_points: int) -> None:
    result = MODIFIER.resample(curve, num_points, method)
    out = _arr(result)

    assert len(out) == num_points
    assert np.array_equal(np.asarray(result.input_values), np.linspace(0.0, 1.0, num_points))
    _assert_bounded_unit(out)
    assert _is_monotone(out, tol=EXACT_TOL)


@given(
    curve=curves(n_min=CUBIC_MIN_POINTS, monotone=True),
    num_points=st.integers(RESAMPLE_MIN_POINTS, RESAMPLE_MAX_POINTS),
)
def test_cubic_resample_bounds_only(curve: CurveData, num_points: int) -> None:
    """'cubic' is not shape-preserving, so only length and bounds are claimed."""
    out = _arr(MODIFIER.resample(curve, num_points, "cubic"))

    assert len(out) == num_points
    _assert_bounded_unit(out)


@pytest.mark.parametrize("method", SHAPE_PRESERVING_RESAMPLERS)
@given(
    curve=curves(n_max=NESTED_MAX_SOURCE_POINTS),
    factor=st.integers(1, NESTED_MAX_FACTOR),
)
def test_resample_on_nested_grid_reproduces_knots(
    method: str, curve: CurveData, factor: int
) -> None:
    """When (N-1) is a multiple of (n-1) the old knots lie on the new grid."""
    n = len(curve.output_values)
    up = MODIFIER.resample(curve, factor * (n - 1) + 1, method)
    back = MODIFIER.resample(up, n, method)

    np.testing.assert_allclose(_arr(up)[::factor], _arr(curve), atol=NESTED_KNOT_TOL, rtol=0.0)
    np.testing.assert_allclose(_arr(back), _arr(curve), atol=NESTED_KNOT_TOL, rtol=0.0)


@pytest.mark.parametrize("method", SHAPE_PRESERVING_RESAMPLERS)
@given(curve=curves(), num_points=st.integers(RESAMPLE_MIN_POINTS, CURVE_MAX_POINTS))
def test_resample_is_idempotent(method: str, curve: CurveData, num_points: int) -> None:
    once = MODIFIER.resample(curve, num_points, method)
    twice = MODIFIER.resample(once, num_points, method)

    np.testing.assert_allclose(_arr(twice), _arr(once), atol=EXACT_TOL, rtol=0.0)


# --- invert / reverse ---------------------------------------------------------


@given(curve=curves())
def test_invert_is_an_involution(curve: CurveData) -> None:
    np.testing.assert_allclose(
        _arr(MODIFIER.invert(MODIFIER.invert(curve))), _arr(curve), atol=EXACT_TOL, rtol=0.0
    )


@given(curve=curves())
def test_reverse_is_an_involution(curve: CurveData) -> None:
    assert np.array_equal(_arr(MODIFIER.reverse(MODIFIER.reverse(curve))), _arr(curve))
