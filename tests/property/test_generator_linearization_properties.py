"""Property-based tests for ``curves/generator.py`` and ``curves/linearization.py`` (TST-06)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.generator import CurveGenerator
from ptpd_calibration.curves.linearization import (
    AutoLinearizer,
    LinearizationMethod,
    TargetResponse,
)
from tests.property.strategies import (
    DENSITY_MAX,
    DENSITY_MAX_STEPS,
    DENSITY_MIN,
    DENSITY_MIN_STEPS,
    densities,
    strict_densities,
)

pytestmark = pytest.mark.property

GENERATOR = CurveGenerator()
OUTPUT_POINTS = GENERATOR.settings.num_output_points
LINEARIZER = AutoLinearizer()
LINEARIZATION_POINTS = LINEARIZER.config.output_points

EXACT_TOL = 1e-12
AFFINE_TOL = 1e-6
LINEAR_IDENTITY_TOL = 0.02
REFINE_NOOP_TOL = 1e-6
LINEAR_DENSITY_MIN, LINEAR_DENSITY_MAX = 0.1, 2.0
SCALE_MIN, SCALE_MAX = 0.1, 10.0
OFFSET_MIN, OFFSET_MAX = 0.0, 1.0
NEAR_CONSTANT_JITTER_MAX = 0.004
"""Range below the generator's 0.01 minimum even after 41 draws."""
MIN_STEPS_FOR_GENERATE = 2
GENERATOR_MIN_RANGE = 0.01
"""Density range below which ``CurveGenerator.generate`` raises ValueError."""

POLYNOMIAL_MIN_STEPS = LINEARIZER.config.polynomial_degree + 1
"""np.polyfit needs degree + 1 points; fewer emit RankWarning."""

CURVE_TYPES = [CurveType.LINEAR, CurveType.PAPER_WHITE, CurveType.AESTHETIC]
LINEARIZATION_METHODS = list(LinearizationMethod)


def _arr(curve: CurveData) -> np.ndarray:
    return np.asarray(curve.output_values, dtype=float)


def _is_monotone(values: np.ndarray, tol: float = 0.0) -> bool:
    return bool(np.all(np.diff(values) >= -tol))


def _linear_densities(n: int) -> list[float]:
    return [float(d) for d in np.linspace(LINEAR_DENSITY_MIN, LINEAR_DENSITY_MAX, n)]


# --- CurveGenerator -----------------------------------------------------------


@pytest.mark.parametrize("curve_type", CURVE_TYPES)
@given(measured=densities)
def test_generate_bounds_monotone_length(curve_type: CurveType, measured: list[float]) -> None:
    curve = GENERATOR.generate(measured, curve_type=curve_type)
    out = _arr(curve)

    assert len(out) == OUTPUT_POINTS
    assert np.array_equal(np.asarray(curve.input_values), np.linspace(0.0, 1.0, OUTPUT_POINTS))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert _is_monotone(out)


@given(
    measured=strict_densities(),
    scale=st.floats(SCALE_MIN, SCALE_MAX),
    offset=st.floats(OFFSET_MIN, OFFSET_MAX),
)
def test_generate_is_affine_invariant(measured: list[float], scale: float, offset: float) -> None:
    """Densities are normalised first, so k*d + b generates the same curve."""
    # Both the original and the scaled densities must clear the generator's
    # minimum range, otherwise generate() raises rather than being compared.
    assume(min(1.0, scale) * (measured[-1] - measured[0]) >= GENERATOR_MIN_RANGE)
    transformed = [scale * d + offset for d in measured]

    np.testing.assert_allclose(
        _arr(GENERATOR.generate(transformed)),
        _arr(GENERATOR.generate(measured)),
        atol=AFFINE_TOL,
        rtol=0.0,
    )


@given(n=st.integers(DENSITY_MIN_STEPS, DENSITY_MAX_STEPS))
def test_linear_densities_generate_identity(n: int) -> None:
    curve = GENERATOR.generate(_linear_densities(n))

    assert np.max(np.abs(_arr(curve) - np.asarray(curve.input_values))) < LINEAR_IDENTITY_TOL


@given(measured=st.lists(st.floats(DENSITY_MIN, DENSITY_MAX), max_size=MIN_STEPS_FOR_GENERATE - 1))
def test_generate_rejects_fewer_than_two_steps(measured: list[float]) -> None:
    with pytest.raises(ValueError):
        GENERATOR.generate(measured)


@given(
    base=st.floats(DENSITY_MIN, DENSITY_MAX),
    jitter=st.lists(
        st.floats(0.0, NEAR_CONSTANT_JITTER_MAX),
        min_size=MIN_STEPS_FOR_GENERATE,
        max_size=DENSITY_MAX_STEPS,
    ),
)
def test_generate_rejects_near_constant_densities(base: float, jitter: list[float]) -> None:
    with pytest.raises(ValueError):
        GENERATOR.generate([base + j for j in jitter])


# --- AutoLinearizer -----------------------------------------------------------


@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
@given(measured=strict_densities(n_min=POLYNOMIAL_MIN_STEPS))
def test_linearize_bounds_length_errors_endpoints(
    method: LinearizationMethod, measured: list[float]
) -> None:
    """Every method: 256 points in [0,1], rms <= max deviation, pinned endpoints, monotone."""
    result = LINEARIZER.linearize(measured, method=method)
    out = _arr(result.curve)

    assert len(out) == LINEARIZATION_POINTS
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert result.residual_error >= 0.0
    assert result.max_deviation >= result.residual_error - EXACT_TOL
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert _is_monotone(out)


@pytest.mark.parametrize(
    "method",
    [m for m in LINEARIZATION_METHODS if m != LinearizationMethod.POLYNOMIAL_FIT],
)
@given(measured=strict_densities())
def test_linearize_accepts_few_steps(method: LinearizationMethod, measured: list[float]) -> None:
    """Non-polynomial methods must work from three steps upwards."""
    result = LINEARIZER.linearize(measured, method=method)
    out = _arr(result.curve)

    assert len(out) == LINEARIZATION_POINTS
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert _is_monotone(out)


@given(n=st.integers(DENSITY_MIN_STEPS, DENSITY_MAX_STEPS))
def test_direct_inversion_of_linear_densities_is_identity(n: int) -> None:
    result = LINEARIZER.linearize(_linear_densities(n), method=LinearizationMethod.DIRECT_INVERSION)

    grid = np.asarray(result.curve.input_values)
    assert np.max(np.abs(_arr(result.curve) - grid)) < LINEAR_IDENTITY_TOL


@given(n=st.integers(POLYNOMIAL_MIN_STEPS, DENSITY_MAX_STEPS))
def test_refine_with_perfect_measurements_is_a_no_op(n: int) -> None:
    """Zero error => zero correction: the identity curve comes back unchanged."""
    grid = np.linspace(0.0, 1.0, LINEARIZATION_POINTS)
    identity = CurveData(name="identity", input_values=list(grid), output_values=list(grid))
    perfect = [float(v) for v in np.linspace(0.0, 1.0, n)]  # == TargetResponse.LINEAR

    result = LINEARIZER.refine_curve(identity, perfect)

    assert result.target_response == TargetResponse.LINEAR
    assert result.residual_error == 0.0
    assert result.max_deviation == 0.0
    np.testing.assert_allclose(_arr(result.curve), grid, atol=REFINE_NOOP_TOL, rtol=0.0)
