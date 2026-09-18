"""Shared Hypothesis strategies for the property-based suites.

All numeric bounds live here as named constants so that individual test
modules never hard-code tolerances or ranges.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, PhysicsConstants

# --- unit-interval values -----------------------------------------------------
UNIT_MIN = 0.0
UNIT_MAX = 1.0
FLOAT_WIDTH = 32
"""Curve values are drawn from float32-representable numbers.

Calibration curves originate from 8/16-bit device data, so single precision is
more than enough resolution. Restricting the width also keeps every finite
difference between two drawn values >= the float32 sub-normal minimum, which
prevents ``scipy.interpolate.PchipInterpolator`` from overflowing when it
divides by a slope built from float64 sub-normals such as ``5e-324`` (a
RuntimeWarning that is an artefact of the strategy, not of the curve code).
"""

unit = st.floats(UNIT_MIN, UNIT_MAX, width=FLOAT_WIDTH, allow_nan=False, allow_infinity=False)
"""A float in the closed unit interval."""

signed_unit = st.floats(
    -UNIT_MAX, UNIT_MAX, width=FLOAT_WIDTH, allow_nan=False, allow_infinity=False
)
"""A float in [-1, 1] (adjustment amounts)."""

# --- curves ------------------------------------------------------------------
CURVE_MIN_POINTS = 3
CURVE_MAX_POINTS = 256
CURVE_NAME = "c"


@st.composite
def curves(
    draw: st.DrawFn,
    n_min: int = CURVE_MIN_POINTS,
    n_max: int = CURVE_MAX_POINTS,
    monotone: bool = False,
    anchored: bool = False,
) -> CurveData:
    """Draw a :class:`CurveData` with ``linspace(0, 1, n)`` inputs.

    Args:
        n_min: Minimum number of points.
        n_max: Maximum number of points.
        monotone: Sort the outputs so the curve is non-decreasing.
        anchored: Force the endpoints to exactly ``0.0`` and ``1.0``.
    """
    n = draw(st.integers(n_min, n_max))
    ys = draw(st.lists(unit, min_size=n, max_size=n))
    if monotone:
        ys = sorted(ys)
    if anchored:
        ys[0], ys[-1] = UNIT_MIN, UNIT_MAX
    return CurveData(
        name=CURVE_NAME,
        input_values=[float(x) for x in np.linspace(UNIT_MIN, UNIT_MAX, n)],
        output_values=ys,
    )


# --- step-tablet densities ---------------------------------------------------
DENSITY_MIN = 0.0
DENSITY_MAX = 3.0
DENSITY_MIN_STEPS = 3
DENSITY_MAX_STEPS = 41
DENSITY_MIN_RANGE = 0.02
"""Minimum dmax - dmin; the generator rejects ranges below 0.01."""

densities = (
    st.lists(
        st.floats(DENSITY_MIN, DENSITY_MAX, allow_nan=False, allow_infinity=False),
        min_size=DENSITY_MIN_STEPS,
        max_size=DENSITY_MAX_STEPS,
    )
    .map(sorted)
    .filter(lambda d: d[-1] - d[0] >= DENSITY_MIN_RANGE)
)
"""Sorted (non-decreasing, duplicates allowed) step-tablet densities."""

STRICT_DENSITY_START_MAX = 0.5
STRICT_DENSITY_MIN_GAP = 1e-3
STRICT_DENSITY_MAX_GAP = 0.06
"""Gap bounds keep strict densities within ~[0, DENSITY_MAX + 0.5]."""


@st.composite
def strict_densities(
    draw: st.DrawFn,
    n_min: int = DENSITY_MIN_STEPS,
    n_max: int = DENSITY_MAX_STEPS,
    min_gap: float = STRICT_DENSITY_MIN_GAP,
    max_gap: float = STRICT_DENSITY_MAX_GAP,
) -> list[float]:
    """Draw strictly increasing densities with a guaranteed minimum step gap.

    Used where the code under test inverts the measured response with
    ``interp1d``/``UnivariateSpline`` (which require strictly increasing
    abscissae) or where a plateau would make the inverse discontinuous.
    """
    n = draw(st.integers(n_min, n_max))
    start = draw(st.floats(DENSITY_MIN, STRICT_DENSITY_START_MAX, allow_nan=False))
    gaps = draw(
        st.lists(
            st.floats(min_gap, max_gap, allow_nan=False, allow_infinity=False),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    values = start + np.concatenate([[0.0], np.cumsum(gaps)])
    return [float(v) for v in values]


# --- simulator parameters ----------------------------------------------------
_PHYSICS = PhysicsConstants()


def _symmetric_bounds(name: str, centre: float) -> tuple[float, float]:
    """Largest window inside the parameter range that is symmetric about ``centre``."""
    rng = DEFAULT_PARAMETER_RANGES[name]
    half = min(centre - rng.min_value, rng.max_value - centre)
    return (centre - half, centre + half)


SIM_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    name: (rng.min_value, rng.max_value) for name, rng in DEFAULT_PARAMETER_RANGES.items()
}
# Humidity is restricted to a window symmetric about the physics optimum so the
# reflection h -> 2*optimum - h (MR6) stays inside the parameter range.
SIM_PARAM_BOUNDS["humidity"] = _symmetric_bounds("humidity", _PHYSICS.humidity_optimal)

sim_params = st.fixed_dictionaries(
    {
        name: st.floats(lo, hi, allow_nan=False, allow_infinity=False)
        for name, (lo, hi) in SIM_PARAM_BOUNDS.items()
    }
)
"""A complete calibration-parameter dictionary for ``ExtendedProcessSimulator``."""


def param_values(name: str) -> st.SearchStrategy[float]:
    """Values for a single simulator dimension, within ``SIM_PARAM_BOUNDS``."""
    lo, hi = SIM_PARAM_BOUNDS[name]
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)
