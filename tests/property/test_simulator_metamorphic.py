"""Metamorphic relations MR1-MR6 for ``mcts/simulator.py`` (TST-09).

All relations are verified on the NumPy path (``simulate_with_numpy``), which
``_numpy_characteristic_curve`` documents as a line-for-line replica of the
torch ``CharacteristicCurve.forward``; the torch-vs-NumPy differential is a
separate item (TST-10). MR-S5 and MR-S17 are strict ``xfail``: they document
the dead ``ferric_oxalate_pct`` parameter and the flat objective (SCI-01) and
will start failing-as-passing once that work lands.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.stats import qmc

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator
from ptpd_calibration.mcts.types import SimulationResult
from tests.property.strategies import SIM_PARAM_BOUNDS, param_values, sim_params

pytestmark = pytest.mark.property

SIM = ExtendedProcessSimulator()
PHYSICS = SIM.physics
SCORER = QualityScorer()

MONOTONE_TOL = 1e-12
CURVE_TOL = 1e-9
GAMMA_TOL = 1e-12
SYMMETRY_TOL = 1e-12
DEFAULT_STEPS = 21
STEPS_MIN, STEPS_MAX = 3, 101
DENSITY_UPPER = 4.0
"""Upper clamp applied to dmax by the characteristic curve."""
SHOULDER_MIN, SHOULDER_MAX = 0.5, 1.0
"""Clip window for shoulder_position in compute_process_parameters."""
FO_DELTA_MIN = 1e-3
SOBOL_LOG2_SAMPLES = 11
"""2^11 = 2048 >= the 2000 samples requested by MR-S17."""
SOBOL_SEED = 0
OBJECTIVE_SPREAD_MIN = 0.3
HUMIDITY_HALF_WINDOW = PHYSICS.humidity_optimal - SIM_PARAM_BOUNDS["humidity"][0]
DEFAULTS = {name: rng.default_value for name, rng in DEFAULT_PARAMETER_RANGES.items()}
HUMIDITY_TOE_CLIP = DEFAULT_PARAMETER_RANGES["humidity"].max_value
"""At the humidity maximum the toe clips to 0.0 (log(0) guard in the simulator)."""
DEV_TEMP_SHOULDER_CLIP = DEFAULT_PARAMETER_RANGES["developer_temp"].max_value

steps = st.integers(STEPS_MIN, STEPS_MAX)


def _simulate(params: dict[str, float], num_steps: int = DEFAULT_STEPS) -> SimulationResult:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # MR-S11: no RuntimeWarning for in-range params
        return SIM.simulate_with_numpy(params, num_steps)


def _curve(result: SimulationResult) -> np.ndarray:
    return np.asarray(result.density_curve, dtype=float)


# --- MR1 / MR2: dmax monotone in exposure and coating -------------------------


@pytest.mark.parametrize("dimension", ["exposure_time", "coating_weight"])
@given(params=sim_params, data=st.data())
def test_dmax_non_decreasing_in_exposure_and_coating(
    dimension: str, params: dict[str, float], data: st.DataObject
) -> None:
    lo, hi = sorted((data.draw(param_values(dimension)), data.draw(param_values(dimension))))
    low = {**params, dimension: lo}
    high = {**params, dimension: hi}

    assert SIM.compute_process_parameters(high).dmax >= (
        SIM.compute_process_parameters(low).dmax - MONOTONE_TOL
    )
    assert _simulate(high).dmax >= _simulate(low).dmax - MONOTONE_TOL


# --- MR3: grid refinement -----------------------------------------------------


@given(params=sim_params, n=steps)
def test_grid_refinement_is_consistent(params: dict[str, float], n: int) -> None:
    coarse = _curve(_simulate(params, n))
    fine = _curve(_simulate(params, 2 * n - 1))

    np.testing.assert_allclose(fine[::2], coarse, atol=CURVE_TOL, rtol=0.0)


# --- MR4: curve shape, bounds, endpoint consistency ---------------------------


@given(params=sim_params, n=steps)
@example(params=DEFAULTS, n=DEFAULT_STEPS)
def test_density_curve_monotone_bounded_and_consistent(params: dict[str, float], n: int) -> None:
    result = _simulate(params, n)
    curve = _curve(result)

    assert len(curve) == n
    assert np.all(np.diff(curve) >= -CURVE_TOL)
    assert np.all(curve >= result.dmin) and np.all(curve <= DENSITY_UPPER)
    assert result.dmin == float(np.min(curve))
    assert result.dmax == float(np.max(curve))
    assert result.density_range == result.dmax - result.dmin
    assert curve[0] == result.dmin
    assert curve[-1] == result.dmax


@pytest.mark.parametrize(
    "overrides",
    [
        {"humidity": HUMIDITY_TOE_CLIP},
        {"developer_temp": DEV_TEMP_SHOULDER_CLIP},
    ],
    ids=["toe_clips_to_zero", "shoulder_clips_to_one"],
)
def test_clipped_toe_and_shoulder_emit_no_warning(overrides: dict[str, float]) -> None:
    """MR-S11: the log(0) at toe=0 must not surface as a RuntimeWarning."""
    result = _simulate({**DEFAULTS, **overrides})
    curve = _curve(result)

    assert np.all(np.isfinite(curve))
    assert np.all(np.diff(curve) >= -CURVE_TOL)


# --- MR5: metal_ratio only moves gamma, linearly ------------------------------


@given(params=sim_params, other_ratio=param_values("metal_ratio"))
def test_metal_ratio_changes_only_gamma_linearly(
    params: dict[str, float], other_ratio: float
) -> None:
    process = SIM.compute_process_parameters(params)
    expected_gamma = (
        params["metal_ratio"] * PHYSICS.pt_gamma_base
        + (1.0 - params["metal_ratio"]) * PHYSICS.pd_gamma_base
    )

    assert process.gamma == pytest.approx(expected_gamma, abs=GAMMA_TOL)
    assert process.dmin == PHYSICS.paper_dmin_base
    assert _simulate(params).gamma == process.gamma

    other = SIM.compute_process_parameters({**params, "metal_ratio": other_ratio})
    assert (
        other.dmin,
        other.dmax,
        other.shoulder_position,
        other.toe_position,
        other.contrast,
    ) == (
        process.dmin,
        process.dmax,
        process.shoulder_position,
        process.toe_position,
        process.contrast,
    )


# --- MR6: humidity symmetry, shoulder monotone in temperature -----------------


@given(params=sim_params, delta=st.floats(0.0, HUMIDITY_HALF_WINDOW))
def test_humidity_is_symmetric_about_optimum(params: dict[str, float], delta: float) -> None:
    """Documents the current symmetric model; convert to xfail when physics is revised."""
    wetter = {**params, "humidity": PHYSICS.humidity_optimal + delta}
    drier = {**params, "humidity": PHYSICS.humidity_optimal - delta}

    assert SIM.compute_process_parameters(wetter).toe_position == pytest.approx(
        SIM.compute_process_parameters(drier).toe_position, abs=SYMMETRY_TOL
    )
    np.testing.assert_allclose(
        _curve(_simulate(wetter)), _curve(_simulate(drier)), atol=SYMMETRY_TOL, rtol=0.0
    )


@given(
    params=sim_params,
    first=param_values("developer_temp"),
    second=param_values("developer_temp"),
)
def test_shoulder_monotone_in_temperature_and_clipped(
    params: dict[str, float], first: float, second: float
) -> None:
    lo, hi = sorted((first, second))
    cool = SIM.compute_process_parameters({**params, "developer_temp": lo}).shoulder_position
    warm = SIM.compute_process_parameters({**params, "developer_temp": hi}).shoulder_position

    if PHYSICS.shoulder_temp_sensitivity >= 0.0:
        assert warm >= cool - MONOTONE_TOL
    else:
        assert warm <= cool + MONOTONE_TOL
    assert SHOULDER_MIN <= cool <= SHOULDER_MAX
    assert SHOULDER_MIN <= warm <= SHOULDER_MAX


# --- MR-S10: default-fill invariance ------------------------------------------


def test_missing_parameters_are_filled_with_config_defaults() -> None:
    assert _simulate({}).density_curve == _simulate(DEFAULTS).density_curve


# --- SCI-01: known failures, documented as strict xfail -----------------------


@pytest.mark.xfail(strict=True, reason="SCI-01: dead parameter / flat objective")
def test_ferric_oxalate_changes_the_curve() -> None:
    """MR-S5: ferric_oxalate_pct only feeds ``contrast``, which the curve ignores."""
    fo_range = DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"]
    low = _curve(_simulate({**DEFAULTS, "ferric_oxalate_pct": fo_range.min_value}))
    high = _curve(_simulate({**DEFAULTS, "ferric_oxalate_pct": fo_range.max_value}))

    assert np.max(np.abs(high - low)) > FO_DELTA_MIN


@pytest.mark.xfail(strict=True, reason="SCI-01: dead parameter / flat objective")
def test_objective_is_reachable_and_spread_over_parameter_box() -> None:
    """MR-S17: over 2^11 Sobol samples the target dmax must be reachable and the
    quality score must vary by at least 0.3 across the box."""
    names = list(DEFAULT_PARAMETER_RANGES)
    lower = [DEFAULT_PARAMETER_RANGES[n].min_value for n in names]
    upper = [DEFAULT_PARAMETER_RANGES[n].max_value for n in names]
    unit_samples = qmc.Sobol(d=len(names), scramble=True, seed=SOBOL_SEED).random_base2(
        SOBOL_LOG2_SAMPLES
    )
    samples = qmc.scale(unit_samples, lower, upper)

    scores: list[float] = []
    dmaxes: list[float] = []
    for row in samples:
        result = _simulate(dict(zip(names, (float(v) for v in row), strict=True)))
        scores.append(SCORER.score(result))
        dmaxes.append(result.dmax)

    assert max(dmaxes) >= SCORER.settings.target_dmax
    assert max(scores) - min(scores) >= OBJECTIVE_SPREAD_MIN
