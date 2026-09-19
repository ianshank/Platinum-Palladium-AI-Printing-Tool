"""Property-based tests for ``chemistry/calculator.py`` (TST-08)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptpd_calibration.chemistry.calculator import (
    METAL_MIX_RATIOS,
    ChemistryCalculator,
    ChemistryRecipe,
    CoatingMethod,
    MetalMix,
    PaperAbsorbency,
)
from tests.property.strategies import unit

pytestmark = pytest.mark.property

CALC = ChemistryCalculator()
SETTINGS = CALC.settings

DIM_MIN, DIM_MAX = 0.5, 40.0
"""Print dimensions in inches; >= 0.5 so the coating-area floor never exceeds the print."""
MARGIN_MIN, MARGIN_MAX = 0.0, 2.0
SCALE_MIN, SCALE_MAX = 0.1, 10.0
ABS_TOL = 1e-6
REL_TOL = 1e-9
ZERO_MARGIN = 0.0
DOUBLE = 2.0
AREA_FACTOR_FOR_DOUBLING = DOUBLE * DOUBLE
CALCULATOR_MIN_COATING_INCHES = 0.5
"""Floor applied by ``ChemistryCalculator.calculate`` to each coating dimension."""

DROP_FIELDS = (
    "ferric_oxalate_drops",
    "ferric_oxalate_contrast_drops",
    "palladium_drops",
    "platinum_drops",
    "na2_drops",
    "total_drops",
)
ML_FIELDS = (
    "ferric_oxalate_ml",
    "ferric_oxalate_contrast_ml",
    "palladium_ml",
    "platinum_ml",
    "na2_ml",
    "total_ml",
)
METADATA_FIELDS = (
    "print_width_inches",
    "print_height_inches",
    "coating_width_inches",
    "coating_height_inches",
    "coating_area_sq_inches",
    "platinum_ratio",
    "palladium_ratio",
    "paper_absorbency",
    "coating_method",
    "contrast_boost",
)

dims = st.floats(DIM_MIN, DIM_MAX, allow_nan=False, allow_infinity=False)
margins = st.floats(MARGIN_MIN, MARGIN_MAX, allow_nan=False, allow_infinity=False)
scales = st.floats(SCALE_MIN, SCALE_MAX, allow_nan=False, allow_infinity=False)
absorbencies = st.sampled_from(list(PaperAbsorbency))
methods = st.sampled_from(list(CoatingMethod))


def _close(a: float, b: float) -> bool:
    return a == pytest.approx(b, rel=REL_TOL, abs=ABS_TOL)


@given(
    width=dims,
    height=dims,
    platinum_ratio=unit,
    absorbency=absorbencies,
    method=methods,
    contrast_boost=unit,
    na2_ratio=unit,
)
def test_recipe_mass_balance_and_ratios(
    width: float,
    height: float,
    platinum_ratio: float,
    absorbency: PaperAbsorbency,
    method: CoatingMethod,
    contrast_boost: float,
    na2_ratio: float,
) -> None:
    """total == sum of parts, A + B == C (Bostick-Sullivan), drops/ml consistent."""
    recipe = CALC.calculate(
        width,
        height,
        platinum_ratio=platinum_ratio,
        paper_absorbency=absorbency,
        coating_method=method,
        contrast_boost=contrast_boost,
        na2_ratio=na2_ratio,
    )

    parts = (
        recipe.ferric_oxalate_drops
        + recipe.ferric_oxalate_contrast_drops
        + recipe.palladium_drops
        + recipe.platinum_drops
        + recipe.na2_drops
    )
    assert _close(recipe.total_drops, parts)
    assert _close(
        recipe.ferric_oxalate_drops + recipe.ferric_oxalate_contrast_drops,
        recipe.palladium_drops + recipe.platinum_drops,
    )
    assert _close(recipe.total_ml * SETTINGS.drops_per_ml, recipe.total_drops)
    for drop_field, ml_field in zip(DROP_FIELDS, ML_FIELDS, strict=True):
        assert _close(
            getattr(recipe, ml_field) * SETTINGS.drops_per_ml, getattr(recipe, drop_field)
        )
        assert getattr(recipe, drop_field) >= 0.0
        assert getattr(recipe, ml_field) >= 0.0
    metal_total = recipe.palladium_drops + recipe.platinum_drops
    assert metal_total > 0.0
    assert _close(recipe.platinum_drops / metal_total, platinum_ratio)
    assert _close(recipe.palladium_ratio, 1.0 - platinum_ratio)
    assert recipe.estimated_cost_usd is not None and recipe.estimated_cost_usd >= 0.0


@given(width=dims, height=dims, platinum_ratio=unit, first=scales, second=scales)
def test_scale_recipe_is_linear_and_composes(
    width: float, height: float, platinum_ratio: float, first: float, second: float
) -> None:
    recipe = CALC.calculate(width, height, platinum_ratio=platinum_ratio)
    scaled = CALC.scale_recipe(recipe, first)

    for field in DROP_FIELDS + ML_FIELDS:
        assert _close(getattr(scaled, field), first * getattr(recipe, field))
    assert recipe.estimated_cost_usd is not None and scaled.estimated_cost_usd is not None
    assert _close(scaled.estimated_cost_usd, first * recipe.estimated_cost_usd)
    for field in METADATA_FIELDS:
        assert getattr(scaled, field) == getattr(recipe, field)
    assert scaled.notes[: len(recipe.notes)] == recipe.notes

    composed = CALC.scale_recipe(scaled, second)
    direct = CALC.scale_recipe(recipe, first * second)
    for field in DROP_FIELDS + ML_FIELDS:
        assert _close(getattr(composed, field), getattr(direct, field))


@given(factor=st.floats(max_value=0.0))
def test_scale_recipe_rejects_non_positive_factor(factor: float) -> None:
    recipe = CALC.calculate(DIM_MIN, DIM_MIN)

    with pytest.raises(ValueError):
        CALC.scale_recipe(recipe, factor)


@given(first=dims, second=dims, other=dims)
def test_total_drops_monotone_in_each_dimension(first: float, second: float, other: float) -> None:
    lo, hi = sorted((first, second))

    assert CALC.calculate(lo, other).total_drops <= CALC.calculate(hi, other).total_drops + ABS_TOL
    assert CALC.calculate(other, lo).total_drops <= CALC.calculate(other, hi).total_drops + ABS_TOL


@given(width=dims, height=dims)
def test_absorbency_and_method_orderings(width: float, height: float) -> None:
    """HIGH >= MEDIUM >= LOW absorbency and ROD <= BRUSH, per the settings multipliers."""
    by_absorbency = {
        level: CALC.calculate(width, height, paper_absorbency=level).total_drops
        for level in PaperAbsorbency
    }
    assert by_absorbency[PaperAbsorbency.HIGH] >= by_absorbency[PaperAbsorbency.MEDIUM] - ABS_TOL
    assert by_absorbency[PaperAbsorbency.MEDIUM] >= by_absorbency[PaperAbsorbency.LOW] - ABS_TOL

    rod = CALC.calculate(width, height, coating_method=CoatingMethod.ROD).total_drops
    brush = CALC.calculate(width, height, coating_method=CoatingMethod.BRUSH).total_drops
    assert rod <= brush + ABS_TOL


@given(width=dims, height=dims, margin=margins)
def test_coating_area_never_exceeds_print_area(width: float, height: float, margin: float) -> None:
    recipe = CALC.calculate(width, height, margin_inches=margin)

    assert recipe.coating_width_inches >= CALCULATOR_MIN_COATING_INCHES
    assert recipe.coating_height_inches >= CALCULATOR_MIN_COATING_INCHES
    assert recipe.coating_area_sq_inches <= width * height + ABS_TOL
    assert _close(
        recipe.coating_area_sq_inches, recipe.coating_width_inches * recipe.coating_height_inches
    )


@given(width=dims, height=dims, first=unit, second=unit)
def test_cost_monotone_in_platinum_ratio(
    width: float, height: float, first: float, second: float
) -> None:
    """Platinum costs at least as much as palladium, so more Pt never lowers the cost."""
    assert SETTINGS.platinum_cost_per_ml >= SETTINGS.palladium_cost_per_ml
    lo, hi = sorted((first, second))

    cheap = CALC.calculate(width, height, platinum_ratio=lo).estimated_cost_usd
    dear = CALC.calculate(width, height, platinum_ratio=hi).estimated_cost_usd

    assert cheap is not None and dear is not None
    assert cheap <= dear + ABS_TOL


@given(width=dims, height=dims)
def test_drops_scale_with_coating_area(width: float, height: float) -> None:
    base = CALC.calculate(width, height, margin_inches=ZERO_MARGIN)
    doubled = CALC.calculate(DOUBLE * width, DOUBLE * height, margin_inches=ZERO_MARGIN)

    assert _close(doubled.total_drops, AREA_FACTOR_FOR_DOUBLING * base.total_drops)


@given(width=dims, height=dims, mix=st.sampled_from(list(MetalMix)))
def test_preset_matches_explicit_ratio(width: float, height: float, mix: MetalMix) -> None:
    preset: ChemistryRecipe = CALC.calculate_from_preset(width, height, metal_mix=mix)
    explicit = CALC.calculate(width, height, platinum_ratio=METAL_MIX_RATIOS[mix])

    for field in DROP_FIELDS + ML_FIELDS:
        assert getattr(preset, field) == getattr(explicit, field)
