"""
Custom assertion helpers for PTPD Calibration tests.

Provides domain-specific assertions for common test scenarios.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

# Maximum realistic density value for photographic processes
# Typical Pt/Pd prints have Dmax around 1.8-2.2, allowing headroom for unusual cases
MAX_REALISTIC_DENSITY = 5.0


def assert_densities_valid(densities: Sequence[float]) -> None:
    """
    Assert that density measurements are valid.

    Valid densities should:
    - Have at least 3 values
    - All be non-negative
    - Have a reasonable range (0 to ~3.0 for typical measurements)
    """
    assert len(densities) >= 3, f"Too few densities: {len(densities)}"

    for i, d in enumerate(densities):
        assert d >= 0, f"Negative density at index {i}: {d}"
        assert d <= MAX_REALISTIC_DENSITY, f"Unrealistic density at index {i}: {d}"


def assert_densities_monotonic(
    densities: Sequence[float],
    increasing: bool = True,
    strict: bool = False,
) -> None:
    """
    Assert that density measurements are monotonic.

    Args:
        densities: The density values to check
        increasing: If True, check for monotonically increasing; else decreasing
        strict: If True, values must be strictly monotonic (no equal values)
    """
    for i in range(1, len(densities)):
        if increasing:
            if strict:
                assert densities[i] > densities[i - 1], (
                    f"Non-strictly increasing at index {i}: {densities[i - 1]} >= {densities[i]}"
                )
            else:
                assert densities[i] >= densities[i - 1], (
                    f"Non-monotonic at index {i}: {densities[i - 1]} > {densities[i]}"
                )
        else:
            if strict:
                assert densities[i] < densities[i - 1], (
                    f"Non-strictly decreasing at index {i}: {densities[i - 1]} <= {densities[i]}"
                )
            else:
                assert densities[i] <= densities[i - 1], (
                    f"Non-monotonic at index {i}: {densities[i - 1]} < {densities[i]}"
                )


def assert_curve_valid(input_values: Sequence[float], output_values: Sequence[float]) -> None:
    """
    Assert that curve data is valid.

    Valid curves should:
    - Have matching input/output lengths
    - Have input values in [0, 1] range
    - Have output values in [0, 1] range
    - Have sufficient points (at least 2)
    """
    assert len(input_values) == len(output_values), (
        f"Mismatched lengths: {len(input_values)} inputs vs {len(output_values)} outputs"
    )

    assert len(input_values) >= 2, f"Too few points: {len(input_values)}"

    for i, v in enumerate(input_values):
        assert 0 <= v <= 1, f"Input value out of range at index {i}: {v}"

    for i, v in enumerate(output_values):
        assert 0 <= v <= 1, f"Output value out of range at index {i}: {v}"


def assert_curve_monotonic(
    output_values: Sequence[float],
    increasing: bool = True,
    tolerance: float = 0.001,
) -> None:
    """
    Assert that curve output values are monotonic within tolerance.

    Args:
        output_values: The output values to check
        increasing: If True, check for monotonically increasing
        tolerance: Allow small violations within this tolerance
    """
    for i in range(1, len(output_values)):
        if increasing:
            assert output_values[i] >= output_values[i - 1] - tolerance, (
                f"Non-monotonic at index {i}: {output_values[i - 1]} > {output_values[i]}"
            )
        else:
            assert output_values[i] <= output_values[i - 1] + tolerance, (
                f"Non-monotonic at index {i}: {output_values[i - 1]} < {output_values[i]}"
            )


def assert_approximately_equal(
    actual: float,
    expected: float,
    tolerance: float = 0.01,
) -> None:
    """
    Assert that two values are approximately equal.

    Args:
        actual: The actual value
        expected: The expected value
        tolerance: Maximum allowed difference
    """
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Values not approximately equal: {actual} vs {expected} (diff={diff}, tolerance={tolerance})"
    )


def assert_arrays_close(
    actual: np.ndarray | Sequence,
    expected: np.ndarray | Sequence,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert that two arrays are element-wise close.

    Uses numpy's allclose function.
    """
    actual_arr = np.array(actual)
    expected_arr = np.array(expected)

    assert np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol), (
        f"Arrays not close. Max difference: {np.max(np.abs(actual_arr - expected_arr))}"
    )


def assert_image_valid(image: Any) -> None:
    """
    Assert that an image object is valid.

    Works with PIL Images and numpy arrays.
    """
    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            assert image.size[0] > 0, "Image width must be positive"
            assert image.size[1] > 0, "Image height must be positive"
            return
    except ImportError:
        pass

    if isinstance(image, np.ndarray):
        assert image.ndim in [2, 3], f"Invalid image dimensions: {image.ndim}"
        assert image.shape[0] > 0, "Image height must be positive"
        assert image.shape[1] > 0, "Image width must be positive"
        return

    raise AssertionError(f"Unknown image type: {type(image)}")


def assert_calibration_record_valid(record: Any) -> None:
    """Assert that a CalibrationRecord is valid."""
    assert record.paper_type, "Paper type must not be empty"
    assert record.exposure_time > 0, "Exposure time must be positive"
    assert 0 <= record.metal_ratio <= 1, "Metal ratio must be in [0, 1]"

    if record.measured_densities:
        assert_densities_valid(record.measured_densities)


def assert_recipe_valid(recipe: dict) -> None:
    """Assert that a chemistry recipe is valid."""
    required_keys = ["platinum", "palladium", "ferric_oxalate"]

    for key in required_keys:
        if key in recipe:
            value = recipe[key]
            assert isinstance(value, (int, float)), f"{key} must be numeric"
            assert value >= 0, f"{key} must be non-negative"


def assert_response_success(response: dict) -> None:
    """Assert that an API response indicates success."""
    assert "success" in response or "error" not in response, (
        f"Response indicates failure: {response}"
    )

    if "success" in response:
        assert response["success"] is True, f"Success is False: {response}"


def assert_response_has_fields(response: dict, fields: list[str]) -> None:
    """Assert that a response contains required fields."""
    for field in fields:
        assert field in response, f"Missing field '{field}' in response: {list(response.keys())}"


# --- Cyanotype Assertions ---


def assert_cyanotype_recipe_valid(recipe: dict | object) -> None:
    """Assert that a cyanotype recipe is valid."""
    if hasattr(recipe, "__dict__"):
        # It's an object, convert to dict-like access
        assert recipe.solution_a_ml > 0, "Solution A must be positive"
        assert recipe.solution_b_ml > 0, "Solution B must be positive"
        assert recipe.total_sensitizer_ml > 0, "Total sensitizer must be positive"
        assert recipe.coating_area_sq_inches > 0, "Coverage area must be positive"
    else:
        # It's a dict
        assert recipe.get("solution_a_ml", 0) > 0, "Solution A must be positive"
        assert recipe.get("solution_b_ml", 0) > 0, "Solution B must be positive"


def assert_cyanotype_exposure_valid(exposure: dict | object) -> None:
    """Assert that a cyanotype exposure result is valid."""
    if hasattr(exposure, "__dict__"):
        assert exposure.exposure_seconds > 0, "Exposure time must be positive"
        assert exposure.exposure_minutes >= 0, "Exposure minutes must be non-negative"
    else:
        assert exposure.get("exposure_seconds", 0) > 0, "Exposure time must be positive"


def assert_cyanotype_solution_ratio_valid(
    solution_a: float,
    solution_b: float,
    formula: str = "classic",
) -> None:
    """Assert that cyanotype solution ratio is valid for the formula."""
    ratio = solution_a / solution_b if solution_b > 0 else 0

    if formula.lower() == "classic":
        # Classic formula typically uses 1:1 ratio
        assert 0.8 <= ratio <= 1.2, f"Classic formula ratio should be ~1:1, got {ratio}"
    elif formula.lower() == "new":
        # New formula may have different ratios
        assert ratio > 0, f"Solution ratio must be positive, got {ratio}"


def assert_uv_exposure_inverse_square(
    exposure1: float,
    distance1: float,
    exposure2: float,
    distance2: float,
    tolerance: float = 0.15,
) -> None:
    """Assert that UV exposure follows inverse square law."""
    expected_ratio = (distance2 / distance1) ** 2
    actual_ratio = exposure2 / exposure1 if exposure1 > 0 else 0

    assert abs(actual_ratio - expected_ratio) / expected_ratio <= tolerance, (
        f"Exposure ratio {actual_ratio} doesn't follow inverse square law "
        f"(expected {expected_ratio})"
    )


# --- Silver Gelatin Assertions ---


def assert_silver_gelatin_chemistry_valid(chemistry: dict | object) -> None:
    """Assert that silver gelatin processing chemistry is valid."""
    if hasattr(chemistry, "__dict__"):
        assert chemistry.developer.total_ml > 0, "Developer volume must be positive"
        assert chemistry.stop_bath_ml > 0, "Stop bath volume must be positive"
        assert chemistry.fixer_ml > 0, "Fixer volume must be positive"
    else:
        assert chemistry.get("developer", {}).get("total_ml", 0) > 0, (
            "Developer volume must be positive"
        )
        assert chemistry.get("stop_bath_ml", 0) > 0, "Stop bath volume must be positive"
        assert chemistry.get("fixer_ml", 0) > 0, "Fixer volume must be positive"


def assert_silver_gelatin_exposure_valid(exposure: dict | object) -> None:
    """Assert that silver gelatin exposure result is valid."""
    if hasattr(exposure, "__dict__"):
        assert exposure.exposure_seconds > 0, "Exposure time must be positive"
        assert exposure.f_stop > 0, "F-stop must be positive"
    else:
        assert exposure.get("exposure_seconds", 0) > 0, "Exposure time must be positive"


def assert_split_filter_valid(split_result: dict) -> None:
    """Assert that split filter calculation is valid."""
    assert "shadow_exposure" in split_result, "Missing shadow exposure"
    assert "highlight_exposure" in split_result, "Missing highlight exposure"

    shadow = split_result["shadow_exposure"]
    highlight = split_result["highlight_exposure"]

    assert shadow > 0, "Shadow exposure must be positive"
    assert highlight > 0, "Highlight exposure must be positive"


def assert_test_strip_times_valid(times: list[float]) -> None:
    """Assert that test strip times are valid."""
    assert len(times) >= 2, f"Need at least 2 test strip times, got {len(times)}"

    for i, time in enumerate(times):
        assert time > 0, f"Test strip time at index {i} must be positive"

    # Times should be increasing
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], (
            f"Test strip times must be increasing: {times[i - 1]} >= {times[i]}"
        )


def assert_fstop_exposure_relationship(
    exposure1: float,
    fstop1: float,
    exposure2: float,
    fstop2: float,
    tolerance: float = 0.15,
) -> None:
    """Assert that exposure changes correctly with f-stop changes."""
    # Each full stop doubles/halves light
    # exposure = k * (f-stop)^2
    expected_ratio = (fstop2 / fstop1) ** 2
    actual_ratio = exposure2 / exposure1 if exposure1 > 0 else 0

    assert abs(actual_ratio - expected_ratio) / expected_ratio <= tolerance, (
        f"Exposure ratio {actual_ratio} doesn't match f-stop relationship "
        f"(expected {expected_ratio})"
    )


def assert_processing_times_valid(
    development_time: float,
    stop_time: float,
    fix_time: float,
    paper_base: str = "fiber",
) -> None:
    """Assert that processing times are valid for the paper type."""
    assert development_time > 0, "Development time must be positive"
    assert stop_time > 0, "Stop time must be positive"
    assert fix_time > 0, "Fix time must be positive"

    # Typical ranges
    if paper_base.lower() == "fiber":
        assert 60 <= development_time <= 300, (
            f"Fiber development time out of range: {development_time}"
        )
        assert 180 <= fix_time <= 600, f"Fiber fix time out of range: {fix_time}"
    else:  # RC
        assert 60 <= development_time <= 180, (
            f"RC development time out of range: {development_time}"
        )
        assert 60 <= fix_time <= 180, f"RC fix time out of range: {fix_time}"


# --- Alternative Process Generic Assertions ---


def assert_alternative_process_result_valid(
    result: dict | object,
    process_type: str,
) -> None:
    """Assert that an alternative process result is valid based on type."""
    if process_type.lower() == "cyanotype":
        if hasattr(result, "solution_a_ml"):
            assert_cyanotype_recipe_valid(result)
        elif hasattr(result, "exposure_seconds"):
            assert_cyanotype_exposure_valid(result)
    elif process_type.lower() in ["silver_gelatin", "darkroom"]:
        if hasattr(result, "developer"):
            assert_silver_gelatin_chemistry_valid(result)
        elif hasattr(result, "exposure_seconds"):
            assert_silver_gelatin_exposure_valid(result)
    else:
        # Generic validation
        assert result is not None, f"Result for {process_type} should not be None"
