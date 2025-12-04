"""
Custom assertion helpers for PTPD Calibration tests.

Provides domain-specific assertions for common test scenarios.
"""

import math
from typing import Any, Sequence

import numpy as np


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
        assert d <= 5.0, f"Unrealistic density at index {i}: {d}"


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
                    f"Non-strictly increasing at index {i}: {densities[i-1]} >= {densities[i]}"
                )
            else:
                assert densities[i] >= densities[i - 1], (
                    f"Non-monotonic at index {i}: {densities[i-1]} > {densities[i]}"
                )
        else:
            if strict:
                assert densities[i] < densities[i - 1], (
                    f"Non-strictly decreasing at index {i}: {densities[i-1]} <= {densities[i]}"
                )
            else:
                assert densities[i] <= densities[i - 1], (
                    f"Non-monotonic at index {i}: {densities[i-1]} < {densities[i]}"
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
                f"Non-monotonic at index {i}: {output_values[i-1]} > {output_values[i]}"
            )
        else:
            assert output_values[i] <= output_values[i - 1] + tolerance, (
                f"Non-monotonic at index {i}: {output_values[i-1]} < {output_values[i]}"
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
