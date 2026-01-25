"""
Test utility functions and helpers for PTPD Calibration tests.

Provides assertion helpers, data generators, and utilities for
common testing patterns.

Usage:
    from tests.utils import (
        assert_curves_equal,
        assert_monotonic,
        generate_synthetic_densities,
        with_performance,
    )

    def test_curve_generation():
        densities = generate_synthetic_densities(num_steps=21)
        curve = generator.generate(densities)
        assert_monotonic(curve.output_values)
        assert_curves_equal(curve.output_values, expected_values, tolerance=0.01)
"""

import numpy as np
from typing import Callable, TypeVar, Any, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import time
from pathlib import Path
from PIL import Image

T = TypeVar("T")


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_curves_equal(
    curve1: Sequence[float],
    curve2: Sequence[float],
    tolerance: float = 0.001,
    msg: str | None = None,
) -> None:
    """Assert two curves are equal within tolerance.

    Args:
        curve1: First curve values.
        curve2: Second curve values.
        tolerance: Maximum allowed difference at any point.
        msg: Optional message prefix for assertion errors.

    Raises:
        AssertionError: If curves differ beyond tolerance.
    """
    prefix = f"{msg}: " if msg else ""

    assert len(curve1) == len(curve2), (
        f"{prefix}Curve lengths differ: {len(curve1)} vs {len(curve2)}"
    )

    for i, (v1, v2) in enumerate(zip(curve1, curve2)):
        diff = abs(v1 - v2)
        assert diff < tolerance, (
            f"{prefix}Curves differ at index {i}: {v1:.6f} vs {v2:.6f} "
            f"(diff={diff:.6f}, tolerance={tolerance})"
        )


def assert_monotonic(
    values: Sequence[float],
    increasing: bool = True,
    strict: bool = False,
    allow_tolerance: float = 0.0,
) -> None:
    """Assert values are monotonically increasing or decreasing.

    Args:
        values: Sequence of values to check.
        increasing: If True, check for increasing; if False, decreasing.
        strict: If True, values must be strictly increasing/decreasing.
        allow_tolerance: Allow small violations (for numerical precision).

    Raises:
        AssertionError: If monotonicity is violated.
    """
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        if increasing:
            if strict:
                valid = diff > -allow_tolerance
            else:
                valid = diff >= -allow_tolerance
            direction = "increasing"
        else:
            if strict:
                valid = diff < allow_tolerance
            else:
                valid = diff <= allow_tolerance
            direction = "decreasing"

        assert valid, (
            f"Not monotonically {direction} at index {i}: "
            f"{values[i-1]:.6f} -> {values[i]:.6f} (diff={diff:.6f})"
        )


def assert_in_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> None:
    """Assert value is within specified range.

    Args:
        value: Value to check.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Name for error messages.

    Raises:
        AssertionError: If value is out of range.
    """
    assert min_val <= value <= max_val, (
        f"{name} {value:.6f} not in range [{min_val}, {max_val}]"
    )


def assert_all_in_range(
    values: Sequence[float],
    min_val: float,
    max_val: float,
    name: str = "values",
) -> None:
    """Assert all values are within specified range.

    Args:
        values: Sequence of values to check.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        name: Name for error messages.

    Raises:
        AssertionError: If any value is out of range.
    """
    for i, v in enumerate(values):
        assert min_val <= v <= max_val, (
            f"{name}[{i}] = {v:.6f} not in range [{min_val}, {max_val}]"
        )


def assert_approximately_equal(
    actual: float,
    expected: float,
    rel_tolerance: float = 0.01,
    abs_tolerance: float = 1e-9,
    name: str = "value",
) -> None:
    """Assert two values are approximately equal.

    Uses both relative and absolute tolerance for robust comparison.

    Args:
        actual: Actual value.
        expected: Expected value.
        rel_tolerance: Relative tolerance (fraction of expected).
        abs_tolerance: Absolute tolerance.
        name: Name for error messages.

    Raises:
        AssertionError: If values differ more than tolerance.
    """
    diff = abs(actual - expected)
    threshold = max(rel_tolerance * abs(expected), abs_tolerance)

    assert diff <= threshold, (
        f"{name}: {actual:.6f} != {expected:.6f} "
        f"(diff={diff:.6f}, threshold={threshold:.6f})"
    )


# =============================================================================
# Performance Testing
# =============================================================================


@dataclass
class PerformanceResult:
    """Result from performance measurement."""
    duration_seconds: float
    iterations: int
    avg_per_iteration: float
    min_duration: float
    max_duration: float


@contextmanager
def assert_performance(
    max_seconds: float,
    operation_name: str = "operation",
):
    """Assert operation completes within time limit.

    Args:
        max_seconds: Maximum allowed duration in seconds.
        operation_name: Name for error messages.

    Yields:
        None

    Raises:
        AssertionError: If operation exceeds time limit.

    Example:
        with assert_performance(1.0, "curve generation"):
            generate_curve(densities)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    assert elapsed < max_seconds, (
        f"{operation_name} took {elapsed:.4f}s, "
        f"expected < {max_seconds}s"
    )


def measure_performance(
    func: Callable[[], T],
    iterations: int = 100,
    warmup: int = 10,
) -> PerformanceResult:
    """Measure function performance over multiple iterations.

    Args:
        func: Function to measure (no arguments).
        iterations: Number of measurement iterations.
        warmup: Number of warmup iterations (not measured).

    Returns:
        PerformanceResult with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Measure
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)

    total = sum(durations)
    return PerformanceResult(
        duration_seconds=total,
        iterations=iterations,
        avg_per_iteration=total / iterations,
        min_duration=min(durations),
        max_duration=max(durations),
    )


# =============================================================================
# Data Generators
# =============================================================================


def generate_synthetic_densities(
    num_steps: int = 21,
    dmin: float = 0.1,
    dmax: float = 2.2,
    gamma: float = 0.85,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> list[float]:
    """Generate synthetic density measurements for testing.

    Creates density values following typical photographic response.

    Args:
        num_steps: Number of steps (patches).
        dmin: Minimum density (paper base).
        dmax: Maximum density.
        gamma: Response curve gamma (< 1 for typical photo response).
        noise_std: Standard deviation of noise to add.
        seed: Random seed for reproducibility.

    Returns:
        List of density values.
    """
    if seed is not None:
        np.random.seed(seed)

    steps = np.linspace(0, 1, num_steps)
    densities = dmin + (dmax - dmin) * (steps ** gamma)

    if noise_std > 0:
        densities += np.random.normal(0, noise_std, num_steps)
        # Ensure physical validity
        densities = np.clip(densities, 0.0, 4.0)

    return list(densities)


def generate_synthetic_step_tablet(
    width: int = 420,
    height: int = 100,
    num_patches: int = 21,
    dmin_value: int = 250,
    dmax_value: int = 20,
    with_border: bool = True,
    border_size: int = 20,
    border_color: int = 250,
) -> np.ndarray:
    """Generate synthetic step tablet image for testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_patches: Number of step patches.
        dmin_value: Pixel value for lightest patch (paper base).
        dmax_value: Pixel value for darkest patch.
        with_border: Add white border around tablet.
        border_size: Border size in pixels.
        border_color: Border pixel value.

    Returns:
        Numpy array of the image (grayscale).
    """
    patch_width = width // num_patches
    img = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_patches):
        # Linear interpolation between dmin and dmax values
        t = i / (num_patches - 1) if num_patches > 1 else 0
        value = int(dmin_value + t * (dmax_value - dmin_value))

        x_start = i * patch_width
        x_end = (i + 1) * patch_width if i < num_patches - 1 else width
        img[:, x_start:x_end] = value

    if with_border:
        # Add border
        full_height = height + 2 * border_size
        full_width = width + 2 * border_size
        full_img = np.full((full_height, full_width), border_color, dtype=np.uint8)
        full_img[border_size:border_size + height, border_size:border_size + width] = img
        return full_img

    return img


def generate_rgb_step_tablet(
    width: int = 420,
    height: int = 100,
    num_patches: int = 21,
    dmin_rgb: tuple[int, int, int] = (250, 248, 245),
    dmax_rgb: tuple[int, int, int] = (20, 18, 15),
) -> np.ndarray:
    """Generate RGB step tablet image for testing.

    Creates a more realistic colored step tablet with slight color variations.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_patches: Number of step patches.
        dmin_rgb: RGB values for lightest patch.
        dmax_rgb: RGB values for darkest patch.

    Returns:
        Numpy array of the image (RGB, shape: height x width x 3).
    """
    patch_width = width // num_patches
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_patches):
        t = i / (num_patches - 1) if num_patches > 1 else 0

        # Interpolate each channel
        r = int(dmin_rgb[0] + t * (dmax_rgb[0] - dmin_rgb[0]))
        g = int(dmin_rgb[1] + t * (dmax_rgb[1] - dmin_rgb[1]))
        b = int(dmin_rgb[2] + t * (dmax_rgb[2] - dmin_rgb[2]))

        x_start = i * patch_width
        x_end = (i + 1) * patch_width if i < num_patches - 1 else width
        img[:, x_start:x_end] = [r, g, b]

    return img


def generate_curve_data(
    num_points: int = 256,
    curve_type: str = "linear",
    gamma: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Generate curve data for testing.

    Args:
        num_points: Number of points in the curve.
        curve_type: Type of curve (linear, gamma, s-curve).
        gamma: Gamma value for gamma curves.

    Returns:
        Tuple of (input_values, output_values).
    """
    x = np.linspace(0, 1, num_points)

    if curve_type == "linear":
        y = x
    elif curve_type == "gamma":
        y = x ** gamma
    elif curve_type == "s-curve":
        # Sigmoid-based S-curve
        y = 1 / (1 + np.exp(-10 * (x - 0.5)))
        y = (y - y.min()) / (y.max() - y.min())  # Normalize to 0-1
    elif curve_type == "inverse":
        y = 1 - x
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

    return list(x), list(y)


# =============================================================================
# File Utilities
# =============================================================================


def save_test_image(
    array: np.ndarray,
    path: Path,
    mode: str | None = None,
) -> Path:
    """Save numpy array as image file.

    Args:
        array: Image data as numpy array.
        path: Path to save to.
        mode: PIL image mode (auto-detected if None).

    Returns:
        Path to saved file.
    """
    if mode is None:
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] == 3:
            mode = "RGB"
        elif array.ndim == 3 and array.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Cannot auto-detect mode for shape {array.shape}")

    img = Image.fromarray(array, mode=mode)
    img.save(path)
    return path


def load_test_fixture(
    fixture_name: str,
    fixtures_dir: Path | None = None,
) -> Path:
    """Load a test fixture file path.

    Args:
        fixture_name: Name of the fixture file.
        fixtures_dir: Directory containing fixtures (defaults to tests/fixtures).

    Returns:
        Path to the fixture file.

    Raises:
        FileNotFoundError: If fixture doesn't exist.
    """
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent / "fixtures"

    path = fixtures_dir / fixture_name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return path


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_images(
    img1: np.ndarray,
    img2: np.ndarray,
    tolerance: int = 5,
) -> tuple[bool, float]:
    """Compare two images for equality.

    Args:
        img1: First image array.
        img2: Second image array.
        tolerance: Maximum per-pixel difference allowed.

    Returns:
        Tuple of (images_match, difference_percentage).
    """
    if img1.shape != img2.shape:
        return False, 100.0

    diff = np.abs(img1.astype(int) - img2.astype(int))
    max_diff = diff.max()
    pixels_different = np.sum(diff > tolerance)
    total_pixels = img1.size
    diff_percentage = (pixels_different / total_pixels) * 100

    return max_diff <= tolerance, diff_percentage


def rmse(values1: Sequence[float], values2: Sequence[float]) -> float:
    """Calculate Root Mean Square Error between two sequences.

    Args:
        values1: First sequence.
        values2: Second sequence.

    Returns:
        RMSE value.
    """
    if len(values1) != len(values2):
        raise ValueError("Sequences must have equal length")

    arr1 = np.array(values1)
    arr2 = np.array(values2)
    return float(np.sqrt(np.mean((arr1 - arr2) ** 2)))


def mae(values1: Sequence[float], values2: Sequence[float]) -> float:
    """Calculate Mean Absolute Error between two sequences.

    Args:
        values1: First sequence.
        values2: Second sequence.

    Returns:
        MAE value.
    """
    if len(values1) != len(values2):
        raise ValueError("Sequences must have equal length")

    arr1 = np.array(values1)
    arr2 = np.array(values2)
    return float(np.mean(np.abs(arr1 - arr2)))
