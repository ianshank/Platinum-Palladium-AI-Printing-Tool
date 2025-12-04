"""
Visual regression test fixtures and configuration.

Provides utilities for screenshot comparison and baseline management.
"""

import os
from pathlib import Path
from typing import Generator

import pytest

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from pixelmatch import pixelmatch
    from pixelmatch.contrib.PIL import pixelmatch as pil_pixelmatch

    PIXELMATCH_AVAILABLE = True
except ImportError:
    PIXELMATCH_AVAILABLE = False
    pixelmatch = None


# Directories
VISUAL_TESTS_DIR = Path(__file__).parent
BASELINE_DIR = VISUAL_TESTS_DIR / "baselines"
ACTUAL_DIR = VISUAL_TESTS_DIR / "actual"
DIFF_DIR = VISUAL_TESTS_DIR / "diffs"


def pytest_configure(config):
    """Add visual test markers."""
    config.addinivalue_line("markers", "visual: mark test as a visual regression test")


def pytest_collection_modifyitems(config, items):
    """Skip visual tests if dependencies are not available."""
    if not PIL_AVAILABLE or not PIXELMATCH_AVAILABLE:
        skip_visual = pytest.mark.skip(
            reason="Visual testing dependencies not installed"
        )
        for item in items:
            if "visual" in item.keywords:
                item.add_marker(skip_visual)


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    """Return and ensure baseline directory exists."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINE_DIR


@pytest.fixture(scope="session")
def actual_dir() -> Path:
    """Return and ensure actual screenshots directory exists."""
    ACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    return ACTUAL_DIR


@pytest.fixture(scope="session")
def diff_dir() -> Path:
    """Return and ensure diff directory exists."""
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    return DIFF_DIR


@pytest.fixture
def visual_comparator(baseline_dir, actual_dir, diff_dir):
    """Create a visual comparison utility."""
    return VisualComparator(baseline_dir, actual_dir, diff_dir)


class VisualComparator:
    """Utility for comparing screenshots against baselines."""

    def __init__(
        self,
        baseline_dir: Path,
        actual_dir: Path,
        diff_dir: Path,
        threshold: float = 0.1,
    ):
        """
        Initialize the visual comparator.

        Args:
            baseline_dir: Directory containing baseline images
            actual_dir: Directory to save actual screenshots
            diff_dir: Directory to save diff images
            threshold: Pixel difference threshold (0-1)
        """
        self.baseline_dir = baseline_dir
        self.actual_dir = actual_dir
        self.diff_dir = diff_dir
        self.threshold = threshold
        self.update_baselines = os.environ.get("UPDATE_BASELINES", "false").lower() == "true"

    def compare(
        self,
        name: str,
        actual_image: Image.Image,
        tolerance: float | None = None,
    ) -> dict:
        """
        Compare an actual image against its baseline.

        Args:
            name: Name of the comparison (used for file names)
            actual_image: The actual screenshot to compare
            tolerance: Override the default threshold

        Returns:
            Dictionary with comparison results
        """
        if not PIL_AVAILABLE:
            return {"error": "PIL not available"}

        tolerance = tolerance or self.threshold
        baseline_path = self.baseline_dir / f"{name}.png"
        actual_path = self.actual_dir / f"{name}.png"
        diff_path = self.diff_dir / f"{name}_diff.png"

        # Save actual image
        actual_image.save(actual_path)

        # If no baseline exists
        if not baseline_path.exists():
            if self.update_baselines:
                actual_image.save(baseline_path)
                return {
                    "status": "created",
                    "message": f"Created new baseline: {baseline_path}",
                    "baseline_path": str(baseline_path),
                }
            return {
                "status": "missing",
                "message": f"Baseline not found: {baseline_path}",
                "actual_path": str(actual_path),
            }

        # Load baseline
        baseline_image = Image.open(baseline_path)

        # Check dimensions match
        if actual_image.size != baseline_image.size:
            if self.update_baselines:
                actual_image.save(baseline_path)
                return {
                    "status": "updated",
                    "message": "Baseline updated (size changed)",
                }
            return {
                "status": "size_mismatch",
                "message": f"Size mismatch: baseline {baseline_image.size} vs actual {actual_image.size}",
                "baseline_size": baseline_image.size,
                "actual_size": actual_image.size,
            }

        # Compare images
        if PIXELMATCH_AVAILABLE:
            result = self._compare_with_pixelmatch(
                baseline_image, actual_image, diff_path, tolerance
            )
        else:
            result = self._compare_simple(baseline_image, actual_image)

        # Update baseline if requested and different
        if self.update_baselines and result["status"] != "match":
            actual_image.save(baseline_path)
            result["status"] = "updated"
            result["message"] = "Baseline updated"

        return result

    def _compare_with_pixelmatch(
        self,
        baseline: Image.Image,
        actual: Image.Image,
        diff_path: Path,
        tolerance: float,
    ) -> dict:
        """Compare images using pixelmatch."""
        width, height = baseline.size

        # Create diff image
        diff = Image.new("RGBA", (width, height))

        # Convert to RGBA
        baseline_rgba = baseline.convert("RGBA")
        actual_rgba = actual.convert("RGBA")

        # Compare
        num_diff_pixels = pil_pixelmatch(
            baseline_rgba,
            actual_rgba,
            diff,
            threshold=tolerance,
            includeAA=True,
        )

        total_pixels = width * height
        diff_percentage = (num_diff_pixels / total_pixels) * 100

        # Save diff if there are differences
        if num_diff_pixels > 0:
            diff.save(diff_path)

        # Consider match if less than 0.1% different
        is_match = diff_percentage < 0.1

        return {
            "status": "match" if is_match else "mismatch",
            "diff_pixels": num_diff_pixels,
            "total_pixels": total_pixels,
            "diff_percentage": diff_percentage,
            "diff_path": str(diff_path) if num_diff_pixels > 0 else None,
        }

    def _compare_simple(
        self,
        baseline: Image.Image,
        actual: Image.Image,
    ) -> dict:
        """Simple image comparison without pixelmatch."""
        import numpy as np

        baseline_array = np.array(baseline.convert("RGB"))
        actual_array = np.array(actual.convert("RGB"))

        diff = np.abs(baseline_array.astype(float) - actual_array.astype(float))
        total_diff = np.sum(diff)
        max_possible = baseline_array.size * 255

        diff_percentage = (total_diff / max_possible) * 100

        return {
            "status": "match" if diff_percentage < 0.1 else "mismatch",
            "diff_percentage": diff_percentage,
        }

    def assert_match(
        self,
        name: str,
        actual_image: Image.Image,
        tolerance: float | None = None,
    ) -> None:
        """
        Assert that an image matches its baseline.

        Raises AssertionError if the images don't match.
        """
        result = self.compare(name, actual_image, tolerance)

        if result["status"] == "missing":
            pytest.skip(f"Baseline not found: {name}")
        elif result["status"] == "mismatch":
            msg = f"Visual regression detected for '{name}': {result.get('diff_percentage', 0):.2f}% different"
            if result.get("diff_path"):
                msg += f"\nDiff saved to: {result['diff_path']}"
            pytest.fail(msg)
        elif result["status"] == "size_mismatch":
            pytest.fail(result["message"])


@pytest.fixture
def capture_component_screenshot(driver):
    """Factory fixture to capture screenshots of specific components."""

    def _capture(selector: str, name: str) -> Image.Image:
        """Capture a screenshot of a specific element."""
        from selenium.webdriver.common.by import By

        element = driver.find_element(By.CSS_SELECTOR, selector)
        screenshot_bytes = element.screenshot_as_png

        import io

        return Image.open(io.BytesIO(screenshot_bytes))

    return _capture


@pytest.fixture
def capture_full_screenshot(driver):
    """Factory fixture to capture full page screenshots."""

    def _capture(name: str) -> Image.Image:
        """Capture a full page screenshot."""
        screenshot_bytes = driver.get_screenshot_as_png()

        import io

        return Image.open(io.BytesIO(screenshot_bytes))

    return _capture
