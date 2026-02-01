"""
Shared fixtures for PTPD Calibration tests.

This module provides common fixtures used across all test suites.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.models import CalibrationRecord, CurveData
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "selenium: Selenium browser tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "visual: Visual regression tests")
    config.addinivalue_line("markers", "slow: Long-running tests")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Automatically add markers based on test location."""
    for item in items:
        # Add markers based on test path
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "/api/" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "/performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "/visual/" in str(item.fspath):
            item.add_marker(pytest.mark.visual)


@pytest.fixture
def real_quad_path():
    """Return path to the real-world .quad fixture file."""
    # Assumes fixture is in tests/fixtures/Platinum_Palladium_V6-CC.quad
    base_dir = Path(__file__).parent
    fixture_path = base_dir / "fixtures" / "Platinum_Palladium_V6-CC.quad"

    if not fixture_path.exists():
        pytest.skip(f"Real world fixture not found at {fixture_path}")

    return fixture_path


@pytest.fixture
def sample_step_tablet_image(tmp_path):
    """Create a synthetic step tablet image for testing."""
    # Create a simple grayscale step tablet
    width, height = 420, 100
    num_patches = 21
    patch_width = width // num_patches

    img = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_patches):
        # Create gradual darkening patches
        value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
        x_start = i * patch_width
        x_end = (i + 1) * patch_width
        img[:, x_start:x_end] = value

    # Add some margin
    full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
    full_img[20 : height + 20, 20 : width + 20, 0] = img
    full_img[20 : height + 20, 20 : width + 20, 1] = img
    full_img[20 : height + 20, 20 : width + 20, 2] = img

    # Save to file
    image_path = tmp_path / "step_tablet.png"
    Image.fromarray(full_img).save(image_path)

    return image_path


@pytest.fixture
def sample_densities():
    """Create sample density measurements."""
    # Simulate typical Pt/Pd response
    steps = np.linspace(0, 1, 21)
    densities = 0.1 + 2.0 * (steps**0.85)
    return list(densities)


@pytest.fixture
def sample_calibration_record():
    """Create a sample calibration record."""
    return CalibrationRecord(
        paper_type="Arches Platine",
        paper_weight=310,
        exposure_time=180.0,
        metal_ratio=0.5,
        chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
        contrast_agent=ContrastAgent.NA2,
        contrast_amount=5.0,
        developer=DeveloperType.POTASSIUM_OXALATE,
        humidity=50.0,
        temperature=21.0,
        measured_densities=[0.1 + i * 0.1 for i in range(21)],
    )


@pytest.fixture
def populated_database():
    """Create a populated calibration database."""
    from ptpd_calibration.ml.database import CalibrationDatabase

    db = CalibrationDatabase()

    papers = ["Arches Platine", "Bergger COT320", "Hahnemühle Platinum Rag"]
    ratios = [0.3, 0.5, 0.7]
    exposures = [150, 180, 210]

    for paper in papers:
        for ratio in ratios:
            for exposure in exposures:
                record = CalibrationRecord(
                    paper_type=paper,
                    exposure_time=float(exposure),
                    metal_ratio=ratio,
                    chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                    contrast_agent=ContrastAgent.NA2,
                    contrast_amount=5.0,
                    developer=DeveloperType.POTASSIUM_OXALATE,
                    measured_densities=[
                        0.1 + 2.0 * (i / 20) ** (0.7 + ratio * 0.3) for i in range(21)
                    ],
                )
                db.add_record(record)

    return db


# =============================================================================
# Curve Fixtures
# =============================================================================


@pytest.fixture
def sample_curve_data():
    """Create sample curve data."""
    input_values = list(np.linspace(0, 1, 256))
    output_values = [x**0.9 for x in input_values]
    return {
        "input_values": input_values,
        "output_values": output_values,
    }


@pytest.fixture
def sample_curve(sample_curve_data):
    """Create a sample CurveData instance."""
    return CurveData(
        name="Test Curve",
        input_values=sample_curve_data["input_values"],
        output_values=sample_curve_data["output_values"],
    )


@pytest.fixture
def linear_curve():
    """Create a linear (identity) curve."""
    values = list(np.linspace(0, 1, 256))
    return CurveData(
        name="Linear Curve",
        input_values=values,
        output_values=values,
    )


@pytest.fixture
def high_contrast_curve():
    """Create a high-contrast S-curve."""
    input_values = list(np.linspace(0, 1, 256))
    # S-curve formula
    output_values = [1 / (1 + np.exp(-10 * (x - 0.5))) for x in input_values]
    # Normalize to 0-1
    min_val, max_val = min(output_values), max(output_values)
    output_values = [(y - min_val) / (max_val - min_val) for y in output_values]

    return CurveData(
        name="High Contrast Curve",
        input_values=input_values,
        output_values=output_values,
    )


# =============================================================================
# Image Fixtures
# =============================================================================


@pytest.fixture
def grayscale_test_image(tmp_path):
    """Create a simple grayscale test image."""
    img_array = np.linspace(0, 255, 100 * 100).reshape(100, 100).astype(np.uint8)
    image_path = tmp_path / "grayscale_test.png"
    Image.fromarray(img_array, mode="L").save(image_path)
    return image_path


@pytest.fixture
def rgb_test_image(tmp_path):
    """Create a simple RGB test image."""
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "rgb_test.png"
    Image.fromarray(img_array, mode="RGB").save(image_path)
    return image_path


@pytest.fixture
def large_test_image(tmp_path):
    """Create a large test image (2000x1500)."""
    img_array = np.random.randint(0, 255, (1500, 2000), dtype=np.uint8)
    image_path = tmp_path / "large_test.png"
    Image.fromarray(img_array, mode="L").save(image_path)
    return image_path


# =============================================================================
# Chemistry Fixtures
# =============================================================================


@pytest.fixture
def chemistry_params():
    """Standard chemistry parameters for testing."""
    return {
        "print_width": 8.0,
        "print_height": 10.0,
        "metal_ratio": 0.5,
        "contrast_agent": "na2",
        "contrast_drops": 5,
        "coating_factor": 1.0,
    }


@pytest.fixture
def platinum_only_params(chemistry_params):
    """Chemistry params for pure platinum."""
    return {**chemistry_params, "metal_ratio": 1.0}


@pytest.fixture
def palladium_only_params(chemistry_params):
    """Chemistry params for pure palladium."""
    return {**chemistry_params, "metal_ratio": 0.0}


# =============================================================================
# API/HTTP Fixtures
# =============================================================================


@pytest.fixture
def api_headers():
    """Standard API request headers."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
