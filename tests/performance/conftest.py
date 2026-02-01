"""
Performance test fixtures and configuration.

Provides fixtures for benchmark tests and performance measurements.
"""

import numpy as np
import pytest


def pytest_configure(config):
    """Add performance test markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


@pytest.fixture
def small_densities():
    """Create a small set of density measurements (21 steps)."""
    steps = np.linspace(0, 1, 21)
    densities = 0.1 + 2.0 * (steps**0.85)
    return list(densities)


@pytest.fixture
def medium_densities():
    """Create a medium set of density measurements (51 steps)."""
    steps = np.linspace(0, 1, 51)
    densities = 0.1 + 2.0 * (steps**0.85)
    return list(densities)


@pytest.fixture
def large_densities():
    """Create a large set of density measurements (256 steps)."""
    steps = np.linspace(0, 1, 256)
    densities = 0.1 + 2.0 * (steps**0.85)
    return list(densities)


@pytest.fixture
def small_curve():
    """Create a small curve (256 points)."""
    input_values = list(np.linspace(0, 1, 256))
    output_values = [x**0.9 for x in input_values]
    return {"input": input_values, "output": output_values}


@pytest.fixture
def large_curve():
    """Create a large curve (1024 points)."""
    input_values = list(np.linspace(0, 1, 1024))
    output_values = [x**0.9 for x in input_values]
    return {"input": input_values, "output": output_values}


@pytest.fixture
def sample_step_tablet_image(tmp_path):
    """Create a sample step tablet image for testing."""
    from PIL import Image

    width, height = 420, 100
    num_patches = 21
    patch_width = width // num_patches

    img_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_patches):
        value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
        x_start = i * patch_width
        x_end = (i + 1) * patch_width
        img_array[:, x_start:x_end] = value

    # Create RGB image
    full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
    full_img[20 : height + 20, 20 : width + 20, 0] = img_array
    full_img[20 : height + 20, 20 : width + 20, 1] = img_array
    full_img[20 : height + 20, 20 : width + 20, 2] = img_array

    image_path = tmp_path / "step_tablet_perf.png"
    Image.fromarray(full_img).save(image_path)

    return image_path


@pytest.fixture
def large_image(tmp_path):
    """Create a large image for performance testing."""
    from PIL import Image

    # Create a 4000x3000 image (like a high-res scan)
    img_array = np.random.randint(0, 255, (3000, 4000), dtype=np.uint8)
    image_path = tmp_path / "large_image.png"
    Image.fromarray(img_array, mode="L").save(image_path)

    return image_path


@pytest.fixture
def populated_database():
    """Create a populated calibration database for performance tests."""
    from ptpd_calibration.core.models import CalibrationRecord
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
    from ptpd_calibration.ml.database import CalibrationDatabase

    db = CalibrationDatabase()

    papers = ["Arches Platine", "Bergger COT320", "Hahnemuhle Platinum Rag"]
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


@pytest.fixture
def large_database():
    """Create a large database for performance stress tests."""
    from ptpd_calibration.core.models import CalibrationRecord
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
    from ptpd_calibration.ml.database import CalibrationDatabase

    db = CalibrationDatabase()

    # Create 1000 records
    for i in range(1000):
        record = CalibrationRecord(
            paper_type=f"Paper_{i % 10}",
            exposure_time=float(120 + (i % 120)),
            metal_ratio=0.3 + (i % 7) * 0.1,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
            measured_densities=[0.1 + 2.0 * (j / 20) ** 0.8 for j in range(21)],
        )
        db.add_record(record)

    return db
