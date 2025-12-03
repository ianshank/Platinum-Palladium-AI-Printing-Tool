"""
Shared fixtures for PTPD Calibration tests.
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType


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
    densities = 0.1 + 2.0 * (steps ** 0.85)
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
                        0.1 + 2.0 * (i / 20) ** (0.7 + ratio * 0.3)
                        for i in range(21)
                    ],
                )
                db.add_record(record)

    return db
