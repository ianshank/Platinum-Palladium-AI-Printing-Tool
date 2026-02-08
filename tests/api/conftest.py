"""
API test fixtures and configuration.

Provides FastAPI test client, mock services, and test data generators.
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Try to import test dependencies
try:
    import httpx
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    httpx = None


def pytest_configure(config):
    """Add API test markers."""
    config.addinivalue_line("markers", "api: mark test as an API endpoint test")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip API tests if FastAPI is not available."""
    if not FASTAPI_AVAILABLE:
        skip_api = pytest.mark.skip(reason="FastAPI not installed")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


@pytest.fixture(scope="session")
def temp_upload_dir():
    """Create a temporary directory for file uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def app():
    """Create the FastAPI application for testing."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    # Set test environment
    os.environ["PTPD_ENV"] = "test"

    from ptpd_calibration.api.server import create_app

    return create_app()


@pytest.fixture(scope="module")
def client(app) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def async_client(app):
    """Create an async test client for async tests."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def sample_densities():
    """Create sample density measurements."""
    import numpy as np

    steps = np.linspace(0, 1, 21)
    densities = 0.1 + 2.0 * (steps**0.85)
    return list(densities)


@pytest.fixture
def sample_curve_data():
    """Create sample curve data."""
    import numpy as np

    input_values = list(np.linspace(0, 1, 256))
    output_values = [x**0.9 for x in input_values]
    return {"input_values": input_values, "output_values": output_values}


@pytest.fixture
def sample_step_tablet_image(tmp_path):
    """Create a sample step tablet image for testing."""
    import numpy as np
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

    # Create RGB image with margin
    full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
    full_img[20 : height + 20, 20 : width + 20, 0] = img_array
    full_img[20 : height + 20, 20 : width + 20, 1] = img_array
    full_img[20 : height + 20, 20 : width + 20, 2] = img_array

    image_path = tmp_path / "step_tablet.png"
    Image.fromarray(full_img).save(image_path)

    return image_path


@pytest.fixture
def sample_quad_content():
    """Create sample .quad file content."""
    return """## Quad profile created by PTPD Calibration
## Profile: Test Profile
## Resolution: 1440
## Media Type: matte

QUAD_RES=1440

[K]
0=0
16=12
32=28
48=44
64=60
80=76
96=92
112=108
128=124
144=140
160=156
176=172
192=188
208=204
224=220
240=236
255=255
"""


@pytest.fixture
def calibration_request_data():
    """Create sample calibration request data."""
    return {
        "paper_type": "Arches Platine",
        "exposure_time": 180.0,
        "metal_ratio": 0.5,
        "contrast_agent": "na2",
        "contrast_amount": 5.0,
        "developer": "potassium_oxalate",
        "chemistry_type": "platinum_palladium",
        "densities": [0.1 + i * 0.1 for i in range(21)],
        "notes": "Test calibration",
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for AI endpoints."""
    return AsyncMock(return_value="This is a mock LLM response for testing purposes.")


@pytest.fixture
def mock_assistant(mock_llm_response):
    """Mock the AI assistant."""
    mock = MagicMock()
    mock.chat = mock_llm_response
    mock.suggest_recipe = mock_llm_response
    mock.troubleshoot = mock_llm_response
    return mock


@pytest.fixture
def headers():
    """Common request headers."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
