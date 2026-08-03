# Testing Guide

## Overview

The PTPD Calibration system uses pytest for backend testing with fixtures, mocking, and integration tests. Target coverage is 80%+ for new code.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures
├── unit/
│   ├── curves/
│   │   ├── test_generator.py
│   │   ├── test_modifier.py
│   │   └── test_exporter.py
│   ├── detection/
│   │   ├── test_detector.py
│   │   └── test_extractor.py
│   └── ...
├── integration/
│   ├── test_api_curves.py               # API integration tests
│   ├── test_api_detection.py
│   └── ...
└── fixtures/
    ├── images/                          # Test images
    ├── curves/                          # Test curve data
    └── mocks.py                         # Mock objects
```

## Writing Tests

### Basic Unit Test

```python
# tests/unit/curves/test_generator.py

import pytest
from ptpd_calibration.curves import CurveGenerator
from ptpd_calibration.curves.exceptions import CurveGenerationError

class TestCurveGenerator:
    """Test suite for CurveGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Provide a CurveGenerator instance."""
        return CurveGenerator()
    
    def test_generate_with_valid_densities(self, generator):
        """Test curve generation with valid input."""
        densities = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        curve = generator.generate(densities)
        
        assert curve is not None
        assert len(curve.points) >= len(densities)
        assert all(0 <= p.output <= 1.0 for p in curve.points)
    
    def test_generate_raises_on_invalid_densities(self, generator):
        """Test that generation fails with invalid input."""
        invalid_densities = [5.0, 6.0]  # Outside normal range
        
        with pytest.raises(CurveGenerationError) as exc_info:
            generator.generate(invalid_densities)
        
        assert "outside valid range" in str(exc_info.value).lower()
    
    def test_generate_empty_list(self, generator):
        """Test that generation requires at least one density."""
        with pytest.raises(ValueError):
            generator.generate([])
```

### Fixture Usage

```python
# tests/conftest.py - Shared fixtures

import pytest
from pathlib import Path
from PIL import Image

@pytest.fixture
def test_image_dir():
    """Provide path to test images."""
    return Path(__file__).parent / "fixtures" / "images"

@pytest.fixture
def sample_tablet_image(test_image_dir):
    """Provide a sample step tablet image for testing."""
    return Image.open(test_image_dir / "tablet_21_step.png")

@pytest.fixture
def sample_densities():
    """Provide sample density measurements."""
    return [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]

@pytest.fixture
def settings_override():
    """Override settings for testing."""
    from ptpd_calibration.config import Settings, DetectionSettings
    return Settings(
        detection=DetectionSettings(
            canny_low_threshold=75,
            morph_kernel_size=7,
        )
    )
```

### Mocking External Dependencies

```python
# tests/unit/llm/test_assistant.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from ptpd_calibration.llm import CalibrationAssistant

class TestCalibrationAssistant:
    """Test LLM assistant with mocked provider."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock the LLM provider."""
        mock = MagicMock()
        mock.generate_text.return_value = "Generated response"
        return mock
    
    @patch('ptpd_calibration.llm.create_client')
    def test_answer_question(self, mock_create_client, mock_llm_client):
        """Test answering user question."""
        mock_create_client.return_value = mock_llm_client
        
        assistant = CalibrationAssistant()
        response = assistant.answer("How do I calibrate?")
        
        assert response == "Generated response"
        mock_llm_client.generate_text.assert_called_once()

    @patch('ptpd_calibration.llm.create_client')
    def test_handles_provider_error(self, mock_create_client):
        """Test graceful error handling for provider failures."""
        from ptpd_calibration.llm.exceptions import LLMProviderError
        
        mock_client = MagicMock()
        mock_client.generate_text.side_effect = LLMProviderError("API down")
        mock_create_client.return_value = mock_client
        
        assistant = CalibrationAssistant()
        
        with pytest.raises(LLMProviderError):
            assistant.answer("Question")
```

### Integration Tests

```python
# tests/integration/test_api_curves.py

import pytest
from fastapi.testclient import TestClient
from ptpd_calibration.api.server import create_app

@pytest.fixture
def client():
    """Provide FastAPI test client."""
    app = create_app()
    return TestClient(app)

def test_generate_curve_endpoint(client):
    """Test curve generation via API."""
    response = client.post(
        "/curves/generate",
        json={
            "densities": [0.1, 0.5, 1.0, 1.5, 2.0],
            "name": "Test Curve",
            "curve_type": "linear",
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "curve" in data
    assert data["name"] == "Test Curve"

def test_export_curve_endpoint(client):
    """Test curve export via API."""
    # First generate a curve
    gen_response = client.post(
        "/curves/generate",
        json={"densities": [0.1, 0.5, 1.0, 1.5, 2.0]}
    )
    curve_id = gen_response.json()["id"]
    
    # Export it
    response = client.post(
        "/curves/export",
        json={
            "curve_id": curve_id,
            "format": "json",
        }
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

def test_upload_scan_invalid_image(client):
    """Test upload endpoint with invalid image."""
    response = client.post(
        "/scan/upload",
        files={"file": ("invalid.txt", b"not an image")}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "INVALID_IMAGE"
```

## Testing Patterns

### Pattern 1: Testing with Different Configurations

```python
@pytest.mark.parametrize("kernel_size,expected", [
    (3, True),   # Valid odd size
    (5, True),
    (7, True),
    (4, False),  # Even size (invalid)
    (6, False),
])
def test_detection_with_different_kernels(kernel_size, expected):
    """Test detection with various kernel sizes."""
    settings = Settings(
        detection=DetectionSettings(morph_kernel_size=kernel_size)
    )
    detector = StepTabletDetector(settings=settings)
    
    if expected:
        # Should succeed
        assert detector.validate_settings()
    else:
        # Should fail
        with pytest.raises(ValueError):
            detector.validate_settings()
```

### Pattern 2: Testing File I/O

```python
def test_save_and_load_curve(tmp_path):
    """Test saving and loading curves."""
    import json
    
    # Create and save curve
    curve = CurveData(
        id="test_curve",
        points=[Point(0.0, 0.0), Point(0.5, 0.5), Point(1.0, 1.0)],
    )
    
    # Save to temporary file
    curve_file = tmp_path / "curve.json"
    with open(curve_file, 'w') as f:
        json.dump(curve.dict(), f)
    
    # Load and verify
    with open(curve_file, 'r') as f:
        loaded = CurveData(**json.load(f))
    
    assert loaded.id == curve.id
    assert len(loaded.points) == len(curve.points)
```

### Pattern 3: Testing Async Functions

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_curve_generation():
    """Test asynchronous curve generation."""
    generator = CurveGenerator()
    
    # Create task
    task = asyncio.create_task(
        generator.generate_async([0.1, 0.5, 1.0, 1.5, 2.0])
    )
    
    # Should complete without timeout
    curve = await asyncio.wait_for(task, timeout=5.0)
    
    assert curve is not None
    assert len(curve.points) > 0
```

### Pattern 4: Testing Error Conditions

```python
def test_curve_export_with_invalid_format():
    """Test error handling for unsupported export formats."""
    exporter = CurveExporter()
    curve = CurveData(...)
    
    with pytest.raises(ValueError) as exc_info:
        exporter.export(curve, format="invalid_format")
    
    assert "unsupported format" in str(exc_info.value).lower()
    assert "invalid_format" in str(exc_info.value)
```

## Running Tests

### Command Line

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ptpd_calibration --cov-report=html tests/

# Run specific file
pytest tests/unit/curves/test_generator.py

# Run specific test class
pytest tests/unit/curves/test_generator.py::TestCurveGenerator

# Run specific test
pytest tests/unit/curves/test_generator.py::TestCurveGenerator::test_generate_with_valid_densities

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run only tests matching pattern
pytest -k "tablet" 

# Run only failed tests
pytest --lf

# Debug mode
pytest --pdb  # Drop into debugger on failure
```

### pytest Configuration

```ini
# pytest.ini or setup.cfg

[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    asyncio: marks tests as async
```

## Coverage Targets

### Per-Component Coverage

```bash
# Measure coverage
pytest --cov=ptpd_calibration --cov-report=term-missing

# Generate HTML report
pytest --cov=ptpd_calibration --cov-report=html

# Check minimum coverage
pytest --cov=ptpd_calibration --cov-fail-under=80
```

### Coverage Goals

| Component | Target | Status |
|-----------|--------|--------|
| Curves | 85% | ✓ |
| Detection | 80% | ✓ |
| ML Models | 75% | ✓ |
| LLM | 70% | ✓ |
| API | 85% | ✓ |
| Config | 90% | ✓ |
| Overall | 80% | Track |

## Fixtures and Test Data

### Creating Test Fixtures

```python
# tests/fixtures/create_fixtures.py

from PIL import Image
import numpy as np
from pathlib import Path

def create_test_tablet_image(
    width: int = 1920,
    height: int = 1440,
    num_patches: int = 21,
) -> Image.Image:
    """Create synthetic step tablet image for testing."""
    # Create gradient image
    gradient = np.linspace(0, 255, num_patches * (width // num_patches))
    image_array = np.tile(gradient[:width], (height, 1))
    
    return Image.fromarray(image_array.astype('uint8'))

# Usage in tests
@pytest.fixture
def test_tablet(tmp_path):
    return create_test_tablet_image()
```

### Reusing Test Data

```python
# tests/fixtures/curves.py

import pytest
import json

@pytest.fixture
def calibration_data():
    """Load reference calibration data."""
    with open(Path(__file__).parent / "data" / "calibration.json") as f:
        return json.load(f)

# Usage
def test_with_reference_data(calibration_data):
    assert calibration_data["name"] == "Reference"
```

## Advanced Testing

### Testing with Docker

```dockerfile
# Dockerfile.test
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"
CMD ["pytest", "--cov=ptpd_calibration", "tests/"]
```

```bash
# Build and run tests
docker build -f Dockerfile.test -t ptpd-test .
docker run ptpd-test
```

### Performance Testing

```python
import pytest
from time import time

def test_curve_generation_performance():
    """Test that curve generation completes within time limit."""
    generator = CurveGenerator()
    densities = [i * 0.1 for i in range(100)]
    
    start = time()
    curve = generator.generate(densities)
    elapsed = time() - start
    
    assert elapsed < 1.0, f"Generation took {elapsed}s, expected < 1.0s"
```

### Mutation Testing

```bash
# Using mutmut for mutation testing
pip install mutmut

# Run mutation tests
mutmut run --tests-dir tests --paths-to-mutate ptpd_calibration

# View results
mutmut results
```

## Best Practices

### 1. Use Descriptive Test Names

```python
# Good
def test_detect_tablet_raises_on_blank_image():
    pass

# Avoid
def test_detect():
    pass
```

### 2. One Assertion per Scenario

```python
# Good: Clear, focused
def test_curve_has_correct_length():
    curve = generate_curve(...)
    assert len(curve.points) == expected_length

def test_curve_values_are_normalized():
    curve = generate_curve(...)
    assert all(0 <= p.output <= 1.0 for p in curve.points)

# Avoid: Multiple unrelated assertions
def test_curve():
    curve = generate_curve(...)
    assert len(curve.points) == expected_length
    assert all(0 <= p.output <= 1.0 for p in curve.points)
    assert curve.name == "Test"
    assert curve.type == "linear"
```

### 3. Use Fixtures for Setup

```python
# Good: Reusable fixture
@pytest.fixture
def generator():
    return CurveGenerator()

def test_a(generator):
    assert generator is not None

def test_b(generator):
    assert generator is not None

# Avoid: Repeating setup
def test_a():
    gen = CurveGenerator()
    assert gen is not None

def test_b():
    gen = CurveGenerator()
    assert gen is not None
```

### 4. Test Public APIs First

```python
# Good: Test the contract
def test_generate_accepts_valid_densities():
    curve = CurveGenerator().generate([0.1, 0.5, 1.0])
    assert curve is not None

# Avoid: Testing internals
def test_internal_cache_is_valid():
    gen = CurveGenerator()
    assert hasattr(gen, '_cache')
```

### 5. Clean Up Resources

```python
# Good: Use fixtures and cleanup
@pytest.fixture
def temp_model_dir(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    yield model_dir
    # Cleanup happens automatically

def test_model_loading(temp_model_dir):
    # Use temp_model_dir
    pass

# Avoid: Manual cleanup
def test_model_loading():
    model_dir = Path("./temp_models")
    model_dir.mkdir()
    try:
        # Test code
        pass
    finally:
        # Cleanup often forgotten
        pass
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest --cov=ptpd_calibration --cov-report=xml tests/
      - uses: codecov/codecov-action@v2
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Testing as part of system design
- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
