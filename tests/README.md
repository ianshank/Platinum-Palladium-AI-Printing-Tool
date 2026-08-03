# PTPD Calibration Test Suite

Comprehensive test suite for the Platinum/Palladium Calibration System.

## Overview

This test suite provides comprehensive coverage including:

- **Unit Tests** - Fast, isolated tests for individual components
- **Integration Tests** - Tests for component interactions
- **API Tests** - Tests for REST API endpoints
- **E2E Tests** - Browser-based end-to-end tests with Selenium
- **Performance Tests** - Benchmarks and load testing
- **Visual Regression Tests** - Screenshot comparison tests

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── README.md                # This file
├── unit/                    # Unit tests
│   ├── test_curves.py
│   ├── test_chemistry.py
│   └── ui/                  # UI component tests
├── integration/             # Integration tests
│   └── test_curve_workflow.py
├── api/                     # API endpoint tests
│   ├── conftest.py
│   ├── test_health_endpoints.py
│   ├── test_calibration_endpoints.py
│   ├── test_curve_endpoints.py
│   └── test_ai_endpoints.py
├── e2e/                     # End-to-end tests
│   ├── conftest.py
│   └── selenium/            # Selenium tests
│       ├── conftest.py
│       ├── pages/           # Page Object Models
│       ├── journeys/        # User journey tests
│       └── components/      # Component tests
├── performance/             # Performance tests
│   ├── conftest.py
│   ├── test_curve_performance.py
│   ├── test_database_performance.py
│   └── locustfile.py        # Load testing
├── visual/                  # Visual regression tests
│   ├── conftest.py
│   ├── baselines/           # Baseline screenshots
│   └── test_ui_visual_regression.py
├── utils/                   # Test utilities
│   ├── assertions.py        # Custom assertions
│   ├── data_builders.py     # Test data builders
│   └── mock_factories.py    # Mock object factories
└── fixtures/                # Test fixtures
    └── Platinum_Palladium_V6-CC.quad
```

## Running Tests

### Prerequisites

```bash
# Install all test dependencies
pip install -e ".[all,dev,test]"
```

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ptpd_calibration --cov-report=html

# Run specific test category
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/api/            # API tests only
```

### Running by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only API tests
pytest -m api

# Run slow tests (include long-running tests)
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto               # Use all available cores
pytest -n 4                  # Use 4 cores
```

### E2E Tests with Selenium

```bash
# Run Selenium tests
pytest tests/e2e/selenium/ -m selenium

# Run with visible browser (non-headless)
PTPD_TEST_HEADLESS=false pytest tests/e2e/selenium/

# Run with specific browser
PTPD_TEST_BROWSER=firefox pytest tests/e2e/selenium/
PTPD_TEST_BROWSER=chrome pytest tests/e2e/selenium/
```

### Performance Tests

```bash
# Run benchmarks
pytest tests/performance/ -m benchmark --benchmark-only

# Run with benchmark comparison
pytest tests/performance/ --benchmark-compare

# Run load tests with Locust
cd tests/performance
locust -f locustfile.py --host=http://localhost:8000
```

### Visual Regression Tests

```bash
# Run visual tests
pytest tests/visual/ -m visual

# Update baselines
UPDATE_BASELINES=true pytest tests/visual/
```

## Writing Tests

### Unit Tests

```python
import pytest
from ptpd_calibration.curves import CurveGenerator

class TestCurveGenerator:
    def test_generate_linear_curve(self, sample_densities):
        generator = CurveGenerator()
        curve = generator.generate(sample_densities)

        assert len(curve.input_values) > 0
        assert len(curve.output_values) == len(curve.input_values)
```

### Using Data Builders

```python
from tests.utils.data_builders import (
    DensityBuilder,
    CurveBuilder,
    CalibrationRecordBuilder,
)

def test_with_builders():
    # Create custom density measurements
    densities = (
        DensityBuilder()
        .with_steps(31)
        .with_range(0.1, 2.5)
        .with_noise(0.01)
        .build()
    )

    # Create custom curve
    curve = (
        CurveBuilder()
        .with_gamma(0.8)
        .with_contrast(0.2)
        .build_model()
    )

    # Create calibration record
    record = (
        CalibrationRecordBuilder()
        .with_paper("Arches Platine")
        .with_exposure(180.0)
        .with_metal_ratio(0.5)
        .build()
    )
```

### Using Custom Assertions

```python
from tests.utils.assertions import (
    assert_densities_valid,
    assert_curve_monotonic,
    assert_approximately_equal,
)

def test_with_assertions():
    densities = [0.1, 0.3, 0.5, 0.7, 1.0]

    assert_densities_valid(densities)
    assert_densities_monotonic(densities, increasing=True)
```

### Using Mock Factories

```python
from tests.utils.mock_factories import (
    MockLLMResponse,
    MockDatabaseFactory,
    patch_llm_client,
)

def test_with_mocks():
    # Create mock database
    mock_db = MockDatabaseFactory.populated_database(10)

    # Use context manager for patching
    with patch_llm_client("Mock response"):
        # Test code that uses LLM
        pass
```

### Page Object Pattern (E2E)

```python
from tests.e2e.selenium.pages.calibration_wizard_page import CalibrationWizardPage

class TestCalibration:
    def test_complete_workflow(self, driver, sample_image):
        page = CalibrationWizardPage(driver)

        page.navigate_to_wizard()
        page.upload_step_tablet(sample_image)
        page.select_paper_type("Arches Platine")
        page.click_analyze()
        page.click_generate_curve()

        assert page.is_curve_displayed()
```

## Test Fixtures

Common fixtures available in all tests:

| Fixture | Description |
|---------|-------------|
| `sample_densities` | 21-step density measurements |
| `sample_calibration_record` | Sample CalibrationRecord |
| `sample_step_tablet_image` | Synthetic step tablet image |
| `populated_database` | Database with sample records |
| `tmp_path` | Temporary directory (pytest built-in) |

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit test |
| `@pytest.mark.integration` | Integration test |
| `@pytest.mark.e2e` | End-to-end test |
| `@pytest.mark.selenium` | Requires Selenium WebDriver |
| `@pytest.mark.api` | API endpoint test |
| `@pytest.mark.performance` | Performance test |
| `@pytest.mark.visual` | Visual regression test |
| `@pytest.mark.slow` | Long-running test |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PTPD_TEST_URL` | App URL for E2E tests | `http://127.0.0.1:7861` |
| `PTPD_TEST_HEADLESS` | Run browser headless | `true` |
| `PTPD_TEST_BROWSER` | Browser to use | `chrome` |
| `UPDATE_BASELINES` | Update visual baselines | `false` |

## Coverage Goals

Target: **80%+ code coverage**

Current coverage by module:
- Core modules: 85%+
- API endpoints: 80%+
- UI components: 70%+
- Utilities: 90%+

## CI/CD Integration

Tests run automatically on:
- Every push to `main` and `develop`
- Every pull request

See `.github/workflows/tests.yml` for the full CI configuration.

## Debugging Tests

```bash
# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run specific test
pytest tests/unit/test_curves.py::test_generate_linear

# Run with debugger on failure
pytest --pdb
```

## Contributing

When adding new tests:

1. Place tests in the appropriate directory
2. Use descriptive test names (`test_<what>_<condition>_<expected>`)
3. Add appropriate markers
4. Use fixtures and builders instead of hardcoded data
5. Document complex test scenarios
6. Ensure tests are deterministic and isolated
