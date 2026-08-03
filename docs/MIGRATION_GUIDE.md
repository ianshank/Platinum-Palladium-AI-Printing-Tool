# Migration Guide - Refactored Architecture

## Overview

This guide helps developers understand the changes made during the refactoring from a monolithic structure to a modular, domain-driven architecture.

## What Changed

### Before: Monolithic Structure

```
src/ptpd_calibration/
├── config.py              # Single large config file (53KB)
├── ui/
│   ├── gradio_app.py     # Massive monolithic app (4,337 lines)
│   ├── handlers.py       # All handlers in one file
│   └── ...
├── curves.py             # Curves in single file
├── detection.py          # Detection in single file
└── ...
```

**Problems**:
- Hard to navigate large files
- Tight coupling between concerns
- Difficult to test individual components
- No clear separation of domains
- Configuration scattered across files

### After: Modular, Domain-Driven Structure

```
src/ptpd_calibration/
├── config/               # Configuration domain
│   ├── __init__.py      # Settings classes (pydantic)
│   └── calculations.py  # Config helpers
├── config.py            # Root settings (legacy, kept for compatibility)
├── core/                # Infrastructure & shared
│   ├── logging.py       # Structured logging
│   ├── models.py        # Shared data models
│   ├── types.py         # Type definitions
│   └── exceptions.py    # Base exception classes
├── ui/                  # Modularized Gradio UI
│   ├── config/          # UI configuration
│   ├── handlers/        # Event handlers
│   ├── validators/      # Input validation
│   ├── tabs/            # Tab components
│   └── gradio_app.py   # Thin orchestration (~100 lines)
├── curves/              # Curves module
│   ├── __init__.py      # Public API
│   ├── generator.py     # CurveGenerator class
│   ├── modifier.py      # CurveModifier class
│   ├── exporter.py      # Export formats
│   ├── exceptions.py    # Domain-specific errors
│   └── utils.py         # Helper functions
├── detection/           # Detection module
├── ml/                  # ML module
├── agents/              # Agent system (NEW)
├── ai/                  # AI analysis (NEW)
├── workflow/            # Workflow automation (NEW)
├── monitoring/          # Performance monitoring (NEW)
└── ... (other domains)
```

**Improvements**:
- Clear separation of concerns
- Domain-focused modules
- Easier to test and maintain
- Centralized configuration
- Structured logging throughout
- Consistent error handling

## Migration Path

### Step 1: Understand Domain Boundaries

Each domain module handles one feature area:

| Domain | Responsible For | Files |
|--------|-----------------|-------|
| `curves/` | Curve generation, export, modification | generator.py, exporter.py, modifier.py |
| `detection/` | Step tablet detection, density extraction | detector.py, extractor.py |
| `ml/` | Machine learning models, predictions | models.py, trainer.py, predictor.py |
| `llm/` | LLM provider integration, chat | client.py, assistant.py |
| `api/` | FastAPI server, routes, models | server.py, models.py, routers |
| `config/` | Configuration management | \_\_init\_\_.py, calculations.py |
| `core/` | Infrastructure layers | logging.py, models.py, exceptions.py |

### Step 2: Update Imports

#### Old Way
```python
# Old: Import everything from main module
from ptpd_calibration import CurveGenerator, StepTabletDetector

# Or use config from monolithic file
from ptpd_calibration.config import Settings
```

#### New Way
```python
# New: Import from specific domain modules
from ptpd_calibration.curves import CurveGenerator
from ptpd_calibration.detection import StepTabletDetector

# Config still works from top-level (backward compatible)
from ptpd_calibration.config import Settings, get_settings

# Or import directly from config domain
from ptpd_calibration.config import get_settings
```

### Step 3: Use Structured Logging

#### Old Way
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Curve generated with {len(points)} points")
```

#### New Way
```python
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)
logger.info(
    "Curve generated",
    extra={
        "num_points": len(points),
        "duration_ms": elapsed,
        "user_id": user.id,
    }
)
```

### Step 4: Handle Exceptions with Domain Classes

#### Old Way
```python
try:
    tablet = detector.detect(image)
except Exception as e:
    logger.error(f"Detection failed: {e}")
    return None
```

#### New Way
```python
from ptpd_calibration.detection.exceptions import TabletDetectionError

try:
    tablet = detector.detect(image)
except TabletDetectionError as e:
    logger.error(
        "Tablet detection failed",
        extra={"error_code": e.error_code}
    )
    # HTTP layer will convert to proper status code
    raise
```

### Step 5: Use Configuration System

#### Old Way
```python
# Hardcoded or environment-based magic strings
CANNY_THRESHOLD = int(os.environ.get("CANNY_THRESHOLD", 50))
MAX_POINTS = 256
```

#### New Way
```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Type-safe, validated configuration
canny_threshold = settings.detection.canny_low_threshold
max_points = settings.curves.max_curve_points

# Override via environment
# PTPD_DETECTION__CANNY_LOW_THRESHOLD=75
```

## Common Migration Scenarios

### Scenario 1: Adding a New Service

**Old Approach**: Add to monolithic file, couple everything together

**New Approach**: Create a new domain module

```python
# src/ptpd_calibration/myfeature/__init__.py
"""My new feature module."""

from .service import MyFeatureService
from .exceptions import MyFeatureError

__all__ = ["MyFeatureService", "MyFeatureError"]

# src/ptpd_calibration/myfeature/service.py
"""Service implementation."""

from ptpd_calibration.core.logging import get_logger
from .exceptions import MyFeatureError

logger = get_logger(__name__)

class MyFeatureService:
    """Service for my feature."""
    
    def process(self, data):
        """Process data."""
        logger.info("Processing started")
        try:
            result = self._do_work(data)
            logger.info("Processing completed")
            return result
        except Exception as e:
            logger.error("Processing failed", extra={"error": str(e)})
            raise MyFeatureError(f"Failed to process: {e}") from e

# src/ptpd_calibration/myfeature/exceptions.py
"""Domain-specific exceptions."""

from ptpd_calibration.core.exceptions import PTCDError

class MyFeatureError(PTCDError):
    """Base exception for myfeature module."""
    error_code = "MYFEATURE_ERROR"
    http_status_code = 400
```

### Scenario 2: Migrating Hardcoded Values

**Old Code**:
```python
# Hardcoded in function
def detect_tablet(image):
    CANNY_LOW = 50
    CANNY_HIGH = 150
    MORPH_SIZE = 5
    MIN_AREA_RATIO = 0.01
    
    edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    # ... etc
```

**New Code**:
```python
from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

class StepTabletDetector:
    def __init__(self):
        self.settings = get_settings()
    
    def detect(self, image):
        """Detect step tablet in image."""
        # Use configured values
        canny_low = self.settings.detection.canny_low_threshold
        canny_high = self.settings.detection.canny_high_threshold
        morph_size = self.settings.detection.morph_kernel_size
        min_area = self.settings.detection.min_contour_area_ratio
        
        logger.debug(
            "Detection parameters",
            extra={
                "canny_low": canny_low,
                "canny_high": canny_high,
                "morph_size": morph_size,
            }
        )
        
        edges = cv2.Canny(image, canny_low, canny_high)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_size, morph_size)
        )
        # ... etc
```

### Scenario 3: Creating FastAPI Routes

**Old Code**: Routes in monolithic server file

**New Code**: Use domain modules with clear separation

```python
# In api/server.py
from fastapi import FastAPI
from ptpd_calibration.curves import CurveGenerator
from ptpd_calibration.api.models import CurveGenerateRequest, CurveGenerateResponse

app = FastAPI()

@app.post("/curves/generate")
async def generate_curve(request: CurveGenerateRequest) -> CurveGenerateResponse:
    """Generate calibration curve."""
    try:
        generator = CurveGenerator()
        curve = generator.generate(request.densities)
        
        return CurveGenerateResponse(
            status="success",
            curve=curve,
        )
    except CurveGenerationError as e:
        # FastAPI exception handler converts to HTTP 422
        raise HTTPException(status_code=422, detail={"error": e.error_code})
```

## Backward Compatibility

### Old Imports Still Work

For backward compatibility, old import paths work:

```python
# These still work (re-exported from __init__.py)
from ptpd_calibration import CurveGenerator
from ptpd_calibration import StepTabletDetector
from ptpd_calibration.config import Settings

# But prefer new imports
from ptpd_calibration.curves import CurveGenerator
from ptpd_calibration.detection import StepTabletDetector
from ptpd_calibration.config import get_settings
```

The `__init__.py` files re-export public APIs for convenience, but new code should use domain-specific imports.

## Testing Migration

### Old Test Structure
```
tests/
├── test_curves.py           # Everything in one file
├── test_detection.py
└── fixtures.py
```

### New Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── curves/
│   │   ├── test_generator.py
│   │   ├── test_modifier.py
│   │   └── test_exporter.py
│   └── detection/
│       └── test_detector.py
├── integration/
│   ├── test_api_curves.py
│   └── test_api_detection.py
└── fixtures/
    ├── images/
    └── mocks.py
```

### Old Test Code
```python
# Single file testing everything
def test_generate_curve():
    generator = CurveGenerator()
    curve = generator.generate([0.1, 0.5, 1.0])
    assert curve is not None

def test_detect_tablet():
    detector = StepTabletDetector()
    # ... test code
```

### New Test Code
```python
# tests/unit/curves/test_generator.py
class TestCurveGenerator:
    @pytest.fixture
    def generator(self):
        return CurveGenerator()
    
    def test_generate_with_valid_densities(self, generator):
        curve = generator.generate([0.1, 0.5, 1.0])
        assert curve is not None

# tests/integration/test_api_curves.py
def test_api_generate_curve(client):
    response = client.post(
        "/curves/generate",
        json={"densities": [0.1, 0.5, 1.0]}
    )
    assert response.status_code == 200
```

## Troubleshooting

### Issue: Import errors after migration

**Symptom**: `ModuleNotFoundError: No module named 'curves'`

**Solution**: Update import paths

```python
# Old
from ptpd_calibration.curves import CurveGenerator

# New (if moved to domain module)
from ptpd_calibration.curves.generator import CurveGenerator
# Or use re-export
from ptpd_calibration.curves import CurveGenerator  # Still works
```

### Issue: Configuration not loading

**Symptom**: Settings appear to have default values, not from environment

**Solution**: Ensure `setup_logging()` is called before using settings

```python
# At application startup
from ptpd_calibration.core.logging import setup_logging
from ptpd_calibration.config import get_settings

setup_logging()  # Configure logging
settings = get_settings()  # Load configuration from environment

# Environment variables must be set before this point
# PTPD_DETECTION__CANNY_LOW_THRESHOLD=75
```

### Issue: Tests failing after migration

**Symptom**: Tests pass locally but fail in CI

**Solution**: Ensure fixtures and mocks are properly set up

```python
# tests/conftest.py - Central fixtures
import pytest

@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings between tests."""
    # Prevent settings from persisting across tests
    yield

@pytest.fixture
def test_settings():
    """Provide test configuration."""
    from ptpd_calibration.config import Settings
    return Settings()
```

## Performance Impact

The refactoring should have **no negative performance impact**. If anything:

- **Faster imports**: Only import what you need (lazy imports)
- **Better caching**: Domain modules can be cached per process
- **Reduced memory**: Smaller module files load faster

Example:

```python
# Before: Everything imported at once
from ptpd_calibration import *  # Slow, imports everything

# After: Import only what you need
from ptpd_calibration.curves import CurveGenerator  # Fast, minimal import
```

## Migration Checklist

When migrating code from old to new structure:

- [ ] Update imports to use domain-specific modules
- [ ] Replace hardcoded values with configuration
- [ ] Replace generic logging with structured logging
- [ ] Replace generic exceptions with domain-specific exceptions
- [ ] Update tests to match new test structure
- [ ] Verify backward compatibility imports work
- [ ] Update documentation with new structure
- [ ] Test in isolated environment
- [ ] Update CI/CD pipelines if needed
- [ ] Remove deprecated code after grace period

## Timeline

- **Phase 1 (Completed)**: Core infrastructure (config, logging, exceptions)
- **Phase 2 (In Progress)**: Domain modules (curves, detection, ml)
- **Phase 3 (Next)**: API migration and endpoint updates
- **Phase 4 (Future)**: Full deprecation of old imports (year+)

## Questions?

Refer to:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) - Configuration details
- [LOGGING.md](LOGGING.md) - Logging patterns
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Exception handling
- CLAUDE.md - Project-wide guidelines
