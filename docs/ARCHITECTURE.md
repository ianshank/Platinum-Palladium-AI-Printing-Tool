# Refactored Backend Architecture

## Overview

The PTPD Calibration backend has been refactored into a modular, domain-driven architecture that supports expanding features while maintaining code clarity and testability.

## System Architecture (C4 Model)

### Level 1: System Context

```
┌─────────────────────────────────────────────────────────────┐
│                     PTPD Calibration System                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Frontend   │  │   Backend    │  │   Database   │      │
│  │  (React TS)  │←→│  (FastAPI)   │←→│  (Optional)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            ↕                                │
│                    ┌───────────────┐                        │
│                    │  External      │                       │
│                    │  Services      │                       │
│                    │ (LLM, GCP, HF) │                       │
│                    └───────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Level 2: Container Diagram

```
Backend FastAPI Container:

┌────────────────────────────────────────────────────────┐
│  FastAPI Server (src/ptpd_calibration/api/server.py)  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Router Layer                                     │ │
│  │                                                  │ │
│  │  • Health / Root endpoints                      │ │
│  │  • Curve endpoints (generate, export, modify)   │ │
│  │  • Detection endpoints (scan upload, analyze)   │ │
│  │  • Chat endpoints (LLM assistant)               │ │
│  │  • Deep learning router (training, prediction)  │ │
│  │  • MCTS router (optimization search)            │ │
│  └──────────────────────────────────────────────────┘ │
│                     ↓                                  │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Service Layer (Domain Modules)                   │ │
│  │                                                  │ │
│  │  curves/         ← Curve generation & export    │ │
│  │  detection/      ← Step tablet detection        │ │
│  │  ml/             ← ML models & predictions      │ │
│  │  llm/            ← LLM integration              │ │
│  │  agents/         ← Multi-agent orchestration    │ │
│  │  ai/             ← AI analysis                  │ │
│  │  advanced/       ← Advanced features            │ │
│  │  workflow/       ← Recipe management            │ │
│  │  calculations/   ← Technical calculations       │ │
│  │  monitoring/     ← Performance metrics          │ │
│  └──────────────────────────────────────────────────┘ │
│                     ↓                                  │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Infrastructure Layer                             │ │
│  │                                                  │ │
│  │  core/logging/   ← Structured logging            │ │
│  │  config/         ← Settings management           │ │
│  │  core/models/    ← Data models                   │ │
│  │  integrations/   ← Hardware/cloud integrations   │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Level 3: Component Diagram

```
Curves Module:

┌─────────────────────────────────────┐
│      curves/                        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ CurveGenerator               │  │
│  │ • generate_linearization()   │  │
│  │ • create_target_curve()      │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ CurveModifier                │  │
│  │ • smooth()                   │  │
│  │ • blend()                    │  │
│  │ • enhance()                  │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ CurveExporter                │  │
│  │ • to_qtr()                   │  │
│  │ • to_piezography()           │  │
│  │ • to_json()                  │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ File I/O                      │  │
│  │ • load_curve()               │  │
│  │ • save_curve()               │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Module Organization

### Domain Modules (Business Logic)

Each domain module encapsulates a feature area:

```
domain_module/
├── __init__.py           # Public API exports
├── models.py            # Domain models (if not in core/)
├── service.py           # Main service class
├── exceptions.py        # Domain-specific exceptions
└── utils.py            # Helper functions
```

Example: `curves/`
```
curves/
├── __init__.py          # Exports: CurveGenerator, CurveExporter, etc.
├── generator.py         # CurveGenerator class
├── modifier.py          # CurveModifier class
├── exporter.py          # CurveExporter, QTRExporter, PiezographyExporter
├── io.py               # load_curve(), save_curve()
├── exceptions.py       # CurveError, CurveExportError
└── utils.py            # interpolate(), smooth_curve(), etc.
```

### Core Modules (Infrastructure)

Infrastructure layers used by all domain modules:

```
core/
├── __init__.py          # Central exports
├── logging.py          # Structured logging (JSONFormatter, get_logger)
├── models.py           # Shared data models (CalibrationRecord, CurveData)
├── types.py            # Type definitions (ChemistryType, CurveType)
└── exceptions.py       # Base exception classes
```

### Configuration Module

Centralized settings management with environment-based overrides:

```
config/
├── __init__.py          # Settings classes (pydantic BaseSettings)
└── calculations.py      # Config calculations/validators
```

Key settings classes:
- `Settings` - Root settings
- `DetectionSettings` - Edge detection, morphology, contour parameters
- `ExtractionSettings` - Sampling, outlier rejection, density calculation
- `APISettings` - Server, CORS, upload limits
- `LLMSettings` - Provider, model, API keys

## Data Flow

### Typical Request Flow

```
1. FastAPI Router (api/server.py)
   ↓
   Validates input (Pydantic model)
   ↓
2. Service Layer (e.g., curves/generator.py)
   ↓
   Calls domain logic
   ↓
3. Infrastructure (logging, config)
   ↓
   Logs structured data, applies settings
   ↓
4. Returns response model
   ↓
5. FastAPI converts to JSON + HTTP status
   ↓
6. Response to client
```

### Error Handling Flow

```
Domain Layer raises exception
   (e.g., TabletDetectionError)
   ↓
Caught by router or middleware
   ↓
Logged with context
   ↓
Converted to HTTP error response
   (422, 400, 500, etc.)
   ↓
Response to client with error details
```

## Patterns and Best Practices

### 1. Module Independence

Each domain module should minimize dependencies on other domains:

```python
# Good: Import only what's needed
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves import CurveGenerator

# Avoid: Circular imports
# Don't: from ptpd_calibration.agents import AgentPlanner (in curves/)
```

### 2. Configuration Injection

Pass configuration to services rather than importing globally:

```python
# Good
from ptpd_calibration.config import get_settings
class CurveGenerator:
    def __init__(self):
        self.settings = get_settings()
        self.max_points = self.settings.curves.max_curve_points

# Avoid
# Hardcoding values: max_points = 256
```

### 3. Logging Context

Use structured logging with context for traceability:

```python
from ptpd_calibration.core.logging import get_logger
logger = get_logger(__name__)

logger.info(
    "Curve generated",
    extra={
        "curve_id": curve.id,
        "num_points": len(curve.points),
        "duration_ms": duration,
    }
)
```

### 4. Optional Dependencies

For heavy/optional features, use try/except imports:

```python
# In __init__.py
from contextlib import suppress

with suppress(ImportError):
    from ptpd_calibration.agents import CalibrationAgent

# Usage code checks availability
try:
    agent = CalibrationAgent()
except NameError:
    raise ImportError("Agents not available - install ptpd-calibration[agents]")
```

### 5. Exception Hierarchy

Use domain-specific exceptions with clear error codes:

```python
# detection/exceptions.py
class DetectionError(Exception):
    """Base exception for detection module."""
    pass

class TabletDetectionError(DetectionError):
    """Raised when step tablet cannot be detected."""
    pass

class PatchExtractionError(DetectionError):
    """Raised when patch extraction fails."""
    pass
```

## Performance Considerations

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def load_model(model_name: str):
    """Cache loaded ML models."""
    return torch.load(f"models/{model_name}.pt")
```

### Image Processing

```python
# Use generators for large image processing
def process_image_patches(image, patch_size):
    """Yield patches without loading entire image."""
    for y in range(0, image.height, patch_size):
        for x in range(0, image.width, patch_size):
            yield image.crop((x, y, x+patch_size, y+patch_size))
```

### Async Operations

```python
# FastAPI supports async handlers
@app.post("/curves/generate")
async def generate_curve(request: CurveRequest):
    """Async curve generation for better concurrency."""
    # Non-blocking I/O for LLM calls, database queries
    result = await llm_client.analyze_densities(request.densities)
    return result
```

## Integration Points

### Frontend Integration

- REST API endpoints in `api/server.py`
- Pydantic response models for type safety
- OpenAPI schema auto-generated from models

### LLM Integration

- Multi-provider support (Anthropic, OpenAI, Vertex AI)
- Circuit breaker pattern for reliability
- Logging of requests/responses for debugging

### Hardware Integration

- Printer drivers (Canon, Epson)
- Spectrophotometer interfaces (X-Rite)
- Weather provider integration

### Cloud Integration

- GCP Vertex AI for ML models
- GCS for artifact storage
- HuggingFace Hub for model repository

## Development Workflow

When adding a new feature:

1. **Create domain module** if it's a new area
2. **Define models** in `core/models.py` or domain-specific
3. **Add configuration** to `config.py` if needed
4. **Write service classes** in domain module
5. **Create FastAPI routes** in `api/server.py`
6. **Add logging** using `core/logging.get_logger()`
7. **Add error handling** with domain-specific exceptions
8. **Write tests** with ~80% coverage
9. **Update CLAUDE.md** if architectural change

## See Also

- [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) - Configuration management details
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Error handling patterns
- [LOGGING.md](LOGGING.md) - Logging architecture and usage
- [TESTING.md](TESTING.md) - Testing strategies
- [API_TYPES.md](API_TYPES.md) - OpenAPI schema and type generation
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migrating from old structure
