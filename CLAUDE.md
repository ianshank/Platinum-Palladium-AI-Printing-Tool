# CLAUDE.md - AI Assistant Guide for Platinum-Palladium AI Printing Tool

This document provides essential context for AI assistants working with this codebase.

## Project Overview

**Pt/Pd Calibration Studio** is an AI-powered calibration system for platinum/palladium alternative photographic printing. It provides tools for creating linearization curves for digital negatives, calculating chemistry recipes, analyzing step tablets, and optimizing the printing workflow.

- **Package Name**: `ptpd-calibration`
- **Version**: 1.1.0
- **Python**: 3.10+
- **UI Framework**: Gradio (web-based)
- **API Framework**: FastAPI
- **License**: MIT

## Quick Commands

```bash
# Install dependencies
pip install -e ".[all,dev,test]"

# Run tests
PYTHONPATH=src pytest tests/unit/ -v

# Run all tests with coverage
PYTHONPATH=src pytest --cov=src/ptpd_calibration --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m api           # API tests
pytest -m "not slow"    # Skip slow tests

# Lint and format
ruff check .
ruff format .

# Type checking
mypy src/ptpd_calibration --ignore-missing-imports

# Launch the Gradio UI
python app.py

# Run API server
ptpd-server
```

## Repository Structure

```
Platinum-Palladium-AI-Printing-Tool/
├── src/ptpd_calibration/        # Main package source
│   ├── core/                    # Core models and types (Pydantic)
│   ├── curves/                  # Curve generation, parsing, export
│   ├── detection/               # Step tablet detection and extraction
│   ├── chemistry/               # Chemistry calculators (Pt/Pd, Cyanotype, etc.)
│   ├── exposure/                # UV exposure calculations
│   ├── imaging/                 # Image processing, split-grade printing
│   ├── deep_learning/           # Neural networks (PyTorch)
│   ├── ml/                      # Traditional ML (scikit-learn)
│   ├── llm/                     # LLM integration (Anthropic, OpenAI)
│   ├── agents/                  # Agentic AI system
│   ├── ai/                      # Platinum/Palladium AI module
│   ├── api/                     # FastAPI endpoints
│   ├── ui/                      # Gradio UI components
│   │   └── tabs/                # Individual UI tabs
│   ├── workflow/                # Recipe management and automation
│   ├── data/                    # Database, cloud sync, version control
│   ├── calculations/            # Enhanced technical calculations
│   ├── integrations/            # Hardware integrations (spectrophotometer, etc.)
│   ├── qa/                      # Quality assurance validation
│   ├── monitoring/              # Performance monitoring
│   ├── education/               # Tutorials and glossary
│   └── config.py                # Application configuration
├── tests/                       # Test suite
│   ├── unit/                    # Fast, isolated unit tests
│   ├── integration/             # Component integration tests
│   ├── api/                     # API endpoint tests
│   ├── e2e/                     # End-to-end Selenium tests
│   ├── performance/             # Benchmarks and load tests
│   ├── visual/                  # Visual regression tests
│   ├── conftest.py              # Shared pytest fixtures
│   └── fixtures/                # Test data files
├── examples/                    # Usage examples and demos
├── scripts/                     # Utility scripts
├── docs/                        # Additional documentation
├── app.py                       # Huggingface Spaces entry point
├── pyproject.toml               # Project configuration
└── requirements.txt             # Core dependencies
```

## Coding Conventions

### Python Standards

- **Type Hints**: Required for all function arguments and return values
- **Models**: Use Pydantic v2 for all data schemas
- **Docstrings**: Google-style docstrings for modules, classes, and functions
- **Line Length**: 100 characters max
- **Error Handling**: Use explicit error handling; avoid bare `try-except`

### Example Code Pattern

```python
from pydantic import BaseModel, Field

class ChemistryRecipe(BaseModel):
    """A coating recipe for Pt/Pd printing.

    Args:
        name: Recipe identifier
        platinum_ml: Platinum solution in milliliters
        palladium_ml: Palladium solution in milliliters
    """
    name: str = Field(..., description="Recipe name")
    platinum_ml: float = Field(..., ge=0, description="Platinum amount (ml)")
    palladium_ml: float = Field(..., ge=0, description="Palladium amount (ml)")

    @property
    def metal_ratio(self) -> float:
        """Calculate Pt:Pd ratio."""
        total = self.platinum_ml + self.palladium_ml
        return self.platinum_ml / total if total > 0 else 0.5
```

### Import Organization

```python
# Standard library
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
from pydantic import BaseModel

# Local imports
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.config import get_settings
```

### Deep Learning Module Conventions

- Use lazy imports for heavy libraries (`torch`, `diffusers`) to keep startup fast
- Configuration via `pydantic-settings`
- All AI outputs validated against Pydantic models
- Models use `torch.nn.Module`

## Testing

### Test Structure

| Directory | Purpose | Marker |
|-----------|---------|--------|
| `tests/unit/` | Fast, isolated tests | `@pytest.mark.unit` |
| `tests/integration/` | Component interaction | `@pytest.mark.integration` |
| `tests/api/` | API endpoint tests | `@pytest.mark.api` |
| `tests/e2e/` | Browser-based tests | `@pytest.mark.selenium` |
| `tests/performance/` | Benchmarks | `@pytest.mark.performance` |
| `tests/visual/` | Screenshot comparison | `@pytest.mark.visual` |

### Common Fixtures

- `sample_densities` - 21-step density measurements
- `sample_calibration_record` - Sample CalibrationRecord
- `sample_step_tablet_image` - Synthetic step tablet image
- `populated_database` - Database with sample records

### Running Tests

```bash
# Always set PYTHONPATH when running tests
PYTHONPATH=src pytest tests/unit/ -v

# Run with coverage
PYTHONPATH=src pytest --cov=src/ptpd_calibration --cov-report=html

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x
```

## Key Data Models

### Core Models (src/ptpd_calibration/core/models.py)

- `CalibrationRecord` - Complete calibration session data
- `CurveData` - Linearization curve points
- `DensityMeasurement` - Single density reading
- `ExtractionResult` - Step tablet scan results
- `PaperProfile` - Paper characteristics
- `PatchData` - Individual step patch data

### Enums (src/ptpd_calibration/core/types.py)

- `ChemistryType` - CLASSIC, NA2, POTASSIUM
- `ContrastAgent` - NA2_DICHROMATE, HYDROGEN_PEROXIDE
- `CurveType` - LINEAR, SOFT, STANDARD, HARD
- `DeveloperType` - AMMONIUM_CITRATE, POTASSIUM_OXALATE
- `PaperSizing` - INTERNAL, EXTERNAL, DOUBLE

## Configuration

Settings managed via `pydantic-settings` in `src/ptpd_calibration/config.py`:

```python
from ptpd_calibration import get_settings, configure

# Get current settings
settings = get_settings()

# Configure programmatically
configure(
    llm_provider="anthropic",
    anthropic_api_key="sk-...",
)
```

### Environment Variables

- `ANTHROPIC_API_KEY` - For LLM features
- `OPENAI_API_KEY` - Alternative LLM provider
- `GRADIO_SERVER_PORT` - UI port (default: 7860)
- `PTPD_DEBUG` - Enable debug mode

## CI/CD Pipeline

GitHub Actions runs on push to `main`/`develop` and all PRs:

1. **Lint & Type Check** - Ruff linter, formatter, MyPy
2. **Unit Tests** - Matrix: Python 3.10-3.12 on Ubuntu/Windows/macOS
3. **Integration Tests** - Component interactions
4. **API Tests** - REST endpoint validation
5. **E2E Tests** - Selenium browser tests
6. **Performance Tests** - Benchmarks
7. **Visual Tests** - Screenshot regression (PRs only)
8. **Coverage Report** - 80% threshold target

## Git Conventions

### Commit Messages

Use conventional commits format:

```
feat: Add neural curve training pipeline
fix: Correct density calculation for high Dmax
docs: Update API documentation
refactor: Simplify curve generation logic
test: Add integration tests for chemistry module
```

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `claude/` - AI-assisted development branches

## Important Patterns

### Optional Dependencies

Many features have optional dependencies. Imports use try/except:

```python
try:
    from ptpd_calibration.deep_learning import NeuralCurvePredictor
except ImportError:
    NeuralCurvePredictor = None  # Torch not installed
```

### Curve Export Formats

The system exports to multiple formats:
- **QTR** (QuadTone RIP) - `.quad` files
- **Piezography** - `.txt` curves
- **CSV/JSON** - Generic data exchange

### Paper Base Reference

Density calculations require paper base reference:
```python
# Visual density = -log10(reflectance / paper_base_reflectance)
density = extractor.calculate_density(rgb_values, paper_base_rgb)
```

## Common Tasks

### Adding a New Chemistry Calculator

1. Create module in `src/ptpd_calibration/chemistry/`
2. Inherit from base calculator pattern
3. Add Pydantic models for parameters
4. Export from `chemistry/__init__.py`
5. Add unit tests in `tests/unit/test_chemistry.py`

### Adding a New UI Tab

1. Create tab module in `src/ptpd_calibration/ui/tabs/`
2. Define Gradio components in `gr.Blocks`
3. Register in `gradio_app.py`
4. Add integration test

### Extending the API

1. Add endpoint in `src/ptpd_calibration/api/`
2. Define request/response Pydantic models
3. Add to router in `server.py`
4. Write API tests in `tests/api/`

## Troubleshooting

### Import Errors

Always ensure `PYTHONPATH=src` when running from project root:
```bash
PYTHONPATH=src python -c "from ptpd_calibration import CurveGenerator"
```

### Test Failures

- Check Python version (3.10+ required)
- Install all dependencies: `pip install -e ".[all,dev,test]"`
- Run with verbose output: `pytest -v -s`

### Gradio Issues

The `app.py` includes a patch for Gradio 4.44 JSON schema issues. If you encounter client-side errors, ensure the patch is being applied.

## Resources

- [GitHub Repository](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool)
- [Huggingface Space](https://huggingface.co/spaces/ianshank/ptpd-calibration)
- Architecture: `ARCHITECTURE.md`
- Deep Learning: `DEEP_LEARNING_IMPLEMENTATION.md`
- Enhanced Calculations: `ENHANCED_CALCULATIONS_SUMMARY.md`
