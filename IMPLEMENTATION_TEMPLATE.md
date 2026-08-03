# PTPD Calibration Studio - Implementation Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: An AI-powered calibration system for platinum/palladium alternative
photographic printing with real hardware integration, deep learning UI exposure,
and multi-user collaboration capabilities.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `PYTHONPATH=src pytest tests/unit/ -v`
- [ ] All integration tests pass: `PYTHONPATH=src pytest tests/integration/ -v`
- [ ] Type checking passes: `mypy src/ptpd_calibration --ignore-missing-imports`
- [ ] Linting passes: `ruff check . && ruff format --check .`
- [ ] Test coverage >= 80% on core modules
- [ ] Hardware integration supports real X-Rite spectrophotometer communication
- [ ] Deep learning features accessible via Gradio UI
- [ ] Session statistics calculated and displayed
- [ ] API latency < 500ms for standard operations
- [ ] System handles 10+ concurrent users without degradation
```

### 1.3 Problem Description (The "Three Paragraphs")

```
PROBLEM DOMAIN:
The system provides calibration tools for alternative photographic printing processes,
primarily platinum/palladium. It must analyze step tablet images, extract density
measurements, generate linearization curves, and export to industry formats (QTR,
Piezography). The core coordination logic involves: image capture → analysis →
curve generation → export, with ML/AI assistance at each stage.

DATA FLOWS:
Data flows from image input (scan/photo) → detection pipeline (OpenCV) → density
extraction → curve fitting (scipy/ML) → export generation. State maintained includes:
calibration records (SQLite), user recipes, curve presets, and ML model weights.
Real-time synchronization needed for multi-user collaboration and hardware device
status (spectrophotometer readings, printer status).

FAILURE MODES:
Hardware failures: Device disconnection during measurement (graceful degradation to
manual input). Network failures: Cloud sync interruption (local-first with eventual
sync). Data corruption: Invalid density readings (outlier rejection + user confirmation).
Invariants: Curves must be monotonic, densities must be in valid range [0.0, 4.0],
all user data must persist across sessions.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.10+ (support through 3.13)
- Required Dependencies:
  - pydantic>=2.5.0 (all data models)
  - pydantic-settings>=2.1.0 (configuration)
  - numpy>=1.24.0 (numerical operations)
  - scipy>=1.11.0 (curve fitting)
  - pillow>=10.0.0 (image handling)
- Security:
  - No hardcoded secrets (use environment variables with PTPD_ prefix)
  - All user inputs validated via Pydantic models
  - File paths sanitized before use
  - API endpoints require authentication in production
- Compatibility:
  - Must run on Linux, macOS, Windows
  - Must support ARM64 and x86_64 architectures
  - SQLite for local storage (PostgreSQL optional for multi-user)
```

### 2.2 Soft Constraints (Preferences)

```
- Style:
  - Google-style docstrings for all public functions
  - 100 character line limit
  - Type hints required for all function arguments and return values
- Architecture:
  - Prefer composition over inheritance
  - Use Pydantic models for all data transfer
  - Lazy imports for heavy dependencies (torch, diffusers)
  - Configuration via pydantic-settings with env prefix
- Performance:
  - Prefer async I/O for network operations
  - Cache expensive computations (curves, ML predictions)
  - Minimize memory allocations in hot paths
- Testing:
  - pytest with markers for test categorization
  - >80% coverage on core logic (curves, detection, chemistry)
  - Integration tests for hardware simulation
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization if it improves clarity
- Add PyPI dependencies not explicitly listed (document in pyproject.toml)
- Refactor adjacent code for consistency with new patterns
- Choose implementation patterns not specified (document decisions)
- Create new modules within src/ptpd_calibration/
- Modify test fixtures and add new test utilities
- Update CLAUDE.md with new build commands or architecture decisions
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- All files in /src/ptpd_calibration/
- All files in /tests/
- Configuration files: pyproject.toml, .env.example
- Documentation: CLAUDE.md, IMPLEMENTATION_TEMPLATE.md
- Entry points: app.py

OUT OF SCOPE:
- /vendor or /third_party directories (if they exist)
- .github/workflows/ (CI/CD - requires review)
- Files with # DO NOT MODIFY header comment
- External API credentials (use environment variables)

REQUIRES CONFIRMATION:
- Changes to pyproject.toml dependencies (document rationale)
- New UI tabs in Gradio (affects user experience)
- Database schema changes (migration required)
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within /src and /tests
- Dependency installation for development
- Running tests and linting
- Refactoring for consistency within modules
- Adding logging and debugging utilities
- Creating fixtures and test utilities
- Bug fixes that don't change public API

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 modules
- Breaking API changes (function signature changes)
- Deletions of >100 lines of production code
- New external service integrations
- Database schema modifications
- Changes to CI/CD configuration

PROHIBITED (do not attempt):
- Force push to any branch
- Commits to main/master branch directly
- Modifications to .git/ directory
- External API calls with side effects (payments, emails)
- Storing secrets in code or configuration files
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Time-boxed exploration: ≤10 min on research before asking
- Max test execution time: 10 minutes for full suite
- Max single file size: 2000 lines (split if larger)
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:
1. Lint: `ruff check . --fix && ruff format .`
2. Type check: `mypy src/ptpd_calibration --ignore-missing-imports`
3. Unit tests: `PYTHONPATH=src pytest tests/unit/ -v -x`
4. Integration tests: `PYTHONPATH=src pytest tests/integration/ -v`
5. Coverage: `PYTHONPATH=src pytest --cov=src/ptpd_calibration --cov-report=term-missing`
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
  → Run `ruff check . --fix` to auto-fix
  → If unfixable, analyze error and fix manually
  → Re-run lint check

ON TYPE ERROR:
  → Read mypy error message carefully
  → Check if type annotation is correct
  → Add type: ignore comment ONLY if external library issue
  → Re-run type check

ON TEST FAILURE:
  → Read failure output completely
  → Identify root cause (implementation bug vs test bug)
  → Fix implementation (not test, unless test is clearly wrong)
  → Re-run failed test specifically: `pytest path/to/test.py::test_name -v`
  → Re-run full test suite

ON IMPORT ERROR:
  → Check if dependency is installed: `pip show package_name`
  → Check if PYTHONPATH is set correctly
  → Verify lazy import pattern for optional dependencies

ON REPEATED FAILURE (same error 3x):
  → Stop and document analysis
  → List attempted fixes
  → Request human guidance with specific question
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass (lint, type, tests)
2. Manual smoke test: `python -c "from ptpd_calibration import get_settings; print(get_settings())"`
3. If UI changes: `python app.py` and verify in browser
4. Generate brief summary of changes made
5. Update CLAUDE.md if new commands or patterns introduced
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md)

```markdown
# Maintain/update the following in CLAUDE.md:

## Build Commands
- `pip install -e ".[all,dev,test]"`: Install all dependencies
- `PYTHONPATH=src pytest tests/unit/ -v`: Run unit tests
- `ruff check . && ruff format .`: Lint and format

## Architecture Decisions
- [date]: [decision]: [rationale]

## Known Issues
- Hardware integration is simulated by default (set spectro_simulate=False for real)
- Gradio 4.44 requires JSON schema patch (see app.py)
```

### 5.2 Information to Preserve Across Sessions

```
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- Hardware-specific configuration requirements
```

### 5.3 Information That Can Be Re-derived

```
- File structure (can be scanned with Glob)
- Dependency versions (in pyproject.toml)
- Current test status (re-run tests)
- Configuration schema (read config.py)
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```
1. Read CLAUDE.md for project context
2. Scan project structure: explore src/ptpd_calibration/ directory
3. Read config.py to understand configuration system
4. Run existing tests to establish baseline: PYTHONPATH=src pytest tests/unit/ -v
5. Verify working environment: python -c "from ptpd_calibration import get_settings"
```

### 6.2 Implementation Order

```
1. Understand existing patterns (read before write)
2. Implement core logic (smallest working version)
3. Add type hints and validation
4. Add error handling with logging
5. Write tests (unit first, then integration)
6. Run verification loop
7. Refactor if needed
8. Update documentation (CLAUDE.md, docstrings)
```

### 6.3 Completion Checklist

```
□ All success criteria met
□ All verification commands pass
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided
□ Known limitations documented
□ Breaking changes noted (if any)
```

---

## SECTION 7: C4 ARCHITECTURE

### 7.1 Context Diagram (Level 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM CONTEXT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────┐                                    ┌─────────────────┐   │
│    │             │                                    │                 │   │
│    │  Printmaker │───── Uses ─────┐                   │   LLM Provider  │   │
│    │   (User)    │                │                   │  (Anthropic/    │   │
│    │             │                │                   │   OpenAI)       │   │
│    └─────────────┘                │                   └────────┬────────┘   │
│                                   │                            │            │
│                                   ▼                            │ API        │
│                    ┌──────────────────────────────┐            │            │
│                    │                              │◄───────────┘            │
│                    │   PTPD Calibration Studio    │                         │
│                    │                              │                         │
│                    │  AI-powered calibration      │                         │
│                    │  system for Pt/Pd printing   │                         │
│                    │                              │                         │
│                    └──────────────┬───────────────┘                         │
│                                   │                                         │
│                    ┌──────────────┼───────────────┐                         │
│                    │              │               │                         │
│                    ▼              ▼               ▼                         │
│             ┌───────────┐  ┌───────────┐  ┌─────────────┐                  │
│             │           │  │           │  │             │                  │
│             │ Spectro-  │  │  Printer  │  │  Weather    │                  │
│             │ photometer│  │  (Epson)  │  │  API        │                  │
│             │ (X-Rite)  │  │           │  │             │                  │
│             └───────────┘  └───────────┘  └─────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Container Diagram (Level 2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTAINER DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PTPD Calibration Studio                          │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │    │
│  │  │                 │    │                 │    │                 │  │    │
│  │  │   Gradio UI     │◄──►│   FastAPI       │◄──►│   Core Engine   │  │    │
│  │  │   (Web App)     │    │   (REST API)    │    │   (Python)      │  │    │
│  │  │                 │    │                 │    │                 │  │    │
│  │  │  Port: 7860     │    │  Port: 8000     │    │  - Curves       │  │    │
│  │  │  gradio>=4.0    │    │  fastapi>=0.109 │    │  - Detection    │  │    │
│  │  │                 │    │                 │    │  - Chemistry    │  │    │
│  │  └────────┬────────┘    └────────┬────────┘    │  - Imaging      │  │    │
│  │           │                      │             │                 │  │    │
│  │           │                      │             └────────┬────────┘  │    │
│  │           │                      │                      │           │    │
│  │           │      ┌───────────────┼──────────────────────┤           │    │
│  │           │      │               │                      │           │    │
│  │           ▼      ▼               ▼                      ▼           │    │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │    │
│  │  │                 │    │                 │    │                 │  │    │
│  │  │   SQLite DB     │    │   ML Pipeline   │    │ Deep Learning   │  │    │
│  │  │   (Storage)     │    │  (scikit-learn) │    │   (PyTorch)     │  │    │
│  │  │                 │    │                 │    │                 │  │    │
│  │  │  - Recipes      │    │  - Prediction   │    │  - Neural Curve │  │    │
│  │  │  - Calibrations │    │  - Clustering   │    │  - Diffusion    │  │    │
│  │  │  - History      │    │  - Active Learn │    │  - Quality      │  │    │
│  │  │                 │    │                 │    │                 │  │    │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  External Integrations:                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Spectro      │  │ Printer      │  │ Weather      │  │ HuggingFace  │     │
│  │ (USB/Serial) │  │ (CUPS/IPP)   │  │ (HTTP)       │  │ (Cloud)      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Component Diagram (Level 3)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT DIAGRAM - Core Engine                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          CORE ENGINE                                 │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │ core/         │  │ curves/       │  │ detection/    │            │    │
│  │  │               │  │               │  │               │            │    │
│  │  │ • models.py   │──│ • generator.py│──│ • extractor.py│            │    │
│  │  │ • types.py    │  │ • parser.py   │  │ • analyzer.py │            │    │
│  │  │               │  │ • export.py   │  │               │            │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │    │
│  │          │                  │                  │                     │    │
│  │          │    ┌─────────────┴─────────────┐    │                     │    │
│  │          │    │                           │    │                     │    │
│  │          ▼    ▼                           ▼    ▼                     │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │ chemistry/    │  │ imaging/      │  │ exposure/     │            │    │
│  │  │               │  │               │  │               │            │    │
│  │  │ • ptpd.py     │  │ • processor.py│  │ • calculator. │            │    │
│  │  │ • cyanotype.py│  │ • split_grade │  │ • uv_monitor. │            │    │
│  │  │ • silver.py   │  │ • negative.py │  │               │            │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                    INTEGRATIONS LAYER                          │  │    │
│  │  │                                                                │  │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │    │
│  │  │  │ spectro     │  │ printer     │  │ weather     │            │  │    │
│  │  │  │ photometer  │  │ drivers     │  │ api         │            │  │    │
│  │  │  │             │  │             │  │             │            │  │    │
│  │  │  │ USB/Serial  │  │ CUPS/IPP    │  │ HTTP REST   │            │  │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │    │
│  │  │                                                                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Code Diagram (Level 4) - Key Classes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CODE DIAGRAM - Key Classes                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CORE MODELS (Pydantic)                                                      │
│  ═══════════════════════                                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CalibrationRecord                                                   │    │
│  │  ├── paper_type: str                                                 │    │
│  │  ├── exposure_time: float                                            │    │
│  │  ├── metal_ratio: float                                              │    │
│  │  ├── chemistry_type: ChemistryType                                   │    │
│  │  ├── contrast_agent: ContrastAgent                                   │    │
│  │  ├── measured_densities: list[float]                                 │    │
│  │  └── @property dmax, dmin, density_range                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CurveData                                                           │    │
│  │  ├── name: str                                                       │    │
│  │  ├── input_values: list[float]                                       │    │
│  │  ├── output_values: list[float]                                      │    │
│  │  ├── metadata: dict[str, Any]                                        │    │
│  │  └── methods: interpolate(), export(), validate()                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  CONFIGURATION (pydantic-settings)                                           │
│  ══════════════════════════════════                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Settings (BaseSettings)                                             │    │
│  │  ├── detection: DetectionSettings                                    │    │
│  │  ├── extraction: ExtractionSettings                                  │    │
│  │  ├── curves: CurveSettings                                           │    │
│  │  ├── ml: MLSettings                                                  │    │
│  │  ├── deep_learning: DeepLearningSettings                             │    │
│  │  ├── integrations: IntegrationSettings                               │    │
│  │  └── ... (20+ subsettings)                                           │    │
│  │                                                                      │    │
│  │  Environment Prefix: PTPD_                                           │    │
│  │  Example: PTPD_DEBUG=true, PTPD_DETECTION_CANNY_LOW_THRESHOLD=50     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  INTEGRATIONS (Protocol-based)                                               │
│  ═════════════════════════════                                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SpectrophotometerProtocol (Protocol)                                │    │
│  │  ├── connect() -> bool                                               │    │
│  │  ├── disconnect() -> None                                            │    │
│  │  ├── read_density() -> DensityMeasurement                            │    │
│  │  ├── calibrate_white() -> bool                                       │    │
│  │  └── get_device_info() -> DeviceInfo                                 │    │
│  │                                                                      │    │
│  │  Implementations:                                                    │    │
│  │  ├── SimulatedSpectrophotometer (testing)                            │    │
│  │  ├── XRiteI1Pro (real hardware - TODO)                               │    │
│  │  └── ColorMunki (real hardware - TODO)                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 8: IMPLEMENTATION PHASES

### Phase 1: Hardware Integration (Priority: Critical)

**Objective:** Enable real spectrophotometer and printer communication.

**Files to Modify:**
- `src/ptpd_calibration/integrations/spectrophotometer.py`
- `src/ptpd_calibration/integrations/printer_drivers.py`
- `tests/integration/test_hardware.py` (new)

**Implementation Steps:**
1. Define abstract protocol for spectrophotometer devices
2. Implement USB/serial communication layer
3. Add X-Rite i1 Pro driver (vendor SDK integration)
4. Add printer driver for CUPS/IPP
5. Create simulation mode fallback
6. Write integration tests with mocked hardware

**Success Criteria:**
- `pytest tests/integration/test_hardware.py -v` passes
- Real device communication works when `spectro_simulate=False`

---

### Phase 2: Deep Learning UI Integration (Priority: High)

**Objective:** Expose neural curve prediction and image quality in Gradio UI.

**Files to Modify:**
- `src/ptpd_calibration/ui/tabs/` (new tabs)
- `src/ptpd_calibration/ui/gradio_app.py`
- `tests/integration/test_ui_deep_learning.py` (new)

**Implementation Steps:**
1. Create `neural_curve_tab.py` for curve prediction UI
2. Create `image_quality_tab.py` for quality assessment
3. Add lazy loading for PyTorch models
4. Implement progress indicators for long operations
5. Add error handling for missing dependencies

**Success Criteria:**
- Neural curve tab functional when PyTorch installed
- Graceful degradation when PyTorch not available
- UI tests pass

---

### Phase 3: Session Statistics (Priority: Medium)

**Objective:** Calculate and display session statistics and best practices.

**Files to Modify:**
- `src/ptpd_calibration/workflow/session_log.py` (fix TODO at line 85)
- `src/ptpd_calibration/ui/tabs/dashboard_tab.py` (new)
- `tests/unit/test_session_stats.py` (new)

**Implementation Steps:**
1. Implement `calculate_stats()` method in SessionLog
2. Add statistics aggregation (prints per session, success rate, etc.)
3. Generate best practice recommendations
4. Create dashboard UI tab
5. Write unit tests for statistics calculations

**Success Criteria:**
- Session statistics calculated correctly
- Dashboard displays metrics
- Best practices generated from data

---

### Phase 4: Multi-User Support (Priority: Medium)

**Objective:** Enable concurrent users with PostgreSQL backend.

**Files to Modify:**
- `src/ptpd_calibration/data/database.py`
- `src/ptpd_calibration/data/cloud_sync.py`
- `src/ptpd_calibration/config.py` (add PostgreSQL settings)
- `tests/integration/test_multiuser.py` (new)

**Implementation Steps:**
1. Add SQLAlchemy with PostgreSQL support
2. Implement connection pooling
3. Add user authentication/session management
4. Implement optimistic locking for concurrent edits
5. Add real-time sync via WebSockets

**Success Criteria:**
- PostgreSQL connection works
- Concurrent writes don't corrupt data
- User sessions isolated

---

## SECTION 9: REUSABLE COMPONENT PATTERNS

### 9.1 Base Service Pattern

```python
"""
Reusable base class for all service components.
Location: src/ptpd_calibration/core/base_service.py
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.config import get_settings

T = TypeVar("T", bound=BaseModel)

class BaseService(ABC, Generic[T]):
    """Base class for all services with common functionality."""

    def __init__(self, config_key: str | None = None):
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self._config = (
            getattr(self.settings, config_key)
            if config_key
            else self.settings
        )

    @abstractmethod
    def validate_input(self, data: T) -> tuple[bool, list[str]]:
        """Validate input data, return (is_valid, errors)."""
        ...

    @abstractmethod
    def process(self, data: T) -> T:
        """Process the input data and return result."""
        ...

    def execute(self, data: T) -> T:
        """Execute with validation and logging."""
        self.logger.debug(f"Processing {type(data).__name__}")

        is_valid, errors = self.validate_input(data)
        if not is_valid:
            self.logger.error(f"Validation failed: {errors}")
            raise ValueError(f"Validation failed: {errors}")

        try:
            result = self.process(data)
            self.logger.info(f"Successfully processed {type(data).__name__}")
            return result
        except Exception as e:
            self.logger.exception(f"Processing failed: {e}")
            raise
```

### 9.2 Hardware Protocol Pattern

```python
"""
Protocol for hardware device integration.
Location: src/ptpd_calibration/integrations/protocols.py
"""
from typing import Protocol, TypeVar, Generic
from pydantic import BaseModel
from enum import Enum

class DeviceStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CALIBRATING = "calibrating"

class DeviceInfo(BaseModel):
    """Device information model."""
    vendor: str
    model: str
    serial_number: str | None = None
    firmware_version: str | None = None

T = TypeVar("T", bound=BaseModel)

class HardwareDevice(Protocol[T]):
    """Protocol for hardware device communication."""

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        ...

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information (None if not connected)."""
        ...

    def connect(self, timeout_seconds: float = 5.0) -> bool:
        """Connect to device. Returns True on success."""
        ...

    def disconnect(self) -> None:
        """Disconnect from device."""
        ...

    def read(self) -> T:
        """Read measurement from device."""
        ...

    def calibrate(self) -> bool:
        """Run device calibration. Returns True on success."""
        ...
```

### 9.3 Repository Pattern

```python
"""
Generic repository pattern for data access.
Location: src/ptpd_calibration/data/repository.py
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
ID = TypeVar("ID")

class Repository(ABC, Generic[T, ID]):
    """Abstract base repository for CRUD operations."""

    @abstractmethod
    def get(self, id: ID) -> T | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    def get_all(self, limit: int = 100, offset: int = 0) -> Sequence[T]:
        """Get all entities with pagination."""
        ...

    @abstractmethod
    def add(self, entity: T) -> T:
        """Add new entity, return with ID assigned."""
        ...

    @abstractmethod
    def update(self, id: ID, entity: T) -> T | None:
        """Update existing entity. Returns None if not found."""
        ...

    @abstractmethod
    def delete(self, id: ID) -> bool:
        """Delete entity. Returns True if deleted."""
        ...

    @abstractmethod
    def find(self, **criteria) -> Sequence[T]:
        """Find entities matching criteria."""
        ...
```

### 9.4 Event Bus Pattern

```python
"""
Event-driven communication between components.
Location: src/ptpd_calibration/core/events.py
"""
from typing import Callable, TypeVar, Generic
from collections import defaultdict
from pydantic import BaseModel
import asyncio

from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

class Event(BaseModel):
    """Base event class."""
    event_type: str
    timestamp: float | None = None

class EventBus:
    """Publish-subscribe event bus."""

    _instance: "EventBus | None" = None

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = defaultdict(list)
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to events of a given type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Publish event to all subscribers."""
        logger.debug(f"Publishing {event.event_type}")
        for handler in self._subscribers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Handler {handler.__name__} failed: {e}")

    async def publish_async(self, event: Event) -> None:
        """Async publish with concurrent handler execution."""
        logger.debug(f"Publishing async {event.event_type}")
        tasks = [
            asyncio.create_task(self._call_handler(handler, event))
            for handler in self._subscribers[event.event_type]
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_handler(
        self, handler: Callable[[Event], None], event: Event
    ) -> None:
        """Call handler, wrapping sync handlers for async context."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.exception(f"Handler {handler.__name__} failed: {e}")
```

---

## SECTION 10: LOGGING AND DEBUGGING FRAMEWORK

### 10.1 Structured Logging Setup

```python
"""
Centralized logging configuration.
Location: src/ptpd_calibration/core/logging.py
"""
import logging
import sys
from typing import Any
from pathlib import Path
from datetime import datetime
import json

from ptpd_calibration.config import get_settings

# Lazy import for structlog (optional dependency)
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
    json_format: bool = False,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config.
        log_file: Optional file path for log output.
        json_format: Use JSON formatting for structured logs.
    """
    settings = get_settings()
    level = level or settings.log_level

    # Root logger configuration
    root_logger = logging.getLogger("ptpd_calibration")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    root_logger.info(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"ptpd_calibration.{name}")


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self._old_factory = None

    def __enter__(self) -> logging.Logger:
        self._old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra = self.context
            return record

        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(self, *args) -> None:
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
```

### 10.2 Debug Utilities

```python
"""
Debugging utilities for development.
Location: src/ptpd_calibration/core/debug.py
"""
import functools
import time
from typing import Callable, TypeVar, ParamSpec, Any
from contextlib import contextmanager

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.config import get_settings

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def timer(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to measure and log function execution time."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        settings = get_settings()
        if not settings.debug:
            return func(*args, **kwargs)

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(
                f"{func.__module__}.{func.__name__} "
                f"completed in {elapsed:.4f}s"
            )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.debug(
                f"{func.__module__}.{func.__name__} "
                f"failed after {elapsed:.4f}s: {e}"
            )
            raise

    return wrapper


def trace(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log function entry/exit with arguments."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        settings = get_settings()
        if not settings.debug:
            return func(*args, **kwargs)

        func_name = f"{func.__module__}.{func.__name__}"

        # Truncate long arguments for readability
        def truncate(val: Any, max_len: int = 100) -> str:
            s = repr(val)
            return s if len(s) <= max_len else s[:max_len] + "..."

        args_repr = [truncate(a) for a in args]
        kwargs_repr = [f"{k}={truncate(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        logger.debug(f"ENTER {func_name}({signature})")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"EXIT  {func_name} -> {truncate(result)}")
            return result
        except Exception as e:
            logger.debug(f"RAISE {func_name} -> {type(e).__name__}: {e}")
            raise

    return wrapper


@contextmanager
def debug_context(name: str, **extra: Any):
    """Context manager for debug logging with timing."""
    settings = get_settings()
    if not settings.debug:
        yield
        return

    extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
    logger.debug(f"START {name}" + (f" ({extra_str})" if extra_str else ""))

    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.debug(f"END   {name} ({elapsed:.4f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.debug(f"FAIL  {name} ({elapsed:.4f}s): {e}")
        raise


class DebugMixin:
    """Mixin class to add debug logging to any class."""

    @property
    def _debug_enabled(self) -> bool:
        return get_settings().debug

    def _debug(self, message: str, **extra: Any) -> None:
        if self._debug_enabled:
            logger.debug(f"[{self.__class__.__name__}] {message}", extra=extra)

    def _debug_enter(self, method: str, **params: Any) -> float:
        if self._debug_enabled:
            params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            logger.debug(f"[{self.__class__.__name__}.{method}] ENTER ({params_str})")
        return time.perf_counter()

    def _debug_exit(self, method: str, start_time: float, result: Any = None) -> None:
        if self._debug_enabled:
            elapsed = time.perf_counter() - start_time
            result_str = f" -> {result!r}" if result is not None else ""
            logger.debug(
                f"[{self.__class__.__name__}.{method}] "
                f"EXIT ({elapsed:.4f}s){result_str}"
            )
```

---

## SECTION 11: COMPREHENSIVE TEST SUITE

### 11.1 Test Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── fixtures/                      # Test data files
│   ├── step_tablets/
│   ├── curves/
│   └── images/
├── unit/                          # Fast, isolated tests
│   ├── test_core_models.py
│   ├── test_curve_generator.py
│   ├── test_chemistry.py
│   ├── test_detection.py
│   └── ...
├── integration/                   # Component interaction tests
│   ├── test_calibration_workflow.py
│   ├── test_hardware_integration.py
│   ├── test_database_operations.py
│   └── ...
├── api/                           # REST API tests
│   ├── test_calibration_endpoints.py
│   ├── test_curve_endpoints.py
│   └── test_authentication.py
├── e2e/                           # End-to-end tests
│   ├── test_user_journeys.py
│   └── test_complete_workflow.py
├── performance/                   # Benchmarks
│   ├── test_curve_generation_perf.py
│   ├── test_image_processing_perf.py
│   └── locustfile.py
└── visual/                        # Visual regression
    ├── test_curve_plots.py
    └── baselines/
```

### 11.2 Enhanced Test Fixtures

```python
"""
Enhanced test fixtures with factories.
Location: tests/conftest.py (additions)
"""
import pytest
from typing import Generator, Any
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import numpy as np

from ptpd_calibration.config import Settings, configure
from ptpd_calibration.core.models import CalibrationRecord, CurveData
from ptpd_calibration.integrations.protocols import DeviceStatus


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_settings() -> Generator[Settings, None, None]:
    """Provide isolated test settings."""
    settings = Settings(
        debug=True,
        log_level="DEBUG",
        data_dir=Path("/tmp/ptpd_test"),
    )
    configure(settings)
    yield settings
    # Reset to default after test
    configure()


@pytest.fixture
def mock_settings(mocker) -> MagicMock:
    """Mock settings for unit tests."""
    mock = MagicMock(spec=Settings)
    mock.debug = True
    mock.log_level = "DEBUG"
    mocker.patch("ptpd_calibration.config.get_settings", return_value=mock)
    return mock


# =============================================================================
# Hardware Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_spectrophotometer(mocker) -> MagicMock:
    """Mock spectrophotometer for hardware tests."""
    mock = MagicMock()
    mock.status = DeviceStatus.CONNECTED
    mock.device_info.vendor = "X-Rite"
    mock.device_info.model = "i1Pro2"
    mock.connect.return_value = True
    mock.read.return_value = MagicMock(
        density=1.5,
        lab_l=50.0,
        lab_a=0.0,
        lab_b=0.0,
    )
    return mock


@pytest.fixture
def mock_printer(mocker) -> MagicMock:
    """Mock printer for hardware tests."""
    mock = MagicMock()
    mock.status = DeviceStatus.CONNECTED
    mock.print_image = AsyncMock(return_value=True)
    return mock


# =============================================================================
# Factory Fixtures
# =============================================================================

@pytest.fixture
def calibration_record_factory():
    """Factory for creating CalibrationRecord instances."""
    def _factory(**overrides) -> CalibrationRecord:
        defaults = {
            "paper_type": "Arches Platine",
            "paper_weight": 310,
            "exposure_time": 180.0,
            "metal_ratio": 0.5,
            "chemistry_type": "PLATINUM_PALLADIUM",
            "contrast_agent": "NA2",
            "contrast_amount": 5.0,
            "developer": "POTASSIUM_OXALATE",
            "humidity": 50.0,
            "temperature": 21.0,
            "measured_densities": [0.1 + i * 0.1 for i in range(21)],
        }
        defaults.update(overrides)
        return CalibrationRecord(**defaults)
    return _factory


@pytest.fixture
def curve_data_factory():
    """Factory for creating CurveData instances."""
    def _factory(
        name: str = "Test Curve",
        points: int = 256,
        gamma: float = 1.0,
        **metadata: Any,
    ) -> CurveData:
        input_values = list(np.linspace(0, 1, points))
        output_values = [x ** gamma for x in input_values]
        return CurveData(
            name=name,
            input_values=input_values,
            output_values=output_values,
            metadata=metadata,
        )
    return _factory


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def event_loop_policy():
    """Configure event loop for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def async_client():
    """Async HTTP client for API testing."""
    from httpx import AsyncClient
    from ptpd_calibration.api.server import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def temp_database(tmp_path) -> Generator[Path, None, None]:
    """Temporary database for testing."""
    db_path = tmp_path / "test.db"
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def populated_test_database(temp_database, calibration_record_factory):
    """Database populated with test data."""
    from ptpd_calibration.data.database import CalibrationDatabase

    db = CalibrationDatabase(temp_database)

    # Add varied test records
    papers = ["Arches Platine", "Bergger COT320", "Hahnemühle Platinum Rag"]
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for paper in papers:
        for ratio in ratios:
            record = calibration_record_factory(
                paper_type=paper,
                metal_ratio=ratio,
            )
            db.add_record(record)

    return db
```

### 11.3 Test Utilities

```python
"""
Test utility functions and helpers.
Location: tests/utils.py
"""
import numpy as np
from typing import Callable, TypeVar
from contextlib import contextmanager
import time

T = TypeVar("T")


def assert_curves_equal(
    curve1: list[float],
    curve2: list[float],
    tolerance: float = 0.001,
) -> None:
    """Assert two curves are equal within tolerance."""
    assert len(curve1) == len(curve2), (
        f"Curve lengths differ: {len(curve1)} vs {len(curve2)}"
    )

    for i, (v1, v2) in enumerate(zip(curve1, curve2)):
        assert abs(v1 - v2) < tolerance, (
            f"Curves differ at index {i}: {v1} vs {v2} (tolerance: {tolerance})"
        )


def assert_monotonic(values: list[float], increasing: bool = True) -> None:
    """Assert values are monotonically increasing or decreasing."""
    for i in range(1, len(values)):
        if increasing:
            assert values[i] >= values[i-1], (
                f"Not monotonically increasing at index {i}: "
                f"{values[i-1]} -> {values[i]}"
            )
        else:
            assert values[i] <= values[i-1], (
                f"Not monotonically decreasing at index {i}: "
                f"{values[i-1]} -> {values[i]}"
            )


def assert_in_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> None:
    """Assert value is within specified range."""
    assert min_val <= value <= max_val, (
        f"{name} {value} not in range [{min_val}, {max_val}]"
    )


@contextmanager
def assert_performance(
    max_seconds: float,
    operation_name: str = "operation",
):
    """Assert operation completes within time limit."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed < max_seconds, (
        f"{operation_name} took {elapsed:.3f}s, "
        f"expected < {max_seconds}s"
    )


def generate_synthetic_densities(
    num_steps: int = 21,
    dmin: float = 0.1,
    dmax: float = 2.2,
    gamma: float = 0.85,
    noise_std: float = 0.0,
) -> list[float]:
    """Generate synthetic density measurements for testing."""
    steps = np.linspace(0, 1, num_steps)
    densities = dmin + (dmax - dmin) * (steps ** gamma)

    if noise_std > 0:
        densities += np.random.normal(0, noise_std, num_steps)
        densities = np.clip(densities, 0, 4.0)  # Physical limits

    return list(densities)


def generate_synthetic_step_tablet(
    width: int = 420,
    height: int = 100,
    num_patches: int = 21,
    dmin_value: int = 250,
    dmax_value: int = 20,
) -> np.ndarray:
    """Generate synthetic step tablet image for testing."""
    patch_width = width // num_patches
    img = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_patches):
        # Linear interpolation between dmin and dmax values
        t = i / (num_patches - 1)
        value = int(dmin_value + t * (dmax_value - dmin_value))

        x_start = i * patch_width
        x_end = (i + 1) * patch_width if i < num_patches - 1 else width
        img[:, x_start:x_end] = value

    return img
```

### 11.4 Example Test Patterns

```python
"""
Example test patterns for unit tests.
Location: tests/unit/test_example_patterns.py
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.generator import CurveGenerator


class TestCurveGenerator:
    """Tests for CurveGenerator class."""

    @pytest.fixture
    def generator(self, test_settings) -> CurveGenerator:
        """Create generator instance for tests."""
        return CurveGenerator()

    @pytest.fixture
    def sample_densities(self) -> list[float]:
        """Sample density measurements."""
        return [0.1 + i * 0.1 for i in range(21)]

    # --- Parameterized Tests ---

    @pytest.mark.unit
    @pytest.mark.parametrize("interpolation", ["linear", "cubic", "monotonic"])
    def test_generate_with_different_interpolations(
        self,
        generator: CurveGenerator,
        sample_densities: list[float],
        interpolation: str,
    ):
        """Test curve generation with different interpolation methods."""
        curve = generator.generate(
            densities=sample_densities,
            interpolation=interpolation,
        )

        assert isinstance(curve, CurveData)
        assert len(curve.input_values) == 256
        assert len(curve.output_values) == 256

    @pytest.mark.unit
    @pytest.mark.parametrize("num_points,expected_len", [
        (16, 16),
        (256, 256),
        (1024, 1024),
    ])
    def test_generate_with_different_point_counts(
        self,
        generator: CurveGenerator,
        sample_densities: list[float],
        num_points: int,
        expected_len: int,
    ):
        """Test curve generation with different output point counts."""
        curve = generator.generate(
            densities=sample_densities,
            num_points=num_points,
        )

        assert len(curve.output_values) == expected_len

    # --- Edge Case Tests ---

    @pytest.mark.unit
    def test_generate_with_minimum_densities(self, generator: CurveGenerator):
        """Test with minimum valid number of density points."""
        densities = [0.1, 1.0, 2.0]  # Minimum 3 points
        curve = generator.generate(densities=densities)

        assert curve is not None
        assert len(curve.output_values) > 0

    @pytest.mark.unit
    def test_generate_with_flat_densities(self, generator: CurveGenerator):
        """Test behavior with flat (constant) density values."""
        densities = [1.0] * 21  # All same value
        curve = generator.generate(densities=densities)

        # Should still produce valid curve
        assert curve is not None
        # Output should be relatively flat
        assert max(curve.output_values) - min(curve.output_values) < 0.1

    # --- Error Handling Tests ---

    @pytest.mark.unit
    def test_generate_with_invalid_densities_raises(
        self, generator: CurveGenerator
    ):
        """Test that invalid densities raise appropriate error."""
        with pytest.raises(ValueError, match="at least 3"):
            generator.generate(densities=[0.1, 0.5])  # Too few points

    @pytest.mark.unit
    def test_generate_with_negative_densities_raises(
        self, generator: CurveGenerator
    ):
        """Test that negative densities raise error."""
        with pytest.raises(ValueError, match="negative"):
            generator.generate(densities=[-0.1, 0.5, 1.0])

    # --- Property-Based Tests ---

    @pytest.mark.unit
    def test_output_is_monotonic(
        self,
        generator: CurveGenerator,
        sample_densities: list[float],
    ):
        """Generated curve should be monotonically increasing."""
        curve = generator.generate(
            densities=sample_densities,
            enforce_monotonicity=True,
        )

        for i in range(1, len(curve.output_values)):
            assert curve.output_values[i] >= curve.output_values[i-1], (
                f"Monotonicity violated at index {i}"
            )

    @pytest.mark.unit
    def test_output_range_is_normalized(
        self,
        generator: CurveGenerator,
        sample_densities: list[float],
    ):
        """Output values should be in [0, 1] range."""
        curve = generator.generate(densities=sample_densities)

        assert min(curve.output_values) >= 0.0
        assert max(curve.output_values) <= 1.0


class TestCurveGeneratorWithMocks:
    """Tests using mocks for isolation."""

    @pytest.mark.unit
    def test_uses_configured_interpolation(self, mock_settings):
        """Test that generator uses interpolation from settings."""
        mock_settings.curves.default_interpolation = "cubic"

        with patch("ptpd_calibration.curves.generator.scipy_interpolate") as mock_interp:
            mock_interp.return_value = np.linspace(0, 1, 256)

            generator = CurveGenerator()
            generator.generate(densities=[0.1, 1.0, 2.0])

            # Verify interpolation was called with correct method
            mock_interp.assert_called()
            call_args = mock_interp.call_args
            assert "cubic" in str(call_args)
```

---

## SECTION 12: AGENT-SPECIFIC INSTRUCTIONS

### 12.1 For Explore Agent

```
When exploring this codebase:
1. Start with src/ptpd_calibration/ directory structure
2. Focus on understanding module dependencies via imports
3. Check tests/ for usage examples of each module
4. Note any TODO/FIXME comments (only 1 exists at session_log.py:85)
5. Identify patterns by reading config.py and core/models.py first
```

### 12.2 For Plan Agent

```
When planning implementations:
1. Reference the C4 diagrams for architectural context
2. Check existing patterns in similar modules before proposing new ones
3. Ensure proposed changes align with pydantic-settings configuration pattern
4. Consider hardware simulation mode for testability
5. Plan tests alongside implementation (TDD preferred)
```

### 12.3 For Bash Agent

```
Common commands for this project:
- Install: pip install -e ".[all,dev,test]"
- Test: PYTHONPATH=src pytest tests/unit/ -v
- Lint: ruff check . && ruff format .
- Type check: mypy src/ptpd_calibration --ignore-missing-imports
- Run UI: python app.py
- Run API: ptpd-server
```

### 12.4 For Code Implementation

```
When writing code:
1. Use type hints for all function parameters and return values
2. Use Pydantic models for all data structures
3. Add logging with get_logger(__name__)
4. Follow existing import organization pattern
5. Add docstrings in Google style
6. Create fixtures for new models in tests/conftest.py
7. Use lazy imports for heavy dependencies (torch, etc.)
```

---

## SECTION 13: PRIORITY IMPLEMENTATION CHECKLIST

### Immediate (Phase 1)

- [ ] Create `src/ptpd_calibration/core/logging.py` with structured logging
- [ ] Create `src/ptpd_calibration/core/debug.py` with debugging utilities
- [ ] Create `src/ptpd_calibration/integrations/protocols.py` with device protocols
- [ ] Create `tests/utils.py` with test helper functions
- [ ] Enhance `tests/conftest.py` with factory fixtures
- [ ] Fix TODO in `src/ptpd_calibration/workflow/session_log.py:85`

### Short-term (Phase 2)

- [ ] Implement real spectrophotometer driver
- [ ] Create neural curve prediction UI tab
- [ ] Add session statistics dashboard
- [ ] Write integration tests for hardware

### Medium-term (Phase 3-4)

- [ ] Add PostgreSQL support for multi-user
- [ ] Implement WebSocket real-time updates
- [ ] Create comprehensive visual regression baselines
- [ ] Add performance benchmarks

---

## TEMPLATE USAGE NOTES

1. **Copy relevant sections** to CLAUDE.md for persistent context
2. **Use C4 diagrams** when discussing architecture changes
3. **Follow verification loop** after every implementation
4. **Update this template** when patterns change
5. **Reference test patterns** when writing new tests
