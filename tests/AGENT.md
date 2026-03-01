# Tests Directory (Backend)

## Purpose
Python test suite for the backend. Organized by test type with shared fixtures and utilities.

## Key Subdirectories
- `api/` — FastAPI endpoint tests (TestClient-based)
- `unit/` — Isolated unit tests for individual modules
- `integration/` — Cross-module integration tests
- `e2e/` — End-to-end workflow tests
- `performance/` — Performance benchmarks
- `sanity/` — Quick smoke tests for CI
- `visual/` — Visual regression tests
- `fixtures/` — Test data (sample images, quad files, density measurements)
- `utils/` — Shared test utilities

## Key Files
- `conftest.py` — Root-level pytest fixtures (app client, test database, sample data)
- `utils.py` — Shared utility functions for tests
- `README.md` — Detailed test documentation

## Conventions
- **pytest**: All tests use pytest (not unittest)
- **Fixture-driven**: Use `conftest.py` fixtures for setup — avoid repetitive setup in test functions
- **Markers**: Use `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.e2e` for selective runs
- **Naming**: `test_{module}_{behavior}.py` or match source module name

## Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/api/ -v                # API tests only
pytest tests/ -v -k "curve"         # Tests matching "curve"
pytest tests/ -v -m "not slow"      # Skip slow tests
pytest tests/sanity/ -v             # Quick CI smoke tests
```

## Pitfalls
- Do NOT use production database/files in tests — always use fixtures
- Some tests require optional dependencies (PyTorch, OpenCV) — skip gracefully if missing
- `conftest.py` fixtures are scoped — check `scope` parameter for session vs function

## Related
- `../src/ptpd_calibration/` — Source code under test
- `../frontend/src/__tests__/` — Frontend test suite (Vitest, not pytest)
- Legacy-vs-new comparison tests (migration comparison suite, if available)
