---
description: Run full verification pipeline (lint, typecheck, backend tests, frontend tests, build)
---

# Full Verification Pipeline

Run this workflow after any significant change to ensure nothing is broken.

## Steps

// turbo-all

1. Run backend tests with coverage:

```bash
cmd /c "set PYTHONPATH=src && python -m pytest tests/ --tb=short --ignore=tests/e2e -q 2>&1"
```

1. Run frontend type checking:

```bash
cd frontend && pnpm typecheck
```

1. Run frontend linting:

```bash
cd frontend && pnpm lint:fix
```

1. Run frontend unit tests:

```bash
cd frontend && pnpm test -- --run
```

1. Run frontend production build:

```bash
cd frontend && pnpm build
```

1. Run backend integration tests:

```bash
cmd /c "set PYTHONPATH=src && python -m pytest tests/integration/test_api_endpoints.py -v --tb=short 2>&1"
```

## Success Criteria

- All backend tests pass (allow pre-existing failures in `test_session_logger`)
- Frontend typecheck passes with 0 errors
- Frontend lint passes with 0 errors
- All frontend tests pass
- Production build succeeds
- All 16 API integration tests pass
