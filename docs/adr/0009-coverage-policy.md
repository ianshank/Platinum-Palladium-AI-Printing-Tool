# ADR-0009: Coverage policy: diff coverage plus tiered floors

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

A single global coverage number (claimed 84%, measured 64% branch-inclusive) excluded the API server and hid that `tests/api` was 70% status-code-only assertions.

## Decision

Branch coverage is on. CI enforces per-package floors as separate `coverage report --include` steps (mcts 70, chemistry 85, curves 80, whole package 60, rising to 80/90/85/70 in phase 2) and `diff-cover --fail-under=90` on changed lines in PRs. Floors only rise. `api/server.py` leaves the omit list once e2e is blocking.

## Consequences

Coverage flags live in CI, not in `addopts`, so local runs stay fast.
