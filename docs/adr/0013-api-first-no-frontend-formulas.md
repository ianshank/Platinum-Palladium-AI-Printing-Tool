# ADR-0013: API-first: domain math lives in Python only

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

`frontend/src/stores/slices/chemistrySlice.ts` re-implemented the coating-recipe formula with different constants from `chemistry/calculator.py`; the two UIs could give different answers for the same input.

## Decision

All domain math (chemistry, exposure, linearization, imaging) is implemented once, in Python, behind an endpoint with a pydantic schema. The frontend consumes generated OpenAPI types (`pnpm generate:types`) and never re-implements a formula. Shared contract goldens are asserted by both pytest and vitest.

## Consequences

The TypeScript chemistry formula is deleted when `POST /api/chemistry/calculate` lands (ARC-06).
