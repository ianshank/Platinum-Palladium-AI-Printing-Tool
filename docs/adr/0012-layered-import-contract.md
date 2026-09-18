# ADR-0012: Layered import contract

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

A cycle `core → curves → llm → ml → core` existed through two lazy imports, plus upward edges `mcts → agents` and `ml → gcp`; layering was implicit.

## Decision

Layers: `core` → domain (`curves`, `chemistry`, `detection`, `imaging`, `exposure`, `zones`, `papers`, `analysis`, `session`, `proofing`) → adapters (`gcp`, storage) → intelligence (`ml`, `neuro_symbolic`, `mcts`) → assistants (`llm`, `agents`) → `api`. `import-linter` enforces the contract as a required check; no lazy imports to hide cycles. `StorageBackend` moves to `core`; `CurveData.save` delegates without importing `curves`; the LLM branch of curve enhancement moves under `llm/`.

## Consequences

Six violations at twelve sites are fixed in place; no package split is required.
