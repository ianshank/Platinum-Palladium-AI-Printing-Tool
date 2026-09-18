# ADR-0012: Layered import contract

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

A cycle `core → curves → llm → ml → core` existed through two lazy imports, plus upward edges `mcts → agents` and `ml → gcp`; layering was implicit.

## Decision

Layers: `core` → domain (`curves`, `chemistry`, `detection`, `imaging`, `exposure`, `zones`, `papers`, `analysis`, `session`, `proofing`) → adapters (`gcp`, storage) → intelligence (`ml`, `neuro_symbolic`, `mcts`) → assistants (`llm`, `agents`) → `api`. `import-linter` is to enforce the contract as a required check; no lazy imports to hide cycles. `StorageBackend` moves to `core`; `CurveData.save` delegates without importing `curves`; the LLM branch of curve enhancement moves under `llm/`.

## Status of enforcement

**Accepted, not yet enforced.** `import-linter` is a declared development dependency; there is no `[tool.importlinter]` contract, no CI step and no pre-commit hook, and the violations below are still present. Enforcement has two preconditions that this record originally missed:

1. The root `__init__.py` imports thirteen subpackages behind `suppress(ImportError)`, so a contract would be checked against an import graph the facade creates rather than the one engineers wrote.
2. A second cycle exists that this record did not name: `config` imports `mcts.config` inside a validator, and `Settings.mcts` is typed `Any` to hide it. `config` sits beneath every layer, so no contract can be drawn until that edge is reversed.

The layer map above also names twenty packages; eleven more exist and are unplaced, and a layers contract is only meaningful over a complete partition. Placing them depends on the quarantine decision in ADR-0011.

Order of work: dismantle the facade, reverse the `config` edge, place the remaining packages, then land the contracts. The two `forbidden` contracts (`config` imports nothing from the package; `core` imports nothing but `core`) hold regardless of the ADR-0005 outcome and come first.

## Consequences

Six violations at twelve sites are to be fixed in place; no package split is required. None are fixed yet: the cycle this record names is present edge for edge at `core/models.py`, `curves/ai_enhance.py`, `llm/assistant.py` and `ml/active_learning.py`.
