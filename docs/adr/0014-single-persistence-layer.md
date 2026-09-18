# ADR-0014: One persistence layer

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

Three persistence mechanisms coexisted: an in-memory/JSON `CalibrationDatabase`, per-session JSON files, and two orphaned SQLite modules under `data/`; the React session log lived only in browser memory.

## Decision

A single SQLite-backed repository module serves calibrations, sessions, and curves through `/api/calibrations`, `/api/sessions`, and the curve store. A migration script imports the JSON stores. `data/` is deleted after migration.

## Consequences

Sequenced after ARC-02 and ARC-06 (plan Phase 3).
