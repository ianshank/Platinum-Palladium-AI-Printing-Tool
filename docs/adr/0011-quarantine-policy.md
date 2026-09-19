# ADR-0011: Quarantine policy for unreachable code

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

About 52% of backend lines (eleven packages plus `deep_learning` and `ui`) have no importer other than examples or an unwired Gradio tab; CI already ignored their tests.

## Decision

Packages with no wired importer move to `experimental/` (candidates for adoption) or `contrib/` (hardware and cloud adapters). They are excluded from coverage floors and required CI. Anything still there after one release without an adopting ADR is deleted. `ai/` is deleted outright.

## Consequences

The remaining core is about 45k lines with legible package boundaries; the bounded-context split is revisited only after ADR-0005 decides.
