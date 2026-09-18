# ADR-0010: The agent framework stays unexposed until hardened

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

The ReAct agent parses model output with `json.loads` and calls tools with unvalidated `**kwargs`, has no token budget, and re-plans on keyword matches; it is not reachable from any route today.

## Decision

No route or UI exposes `agents/` until SEC-10 (prompt-injection defenses), SEC-11 (validated tool arguments, budgets, confirmations for write tools), SEC-12 (schema-validated memory and checkpoints) and SEC-20 (`schemathesis` and `promptfoo` red-team gates) pass in CI.

## Consequences

The package may be quarantined under ADR-0011 in the meantime.
