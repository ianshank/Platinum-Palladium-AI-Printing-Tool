# ADR-0008: Documentation placement rules

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

Thirteen root markdown files, four of them superseded roadmaps, contradicted each other on migration status, test counts and coverage; 23 `AGENT.md` files were loaded by nothing and drifted. Agent sessions ingested this as context.

## Decision

The root may contain only `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE`, `AGENTS.md`, `CLAUDE.md`. Status goes in PR descriptions, decisions in `docs/adr/`, plans in `docs/plans/` (deleted when executed), the roadmap in `docs/roadmap.md`, history in `docs/archive/`. No `AGENT.md`, `*_SUMMARY.md`, `*_NEXT_STEPS*.md`, `QUICK_REFERENCE.md`, or `plan.md` anywhere. Enforced by `scripts/check-doc-sprawl.sh`.

## Consequences

Superseded roadmaps were deleted or archived on 2026-09-18; `CLAUDE.md` was rewritten under 6 KB with correct paths.
