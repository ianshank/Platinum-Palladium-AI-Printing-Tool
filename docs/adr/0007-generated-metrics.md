# ADR-0007: README metrics are generated, never typed

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

The README, CLAUDE.md, CHANGELOG, KB summaries and PR bodies carried five different frontend test counts and four coverage figures, none reproducible by CI.

## Decision

No numeric quality claim is written into markdown. CI produces `reports/metrics.json` via `scripts/test_metrics.py` and renders it into the job summary; documents link to the workflow. `scripts/check-doc-sprawl.sh` and review enforce the rule.

## Consequences

Readers get current numbers from the latest run instead of stale prose.
