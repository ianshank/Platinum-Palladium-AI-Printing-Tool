# ADR-0002: One CI workflow, one required check

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

Three overlapping workflows (`ci.yml`, `ci-cd.yml`, `tests.yml`) used three Python versions, a nonexistent `test` extra, doubly masked lint, `|| true` on tests, and a deploy job that force-pushed any `claude/*` branch and reported success when the token was missing. None had produced a green run on the trunk.

## Decision

`.github/workflows/ci.yml` is the only workflow. `all-green` is the single required status check. Every action is SHA-pinned, `permissions: {}` at the top with per-job grants, concurrency per PR, no duplicate push+PR runs for feature branches. Advisory checks (e2e, dependency review, CodeQL, audits, ruff `S`) ratchet to blocking on the dated schedule in `docs/plans/2026-09-review/expert-devops.md` §2. Deploys run only from `v*` tags after the gate, behind the `huggingface` environment, and fail loudly on missing secrets.

## Consequences

About nine jobs per PR instead of thirty-three. The owner must create the `release` and `huggingface` environments with a required reviewer.
