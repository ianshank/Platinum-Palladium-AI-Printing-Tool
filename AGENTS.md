# AGENTS.md

Rules for every agent (Claude Code, Copilot, Antigravity, or otherwise) that
edits this repository. `CLAUDE.md` includes this file. Enforced by
`scripts/check-doc-sprawl.sh` in CI and in pre-commit (ADR-0008).

## Documentation and file placement

1. The repository root may contain only: `README.md`, `CONTRIBUTING.md`,
   `SECURITY.md`, `CHANGELOG.md`, `LICENSE`, `AGENTS.md`, `CLAUDE.md`. Any other
   `*.md` at the root fails CI.
2. Never create files named `*_SUMMARY.md`, `*_NEXT_STEPS*.md`,
   `*_IMPLEMENTATION*.md`, `INVESTIGATION*.md`, `QUICK_REFERENCE.md`, `plan.md`,
   or dated status reports anywhere in the repository. Status goes in the PR
   description; decisions go in `docs/adr/NNNN-title.md`; plans go in
   `docs/plans/YYYY-MM-title.md` and are deleted when executed; the roadmap is
   `docs/roadmap.md` only.
3. Numeric claims (test counts, coverage, migration percentages) are not written
   into markdown. Link the CI run or its job summary.
4. Do not create per-directory `AGENT.md` files; Claude Code does not load them
   and they drift. Nested `CLAUDE.md` files are allowed only under `frontend/`,
   `src/ptpd_calibration/`, and `tests/`, each under 1.5 KB, containing commands
   and hard constraints only.
5. Agent working state (`kb/`) rides in its own commit, never mixed with code
   changes, and follows `docs/agents/kb-protocol.md`.
6. Deleting a stale document never needs an ADR; adding a root file does.

## Code

7. Backwards compatible by default; no hard-coded tunables (settings classes
   with environment overrides); reusable helpers over copy-paste; module
   loggers with debug logging on guarded paths; explicit return types.
8. Tests accompany every change. Never skip, disable, or quarantine a test to
   get green; fix the cause or mark it `xfail(strict=True)` with a reason and a
   plan item.
9. Do not modify `src/ptpd_calibration/ui/` (frozen legacy UI).
10. Security findings are fixed before their details are committed to this
    public repository.

## Process

11. Trunk-based development on `main`; branches live less than a week; one
    concern per PR; Conventional Commits; `all-green` is the required check.
12. Before ending a session that did meaningful work, run the matching handoff
    skill (`.claude/skills/<role>-handoff/SKILL.md`).
