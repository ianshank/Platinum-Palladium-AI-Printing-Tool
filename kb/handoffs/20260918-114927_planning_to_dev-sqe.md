# Planning → DEV-SQE Handoff

**Date**: 2026-09-18 11:49:27 UTC
**Session ID**: e274d4ad-70e6-5d62-ba38-6b277fa971b0
**Branch**: claude/ptpd-validation-sdlc-plan-j2a2gc (base commit 8b01859 on trunk c9ef03a)
**Plan**: docs/plans/2026-09-validation-sdlc-plan.md (Phase 0 is what this handoff covers)

## Tasks Ready for Implementation (Phase 0 — establish truth)

Owner decisions listed in plan §9 gate TASK-001 only; every other task can start immediately.

### TASK-001: Trunk reconciliation and branch protection (ARC-01 / OPS-01)
**Priority**: Critical · **Blocked by**: owner decision §9 Q1
**Acceptance Criteria**:
- [ ] GitHub default branch is `main`, fast-forwarded to c9ef03a (`main..trunk` is empty, so nothing is lost)
- [ ] Rulesets: PR required, `all-green` required status, linear history, no force-push, on `main` and `v*`
- [ ] Workflow permissions read-only; `release` and `huggingface` environments with a required reviewer
- [ ] PRs #13/#14/#16/#17 closed with rationale; #34/#36 retargeted or closed per plan §3.5

### TASK-002: Consolidated CI workflow (OPS-08, with D08/D09 first)
**Priority**: Critical
**Acceptance Criteria**:
- [ ] Deploy job cannot run for non-tag refs and fails loudly when `HF_TOKEN`/`HF_USERNAME` are unset (first commit, before anything else)
- [ ] `.github/workflows/ci.yml` replaced by `docs/plans/2026-09-review/ci-proposed.yml` at phase-1 settings; `ci-cd.yml` and `tests.yml` deleted
- [ ] pnpm installed before `setup-node`; `all-green` passes on a no-op PR; ≤ 9 jobs per PR; SHAs re-verified with `pinact`
**Files**: .github/workflows/*

### TASK-003: uv migration and importable package (OPS-03 / ARC-13 / OPS-18)
**Priority**: Critical
**Acceptance Criteria**:
- [ ] `pyproject.toml` declares matplotlib (or `curves.visualization` is lazy), tifffile, python-dotenv, psutil, pyyaml; extras per expert-devops.md §3; `[dependency-groups] dev`
- [ ] `.python-version` = 3.12 committed (remove the `.gitignore` line that ignores it); `uv.lock` committed; `uv lock --check` in CI; `pylock.toml` exported
- [ ] `requirements-dl.txt` deleted; `requirements.txt` generated transitionally
- [ ] `uv sync --frozen --extra server && python -c "import ptpd_calibration"` succeeds in a clean container
- [ ] `ptpd` console-script entry fixed or removed
**Files**: pyproject.toml, uv.lock, .python-version, .gitignore, requirements*.txt

### TASK-004: Fix the three red backend tests (TST-01)
**Priority**: High
**Acceptance Criteria**:
- [ ] `src/ptpd_calibration/session/logger.py:470` uses `stats[record.paper_type]`; the blanket `except` no longer swallows `NameError`; a regression test with 3 records over 2 papers passes
- [ ] `tests/unit/test_neuro_symbolic.py:1067` seeded or asserted statistically
- [ ] Baseline command (expert-testing.md §1) reports 0 failed

### TASK-005: Fix the five red frontend tests (TST-02 / OPS-06 / ARC-10)
**Priority**: High
**Acceptance Criteria**:
- [ ] `useKeyboardShortcuts.ts` guards `HTMLSelectElement`
- [ ] Layout, CurveEditor, uiSlice tests updated to current behaviour (salvage #34 commits 90e2c94/ba84f70 where still relevant)
- [ ] `frontend/package.json` has `packageManager`; `frontend/.nvmrc` present; `pnpm test:run` 831/831

### TASK-006: Lint, format, and mypy baseline (OPS-05 / OPS-07)
**Priority**: High
**Acceptance Criteria**:
- [ ] `ruff check src tests app.py scripts` and `ruff format --check` exit 0 (86 errors, 34 files today)
- [ ] `[tool.mypy] files` allowlist = the nine zero-error packages; `mypy` exit 0 on it; expansion order documented in CONTRIBUTING.md

### TASK-007: Strict pytest configuration (TST-03 / OPS-04)
**Priority**: High
**Acceptance Criteria**:
- [ ] 13 markers registered; `addopts = "-ra --strict-markers --strict-config"`; `xfail_strict = true`; `timeout = 120`; coverage flags moved to CI
- [ ] The five collection errors gated by markers/`importorskip`; `pytest --collect-only -q` reports 0 errors and 0 warnings

### TASK-008: Repository hygiene and governance files (OPS-02 / OPS-15)
**Priority**: High
**Acceptance Criteria**:
- [ ] `git rm -r --cached node_modules .gradio hf_check`; UTF-16 `.husky/pre-push` deleted or rewritten
- [ ] `.pre-commit-config.yaml` per expert-devops.md §5; `CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md`, `.github/dependabot.yml`
- [ ] `git ls-files -i -c --exclude-standard` is empty; `pre-commit run --all-files` clean

### TASK-009: Immediate security fixes (SEC-01 / SEC-02 / SEC-03 / SEC-07)
**Priority**: Critical
**Acceptance Criteria**:
- [ ] `/api/curves/upload-quad` and `/api/curves/export` use uuid names, extension allowlist, streamed size cap, cleanup; client-supplied names never touch a path
- [ ] Request-size middleware; `max_length` on list and string fields
- [ ] CORS: explicit origins; `allow_credentials` default false; startup guard rejects `*` with credentials
- [ ] Tests: traversal filename → 400 with sentinel intact; > 50 MB → 413; foreign-origin preflight returns no allow-origin
**Reference**: full security report delivered to the owner out-of-band (SEC-01..22); do not commit exploit details

### TASK-010: Documentation purge and canonical set (ARC-14 / ARC-16)
**Priority**: Medium
**Acceptance Criteria**:
- [ ] Root contains exactly README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, LICENSE, AGENTS.md, CLAUDE.md
- [ ] Dispositions per expert-architecture.md §4.1; `CLAUDE.md` ≤ 6 KB with no nonexistent path; 23 `AGENT.md` deleted; `scripts/check-doc-sprawl.sh` in CI
- [ ] Migration status, Celery claim, and env-var names (`PTPD_LLM_*`) corrected

### TASK-011: KB protocol minimization (ARC-17, ADR-0015)
**Priority**: Medium · **Blocked by**: owner decision §9 Q8
**Acceptance Criteria**:
- [ ] SessionStart hook opt-in, startup-only, ≤ 4096 bytes, staleness banner; no `compact`/`resume` injection
- [ ] Blocking Stop prompt hook removed; explicit `/handoff` skill
- [ ] Ledger JSON-schema validated in CI; February summaries archived under `kb/archive/2026-02/`

### TASK-012: Honest README metrics (TST-17)
**Priority**: Medium
**Acceptance Criteria**:
- [ ] Test counts and coverage in README are produced by a script from the latest `main` CI artifact
- [ ] "AlphaZero", "self-play", "Expert Iteration" wording removed until SCI-07 lands

## Context Files
- docs/plans/2026-09-validation-sdlc-plan.md
- docs/plans/2026-09-review/verification-matrix.md
- docs/plans/2026-09-review/expert-devops.md and ci-proposed.yml
- docs/plans/2026-09-review/expert-testing.md
- docs/plans/2026-09-review/expert-architecture.md
- docs/plans/2026-09-review/expert-science.md (Phase 1 context)

## Dependencies
- TASK-001 before TASK-002's final required-check wiring (rulesets need the check name)
- TASK-003 before TASK-006/007 (uv-managed dev group)
- TASK-004/005 before TASK-002's blocking gate can be green

## Notes for DEV-SQE
- Reproduce each failure before fixing it (commands in expert-testing.md §1 and expert-devops.md §1).
- Keep PRs under ~400 changed lines; one concern per PR; do not bundle `kb/` changes with code.
- Phase 1 (SCI-06, SCI-08, SCI-01, SCI-02, SCI-05, TST-04..10) follows once the trunk is green; do not start MCTS changes before Phase 0 exits.

---
**Handoff Document**: Immutable after creation
**Next Phase**: DEV-SQE implementation
