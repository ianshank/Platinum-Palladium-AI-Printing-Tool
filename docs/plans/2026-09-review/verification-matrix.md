# Peer-review finding verification matrix (draft, 2026-09-18)

Legend: CONFIRMED / PARTIAL / REFUTED / NEW (not in peer review)

## Where models agreed

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A1 | Committed node_modules/.gradio + duplicate package tree | PARTIAL | 268 tracked files under root `node_modules/` (husky/lint-staged toolchain only, not frontend deps; repo pack is 3.83 MiB so size is not the issue). `.gradio/certificate.pem` is ONE tracked file and it is a public CA bundle (BEGIN CERTIFICATE, no private key) so the "flagged data / uploaded scans leak" concern does not apply. `hf_check/` is ONE stale file (`papers/profiles.py`, older, lacks CYANOTYPE_PAPERS). All three are listed in .gitignore but were committed before the rule (reactive .gitignore). |
| A2 | MCTS/NN validated only against self-authored simulator | CONFIRMED | engine._evaluate → simulator.simulate → scorer.score (mcts/engine.py:316-338); PhysicsConstants are hand-set defaults (mcts/config.py:322-402); no measured-print dataset in repo; only fixture is tests/fixtures/create_dummy_image.py. |
| A3 | Do not build MCTS dashboard / Celery / PWA / i18n next | CONFIRMED as a priority call | README "In Progress" list; architecture.md already claims Celery+Redis exists (it does not). |
| A4 | Migrate to uv + lockfile, Ruff, fast type checker | PARTIAL | Ruff is ALREADY configured (pyproject [tool.ruff], 9 rule families, no S rules) and mypy is configured; what is missing is a lockfile and a passing baseline (ruff: 32 errors; format: 34 files; mypy: 721 errors in 64 files). |
| A5 | Three manifests, no lockfile, pnpm vs npm | CONFIRMED with nuance | No uv.lock/poetry.lock/pylock.toml. requirements.txt adds matplotlib + tifffile that pyproject omits. frontend/ correctly uses pnpm-lock.yaml; root package-lock.json is for husky only (mixed managers, minor). |
| A6 | Property-based/metamorphic testing highest ROI | CONFIRMED gap | zero Hypothesis imports; 10/142 test files use parametrize. |
| A7 | Agent/LLM layer needs OWASP LLM/Agentic hardening | PARTIAL | Tool registry is a fixed allowlist of domain tools (agents/tools.py:148-300) with no shell/exec; agent loop has max_iterations and orchestrator task_timeout_seconds, but no token budget; prompts f-string user fields (llm/prompts.py:59-123). See security report. |
| A8 | Coverage % alone inadequate gate | CONFIRMED | coverage omits ui/ and api/server.py (pyproject [tool.coverage.run]); CI never produced the 84% figure (CI red). |
| A9 | Supply-chain controls missing | CONFIRMED | no SBOM/provenance/dependency-review/scorecard/dependabot; actions pinned by tag. |
| A10 | Seeded determinism + golden files missing | CONFIRMED | unseeded global `random` at engine.py:295,374; tree.py:228,234; training.py:121,396; no golden/snapshot tests in tests/unit/mcts. |

## Where models disagreed

| # | Topic | Resolution |
|---|---|---|
| D1 | Is MCTS the right algorithm | Parameter space is 5 continuous params (not 6-10) discretized into bins and chosen in a fixed decision_order (mcts/config.py:37-73,137-148). This is a sequential-decision framing imposed on what is functionally a 5-dim continuous optimization with a cheap simulator. Ablation is cheap and decisive. Recommendation: run ablation first (both models' positions converge here). |
| D2 | Dual UI | CONFIRMED worse than either said: the HF Space (updated 29 Nov 2025, sdk gradio 4.44.0) has a different file tree from main; the deploy job in ci-cd.yml force-pushes HEAD of main OR any claude/* branch to the Space, and has evidently not run successfully since Nov 2025. app.py monkeypatches Gradio. |
| D3 | Doc sprawl | 13 root markdown files (not 10). CLAUDE.md says 12/15 migrated, ~75% cov; README says 15/15, 84%; docs/migration/progress.json says 15/15, 80%. 23 AGENT.md files added 2026-03-01. Delete-then-add sequencing recommended. |
| D4 | Framing (architecture vs epistemic) | Both; but a THIRD blocker precedes both: main has been red on the primary CI workflow since at least 2026-02-07 (5/5 runs failed) and nobody noticed for 7 months. |
| D5 | Celery/Redis | Deprioritize (single-operator). architecture.md must stop claiming it exists. |
| D6 | "4,400+ tests / 84%" trust | REFUTED as stated by Opus: pytest collects 4,687 tests, so "4,400+" is substantiated; 104/726/9 are subsets (tests/api, vitest, playwright). BUT the numbers are stale/unverifiable in CI: vitest today = 826 pass / 5 fail; the 84% has never been reproduced by a green CI run. |
| D7 | Coverage mechanism | Combine: diff-coverage on changed lines + tiered floors on curves/chemistry/mcts. |

## New findings not in the peer review

| # | Finding | Evidence | Severity |
|---|---|---|---|
| N1 | Primary CI red on main since Feb 2026 | actions runs 21805931836 etc. all `failure`; setup-node cache:pnpm precedes pnpm/action-setup in ci.yml (frontend job dies in 4 s); backend job blocked by ruff/mypy errors. | Critical (process) |
| N2 | HF Space deploy from any claude/* branch with --force | ci-cd.yml deploy-huggingface `if: startsWith(github.ref, 'refs/heads/claude/')` + `git push hf HEAD:main --force` | High (security/process) |
| N3 | Path traversal + unbounded read in /api/curves/upload-quad | server.py:381-395 `upload_dir / file.filename`, `await file.read()` no cap; contrast with upload_scan hardening at 228-262 | High |
| N4 | Real runtime bug: undefined name `paper_stats` | session/logger.py:470 (F821) inside try/except that swallows it as a warning → statistics silently wrong | Medium |
| N5 | Frontend suite currently failing | 5 failures incl. uiSlice sidebarOpen default test stale since 2026-02-22 fix | Medium |
| N6 | tests.yml installs nonexistent `[test]` extra | pyproject has no `test` extra → job cannot pass | Medium |
| N7 | PR #36 mega-PR risk | 554 files, +177,966/-9,291, 140 commits, self-reported 823 type errors, 15% coverage; will conflict with everything | High (process) |
| N8 | Four open PRs (#13,#14,#16,#17) target a stale non-main base branch | list_pull_requests | Low (hygiene) |
| N9 | torch.load(weights_only=False) and pickle.load on model files | pipelines.py:391, mcts/training.py:548, ml/predictor.py:217 | Medium |
| N10 | 721 mypy errors while ci.yml treats mypy as blocking | mypy run today | Medium |
| N11 | Unregistered pytest markers (15 kinds) | grep of tests/ | Low |
| N12 | Unverifiable "Celery + Redis" in docs/architecture.md | no celery in deps or src | Low |
| N13 | Last commit on main 2026-03-01; project dormant 6.5 months except PR #36 (Aug) | git log | Context |

## Live CI evidence from PR #37 (docs-only change, head 9ac33cf, 2026-09-18)

- Frontend job (ci.yml) failed at 11:34:40Z after 9 s with `##[error]Unable to locate executable file: pnpm` during `actions/setup-node@v4` (`cache: pnpm`), which runs BEFORE `pnpm/action-setup@v4`. Root cause of the frontend failures on main since 2026-02-07. Fix: move `pnpm/action-setup` above `setup-node` (2-line reorder).
- A docs-only PR triggers 4 workflow runs (ci-cd.yml fires twice: on push to claude/* AND on pull_request) and 20 check runs (9-cell OS x Python matrix in tests.yml, 3 test-type matrix in ci-cd.yml x2, lint x3, backend, frontend, e2e).
- Runner warnings: all actions pinned to Node-20 majors (`actions/checkout@v4`, `setup-node@v4`, `upload-artifact@v4`) are being force-run on Node 24; GitHub deprecation notice 2025-09-19.
- Backend job (ci.yml, Python 3.12): `ruff check src/` exits 1 with "Found 32 errors" (matches local run). mypy and pytest steps never reached.
- Lint & Type Check (tests.yml): `ruff check .` exits 1 with "Found 129 errors. 84 fixable" (32 in src + 97 in tests; e.g. I001 import sorting in tests/unit/test_zone_mapping.py:6).
- Unit Tests matrix (tests.yml, all 9 OS x Python cells): `pip install -e ".[all,dev,test]"` then `pytest tests/unit/ -x` → "collected 0 items / 1 error": `src/ptpd_calibration/__init__.py:59` → `curves/__init__.py:48` → `curves/visualization.py:12` `import matplotlib` → ModuleNotFoundError. N14: the package is UNIMPORTABLE from pyproject metadata alone because matplotlib (and tifffile) live only in requirements.txt while the top-level `__init__` eagerly imports the visualization module. A `pip install ptpd-calibration` user gets a broken package.

## N15 (governance): `main` is not the trunk
- `git ls-remote --symref origin HEAD` → default branch is `claude/implement-chat-requirements-014YeEiyBSMJL91puKEgVcih` at c9ef03a (2026-03-01).
- `main` = 43537f1 (2026-02-08), an ancestor of the default branch, 135 commits behind (`git rev-list --count 43537f1..c9ef03a`).
- PR #35 (AGENT.md files) and #33 merged into the default branch, not main. PRs #34/#36 target main; #13/#14/#16/#17 target the default branch. The security expert confirmed via GitHub API: no branch is protected, 26 `claude/*` branches exist.
- Consequence: every "PR to main" CI run merges 135 unrelated commits; the HF deploy job pushes `HEAD:main` of whichever branch fired. Fix: make `main` the default, fast-forward it to c9ef03a, protect it, and retarget open PRs.
- ci-cd.yml unit job (requirements.txt installed) is the only job that actually runs tests: 3 failed / 4,017 passed / 120 skipped on head 9ac33cf. Two failures are the `paper_stats` NameError (N4) — tests DO catch it, so N4 is a known-red test on the trunk, not silent. One was introduced by this session's interim state file and fixed in 8b01859.
