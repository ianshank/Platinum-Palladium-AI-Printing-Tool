# Validation-First SDLC Plan (September 2026)

**Status:** proposed, awaiting the owner decisions listed in §9
**Baseline commit:** `c9ef03a` (GitHub default branch, 2026-03-01); `main` = `43537f1` (2026-02-08)
**Inputs:** a three-model peer review (Gemini 3.8 Flash Thinking, Claude Opus 5 Thinking, Kimi K3) of this repository, verified finding-by-finding against the code, plus five domain-expert deep dives commissioned for this plan (scientific validation, application and AI security, DevOps and supply chain, test architecture, software architecture and documentation). Every expert claim is cited `path:line` in the reports under `docs/plans/2026-09-review/`.

---

## 1. Executive summary

The peer review was right about the diagnosis and understated it. Verifying its claims against the code turned up four blockers that sit *in front of* everything the review recommended, and each one changes what the project should do next.

1. **The repository has no working definition of "green".** GitHub's default branch is a `claude/*` feature branch 134 commits ahead of `main`; `main` is stale since February and unprotected. The primary `CI` workflow has failed on every run on `main` since 2026-02-07: the frontend job dies in nine seconds on a step-ordering bug, the nine-cell unit-test matrix collects zero tests because the package cannot be imported from its own metadata, and the one job that does run tests has three deterministic failures. The Hugging Face deploy job reports success while doing nothing (it exits 0 when the token is unset), and would force-push any `claude/*` branch to production the moment the variables are set. Every claim in the README ("4,400+ tests", "84% coverage", "726 passing") is therefore unverifiable by construction, even where the underlying number is real.
2. **The product does not run the algorithm the README describes.** No production code instantiates the tree search; `/api/mcts/search` runs two heuristic subagents and one simulation and reports 800 simulations. The "self-play" trainer samples uniform random parameters and trains the policy head to imitate that sampler; the network is never consulted during search. The user's target curve is accepted and silently discarded. The simulator's objective is nearly flat over its whole domain (quality spans 0.41 to 0.48), one of the six search dimensions has no effect on the output, and simulated densities are written into a field named `measured_densities`. The peer review's "closed evaluation loop" is correct; there is also no loop.
3. **The shipped React application cannot produce the product's core deliverable.** No API endpoint applies a curve to an image or exports a digital negative; `imaging/` is reachable only from the Gradio UI, and the `/api/export/*` endpoints that `CLAUDE.md` documents do not exist. The React chemistry calculator re-implements the recipe formula client-side with different constants from the Python one. About 52% of backend code (46,874 of 91,731 lines) has no importer other than examples or an unwired Gradio tab, including the entire 13,819-line `deep_learning` package.
4. **The public surface has pre-authentication file-system defects** (path traversal on two endpoints, unbounded upload reads, permissive CORS with credentials, image-decode amplification of roughly 40,000×) and no authentication or rate limiting on LLM routes that spend a server-side key.

The plan therefore reorders the peer review's recommendations: establish truth (trunk, CI, tests, README) first; make the scientific core honest and measurable second; run the MCTS-versus-baselines ablation third and let it decide whether 5,161 lines of search code survive; ship one product (FastAPI plus React as a Docker Space, with the digital-negative pipeline exposed through the API) fourth; and only then invest in the real-print holdout dataset, physics fitting, and agent-evaluation harness that turn this into a validated system. Items the peer review agreed to defer (MCTS dashboard, Celery/Redis queue, PWA, i18n) stay deferred. The draft mega-PR #36 should be closed and salvaged in pieces; its real delta against the trunk is 81 files, not 554.

Estimated effort to the end of Phase 3 is roughly twelve working weeks for one engineer with agent assistance. Phase 4 is open-ended because it depends on darkroom time.

---

## 2. What was verified

### 2.1 Where the peer-review models agreed

| # | Peer-review claim | Verdict | Evidence |
|---|---|---|---|
| A1 | Committed `node_modules`/`.gradio`, duplicate package tree | Partial | 268 tracked files under root `node_modules/` are the husky/lint-staged toolchain, not frontend dependencies (pack size 3.83 MiB). `.gradio/certificate.pem` is one public CA bundle, not flagged data. `hf_check/` is one stale copy of `papers/profiles.py`. All three are in `.gitignore` but were committed before the rule. No history rewrite is needed. |
| A2 | MCTS/NN validated only against a self-authored simulator | Confirmed and worse | See §3.1. No ingress for measured data exists at all. |
| A3 | Do not build MCTS dashboard, Celery, PWA, i18n next | Confirmed | `docs/architecture.md` already claims a Celery+Redis container exists; it does not. |
| A4 | Migrate to uv, Ruff, fast type checker | Partial | Ruff and mypy are already configured. Missing: a lockfile and a passing baseline (ruff 32 errors in `src/`, 86 in `src`+`tests`, 34 files unformatted; mypy 721 errors in 64 files; `uv lock --dry-run` resolves cleanly, so migration has no resolver blocker). |
| A5 | Three manifests, no lockfile | Confirmed | No `uv.lock`/`poetry.lock`/`pylock.toml`. `requirements.txt` carries `matplotlib` and `tifffile`, which `pyproject.toml` omits; `python-dotenv`, `psutil`, `pyyaml` are hard imports declared nowhere; eleven packages in `requirements-dl.txt` have zero import sites. |
| A6 | Property-based and metamorphic testing is the highest-ROI addition | Confirmed | Zero Hypothesis usage; 10 of 142 test files use `parametrize`. Exploratory probes found four real defects in an afternoon (§3.3). |
| A7 | LLM/agent layer needs OWASP hardening | Partial | The ReAct agent and its tools are not reachable from any route; the reachable surface is a tool-less assistant. Real issues are stored prompt injection through anonymous calibration records, denial-of-wallet, and SDK clients created without timeouts. |
| A8 | Coverage percentage alone is inadequate | Confirmed | Coverage omits `ui/` and `api/server.py`; measured whole-package coverage is 64% branch-inclusive, so the `ci.yml` gate of 70 would fail even with a clean collection. |
| A9 | Supply-chain controls missing | Confirmed | No SBOM, provenance, dependency review, Scorecard, or Dependabot; 47 `uses:` pinned by mutable tag; no `permissions:` blocks; token embedded in a git remote URL. |
| A10 | Seeded determinism and golden files missing | Confirmed | Seven unseeded global `random` call sites in `mcts/`; no golden or snapshot tests. |

### 2.2 Where the models disagreed, and the resolution

| Topic | Resolution |
|---|---|
| Is MCTS the right algorithm | The search space is five continuous parameters (not six to ten) discretized into bins and chosen in a fixed order: a sequential framing imposed on a continuous optimization with a cheap simulator. The ablation is cheap and decisive, and because the API never calls the engine, deleting it would change no user-facing behaviour today. Run the ablation (Phase 2) after the objective is made non-flat (Phase 1). |
| Dual UI | Retire Gradio, as Opus argued, but close the product gaps first: eleven Gradio features (digital negative export foremost) have no API or React equivalent. ~17–19 engineer-days. |
| Root doc sprawl | Thirteen root markdown files, not ten; five disagree on migration status, five on test counts, four on coverage. Delete nine, move two, rewrite two, then add the canonical set (Gemini's list). |
| Framing: architectural vs epistemic | Both are right; a process blocker (no trunk, red CI) precedes both. |
| Celery/Redis | Deferred; single-operator tool. Remove the claim from the architecture doc. |
| "4,400+ tests" is irreconcilable | Refuted: 4,687 node IDs collect; parametrize inflation is 2.1%. The 104/726/9 figures are subsets. What is true is that the numbers are stale (vitest is 831 with 5 failing) and unverifiable in CI. |
| Coverage gate mechanism | Both: diff-coverage on changed lines plus tiered floors on `curves/`, `chemistry/`, `mcts/`. |
| Bounded-context refactor | Not now. Six layering violations at twelve import sites and one four-package cycle can be fixed in place and enforced with `import-linter`; a package split before the ablation risks moving 17,260 lines that may be deleted. |

### 2.3 Findings the peer review missed

| # | Finding | Severity |
|---|---|---|
| N1 | `CI` workflow red on `main` since 2026-02-07; `setup-node` with `cache: pnpm` precedes `pnpm/action-setup` | Critical (process) |
| N2 | `ci-cd.yml` force-pushes `HEAD:main` to the public Space on any push to `claude/*`; exits 0 (green) when secrets are unset, which is the state today | High |
| N3 | Unsanitized client filename and unbounded read in `/api/curves/upload-quad`; unsanitized `name` in `/api/curves/export` | High |
| N4 | Undefined name `paper_stats` at `session/logger.py:470`; two unit tests have been red on the trunk because of it | Medium |
| N5 | Five failing vitest tests: one genuine hook gap, four tests drifted from implementation | Medium |
| N6 | `tests.yml` installs a `test` extra that does not exist, so `--timeout` and `--benchmark-json` are unknown arguments | Medium |
| N7 | PR #36: real delta vs trunk is 6 commits / 81 files / +26,809, self-reported 823 type errors and 15% coverage | High (process) |
| N8 | Open PRs #13/#14/#16/#17 target the trunk but are superseded by merged work; #34/#36 target stale `main` | Low |
| N9 | `torch.load(weights_only=False)` and `pickle.load` on model artifacts | Medium |
| N10 | 721 mypy errors while `ci.yml` treats mypy as blocking; nine packages have zero errors | Medium |
| N11 | Fifteen pytest marker kinds, none registered; six `--ignore` flags quarantine tests silently | Low |
| N12 | `docs/architecture.md` describes a Celery+Redis container that does not exist; `[project.scripts] ptpd` targets a module that does not exist | Low |
| N13 | Trunk dormant from 2026-03-01 until PR #36 in August | Context |
| N14 | Package unimportable from `pyproject.toml` metadata alone (eager `matplotlib` import in `curves/visualization.py`) | High |
| N15 | GitHub default branch is `claude/implement-chat-requirements-…`; `main` is 134 commits behind and unprotected | High (governance) |
| N16 | 52% of backend LOC is unreachable from any wired entry point; `deep_learning/` (13,819 LOC) is imported only by a Gradio tab that is never built | High (maintenance) |
| N17 | The React app has no digital-negative export, image preview, chemistry, exposure, papers, linearization, wedge-analysis, or persistent session API | High (product) |
| N18 | Frontend chemistry formula differs from the Python one (`chemistrySlice.ts:193-213` vs `chemistry/calculator.py:283`) | Medium |
| N19 | Import cycle `core → curves → llm → ml → core` created by two lazy imports; `mcts → agents` and `ml → gcp` upward edges | Medium |
| N20 | KB hook injects 36 KB of February-2026 state (asserting "726 passed, 0 failed", "Ready for PR") on every startup, resume, and compaction; a blocking Stop hook forces `kb/` writes into every PR; three inconsistent protocol specs coexist | Medium (agent context poisoning) |
| N21 | `.husky/pre-push` is UTF-16LE and runs a script that does not exist; `.claude/settings.local.json` is committed | Low |

### 2.4 Measured baseline (2026-09-18, Python 3.11.15, Node 22, no torch)

| Metric | Value |
|---|---|
| Backend tests collected | 4,687 (5 collection errors: psutil, gradio missing) |
| Backend `tests/api` + `tests/unit` run | 4,036 passed, 3 failed, 148 skipped, 94 s |
| Frontend vitest | 826 passed, 5 failed (831) |
| Frontend `tsc --noEmit` | 0 errors |
| `ruff check src/` / `ruff check src tests` / `ruff check .` | 32 / 86 / 129 errors |
| `ruff check --select S src/` | 65 findings |
| `ruff format --check src tests` | 34 files would change |
| `mypy src/ptpd_calibration --ignore-missing-imports` | 721 errors in 64 files (deep_learning 351, api 97, ui 96) |
| Whole-package coverage, branch-inclusive | 64% |
| Scientific-core coverage (line / branch) | chemistry 94.8 / 85.7; curves 89.9 / 71.6; mcts 78.2 / 69.2 |
| Tests with only vacuous assertions (AST heuristic) | 18.4%; tolerance-bearing numeric assertions 3.9% |
| Job slots triggered per PR push today / proposed | 33 (4 workflow runs) / 9 |
| Backend LOC / packages / unreachable LOC | 91,731 / 31 / 46,874 |
| KB context injected at session start | 36,156 bytes (plus 18.9 KB `CLAUDE.md`) |

---

## 3. Expert findings (condensed)

Full reports with `path:line` citations: `docs/plans/2026-09-review/expert-*.md`. Plan-item IDs (SCI-, SEC-, OPS-, TST-, ARC-) refer to the tables in those reports.

### 3.1 Scientific validation (SCI-01 … SCI-12)

- **No ground-truth ingress.** The value target is `QualityScorer.score(ExtendedProcessSimulator.simulate(params))`. All eight `PhysicsConstants` construction sites use defaults; no function fits them from records. `/api/mcts/feedback` logs and discards measurements. `MCTSSearchRequest.target_curve` is placed in context and never read.
- **The engine is unreachable and the trainer does not self-play.** No `MCTSEngine` caller exists outside tests. `training.py` samples uniform random parameters, simulates once, and uses the random bin as the policy target; the network is never used in `_expand` or `_evaluate`. "AlphaZero-style" describes an aspiration, not the code.
- **The objective is flat.** With defaults, reachable `dmax` spans 0.52 to 0.89 against a target of 2.0, so the dmax term contributes ~0.0006; 61% of the parameter box hits a clamp where coating and exposure have no effect; `ferric_oxalate_pct` computes a `contrast` that is never passed to either curve path (Δ = 0.0). Four physics constants have no consumer.
- **Axis and unit mismatches.** Simulated curves are sampled on linear relative exposure while step tablets are log-exposure; `coating_weight` is in units roughly 65× off from the chemistry module; `export.py` writes developer temperature into the ambient `temperature` field and simulated densities into `measured_densities`.
- **Tests do not constrain behaviour.** Simulator tests assert on intermediate dataclasses, never on the density curve; the engine "convergence" test asserts `quality > 0.1` when the floor is 0.41.
- Deliverables designed: 17 simulator/scorer metamorphic relations and 3 engine relations with tolerances; seed injection at 7 sites plus torch determinism; golden fixtures; a pre-registered ablation (random, Sobol, TPE, GP-UCB, differential evolution, grid; 30 seeds × 3 budgets × 3 targets; Wilcoxon + Cliff's δ; delete rule); a real-print holdout schema with a per-value `MeasurementEnvelope`, ≥ 40 runs over ≥ 4 paper batches, split by paper batch and chemistry lot; a `CalibrationParameters` model with physical-unit `NewType`s at the engine boundary.

### 3.2 Application and AI security (SEC-01 … SEC-22)

The full report is delivered to the owner out-of-band and intentionally not committed while the pre-authentication defects are unfixed. Fix-level summary:

| Priority | Items | What changes |
|---|---|---|
| Immediate | SEC-01, SEC-02, SEC-03, SEC-07, SEC-15 | Sanitize filenames and names on the two file-writing endpoints (uuid names, extension allowlist, streamed size cap, cleanup); global request-size middleware and `max_length` on list and string fields; explicit CORS origins with `allow_credentials` off by default; deploy only from tags with an environment gate, no `--force`, least-privilege `permissions:`, SHA-pinned actions, branch protection. |
| Phase 1 | SEC-04, SEC-05, SEC-13, SEC-14 | Image decode hardening (format allowlist, 40 MP pixel cap, header pre-check, multi-frame reject, downsample) and off-event-loop compute with timeouts (the MCTS training task calls `time.sleep` inside an `async def`, blocking the server for up to ~1000 s); `weights_only=True`/safetensors and no `pickle` for model artifacts; `.quad` parser caps and overflow handling. |
| Phase 3 | SEC-06, SEC-08, SEC-09, SEC-16, SEC-17, SEC-18 | Authentication and rate limiting (`rate_limit_per_minute` exists but is never read); `SecretStr` keys, SDK timeouts, removal of the process-global runtime key that the Gradio "Save API Key" button writes and nothing reads; per-IP quotas, message length cap, daily spend circuit breaker; lockfile, `pip-audit`, Dependabot, gitleaks, Ruff `S` rules; tenant isolation of shared in-memory state; error-detail hygiene and request IDs. |
| Phase 4 (before any agent exposure) | SEC-10, SEC-11, SEC-12, SEC-20, SEC-21, SEC-22 | Delimited data blocks and length caps on every user- or DB-derived prompt field; pydantic-validated tool arguments, wired tool timeouts, token budgets, confirmation for write tools; schema-validated agent memory and checkpoints; `schemathesis` and `promptfoo` (`owasp:llm`, `owasp:agentic`) in CI; `THREAT_MODEL.md` and `SECURITY.md`; fix documented env-var names (the prefix is `PTPD_LLM_`, not `PTPD_ANTHROPIC_`). |

### 3.3 Test architecture (TST-01 … TST-18)

- Real baseline: backend 4,036 / 3 / 148; frontend 826 / 5. The three backend failures are the `paper_stats` bug (two tests) and an unseeded symbolic-regression test. The five frontend failures are one genuine hook gap (`HTMLSelectElement` not guarded in keyboard shortcuts) and four tests that drifted from implementation.
- Exploratory Hypothesis probes found: `CurveModifier.smooth(SPLINE)` crashes for curves under 10 points; every `adjust_*` breaks monotonicity when endpoints are not (0, 1) because `preserve_endpoints` re-pins original values; `save_curve(.quad)` → `load_curve(.quad)` always fails ("No curve data found"); `.quad` export writes 16-bit values that the parser truncates to 8-bit.
- Quality audit: 18.4% of tests have only vacuous assertions, `tests/api` is 70% status-code-only, engine tests cannot detect a broken search. The recommended headline is a CI-generated per-suite table of collected / passed / skipped-by-reason / failed plus line and branch coverage per tier.
- The "equivalence" tests never call the React component, the API, or Gradio; one re-implements the math inline. Delete one, convert one to a unit test, replace the concept with shared contract goldens consumed by both pytest and vitest.
- Fully specified: strict pytest config (13 registered markers, `--strict-markers --strict-config`, `xfail_strict`, `timeout=120`, `filterwarnings=error` with a three-line justified ignore list derived from the 81 warnings actually observed, including two Pillow deprecations that Pillow 13 removes on 2026-10-15); mutation testing scope (8 files, 1,446 statements, mutmut 3, ratchet gate); torch-vs-NumPy differential in a dedicated `dl` job; MCTS-vs-exhaustive oracle on 2 dimensions × 5 bins; Playwright snapshot and axe policies (today no `toHaveScreenshot` call exists and `jest-axe` covers 1 of 15 components).

### 3.4 DevOps, CI, and supply chain (OPS-01 … OPS-18)

- Thirty concrete CI defects with `file:line` (D01–D30): step ordering, nonexistent extra, three Python versions with `.python-version` git-ignored, doubly masked lint, `|| true` e2e, an `if: always()` sanity gate that imports the legacy Gradio app as the deploy gate, deploy from any `claude/*` push with token in the remote URL, duplicate `push`+`pull_request` triggers, a `develop` trigger for a branch that does not exist, a nine-cell OS×Python matrix with `-x`, codecov without a token, no `permissions:`/`concurrency`/`timeout-minutes`, tag-pinned third-party actions, an `htmlcov/` upload that is never produced, and a 70% coverage gate against a measured 64%.
- Per PR push today: 33 job slots across four workflow runs (`ci-cd.yml` runs twice per commit), 1,684 runner-seconds while failing fast, an estimated 110–120 runner-minutes if healthy. Proposed: 9 jobs per PR, ~30 runner-minutes, ~12 minutes wall-clock, one required `all-green` check.
- Consolidated workflow written and validated (`ci-proposed.yml`): path-filtered backend (`uv sync --frozen`, `uv lock --check`, ruff check+format, mypy allowlist, pytest strict, per-package coverage floors, `diff-cover`) and frontend (pnpm frozen, tsc, eslint, prettier, vitest with the existing 80/75 thresholds, build), e2e against the real FastAPI factory, security (gitleaks on history, ruff `S`, `pip-audit` on the exported lock, `pnpm audit`), dependency review, CodeQL, weekly Scorecard, SBOM (syft SPDX over both lockfiles) with build-provenance and SBOM attestations on tags, GHCR release image, and a tag-only deploy job with environment approval, header-based auth, `--force-with-lease`, and a `workflow_dispatch(deploy_ref)` rollback. All 47 `uses:` are SHA-pinned to SHAs resolved via `git ls-remote` on 2026-09-18 (re-verify with `pinact` before merge). A dated ratchet table moves advisory checks to blocking over 30–90 days.
- uv migration designed: `[project]` with `python-dotenv`, `psutil`, `pyyaml`, `tifffile` declared; extras `plot, ml, llm, api, gcp, ui, hardware, qr, dl, server, all`; PEP 735 `[dependency-groups]`; torch from the CPU index on Linux; `.python-version = 3.12`; committed `uv.lock`; `pylock.toml` export; `requirements-dl.txt` deleted; `requirements.txt` generated transitionally then deleted. mypy allowlist of the nine zero-error packages as the blocking baseline.
- Docker Space designed (`Dockerfile.proposed`): node build → uv sync `--extra server` → slim runtime, UID 1000, port 7860, health check. Two code changes required: FastAPI must serve the SPA with a catch-all fallback (no static serving exists today), and the frontend must be built with `VITE_API_URL=/`. Recommended pattern: CI builds an attested image to GHCR and the Space's Dockerfile is a one-line `FROM ghcr.io/…@sha256:` so rollback is a digest change.
- Hygiene runbook: `git rm -r --cached`, `.pre-commit-config.yaml` (large files, ruff, uv-lock, gitleaks, actionlint, zizmor, prettier), `CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md` outlines, rulesets for `main` and `v*` tags, Dependabot or Renovate config; retire root npm/husky once pre-commit works.

### 3.5 Architecture, product, and documentation (ARC-01 … ARC-19)

- **Module inventory (31 packages):** CORE-KEEP 12 (core, curves, detection, imaging, chemistry, exposure, papers, session, analysis, zones, proofing, api); KEEP-BEHIND-EXTRA (llm, `ml/deep`, gcp, mcts, neuro_symbolic); QUARANTINE to `experimental/` or `contrib/` 11 (advanced, batch, calculations, data, education, integrations, monitoring, qa, vertex, workflow, deep_learning; agents pending the ablation); DELETE `ai` and, after the gaps close, `ui`. UV-exposure and drying-time calculators exist in five places; three persistence layers coexist (two SQLite, two JSON).
- **Import graph:** one strongly connected component `{core, curves, llm, ml}` via `core/models.py:205` → `curves/ai_enhance.py:247` → `llm/assistant.py:18` → `ml`; upward edges `mcts → agents` and `ml → gcp`. Six violations at twelve sites, all fixable in place; enforce with an `import-linter` contract `core → domain → adapters → intelligence → assistants → api`.
- **Gradio retirement:** 23 wired tabs plus two dead builders. Must-close before deleting `ui/`: Digital Negative (`POST /api/export/negative`, 16-bit TIFF/PNG), Image Preview, Chemistry API (delete the TypeScript formula), Session persistence, Auto-Linearization, Exposure, Papers, Wedge Analysis, curve compare/blend, histogram, About: ~17–19 engineer-days with the API-first pattern. Explicit drops: Batch, Soft Proofing v1, browser-side LLM-key Settings, Scanner Calibration (unwired even in Gradio), Neural Curves (unwired).
- **Docs:** delete 9 of 13 root files (four superseded roadmaps, three deep-learning write-ups, a 68 KB generic prompt template, one investigation summary), move `plan.md` to `docs/plans/2026-02-mcts-engine.md` and the gap analysis to `docs/archive/`, rewrite `CLAUDE.md` to ≤ 6 KB with correct paths (it cites `migration/`, `legacy/`, `ptpd_calibration.cli`, and `/api/export/*`, none of which exist). Delete all 23 `AGENT.md` files (referenced by nothing, not auto-loaded); keep one root `AGENTS.md` with the anti-sprawl rule block, enforced by a CI script.
- **KB protocol:** the SessionStart hook injects 36 KB on startup, resume, and compaction, mostly stale February state; a blocking Stop prompt hook coerces `kb/` writes into every PR; `.agent/rules/kb-protocol.md` describes a different protocol (for a second agent tool) with events that do not exist. Minimal replacement: opt-in, startup-only, ≤ 4 KB injection with a staleness banner; explicit `/handoff` skill instead of a blocking hook; JSON-schema-validated ledger entries carrying a commit and CI URL; generated summaries; `kb/` never in feature PRs.
- **PR dispositions:** #36 close and salvage (OpenAPI generation with one committed `schema.ts`, `server.py` type fixes, calculator de-duplication into `exposure/`); #34 close and cherry-pick the two commits that fix the five failing vitest tests; #13/#14/#16/#17 close as superseded by `frontend/`, `agents/`, `mcts/` already on the trunk.

---

## 4. Decisions (ADRs to write in Phase 0)

The expert reports each propose ADR lists with their own numbering; this table is the reconciled set.

| ADR | Decision | Status |
|---|---|---|
| 0001 | `main` is the trunk: make it the default branch, fast-forward it to `c9ef03a`, protect it with rulesets, delete the long-lived `claude/implement-chat-requirements-*` branch, retarget or close open PRs | proposed |
| 0002 | One CI workflow with one required `all-green` check; advisory steps ratchet to blocking on a dated schedule; deploy only from `v*` tags behind an environment approval | proposed |
| 0003 | uv with a committed `uv.lock` is the Python package manager; declared extras replace `requirements*.txt`; torch is an optional extra from the CPU index | proposed |
| 0004 | Retire Gradio after the §3.5 gap list is closed or explicitly dropped; FastAPI serves the built React app; the Space becomes a Docker Space built from an attested GHCR image | proposed |
| 0005 | MCTS, neuro_symbolic, and agents stay behind an extra and are gated by a pre-registered ablation; the engine is deleted unless it beats random search and TPE under equal simulation budgets | proposed |
| 0006 | Simulated and measured data carry a `provenance` field; nothing trains on simulated records by default | proposed |
| 0007 | README metrics are generated from CI artifacts, never typed | proposed |
| 0008 | Root markdown is limited to the canonical set; roadmap at `docs/roadmap.md`, decisions at `docs/adr/`, plans at `docs/plans/` and deleted when executed; no `AGENT.md` files | proposed |
| 0009 | Coverage policy: diff-coverage on changed lines (90%) plus tiered floors on the scientific core that ratchet (70/85/80 → 90/90/85) | proposed |
| 0010 | The agent framework stays unexposed until SEC-10/11/12/20 pass | proposed |
| 0011 | Quarantine policy: `experimental/` and `contrib/` are excluded from coverage and required CI; anything there for more than one release without an adopting ADR is deleted | proposed |
| 0012 | Layered import contract enforced by `import-linter`; no lazy imports to hide cycles | proposed |
| 0013 | API-first: all domain math (chemistry, exposure, linearization, imaging) lives in Python; the frontend consumes generated OpenAPI types and never re-implements formulas | proposed |
| 0014 | One persistence layer (SQLite via a single repository module) replaces the JSON calibration store, per-session JSON files, and the orphaned `data/` SQLite modules | proposed |
| 0015 | KB protocol minimized per §3.5; the blocking Stop hook is removed; `kb/` changes never ride in feature PRs | proposed |

---

## 5. Phased plan

Effort: S ≤ 1 day, M 2–5 days, L > 1 week. IDs refer to expert report tables.

### Phase 0 — Establish truth (weeks 1–2)

Goal: a protected trunk with one green CI run, an importable package, and a README that makes no claim CI cannot reproduce.

| ID | Item | Effort | Acceptance |
|---|---|---|---|
| ARC-01 / OPS-01 | ADR-0001: default branch → `main`, fast-forward to `c9ef03a`, rulesets for `main` and `v*`, read-only workflow permissions, `release`/`huggingface` environments; close #13/#14/#16/#17 with rationale | S | `default_branch: main`; `protected: true`; direct push rejected |
| OPS-08 (D08/D09 first) | Disable deploy for non-tag refs and make missing secrets fail loudly, then replace the three workflows with `ci-proposed.yml` at phase-1 settings | S then M | One workflow file; `all-green` passes on a no-op PR; ≤ 9 jobs per PR |
| OPS-03 / ARC-13 | ADR-0003 uv migration: declared hard deps, extras, `[dependency-groups]`, `.python-version`, `uv.lock`, `pylock.toml`; delete `requirements-dl.txt`; fix or drop the `ptpd` entry point (OPS-18) | M | `uv sync --frozen --extra server` then `import ptpd_calibration` succeeds in a clean container |
| TST-01 | Fix `paper_stats` NameError and narrow the swallowing `except`; seed the symbolic-regression test | S | 0 backend failures |
| TST-02 / OPS-06 / ARC-10 | Fix the five vitest failures (salvage #34 commits where still relevant); `packageManager` field; `.nvmrc`; close #34 | S | 831/831; frontend job green |
| OPS-05 / OPS-07 | Lint and format baseline to zero on `src tests app.py scripts`; mypy allowlist of the nine zero-error packages as the blocking set | M + S | `ruff check` and `ruff format --check` exit 0; `mypy` exit 0 on the allowlist |
| TST-03 / OPS-04 | Strict pytest config (markers, `--strict-markers --strict-config`, `xfail_strict`, timeout); coverage flags moved to CI; the five collection errors gated by markers | S | `pytest --collect-only -q` reports 0 errors and 0 warnings |
| OPS-02 / OPS-15 | Untrack `node_modules`, `.gradio`, `hf_check`; fix or delete the UTF-16 pre-push hook; `.pre-commit-config.yaml`; `CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md`; Dependabot | S | `git ls-files -i -c --exclude-standard` empty; `pre-commit run --all-files` clean |
| SEC-01 / SEC-02 / SEC-03 / SEC-07 | Path and name sanitization on both file-writing endpoints; request-size middleware and field bounds; CORS defaults | S each | Traversal filename → 400 with sentinel intact; > 50 MB → 413; foreign-origin preflight returns no allow-origin |
| ARC-14 / ARC-16 | Root doc purge to the canonical set; `CLAUDE.md` rewrite ≤ 6 KB; root `AGENTS.md` with the rule block and `scripts/check-doc-sprawl.sh` in CI; delete 23 `AGENT.md`; untrack `settings.local.json` | M | Root has exactly README, CONTRIBUTING, SECURITY, CHANGELOG, LICENSE, AGENTS.md, CLAUDE.md |
| ARC-17 | ADR-0015 KB protocol minimization: ≤ 4 KB opt-in startup injection, no Stop hook, schema-validated ledger, archived February summaries | S | Hook output ≤ 4096 bytes; `settings.json` has no `compact`/`resume` injection and no prompt Stop hook |
| TST-17 | README metrics generated from CI artifacts; remove "AlphaZero", "self-play", and "Expert Iteration" wording until SCI-07 lands | S | No hand-typed test count or coverage figure in README |

Exit criterion: one green run of the consolidated workflow on protected `main`.

### Phase 1 — Make the scientific core honest and measurable (weeks 3–5)

| ID | Item | Effort | Depends on |
|---|---|---|---|
| SCI-06 | Truthful API: `/api/mcts/search` either runs `MCTSEngine.search(target_curve=…)` or stops reporting `num_simulations`; `target_curve` provably consumed; `/feedback` persists or is removed | S | Phase 0 |
| SCI-08 | ADR-0006 `provenance` field on `CalibrationRecord`; `developer_temp_c` field; `export.py` stops writing into `temperature`/`measured_densities` | S | — |
| SCI-01 | Reachable objective: wire `contrast`, rescale coating/exposure constants so `dmax` spans at least [0.6, target + 0.3], clamp hit < 5%, remove `log(0)`, delete or wire the four dead constants | M | — |
| SCI-02 | Log-exposure axis for simulated curves; scorer compares on the same axis | S | — |
| SCI-05 | Seed injection at all 7 `random` sites plus torch; `PTPD_MCTS_SEED`; golden fixtures for 3 seeds | M | — |
| ARC-03 / ARC-04 | Break the four-package cycle and the two upward edges (twelve sites); ADR-0012 `import-linter` contract as a required check | S + S | — |
| ARC-02 | ADR-0011 quarantine: move the eleven orphan packages to `experimental/`/`contrib/`, delete `ai`, exclude from coverage and required CI | M | ARC-01 |
| TST-04 | `filterwarnings = error` migration (Pydantic `class Config`, Pillow `mode` before 2026-10-15, `log(0)`, import aliases) | M | TST-03 |
| TST-05 / 06 / 07 / 08 | Hypothesis property suites for `modifier`, `generator`, `linearization`, `.quad` round trip (fix `_load_text_curve`, keep 16-bit), `chemistry`, `quality`, `constraints`; fix the spline crash; fix or strict-xfail the endpoint-pinning defect | M + S + S + S | TST-03 |
| TST-09 / SCI-03 | Metamorphic suite for simulator, scorer, engine (MR-S1…S17, MR-E1…E3) with `xfail(strict=True)` on MR-S5/S17 until SCI-01; NumPy golden of 8 parameter sets × 21 steps | M | SCI-01, SCI-02 |
| TST-10 | `dl` CI job with CPU torch; torch-vs-NumPy differential at `atol=1e-4` with edge cases | M | TST-09 |
| SEC-04 / SEC-05 | Image decode hardening; CPU-bound work off the event loop with timeouts | M + M | — |
| SEC-13 / SEC-14 | Safe deserialization; `.quad` parser caps | M + S | — |
| TST-13 / OPS-11 | ADR-0009 coverage policy: branch on, tiered floors, `diff-cover`, stop omitting `api/server.py` | M | TST-03 |

Exit criterion: property, metamorphic, and golden suites green; simulator objective spread ≥ 0.3 over the box; no test can pass silently; import contract enforced.

### Phase 2 — Decide MCTS (weeks 6–7)

| ID | Item | Effort | Depends on |
|---|---|---|---|
| TST-11 | MCTS-vs-exhaustive oracle (2 dims × 5 bins), seeded golden, regret non-increasing in simulations | M | SCI-05, TST-09 |
| SCI-09 / ARC-19 | Pre-registered ablation: random, Sobol, TPE, GP-UCB, differential evolution, grid; budgets {100, 400, 1600}; 30 seeds; Wilcoxon + Cliff's δ; decision rule applied and recorded as the ADR-0005 outcome | M (~4 d) | SCI-01, SCI-02, SCI-05 |
| SCI-07 | Only if MCTS survives: real self-play (policy target = root visit counts, value target = search quality; network used in expansion and evaluation) and an "MCTS+network" ablation arm | L | SCI-09 |
| X-07 | If MCTS is deleted: remove `engine.py`, `tree.py`, `training.py`, `networks.py`, the search plumbing in `mcts_router`, the README section, and `neuro_symbolic`/`agents` if nothing else needs them; keep `simulator.py`, `quality.py`, `constraints.py`, `export.py` under a `calibration/optimize` name | M | SCI-09 |

Exit criterion: an ADR with the measured result and a codebase that matches it.

### Phase 3 — Ship one product (weeks 8–12)

| ID | Item | Effort | Depends on |
|---|---|---|---|
| ARC-05 | ADR-0004 Gradio retirement decision with the explicit drop list; Docker Space skeleton | S | — |
| ARC-06 | ADR-0013 API-first endpoints for chemistry, exposure, papers, linearize, wedge analysis, sessions with generated TypeScript types; delete the TypeScript chemistry formula | M | ARC-05 |
| ARC-07 | Digital negative and image preview endpoints (`POST /api/export/negative`, `POST /api/image/preview`) wired into `ExportPanel`/`ImagePreview`; 16-bit TIFF and PNG byte-compared against `imaging.processor` fixtures | L | ARC-05 |
| ARC-08 | Remaining UI gaps: curve compare, blend UI, presets, histogram, About; `/api/sessions` persistence | M | ARC-06 |
| ARC-18 | ADR-0014 persistence consolidation: one SQLite repository for calibrations, sessions, curves; migration script; `data/` deleted | M | ARC-02, ARC-06 |
| ARC-09 / OPS-17 | Delete Gradio: `ui/`, `app.py`, Gradio tests, `[ui]` extra, ruff/coverage exceptions, `.claude` deny rules, `requirements.txt`, `.gradio/`; retire equivalence tests (TST-15) | M | ARC-06, ARC-07, ARC-08 |
| OPS-13 / OPS-14 | FastAPI serves the SPA; `Dockerfile` and `.dockerignore`; README `sdk: docker`; Space secrets and variables; tag-only deploy with approval; thin `FROM ghcr.io/…@sha256` Space Dockerfile; rollback drill | M + M | OPS-03, OPS-06, OPS-08 |
| SEC-06 / SEC-08 / SEC-09 | Authentication and rate limiting; key hygiene and SDK timeouts; quotas and spend circuit breaker (required before the Space exposes chat with server-side keys) | M each | — |
| OPS-09 / OPS-10 / OPS-12 / SEC-16 | SHA re-verification and Dependabot; security jobs live (gitleaks, ruff `S`, `pip-audit`, `pnpm audit`, dependency review, CodeQL, Scorecard); SBOM and attestations on tags | S each | OPS-08 |
| SEC-17 / SEC-18 | Tenant isolation or explicit shared-demo mode; error hygiene and request IDs | M + S | SEC-06 |
| ARC-11 | Close #36; salvage OpenAPI generation, `server.py` type fixes, calculator de-duplication into `exposure/` as ≤ 4 small PRs | M | ARC-01, ARC-02 |
| TST-12 | Mutation testing nightly with ratchet on the 8-file scientific scope | M | TST-05…08 |
| TST-14 | `schemathesis` against the ASGI app with 5xx = failure; API tests assert payload values against Python goldens; un-skip the 13 chat tests via `AsyncClient` | M | TST-01 |
| TST-15 / TST-16 | Contract goldens shared by pytest and vitest; Playwright snapshot policy; axe gating on 5 routes and 15 components | S + M | TST-07, TST-02 |
| OPS-16 | Phase-2 ratchet: e2e, dependency review, CodeQL into the gate; audits and ruff `S` blocking; floors 80/90/85/70; mypy +9 packages | M | +30–60 days |
| X-08 | Tag `v1.0.0` with SBOM and provenance; CHANGELOG entry generated from merged PRs; ADR-0001…0015 written (ARC-15) | S | all above |

Exit criterion: the public artifact is the FastAPI+React app, deployed from a tag, with attestations, and it can export a digital negative.

### Phase 4 — Ground truth and agent evaluation (week 12 onward)

| ID | Item | Effort | Depends on |
|---|---|---|---|
| SCI-10 | Real-print holdout dataset: pydantic schema with `MeasurementEnvelope`; ingestion CLI; ≥ 12 initial runs (one batch, replicates), growing to ≥ 40 runs over ≥ 4 paper batches; `pytest -m realdata` report of sim-to-real RMSE, dmax error, rank correlation | L | SCI-02, SCI-08 |
| SCI-11 | `fit_physics_constants(records)` with covariance; leave-one-batch-out RMSE; versioned constants tied to a dataset hash | L | SCI-10, SCI-01 |
| SCI-12 | Physical-unit `NewType`s and a `CalibrationParameters` model replacing `dict[str, float]` at the engine boundary; `mypy --strict` on `mcts/`, `core/models.py`, `ml/predictor.py`, `chemistry/` | M | SCI-08 |
| SEC-10 / SEC-11 / SEC-12 / SEC-20 | Prompt-injection defenses; agent tool-argument validation, budgets, confirmations; memory and checkpoint schemas; `promptfoo` red-team (`owasp:llm`, `owasp:agentic`) as a nightly gate | M each | SEC-06/09 |
| X-09 | First sim-to-real validation report with uncertainty bands, published under `docs/validation/` | M | SCI-10, SCI-11 |
| X-10 | Re-run the SCI-09 ablation against real prints; revisit ADR-0005 | M | SCI-10 |

Exit criterion: a recommendation shown to a user carries a provenance flag and an uncertainty band derived from measured prints.

### Explicitly deferred

MCTS dashboard with live visualization, Celery/Redis queue, PWA offline mode, i18n, bounded-context package split until Phase 2 decides which packages survive, and any exposure of the ReAct agent framework.

---

## 6. Sequencing and critical path

```
Week 1-2   ARC-01/OPS-01  OPS-08  OPS-03  TST-01  TST-02  OPS-05/07  TST-03  OPS-02/15  SEC-01..03,07  ARC-14/16/17  TST-17
Week 3-5   SCI-06  SCI-08  SCI-01  SCI-02  SCI-05  ARC-02/03/04  TST-04  TST-05..10  SEC-04/05/13/14  TST-13
Week 6-7   TST-11  SCI-09 -> ADR-0005 -> (SCI-07 | X-07)
Week 8-12  ARC-05..09  ARC-18  OPS-13/14  SEC-06/08/09  OPS-09/10/12  SEC-17/18  ARC-11  TST-12  TST-14  TST-15/16  OPS-16  X-08
Week 12+   SCI-10  SCI-11  SCI-12  SEC-10/11/12/20  X-09  X-10
```

Critical path: ARC-01 → OPS-08 → OPS-03 → TST-03 → TST-09 → SCI-09 → ADR-0005 → ARC-07 → OPS-14. Everything in Phase 1 that does not touch `mcts/` can proceed in parallel with SCI-01/02/05; the Gradio gap closure (ARC-06/07/08) can start in Phase 1 if a second engineer is available, since it does not depend on the ablation.

---

## 7. Governance rules to adopt now

1. Trunk-based development on protected `main`; branches live less than a week; no PR over ~400 changed lines (excluding generated and lock files) without a design note.
2. No PR merges red. "Advisory" steps have a dated ratchet: mypy error count and mutation score may only fall or rise respectively.
3. No metric appears in `README.md` unless a script produced it from the latest `main` CI artifact.
4. No new root-level markdown beyond the canonical set; roadmap changes go to `docs/roadmap.md`, decisions to `docs/adr/`, plans to `docs/plans/` and are deleted when executed.
5. Agent working state (`kb/`, `.agent/`) never rides in a feature PR; until ADR-0015 lands, any session that touches `kb/` must keep `tests/unit/test_kb_protocol.py` green (this plan's own session tripped it once).
6. Simulated data never enters a field named `measured_*`; records carry `provenance`.
7. Security findings are fixed before their details are committed to the public repository.
8. Domain math lives in Python only; the frontend consumes generated types.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Fast-forwarding `main` surprises anyone with a stale checkout | `main..trunk` is empty so nothing is lost; announce in CHANGELOG |
| PR #36 conflicts with Phase 0 hygiene | Close #36; cherry-pick the OpenAPI type generation and calculation de-duplication as small PRs after Phase 0 |
| The ablation shows MCTS is not better | That is a valid, publishable result; X-07 removes ~3,000 lines and possibly 12,000 more |
| Holdout dataset needs darkroom time the owner may not have | Phase 4 is designed to be useful at 12 runs; the first 12 are enough to reject the current physics on dmax alone |
| Retiring Gradio drops features users rely on | §3.5 enumerates the gaps; each is closed or explicitly dropped in ADR-0004 before deletion |
| A public Space with server-side LLM keys is being billed | The deploy job has been a no-op since November 2025 (secrets unset), so the Space runs the old Gradio build; verify (§9, Q4) and do not set secrets until SEC-06/09 land |
| Pillow 13 (2026-10-15) breaks image processing | TST-04 fixes the two `mode` deprecations in Phase 1 |
| Quarantining 47k LOC removes capabilities someone wanted | ADR-0011 gives one release to adopt anything quarantined; nothing is deleted without an ADR |

---

## 9. Decisions needed from the owner

1. **Trunk.** Confirm ADR-0001 (make `main` default and fast-forward it to `c9ef03a`). Alternative: rename the current default branch to `main`.
2. **PR #36 and #34.** Close and salvage as proposed, or attempt to land. The plan assumes close-and-salvage.
3. **Operating model.** Single-operator darkroom tool plus portfolio artifact (plan as written) versus multi-user service (pull SEC-06/17 and a job runtime forward into Phase 1).
4. **Hugging Face Space.** Confirm no `PTPD_LLM_*` secrets are set on the Space; confirm the Space is public. The plan assumes both.
5. **Module retention.** Approve or amend the keep/quarantine/delete matrix in `expert-architecture.md` §1, in particular whether `deep_learning/` (13,819 LOC, no wired importer) and `integrations/` (8,675 LOC, hardware drivers) are worth adopting.
6. **MCTS deletion rule.** Approve the pre-registered thresholds in `expert-science.md` §6 before the ablation runs, so the result cannot be argued after the fact.
7. **Python floor.** Approve `requires-python >= 3.11` (3.10 reaches end of life in October 2026).
8. **KB protocol.** Approve ADR-0015 (remove the blocking Stop hook, cap injection, delete `.agent/` and the 23 `AGENT.md` files).

---

## Appendices (in `docs/plans/2026-09-review/`)

- `verification-matrix.md` — the peer-review verification with evidence, including live CI logs from PR #37
- `expert-science.md` — scientific validation review, SCI-01…12
- `expert-testing.md` — test architecture review, TST-01…18
- `expert-devops.md`, `ci-proposed.yml`, `Dockerfile.proposed` — CI and supply-chain review, OPS-01…18
- `expert-architecture.md` — architecture, product, and documentation review, ARC-01…19
- Security review (SEC-01…22) — delivered out-of-band; summarized in §3.2
