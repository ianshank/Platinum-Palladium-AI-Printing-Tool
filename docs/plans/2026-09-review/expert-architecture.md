# Expert Report: Software Architecture, Product & Documentation

Repository: /home/user/Platinum-Palladium-AI-Printing-Tool (checkout at c9ef03a, 2026-03-01)
Date: 2026-09-18. Read-only analysis. Supporting data: `scratchpad/depgraph.py`, `scratchpad/depgraph.json`.

Legend: **[V]** = verified by running a command / reading the file in this session. **[I]** = inferred. **[G]** = given as already-verified by the coordinator.

## 0. Headline findings that change the plan

1. **`main` is not the trunk. [V]** `origin/main` = 43537f1 (2026-02-08). The branch `claude/implement-chat-requirements-014YeEiyBSMJL91puKEgVcih` is **134 commits ahead of main and 0 behind**, and its first-parent history contains the merges of PRs #19–#35 (`git log --first-parent origin/main..<branch>`). The 4 "stale-base" PRs (#13/#14/#16/#17) target the branch that is actually the live integration line; **`main` is the stale ref**. PR #36 and #34 are "554 files / 529 files" only because they are measured against the abandoned `main`; their real deltas against the trunk are 81 files (#36) and 87 files (#34).
2. **`deep_learning/` (13,819 LOC, 15% of the backend) is unreachable from every wired entry point. [V]** `api/deep_learning.py:441-443` imports `ptpd_calibration.ml.deep.*`, not `deep_learning`. The only importer is `ui/tabs/neural_curve.py`, which is not imported by `gradio_app.py` (the tab is never built). CI (`ci-cd.yml`) already `--ignore`s its tests.
3. **About 52% of backend LOC has no importer other than `examples/` or an unwired tab. [V]** Orphans (fan-in = nobody): advanced, ai, batch, calculations, data, education, integrations, monitoring, qa, vertex, workflow (25,978 LOC) + deep_learning (13,819) + ui (7,077) = 46,874 of 91,731 LOC.
4. **The React app + API does not cover the product's core deliverable. [V]** No API endpoint applies a curve to an image or exports a digital negative (`imaging/` is imported only by `ui/`); CLAUDE.md's `/api/export/*` endpoints do not exist in `api/server.py`. Chemistry is re-implemented client-side with a different formula (`frontend/src/stores/slices/chemistrySlice.ts:193-213` vs `chemistry/calculator.py:283`). Session log is in-memory in React (`stores/index.ts:53` persists UI prefs only) while Gradio persists JSON files (`session/logger.py:343`).
5. **The KB/agent scaffolding injects ~36 KB of Feb-2026 state on every start, resume and compaction [V]**, includes claims that are false today ("726 passed, 0 failed", "Ready for PR"), and three mutually inconsistent protocol specs (`CLAUDE.md`, `.agent/rules/kb-protocol.md`, PR #36's ledger vocabulary).
6. **Current test reality [V]:** frontend vitest on trunk: 826 passed / 5 failed / 831 (files: `uiSlice.test.ts`, `Layout.test.tsx`, `CurveEditor.test.tsx`, `useKeyboardShortcuts.test.ts`). Backend: 4,522 tests collected; 5 collection errors here are environmental (venv lacks `gradio`, `psutil`). Last CI on main: 5/5 failures (Feb 7–8); on trunk `ci-cd.yml`: failure on the last 2 pushes (#33 Feb 23, #35 Mar 1).

---

## 1. Module inventory (31 packages)

Reachability columns: **API** = imported (directly or transitively) by `api/server.py`, `api/mcts_router.py`, `api/deep_learning.py`; **React** = an endpoint the frontend calls (`frontend/src/api/{client,hooks,mcts}.ts`) leads to it; **Gradio-only** = imported by `ui/` and by nothing else. Test-file counts: [G] for the 11 packages the coordinator supplied; the rest are `grep -rlE "ptpd_calibration\.<pkg>([. ]|$)" tests/` [V] (slightly stricter than the coordinator's grep; differences are ±1).

| Package | Purpose (from `__init__` docstring) | LOC | Test files | API | React | Gradio-only | Optional heavy deps | Recommendation |
|---|---|---|---|---|---|---|---|---|
| core | Core data models and types | 2,076 | 36 | yes | yes | no | psutil (guarded, `core/debug.py:375`, **undeclared**) | **CORE-KEEP** |
| curves | Curve generation, export, modification | 4,378 | 15 | yes | yes | no | matplotlib (plots only) | **CORE-KEEP** |
| detection | Step tablet detection and extraction | 1,390 | 6 | yes (`/api/scan/upload`) | yes | `scanner.py` only via unwired tab | none | **CORE-KEEP** |
| imaging | Digital negative creation and curve application | 2,533 | 10 | **no** | **no** | yes (Image Preview, Digital Negative tabs) | tifffile, matplotlib | **CORE-KEEP** — must gain API endpoints (ARC-07) |
| chemistry | Chemistry calculation module | 1,787 | 8 | **no** | no (TS re-implementation) | yes (`ui/tabs/chemistry.py`) | none | **CORE-KEEP** — expose via API, delete TS formula (ARC-08) |
| exposure | Exposure calculators | 1,170 | 10 | no | no | yes (Exposure Calculator tab) | none | **CORE-KEEP** — expose via API |
| papers | Paper profiles database | 810 | 4 | no | no | yes (Paper Profiles tab) | none | **CORE-KEEP** — expose via API |
| session | Print session logging | 510 | 5 | **no** | no (React log is in-memory) | yes | none | **CORE-KEEP** — expose via API |
| analysis | Wedge analysis | 783 | 6 | no | no | yes (Step Wedge Analysis tab) | none | **CORE-KEEP** — expose via API or fold into `detection` |
| zones | Zone System mapping | 429 | 5 | no | no | yes | none | **CORE-KEEP** (small; API or explicit drop) |
| proofing | Soft proofing simulation | 373 | 6 | no | no | yes | none | **CORE-KEEP** (small; API or explicit drop) |
| api | FastAPI web API | 1,895 | 7 | root | root | no | fastapi; torch check guarded (`api/deep_learning.py:17`) | **CORE-KEEP** |
| llm | LLM integration | 863 | 10 | yes (`/api/chat*`, `/api/curves/enhance`) | yes | no | anthropic, openai, google (guarded) | **KEEP-BEHIND-EXTRA** `[llm]` (already) |
| ml | ML prediction/refinement (+ `ml/deep`) | 5,427 | 9 | yes (`CalibrationDatabase`; `/api/deep` uses `ml.deep`) | partly (`/api/deep/*` has no frontend hook [V]) | no | sklearn; torch guarded (`ml/deep/models.py:23`) | **CORE-KEEP** for `database.py`/`predictor.py`; `ml/deep` **KEEP-BEHIND-EXTRA** `[torch]` |
| gcp | Google Cloud Platform integration | 290 | 6 | indirect: `ml/database.py:13` imports `gcp.storage.StorageBackend` | indirect | no | google-cloud-storage (guarded `gcp/storage.py:9`) | **KEEP-BEHIND-EXTRA** `[gcp]`; move `StorageBackend` protocol into `core` so `ml` stops importing `gcp` |
| mcts | Monte Carlo Tree Search | 5,161 | 16 | yes (`/api/mcts/*`) | yes (`MCTSPage`) | no | torch optional, numpy fallback (`mcts/simulator.py:18-20`) | **KEEP-BEHIND-EXTRA** `[mcts]` pending ablation (ADR-0003) |
| neuro_symbolic | Neuro-symbolic constraints | 3,578 | 5 | indirect via `mcts/constraints.py:31` | indirect | no | scipy | **KEEP-BEHIND-EXTRA** with mcts; falls with it |
| agents | Agentic system for autonomous assistance | 8,521 | 12 | indirect only via `mcts/agents.py:13-14` (`agents.logging`, `agents.subagents.base`) | only via `/api/mcts` | no | psutil (guarded `agents/health.py:277`) | **QUARANTINE** (`experimental/agents`) unless ADR-0003 keeps `mcts.agents`; sever the `mcts → agents` edge (2 small imports) |
| deep_learning | Deep Learning module | 13,819 | 7 | **no** (`api/deep_learning.py` uses `ml.deep`) | no | only via **unwired** `ui/tabs/neural_curve.py` | torch, torchvision, diffusers, timm, ultralytics, segment_anything, clip, lpips, cv2, skimage, controlnet_aux, anthropic, openai | **QUARANTINE** (`experimental/deep_learning`), excluded from coverage/CI; DELETE after one release unless an ADR adopts it. Evidence: zero wired importers; `ci-cd.yml` already ignores `tests/unit/test_deep_learning*`; `requirements-dl.txt` is a separate install. |
| integrations | Hardware/services integrations (spectro, printers, weather, ICC) | 8,675 | 15 | no | no | no — **orphan** (`examples/integrations_demo.py` only) | cups, serial, win32api/print, zeroconf, httpx | **QUARANTINE** (`contrib/hardware`). Evidence: fan-in = nobody; `ci-cd.yml` ignores `tests/unit/test_integrations.py`; Windows/CUPS drivers cannot run in CI. |
| vertex | Vertex AI integration | 2,154 | 3 | no | no | orphan | google, vertexai | **QUARANTINE** (`contrib/gcp`) → DELETE if unused for a release. Evidence: fan-in = nobody; duplicates `llm` + `gcp`. |
| data | Data management (sqlite PrintDatabase, boto3 cloud_sync, version_control, export_import) | 2,924 | 4 | no | no | orphan | boto3, yaml | **QUARANTINE** then consolidate persistence (ADR-0006). Evidence: third persistence layer beside `ml.database` (JSON) and `session.logger` (JSON files); fan-in = nobody. |
| education | Educational components (tips, glossary, tutorials) | 3,820 | 2 | no | no | orphan (`examples/education_demo.py`) | pydantic | **QUARANTINE** → DELETE; move text content to `docs/`. Evidence: no importer, 2 test files, ruff per-file-ignores admit "tutorial code". |
| advanced | Style transfer, negative blender, QR metadata, print comparison | 1,771 | 4 | no | no | orphan | pyzbar, qrcode, scipy | **QUARANTINE**. Evidence: no importer; features absent from both UIs. |
| qa | Quality assurance | 1,750 | 4 | no | no | orphan (`examples/qa_example.py`) | none | **QUARANTINE**. Also duplicates `calculate_drying_time`/exposure adjust (see §1.1). |
| monitoring | Performance monitoring/profiling | 1,480 | 4 | no | no | orphan | psutil (**unguarded** `monitoring/performance.py:21`, undeclared), pynvml | **QUARANTINE** → DELETE. Evidence: `ci-cd.yml` ignores `tests/unit/test_performance_monitoring.py`; undeclared dep breaks collection. |
| calculations | Enhanced calculations | 1,474 | 2 | no | no | orphan (`examples/enhanced_calculations_demo.py`) | pydantic | **QUARANTINE** → DELETE after de-duplicating into `exposure`/`chemistry` (§1.1). |
| ai | "Comprehensive AI system" (`platinum_palladium_ai.py`) | 1,388 | 2 | no | no | orphan, no example | PIL, numpy | **DELETE**. Evidence: single-file, no importer, no example, overlaps `llm`/`agents`/`ml`. |
| workflow | Workflow automation & recipe management | 1,308 | 4 | no | no | orphan (`examples/recipe_management_demo.py`) | yaml | **QUARANTINE**. |
| batch | Batch processing | 354 | 2 | no | no | orphan (Gradio Batch tab does **not** import it [V]) | none | **QUARANTINE**; re-home into `imaging` if a batch API is ever built. |
| ui | Gradio UI (4,337-line `gradio_app.py` + 6 tab modules + `manage_rag.py`) | 7,077 | 10 | root | — | root | gradio, gradio_client, torch (via neural_curve) | **DELETE** after ADR-0001 (§3). `ui/manage_rag.py` has no importer at all [V] — delete immediately. |

### 1.1 Duplication evidence (supports the QUARANTINE/DELETE calls) [V]
- UV exposure / drying-time calculators exist in 5 places: `calculations/enhanced.py:312,1294`, `qa/quality_assurance.py:851,1044`, `integrations/weather.py:198`, `vertex/agents.py:260`, `deep_learning/uv_exposure.py:628` (plus `imaging/split_grade.py:862`). The canonical home is `exposure/`.
- Persistence: `data/database.py:86` (sqlite `PrintDatabase`), `data/repository.py:25` (sqlite), `ml/database.py:16` (JSON `CalibrationDatabase`), `session/logger.py:343` (JSON per session).
- Pyproject extras: `ml, llm, api, gcp, ui, all, dev` — no `torch`/`mcts`/`hardware` extra; `psutil` and `python-dotenv` (used at `config.py:14`) are undeclared (`ci-cd.yml` installs psutil by hand). `[project.scripts] ptpd = "ptpd_calibration.cli:main"` points at a module that does not exist [V].
- 268 files under root `node_modules/` are committed to git (`git ls-files node_modules | wc -l`) [V].

---

## 2. Import dependency graph and layering violations

Method [V]: AST walk of every `.py` under `src/ptpd_calibration/<pkg>/`, resolving absolute and relative imports to top-level package (`scratchpad/depgraph.py`; output in `depgraph.json`). Lazy (function-level) imports are included, and flagged below where relevant.

### 2.1 Edges (importer → importees)
```
agents         -> core, curves, llm, ml
ai             -> chemistry, core, exposure, imaging
analysis       -> core, curves, detection
api            -> core, curves, detection, llm, mcts, ml
batch          -> core, imaging
chemistry      -> core
core           -> curves            (lazy: core/models.py:205 CurveData.save)
curves         -> core, llm         (lazy: curves/ai_enhance.py:247)
data           -> core
deep_learning  -> detection         (lazy: deep_learning/detection.py:865)
detection      -> core
exposure       -> core
imaging        -> core
integrations   -> core
llm            -> core, ml          (llm/assistant.py:18)
mcts           -> agents, core, ml, neuro_symbolic
ml             -> core, gcp         (ml/database.py:13)
neuro_symbolic -> core, curves
ui             -> analysis, chemistry, core, curves, deep_learning, detection, exposure, imaging, llm, papers, proofing, session, zones
vertex         -> core, curves, detection
workflow       -> core
```
Fan-in = nobody: advanced, ai, api, batch, calculations, data, education, integrations, monitoring, qa, ui, vertex, workflow.

### 2.2 Cycles [V]
- Strongly connected component: **{core, curves, llm, ml}**: `core → curves` (models.py:205) → `llm` (ai_enhance.py:247) → `ml` (assistant.py:18) → `core`.
- Mutual import: **core ↔ curves** (`core/models.py:205` ↔ `curves/linearization.py:14`).

### 2.3 Violations of the proposed layering
Layers: L0 core → L1 (curves, chemistry, detection, imaging, exposure, zones, papers) → L2 (ml, mcts, deep_learning, neuro_symbolic) → L3 (agents, llm) → L4 api → L5 ui.

| # | Importer → importee | Direction | Sites | Fix |
|---|---|---|---|---|
| V1 | core → curves | UPWARD (L0→L1), forms cycle | `core/models.py:205` (1) | Remove `CurveData.save()` convenience; callers use `curves.save_curve(curve, path)`. |
| V2 | curves → llm | UPWARD (L1→L3), forms cycle | `curves/ai_enhance.py:247` (1) | Move `CurveAIEnhancer`'s LLM branch to `llm/curve_enhance.py` (or inject an `Assistant` protocol). |
| V3 | mcts → agents | UPWARD (L2→L3) | `mcts/agents.py:13,14` (2) | Move `get_agent_logger` to `core.logging`; move `SubagentMessage/Result` base to `core` or make `mcts.agents` an optional adapter under `agents/`. |
| V4 | ml → gcp | L2 → infra adapter (gcp unassigned) | `ml/database.py:13` (1) | Define `StorageBackend` protocol in `core.storage`; `gcp.storage` implements it; inject. |
| V5 | agents → llm | LATERAL (L3) | `agents/agent.py:15`, `agents/subagents/base.py:21`, `agents/health.py:199`, +1 (4) | Acceptable if L3 is defined as "assistants" with llm below agents; recommend split L3a llm → L3b agents. |
| V6 | mcts → ml, mcts → neuro_symbolic | LATERAL (L2) | `mcts/simulator.py:46` (lazy), `mcts/constraints.py:31` | Acceptable if L2 is ordered ml/neuro_symbolic → mcts. Codify. |
| — | deep_learning → detection, llm → ml, api → *, ui → * | downward | — | allowed |

Unassigned packages (15) and proposed placement: analysis, proofing, session, batch → L1 domain; gcp, data, monitoring, integrations → "adapters" layer that may import only core; vertex → L3 (LLM provider adapter); ai, education, qa, workflow, advanced, calculations → `experimental/` (outside the contract). Enforce with `import-linter` layered contract (ARC-04).

---

## 3. Gradio retirement assessment

### 3.1 What ships today [V]
- `app.py` is Gradio-only, monkey-patches `gradio.blocks.Blocks.get_config_file` (schema regression workaround), and imports `ptpd_calibration.ui.gradio_app`. README front-matter: `sdk: gradio`, `sdk_version: "4.44.0"`. `ci-cd.yml` verifies `create_gradio_app` imports and force-pushes to the HF Space on every push to `main`/`claude/*`.
- Gradio builds 5 top-level groups with 23 wired tabs (`gradio_app.py:4278-4318`). Two builders are dead: `build_scanner_calibration_tab` (312 LOC, defined, never called) and `ui/tabs/neural_curve.py` (1,230 LOC, never imported). `ui/manage_rag.py` (105 LOC) has no importer.
- React: 8 routes (`frontend/src/App.tsx:82-89`): `/`, `/calibration`, `/curves`, `/chemistry`, `/assistant`, `/mcts`, `/session`, `/settings`. API endpoints used by the frontend: health, analyze, scan/upload, curves/{generate,modify,smooth,enhance,upload,parse,{id},{id}/export,{id}/enforce}, calibrations, chat/{,recipe,troubleshoot}, statistics, mcts/*.
- The "equivalence tests" (`frontend/src/__tests__/equivalence/*.ts`) never call Gradio or the backend; they unit-test a tolerance comparator against static fixtures [V]. Deleting `ui/` does not affect them; they should be renamed (they do not verify equivalence).

### 3.2 Tab-by-tab gap table (Gradio LOC from builder boundaries [V])

| Gradio tab (LOC) | Backend pkg | React / API today | Gap | Disposition | Effort |
|---|---|---|---|---|---|
| Dashboard (159) | session | `DashboardPage` + `/api/statistics` | none | keep | — |
| Calibration Wizard (691) | detection, curves | `CalibrationWizard` 5 steps + `/api/scan/upload`, `/api/curves/generate` | none material | keep | — |
| Step Tablet Reader (66) | detection | `ScanUpload` + `/api/scan/upload` | none | keep | — |
| Step Wedge Analysis (332) | analysis | none; no endpoint | quality metrics/density curve of a wedge | close: `POST /api/analysis/wedge` + panel in Calibration page | M (2d) |
| Curve Display (281) | curves | `CurvesPage` upload/edit/export | multi-curve compare + stats table | close: compare tab (client-side, uses `/api/curves/{id}`) | S (1d) |
| Curve Generator (75) | curves | wizard Step 3/4 | none | keep | — |
| Curve Editor (870) | curves | `CurveEditor` + modify/smooth/enhance/parse/upload/export | `blend` endpoint exists (`server.py:566`) but no frontend hook [V] | close: blend UI | S (0.5d) |
| AI Curve Enhancement / Quick tools (64) | llm | `AIAssistant` + `/api/chat/recipe`, `/troubleshoot` | none | keep | — |
| Image Preview (176) | imaging | `ImagePreview` component (no imaging endpoint) | server-side curve application not exposed | close: `POST /api/image/preview` (returns PNG) | M (2d) |
| Digital Negative (233) | imaging | **none**; `/api/export/negative` in CLAUDE.md does not exist | **product-critical**: invert + apply curve + 16-bit TIFF/PNG export | close: `POST /api/export/negative` + `ExportPanel` wiring | L (3–4d incl. 16-bit tests) |
| Interactive Editor (300) | curves | `CurveEditor` control points | presets | close: presets in `CurveEditor` | S (0.5d) |
| Batch Processing (160) | (inline imaging) | none | batch queue | **drop** for v1 (ADR: no Celery/Redis); revisit as client loop | 0 |
| Histogram Analysis (90) | (inline) | `imageSlice.histogram` type only | zone-banded histogram | close client-side (canvas) | S (1d) |
| Zone System (138) | zones | none | zone mapping visual | close: `POST /api/image/zones` or drop | S–M (1.5d) or 0 |
| Soft Proofing (172) | proofing | none | paper simulation preview | **drop** for v1 unless validated (VALIDATION.md); package stays | 0 (or M) |
| Chemistry Calculator (194) | chemistry | client-side re-implementation with **different formula** | equivalence broken (area×coating factor, 60/40 split, Na2 = level×0.5 vs Python drops/in², Na2 ratio 25%) | close: `POST /api/chemistry/calculate`; delete TS formula; generated types | S–M (1.5d) |
| Exposure Calculator (183) | exposure | none | UV exposure/time calc | close: `POST /api/exposure/calculate` + small page | S–M (1.5d) |
| Paper Profiles (121) | papers | none | browse/select paper profile | close: `GET /api/papers`, use in wizard | S (1d) |
| Print Session Log (98) | session | `SessionLogPage` in Zustand memory only | data lost on reload; Gradio persists JSON | close: `/api/sessions` CRUD backed by `session.logger` | M (2d) |
| Settings (188) | config | `SettingsPage` (UI prefs) | runtime LLM provider/API key entry in browser | **drop**: API keys via server env only (THREAT_MODEL) | 0 |
| Auto-Linearization (178) | curves.linearization | none | generate linearization curve from measurements | close: `POST /api/curves/linearize` + wizard step option | S–M (1.5d) |
| About (174) | — | none | static | close: markdown page | S (0.25d) |
| Scanner Calibration (312, unwired) | detection.scanner | — | dead in Gradio too | drop | 0 |
| Neural Curves (1,230, unwired) | deep_learning | — | dead | drop with quarantine | 0 |

**Must-close before deleting `ui/`:** Digital Negative, Image Preview, Chemistry API, Session API, Auto-Linearization, Exposure, Papers, Wedge Analysis, Curve compare/blend, Histogram, About ≈ **17–19 engineer-days** [I] with the API-first pattern (endpoint + pydantic schema + generated TS type + hook + panel + test). **Explicit drops (document in ADR-0001):** Batch, Soft Proofing (v1), Settings LLM-key entry, Scanner Calibration, Neural Curves.

### 3.3 Deletion set once gaps are closed [V]
`src/ptpd_calibration/ui/**`, `app.py`, README front-matter (`sdk: gradio`…), `docs/huggingface-deployment.md` (duplicate front-matter), `tests/unit/ui/*` (8 files), `tests/unit/test_neural_curve_tab.py`, `tests/e2e/test_user_journeys.py`, `tests/e2e/selenium/**`, `tests/visual/**`, `tests/sanity/test_deployment_sanity.py`, `ci-cd.yml` "Verify app.py can be loaded" step and HF force-push, pyproject `[ui]` extra + `ruff` per-file-ignores for `ui/**` + coverage `omit` for `*/ui/*`, `.claude/settings.json` deny rules for `ui/**`, `.gradio/`, `frontend` scripts `migrate:*` (they point at a non-existent `migration/` dir). Replace with a `Dockerfile` (FastAPI serving `frontend/dist`) and `sdk: docker` in the Space README.

---

## 4. Docs consolidation

### 4.1 Root markdown disposition (13 files) [V for dates/sizes]

| File (size, last commit) | Nature | Disposition | Target |
|---|---|---|---|
| README.md (6.6 KB, HF front-matter) | product README + Space config | **keep-at-root**, shorten; replace "84% coverage / 726 tests / 15/15" prose with CI badges; drop `sdk: gradio` when ADR-0001 lands | README.md |
| CHANGELOG.md (5.6 KB, 2026-02-22) | Keep-a-Changelog but only "Unreleased" entries | **keep-at-root**; cut a `v1.0.0` tag so entries stop accumulating under Unreleased | CHANGELOG.md |
| CLAUDE.md (18.9 KB) | agent instructions; contradictory | **keep-at-root but rewrite ≤ 6 KB**; move KB protocol to `docs/agents/kb-protocol.md`; fix wrong paths (`migration/`→`docs/migration/`, non-existent `legacy/` symlink, non-existent `ptpd_calibration.cli`, non-existent `/api/export/*`) | CLAUDE.md |
| plan.md (35 KB, 2026-02-16) | MCTS/AlphaZero 11-phase implementation plan (executed by PR #31) | **move-to-docs** as historical context for ADR-0003 | `docs/plans/2026-02-mcts-engine.md` |
| ANALYSIS_GAPS_AND_AI_INTEGRATION.md (44 KB, 2025-12-07; header says "December 2024") | market/gap analysis, 12-month AI roadmap | **move-to-docs/archive**; extract still-valid items into `docs/roadmap.md` | `docs/archive/2025-12-gap-analysis.md` |
| AGENTIC_NEXT_STEPS.md (15.8 KB, 2026-02-01) | roadmap v1 | **delete** (superseded by V3) | — |
| NEXT_STEPS_AGENTIC_DEVELOPMENT.md (17.3 KB, 2026-02-01) | roadmap v2, same day | **delete** (superseded) | — |
| INVESTIGATION_SUMMARY.md (15.8 KB, 2026-02-02) | investigation of the two above | **delete** (superseded by V3) | — |
| AGENTIC_NEXT_STEPS_V3.md (13.9 KB, 2026-02-08) | consolidated roadmap; notes tracker was stale (0/15 vs 12/15) | **merge-into** `docs/roadmap.md`, then delete | `docs/roadmap.md` |
| DEEP_LEARNING_IMPLEMENTATION.md (18.4 KB, 2025-12-07) | DL features 1–2 docs | **merge-into** `docs/deep_learning_features.md` (exists, 16 KB) or move with the package to `experimental/deep_learning/README.md`; delete | `experimental/deep_learning/README.md` |
| IMPLEMENTATION_SUMMARY.md (10.3 KB, 2025-12-07) | DL features 3–4 summary | same as above; delete | same |
| QUICK_REFERENCE.md (6.4 KB, 2025-12-07) | DL quick reference | same as above; delete | same |
| IMPLEMENTATION_TEMPLATE.md (67.8 KB, 2026-01-25) | generic "prompting as constraint programming" template with a project C4 copy | **delete** (prompt template, not project documentation); if the C4 section has value it is already in `docs/architecture.md` | — |

### 4.2 Contradictions found [V]
| Topic | Claims |
|---|---|
| Migration status | CLAUDE.md: 12/15, ~75% · README: 15/15, 84% backend coverage · `docs/migration/progress.json`: 15/15, coverage 80 · `docs/migration/config.json`: "Phase 1: Project Setup", coverage current 0 · V3 (Feb 8): 12/15 |
| Frontend test count | 592 (V3) · 680/720 (progress.json changelog) · 726 (CHANGELOG, kb summaries, README) · 812/812 (PR #34 body) · **actual on trunk: 831 (826 pass, 5 fail)** |
| Backend coverage | README 84% · CLAUDE.md ~75% · PR #36 self-report 15% · `ci.yml` gate 70% (blocking) · `tests.yml` gate 80% (`continue-on-error`) · pyproject omits `ui/*` and `api/server.py` from coverage |
| CI | three workflows: `ci.yml` (py3.12, ruff+mypy blocking, cov 70) · `tests.yml` (3.10–3.12 × 3 OS, installs extra `.[test]` which does not exist in pyproject) · `ci-cd.yml` (py3.11, lint advisory, HF deploy). Last 5 `ci.yml` runs on main: failure. |
| Paths/commands | CLAUDE.md `migration/` (real: `docs/migration/`), `legacy/` symlink (absent), `python -m ptpd_calibration.cli` and `ptpd` script (module absent), `/api/export/{negative,curve,profile}` (absent), `pnpm migrate:verify` → `migration/equivalence-tests` (absent; real tests in `src/__tests__/equivalence`) |
| Phase naming | CLAUDE.md "Phase 3 – Migration Completion & Hardening" vs `.claude/settings.json` env `MIGRATION_PHASE` same vs `config.json` Phase 1 vs plan.md Phases 1–11 (MCTS) vs PR #36 "Phases 1-2" (hygiene) |

### 4.3 Canonical doc set
- `README.md` (≤ 150 lines): what it is, 3-command quick start (`uv sync`, `uv run ptpd-server`, `pnpm dev`), links.
- `CONTRIBUTING.md`: branching (§6), required checks, verification loop, how to add an endpoint (API-first), where experimental code lives, doc rules (§4.4).
- `SECURITY.md`: reporting, no secrets in browser, upload limits (`/api/curves/upload-quad` path noted as unsafe by the coordinator).
- `docs/ARCHITECTURE.md`: existing C4 (`docs/architecture.md`, 2026-02-23) updated: mark Gradio container "retired", add layering contract (§2.3), adapters, experimental/ boundary.
- `docs/VALIDATION.md`: sim-to-real protocol — reference step tablet, densitometer readings, acceptance tolerances (ΔD ≤ 0.02, curve RMSE), fixture provenance, ablation procedure for MCTS vs. baseline linearization.
- `docs/THREAT_MODEL.md`: assets (API keys, uploaded scans, session data), trust boundaries (browser ↔ API ↔ LLM providers ↔ HF Space), file upload handling, LLM prompt injection via chat.
- `docs/roadmap.md`: single living roadmap (from V3 + gap analysis survivors).
- `docs/adr/` — first eight:
  - ADR-0001 Retire the Gradio UI; ship FastAPI + React as a Docker Hugging Face Space; explicit drop list (§3.2).
  - ADR-0002 `uv` as the package manager; one `uv.lock`; extras `api, llm, ml, torch, gcp, hardware`; declare `psutil`, `python-dotenv`; remove `[ui]` and `requirements*.txt`.
  - ADR-0003 MCTS + neuro-symbolic stay behind `[mcts]` extra until an ablation (VALIDATION.md) shows benefit over baseline linearization; sunset date and removal criteria.
  - ADR-0004 Quarantine policy: `experimental/` and `contrib/` trees are excluded from coverage and required CI; anything there for > 1 release without an adopting ADR is deleted.
  - ADR-0005 Layered import contract enforced by `import-linter` (core → domain → adapters/optimization → assistants → api); no lazy imports to hide cycles.
  - ADR-0006 One persistence layer (SQLite via a single repository module); retire `data.PrintDatabase`, `ml.CalibrationDatabase` JSON store and per-session JSON files.
  - ADR-0007 API-first: all domain math (chemistry, exposure, linearization, imaging) lives in Python; the frontend consumes generated OpenAPI types and never re-implements formulas.
  - ADR-0008 Trunk-based development on `main` with one CI workflow and required checks; the long-lived `claude/implement-chat-requirements-*` branch is fast-forwarded into `main` and deleted.

### 4.4 Proposed rule text (AGENTS.md and CLAUDE.md, identical section)
```
## Documentation and file-placement rules (enforced by CI `scripts/check-doc-sprawl.sh`)
1. The repository root may contain only: README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, LICENSE,
   AGENTS.md, CLAUDE.md. Any other *.md at root fails CI.
2. Never create files named *_SUMMARY.md, *_NEXT_STEPS*.md, *_IMPLEMENTATION*.md, *_GUIDE.md, plan.md,
   INVESTIGATION*.md, QUICK_REFERENCE.md, or dated status reports anywhere in the repo.
   Status goes in the PR description; decisions go in docs/adr/NNNN-title.md; plans go in
   docs/plans/YYYY-MM-title.md and are deleted when executed; the roadmap is docs/roadmap.md only.
3. Numeric claims (test counts, coverage %, migration %) are not written into markdown. Link the CI run.
4. Per-directory agent guidance: do not create AGENT.md files. Claude Code loads CLAUDE.md only;
   at most three nested CLAUDE.md files are allowed (frontend/, src/ptpd_calibration/, tests/), each ≤ 1.5 KB,
   containing only commands and hard constraints, never architecture prose (that lives in docs/ARCHITECTURE.md).
5. Agent working state (kb/, .agent/) is never committed in a feature PR. Ledger changes are squashed into one
   event at merge by the maintainer.
6. Deleting a stale doc never requires an ADR; adding a root file does.
```
**AGENT.md policy [V]:** the 23 `AGENT.md` files (47.6 KB, PR #35) are referenced by nothing in `CLAUDE.md`, `.claude/`, `.agent/` or `README.md`, and Claude Code does not auto-load that filename; they are inert for this toolchain and already drift (e.g. `.claude/AGENT.md` says `settings.local.json` is "not committed" while it is tracked). Delete all 23; keep one root `AGENTS.md` (≤ 4 KB, the rule block above + pointers) that `CLAUDE.md` `@`-includes.

---

## 5. Knowledge-base protocol assessment (kb/, .agent/, .claude/)

### 5.1 Injection volume [V — measured by running the hooks]
| Condition | Bytes injected at SessionStart |
|---|---|
| `CLAUDE_ROLE` unset (default `all`) — the configured case | **36,156** |
| planning | 6,847 |
| dev-sqe | 11,567 |
| pre-pr | 17,686 |
`.claude/settings.json` runs `kb-start.sh` on `startup`, `resume` **and `compact`** matchers, so the 36 KB (≈9k tokens) is re-injected after every compaction, on top of CLAUDE.md (18.9 KB ≈ 4.7k tokens). Roughly 14k tokens of mostly stale context per session start [I on token math].

### 5.2 Stale / conflicting state [V]
- `kb/sessions/dev-sqe.state.json` and `pre-pr.state.json`: frozen at 2026-02-22 (branch `claude/compassionate-pike`, commit 5ac3a6e), asserting "726 passed, 0 failed", "typecheck_errors: 0", "pr_readiness: ready". Today: 5 failing frontend tests, CI red on the last two trunk pushes. The pre-pr hook injects the **dev-sqe state in full** as well.
- `kb/summaries/dev-sqe.md` (8 KB) and `pre-pr.md` are Feb-22/23 narratives ("All checks passing… Ready for PR"). `planning.md` and `design-contract-readiness.md` are empty.
- `kb/sessions/planning.state.json`: was null-valued before today; now overwritten by the current session's parent (2026-09-18) with `last_planning_doc: null`, `last_adr: null`.
- Ledger: 8 events (7 up to 2026-02-22 + 1 appended 2026-09-18). PR #36's branch appends **5 more** with a different vocabulary (`PRE-PR-HANDOFF`, `DEV-SQE-COMPLETION`, `tasks_defined[]`, `workflow_id`) dated 2026-08-03; they never reached the trunk. Merging #36 would make two branches' append-only logs collide.
- Three inconsistent protocol specs: (a) `CLAUDE.md` + `.claude/skills/*` (events PLANNING-HANDOFF / DEV-SQE-HANDOFF / DESIGN-GAP / QA-GAP / SCOPE-CHANGE); (b) `.agent/rules/kb-protocol.md` (events PLAN-LOCKED / DEV-PROGRESS / DEV-COMPLETE, `kb/specs/`, `kb/ledger/lock.dir` — none of these exist in the repo); (c) `kb/AGENT.md`. `.agent/workflows/handoff.md` reads `~/.gemini/antigravity/brain/*` and PR #34's body cites `C:\Users\iansh\.gemini\antigravity\brain\...` — a second agent tool (Antigravity) writes the same KB under protocol (b).
- `.claude/settings.json` `Stop` hook is a *prompt* hook that returns `decision: block` until handoff files are written, so every session that did "meaningful work" is coerced into mutating `kb/` — this is how stale narratives accumulate and why feature PRs carry `kb/` churn (#34: 8 kb files; #36: 12).
- `.claude/settings.local.json` is committed (Windows paths, a full commit message in an allow-rule) although `.claude/AGENT.md` says it is never committed; `settings.json` pins `claude-sonnet-4-20250514`; `.claude/agents|commands|skills` reference `migration/…` paths that do not exist.

### 5.3 Poisoning risk
High for any agent that trusts the injected text: it will believe tests pass, that the branch is PR-ready, that the trunk is `claude/compassionate-pike`, that `migration/` exists, and (under protocol b) that a `PR-READY` last event means "start Planning". Injection on `compact` means the false state is re-asserted mid-task.

### 5.4 Minimal, consistent protocol (recommendation)
1. One protocol, one owner: delete `.agent/` (or move to `docs/agents/antigravity/` un-loaded) and the 23 `AGENT.md`; the protocol lives in `docs/agents/kb-protocol.md`, referenced (not inlined) from CLAUDE.md.
2. SessionStart hook: opt-in (`CLAUDE_ROLE` must be set), `startup` only (never `compact`/`resume`), hard cap 4 KB total; inject only (a) last 5 ledger lines, (b) the role's state JSON ≤ 1 KB, (c) *paths* of pending handoff docs. Print a `STALE` banner when `state.git_commit` is not an ancestor of HEAD.
3. Remove the blocking `Stop` prompt hook; handoff is an explicit `/handoff` skill.
4. Ledger schema: 5 event types, JSON-schema validated by `scripts/verify-kb-protocol.sh` in CI; entries carry `commit`, `ci_run_url`; numeric claims without a CI URL are rejected.
5. Summaries: delete hand-written summaries; `kb/summaries/*.md` become generated (`git log --since` + last CI status) and are git-ignored. Archive today's `dev-sqe.md`/`pre-pr.md` to `kb/archive/2026-02/`.
6. `kb/` changes never ride in feature PRs (CI check: PR touching `kb/` must be labelled `kb-only`).

---

## 6. PR strategy

### 6.1 Verified branch topology [V]
- `origin/main` 43537f1 (2026-02-08). Trunk `claude/implement-chat-requirements-014YeEiyBSMJL91puKEgVcih` = HEAD c9ef03a (2026-03-01), 134 ahead / 0 behind main; contains merges of #19, #20, #21, #22, #23, #26, #29, #30, #31, #32, #33, #35.
- CI on trunk (`ci-cd.yml`, push to `claude/*`): #35 failure, #33 failure, #32 success, #31 failure, #30/#29/#26 cancelled, #23/#22/#21 failure. `ci.yml` never runs on the trunk (triggers on `main`/`feature/**` pushes and PRs to `main` only).

### 6.2 Disposition
| PR | Facts [V] | Disposition | Rationale / salvage |
|---|---|---|---|
| **#36** draft "Code hygiene and modularity (Phases 1-2)" (Aug 3, base main) | vs main: 140 commits/554 files; **vs trunk: 6 commits / 81 files / +26,809 −1,394**. Adds `openapi.json` three times (3,546 lines each: root, `src/.../api/`, `frontend/src/api/`), generated `schema.ts`, 7 docs guides, `calculations/core.py` (791), Gradio UI modularization (`ui/` 6 files), two more root MDs (`REFACTORING_GUIDE.md`, `API_SCHEMA_GENERATION.md`), kb ledger/sessions/handoffs. Self-reported: 823 type errors, 15% coverage, 9 TS build errors. `mergeable_state: unstable`. | **Close.** Salvage as ≤ 4 small PRs onto trunk: (1) OpenAPI export script + `pnpm generate:types` wiring, committing **one** generated `schema.ts` (or generating in CI); (2) `api/server.py` type fixes (commit 1f6dbee) if they pass mypy; (3) calculation de-dup — but into `exposure/`/`chemistry/`, not a new `calculations/core.py`; (4) `docs/TESTING.md` + `ERROR_HANDLING.md` content folded into CONTRIBUTING.md. | Drop the Gradio modularization (ui is being retired), drop all `kb/` and root-MD additions, drop duplicate openapi copies. |
| **#34** "stabilize frontend test suite and refactor headings" (Feb 24, base main) | vs main 118 commits; **vs its trunk merge-base (Feb 16): 3 commits / 87 files / +1,140 −550**; **31 commits behind trunk**; merging now would delete 35,568 lines; `mergeable_state: dirty`. Claims 812/812 tests; trunk today 826/831. | **Close; cherry-pick-salvage** commits 90e2c94 (headings/test stabilization) and ba84f70 (bugs 1–17) onto a fresh branch, resolve against trunk, keep only what still fails (`uiSlice` sidebar default, `Layout` toggle, `CurveEditor` save, `useKeyboardShortcuts` select). | Overlaps with #32's bug-fix sprint already merged. |
| **#17** "agentic coding template system" (Jan 17, base = trunk) | +9,885 / 26 files; generic config/logging/error/agent/health scaffolding; nothing references it. | **Close.** No salvage (health endpoints idea → issue). | Duplicates `core/` + `agents/`; template not product. |
| **#16** "AlchemistZero AlphaZero calibration" (Jan 10, base = trunk) | +4,814 / 21 files; dirty. | **Close** as superseded by `mcts/` (merged via #31, executed `plan.md`). | Two MCTS implementations would compete. |
| **#14** "agent system phases 1-2 router/skills" (Dec 15, base = trunk) | +12,459 / 28 files; dirty. | **Close** as superseded by the `agents/` package on trunk (8,521 LOC). | If ADR-0003 quarantines agents, nothing to salvage. |
| **#13** "React frontend + testing infrastructure" (Dec 15, base = trunk) | +10,585 / 77 files; dirty. | **Close** as superseded by `frontend/` on trunk (15/15). | — |

### 6.3 Branching model going forward
1. **Fast-forward `main` to the trunk tip** (`git push origin c9ef03a:main`; safe: `integration..main` = 0 commits), then delete `claude/implement-chat-requirements-*` and the four stale heads. Update HF deploy to run only from `main` tags.
2. **Trunk-based**: branches ≤ 3 days, target ≤ 400 changed lines (excluding generated/lock files), one concern per PR, squash-merge, linear history, no long-lived integration branches, no bot `kb/` churn.
3. **One workflow** (`ci.yml`, delete `ci-cd.yml` and `tests.yml`): `uv sync --extra api --extra ml --extra llm`; `ruff check` + `ruff format --check`; `mypy` on a listed set of core packages (blocking) and advisory elsewhere; `pytest tests/unit tests/api` excluding `experimental/`/`contrib/`; coverage gate 70% on core packages; `pnpm typecheck && pnpm lint && pnpm test:run && pnpm build`; `import-linter`; `scripts/check-doc-sprawl.sh`; `scripts/verify-kb-protocol.sh`.
4. **Branch protection on `main`**: require PR, the checks above, up-to-date branch, linear history; deploy to HF on `v*` tags only.

---

## 7. Bounded-context refactor (ptpd_domain / ptpd_simulation / ptpd_optimization)

**Recommendation: not now. Fix the graph in place, quarantine, run the MCTS ablation, then decide on a sub-package split (not separate distributions).**

Why [V + I]:
- The layering problems are small and local: 6 violations at 12 import sites (§2.3), one 4-node cycle created by two lazy imports. `import-linter` + four ~20-line changes remove them without moving files.
- The candidate `ptpd_simulation` (mcts.simulator, ml.deep.process_sim, neuro_symbolic) and `ptpd_optimization` (mcts, agents) contents are exactly what ADR-0003's ablation may delete: 5,161 + 3,578 + 8,521 = 17,260 LOC. Moving them first is work that may be thrown away; moving them after quarantine of the 46.9 kLOC orphan set leaves a ~45 kLOC core where package-level boundaries are already legible.
- There is no second consumer (no CLI — `ptpd_calibration.cli` does not exist — no plugin, no separate service) that would benefit from separate distributions; a single wheel with sub-packages and an import contract gives the same one-way guarantee with none of the release overhead.

Sequencing:
1. **Now (Sprint 1–2):** ARC-02/03/04 (quarantine, fix cycles, import-linter contract with layers `core → domain(curves, chemistry, detection, imaging, exposure, zones, papers, analysis, session, proofing) → adapters(gcp, storage) → intelligence(ml, neuro_symbolic, mcts) → assistants(llm, agents) → api`).
2. **After ADR-0001 ships (Sprint 3–4):** Gradio deletion; API-first endpoints; persistence consolidation (ADR-0006).
3. **After the ablation (ADR-0003, ~1 quarter):** if MCTS survives, introduce sub-packages `ptpd_calibration.simulation` (process_sim + mcts.simulator + constraints) and `ptpd_calibration.optimization` (mcts engine/tree/training + whatever of `agents` it still needs) with contract `domain ← simulation ← optimization`; if it does not, delete `mcts/`, `neuro_symbolic/`, `agents/` and the question is moot. Promote to separate distributions only when an external consumer appears.

---

## Plan items

| ID | Title | Effort | Dependencies | Acceptance criteria |
|---|---|---|---|---|
| ARC-01 | Fast-forward `main` to trunk c9ef03a; delete long-lived `claude/implement-chat-requirements-*` and stale heads; close #13/#14/#16/#17; enable branch protection | S | none | `origin/main == c9ef03a`; 4 PRs closed with rationale comments; `main` requires PR + checks + linear history; no open PR has a non-`main` base |
| ARC-02 | Quarantine orphan packages: move advanced, ai(delete), batch, calculations, data, education, integrations, monitoring, qa, vertex, workflow, deep_learning to `experimental/` or `contrib/`; exclude from coverage/CI; write ADR-0004 | M | ARC-01 | `src/ptpd_calibration/` contains only CORE/EXTRA packages from §1; `pytest` core suite green without torch/psutil/google; coverage `source` excludes quarantined trees; ADR-0004 merged with sunset dates |
| ARC-03 | Break the core↔curves↔llm↔ml cycle and the mcts→agents / ml→gcp edges (V1–V4) | S | none | `python scratchpad/depgraph.py` reports no SCC > 1 and no UPWARD edges; `StorageBackend` protocol lives in `core`; `CurveData.save` removed or delegated without importing `curves` |
| ARC-04 | Add `import-linter` layered contract (ADR-0005) as a required CI check | S | ARC-03 | `lint-imports` passes on `main`; contract file lists layers from §7 step 1; CI fails on a deliberately added upward import |
| ARC-05 | ADR-0001 Gradio retirement decision + explicit drop list; Docker HF Space skeleton (`Dockerfile` serving `frontend/dist` from FastAPI, `sdk: docker`) | S | none | ADR merged; Space builds from Docker on a test branch; drop list (Batch, Soft Proofing v1, Settings key entry, Scanner Cal, Neural Curves) recorded |
| ARC-06 | API-first endpoints for chemistry, exposure, papers, linearize, wedge analysis, sessions (ADR-0007) with generated TS types | M | ARC-05 | 6 endpoints with pydantic schemas + tests; `pnpm generate:types` produces `schema.ts` from `/openapi.json`; frontend hooks call them; TS chemistry formula deleted; chemistry API output equals `chemistry.calculator` for 5 fixtures |
| ARC-07 | Digital negative + image preview endpoints (`POST /api/export/negative`, `POST /api/image/preview`) wired into `ExportPanel`/`ImagePreview` | L | ARC-05 | 16-bit TIFF and PNG exports byte-compared against `imaging.processor` fixtures; upload size/type limits enforced; e2e test downloads a negative |
| ARC-08 | Close remaining UI gaps: curve compare tab, blend UI, presets, histogram, About; add `/api/sessions` persistence to `SessionLogPage` | M | ARC-06 | Each §3.2 "close" row has a page/panel + test; session log survives reload; frontend vitest 0 failures |
| ARC-09 | Delete Gradio: `ui/`, `app.py`, Gradio tests (§3.3), `[ui]` extra, ruff/coverage exceptions, `.claude` deny rules; README front-matter → docker | M | ARC-06, ARC-07, ARC-08 | `grep -r gradio src tests pyproject.toml README.md` returns nothing; Space deploys the React app; backend LOC ≤ 45k |
| ARC-10 | Fix the 5 failing frontend tests (salvage #34 commits 90e2c94/ba84f70 where still relevant); close #34 | S | ARC-01 | `pnpm test:run` 0 failed on `main`; #34 closed with link to the salvage PR |
| ARC-11 | Close #36; salvage OpenAPI generation (one committed `schema.ts`), `server.py` type fixes, calculator de-dup into `exposure/` | M | ARC-01, ARC-02 | ≤ 4 PRs each < 800 lines; no `openapi.json` duplicates; mypy on `api/` passes; the 5 duplicate UV/drying implementations reduced to one in `exposure/` |
| ARC-12 | Single CI workflow with required checks; delete `ci-cd.yml`, `tests.yml`; deploy on tags only | S | ARC-01 | Only `ci.yml` exists; it is green on `main`; required-checks list matches §6.3 step 3; HF deploy job runs only on `v*` tags |
| ARC-13 | `uv` migration (ADR-0002): `uv.lock`, extras `api, llm, ml, torch, gcp, hardware`, declare `psutil`/`python-dotenv`, remove `requirements*.txt`, fix/remove `ptpd` script entry, untrack root `node_modules/` | S | none | `uv sync` reproduces CI env; `git ls-files node_modules` = 0; `ptpd-server` runs; no undeclared imports (`deptry` clean) |
| ARC-14 | Root doc purge + canonical set: delete/move per §4.1; write CONTRIBUTING.md, SECURITY.md, docs/roadmap.md; rename `docs/architecture.md` → `docs/ARCHITECTURE.md` and update; add `scripts/check-doc-sprawl.sh` to CI | M | none | Root has exactly README, CONTRIBUTING, SECURITY, CHANGELOG, LICENSE, AGENTS.md, CLAUDE.md; sprawl check green; ARCHITECTURE.md shows retired Gradio container + layers |
| ARC-15 | Write ADR-0001…0008 (`docs/adr/`), `docs/VALIDATION.md` (sim-to-real + MCTS ablation protocol), `docs/THREAT_MODEL.md` | M | ARC-14 | 8 ADRs with status/decision/consequences; VALIDATION.md defines fixtures, tolerances, ablation metric; THREAT_MODEL.md covers upload, API keys, LLM injection |
| ARC-16 | Rewrite CLAUDE.md (≤ 6 KB, no numbers, correct paths) + root AGENTS.md with the §4.4 rule block; delete 23 AGENT.md; delete `.agent/`; untrack `.claude/settings.local.json`; unpin stale model | S | ARC-14 | `find . -name AGENT.md` empty; CLAUDE.md ≤ 6 KB and contains no path that does not exist (script check); `.agent/` absent |
| ARC-17 | KB protocol minimisation (§5.4): opt-in 4 KB startup hook, no compact/resume injection, remove blocking Stop hook, ledger JSON-schema check in CI, archive Feb-2026 summaries/state | S | ARC-16 | `CLAUDE_ROLE=all bash .claude/hooks/kb-start.sh | wc -c` ≤ 4096; `settings.json` has no `compact`/`resume` SessionStart and no prompt `Stop` hook; `verify-kb-protocol.sh` required in CI; stale files under `kb/archive/` |
| ARC-18 | ADR-0006 persistence consolidation: one SQLite repository for calibrations, sessions, curves; migrate `ml.CalibrationDatabase` JSON and `session` JSON files | M | ARC-02, ARC-06 | One `storage/` module; `/api/calibrations` and `/api/sessions` backed by it; migration script with tests; `data/` deleted |
| ARC-19 | MCTS ablation (ADR-0003): baseline linearization vs MCTS on VALIDATION.md fixtures; decide keep/delete for mcts, neuro_symbolic, agents; then (only if kept) sub-package split per §7 step 3 | L | ARC-15, ARC-18 | Ablation report with metric and decision recorded in ADR-0003; either `mcts/`+`neuro_symbolic/`+`agents/` deleted, or `simulation/`/`optimization/` sub-packages exist with a passing import contract `domain ← simulation ← optimization` |
