# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-09-18

### Changed (Phase 0: establish truth — see docs/plans/2026-09-validation-sdlc-plan.md)

- **One CI workflow**: `.github/workflows/ci.yml` replaces `ci.yml`, `ci-cd.yml`, and `tests.yml`; single required check `all-green`; SHA-pinned actions; least-privilege permissions; deploy only from `v*` tags behind an environment approval and failing loudly on missing secrets (ADR-0002)
- **uv with a committed lockfile**: `uv.lock`, `pylock.toml`, `.python-version`; every unconditional import is now a declared dependency (matplotlib, tifffile, python-dotenv, psutil, pyyaml); extras reorganised; `requirements-dl.txt` removed; `requirements.txt` is generated (ADR-0003)
- **Strict pytest configuration**: registered markers, `--strict-markers --strict-config`, `xfail_strict`, per-test timeout; coverage flags moved to CI (ADR-0009)
- **Documentation consolidated**: nine root roadmap/summary files deleted or archived, `CLAUDE.md` rewritten under 6 KB, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/roadmap.md`, fifteen ADRs under `docs/adr/`; 23 `AGENT.md` files removed; placement rules enforced by `scripts/check-doc-sprawl.sh` (ADR-0008)
- **Metrics are generated**: `scripts/test_metrics.py` renders test and coverage numbers from CI artifacts into the job summary; the README no longer carries typed counts (ADR-0007)
- **Knowledge-base protocol minimised**: capped startup-only digest with staleness banner, non-blocking Stop hook, February 2026 summaries archived, protocol in `docs/agents/kb-protocol.md` (ADR-0015)
- **Repository hygiene**: root `node_modules/`, `.gradio/certificate.pem`, `hf_check/`, root npm/husky tooling and `.claude/settings.local.json` untracked; `.pre-commit-config.yaml`, `CODEOWNERS`, Dependabot added

### Fixed

- `session/logger.py`: `get_paper_statistics` referenced an undefined name and silently counted one record per session
- Five stale or incorrect frontend tests; `useKeyboardShortcuts` now ignores `<select>` elements
- Pydantic v2 and Pillow deprecation warnings that would break under `filterwarnings = error`

### Security

- Client-supplied filenames and names are no longer used in server paths on `/api/curves/upload-quad` and `/api/curves/export`; uploads are size-capped and extension-allowlisted; request bodies and list/string fields are bounded; CORS no longer combines `*` with credentials (SEC-01/02/03/07)

## [Unreleased] — 2026-02-22

### Fixed (Bug-Fix Sprint)

- **P0 — CurveEditor HTTP 400**: `adjustment_type` was `'none'` on every curve-modify call; changed to `'brightness'` so the API accepts the request
- **Sidebar flash on first load**: `uiSlice` `sidebarOpen` default was `true`; changed to `false` so the sidebar starts closed on mobile
- **E2E test — duplicate `h1`**: Dashboard heading check scoped to `<main>` to avoid matching the duplicate heading in the Layout header
- **E2E test — focus ring detection**: Added `body.click()` to establish page focus before `Tab`; switched selector to `:focus-visible` for spec-correct matching
- **Backend `analyze_densities` crash**: Added empty-list guard before `np.array([]).max()` — now returns HTTP 422 instead of 500
- **Backend `create_calibration` 500**: Wrapped `CalibrationRecord(...)` construction in `try/except pydantic.ValidationError` — now returns HTTP 422
- **Backend `get_calibration` 500**: `UUID(calibration_id)` `ValueError` now caught and returned as HTTP 422 with descriptive message
- **Backend `smooth_curve` 400**: `CurveModifier.smooth()` does not accept `preserve_endpoints` as a kwarg — moved to `CurveModifier.__init__()`
- **Backend quad validation false-positive**: `parse_quad_content` now checks `not profile.raw_sections and not profile.active_channels`; previous check on `profile.channels` was always `False` because `_post_process()` unconditionally injects 7 default disabled channels
- **`CurveType` enum missing values**: Added `SPLINE = "spline"` and `POLYNOMIAL = "polynomial"` to resolve `ValueError` in callers using those strings

### Test Results (2026-02-22)

- **Frontend Playwright e2e**: 9/9 passed ✅
- **Backend pytest (`tests/api/`)**: 104 passed, 13 skipped, 0 failed ✅
- **Frontend vitest**: 726 passed, 0 failed ✅
- **TypeScript**: 0 errors ✅

---

## [Unreleased] — 2026-02-16

### Added

- **AlphaZero-Style MCTS Calibration Engine** (`src/ptpd_calibration/mcts/`, 5,181 LOC)
  - `MCTSEngine`: Monte Carlo Tree Search with UCB1/PUCT exploration for optimal printing parameters
  - `DualNetwork`: PyTorch policy + value network guiding search via Expert Iteration
  - `ExtendedProcessSimulator`: Physics-based Pt/Pd process model (sensitizer diffusion, UV exposure, humidity)
  - `QualityScorer`: Multi-metric print quality evaluation (Dmax, tonal range, linearity, smoothness)
  - `ActionPruner` + `ConstraintChecker`: Photochemistry-aware parameter validation
  - `MCTSTrainer` + `ReplayBuffer`: Self-play training pipeline with prioritized experience replay
  - `MCTSResultExporter`: Export optimized parameters to JSON, CSV, and QTR-compatible formats
  - Domain subagents: `ChemistrySubagent`, `ExposureSubagent`, `CalibrationCoordinatorSubagent`
- **Frontend MCTS Integration**
  - Zustand `mctsSlice` for search state management
  - TanStack Query hooks and Axios client methods for MCTS endpoints
  - TypeScript type definitions for all MCTS data structures
- **Frontend Quad File Upload**
  - `CurveUpload` component with drag-and-drop and paste modes
  - `useUploadQuadFile` / `useParseQuadContent` mutation hooks
  - API client methods `uploadQuad()` / `parseQuad()` with FormData handling
  - Type definitions: `QuadUploadResponse`, `QuadParseResponse`, `QuadChannel`, `QUAD_CHANNELS`

### Fixed

- **NumPy RNG enum bug** in `data_generators.py`: `numpy.random.default_rng().choice()` on Python enums returns `numpy.str_` not enum members — fixed 5 call sites
- **Physics simulator torch/numpy mix**: Replaced `torch.clamp` with `numpy.clip` in `ExtendedProcessSimulator`
- **Agent communication deadlock**: Fixed `asyncio.wait_for` timeout handling in `MessageBus.request()`
- **Agent persistence type error**: Fixed `AgentMemory.save()` argument type
- **141 test issues resolved**: Deep learning data generators (22), vertex integration env vars (3), neural curve field names (1), dashboard HTML wrapping (1), async mocking (52), TS errors (21), FE test failures (20), BE import errors (21)
- Added `generate_batch()` methods to 4 data generator classes
- Added `monkeypatch.delenv()` isolation for LLM API key tests
- Added `pytest.mark.skipif` for Windows-incompatible shell hook tests

### Test Results

- **Frontend**: 792 passed, 0 failed
- **Backend**: 4,423 passed, 0 failed, 112 skipped
- **Build**: ~280 KB gzipped (under 500 KB target)

---

## [Previous] — Frontend Migration Phase 2

### Added (Phase 2)

- **Frontend Migration (Phase 2)**
  - Added `ScanUpload` component with drag-and-drop and progress tracking.
  - Added `CurveEditor` component using Recharts and Radix UI.
  - Added `CalibrationWizard` component for multi-step calibration workflow.
  - Added `CalibrationPage` and `CurvesPage` for routing.
  - Updated `App.tsx` to include new routes.
  - Added comprehensive unit tests for all new components.
  - Added E2E tests for the full calibration journey using Playwright.
- **Type Safety**
  - synchronized frontend types with backend Pydantic models in `frontend/src/types/models.ts`.
  - Updated API client with strong typing.

### Changed

- Refactored `App.tsx` to use dedicated page components.
- Updated `test_journey.py` to target the React frontend (port 3000).

### Deprecated

- Legacy Gradio interface for calibration workflows is now superseded by the React frontend.
