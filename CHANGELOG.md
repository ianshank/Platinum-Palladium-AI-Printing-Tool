# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
