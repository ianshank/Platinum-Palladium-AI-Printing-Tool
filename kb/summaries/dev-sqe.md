# DEV-SQE Summary

## 2026-02-09 | Code Review Hardening

### What was done
- **Bug fix**: Fixed `ctrlMatch` false-positive in `useKeyboardShortcuts` (Ctrl held incorrectly matching `ctrl:false` shortcuts)
- **useUndoRedo hardened**: Removed dead `initialRef`, added `logger.debug` tracing, capped redo stack via `capStack()`, config-driven `maxHistory` default from `config.ui.undoHistoryLimit`, validated with `Math.max(1, Math.floor())`
- **CurveEditor hardened**: Added logging to all handlers, `role="alert"` on errors, `role="img"` with aria-label on chart, typed `AdjustmentType` union, `config.calibration.maxCurvePoints` instead of hard-coded 256, extracted helpers
- **Keyboard shortcuts**: Added `<select>` exclusion, memoized shortcuts array with `useMemo`
- **Equivalence tests**: Removed dead `callCurveModifyEndpoint`, `readonly number[]` signatures, NaN handling, removed 5 vacuous tests, fixed misleading fixture labels
- **New tests**: 12 edge-case tests (ctrl/meta false-positive, select exclusion, contentEditable, Mac Cmd support, case-insensitive keys, null initial value, maxHistory boundary/clamp, reset tracking, stable callback refs)

### Verification
- **TypeScript**: 0 errors
- **Tests**: 726 passed, 0 failed (~80% coverage) — 6 net new tests (12 added, 5 vacuous removed, 1 fixture deduplicated)
- **Build**: Success (bundle ~280KB gzipped)

### Files Changed (10 files, +382 -218)
- `frontend/src/hooks/useUndoRedo.ts` - Dead code removal, logging, capStack, config-driven default
- `frontend/src/hooks/useUndoRedo.test.ts` - 5 new edge-case tests
- `frontend/src/hooks/useKeyboardShortcuts.ts` - ctrlMatch fix, select exclusion, useMemo
- `frontend/src/hooks/useKeyboardShortcuts.test.ts` - 7 new tests for bugs and exclusions
- `frontend/src/hooks/index.ts` - Export UseUndoRedoReturn type
- `frontend/src/components/curves/CurveEditor.tsx` - Logging, a11y, types, config-driven values
- `frontend/src/__tests__/equivalence/setup.ts` - Dead code removal, readonly sigs, NaN handling
- `frontend/src/__tests__/equivalence/CurveEditor.equiv.test.ts` - Remove vacuous tests, fix casts
- `frontend/src/__tests__/equivalence/ScanAnalysis.equiv.test.ts` - Fix misleading labels
- `frontend/src/config/index.ts` - Added `ui.undoHistoryLimit` config entry

---

## 2026-02-09 | Hardening: All 5 Gaps Closed

### What was done
- **gap-3 closed**: Undo/redo for CurveEditor via new `useUndoRedo` generic hook with Undo/Redo UI buttons
- **gap-1 closed**: Equivalence test framework with `CurveEditor.equiv.test.ts` and `ScanAnalysis.equiv.test.ts`, shared fixtures/tolerance utilities in `setup.ts`
- **Keyboard shortcuts**: Wired `useAppShortcuts()` into Layout (Ctrl+1-5 tab navigation, Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y undo/redo)
- **Mobile responsive**: Added `px-4 sm:px-6 lg:px-8` responsive padding to all 7 page containers

### Verification
- **TypeScript**: 0 errors
- **Tests**: 720 passed, 0 failed (~80% coverage) — 40 new tests
- **Build**: Success (bundle ~280KB gzipped, under 500KB target)

### Gaps Status: ALL CLOSED
- gap-1: Equivalence test framework (CLOSED)
- gap-2: CurveEditor save via API (CLOSED)
- gap-3: Undo/redo stack for curve editing (CLOSED)
- gap-4: ImageUpload/ImagePreview standalone (CLOSED)
- gap-5: ExportPanel standalone (CLOSED)

### Files Changed
- `frontend/src/hooks/useUndoRedo.ts` - New generic undo/redo hook
- `frontend/src/hooks/useUndoRedo.test.ts` - 12 tests for undo/redo hook
- `frontend/src/hooks/index.ts` - Export new hook
- `frontend/src/components/curves/CurveEditor.tsx` - Integrated undo/redo buttons
- `frontend/src/components/Layout/Layout.tsx` - Wired useAppShortcuts
- `frontend/src/components/Layout/Layout.test.tsx` - Mock for useAppShortcuts
- `frontend/src/__tests__/equivalence/setup.ts` - Equivalence test framework
- `frontend/src/__tests__/equivalence/CurveEditor.equiv.test.ts` - Curve equivalence tests
- `frontend/src/__tests__/equivalence/ScanAnalysis.equiv.test.ts` - Scan analysis equivalence tests
- `frontend/src/pages/*.tsx` - All 7 pages updated with responsive padding
- `docs/migration/progress.json` - All gaps closed
- `kb/ledger/ledger.jsonl` - Event logged
- `kb/sessions/dev-sqe.state.json` - State updated

---

## 2026-02-09 | Sprint 0+2: Migration Completion (15/15)

### What was done
- **CurveEditor save** wired to `useSaveCurve` API mutation hook (Sprint 0, gap-2 closed)
- **ImageUpload** standalone component extracted with react-dropzone, progress tracking, preview, validation (26 tests)
- **ImagePreview** standalone component with react-zoom-pan-pinch, side-by-side comparison, metadata overlay (29 tests)
- **ExportPanel** standalone component with configurable formats, file download, retry (28 tests)
- Fixed pre-existing ChemistryCalculator test type assertion (lint hook conflict)
- Made `addToast` calls defensive with optional chaining for test robustness
- Replaced `userEvent` with `fireEvent` in new tests to avoid jsdom/react-dropzone timeouts

### Verification
- **TypeScript**: 0 errors
- **Tests**: 680 passed, 0 failed (~80% coverage)
- **Build**: Success (bundle under 500KB gzipped)
- **Lint**: 0 errors in modified files (warnings only: `explicit-function-return-type`)

### Gaps Closed
- gap-2: CurveEditor save via API
- gap-4: ImageUpload/ImagePreview standalone
- gap-5: ExportPanel standalone

### Gaps Remaining
- gap-1: No equivalence tests between Gradio and React (medium)
- gap-3: No undo/redo stack for curve editing (medium)

### Files Changed
- `frontend/src/api/hooks.ts` - Added `useSaveCurve` hook
- `frontend/src/components/curves/CurveEditor.tsx` - Wired save, added accessibility labels
- `frontend/src/components/curves/CurveEditor.test.tsx` - Updated tests for API save
- `frontend/src/components/upload/ImageUpload.tsx` - New component
- `frontend/src/components/upload/ImageUpload.test.tsx` - New tests (26)
- `frontend/src/components/preview/ImagePreview.tsx` - New component
- `frontend/src/components/preview/ImagePreview.test.tsx` - New tests (29)
- `frontend/src/components/export/ExportPanel.tsx` - New component
- `frontend/src/components/export/ExportPanel.test.tsx` - New tests (28)
- `docs/migration/progress.json` - Updated to 15/15
