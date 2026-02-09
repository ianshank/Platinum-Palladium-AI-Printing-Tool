# DEV-SQE Summary

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
