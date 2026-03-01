# Calibration Components Directory

## Purpose
Top-level calibration UI components: the wizard wrapper and scan upload interface for step tablet analysis.

## Key Files
- `CalibrationWizard.tsx` — Orchestrates the multi-step wizard flow (`CalibrationWizard.test.tsx`)
- `ScanUpload.tsx` — Step tablet image upload with preview and tablet type selection (`ScanUpload.test.tsx`)

## Conventions
- **CalibrationWizard** is the entry point — it renders the wizard steps from `../wizard/`
- **ScanUpload** handles file validation (image types only), preview via `URL.createObjectURL`, and upload progress tracking
- Tablet types: `stouffer_21`, `stouffer_31`, `stouffer_41` — selected via dropdown

## Data Flow
1. ScanUpload → `api.scan.upload(file, tabletType)` → backend detects patches
2. Response populates `calibrationSlice.measurements` with density values
3. CalibrationWizard advances to analysis step

## Testing
```bash
pnpm test -- --run src/components/calibration/
```

## Pitfalls
- File validation: Accept only image formats (JPEG, PNG, TIFF) — reject non-image files before upload
- Upload progress needs `onUploadProgress` from Axios (not fetch API)
- Memory: Revoke object URLs on component unmount

## Related
- `../wizard/` — Step components rendered by CalibrationWizard
- `../../stores/slices/calibrationSlice.ts` — Calibration state
- `../../stores/slices/imageSlice.ts` — Uploaded image state
- Backend: `../../../../src/ptpd_calibration/detection/` — Step tablet detection
