# Wizard Directory

## Purpose
Multi-step calibration wizard guiding users through the full calibration workflow: upload scan → analyze → configure → preview → export.

## Key Files
- `index.ts` — Barrel export
- `Step1Upload.tsx` — Upload step tablet scan image
- `Step2Analyze.tsx` — Automated/manual density analysis
- `Step3Configure.tsx` — Curve generation parameters (+ `Step3Configure.test.tsx`)
- `Step4Preview.tsx` — Curve preview and visual verification
- `Step5Export.tsx` — Export format selection and download (+ `Step5Export.test.tsx`)
- `__tests__/` — Additional integration tests for multi-step flows

## Conventions
- **Linear flow**: Steps must be completed in order (1→5). Step N enables step N+1
- **Store-driven state**: Wizard state lives in `calibrationSlice` (currentStep, measurements, curve data)
- **Each step is self-contained**: Step components handle their own validation and can independently render
- **Error boundaries**: Each step should handle its own error state gracefully

## Data Flow
1. Step1: Image upload → `imageSlice.current`
2. Step2: Analysis → `calibrationSlice.measurements`
3. Step3: Config → `calibrationSlice.current` (generated curve)
4. Step4: Preview → reads `curveSlice.current`
5. Step5: Export → calls `api.curves.export()` or `api.export.negative()`

## Testing
```bash
pnpm test -- --run src/components/wizard/
pnpm test -- --run src/components/wizard/__tests__/
```

## Pitfalls
- Do NOT allow skipping steps — enforce sequential completion via store state
- Step transitions should be atomic — update all relevant store slices together
- File uploads in Step1 need cleanup (URL.revokeObjectURL) on unmount

## Related
- `../calibration/` — CalibrationWizard wraps this wizard flow
- `../../stores/slices/calibrationSlice.ts` — Wizard state management
- `../../stores/slices/curveSlice.ts` — Curve data for steps 3-5
