# Types Directory

## Purpose
Shared TypeScript type definitions for the frontend. Mirrors backend Pydantic models for type-safe API communication.

## Key Files
- `index.ts` — Barrel exports of all shared types and interfaces
- `models.ts` — API response/request types mirroring `src/ptpd_calibration/core/models.py`: `CurveData`, `CalibrationRecord`, `DensityMeasurement`, `PatchData`, `ScanUploadResponse`, etc.
- `mcts.ts` — MCTS-specific types: search parameters, simulation results, tree state
- `jest-axe.d.ts` — Type augmentations for jest-axe accessibility matchers
- `vitest-axe.d.ts` — Type augmentations for vitest-axe accessibility matchers

## Conventions
- **Backend sync**: Types in `models.ts` must stay in sync with `src/ptpd_calibration/core/models.py` and `core/types.py`
- **Interface for shapes**: Use `interface` for object types, `type` for unions/intersections (per CLAUDE.md)
- **Type-only imports**: Use `import type { ... }` for types consumed only at compile time
- **No runtime code**: This directory is types-only — no runtime logic, functions, or constants

## Adding Backend Types
1. Identify the Pydantic model in `src/ptpd_calibration/core/models.py`
2. Create matching TypeScript interface in `models.ts`
3. Map Python types: `str→string`, `int/float→number`, `bool→boolean`, `Optional→?`, `list→Array`
4. Import in API client/hooks with `import type`

## Pitfalls
- Do NOT let types drift from backend models — this causes runtime deserialization errors
- Python `Optional[X]` maps to `X | undefined` (or `X?` on interfaces), not `X | null`
- NumPy arrays in Python become `number[]` in TypeScript

## Related
- `../api/client.ts` — Imports these types for API calls
- `../api/hooks.ts` — Uses these types for query/mutation generics
- Backend: `src/ptpd_calibration/core/models.py` — Source of truth for data models
- Backend: `src/ptpd_calibration/core/types.py` — Source of truth for enums
