# Core Directory

## Purpose
Shared data models, types, and base services used across the entire backend. This is the foundation layer — everything else depends on it.

## Key Files
- `models.py` — Pydantic models: `PatchData`, `DensityMeasurement`, `CalibrationRecord`, `CurveData`, `StepTabletScan`, etc.
- `types.py` — Enums and type aliases: `ChemistryType`, `CurveType`, `ContrastAgent`, `DeveloperType`, `MeasurementUnit`, `PaperSizing`
- `base_service.py` — Abstract base service class for dependency injection
- `events.py` — Event system for cross-module communication
- `logging.py` — Structured logging configuration
- `debug.py` — Debug utilities

## Conventions
- **Pydantic v2**: All models use `BaseModel` with `ConfigDict`, `Field` validators, `field_validator`, `model_validator`
- **NumPy interop**: Models accept `np.ndarray` via `arbitrary_types_allowed=True` and convert to tuples in validators
- **UUID primary keys**: Use `uuid4` for model IDs
- **Type-first design**: Define types/enums in `types.py`, reference in models — never use raw strings for constrained values

## Frontend Type Sync
Types here must stay in sync with `frontend/src/types/models.ts`. When modifying:
1. Update the Pydantic model in `models.py`
2. Update the corresponding TypeScript interface in `frontend/src/types/models.ts`
3. Verify API client types still match

## Testing
```bash
pytest tests/unit/ -v -k "model or type"
```

## Pitfalls
- Do NOT add domain-specific logic here — this is pure data modeling
- Do NOT break model field names — they map directly to API JSON and frontend types
- Float precision matters for density values — use `ge=0.0` constraints consistently

## Related
- `../api/` — Endpoints use these models for request/response serialization
- `../curves/` — Curve algorithms operate on `CurveData`, `DensityMeasurement`
- `../detection/` — Scanner produces `PatchData`, `StepTabletScan`
- Frontend: `frontend/src/types/models.ts` — TypeScript mirror
