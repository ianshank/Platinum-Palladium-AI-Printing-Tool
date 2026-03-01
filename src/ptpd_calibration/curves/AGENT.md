# Curves Directory

## Purpose
Curve generation, modification, export, and AI enhancement for calibration curves used in platinum/palladium digital negative creation.

## Key Files
- `generator.py` — `CurveGenerator`: Creates calibration curves from density measurements (ChartThrob-equivalent algorithm)
- `modifier.py` — `CurveModifier`: Adjusts existing curves (brightness, contrast, highlight/shadow control)
- `ai_enhance.py` — `CurveAIEnhancer`: LLM-powered curve optimization
- `analysis.py` — Curve analysis and comparison utilities
- `export.py` — Export to QTR (.quad), Piezography, CSV, JSON, ACV formats
- `linearization.py` — Linearization and monotonicity enforcement
- `parser.py` — Parse QTR quad files
- `visualization.py` — Curve plotting utilities
- `__init__.py` — Re-exports: `CurveGenerator`, `CurveModifier`, `CurveAIEnhancer`, `BlendMode`, `SmoothingMethod`, `EnhancementGoal`, `load_quad_file`, `load_quad_string`, `save_curve`

## Conventions
- **Float precision**: Curve values are 0.0–1.0 (input) mapped to 0.0–1.0 (output). Use at least 3 decimal places
- **Monotonicity**: Output curves should generally be monotonically increasing (darker input → denser output). `linearization.py` enforces this
- **NumPy arrays**: Internal computation uses `np.ndarray`; convert to lists for API serialization
- **Smoothing methods**: `SmoothingMethod` enum — cubic spline, moving average, Savitzky-Golay

## Testing
```bash
pytest tests/unit/ -v -k "curve"
pytest tests/integration/ -v -k "curve"
```
Equivalence tests: `migration/equivalence-tests/` compares legacy vs new curve output within tolerance of 0.001.

## Pitfalls
- Do NOT break the `save_curve()` / `load_quad_file()` format — third-party tools (QuadTone RIP) depend on exact formatting
- Do NOT modify curves in-place — always return new `CurveData` instances
- AI enhancement requires LLM API keys configured — handle missing keys gracefully

## Related
- `../core/models.py` — `CurveData`, `DensityMeasurement` models
- `../api/server.py` — `/api/curves/*` endpoints call these functions
- `../detection/` — Produces measurements that feed into curve generation
- Frontend: `frontend/src/components/curves/` — CurveEditor visualizes and edits these curves
