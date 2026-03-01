# Chemistry Directory

## Purpose
Chemistry calculators for alternative photographic printing processes. Computes precise solution volumes, drop counts, and ratios for different printing chemistries.

## Key Files
- `calculator.py` — Primary Pt/Pd chemistry calculator: metal ratios, sensitizer volumes, contrast agent amounts
- `cyanotype_calculator.py` — Cyanotype-specific chemistry (ferric ammonium citrate + potassium ferricyanide)
- `silver_gelatin_calculator.py` — Silver gelatin developer formulations
- `__init__.py` — Package exports

## Conventions
- **Domain accuracy**: Formulas are based on published photographic chemistry references — verify changes against source material
- **Units**: All volumes in milliliters (mL), all ratios as decimal fractions (0.0–1.0)
- **Drop counts**: 20 drops per mL standard (configurable). Drop count precision is critical for small volumes
- **Paper sizing**: Calculations account for paper size and sizing type via `PaperSizing` enum from `core/types.py`

## Key Concepts
- **Metal ratio**: Proportion of platinum to palladium (0.0 = pure Pd, 1.0 = pure Pt)
- **Contrast agents**: `Na2` (sodium chloroplatinate), `dichromate`, `hydrogen peroxide` — affect curve shape
- **Developer types**: `potassium_oxalate`, `ammonium_citrate`, etc.

## Testing
```bash
pytest tests/unit/ -v -k "chemistry or calculator"
```

## Pitfalls
- Do NOT round intermediate calculations — only round final display values
- Metal ratio boundaries (0.0 and 1.0) are valid — don't exclude edge cases
- Chemistry amounts scale linearly with paper area — ensure no off-by-one in area calculations
- Each calculator type has different required inputs — don't assume a common interface

## Related
- `../core/types.py` — `ChemistryType`, `ContrastAgent`, `DeveloperType`, `PaperSizing` enums
- `../api/server.py` — Chemistry endpoints
- Frontend: `frontend/src/components/chemistry/` — ChemistryCalculator UI
- Frontend: `frontend/src/stores/slices/chemistrySlice.ts` — Chemistry state
