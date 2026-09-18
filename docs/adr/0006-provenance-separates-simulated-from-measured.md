# ADR-0006: Provenance separates simulated from measured data

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

`mcts/export.py` wrote simulated densities into `CalibrationRecord.measured_densities` and developer temperature into the ambient `temperature` field; the sklearn predictor trained on whatever records existed.

## Decision

`CalibrationRecord.provenance` is `"measured"` (default) or `"simulated"`; `developer_temp_c` is a separate optional field. Exporters set `provenance="simulated"`. Training and similarity queries exclude simulated records unless `include_simulated=True` is passed explicitly. Old JSON records load as measured.

## Consequences

Any future data source must declare its provenance; UI surfaces should show it.
