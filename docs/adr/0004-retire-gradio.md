# ADR-0004: Retire the Gradio UI after closing the product gaps

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

Two UIs were maintained; the Gradio Space had not been redeployed since November 2025; eleven Gradio features (digital negative export foremost) had no API or React equivalent; the React chemistry calculator re-implemented the formula with different constants.

## Decision

The FastAPI + React application is the product. Before `src/ptpd_calibration/ui/` and `app.py` are deleted, the must-close list in `docs/plans/2026-09-review/expert-architecture.md` §3.2 is implemented API-first (digital negative, image preview, chemistry, exposure, papers, linearization, wedge analysis, sessions, curve compare/blend, histogram). Explicit drops: batch processing, soft proofing v1, browser-side LLM-key settings, scanner calibration, neural curves. The Space becomes a Docker Space built from the attested GHCR image.

## Consequences

Until then `ui/` is frozen: excluded from lint and coverage, never modified. The README keeps the Gradio front matter until cutover.
