# ADR-0016: Bit depth is preserved end to end

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §ARC-07 precondition

## Context

`ImageProcessor` dispatched on the Pillow mode. A 16-bit scan decodes as `I;16`, which matched no branch in `apply_curve`, `invert` or the grayscale conversion in `create_digital_negative`, so each one fell through to `convert("RGB")` and produced 8-bit colour. `export` then multiplied that result by 257 and wrote a file that declared 16 bits per sample. The output was a single-channel scanner file turned into a three-channel negative holding fewer distinct levels than the scanner produced, and every sample in it divided by 257.

Banding in the highlights is exactly what a 16-bit negative exists to avoid, so the failure was silent in the file format and visible only in the print. No test covered a high-depth source, so the whole path was unexercised.

`load_image` had the same shape of bug for array sources: `astype(np.uint8)` truncates modulo 256, so a `uint16` array arrived scrambled rather than merely coarsened.

## Decision

A single-channel source carrying more than eight bits per sample keeps its depth through load, curve application, inversion and export. `ColorMode.GRAYSCALE` treats such an image as already grayscale; `ColorMode.RGB` remains an explicit caller request for 8-bit colour and is honoured.

Curve application at 16 bits uses a 65536-entry table built by the same builder as the 256-entry table, so the two depths cannot drift apart in rounding or clipping. Scaling to 16 bits on export applies only to genuinely 8-bit data; an array already on the 0–65535 scale passes through unchanged.

`PTPD_IMAGING_PRESERVE_BIT_DEPTH=false` restores the previous behaviour for downstream tooling that requires 8-bit output. `PTPD_IMAGING_LUT_CACHE_ENTRIES` bounds the lookup-table cache, which was unbounded for the life of the process.

## Consequences

The digital-negative export endpoint (ARC-07) can be built on a pipeline that does not quantise its input. Any future stage added to the imaging pipeline must handle the high-depth grayscale modes explicitly rather than converting to RGB as a fallback. Callers that relied on `apply_curve` returning `RGB` for a 16-bit source now receive `I;16`; the setting above is the escape hatch.
