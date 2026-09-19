# ADR-0016: Bit depth is preserved end to end

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §ARC-07 precondition

## Context

A 16-bit scan decodes as Pillow mode `I;16`. Every stage of the imaging pipeline dispatched on that mode and none of them handled it, each in its own way:

- `ImageProcessor.apply_curve` and `invert` matched no branch and fell through to `convert("RGB")`, turning a single-channel scan into three-channel colour.
- `create_digital_negative` with its default `ColorMode.GRAYSCALE`, which is what `BatchProcessor` and the Gradio tab both use, did match its condition and called `convert("L")`. Pillow's `convert` **clips** `I;16` rather than scaling it, so every sample above 255 became white. The result was a single-channel 8-bit negative with most of the frame blown out, not merely a coarsened one.
- `PlatinumPalladiumAI.generate_digital_negative` carried its own copy of that `convert("L")` block and so had the same defect independently of the processor.
- `export` then multiplied by 257 for the `TIFF_16BIT` and `PNG_16BIT` formats and wrote a file declaring 16 bits per sample. Every sample in it divided by 257, which is the signature of an 8-bit image widened after the fact.
- `_maybe_downsample` in the decode guard called `Image.thumbnail`, which Pillow cannot do for any `I;16` mode. Any scan longer than `downsample_max_side` (4096 by default, so most real scanner output) raised `ValueError: image has wrong mode` before any of the above ran. `preview_curve_effect` raised the same way.
- `load_image` truncated a `uint16` array with `astype(np.uint8)`, which wraps modulo 256: level 256 arrived as 0.

Banding in the highlights is what a 16-bit negative exists to avoid, so the failure was invisible in the file header and visible only in the print. No test covered a high-depth source, so none of it was exercised.

## Decision

A single-channel source carrying more than eight bits per sample keeps its depth through decode, load, curve application, inversion, preview and export.

The decode guard converts to `I`, which Pillow can resample, before downsampling; that conversion is lossless because `I` is wider. Curve application at 16 bits uses a 65536-entry table built by the same builder as the 256-entry table, so the two depths cannot drift apart in rounding or clipping. Inversion uses the 16-bit maximum.

Depth is reduced only where an output format demands it, and then by scaling, never by clipping or wrapping. `TIFF_16BIT` and `PNG_16BIT` carry sixteen bits; `TIFF`, `PNG` and the JPEG formats are the 8-bit entries beside them and get a scaled-down image; `ORIGINAL` means "whatever came in" and keeps the source depth. `ColorMode.RGB` is an explicit request for 8-bit colour and is served by scaling first, then colourising.

`PTPD_IMAGING_PRESERVE_BIT_DEPTH=false` restores 8-bit output for downstream tooling that requires it. It selects a depth, not a defect: the array path scales rather than wrapping either way. `PTPD_IMAGING_LUT_CACHE_ENTRIES` bounds the lookup-table cache, which was unbounded for the life of the process; the cache is locked because `BatchProcessor` shares one `ImageProcessor` across a thread pool.

## Consequences

The digital-negative export endpoint (ARC-07) can be built on a pipeline that neither quantises its input nor refuses it. `export_to_bytes` now honours a 16-bit request, which it previously ignored, so an HTTP endpoint returning bytes gets what it asked for.

Any future stage added to the imaging pipeline must handle the high-depth grayscale modes explicitly rather than converting to RGB or to "L" as a fallback, and any new grayscale conversion outside `ImageProcessor` reintroduces the AI facade's defect.

Callers that relied on `apply_curve` returning `RGB` for a 16-bit source now receive `I;16`. In particular `apply_curves_per_channel` can no longer follow `apply_curve` on such a source, because it requires RGB; no caller in this repository chains them. The `preserve_bit_depth` setting is the escape hatch.

Float (`F`) scans remain outside this decision. That mode carries no declared range, so choosing a normalisation would change the tone curve; `_as_16bit` scales only the unambiguous cases and leaves the rest alone.
