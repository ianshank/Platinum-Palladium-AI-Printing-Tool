# Detection Directory

## Purpose
Step tablet detection and density extraction from scanned prints. Uses OpenCV for image processing to automatically identify and measure step tablet patches.

## Key Files
- `detector.py` — `StepTabletDetector`: Locates step tablet patches in scanned images using contour detection
- `extractor.py` — `DensityExtractor`: Extracts density values from detected patch regions
- `reader.py` — `StepTabletReader`: High-level orchestrator combining detection + extraction
- `scanner.py` — Scanner interface utilities and image preprocessing
- `__init__.py` — Re-exports `StepTabletReader`

## Pipeline
1. `scanner.py` — Preprocess image (denoise, normalize, color correct)
2. `detector.py` — Find patch bounding boxes via contour analysis
3. `extractor.py` — Sample RGB/LAB values within patches, compute density
4. `reader.py` — Orchestrate pipeline, return `StepTabletScan` with `PatchData` array

## Conventions
- **OpenCV + NumPy**: All image operations use `cv2` and `np.ndarray`
- **Color spaces**: Work in LAB for density calculation (L* channel correlates with visual density)
- **Tablet types**: Support Stouffer 21, 31, 41 step wedges — patch count affects detection parameters
- **Robustness**: Handle rotation, partial occlusion, and variable lighting conditions

## Testing
```bash
pytest tests/unit/ -v -k "detection or detector or extractor"
```
Test fixtures in `tests/fixtures/` contain sample step tablet images.

## Pitfalls
- OpenCV `imread` returns BGR, not RGB — convert before color space transformations
- Density values must be ≥0.0 — negative values indicate measurement error
- Patch ordering matters: index 0 = lightest (paper white), highest index = darkest
- Large images can be slow — downsample for detection, full-res for extraction

## Related
- `../core/models.py` — `PatchData`, `StepTabletScan`, `DensityMeasurement`
- `../api/server.py` — `/api/scan/upload` calls `StepTabletReader`
- `../curves/generator.py` — Consumes density measurements for curve generation
- Frontend: `frontend/src/components/calibration/ScanUpload.tsx` — Upload UI
