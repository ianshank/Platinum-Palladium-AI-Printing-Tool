# Imaging Directory

## Purpose
Image processing utilities for print analysis: histogram computation, colorspace conversion, split-grade printing support, and general image manipulation.

## Key Files
- `processor.py` — Core image processing: resize, crop, rotate, color correction, TIFF handling
- `histogram.py` — Histogram computation and analysis (per-channel, luminance, density distribution)
- `split_grade.py` — Split-grade printing: separate highlight and shadow exposures for contrast control
- `__init__.py` — Package exports

## Conventions
- **OpenCV + Pillow**: Use OpenCV (`cv2`) for computation, Pillow for format I/O (especially TIFF with ICC profiles)
- **NumPy arrays**: All intermediate data is `np.ndarray` (dtype typically `float32` or `uint8`)
- **Color spaces**: BGR (OpenCV default) → RGB → LAB/HSV as needed. Always be explicit about color space
- **16-bit TIFF**: Digital negatives must be exported as 16-bit TIFF — preserve full dynamic range

## Key Operations
- Negative inversion: Input image → inverted for contact printing
- Curve application: Apply calibration curve LUT to image data
- Split-grade: Decompose into highlight/shadow layers with separate curves
- Histogram: Compute and display density distribution for print evaluation

## Testing
```bash
pytest tests/unit/ -v -k "imaging or processor or histogram"
```

## Pitfalls
- BGR vs RGB: OpenCV loads as BGR — conversion errors cause incorrect color rendering
- 8-bit vs 16-bit: Truncating 16-bit data to 8-bit loses critical shadow detail
- ICC profiles: TIFF files may contain ICC profiles — preserve them through processing pipeline
- Memory: Large TIFF files (300+ DPI, 16-bit) can be several hundred MB — process in chunks if needed

## Related
- `../detection/` — Uses imaging utilities for scan preprocessing
- `../curves/` — Curves are applied to images via LUT operations
- `../api/server.py` — `/api/export/negative` uses processor for TIFF generation
- Frontend: `../../../frontend/src/components/preview/` — Image preview in browser
