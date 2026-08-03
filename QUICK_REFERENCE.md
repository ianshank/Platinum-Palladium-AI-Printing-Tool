# Deep Learning Features - Quick Reference

## 🎯 Detection (YOLOv8 + SAM)

### Basic Usage
```python
from ptpd_calibration.deep_learning import DeepTabletDetector

detector = DeepTabletDetector()
result = await detector.detect(image)
print(f"{result.num_patches} patches, confidence: {result.tablet_confidence:.3f}")
```

### Custom Config
```python
from ptpd_calibration.deep_learning import DetectionModelSettings, DetectionBackend

settings = DetectionModelSettings(
    detection_backend=DetectionBackend.YOLOV8,
    yolo_model_size="m",              # n, s, m, l, x
    yolo_confidence_threshold=0.25,
    device="auto",                     # auto, cpu, cuda, mps
    fallback_to_classical=True,
)
detector = DeepTabletDetector(settings)
```

### Result Access
```python
result = await detector.detect(image)

# Tablet info
x, y, w, h = result.tablet_bbox
confidence = result.tablet_confidence

# Patches
for patch in result.patches:
    patch.index         # 0, 1, 2, ...
    patch.bbox          # (x, y, w, h)
    patch.confidence    # 0.0 - 1.0
    patch.mask          # numpy array or None
    patch.centroid      # (x, y)
```

---

## 📊 Image Quality Assessment (Vision Transformer)

### Basic Usage
```python
from ptpd_calibration.deep_learning import VisionTransformerIQA

iqa = VisionTransformerIQA()
result = await iqa.assess_quality(image)
print(f"{result.quality_level.value}: {result.overall_score:.3f}")
```

### Custom Config
```python
from ptpd_calibration.deep_learning import ImageQualitySettings, IQAMetric

settings = ImageQualitySettings(
    primary_metric=IQAMetric.MANIQA,
    secondary_metrics=[IQAMetric.CLIP_IQA, IQAMetric.MUSIQ],
    analyze_zones=True,
    zone_count=11,                    # Ansel Adams zones 0-10
    device="auto",
    cache_embeddings=True,
)
iqa = VisionTransformerIQA(settings)
```

### Result Access
```python
result = await iqa.assess_quality(image)

# Overall quality
result.overall_score       # 0.0 - 1.0
result.quality_level       # EXCELLENT, GOOD, ACCEPTABLE, MARGINAL, POOR

# Metrics
result.metric_scores       # {'maniqa': 0.85, 'clip_iqa': 0.82, ...}
result.sharpness          # 0.0 - 1.0
result.noise_level        # 0.0 - 1.0
result.dynamic_range      # 0.0 - 1.0
result.contrast           # 0.0 - 2.0

# Zone analysis
result.highlight_quality  # 0.0 - 1.0
result.midtone_quality    # 0.0 - 1.0
result.shadow_quality     # 0.0 - 1.0

for zone in result.zone_scores:
    zone.zone              # 0 - 10
    zone.zone_name         # "Zone V: Middle Gray"
    zone.score             # 0.0 - 1.0
    zone.pixel_percentage  # 0.0 - 100.0
    zone.detail_preservation  # 0.0 - 1.0
    zone.issues            # ["issue1", "issue2", ...]

# Recommendations
result.recommendations     # ["suggestion1", "suggestion2", ...]
result.issues             # ["issue1", "issue2", ...]
result.embedding          # [0.1, 0.2, ...] or None
```

---

## 🚀 Combined Pipeline

```python
import asyncio
from ptpd_calibration.deep_learning import (
    DeepTabletDetector,
    VisionTransformerIQA,
)

async def analyze_print(image):
    # Run detection and quality in parallel
    detector = DeepTabletDetector()
    iqa = VisionTransformerIQA()

    detection, quality = await asyncio.gather(
        detector.detect(image),
        iqa.assess_quality(image),
    )

    return {
        'patches': detection.num_patches,
        'quality': quality.quality_level.value,
        'score': quality.overall_score,
        'recommendations': quality.recommendations,
    }

# Usage
result = await analyze_print(my_image)
```

---

## ⚙️ Environment Variables

### Detection
```bash
PTPD_DL_DETECTION_DETECTION_BACKEND=yolov8
PTPD_DL_DETECTION_YOLO_MODEL_SIZE=m
PTPD_DL_DETECTION_DEVICE=cuda
PTPD_DL_DETECTION_FALLBACK_TO_CLASSICAL=true
```

### Image Quality
```bash
PTPD_DL_IQA_PRIMARY_METRIC=maniqa
PTPD_DL_IQA_ANALYZE_ZONES=true
PTPD_DL_IQA_ZONE_COUNT=11
PTPD_DL_IQA_DEVICE=cuda
PTPD_DL_IQA_CACHE_EMBEDDINGS=true
```

---

## 🎨 Quality Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| EXCELLENT | ≥ 0.90 | Outstanding quality |
| GOOD | 0.75 - 0.89 | High quality |
| ACCEPTABLE | 0.60 - 0.74 | Usable quality |
| MARGINAL | 0.40 - 0.59 | Borderline quality |
| POOR | < 0.40 | Low quality |

---

## 📈 Performance Tips

1. **GPU Acceleration**: Set `device="cuda"` for 5-10x speedup
2. **Batch Processing**: Use `asyncio.gather()` for parallel processing
3. **Model Caching**: Initialize once, reuse detector/iqa instances
4. **Memory Management**: Call `.clear_cache()` when done

```python
# Good - reuse instances
detector = DeepTabletDetector()
for image in images:
    result = await detector.detect(image)
detector.clear_cache()

# Bad - recreate each time
for image in images:
    detector = DeepTabletDetector()  # Slow!
    result = await detector.detect(image)
```

---

## 🔧 Troubleshooting

### "Models not available"
- Install: `pip install torch ultralytics segment-anything timm`
- Will fallback to classical methods automatically

### "CUDA out of memory"
- Reduce batch size
- Use smaller model: `yolo_model_size="s"`
- Disable FP16: `half_precision=False`
- Or use CPU: `device="cpu"`

### Low detection confidence
- Increase `yolo_confidence_threshold`
- Try larger model: `yolo_model_size="l"`
- Ensure good image quality and lighting

### Quality scores seem off
- Check thresholds in `ImageQualitySettings`
- Try different metrics
- Ensure proper image normalization

---

## 📚 Available Metrics

### Detection Backends
- `YOLOV8` - YOLOv8 (recommended)
- `YOLOV9` - YOLOv9
- `DETR` - Detection Transformer
- `CLASSICAL` - Classical CV fallback

### Segmentation Backends
- `SAM` - Segment Anything Model
- `SAM2` - SAM 2.0
- `MOBILE_SAM` - Lightweight SAM
- `CLASSICAL` - Contour-based fallback

### IQA Metrics
- `MANIQA` - Multi-dimension Attention (recommended)
- `MUSIQ` - Multi-Scale IQA
- `TOPIQ` - Task-Oriented IQA
- `CLIP_IQA` - CLIP-based perceptual
- `NIQE` - Natural Image Quality
- `BRISQUE` - Blind/Referenceless Quality

---

## 📍 File Locations

- Detection: `src/ptpd_calibration/deep_learning/detection.py`
- IQA: `src/ptpd_calibration/deep_learning/image_quality.py`
- Config: `src/ptpd_calibration/deep_learning/config.py`
- Types: `src/ptpd_calibration/deep_learning/types.py`
- Models: `src/ptpd_calibration/deep_learning/models.py`
- Docs: `DEEP_LEARNING_IMPLEMENTATION.md`
