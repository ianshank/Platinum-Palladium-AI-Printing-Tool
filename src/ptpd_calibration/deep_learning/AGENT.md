# Deep Learning Directory

## Purpose
Optional deep learning modules for advanced features: defect detection, print quality assessment, neural curve optimization, recipe recommendation, and multimodal analysis. All modules require PyTorch — the system operates without them.

## Key Files
- `models.py` — Neural network architectures (CNN, ResNet-based models)
- `types.py` — Type definitions for DL inputs/outputs
- `config.py` — Training hyperparameters and model configuration
- `neural_curve.py` — Neural network-based curve prediction
- `defect_detection.py` — Print defect identification from scans
- `image_quality.py` — Print quality scoring model
- `recipe_recommendation.py` — Chemistry recipe suggestion via learned embeddings
- `print_comparison.py` — Compare two prints for quality differences
- `multimodal_assistant.py` — Vision-language model integration
- `diffusion_enhance.py` — Image enhancement via diffusion models
- `uv_exposure.py` — UV exposure optimization
- `federated_learning.py` — Privacy-preserving model training
- `detection.py` — DL-enhanced step tablet detection
- `training/` — Training scripts and data loaders

## Conventions
- **Optional dependency**: ALL torch imports MUST be lazy (inside functions or guarded by try/except)
- **Graceful degradation**: Every feature has a non-DL fallback path
- **GPU handling**: Check `torch.cuda.is_available()` — never assume GPU presence
- **Model storage**: Trained models stored via API state dict, not filesystem hardcoding

## Testing
```bash
pytest tests/unit/ -v -k "deep_learning" -m "not slow"
```
Skip DL tests if PyTorch not installed: `@pytest.mark.skipif(not HAS_TORCH, ...)`

## Pitfalls
- NEVER add `torch` or `torchvision` to core dependencies — they are optional extras
- Model loading can be slow — cache loaded models, don't reload per request
- Training jobs should use Celery task queue for background processing
- Memory: Large models + large images can exhaust RAM — check before loading

## Related
- `../api/deep_learning.py` — DL router (conditionally included)
- `../api/server.py` — Wraps DL router import in try/except
- `../detection/` — Non-DL detection (fallback when DL unavailable)
- `../curves/` — Non-DL curve generation (always available)
