"""
Deep learning modules for tone curve prediction.

This package provides PyTorch-based neural network models for learning
tone curves from calibration data, enabling:

- Metadata-conditioned curve prediction (process parameters -> curve)
- Content-aware local curve adjustments
- Spatial uniformity correction (learned dodge/burn)
- End-to-end differentiable process simulation

All modules are optional and require PyTorch to be installed:
    pip install ptpd-calibration[deep]
"""

from ptpd_calibration.ml.deep.exceptions import (
    DeepLearningError,
    ModelNotTrainedError,
    PyTorchNotAvailableError,
)

__all__ = [
    # Exceptions
    "DeepLearningError",
    "ModelNotTrainedError",
    "PyTorchNotAvailableError",
]


def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# Mapping of class names to their module paths for lazy imports
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Models
    "CurveMLP": ("ptpd_calibration.ml.deep.models", "CurveMLP"),
    "CurveCNN": ("ptpd_calibration.ml.deep.models", "CurveCNN"),
    "ContentAwareCurveNet": ("ptpd_calibration.ml.deep.models", "ContentAwareCurveNet"),
    "UniformityCorrectionNet": ("ptpd_calibration.ml.deep.models", "UniformityCorrectionNet"),
    # Process simulation
    "ProcessSimulator": ("ptpd_calibration.ml.deep.process_sim", "ProcessSimulator"),
    # Dataset
    "CalibrationDataset": ("ptpd_calibration.ml.deep.dataset", "CalibrationDataset"),
    # Training
    "CurveTrainer": ("ptpd_calibration.ml.deep.training", "CurveTrainer"),
    # Predictor
    "DeepCurvePredictor": ("ptpd_calibration.ml.deep.predictor", "DeepCurvePredictor"),
}


def __getattr__(name: str):
    """
    Lazy import of PyTorch-dependent modules.

    Uses importlib for explicit module resolution without relying on locals().
    """
    if name in _LAZY_IMPORTS:
        if not _check_torch_available():
            raise PyTorchNotAvailableError(
                f"PyTorch is required to use {name}. "
                "Install with: pip install ptpd-calibration[deep]"
            )

        import importlib

        module_path, class_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
