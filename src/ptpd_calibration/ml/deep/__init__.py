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


# Lazy imports to avoid requiring PyTorch at module load time
def __getattr__(name: str):
    """Lazy import of PyTorch-dependent modules."""
    torch_modules = {
        "CurveMLP",
        "CurveCNN",
        "ContentAwareCurveNet",
        "UniformityCorrectionNet",
        "ProcessSimulator",
        "CalibrationDataset",
        "CurveTrainer",
        "DeepCurvePredictor",
    }

    if name in torch_modules:
        if not _check_torch_available():
            raise PyTorchNotAvailableError(
                f"PyTorch is required to use {name}. "
                "Install with: pip install ptpd-calibration[deep]"
            )

        if name in {"CurveMLP", "CurveCNN", "ContentAwareCurveNet", "UniformityCorrectionNet"}:
            from ptpd_calibration.ml.deep.models import (
                ContentAwareCurveNet,
                CurveCNN,
                CurveMLP,
                UniformityCorrectionNet,
            )

            return locals()[name]

        elif name == "ProcessSimulator":
            from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

            return ProcessSimulator

        elif name == "CalibrationDataset":
            from ptpd_calibration.ml.deep.dataset import CalibrationDataset

            return CalibrationDataset

        elif name == "CurveTrainer":
            from ptpd_calibration.ml.deep.training import CurveTrainer

            return CurveTrainer

        elif name == "DeepCurvePredictor":
            from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

            return DeepCurvePredictor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
