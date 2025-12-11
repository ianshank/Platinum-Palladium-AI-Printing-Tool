"""
Machine learning module for calibration prediction and refinement.

This module provides:
- CalibrationDatabase: Store and query calibration records
- CurvePredictor: Classical ML curve prediction (scikit-learn)
- ActiveLearner: Active learning for iterative model improvement
- TransferLearner: Transfer learning for new papers/chemistry

For deep learning models, see the `deep` submodule:
- deep.DeepCurvePredictor: PyTorch-based curve prediction
- deep.ProcessSimulator: Differentiable process simulation
"""

from ptpd_calibration.ml.active_learning import ActiveLearner
from ptpd_calibration.ml.database import CalibrationDatabase
from ptpd_calibration.ml.predictor import CurvePredictor
from ptpd_calibration.ml.transfer import TransferLearner

__all__ = [
    "CalibrationDatabase",
    "CurvePredictor",
    "ActiveLearner",
    "TransferLearner",
]


# Lazy import for deep learning module
def __getattr__(name: str):
    """Lazy import of deep learning module."""
    if name == "deep":
        from ptpd_calibration.ml import deep

        return deep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
