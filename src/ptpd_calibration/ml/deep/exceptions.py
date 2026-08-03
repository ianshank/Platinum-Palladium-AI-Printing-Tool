"""
Exceptions for deep learning modules.
"""


class DeepLearningError(Exception):
    """Base exception for deep learning module errors."""

    pass


class PyTorchNotAvailableError(DeepLearningError):
    """Raised when PyTorch is required but not installed."""

    pass


class ModelNotTrainedError(DeepLearningError):
    """Raised when prediction is attempted on an untrained model."""

    pass


class InvalidModelConfigError(DeepLearningError):
    """Raised when model configuration is invalid."""

    pass


class TrainingError(DeepLearningError):
    """Raised when an error occurs during training."""

    pass


class DatasetError(DeepLearningError):
    """Raised when there's an issue with the training dataset."""

    pass


class CheckpointError(DeepLearningError):
    """Raised when there's an issue with model checkpoints."""

    pass
