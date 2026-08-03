"""
Training utilities for deep learning models.

This module provides:
- Synthetic data generators for training neural networks
- Training pipelines for all models
- Data augmentation utilities
- Validation and evaluation tools

Key design principle: Input data should NOT match expected output
to prevent overfitting and hallucinations. All generators produce
varied, realistic data with noise and realistic variations.
"""

from ptpd_calibration.deep_learning.training.data_generators import (
    CurveDataGenerator,
    DefectDataGenerator,
    DetectionDataGenerator,
    ExposureDataGenerator,
    RecipeDataGenerator,
    SyntheticDataConfig,
)
from ptpd_calibration.deep_learning.training.pipelines import (
    CurveTrainingPipeline,
    DefectTrainingPipeline,
    DetectionTrainingPipeline,
    EarlyStopping,
    ExposureTrainingPipeline,
    RecipeTrainingPipeline,
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
)

__all__ = [
    # Configuration
    "SyntheticDataConfig",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResult",
    "EarlyStopping",
    # Data generators
    "DetectionDataGenerator",
    "CurveDataGenerator",
    "ExposureDataGenerator",
    "DefectDataGenerator",
    "RecipeDataGenerator",
    # Training pipelines
    "DetectionTrainingPipeline",
    "CurveTrainingPipeline",
    "ExposureTrainingPipeline",
    "DefectTrainingPipeline",
    "RecipeTrainingPipeline",
]
