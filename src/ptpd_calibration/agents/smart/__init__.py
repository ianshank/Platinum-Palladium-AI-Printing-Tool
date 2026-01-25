"""Smart capabilities module for advanced agent features.

This module provides intelligent capabilities including:
- Active learning from user feedback
- Environmental adaptation
- Predictive quality assessment
"""

from .learning import (
    ActiveLearner,
    FeedbackRecord,
    FeedbackType,
    LearningSettings,
    LearningModel,
    PatternMatcher,
)
from .environment import (
    EnvironmentAdapter,
    EnvironmentConditions,
    EnvironmentSettings,
    AdaptationResult,
    SeasonalProfile,
)
from .prediction import (
    QualityPredictor,
    PredictionResult,
    PredictionSettings,
    QualityFeatures,
    HistoricalData,
)

__all__ = [
    # Learning
    "ActiveLearner",
    "FeedbackRecord",
    "FeedbackType",
    "LearningSettings",
    "LearningModel",
    "PatternMatcher",
    # Environment
    "EnvironmentAdapter",
    "EnvironmentConditions",
    "EnvironmentSettings",
    "AdaptationResult",
    "SeasonalProfile",
    # Prediction
    "QualityPredictor",
    "PredictionResult",
    "PredictionSettings",
    "QualityFeatures",
    "HistoricalData",
]
