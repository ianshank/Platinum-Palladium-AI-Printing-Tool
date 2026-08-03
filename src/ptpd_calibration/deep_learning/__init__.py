"""
Deep Learning module for Platinum-Palladium AI Printing Tool.

This module provides cutting-edge AI/ML capabilities including:
- Deep learning step tablet detection (YOLOv8 + SAM)
- Vision Transformer image quality assessment
- Diffusion model enhancement
- Neural curve prediction (Transformer-based)
- UV exposure neural predictor
- Automated defect detection (U-Net + ResNet)
- Recipe recommendation engine
- Deep print comparison (LPIPS)
- Multi-modal AI assistant
- Federated community learning

All components are designed with:
- No hardcoded values (configuration-driven)
- Comprehensive type hints
- Async support where appropriate
- GPU acceleration when available
- Graceful fallback to CPU
"""

from ptpd_calibration.deep_learning.config import (
    DeepLearningSettings,
    DefectDetectionSettings,
    DetectionModelSettings,
    DiffusionSettings,
    FederatedLearningSettings,
    ImageQualitySettings,
    MultiModalSettings,
    NeuralCurveSettings,
    PrintComparisonSettings,
    RecipeRecommendationSettings,
    UVExposureSettings,
    get_deep_learning_settings,
)
from ptpd_calibration.deep_learning.models import (
    CurvePredictionResult,
    DeepDetectionResult,
    DefectDetectionResult,
    # Defect detection models
    DetectedDefect,
    # Detection models
    DetectedPatch,
    # Result models
    DiffusionEnhancementResult,
    EnhancementRegion,
    ImageQualityResult,
    PrintComparisonResult,
    # Recipe recommendation models
    RecipeRecommendation,
    RecipeRecommendationResult,
    # UV exposure models
    UVExposurePrediction,
    # Print comparison models
    ZoneComparison,
    # Image quality models
    ZoneQualityScore,
)
from ptpd_calibration.deep_learning.types import (
    # Federated types
    AggregationStrategy,
    ComparisonMode,
    ComparisonResult,
    # Neural prediction types
    CurvePredictorArchitecture,
    DefectSeverity,
    # Defect types
    DefectType,
    # Detection types
    DetectionBackend,
    # Diffusion types
    DiffusionScheduler,
    EnhancementMode,
    # Quality assessment types
    IQAMetric,
    # Comparison types
    PerceptualMetric,
    PrivacyLevel,
    QualityLevel,
    RecipeCategory,
    # Recommendation types
    RecommendationStrategy,
    SegmentationBackend,
    SimilarityMetric,
    ToolType,
    UncertaintyMethod,
    # Multi-modal types
    VisionLanguageModel,
)

# Lazy imports for implementation classes (to avoid dependency errors)
# These are imported on-demand when needed
__lazy_imports = {
    "DeepTabletDetector": "ptpd_calibration.deep_learning.detection",
    "VisionTransformerIQA": "ptpd_calibration.deep_learning.image_quality",
    "DiffusionEnhancer": "ptpd_calibration.deep_learning.diffusion_enhance",
    "NeuralCurvePredictor": "ptpd_calibration.deep_learning.neural_curve",
    "CurveTransformer": "ptpd_calibration.deep_learning.neural_curve",
    "UVExposurePredictor": "ptpd_calibration.deep_learning.uv_exposure",
    "ExposureNet": "ptpd_calibration.deep_learning.uv_exposure",
    "DefectDetector": "ptpd_calibration.deep_learning.defect_detection",
    "DefectSegmentationNet": "ptpd_calibration.deep_learning.defect_detection",
    "DefectClassifierNet": "ptpd_calibration.deep_learning.defect_detection",
    "RecipeRecommender": "ptpd_calibration.deep_learning.recipe_recommendation",
    "RecipeEncoder": "ptpd_calibration.deep_learning.recipe_recommendation",
    "ImageEncoder": "ptpd_calibration.deep_learning.recipe_recommendation",
    "CollaborativeFilter": "ptpd_calibration.deep_learning.recipe_recommendation",
    "DeepPrintComparator": "ptpd_calibration.deep_learning.print_comparison",
    "LPIPSWrapper": "ptpd_calibration.deep_learning.print_comparison",
}


def __getattr__(name: str):
    """Lazy import implementation classes."""
    if name in __lazy_imports:
        import importlib

        module_path = __lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    "DeepLearningSettings",
    "DetectionModelSettings",
    "DiffusionSettings",
    "FederatedLearningSettings",
    "ImageQualitySettings",
    "MultiModalSettings",
    "NeuralCurveSettings",
    "PrintComparisonSettings",
    "RecipeRecommendationSettings",
    "DefectDetectionSettings",
    "UVExposureSettings",
    "get_deep_learning_settings",
    # Types
    "DetectionBackend",
    "SegmentationBackend",
    "IQAMetric",
    "QualityLevel",
    "DiffusionScheduler",
    "EnhancementMode",
    "CurvePredictorArchitecture",
    "UncertaintyMethod",
    "DefectType",
    "DefectSeverity",
    "RecommendationStrategy",
    "SimilarityMetric",
    "RecipeCategory",
    "PerceptualMetric",
    "ComparisonMode",
    "ComparisonResult",
    "VisionLanguageModel",
    "ToolType",
    "AggregationStrategy",
    "PrivacyLevel",
    # Models
    "DetectedPatch",
    "DeepDetectionResult",
    "ZoneQualityScore",
    "ImageQualityResult",
    "DiffusionEnhancementResult",
    "CurvePredictionResult",
    "EnhancementRegion",
    "UVExposurePrediction",
    "DetectedDefect",
    "DefectDetectionResult",
    "RecipeRecommendation",
    "RecipeRecommendationResult",
    "ZoneComparison",
    "PrintComparisonResult",
    # Implementation classes (lazy-loaded)
    "DeepTabletDetector",
    "VisionTransformerIQA",
    "DiffusionEnhancer",
    "NeuralCurvePredictor",
    "CurveTransformer",
    "UVExposurePredictor",
    "ExposureNet",
    "DefectDetector",
    "DefectSegmentationNet",
    "DefectClassifierNet",
    "RecipeRecommender",
    "RecipeEncoder",
    "ImageEncoder",
    "CollaborativeFilter",
    "DeepPrintComparator",
    "LPIPSWrapper",
]
