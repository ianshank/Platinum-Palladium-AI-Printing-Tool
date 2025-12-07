"""
Tests for deep learning configuration.

Tests configuration settings, environment variable loading,
and default values for all deep learning components.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

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
    configure_deep_learning,
    get_deep_learning_settings,
)
from ptpd_calibration.deep_learning.types import (
    AggregationStrategy,
    ComparisonMode,
    ControlNetType,
    CurvePredictorArchitecture,
    DefectDetectorArchitecture,
    DetectionBackend,
    DiffusionModelType,
    DiffusionScheduler,
    IQAMetric,
    PerceptualMetric,
    PrivacyLevel,
    RecommendationStrategy,
    SegmentationBackend,
    SimilarityMetric,
    UncertaintyMethod,
    VisionLanguageModel,
)


class TestDetectionModelSettings:
    """Test DetectionModelSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = DetectionModelSettings()

        assert settings.detection_backend == DetectionBackend.YOLOV8
        assert settings.segmentation_backend == SegmentationBackend.SAM
        assert settings.yolo_model_size == "m"
        assert settings.yolo_confidence_threshold == 0.25
        assert settings.yolo_iou_threshold == 0.45
        assert settings.yolo_max_detections == 50
        assert settings.sam_model_type == "vit_h"
        assert settings.device == "auto"
        assert settings.half_precision is True
        assert settings.fallback_to_classical is True

    def test_custom_values(self):
        """Test custom configuration values."""
        settings = DetectionModelSettings(
            detection_backend=DetectionBackend.YOLOV9,
            yolo_model_size="l",
            yolo_confidence_threshold=0.5,
            device="cuda:0",
            half_precision=False,
        )

        assert settings.detection_backend == DetectionBackend.YOLOV9
        assert settings.yolo_model_size == "l"
        assert settings.yolo_confidence_threshold == 0.5
        assert settings.device == "cuda:0"
        assert settings.half_precision is False

    def test_validation_bounds(self):
        """Test validation of bounded values."""
        # Valid bounds
        settings = DetectionModelSettings(
            yolo_confidence_threshold=0.0,
            yolo_iou_threshold=1.0,
        )
        assert settings.yolo_confidence_threshold == 0.0
        assert settings.yolo_iou_threshold == 1.0

        # Invalid bounds should raise
        with pytest.raises(ValueError):
            DetectionModelSettings(yolo_confidence_threshold=1.5)

        with pytest.raises(ValueError):
            DetectionModelSettings(yolo_confidence_threshold=-0.1)

    def test_env_variable_override(self):
        """Test environment variable configuration."""
        with patch.dict(os.environ, {
            "PTPD_DL_DETECTION_DEVICE": "cuda:1",
            "PTPD_DL_DETECTION_YOLO_MODEL_SIZE": "x",
        }):
            settings = DetectionModelSettings()
            assert settings.device == "cuda:1"
            assert settings.yolo_model_size == "x"


class TestImageQualitySettings:
    """Test ImageQualitySettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = ImageQualitySettings()

        assert settings.primary_metric == IQAMetric.MANIQA
        assert IQAMetric.CLIP_IQA in settings.secondary_metrics
        assert settings.vit_model_name == "vit_base_patch16_224"
        assert settings.input_size == 224
        assert settings.analyze_zones is True
        assert settings.zone_count == 11
        assert settings.excellent_threshold == 0.9
        assert settings.cache_embeddings is True

    def test_zone_count_validation(self):
        """Test zone count bounds."""
        settings = ImageQualitySettings(zone_count=5)
        assert settings.zone_count == 5

        settings = ImageQualitySettings(zone_count=21)
        assert settings.zone_count == 21

        with pytest.raises(ValueError):
            ImageQualitySettings(zone_count=4)

        with pytest.raises(ValueError):
            ImageQualitySettings(zone_count=22)


class TestDiffusionSettings:
    """Test DiffusionSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = DiffusionSettings()

        assert settings.model_type == DiffusionModelType.STABLE_DIFFUSION_XL
        assert settings.scheduler == DiffusionScheduler.EULER
        assert settings.use_controlnet is True
        assert settings.controlnet_type == ControlNetType.CANNY
        assert settings.num_inference_steps == 30
        assert settings.guidance_scale == 7.5
        assert settings.strength == 0.5
        assert settings.enable_attention_slicing is True
        assert settings.enable_vae_slicing is True

    def test_inference_steps_validation(self):
        """Test inference steps bounds."""
        settings = DiffusionSettings(num_inference_steps=10)
        assert settings.num_inference_steps == 10

        settings = DiffusionSettings(num_inference_steps=150)
        assert settings.num_inference_steps == 150

        with pytest.raises(ValueError):
            DiffusionSettings(num_inference_steps=5)


class TestNeuralCurveSettings:
    """Test NeuralCurveSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = NeuralCurveSettings()

        assert settings.architecture == CurvePredictorArchitecture.TRANSFORMER
        assert settings.d_model == 256
        assert settings.n_heads == 8
        assert settings.n_layers == 6
        assert settings.output_points == 256
        assert settings.uncertainty_method == UncertaintyMethod.ENSEMBLE
        assert settings.ensemble_size == 5

    def test_transformer_dimensions(self):
        """Test transformer dimension constraints."""
        settings = NeuralCurveSettings(
            d_model=512,
            n_heads=16,
            n_layers=12,
        )
        assert settings.d_model == 512
        assert settings.n_heads == 16
        assert settings.n_layers == 12


class TestUVExposureSettings:
    """Test UVExposureSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = UVExposureSettings()

        assert settings.model_architecture == "mlp_with_residual"
        assert settings.hidden_layers == [256, 128, 64]
        assert settings.activation == "gelu"
        assert settings.predict_confidence_interval is True
        assert settings.confidence_level == 0.95
        assert settings.uncertainty_method == UncertaintyMethod.ENSEMBLE

    def test_input_features(self):
        """Test input feature configuration."""
        settings = UVExposureSettings()

        expected_features = [
            "target_density",
            "paper_type",
            "chemistry_ratio",
            "uv_source",
            "humidity",
            "temperature",
            "coating_thickness",
            "negative_dmax",
        ]
        assert settings.input_features == expected_features


class TestDefectDetectionSettings:
    """Test DefectDetectionSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = DefectDetectionSettings()

        assert settings.segmentation_architecture == DefectDetectorArchitecture.UNET_PLUS_PLUS
        assert settings.segmentation_encoder == "resnet50"
        assert settings.classifier_backbone == "resnet34"
        assert settings.num_classes == 25
        assert settings.confidence_threshold == 0.5
        assert settings.use_multi_scale is True
        assert settings.estimate_severity is True


class TestRecipeRecommendationSettings:
    """Test RecipeRecommendationSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = RecipeRecommendationSettings()

        assert settings.strategy == RecommendationStrategy.HYBRID
        assert settings.similarity_metric == SimilarityMetric.COSINE
        assert settings.recipe_embedding_dim == 128
        assert settings.top_k == 10
        assert settings.generate_explanations is True


class TestPrintComparisonSettings:
    """Test PrintComparisonSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = PrintComparisonSettings()

        assert settings.primary_metric == PerceptualMetric.LPIPS_ALEX
        assert settings.comparison_mode == ComparisonMode.ZONE_BASED
        assert settings.lpips_net == "alex"
        assert settings.use_multi_scale is True
        assert settings.generate_attention_maps is True


class TestMultiModalSettings:
    """Test MultiModalSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = MultiModalSettings()

        assert settings.vision_language_model == VisionLanguageModel.CLAUDE_4_SONNET
        assert settings.fallback_model == VisionLanguageModel.GPT_4O
        assert settings.max_image_size == 2048
        assert settings.image_detail == "high"
        assert settings.use_rag is True
        assert settings.stream_response is True

    def test_enabled_tools(self):
        """Test default enabled tools."""
        settings = MultiModalSettings()

        expected_tools = [
            "exposure_calculator",
            "chemistry_calculator",
            "curve_adjustment",
            "defect_diagnosis",
            "recipe_lookup",
            "quality_assessment",
        ]
        assert settings.enabled_tools == expected_tools


class TestFederatedLearningSettings:
    """Test FederatedLearningSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = FederatedLearningSettings()

        assert settings.enabled is False
        assert settings.aggregation_strategy == AggregationStrategy.FEDAVG
        assert settings.privacy_level == PrivacyLevel.DIFFERENTIAL
        assert settings.differential_privacy_epsilon == 1.0
        assert settings.gradient_compression is True
        assert settings.auto_participate is False

    def test_privacy_bounds(self):
        """Test differential privacy bounds."""
        settings = FederatedLearningSettings(
            differential_privacy_epsilon=0.1,
            differential_privacy_delta=1e-10,
        )
        assert settings.differential_privacy_epsilon == 0.1
        assert settings.differential_privacy_delta == 1e-10


class TestDeepLearningSettings:
    """Test main DeepLearningSettings aggregator."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = DeepLearningSettings()

        assert settings.enabled is True
        assert settings.default_device == "auto"
        assert settings.cache_models is True
        assert settings.download_models is True
        assert settings.offline_mode is False

    def test_subsettings_present(self):
        """Test that all subsettings are present."""
        settings = DeepLearningSettings()

        assert isinstance(settings.detection, DetectionModelSettings)
        assert isinstance(settings.image_quality, ImageQualitySettings)
        assert isinstance(settings.diffusion, DiffusionSettings)
        assert isinstance(settings.neural_curve, NeuralCurveSettings)
        assert isinstance(settings.uv_exposure, UVExposureSettings)
        assert isinstance(settings.defect_detection, DefectDetectionSettings)
        assert isinstance(settings.recipe_recommendation, RecipeRecommendationSettings)
        assert isinstance(settings.print_comparison, PrintComparisonSettings)
        assert isinstance(settings.multimodal, MultiModalSettings)
        assert isinstance(settings.federated, FederatedLearningSettings)

    def test_model_cache_dir_resolution(self):
        """Test cache directory path resolution."""
        settings = DeepLearningSettings()
        expected_default = Path.home() / ".ptpd" / "models"
        assert settings.model_cache_dir == expected_default

    def test_get_device(self):
        """Test device selection."""
        settings = DeepLearningSettings(default_device="cpu")
        assert settings.get_device() == "cpu"

        settings = DeepLearningSettings(default_device="cuda:0")
        assert settings.get_device() == "cuda:0"


class TestGlobalSettings:
    """Test global settings functions."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns singleton."""
        settings1 = get_deep_learning_settings()
        settings2 = get_deep_learning_settings()
        assert settings1 is settings2

    def test_configure_settings(self):
        """Test settings configuration."""
        new_settings = DeepLearningSettings(
            enabled=False,
            default_device="cpu",
        )
        configured = configure_deep_learning(new_settings)
        assert configured.enabled is False
        assert configured.default_device == "cpu"

    def test_configure_with_kwargs(self):
        """Test settings configuration with kwargs."""
        configured = configure_deep_learning(
            enabled=True,
            offline_mode=True,
        )
        assert configured.enabled is True
        assert configured.offline_mode is True
