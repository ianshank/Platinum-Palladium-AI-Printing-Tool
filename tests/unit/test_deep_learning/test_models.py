"""
Tests for deep learning Pydantic models.

Tests model validation, serialization, and methods
for all deep learning result models.
"""

from datetime import datetime
from uuid import UUID, uuid4

import numpy as np

from ptpd_calibration.deep_learning.models import (
    BaseAIResult,
    CurvePredictionResult,
    DeepDetectionResult,
    DefectDetectionResult,
    DetectedDefect,
    DetectedPatch,
    DiffusionEnhancementResult,
    FederatedRoundResult,
    FederatedUpdate,
    ImageAnalysis,
    ImageQualityResult,
    MultiModalResponse,
    PrintComparisonResult,
    RecipeRecommendation,
    RecipeRecommendationResult,
    ToolCall,
    UVExposurePrediction,
    ZoneComparison,
    ZoneQualityScore,
)
from ptpd_calibration.deep_learning.types import (
    ComparisonResult,
    DefectSeverity,
    DefectType,
    EnhancementMode,
    QualityLevel,
    RecipeCategory,
)


class TestBaseAIResult:
    """Test BaseAIResult model."""

    def test_default_fields(self):
        """Test default field values."""

        class TestResult(BaseAIResult):
            pass

        result = TestResult()

        assert isinstance(result.id, UUID)
        assert isinstance(result.timestamp, datetime)
        assert result.inference_time_ms == 0.0
        assert result.device_used == "cpu"
        assert result.model_version == "1.0.0"

    def test_custom_fields(self):
        """Test custom field values."""

        class TestResult(BaseAIResult):
            pass

        custom_id = uuid4()
        result = TestResult(
            id=custom_id,
            inference_time_ms=123.45,
            device_used="cuda:0",
            model_version="2.0.0",
        )

        assert result.id == custom_id
        assert result.inference_time_ms == 123.45
        assert result.device_used == "cuda:0"
        assert result.model_version == "2.0.0"


class TestDetectedPatch:
    """Test DetectedPatch model."""

    def test_basic_patch(self):
        """Test basic patch creation."""
        patch = DetectedPatch(
            index=0,
            bbox=(10, 20, 100, 50),
            confidence=0.95,
        )

        assert patch.index == 0
        assert patch.bbox == (10, 20, 100, 50)
        assert patch.confidence == 0.95
        assert patch.mask is None
        assert patch.mask_area == 0

    def test_bbox_conversion_from_list(self):
        """Test bbox conversion from list."""
        patch = DetectedPatch(
            index=0,
            bbox=[10, 20, 100, 50],
            confidence=0.9,
        )

        assert patch.bbox == (10, 20, 100, 50)

    def test_bbox_conversion_from_numpy(self):
        """Test bbox conversion from numpy array."""
        bbox_array = np.array([10, 20, 100, 50])
        patch = DetectedPatch(
            index=0,
            bbox=bbox_array,
            confidence=0.9,
        )

        assert patch.bbox == (10, 20, 100, 50)

    def test_with_mask(self):
        """Test patch with segmentation mask."""
        mask = np.ones((50, 100), dtype=np.uint8)
        patch = DetectedPatch(
            index=0,
            bbox=(10, 20, 100, 50),
            confidence=0.95,
            mask=mask,
            mask_area=5000,
        )

        assert patch.mask is not None
        assert patch.mask.shape == (50, 100)
        assert patch.mask_area == 5000


class TestDeepDetectionResult:
    """Test DeepDetectionResult model."""

    def test_basic_result(self):
        """Test basic detection result."""
        result = DeepDetectionResult(
            tablet_bbox=(0, 0, 640, 480),
            tablet_confidence=0.98,
            rotation_angle=0.5,
            orientation="horizontal",
            num_patches=21,
        )

        assert result.tablet_bbox == (0, 0, 640, 480)
        assert result.tablet_confidence == 0.98
        assert result.rotation_angle == 0.5
        assert result.num_patches == 21
        assert result.used_fallback is False

    def test_with_patches(self):
        """Test result with patches."""
        patches = [
            DetectedPatch(index=i, bbox=(i * 30, 0, 30, 100), confidence=0.9) for i in range(21)
        ]

        result = DeepDetectionResult(
            tablet_bbox=(0, 0, 630, 100),
            tablet_confidence=0.95,
            patches=patches,
            num_patches=21,
        )

        assert len(result.patches) == 21
        assert result.get_patch_bounds()[0] == (0, 0, 30, 100)

    def test_fallback_mode(self):
        """Test fallback detection mode."""
        result = DeepDetectionResult(
            tablet_bbox=(0, 0, 640, 480),
            tablet_confidence=0.7,
            used_fallback=True,
            fallback_reason="YOLOv8 unavailable",
        )

        assert result.used_fallback is True
        assert result.fallback_reason == "YOLOv8 unavailable"


class TestImageQualityResult:
    """Test ImageQualityResult model."""

    def test_basic_result(self):
        """Test basic quality result."""
        result = ImageQualityResult(
            overall_score=0.85,
            quality_level=QualityLevel.GOOD,
        )

        assert result.overall_score == 0.85
        assert result.quality_level == QualityLevel.GOOD

    def test_with_zone_scores(self):
        """Test result with zone scores."""
        zone_scores = [
            ZoneQualityScore(
                zone=i,
                zone_name=f"Zone {i}",
                score=0.9 - i * 0.05,
                pixel_percentage=9.0,
            )
            for i in range(11)
        ]

        result = ImageQualityResult(
            overall_score=0.85,
            quality_level=QualityLevel.GOOD,
            zone_scores=zone_scores,
            highlight_quality=0.9,
            midtone_quality=0.85,
            shadow_quality=0.8,
        )

        assert len(result.zone_scores) == 11
        assert result.highlight_quality == 0.9

    def test_with_recommendations(self):
        """Test result with recommendations."""
        result = ImageQualityResult(
            overall_score=0.6,
            quality_level=QualityLevel.ACCEPTABLE,
            recommendations=["Increase exposure", "Check coating uniformity"],
            issues=["Low shadow detail", "Slight noise in highlights"],
        )

        assert len(result.recommendations) == 2
        assert len(result.issues) == 2


class TestDiffusionEnhancementResult:
    """Test DiffusionEnhancementResult model."""

    def test_basic_result(self):
        """Test basic enhancement result."""
        result = DiffusionEnhancementResult(
            original_size=(1024, 768),
            output_size=(1024, 768),
            enhancement_mode=EnhancementMode.TONAL_ENHANCEMENT,
        )

        assert result.original_size == (1024, 768)
        assert result.enhancement_mode == EnhancementMode.TONAL_ENHANCEMENT
        assert result.structure_preservation == 1.0

    def test_with_enhanced_image(self):
        """Test result with enhanced image."""
        enhanced = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)

        result = DiffusionEnhancementResult(
            enhanced_image=enhanced,
            original_size=(1024, 768),
            output_size=(1024, 768),
            enhancement_mode=EnhancementMode.INPAINTING,
            quality_improvement=15.5,
        )

        assert result.enhanced_image is not None
        assert result.enhanced_image.shape == (768, 1024, 3)
        assert result.quality_improvement == 15.5


class TestCurvePredictionResult:
    """Test CurvePredictionResult model."""

    def test_basic_result(self):
        """Test basic curve prediction."""
        input_vals = list(np.linspace(0, 1, 256))
        output_vals = list(np.linspace(0, 1, 256) ** 0.9)

        result = CurvePredictionResult(
            input_values=input_vals,
            output_values=output_vals,
            num_points=256,
        )

        assert len(result.input_values) == 256
        assert len(result.output_values) == 256
        assert result.is_monotonic is True

    def test_numpy_conversion(self):
        """Test numpy array conversion."""
        input_vals = np.linspace(0, 1, 256)
        output_vals = np.linspace(0, 1, 256) ** 0.9

        result = CurvePredictionResult(
            input_values=input_vals,
            output_values=output_vals,
            num_points=256,
        )

        x, y = result.to_numpy()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == (256,)

    def test_with_uncertainty(self):
        """Test result with uncertainty."""
        result = CurvePredictionResult(
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.4, 1.0],
            num_points=3,
            uncertainty=[0.02, 0.03, 0.02],
            mean_uncertainty=0.023,
            confidence=0.92,
        )

        assert result.uncertainty is not None
        assert len(result.uncertainty) == 3
        assert result.mean_uncertainty == 0.023


class TestUVExposurePrediction:
    """Test UVExposurePrediction model."""

    def test_basic_prediction(self):
        """Test basic exposure prediction."""
        result = UVExposurePrediction(
            predicted_seconds=180.0,
            predicted_minutes=3.0,
            lower_bound_seconds=150.0,
            upper_bound_seconds=210.0,
            base_exposure=180.0,
        )

        assert result.predicted_seconds == 180.0
        assert result.predicted_minutes == 3.0
        assert result.confidence_level == 0.95

    def test_format_time(self):
        """Test time formatting."""
        result = UVExposurePrediction(
            predicted_seconds=45.0,
            predicted_minutes=0.75,
            lower_bound_seconds=40.0,
            upper_bound_seconds=50.0,
            base_exposure=45.0,
        )
        assert result.format_time() == "45 seconds"

        result = UVExposurePrediction(
            predicted_seconds=180.0,
            predicted_minutes=3.0,
            lower_bound_seconds=150.0,
            upper_bound_seconds=210.0,
            base_exposure=180.0,
        )
        assert result.format_time() == "3 minutes"

        result = UVExposurePrediction(
            predicted_seconds=210.0,
            predicted_minutes=3.5,
            lower_bound_seconds=180.0,
            upper_bound_seconds=240.0,
            base_exposure=210.0,
        )
        assert result.format_time() == "3 min 30 sec"


class TestDefectDetectionResult:
    """Test DefectDetectionResult model."""

    def test_no_defects(self):
        """Test result with no defects."""
        result = DefectDetectionResult(
            defects=[],
            num_defects=0,
            overall_severity=DefectSeverity.NEGLIGIBLE,
            print_acceptable=True,
        )

        assert result.num_defects == 0
        assert result.print_acceptable is True

    def test_with_defects(self):
        """Test result with defects."""
        defects = [
            DetectedDefect(
                defect_type=DefectType.BRUSH_MARK,
                severity=DefectSeverity.MINOR,
                confidence=0.85,
                bbox=(100, 200, 50, 30),
                area_pixels=1500,
            ),
            DetectedDefect(
                defect_type=DefectType.DUST,
                severity=DefectSeverity.NEGLIGIBLE,
                confidence=0.9,
                bbox=(300, 400, 20, 20),
                area_pixels=400,
            ),
        ]

        result = DefectDetectionResult(
            defects=defects,
            num_defects=2,
            defects_by_type={"brush_mark": 1, "dust": 1},
            defects_by_severity={"minor": 1, "negligible": 1},
            overall_severity=DefectSeverity.MINOR,
            print_acceptable=True,
        )

        assert result.num_defects == 2
        assert len(result.defects) == 2


class TestRecipeRecommendationResult:
    """Test RecipeRecommendationResult model."""

    def test_basic_result(self):
        """Test basic recommendation result."""
        recommendations = [
            RecipeRecommendation(
                recipe_id=uuid4(),
                recipe_name="Warm Portrait",
                similarity_score=0.95,
                rank=1,
                paper_type="Arches Platine",
                chemistry_type="platinum_palladium",
                metal_ratio=0.3,
                exposure_time=180.0,
                categories=[RecipeCategory.PORTRAIT, RecipeCategory.WARM_TONE],
                explanation="Similar tone and paper type",
            )
        ]

        result = RecipeRecommendationResult(
            recommendations=recommendations,
            num_recommendations=1,
        )

        assert result.num_recommendations == 1
        assert result.recommendations[0].rank == 1


class TestPrintComparisonResult:
    """Test PrintComparisonResult model."""

    def test_basic_result(self):
        """Test basic comparison result."""
        result = PrintComparisonResult(
            overall_similarity=0.92,
            comparison_result=ComparisonResult.VERY_SIMILAR,
            lpips_score=0.08,
            ssim_score=0.94,
            psnr_score=32.5,
        )

        assert result.overall_similarity == 0.92
        assert result.comparison_result == ComparisonResult.VERY_SIMILAR
        assert result.lpips_score == 0.08

    def test_with_zone_comparisons(self):
        """Test result with zone comparisons."""
        zone_comparisons = [
            ZoneComparison(
                zone="shadows",
                similarity=0.9,
                lpips_score=0.1,
                ssim_score=0.92,
            ),
            ZoneComparison(
                zone="midtones",
                similarity=0.95,
                lpips_score=0.05,
                ssim_score=0.96,
            ),
            ZoneComparison(
                zone="highlights",
                similarity=0.88,
                lpips_score=0.12,
                ssim_score=0.9,
            ),
        ]

        result = PrintComparisonResult(
            overall_similarity=0.91,
            comparison_result=ComparisonResult.VERY_SIMILAR,
            lpips_score=0.09,
            ssim_score=0.93,
            psnr_score=31.0,
            zone_comparisons=zone_comparisons,
        )

        assert len(result.zone_comparisons) == 3


class TestMultiModalResponse:
    """Test MultiModalResponse model."""

    def test_basic_response(self):
        """Test basic response."""
        response = MultiModalResponse(
            response_text="Based on your image, I recommend...",
            response_type="analysis",
        )

        assert "recommend" in response.response_text
        assert response.response_type == "analysis"
        assert response.num_tool_calls == 0

    def test_with_tool_calls(self):
        """Test response with tool calls."""
        tool_calls = [
            ToolCall(
                tool_name="exposure_calculator",
                tool_input={"target_density": 1.8},
                tool_output={"exposure_seconds": 180},
                execution_time_ms=50.0,
                success=True,
            )
        ]

        response = MultiModalResponse(
            response_text="The recommended exposure is 3 minutes.",
            tool_calls=tool_calls,
            num_tool_calls=1,
        )

        assert response.num_tool_calls == 1
        assert response.tool_calls[0].tool_name == "exposure_calculator"

    def test_with_image_analysis(self):
        """Test response with image analysis."""
        analyses = [
            ImageAnalysis(
                image_index=0,
                description="A step tablet print showing good tonal separation",
                detected_issues=["Slight brush marks in zone 3"],
                recommendations=["Consider using glass rod coating"],
            )
        ]

        response = MultiModalResponse(
            response_text="I've analyzed your print.",
            image_analyses=analyses,
            images_analyzed=1,
        )

        assert response.images_analyzed == 1


class TestFederatedModels:
    """Test federated learning models."""

    def test_federated_update(self):
        """Test federated update model."""
        update = FederatedUpdate(
            client_id="client_001",
            round_number=5,
            num_samples=100,
            local_loss=0.125,
            local_accuracy=0.92,
            training_time_seconds=45.5,
        )

        assert update.client_id == "client_001"
        assert update.round_number == 5
        assert update.num_samples == 100

    def test_federated_round_result(self):
        """Test federated round result model."""
        updates = [
            FederatedUpdate(
                client_id=f"client_{i}",
                round_number=1,
                num_samples=100,
                local_loss=0.1 + i * 0.01,
            )
            for i in range(5)
        ]

        result = FederatedRoundResult(
            round_number=1,
            num_participants=5,
            aggregation_strategy="fedavg",
            global_loss=0.11,
            global_accuracy=0.89,
            participant_updates=updates,
            privacy_level="differential",
            noise_added=True,
        )

        assert result.num_participants == 5
        assert result.global_loss == 0.11
        assert result.noise_added is True
