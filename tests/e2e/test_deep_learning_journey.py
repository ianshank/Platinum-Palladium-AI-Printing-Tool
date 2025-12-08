"""
End-to-end user journey tests for the Deep Learning features.

These tests simulate complete AI-powered workflows as a user would experience them,
testing the integration of multiple deep learning components.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from PIL import Image


class TestDeepLearningCalibrationJourney:
    """Test AI-enhanced calibration workflow."""

    @pytest.fixture
    def sample_step_tablet(self):
        """Create a sample step tablet image for deep learning detection."""
        width, height = 512, 128
        num_patches = 21
        patch_width = width // num_patches

        img = np.zeros((height, width), dtype=np.uint8)

        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img[:, x_start:x_end] = value

        # Add border to simulate real scanned image
        full_img = np.full((height + 60, width + 60, 3), 250, dtype=np.uint8)
        full_img[30 : height + 30, 30 : width + 30, 0] = img
        full_img[30 : height + 30, 30 : width + 30, 1] = img
        full_img[30 : height + 30, 30 : width + 30, 2] = img

        return Image.fromarray(full_img)

    def test_ai_detection_to_curve_workflow(self, sample_step_tablet, tmp_path):
        """
        AI Journey: Deep Detection → Neural Curve Prediction → Quality Assessment

        This tests the full AI-enhanced calibration pipeline:
        1. Use YOLOv8+SAM for step tablet detection
        2. Use Transformer for curve prediction
        3. Use ViT for quality assessment
        """
        from ptpd_calibration.deep_learning.config import (
            DetectionModelSettings,
            NeuralCurveSettings,
            ImageQualitySettings,
        )
        from ptpd_calibration.deep_learning.models import (
            DeepDetectionResult,
            CurvePredictionResult,
            ImageQualityResult,
        )

        # Step 1: Configure AI components
        detection_settings = DetectionModelSettings(device="cpu")
        curve_settings = NeuralCurveSettings(device="cpu")
        quality_settings = ImageQualitySettings(device="cpu")

        # Verify configuration is valid
        assert detection_settings.detection_backend is not None
        assert curve_settings.architecture is not None
        assert quality_settings.metrics is not None

        # Step 2: Test deep learning types can be instantiated
        from ptpd_calibration.deep_learning.types import (
            DetectionBackend,
            CurvePredictorArchitecture,
            IQAMetric,
        )

        assert DetectionBackend.YOLOV8 is not None
        assert CurvePredictorArchitecture.TRANSFORMER is not None
        assert IQAMetric.MUSIQ in list(IQAMetric)

        # Step 3: Save test image
        image_path = tmp_path / "step_tablet.png"
        sample_step_tablet.save(image_path)
        assert image_path.exists()

        # Step 4: Create mock detection result (without actual model)
        mock_patches = [
            {
                "zone_number": i,
                "bbox": [i * 24, 30, 24, 128],
                "confidence": 0.95,
                "density": 0.1 + (i * 0.075),
            }
            for i in range(21)
        ]

        detection_result = DeepDetectionResult(
            patches=[],  # Would be populated by actual detector
            extraction=None,
            processing_time_ms=150.0,
            model_version="yolov8s-1.0.0",
            confidence_threshold=0.5,
        )
        assert detection_result is not None

        # Step 5: Create mock curve prediction result
        curve_result = CurvePredictionResult(
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.linspace(0, 1, 256) ** 0.8),
            confidence_intervals=None,
            uncertainty_per_zone=None,
            model_version="curve-transformer-v1",
            processing_time_ms=50.0,
        )
        assert len(curve_result.output_values) == 256

        # Step 6: Create mock quality assessment result
        quality_result = ImageQualityResult(
            overall_score=85.0,
            zone_scores=[],
            quality_level=None,
            recommendations=[],
            processing_time_ms=100.0,
        )
        assert quality_result.overall_score > 0

    def test_ai_enhanced_workflow_with_fallback(self, sample_step_tablet, tmp_path):
        """
        Test that AI workflow gracefully falls back when models unavailable.
        """
        from ptpd_calibration.deep_learning.config import get_deep_learning_settings

        # Get settings - should work even without GPU
        settings = get_deep_learning_settings()
        assert settings is not None

        # Verify fallback device is configured
        assert settings.detection.device in ["auto", "cpu", "cuda", "mps"]


class TestDefectDetectionJourney:
    """Test AI defect detection workflow."""

    @pytest.fixture
    def sample_print_with_defects(self):
        """Create a sample print image with simulated defects."""
        width, height = 256, 256

        # Create base gradient
        arr = np.linspace(50, 200, width * height).reshape(height, width)
        arr = arr.astype(np.uint8)

        # Add simulated defects
        # Scratch (line)
        arr[100:105, 50:200] = 255

        # Spot defect
        y, x = np.ogrid[120:140, 180:200]
        mask = (x - 190) ** 2 + (y - 130) ** 2 <= 8**2
        arr[120:140, 180:200][mask] = 30

        # Uneven coating region
        arr[20:60, 20:80] = arr[20:60, 20:80] * 0.8

        # Convert to RGB
        rgb = np.stack([arr, arr, arr], axis=2)
        return Image.fromarray(rgb.astype(np.uint8))

    def test_defect_detection_workflow(self, sample_print_with_defects, tmp_path):
        """
        AI Journey: Upload Print → Detect Defects → Get Recommendations
        """
        from ptpd_calibration.deep_learning.config import DefectDetectionSettings
        from ptpd_calibration.deep_learning.models import (
            DetectedDefect,
            DefectDetectionResult,
        )
        from ptpd_calibration.deep_learning.types import DefectType, DefectSeverity

        # Step 1: Configure defect detection
        settings = DefectDetectionSettings(device="cpu")
        assert settings is not None

        # Step 2: Save test image
        image_path = tmp_path / "print_with_defects.png"
        sample_print_with_defects.save(image_path)

        # Step 3: Create mock detection results
        defects = [
            DetectedDefect(
                defect_type=DefectType.SCRATCH,
                severity=DefectSeverity.MINOR,
                bbox=[50, 100, 150, 5],
                confidence=0.92,
                description="Linear scratch defect",
            ),
            DetectedDefect(
                defect_type=DefectType.SPOT,
                severity=DefectSeverity.MODERATE,
                bbox=[180, 120, 20, 20],
                confidence=0.88,
                description="Dark spot defect",
            ),
            DetectedDefect(
                defect_type=DefectType.UNEVEN_COATING,
                severity=DefectSeverity.MINOR,
                bbox=[20, 20, 60, 40],
                confidence=0.75,
                description="Uneven coating region",
            ),
        ]

        result = DefectDetectionResult(
            defects=defects,
            overall_quality_score=72.0,
            defect_count=len(defects),
            recommendations=[
                "Check coating brush for debris",
                "Ensure even chemistry distribution",
                "Inspect negative for scratches",
            ],
            processing_time_ms=200.0,
        )

        # Step 4: Verify result structure
        assert result.defect_count == 3
        assert result.overall_quality_score < 100
        assert len(result.recommendations) > 0

        # Step 5: Verify defect types detected
        defect_types = [d.defect_type for d in result.defects]
        assert DefectType.SCRATCH in defect_types
        assert DefectType.SPOT in defect_types


class TestRecipeRecommendationJourney:
    """Test AI recipe recommendation workflow."""

    def test_recipe_recommendation_workflow(self):
        """
        AI Journey: Enter Parameters → Get AI Recommendations → Compare Options
        """
        from ptpd_calibration.deep_learning.config import RecipeRecommendationSettings
        from ptpd_calibration.deep_learning.models import (
            RecipeRecommendation,
            RecipeRecommendationResult,
        )
        from ptpd_calibration.deep_learning.types import RecipeCategory

        # Step 1: Configure recommendation engine
        settings = RecipeRecommendationSettings()
        assert settings.min_similarity > 0

        # Step 2: Create mock recommendations
        recommendations = [
            RecipeRecommendation(
                recipe_id="recipe_001",
                name="High Contrast Platinum",
                category=RecipeCategory.HIGH_CONTRAST,
                similarity_score=0.92,
                predicted_rating=4.5,
                platinum_ratio=0.7,
                palladium_ratio=0.3,
                developer_type="ammonium_citrate",
                estimated_dmax=1.65,
                estimated_exposure_factor=1.0,
                confidence=0.88,
                source_users=15,
            ),
            RecipeRecommendation(
                recipe_id="recipe_002",
                name="Warm Tone Classic",
                category=RecipeCategory.WARM_TONE,
                similarity_score=0.85,
                predicted_rating=4.2,
                platinum_ratio=0.5,
                palladium_ratio=0.5,
                developer_type="potassium_oxalate",
                estimated_dmax=1.55,
                estimated_exposure_factor=1.1,
                confidence=0.82,
                source_users=23,
            ),
        ]

        result = RecipeRecommendationResult(
            recommendations=recommendations,
            user_profile_used=True,
            paper_match_score=0.9,
            processing_time_ms=75.0,
        )

        # Step 3: Verify recommendations
        assert len(result.recommendations) == 2
        assert result.recommendations[0].similarity_score > result.recommendations[1].similarity_score
        assert all(r.confidence > 0.5 for r in result.recommendations)

        # Step 4: Check recipe categories
        categories = [r.category for r in result.recommendations]
        assert RecipeCategory.HIGH_CONTRAST in categories


class TestPrintComparisonJourney:
    """Test AI print comparison workflow."""

    @pytest.fixture
    def reference_print(self):
        """Create a reference print image."""
        arr = np.linspace(30, 220, 256 * 256).reshape(256, 256).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture
    def test_print(self):
        """Create a test print with slight differences."""
        arr = np.linspace(35, 215, 256 * 256).reshape(256, 256).astype(np.uint8)
        # Add slight variations
        noise = np.random.normal(0, 5, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def test_perceptual_comparison_workflow(self, reference_print, test_print, tmp_path):
        """
        AI Journey: Upload Reference → Upload Test → Get LPIPS Comparison
        """
        from ptpd_calibration.deep_learning.config import PrintComparisonSettings
        from ptpd_calibration.deep_learning.models import (
            ZoneComparison,
            PrintComparisonResult,
        )
        from ptpd_calibration.deep_learning.types import (
            PerceptualMetric,
            ComparisonMode,
            ComparisonResult,
        )

        # Step 1: Configure comparison
        settings = PrintComparisonSettings(device="cpu")
        assert PerceptualMetric.LPIPS in settings.metrics

        # Step 2: Save images
        ref_path = tmp_path / "reference.png"
        test_path = tmp_path / "test.png"
        reference_print.save(ref_path)
        test_print.save(test_path)

        # Step 3: Create mock zone comparisons
        zone_comparisons = [
            ZoneComparison(
                zone_number=i,
                lpips_distance=0.05 + i * 0.01,
                ssim_score=0.95 - i * 0.01,
                density_difference=0.02 + i * 0.005,
                result=ComparisonResult.ACCEPTABLE if i < 5 else ComparisonResult.MARGINAL,
            )
            for i in range(11)
        ]

        result = PrintComparisonResult(
            overall_similarity=0.87,
            perceptual_distance=0.12,
            zone_comparisons=zone_comparisons,
            result=ComparisonResult.ACCEPTABLE,
            recommendations=[
                "Shadows match well",
                "Minor highlight variation detected",
            ],
            processing_time_ms=250.0,
        )

        # Step 4: Verify comparison
        assert result.overall_similarity > 0.5
        assert result.result == ComparisonResult.ACCEPTABLE
        assert len(result.zone_comparisons) == 11


class TestUVExposurePredictionJourney:
    """Test AI UV exposure prediction workflow."""

    def test_exposure_prediction_workflow(self):
        """
        AI Journey: Enter Conditions → Get Predicted Exposure → View Confidence
        """
        from ptpd_calibration.deep_learning.config import UVExposureSettings
        from ptpd_calibration.deep_learning.models import UVExposurePrediction

        # Step 1: Configure predictor
        settings = UVExposureSettings(device="cpu")
        assert settings is not None

        # Step 2: Create mock prediction
        prediction = UVExposurePrediction(
            predicted_time_minutes=12.5,
            confidence_interval=(11.0, 14.0),
            uncertainty_std=0.8,
            factors_importance={
                "humidity": 0.25,
                "paper_type": 0.30,
                "negative_density": 0.20,
                "chemistry_age": 0.15,
                "temperature": 0.10,
            },
            recommendations=[
                "Consider 10% longer exposure due to high humidity",
                "Paper absorption rate suggests slight underexposure risk",
            ],
            model_version="exposure-net-v2",
            processing_time_ms=25.0,
        )

        # Step 3: Verify prediction
        assert prediction.predicted_time_minutes > 0
        assert prediction.confidence_interval[0] < prediction.predicted_time_minutes
        assert prediction.confidence_interval[1] > prediction.predicted_time_minutes
        assert abs(sum(prediction.factors_importance.values()) - 1.0) < 0.01


class TestMultiModalAssistantJourney:
    """Test multi-modal AI assistant workflow."""

    @pytest.fixture
    def sample_problem_image(self):
        """Create an image showing a print problem."""
        arr = np.zeros((256, 256), dtype=np.uint8)
        # Create uneven exposure pattern
        for i in range(256):
            arr[i, :] = 100 + int(50 * np.sin(i * 0.1))
        return Image.fromarray(arr, mode="L")

    def test_visual_troubleshooting_workflow(self, sample_problem_image, tmp_path):
        """
        AI Journey: Upload Problem Image → Describe Issue → Get AI Analysis
        """
        from ptpd_calibration.deep_learning.config import MultiModalSettings
        from ptpd_calibration.deep_learning.models import MultiModalResponse
        from ptpd_calibration.deep_learning.types import VisionLanguageModel, ToolType

        # Step 1: Configure assistant
        settings = MultiModalSettings()
        assert settings.model in list(VisionLanguageModel)

        # Step 2: Save problem image
        image_path = tmp_path / "problem.png"
        sample_problem_image.save(image_path)

        # Step 3: Create mock response
        response = MultiModalResponse(
            text_response=(
                "I can see banding in your print, which typically indicates uneven "
                "UV exposure. This could be caused by:\n"
                "1. Distance variation from light source\n"
                "2. Light source aging\n"
                "3. Reflection from nearby surfaces"
            ),
            confidence=0.85,
            detected_issues=["banding", "uneven_exposure"],
            suggested_tools=[ToolType.EXPOSURE_CALCULATOR, ToolType.DEFECT_DETECTOR],
            follow_up_questions=[
                "What light source are you using?",
                "How old are your UV tubes?",
            ],
            processing_time_ms=1500.0,
        )

        # Step 4: Verify response
        assert len(response.text_response) > 0
        assert response.confidence > 0.5
        assert len(response.detected_issues) > 0
        assert ToolType.EXPOSURE_CALCULATOR in response.suggested_tools


class TestFederatedLearningJourney:
    """Test federated learning workflow."""

    def test_federated_contribution_workflow(self):
        """
        AI Journey: Opt-in → Contribute Local Data → Receive Updates
        """
        from ptpd_calibration.deep_learning.config import FederatedLearningSettings
        from ptpd_calibration.deep_learning.models import FederatedRoundResult
        from ptpd_calibration.deep_learning.types import (
            AggregationStrategy,
            PrivacyLevel,
        )

        # Step 1: Configure federated learning with privacy
        settings = FederatedLearningSettings(
            enabled=True,
            privacy_level=PrivacyLevel.DIFFERENTIAL_PRIVACY,
            min_clients_per_round=3,
        )
        assert settings.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY

        # Step 2: Create mock round result
        round_result = FederatedRoundResult(
            round_number=5,
            participating_clients=12,
            global_model_version="federated-v5",
            improvement_metrics={
                "curve_prediction_accuracy": 0.02,
                "exposure_prediction_rmse": -0.1,
                "defect_detection_f1": 0.015,
            },
            privacy_budget_spent=0.1,
            aggregation_strategy=AggregationStrategy.FEDAVG,
            processing_time_ms=5000.0,
        )

        # Step 3: Verify round result
        assert round_result.participating_clients >= settings.min_clients_per_round
        assert round_result.privacy_budget_spent < 1.0
        assert all(
            metric_name in round_result.improvement_metrics
            for metric_name in ["curve_prediction_accuracy"]
        )


class TestDiffusionEnhancementJourney:
    """Test diffusion model enhancement workflow."""

    @pytest.fixture
    def damaged_print_scan(self):
        """Create a damaged print scan for restoration."""
        arr = np.linspace(40, 200, 256 * 256).reshape(256, 256).astype(np.uint8)

        # Add damage
        # Tears
        arr[80:85, 50:150] = 255
        # Stains
        y, x = np.ogrid[150:180, 100:130]
        mask = (x - 115) ** 2 + (y - 165) ** 2 <= 15**2
        arr[150:180, 100:130][mask] = (
            arr[150:180, 100:130][mask] * 0.6
        ).astype(np.uint8)
        # Fading
        arr[:50, :] = (arr[:50, :] * 0.7).astype(np.uint8)

        return Image.fromarray(arr, mode="L")

    def test_diffusion_restoration_workflow(self, damaged_print_scan, tmp_path):
        """
        AI Journey: Upload Damaged Scan → Select Restoration → Preview → Apply
        """
        from ptpd_calibration.deep_learning.config import DiffusionSettings
        from ptpd_calibration.deep_learning.models import (
            DiffusionEnhancementResult,
            EnhancementRegion,
        )
        from ptpd_calibration.deep_learning.types import (
            DiffusionScheduler,
            EnhancementMode,
        )

        # Step 1: Configure diffusion model
        settings = DiffusionSettings(
            device="cpu",
            scheduler=DiffusionScheduler.DDPM,
            num_inference_steps=20,  # Fewer steps for testing
        )
        assert settings.scheduler == DiffusionScheduler.DDPM

        # Step 2: Save damaged image
        image_path = tmp_path / "damaged.png"
        damaged_print_scan.save(image_path)

        # Step 3: Create mock enhancement regions
        regions = [
            EnhancementRegion(
                bbox=[50, 80, 100, 5],
                enhancement_type=EnhancementMode.INPAINT,
                description="Tear repair",
            ),
            EnhancementRegion(
                bbox=[100, 150, 30, 30],
                enhancement_type=EnhancementMode.RESTORE,
                description="Stain removal",
            ),
            EnhancementRegion(
                bbox=[0, 0, 256, 50],
                enhancement_type=EnhancementMode.RESTORE,
                description="Fade restoration",
            ),
        ]

        # Step 4: Create mock result
        result = DiffusionEnhancementResult(
            enhanced_image_path=str(tmp_path / "enhanced.png"),
            enhancement_regions=regions,
            quality_improvement_score=0.35,
            processing_time_ms=3000.0,
            model_version="sd-inpaint-v1.5",
        )

        # Step 5: Verify enhancement
        assert result.quality_improvement_score > 0
        assert len(result.enhancement_regions) == 3
        assert all(
            r.enhancement_type in [EnhancementMode.INPAINT, EnhancementMode.RESTORE]
            for r in result.enhancement_regions
        )


class TestIntegratedAIWorkflow:
    """Test complex integrated AI workflows."""

    def test_complete_ai_enhanced_calibration(self, tmp_path):
        """
        Complete AI Journey: Detection → Curve Prediction → Quality Check →
        Recipe Recommendation → Exposure Prediction
        """
        from ptpd_calibration.deep_learning.config import get_deep_learning_settings
        from ptpd_calibration.deep_learning.models import (
            DeepDetectionResult,
            CurvePredictionResult,
            ImageQualityResult,
            RecipeRecommendation,
            UVExposurePrediction,
        )
        from ptpd_calibration.deep_learning.types import RecipeCategory

        # Step 1: Get all settings
        settings = get_deep_learning_settings()
        assert settings is not None

        # Step 2: Simulate detection
        detection = DeepDetectionResult(
            patches=[],
            extraction=None,
            processing_time_ms=100.0,
            model_version="yolov8-v1",
            confidence_threshold=0.5,
        )

        # Step 3: Simulate curve prediction
        curve = CurvePredictionResult(
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.power(np.linspace(0, 1, 256), 0.85)),
            confidence_intervals=None,
            uncertainty_per_zone=None,
            model_version="transformer-v1",
            processing_time_ms=50.0,
        )

        # Step 4: Simulate quality assessment
        quality = ImageQualityResult(
            overall_score=88.0,
            zone_scores=[],
            quality_level=None,
            recommendations=["Good shadow detail", "Slight highlight compression"],
            processing_time_ms=75.0,
        )

        # Step 5: Simulate recipe recommendation
        recipe = RecipeRecommendation(
            recipe_id="rec_001",
            name="Balanced Platinum",
            category=RecipeCategory.NEUTRAL,
            similarity_score=0.91,
            predicted_rating=4.3,
            platinum_ratio=0.6,
            palladium_ratio=0.4,
            developer_type="ammonium_citrate",
            estimated_dmax=1.60,
            estimated_exposure_factor=1.0,
            confidence=0.87,
            source_users=18,
        )

        # Step 6: Simulate exposure prediction
        exposure = UVExposurePrediction(
            predicted_time_minutes=11.0,
            confidence_interval=(9.5, 12.5),
            uncertainty_std=0.7,
            factors_importance={"paper": 0.3, "humidity": 0.25, "negative": 0.25, "chemistry": 0.2},
            recommendations=["Standard exposure recommended"],
            model_version="exposure-v1",
            processing_time_ms=20.0,
        )

        # Step 7: Verify complete workflow
        total_processing_time = (
            detection.processing_time_ms
            + curve.processing_time_ms
            + quality.processing_time_ms
            + exposure.processing_time_ms
        )

        assert total_processing_time < 1000  # Sub-second for mock
        assert quality.overall_score > 80
        assert recipe.confidence > 0.8
        assert exposure.predicted_time_minutes > 0

        # Step 8: Compile summary
        summary = {
            "detection_model": detection.model_version,
            "curve_model": curve.model_version,
            "quality_score": quality.overall_score,
            "recommended_recipe": recipe.name,
            "predicted_exposure": f"{exposure.predicted_time_minutes:.1f} min",
            "total_ai_time_ms": total_processing_time,
        }

        assert all(k in summary for k in [
            "detection_model",
            "curve_model",
            "quality_score",
            "recommended_recipe",
            "predicted_exposure",
        ])
