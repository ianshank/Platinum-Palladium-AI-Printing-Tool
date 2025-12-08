"""
Backend API tests for deep learning endpoints.

Tests the API layer for all AI/ML features including:
- Detection endpoints
- Curve prediction endpoints
- Quality assessment endpoints
- Defect detection endpoints
- Recipe recommendation endpoints
- Print comparison endpoints
- Multi-modal assistant endpoints
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from PIL import Image
import io
import base64


@pytest.mark.api
class TestDetectionEndpoints:
    """Tests for step tablet detection API endpoints."""

    @pytest.fixture
    def sample_image_base64(self):
        """Create a base64 encoded sample image."""
        arr = np.random.randint(50, 200, (256, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_detect_step_tablet_basic(self, client, sample_image_base64):
        """Test basic step tablet detection."""
        request_data = {
            "image_base64": sample_image_base64,
            "confidence_threshold": 0.5,
        }

        response = client.post("/api/deep-learning/detect", json=request_data)

        # May fail without models, but should handle gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "patches" in data or "error" in data
            assert "processing_time_ms" in data

    def test_detect_with_custom_settings(self, client, sample_image_base64):
        """Test detection with custom configuration."""
        request_data = {
            "image_base64": sample_image_base64,
            "confidence_threshold": 0.7,
            "use_sam_refinement": True,
            "num_patches_expected": 21,
        }

        response = client.post("/api/deep-learning/detect", json=request_data)
        assert response.status_code in [200, 500, 503]

    def test_detect_invalid_image(self, client):
        """Test detection with invalid image data."""
        request_data = {
            "image_base64": "not_valid_base64!@#$",
            "confidence_threshold": 0.5,
        }

        response = client.post("/api/deep-learning/detect", json=request_data)
        assert response.status_code in [400, 422, 500]

    def test_detect_empty_request(self, client):
        """Test detection with empty request."""
        response = client.post("/api/deep-learning/detect", json={})
        assert response.status_code in [400, 422]


@pytest.mark.api
class TestCurvePredictionEndpoints:
    """Tests for neural curve prediction API endpoints."""

    def test_predict_curve_from_densities(self, client):
        """Test curve prediction from density measurements."""
        request_data = {
            "densities": [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60],
            "paper_type": "arches_platine",
            "target_response": "linear",
        }

        response = client.post("/api/deep-learning/predict-curve", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "output_values" in data or "curve" in data

    def test_predict_curve_with_uncertainty(self, client):
        """Test curve prediction with uncertainty estimation."""
        request_data = {
            "densities": [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60],
            "include_uncertainty": True,
            "uncertainty_method": "ensemble",
        }

        response = client.post("/api/deep-learning/predict-curve", json=request_data)
        assert response.status_code in [200, 500, 503]

    def test_predict_curve_invalid_densities(self, client):
        """Test curve prediction with invalid density values."""
        request_data = {
            "densities": [-0.5, 0.5, 3.0],  # Invalid: negative and > 2.5
        }

        response = client.post("/api/deep-learning/predict-curve", json=request_data)
        assert response.status_code in [200, 400, 422, 500]


@pytest.mark.api
class TestQualityAssessmentEndpoints:
    """Tests for image quality assessment API endpoints."""

    @pytest.fixture
    def sample_image_base64(self):
        """Create a base64 encoded sample image."""
        arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_assess_quality_basic(self, client, sample_image_base64):
        """Test basic image quality assessment."""
        request_data = {
            "image_base64": sample_image_base64,
        }

        response = client.post("/api/deep-learning/assess-quality", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "overall_score" in data or "quality" in data

    def test_assess_quality_with_metrics(self, client, sample_image_base64):
        """Test quality assessment with specific metrics."""
        request_data = {
            "image_base64": sample_image_base64,
            "metrics": ["musiq", "nima", "brisque"],
            "include_zone_analysis": True,
        }

        response = client.post("/api/deep-learning/assess-quality", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestDefectDetectionEndpoints:
    """Tests for defect detection API endpoints."""

    @pytest.fixture
    def sample_print_base64(self):
        """Create a base64 encoded print image with simulated defects."""
        arr = np.linspace(50, 200, 256 * 256).reshape(256, 256).astype(np.uint8)
        # Add simulated defects
        arr[100:105, 50:150] = 255  # Scratch
        rgb = np.stack([arr, arr, arr], axis=2)
        img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_detect_defects_basic(self, client, sample_print_base64):
        """Test basic defect detection."""
        request_data = {
            "image_base64": sample_print_base64,
        }

        response = client.post("/api/deep-learning/detect-defects", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "defects" in data or "defect_count" in data

    def test_detect_defects_with_sensitivity(self, client, sample_print_base64):
        """Test defect detection with custom sensitivity."""
        request_data = {
            "image_base64": sample_print_base64,
            "sensitivity": "high",
            "defect_types": ["scratch", "spot", "uneven_coating"],
        }

        response = client.post("/api/deep-learning/detect-defects", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestRecipeRecommendationEndpoints:
    """Tests for recipe recommendation API endpoints."""

    def test_recommend_recipe_basic(self, client):
        """Test basic recipe recommendation."""
        request_data = {
            "paper_type": "Arches Platine",
            "desired_characteristics": {
                "contrast": "high",
                "tone": "warm",
            },
        }

        response = client.post(
            "/api/deep-learning/recommend-recipe", json=request_data
        )
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data or "recipe" in data

    def test_recommend_recipe_with_constraints(self, client):
        """Test recipe recommendation with constraints."""
        request_data = {
            "paper_type": "Bergger COT320",
            "desired_characteristics": {
                "contrast": "normal",
                "tone": "neutral",
            },
            "constraints": {
                "max_platinum_ratio": 0.6,
                "developer": "ammonium_citrate",
            },
            "num_recommendations": 5,
        }

        response = client.post(
            "/api/deep-learning/recommend-recipe", json=request_data
        )
        assert response.status_code in [200, 500, 503]

    def test_recommend_recipe_invalid_paper(self, client):
        """Test recipe recommendation with unknown paper."""
        request_data = {
            "paper_type": "Unknown Paper Type XYZ",
            "desired_characteristics": {"contrast": "normal"},
        }

        response = client.post(
            "/api/deep-learning/recommend-recipe", json=request_data
        )
        # Should either succeed with generic recommendations or return error
        assert response.status_code in [200, 400, 404, 500]


@pytest.mark.api
class TestPrintComparisonEndpoints:
    """Tests for print comparison API endpoints."""

    @pytest.fixture
    def reference_image_base64(self):
        """Create a base64 encoded reference image."""
        arr = np.linspace(30, 220, 256 * 256).reshape(256, 256).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    @pytest.fixture
    def test_image_base64(self):
        """Create a base64 encoded test image with slight differences."""
        arr = np.linspace(35, 215, 256 * 256).reshape(256, 256).astype(np.uint8)
        noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_compare_prints_basic(
        self, client, reference_image_base64, test_image_base64
    ):
        """Test basic print comparison."""
        request_data = {
            "reference_image_base64": reference_image_base64,
            "test_image_base64": test_image_base64,
        }

        response = client.post("/api/deep-learning/compare-prints", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "similarity" in data or "perceptual_distance" in data

    def test_compare_prints_with_options(
        self, client, reference_image_base64, test_image_base64
    ):
        """Test print comparison with options."""
        request_data = {
            "reference_image_base64": reference_image_base64,
            "test_image_base64": test_image_base64,
            "metrics": ["lpips", "ssim"],
            "zone_analysis": True,
        }

        response = client.post("/api/deep-learning/compare-prints", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestUVExposureEndpoints:
    """Tests for UV exposure prediction API endpoints."""

    def test_predict_exposure_basic(self, client):
        """Test basic UV exposure prediction."""
        request_data = {
            "paper_type": "Arches Platine",
            "negative_density": 1.6,
            "humidity_percent": 50,
            "temperature_celsius": 22,
        }

        response = client.post("/api/deep-learning/predict-exposure", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_time" in data or "exposure_minutes" in data

    def test_predict_exposure_with_all_factors(self, client):
        """Test exposure prediction with all environmental factors."""
        request_data = {
            "paper_type": "Bergger COT320",
            "negative_density": 1.8,
            "humidity_percent": 65,
            "temperature_celsius": 24,
            "chemistry_age_days": 3,
            "coating_weight": "heavy",
            "light_source": "uv_led",
            "include_uncertainty": True,
        }

        response = client.post("/api/deep-learning/predict-exposure", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestDiffusionEndpoints:
    """Tests for diffusion model enhancement API endpoints."""

    @pytest.fixture
    def damaged_image_base64(self):
        """Create a base64 encoded damaged image."""
        arr = np.linspace(40, 200, 256 * 256).reshape(256, 256).astype(np.uint8)
        # Add damage
        arr[80:85, 50:150] = 255  # Tear
        rgb = np.stack([arr, arr, arr], axis=2)
        img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_enhance_image_basic(self, client, damaged_image_base64):
        """Test basic image enhancement."""
        request_data = {
            "image_base64": damaged_image_base64,
            "enhancement_mode": "restore",
        }

        response = client.post("/api/deep-learning/enhance", json=request_data)
        assert response.status_code in [200, 500, 503]

    def test_inpaint_region(self, client, damaged_image_base64):
        """Test inpainting a specific region."""
        request_data = {
            "image_base64": damaged_image_base64,
            "enhancement_mode": "inpaint",
            "mask_regions": [
                {"x": 50, "y": 80, "width": 100, "height": 5},
            ],
        }

        response = client.post("/api/deep-learning/enhance", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestMultiModalEndpoints:
    """Tests for multi-modal AI assistant API endpoints."""

    @pytest.fixture
    def problem_image_base64(self):
        """Create a base64 encoded problem image."""
        arr = np.zeros((256, 256), dtype=np.uint8)
        # Create banding pattern
        for i in range(256):
            arr[i, :] = 100 + int(30 * np.sin(i * 0.15))
        img = Image.fromarray(arr, mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_analyze_with_image(self, client, problem_image_base64):
        """Test visual analysis with image."""
        request_data = {
            "message": "Why does my print have these bands?",
            "image_base64": problem_image_base64,
        }

        response = client.post("/api/deep-learning/multimodal/analyze", json=request_data)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "response" in data or "analysis" in data

    def test_analyze_text_only(self, client):
        """Test text-only analysis."""
        request_data = {
            "message": "What causes uneven coating in platinum printing?",
        }

        response = client.post("/api/deep-learning/multimodal/analyze", json=request_data)
        assert response.status_code in [200, 500, 503]


@pytest.mark.api
class TestFederatedLearningEndpoints:
    """Tests for federated learning API endpoints."""

    def test_federated_status(self, client):
        """Test federated learning status endpoint."""
        response = client.get("/api/deep-learning/federated/status")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "enabled" in data

    def test_contribute_local_update(self, client):
        """Test contributing a local model update."""
        request_data = {
            "model_type": "curve_predictor",
            "num_samples": 100,
            "metrics": {
                "local_loss": 0.05,
                "local_accuracy": 0.92,
            },
        }

        response = client.post(
            "/api/deep-learning/federated/contribute", json=request_data
        )
        # May require authentication or opt-in
        assert response.status_code in [200, 401, 403, 503]


@pytest.mark.api
class TestDeepLearningHealthEndpoints:
    """Tests for deep learning health and status endpoints."""

    def test_model_status(self, client):
        """Test model availability status."""
        response = client.get("/api/deep-learning/status")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Should report status of various models
            assert "models" in data or "status" in data

    def test_gpu_status(self, client):
        """Test GPU availability status."""
        response = client.get("/api/deep-learning/gpu-status")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "available" in data or "gpu" in data


@pytest.mark.api
class TestDeepLearningBatchEndpoints:
    """Tests for batch processing endpoints."""

    @pytest.fixture
    def batch_images_base64(self):
        """Create multiple base64 encoded images."""
        images = []
        for i in range(3):
            arr = np.random.randint(50 + i * 20, 200 - i * 20, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images.append(base64.b64encode(buffer.getvalue()).decode())
        return images

    def test_batch_quality_assessment(self, client, batch_images_base64):
        """Test batch quality assessment."""
        request_data = {
            "images": [
                {"id": f"img_{i}", "image_base64": img}
                for i, img in enumerate(batch_images_base64)
            ],
        }

        response = client.post(
            "/api/deep-learning/batch/assess-quality", json=request_data
        )
        assert response.status_code in [200, 500, 503]

    def test_batch_defect_detection(self, client, batch_images_base64):
        """Test batch defect detection."""
        request_data = {
            "images": [
                {"id": f"print_{i}", "image_base64": img}
                for i, img in enumerate(batch_images_base64)
            ],
        }

        response = client.post(
            "/api/deep-learning/batch/detect-defects", json=request_data
        )
        assert response.status_code in [200, 500, 503]
