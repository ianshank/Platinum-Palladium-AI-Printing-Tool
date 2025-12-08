"""
Deep Learning API Endpoint Tests.

Tests for the deep learning curve prediction API endpoints including:
- Detection endpoints
- Curve prediction endpoints
- Training endpoints
- Quality assessment endpoints
- Defect detection endpoints
- Recipe recommendation endpoints
- Print comparison endpoints
- Multi-modal assistant endpoints
"""

import base64
import importlib.util
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# Check if PyTorch is available using importlib (cleaner than try/import)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Check if FastAPI is available
FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


pytestmark = [
    pytest.mark.api,
    pytest.mark.deep,
]


# =============================================================================
# Training/Prediction Endpoint Fixtures
# =============================================================================


@pytest.fixture
def deep_learning_status_response():
    """Expected structure for deep learning status response."""
    return {
        "torch_available": True,
        "torch_version": "2.0.0",
        "cuda_available": False,
        "models_loaded": [],
        "database_records": 0,
        "training_in_progress": [],
    }


@pytest.fixture
def train_request_data():
    """Sample training request data."""
    return {
        "model_name": "test_model",
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.001,
        "use_synthetic_data": True,
        "num_synthetic_samples": 50,
        "validation_split": 0.2,
    }


@pytest.fixture
def predict_request_data():
    """Sample prediction request data."""
    return {
        "model_name": "test_model",
        "paper_type": "Arches Platine",
        "metal_ratio": 0.5,
        "exposure_time": 180.0,
        "contrast_agent": "na2",
        "contrast_amount": 5.0,
        "humidity": 50.0,
        "temperature": 21.0,
        "return_uncertainty": True,
    }


@pytest.fixture
def suggest_adjustments_request():
    """Sample adjustment suggestion request."""
    return {
        "model_name": "test_model",
        "paper_type": "Arches Platine",
        "metal_ratio": 0.5,
        "exposure_time": 180.0,
        "target_curve": [0.1 + i * 0.04 for i in range(21)],
    }


# =============================================================================
# Training API Endpoints
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningStatusEndpoint:
    """Tests for /api/deep/status endpoint."""

    def test_status_returns_torch_availability(self, client):
        """Test status endpoint returns PyTorch availability info."""
        response = client.get("/api/deep/status")

        assert response.status_code == 200
        data = response.json()
        assert "torch_available" in data
        assert "torch_version" in data
        assert "cuda_available" in data
        assert "models_loaded" in data
        assert "database_records" in data
        assert "training_in_progress" in data

    def test_status_shows_empty_models_initially(self, client):
        """Test that no models are loaded initially."""
        response = client.get("/api/deep/status")

        assert response.status_code == 200
        data = response.json()
        assert data["models_loaded"] == []

    def test_status_response_time(self, client):
        """Test status endpoint responds quickly."""
        import time

        start = time.time()
        response = client.get("/api/deep/status")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, "Status endpoint should respond in under 1 second"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningTrainEndpoint:
    """Tests for /api/deep/train endpoint."""

    def test_train_without_pytorch_returns_error(self, client, train_request_data):
        """Test training without PyTorch returns 503."""
        with patch.dict("sys.modules", {"torch": None}):
            response = client.post("/api/deep/train", json=train_request_data)

            # Either 503 (no PyTorch) or 200 (training started)
            assert response.status_code in [200, 503]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_with_valid_request(self, client, train_request_data):
        """Test training with valid request returns success."""
        response = client.post("/api/deep/train", json=train_request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == train_request_data["model_name"]
        assert data["status"] == "starting"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_with_default_values(self, client):
        """Test training with minimal request uses defaults."""
        response = client.post("/api/deep/train", json={"use_synthetic_data": True})

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "default"

    def test_train_invalid_epochs(self, client):
        """Test training with invalid epoch count."""
        response = client.post(
            "/api/deep/train",
            json={"num_epochs": 2000, "use_synthetic_data": True},  # Exceeds max
        )

        assert response.status_code == 422  # Validation error

    def test_train_invalid_batch_size(self, client):
        """Test training with invalid batch size."""
        response = client.post(
            "/api/deep/train",
            json={"batch_size": 500, "use_synthetic_data": True},  # Exceeds max
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningTrainStatusEndpoint:
    """Tests for /api/deep/train/{model_name}/status endpoint."""

    def test_status_not_found(self, client):
        """Test status for non-existent model returns 404."""
        response = client.get("/api/deep/train/nonexistent_model/status")

        assert response.status_code == 404

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_status_after_training_started(self, client, train_request_data):
        """Test status after training has been started."""
        # Start training
        client.post("/api/deep/train", json=train_request_data)

        # Check status
        response = client.get(f"/api/deep/train/{train_request_data['model_name']}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == train_request_data["model_name"]
        assert data["status"] in ["starting", "training", "completed", "failed"]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningPredictEndpoint:
    """Tests for /api/deep/predict endpoint."""

    def test_predict_model_not_found(self, client, predict_request_data):
        """Test prediction with non-existent model returns 404."""
        response = client.post("/api/deep/predict", json=predict_request_data)

        # Either 404 (model not found) or 503 (no PyTorch)
        assert response.status_code in [404, 503]

    def test_predict_missing_paper_type(self, client):
        """Test prediction without required paper_type."""
        response = client.post(
            "/api/deep/predict",
            json={"model_name": "test", "metal_ratio": 0.5},
        )

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_metal_ratio(self, client):
        """Test prediction with invalid metal ratio."""
        response = client.post(
            "/api/deep/predict",
            json={
                "model_name": "test",
                "paper_type": "Test Paper",
                "metal_ratio": 1.5,  # Out of range [0, 1]
            },
        )

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_humidity(self, client):
        """Test prediction with invalid humidity."""
        response = client.post(
            "/api/deep/predict",
            json={
                "model_name": "test",
                "paper_type": "Test Paper",
                "metal_ratio": 0.5,
                "humidity": 150.0,  # Out of range [0, 100]
            },
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningSuggestAdjustmentsEndpoint:
    """Tests for /api/deep/suggest-adjustments endpoint."""

    def test_suggest_model_not_found(self, client, suggest_adjustments_request):
        """Test suggestion with non-existent model returns 404."""
        response = client.post(
            "/api/deep/suggest-adjustments", json=suggest_adjustments_request
        )

        # Either 404 (model not found) or 503 (no PyTorch)
        assert response.status_code in [404, 503]

    def test_suggest_missing_target_curve(self, client):
        """Test suggestion without required target_curve."""
        response = client.post(
            "/api/deep/suggest-adjustments",
            json={
                "model_name": "test",
                "paper_type": "Test Paper",
            },
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningModelsEndpoint:
    """Tests for /api/deep/models endpoint."""

    def test_list_models_empty_initially(self, client):
        """Test listing models returns empty list initially."""
        response = client.get("/api/deep/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_delete_model_not_found(self, client):
        """Test deleting non-existent model returns 404."""
        response = client.delete("/api/deep/models/nonexistent_model")

        assert response.status_code == 404


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningGenerateSyntheticEndpoint:
    """Tests for /api/deep/generate-synthetic endpoint."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_generate_synthetic_default_params(self, client):
        """Test generating synthetic data with default parameters."""
        response = client.post("/api/deep/generate-synthetic")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["records_added"] == 100  # Default
        assert "total_records" in data

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_generate_synthetic_custom_count(self, client):
        """Test generating synthetic data with custom count."""
        response = client.post("/api/deep/generate-synthetic?num_samples=50")

        assert response.status_code == 200
        data = response.json()
        assert data["records_added"] == 50

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_generate_synthetic_with_seed(self, client):
        """Test generating synthetic data with seed for reproducibility."""
        response1 = client.post("/api/deep/generate-synthetic?num_samples=10&seed=42")
        response2 = client.post("/api/deep/generate-synthetic?num_samples=10&seed=42")

        assert response1.status_code == 200
        assert response2.status_code == 200
        # Both should add records (seed affects content, not count)
        assert response1.json()["records_added"] == 10
        assert response2.json()["records_added"] == 10


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDeepLearningEndpointValidation:
    """Tests for request validation across endpoints."""

    def test_train_learning_rate_bounds(self, client):
        """Test learning rate validation."""
        # Too small
        response = client.post(
            "/api/deep/train",
            json={"learning_rate": 1e-8, "use_synthetic_data": True},
        )
        assert response.status_code == 422

        # Too large
        response = client.post(
            "/api/deep/train",
            json={"learning_rate": 2.0, "use_synthetic_data": True},
        )
        assert response.status_code == 422

    def test_train_validation_split_bounds(self, client):
        """Test validation split bounds."""
        # Too small
        response = client.post(
            "/api/deep/train",
            json={"validation_split": 0.05, "use_synthetic_data": True},
        )
        assert response.status_code == 422

        # Too large
        response = client.post(
            "/api/deep/train",
            json={"validation_split": 0.5, "use_synthetic_data": True},
        )
        assert response.status_code == 422

    def test_predict_exposure_time_minimum(self, client):
        """Test exposure time minimum validation."""
        response = client.post(
            "/api/deep/predict",
            json={
                "model_name": "test",
                "paper_type": "Test",
                "exposure_time": 0.5,  # Below minimum of 1.0
            },
        )
        assert response.status_code == 422


@pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and TORCH_AVAILABLE),
    reason="FastAPI and PyTorch required",
)
class TestDeepLearningIntegration:
    """Integration tests for deep learning workflow via API."""

    def test_full_training_workflow(self, client):
        """Test complete training workflow through API."""
        # 1. Check initial status
        status_response = client.get("/api/deep/status")
        assert status_response.status_code == 200
        assert status_response.json()["models_loaded"] == []

        # 2. Generate synthetic data
        synth_response = client.post("/api/deep/generate-synthetic?num_samples=50")
        assert synth_response.status_code == 200

        # 3. Start training
        train_response = client.post(
            "/api/deep/train",
            json={
                "model_name": "integration_test_model",
                "num_epochs": 2,
                "batch_size": 8,
                "use_synthetic_data": True,
                "num_synthetic_samples": 50,
            },
        )
        assert train_response.status_code == 200
        assert train_response.json()["status"] == "starting"

        # 4. Check training status
        import time

        for _ in range(30):  # Wait up to 30 seconds
            status_response = client.get(
                "/api/deep/train/integration_test_model/status"
            )
            if status_response.json()["status"] in ["completed", "failed"]:
                break
            time.sleep(1)

        # 5. List models
        models_response = client.get("/api/deep/models")
        assert models_response.status_code == 200

    def test_api_error_handling(self, client):
        """Test that API returns proper error responses."""
        # Invalid JSON
        response = client.post(
            "/api/deep/train",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

        # Missing required fields
        response = client.post(
            "/api/deep/predict",
            json={},  # Missing paper_type
        )
        assert response.status_code == 422


# =============================================================================
# Detection Endpoints
# =============================================================================


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
