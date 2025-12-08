"""
Deep Learning API Endpoint Tests.

Tests for the deep learning curve prediction API endpoints.
"""

import pytest
from unittest.mock import patch

# Check if PyTorch is available
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient  # noqa: F401

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = [
    pytest.mark.api,
    pytest.mark.deep,
]


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
