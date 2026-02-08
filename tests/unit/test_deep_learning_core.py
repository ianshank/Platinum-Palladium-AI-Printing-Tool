"""
Tests for deep learning curve prediction modules.

These tests cover:
- Model architectures (CurveMLP, CurveCNN, etc.)
- Dataset and data loading
- Training loop and metrics
- Process simulation
- Full predictor workflow

Tests are marked with @pytest.mark.deep to allow skipping when PyTorch is unavailable.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ptpd_calibration.config import DeepLearningSettings, get_settings
from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
from ptpd_calibration.ml.database import CalibrationDatabase
from ptpd_calibration.ml.deep.exceptions import (
    DatasetError,
    DeepLearningError,
    ModelNotTrainedError,
)

# Check if PyTorch is available using importlib (cleaner than try/import per Copilot feedback)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Skip all tests in this module if PyTorch is not available
pytestmark = [
    pytest.mark.deep,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dl_settings() -> DeepLearningSettings:
    """Create deep learning settings for testing."""
    return DeepLearningSettings(
        model_type="curve_mlp",
        num_control_points=8,
        lut_size=64,
        hidden_dims=[32, 64, 32],
        dropout_rate=0.1,
        use_batch_norm=True,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=5,
        early_stopping_patience=3,
        use_ensemble=False,
        device="cpu",
    )


@pytest.fixture
def sample_records() -> list[CalibrationRecord]:
    """Create sample calibration records for testing."""
    records = []
    papers = ["Arches Platine", "Bergger COT320"]
    ratios = [0.3, 0.5, 0.7]
    exposures = [150.0, 180.0, 210.0]

    for paper in papers:
        for ratio in ratios:
            for exposure in exposures:
                # Generate realistic density curve
                steps = np.linspace(0, 1, 21)
                gamma = 0.7 + ratio * 0.4  # Gamma varies with metal ratio
                densities = 0.1 + 1.8 * (steps**gamma)
                # Add some noise
                densities += np.random.normal(0, 0.02, 21)
                densities = np.clip(densities, 0.0, 3.0)

                record = CalibrationRecord(
                    paper_type=paper,
                    exposure_time=exposure,
                    metal_ratio=ratio,
                    chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                    contrast_agent=ContrastAgent.NA2,
                    contrast_amount=5.0,
                    developer=DeveloperType.POTASSIUM_OXALATE,
                    humidity=50.0,
                    temperature=21.0,
                    measured_densities=list(densities),
                )
                records.append(record)

    return records


@pytest.fixture
def populated_db(sample_records: list[CalibrationRecord]) -> CalibrationDatabase:
    """Create a populated calibration database."""
    db = CalibrationDatabase()
    for record in sample_records:
        db.add_record(record)
    return db


@pytest.fixture
def single_record() -> CalibrationRecord:
    """Create a single calibration record for prediction tests."""
    return CalibrationRecord(
        paper_type="Arches Platine",
        exposure_time=180.0,
        metal_ratio=0.5,
        chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
        contrast_agent=ContrastAgent.NA2,
        contrast_amount=5.0,
        developer=DeveloperType.POTASSIUM_OXALATE,
        humidity=50.0,
        temperature=21.0,
        measured_densities=[0.1 + i * 0.1 for i in range(21)],
    )


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_deep_learning_error_base(self) -> None:
        """Test DeepLearningError is proper exception."""
        with pytest.raises(DeepLearningError):
            raise DeepLearningError("Test error")

    def test_model_not_trained_error(self) -> None:
        """Test ModelNotTrainedError inheritance."""
        with pytest.raises(DeepLearningError):
            raise ModelNotTrainedError("Model not trained")

    def test_dataset_error(self) -> None:
        """Test DatasetError inheritance."""
        with pytest.raises(DeepLearningError):
            raise DatasetError("Dataset error")


# =============================================================================
# Settings Tests
# =============================================================================


class TestDeepLearningSettings:
    """Tests for DeepLearningSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = DeepLearningSettings()

        assert settings.model_type == "curve_mlp"
        assert settings.num_control_points == 16
        assert settings.lut_size == 256
        assert settings.learning_rate == 1e-3
        assert settings.batch_size == 32

    def test_settings_validation(self) -> None:
        """Test settings validation."""
        from pydantic import ValidationError

        # Valid settings
        settings = DeepLearningSettings(
            num_control_points=32,
            lut_size=512,
            dropout_rate=0.2,
        )
        assert settings.num_control_points == 32

        # Invalid settings should raise
        with pytest.raises(ValidationError):
            DeepLearningSettings(num_control_points=2)  # Too few

        with pytest.raises(ValidationError):
            DeepLearningSettings(dropout_rate=1.5)  # Out of range

    def test_settings_in_global_config(self) -> None:
        """Test deep_learning settings in global config."""
        settings = get_settings()
        assert hasattr(settings, "deep_learning")
        assert isinstance(settings.deep_learning, DeepLearningSettings)


# =============================================================================
# Model Tests
# =============================================================================


class TestCurveMLP:
    """Tests for CurveMLP model."""

    def test_model_creation(self, dl_settings: DeepLearningSettings) -> None:
        """Test model creation from settings."""
        from ptpd_calibration.ml.deep.models import CurveMLP

        num_features = 15
        model = CurveMLP.from_settings(num_features, dl_settings)

        assert model.num_features == num_features
        assert model.num_control_points == dl_settings.num_control_points
        assert model.lut_size == dl_settings.lut_size

    def test_forward_pass(self, dl_settings: DeepLearningSettings) -> None:
        """Test forward pass produces correct output shapes."""
        import torch

        from ptpd_calibration.ml.deep.models import CurveMLP

        num_features = 15
        batch_size = 4
        model = CurveMLP.from_settings(num_features, dl_settings)

        # Create dummy input
        x = torch.randn(batch_size, num_features)

        # Forward pass
        lut, control_points = model(x, return_control_points=True)

        assert lut.shape == (batch_size, dl_settings.lut_size)
        assert control_points.shape == (batch_size, dl_settings.num_control_points)

    def test_output_monotonicity(self, dl_settings: DeepLearningSettings) -> None:
        """Test that output curves are monotonic."""
        import torch

        from ptpd_calibration.ml.deep.models import CurveMLP

        num_features = 15
        model = CurveMLP.from_settings(num_features, dl_settings)

        x = torch.randn(10, num_features)
        lut, _ = model(x)

        # Check monotonicity
        diffs = lut[:, 1:] - lut[:, :-1]
        assert torch.all(diffs >= 0.0), "Curves should be monotonically increasing"

    def test_output_range(self, dl_settings: DeepLearningSettings) -> None:
        """Test that output is in [0, 1] range."""
        import torch

        from ptpd_calibration.ml.deep.models import CurveMLP

        num_features = 15
        model = CurveMLP.from_settings(num_features, dl_settings)

        x = torch.randn(10, num_features)
        lut, _ = model(x)

        assert torch.all(lut >= 0.0), "LUT values should be >= 0"
        assert torch.all(lut <= 1.0), "LUT values should be <= 1"


class TestCurveCNN:
    """Tests for CurveCNN model."""

    def test_model_creation(self, dl_settings: DeepLearningSettings) -> None:
        """Test CNN model creation."""
        from ptpd_calibration.ml.deep.models import CurveCNN

        model = CurveCNN(
            num_features=15,
            lut_size=dl_settings.lut_size,
            dropout_rate=dl_settings.dropout_rate,
        )

        assert model.num_features == 15
        assert model.lut_size == dl_settings.lut_size

    def test_forward_pass(self, dl_settings: DeepLearningSettings) -> None:
        """Test CNN forward pass."""
        import torch

        from ptpd_calibration.ml.deep.models import CurveCNN

        model = CurveCNN(num_features=15, lut_size=dl_settings.lut_size)
        x = torch.randn(4, 15)
        lut, _ = model(x)

        assert lut.shape == (4, dl_settings.lut_size)


class TestMonotonicLayer:
    """Tests for MonotonicLayer."""

    def test_monotonicity_enforcement(self) -> None:
        """Test that layer enforces monotonicity."""
        import torch

        from ptpd_calibration.ml.deep.models import MonotonicLayer

        layer = MonotonicLayer(normalize=True)

        # Input with some negative values
        x = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.2]])

        output = layer(x)

        # Check monotonicity
        diffs = output[:, 1:] - output[:, :-1]
        assert torch.all(diffs >= 0), "Output should be monotonically increasing"

    def test_normalization(self) -> None:
        """Test output normalization to [0, 1]."""
        import torch

        from ptpd_calibration.ml.deep.models import MonotonicLayer

        layer = MonotonicLayer(normalize=True)
        x = torch.randn(5, 10)
        output = layer(x)

        assert torch.allclose(output[:, 0], torch.zeros(5), atol=1e-5)
        assert torch.allclose(output[:, -1], torch.ones(5), atol=1e-5)


class TestContentAwareCurveNet:
    """Tests for ContentAwareCurveNet."""

    def test_model_creation(self) -> None:
        """Test content-aware model creation."""
        from ptpd_calibration.ml.deep.models import ContentAwareCurveNet

        model = ContentAwareCurveNet(
            in_channels=1,
            base_channels=16,
            num_levels=3,
            adjustment_range=(0.9, 1.1),
        )

        assert model.adjustment_range == (0.9, 1.1)

    def test_forward_pass(self) -> None:
        """Test content-aware forward pass."""
        import torch

        from ptpd_calibration.ml.deep.models import ContentAwareCurveNet

        model = ContentAwareCurveNet(in_channels=1, base_channels=8, num_levels=2)

        # Create dummy image
        x = torch.randn(2, 1, 64, 64)
        output = model(x)

        assert output.shape == (2, 1, 64, 64)

    def test_adjustment_range(self) -> None:
        """Test that output is within adjustment range."""
        import torch

        from ptpd_calibration.ml.deep.models import ContentAwareCurveNet

        low, high = 0.8, 1.2
        model = ContentAwareCurveNet(
            in_channels=1,
            base_channels=8,
            num_levels=2,
            adjustment_range=(low, high),
        )

        x = torch.randn(2, 1, 32, 32)
        output = model(x)

        assert torch.all(output >= low)
        assert torch.all(output <= high)


class TestUniformityCorrectionNet:
    """Tests for UniformityCorrectionNet."""

    def test_model_creation(self) -> None:
        """Test uniformity correction model creation."""
        from ptpd_calibration.ml.deep.models import UniformityCorrectionNet

        model = UniformityCorrectionNet(
            kernel_size=15,
            sigma=5.0,
            correction_range=(0.9, 1.1),
        )

        assert model.kernel_size == 15
        assert model.sigma == 5.0

    def test_forward_pass(self) -> None:
        """Test uniformity correction forward pass."""
        import torch

        from ptpd_calibration.ml.deep.models import UniformityCorrectionNet

        model = UniformityCorrectionNet(kernel_size=7, sigma=2.0)
        x = torch.randn(2, 1, 32, 32)
        output = model(x)

        assert output.shape == (2, 1, 32, 32)

    def test_smoothness(self) -> None:
        """Test that correction map is smooth."""
        import torch

        from ptpd_calibration.ml.deep.models import UniformityCorrectionNet

        model = UniformityCorrectionNet(kernel_size=15, sigma=5.0)
        x = torch.randn(1, 1, 64, 64)
        output = model(x)

        # Check that gradients are small (smooth)
        grad_x = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        grad_y = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])

        assert torch.mean(grad_x) < 0.1, "Correction map should be smooth"
        assert torch.mean(grad_y) < 0.1, "Correction map should be smooth"


# =============================================================================
# Dataset Tests
# =============================================================================


class TestFeatureEncoder:
    """Tests for FeatureEncoder."""

    def test_encoder_from_database(self, populated_db: CalibrationDatabase) -> None:
        """Test encoder creation from database."""
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder

        encoder = FeatureEncoder.from_database(populated_db)

        assert len(encoder.paper_to_idx) == 2  # Two paper types
        assert encoder.num_features > 0

    def test_encoder_encode(
        self, populated_db: CalibrationDatabase, single_record: CalibrationRecord
    ) -> None:
        """Test encoding a record."""
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder

        encoder = FeatureEncoder.from_database(populated_db)
        features = encoder.encode(single_record)

        assert features.shape == (encoder.num_features,)
        assert features.dtype == np.float32

    def test_encoder_serialization(self, populated_db: CalibrationDatabase) -> None:
        """Test encoder serialization."""
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder

        encoder = FeatureEncoder.from_database(populated_db)

        # Serialize
        data = encoder.to_dict()
        assert "paper_to_idx" in data
        assert "log_exposure_mean" in data

        # Deserialize
        loaded = FeatureEncoder.from_dict(data)
        assert loaded.num_features == encoder.num_features

    def test_encoder_empty_database_error(self) -> None:
        """Test error on empty database."""
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder

        empty_db = CalibrationDatabase()
        with pytest.raises(DatasetError):
            FeatureEncoder.from_database(empty_db)


class TestCalibrationDataset:
    """Tests for CalibrationDataset."""

    def test_dataset_creation(self, populated_db: CalibrationDatabase) -> None:
        """Test dataset creation."""
        from ptpd_calibration.ml.deep.dataset import CalibrationDataset

        dataset = CalibrationDataset(populated_db, target_length=64)

        assert len(dataset) == len(populated_db)
        assert dataset.target_length == 64

    def test_dataset_getitem(self, populated_db: CalibrationDatabase) -> None:
        """Test getting items from dataset."""
        import torch

        from ptpd_calibration.ml.deep.dataset import CalibrationDataset

        dataset = CalibrationDataset(populated_db, target_length=64)
        features, densities = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(densities, torch.Tensor)
        assert features.shape == (dataset.num_features,)
        assert densities.shape == (64,)

    def test_dataset_split(self, populated_db: CalibrationDatabase) -> None:
        """Test dataset splitting."""
        from ptpd_calibration.ml.deep.dataset import CalibrationDataset

        dataset = CalibrationDataset(populated_db, target_length=64)
        train, val = dataset.split(val_ratio=0.2, seed=42)

        # Check sizes add up
        assert len(train) + len(val) == len(dataset)

        # Check no overlap (by sampling)
        train_items = [train[i][0].numpy().tobytes() for i in range(min(5, len(train)))]
        val_items = [val[i][0].numpy().tobytes() for i in range(min(5, len(val)))]

        for item in train_items:
            assert item not in val_items


class TestDataAugmentation:
    """Tests for DataAugmentation."""

    def test_augmentation_features(self) -> None:
        """Test feature augmentation."""
        from ptpd_calibration.ml.deep.dataset import DataAugmentation

        aug = DataAugmentation(enabled=True, noise_std=0.1)
        features = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        augmented = aug.augment_features(features)

        # Should be different due to noise
        assert not np.allclose(features, augmented)

    def test_augmentation_disabled(self) -> None:
        """Test disabled augmentation returns unchanged data."""
        from ptpd_calibration.ml.deep.dataset import DataAugmentation

        aug = DataAugmentation(enabled=False)
        features = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        augmented = aug.augment_features(features)
        assert np.array_equal(features, augmented)


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_create_dataloaders(self, populated_db: CalibrationDatabase) -> None:
        """Test dataloader creation."""
        from ptpd_calibration.ml.deep.dataset import create_dataloaders

        train_loader, val_loader, encoder = create_dataloaders(
            populated_db,
            batch_size=4,
            val_ratio=0.2,
            target_length=64,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert encoder is not None

        # Check batch
        batch = next(iter(train_loader))
        features, densities = batch
        assert features.shape[0] <= 4
        assert densities.shape[1] == 64


# =============================================================================
# Training Tests
# =============================================================================


class TestCurveLoss:
    """Tests for CurveLoss."""

    def test_loss_computation(self) -> None:
        """Test loss computation."""
        import torch

        from ptpd_calibration.ml.deep.training import CurveLoss

        loss_fn = CurveLoss(
            mse_weight=1.0,
            monotonicity_weight=0.1,
            smoothness_weight=0.05,
        )

        predicted = torch.linspace(0, 1, 64).unsqueeze(0)
        target = torch.linspace(0, 1, 64).unsqueeze(0) + 0.01

        loss, components = loss_fn(predicted, target)

        assert loss.item() > 0
        assert "mse" in components
        assert "monotonicity" in components
        assert "smoothness" in components

    def test_monotonicity_penalty(self) -> None:
        """Test monotonicity penalty is higher for non-monotonic curves."""
        import torch

        from ptpd_calibration.ml.deep.training import CurveLoss

        loss_fn = CurveLoss(mse_weight=0.0, monotonicity_weight=1.0, smoothness_weight=0.0)

        # Monotonic curve
        monotonic = torch.linspace(0, 1, 64).unsqueeze(0)
        # Non-monotonic curve
        non_monotonic = monotonic.clone()
        non_monotonic[0, 30] = 0.2  # Create decrease

        target = torch.zeros_like(monotonic)

        loss_mono, _ = loss_fn(monotonic, target)
        loss_non_mono, _ = loss_fn(non_monotonic, target)

        assert loss_non_mono > loss_mono


class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_early_stopping_improvement(self) -> None:
        """Test early stopping with improving values."""
        from ptpd_calibration.ml.deep.training import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01)

        # Improving values should not trigger stop
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        assert not es.should_stop

    def test_early_stopping_no_improvement(self) -> None:
        """Test early stopping without improvement."""
        from ptpd_calibration.ml.deep.training import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01)

        es(1.0)  # Best
        es(1.1)  # Worse
        es(1.1)  # Worse
        assert es(1.1)  # Third time - should stop

        assert es.should_stop


class TestCurveTrainer:
    """Tests for CurveTrainer."""

    def test_trainer_creation(self, dl_settings: DeepLearningSettings) -> None:
        """Test trainer creation."""
        import torch

        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import CurveTrainer

        model = CurveMLP.from_settings(15, dl_settings)
        trainer = CurveTrainer(model, dl_settings)

        assert trainer.model is model
        assert trainer.device == torch.device("cpu")

    def test_training_loop(
        self, populated_db: CalibrationDatabase, dl_settings: DeepLearningSettings
    ) -> None:
        """Test basic training loop."""
        from ptpd_calibration.ml.deep.dataset import create_dataloaders
        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import CurveTrainer

        train_loader, val_loader, encoder = create_dataloaders(
            populated_db,
            batch_size=4,
            val_ratio=0.2,
            target_length=dl_settings.lut_size,
        )

        model = CurveMLP.from_settings(encoder.num_features, dl_settings)
        trainer = CurveTrainer(model, dl_settings)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
        )

        assert len(history.metrics) > 0
        assert history.best_val_loss < float("inf")

    def test_evaluation(
        self, populated_db: CalibrationDatabase, dl_settings: DeepLearningSettings
    ) -> None:
        """Test model evaluation."""
        from ptpd_calibration.ml.deep.dataset import create_dataloaders
        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import CurveTrainer

        train_loader, val_loader, encoder = create_dataloaders(
            populated_db,
            batch_size=4,
            val_ratio=0.2,
            target_length=dl_settings.lut_size,
        )

        model = CurveMLP.from_settings(encoder.num_features, dl_settings)
        trainer = CurveTrainer(model, dl_settings)
        trainer.train(train_loader, val_loader, num_epochs=2)

        metrics = trainer.evaluate(val_loader)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "mean_correlation" in metrics


# =============================================================================
# Process Simulator Tests
# =============================================================================


class TestCharacteristicCurve:
    """Tests for CharacteristicCurve."""

    def test_curve_creation(self) -> None:
        """Test characteristic curve creation."""
        from ptpd_calibration.ml.deep.process_sim import CharacteristicCurve

        curve = CharacteristicCurve(gamma=1.8, dmin=0.1, dmax=2.0)

        assert curve._initial_gamma == 1.8
        assert curve._initial_dmin == 0.1

    def test_curve_forward(self) -> None:
        """Test characteristic curve forward pass."""
        import torch

        from ptpd_calibration.ml.deep.process_sim import CharacteristicCurve

        curve = CharacteristicCurve(gamma=1.0, dmin=0.0, dmax=1.0, learnable=False)

        exposure = torch.linspace(0, 1, 11)
        density = curve(exposure)

        # With gamma=1.0 and dmin=0, dmax=1, should be ~identity
        assert density.shape == exposure.shape
        assert torch.all(density >= 0.0)
        assert torch.all(density <= 1.5)  # Some headroom for shoulder


class TestProcessSimulator:
    """Tests for ProcessSimulator."""

    def test_simulator_creation(self, dl_settings: DeepLearningSettings) -> None:
        """Test simulator creation."""
        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        sim = ProcessSimulator(dl_settings)
        params = sim.get_parameters()

        assert params.gamma > 0
        assert params.dmin >= 0

    def test_negative_to_transmission(self, dl_settings: DeepLearningSettings) -> None:
        """Test negative density to transmission conversion."""
        import torch

        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        sim = ProcessSimulator(dl_settings)

        # Density 0 -> transmission 1
        # Density 1 -> transmission 0.1
        # Density 2 -> transmission 0.01
        density = torch.tensor([0.0, 1.0, 2.0])
        transmission = sim.negative_to_transmission(density)

        assert torch.isclose(transmission[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(transmission[1], torch.tensor(0.1), atol=1e-5)
        assert torch.isclose(transmission[2], torch.tensor(0.01), atol=1e-5)

    def test_full_simulation(self, dl_settings: DeepLearningSettings) -> None:
        """Test full process simulation."""
        import torch

        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        sim = ProcessSimulator(dl_settings, learnable=False)

        negative_density = torch.linspace(0, 2, 21)
        print_density = sim(negative_density)

        assert print_density.shape == negative_density.shape
        assert torch.all(print_density >= 0.0)

    def test_simulation_with_intermediates(self, dl_settings: DeepLearningSettings) -> None:
        """Test simulation returning intermediate values."""
        import torch

        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        sim = ProcessSimulator(dl_settings)

        negative_density = torch.linspace(0, 2, 11)
        print_density, intermediates = sim(negative_density, return_intermediates=True)

        assert "transmission" in intermediates
        assert "exposure" in intermediates
        assert "print_density" in intermediates


class TestProcessParameters:
    """Tests for ProcessParameters."""

    def test_to_tensor(self) -> None:
        """Test conversion to tensor."""
        from ptpd_calibration.ml.deep.process_sim import ProcessParameters

        params = ProcessParameters(gamma=1.8, dmin=0.1, dmax=2.0)
        tensor = params.to_tensor()

        assert tensor.shape == (6,)
        assert tensor[0] == pytest.approx(1.8)

    def test_from_tensor(self) -> None:
        """Test creation from tensor."""
        import torch

        from ptpd_calibration.ml.deep.process_sim import ProcessParameters

        tensor = torch.tensor([1.8, 0.1, 2.0, 0.85, 0.15, 1.0])
        params = ProcessParameters.from_tensor(tensor)

        assert params.gamma == pytest.approx(1.8)
        assert params.dmin == pytest.approx(0.1)


# =============================================================================
# Predictor Tests
# =============================================================================


class TestDeepCurvePredictor:
    """Tests for DeepCurvePredictor."""

    def test_predictor_creation(self, dl_settings: DeepLearningSettings) -> None:
        """Test predictor creation."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        predictor = DeepCurvePredictor(dl_settings)

        assert not predictor.is_trained
        assert predictor.model is None

    def test_predict_without_training_error(
        self, dl_settings: DeepLearningSettings, single_record: CalibrationRecord
    ) -> None:
        """Test that prediction without training raises error."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        predictor = DeepCurvePredictor(dl_settings)

        with pytest.raises(ModelNotTrainedError):
            predictor.predict(single_record)

    def test_training_and_prediction(
        self,
        populated_db: CalibrationDatabase,
        single_record: CalibrationRecord,
        dl_settings: DeepLearningSettings,
    ) -> None:
        """Test full training and prediction workflow."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        predictor = DeepCurvePredictor(dl_settings)

        # Train
        stats = predictor.train(populated_db, num_epochs=3)

        assert "num_samples" in stats
        assert "best_val_loss" in stats
        assert predictor.is_trained

        # Predict
        result = predictor.predict(single_record)

        assert result.curve is not None
        assert len(result.curve) == dl_settings.lut_size
        assert result.curve.min() >= 0.0
        assert result.curve.max() <= 1.0

    def test_save_and_load(
        self,
        populated_db: CalibrationDatabase,
        single_record: CalibrationRecord,
        dl_settings: DeepLearningSettings,
        tmp_path: Path,
    ) -> None:
        """Test saving and loading predictor."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        # Train and save
        predictor = DeepCurvePredictor(dl_settings)
        predictor.train(populated_db, num_epochs=2)

        original_prediction = predictor.predict(single_record)

        save_path = tmp_path / "predictor"
        predictor.save(save_path)

        # Verify files exist
        assert (save_path / "model.pt").exists()
        assert (save_path / "encoder.json").exists()
        assert (save_path / "metadata.json").exists()

        # Load and compare
        loaded = DeepCurvePredictor.load(save_path)
        loaded_prediction = loaded.predict(single_record)

        assert np.allclose(
            original_prediction.curve,
            loaded_prediction.curve,
            atol=1e-5,
        )

    def test_to_curve_data(
        self,
        populated_db: CalibrationDatabase,
        single_record: CalibrationRecord,
        dl_settings: DeepLearningSettings,
    ) -> None:
        """Test conversion to CurveData model."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        predictor = DeepCurvePredictor(dl_settings)
        predictor.train(populated_db, num_epochs=2)

        result = predictor.predict(single_record)
        curve_data = predictor.to_curve_data(
            result,
            name="Test Curve",
            paper_type="Arches Platine",
        )

        assert curve_data.name == "Test Curve"
        assert curve_data.paper_type == "Arches Platine"
        assert len(curve_data.input_values) == len(result.curve)

    def test_suggest_adjustments(
        self,
        populated_db: CalibrationDatabase,
        single_record: CalibrationRecord,
        dl_settings: DeepLearningSettings,
    ) -> None:
        """Test adjustment suggestions."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        predictor = DeepCurvePredictor(dl_settings)
        predictor.train(populated_db, num_epochs=2)

        # Create a target curve that's brighter
        target = np.linspace(0.1, 1.0, dl_settings.lut_size)

        suggestions = predictor.suggest_adjustments(single_record, target)

        assert "suggestions" in suggestions
        assert "adjustments" in suggestions
        assert "current_curve" in suggestions


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullWorkflow:
    """Integration tests for the full deep learning workflow."""

    def test_end_to_end_workflow(
        self,
        populated_db: CalibrationDatabase,
        dl_settings: DeepLearningSettings,
        tmp_path: Path,
    ) -> None:
        """Test complete workflow from data to prediction."""
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        # 1. Create predictor
        predictor = DeepCurvePredictor(dl_settings)

        # 2. Train on database
        stats = predictor.train(populated_db, num_epochs=3)
        assert stats["num_samples"] > 0

        # 3. Make predictions for all papers in database
        papers = ["Arches Platine", "Bergger COT320"]
        predictions = {}

        for paper in papers:
            record = CalibrationRecord(
                paper_type=paper,
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            predictions[paper] = predictor.predict(record)

        # Different papers should give different curves
        assert not np.allclose(
            predictions["Arches Platine"].curve,
            predictions["Bergger COT320"].curve,
            atol=0.01,
        )

        # 4. Save and reload
        save_path = tmp_path / "workflow_test"
        predictor.save(save_path)

        loaded = DeepCurvePredictor.load(save_path)

        # 5. Verify loaded model produces same results
        for paper in papers:
            record = CalibrationRecord(
                paper_type=paper,
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            new_pred = loaded.predict(record)
            assert np.allclose(
                predictions[paper].curve,
                new_pred.curve,
                atol=1e-5,
            )
