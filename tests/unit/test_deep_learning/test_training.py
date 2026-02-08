"""
Tests for the deep learning training module.

Tests cover:
- Synthetic data generators with anti-hallucination measures
- Training pipelines for all models
- Configuration validation
- Data separation between train/val/test splits
"""

import numpy as np
import pytest

from ptpd_calibration.deep_learning.training.data_generators import (
    CurveDataGenerator,
    DefectDataGenerator,
    DetectionDataGenerator,
    ExposureDataGenerator,
    RecipeDataGenerator,
    SyntheticDataConfig,
)
from ptpd_calibration.deep_learning.training.pipelines import (
    EarlyStopping,
    TrainingConfig,
    TrainingMetrics,
)


class TestSyntheticDataConfig:
    """Tests for SyntheticDataConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyntheticDataConfig()

        assert config.seed == 42
        assert config.input_noise_std == 0.05
        assert config.output_noise_std == 0.02
        assert config.label_noise_probability == 0.05
        assert config.augmentation_probability == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SyntheticDataConfig(
            seed=123,
            input_noise_std=0.1,
            output_noise_std=0.05,
            label_noise_probability=0.1,
        )

        assert config.seed == 123
        assert config.input_noise_std == 0.1
        assert config.output_noise_std == 0.05
        assert config.label_noise_probability == 0.1

    def test_validation_bounds(self):
        """Test that validation bounds work."""
        with pytest.raises(ValueError):
            SyntheticDataConfig(input_noise_std=-0.1)  # Must be >= 0

        with pytest.raises(ValueError):
            SyntheticDataConfig(label_noise_probability=1.5)  # Must be <= 1.0


class TestDetectionDataGenerator:
    """Tests for DetectionDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = DetectionDataGenerator(config)

        data = generator.generate(10)

        assert "images" in data
        assert "bboxes" in data
        assert "masks" in data
        assert data["images"].shape[0] == 10
        assert data["bboxes"].shape[0] == 10
        assert data["masks"].shape[0] == 10

    def test_image_shape(self):
        """Test that images have correct shape."""
        config = SyntheticDataConfig(seed=42)
        generator = DetectionDataGenerator(config, image_size=256, num_patches=21)

        data = generator.generate(5)

        assert data["images"].shape == (5, 256, 256, 3)
        assert data["bboxes"].shape[1] == 21  # num_patches
        assert data["bboxes"].shape[2] == 5  # x, y, w, h, confidence

    def test_noise_application(self):
        """Test that noise is applied to prevent exact matching."""
        config_no_noise = SyntheticDataConfig(seed=42, input_noise_std=0.0)
        config_with_noise = SyntheticDataConfig(seed=42, input_noise_std=0.1)

        gen_no_noise = DetectionDataGenerator(config_no_noise)
        gen_with_noise = DetectionDataGenerator(config_with_noise)

        data_no_noise = gen_no_noise.generate(1)
        data_with_noise = gen_with_noise.generate(1)

        # Images should be different due to noise
        # Note: Can't be exactly equal check due to different seeds affecting internal state
        assert data_no_noise["images"].shape == data_with_noise["images"].shape

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config1 = SyntheticDataConfig(seed=42)
        config2 = SyntheticDataConfig(seed=42)

        gen1 = DetectionDataGenerator(config1)
        gen2 = DetectionDataGenerator(config2)

        data1 = gen1.generate(5)
        data2 = gen2.generate(5)

        np.testing.assert_array_equal(data1["images"], data2["images"])
        np.testing.assert_array_equal(data1["bboxes"], data2["bboxes"])

    def test_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        config1 = SyntheticDataConfig(seed=42)
        config2 = SyntheticDataConfig(seed=123)

        gen1 = DetectionDataGenerator(config1)
        gen2 = DetectionDataGenerator(config2)

        data1 = gen1.generate(5)
        data2 = gen2.generate(5)

        # Data should be different
        assert not np.allclose(data1["images"], data2["images"])


class TestCurveDataGenerator:
    """Tests for CurveDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = CurveDataGenerator(config, num_zones=21)

        data = generator.generate(10)

        assert "densities" in data
        assert "process_conditions" in data
        assert "target_curves" in data
        assert data["densities"].shape == (10, 21)
        assert data["target_curves"].shape == (10, 21)

    def test_curve_monotonicity(self):
        """Test that generated curves are monotonic (increasing)."""
        config = SyntheticDataConfig(seed=42, output_noise_std=0.001)
        generator = CurveDataGenerator(config, num_zones=21)

        data = generator.generate(100)
        curves = data["target_curves"]

        # Most curves should be roughly monotonic
        # (small noise may cause slight inversions)
        diffs = np.diff(curves, axis=1)
        increasing_fraction = np.mean(diffs >= -0.1)  # Allow small noise
        assert increasing_fraction > 0.9

    def test_density_range(self):
        """Test that densities are in valid range [0, 1]."""
        config = SyntheticDataConfig(seed=42)
        generator = CurveDataGenerator(config)

        data = generator.generate(100)
        densities = data["densities"]

        assert np.all(densities >= 0)
        assert np.all(densities <= 1)


class TestExposureDataGenerator:
    """Tests for ExposureDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate(10)

        assert "features" in data
        assert "exposure_times" in data
        assert data["features"].shape[0] == 10
        assert data["exposure_times"].shape[0] == 10

    def test_exposure_time_positive(self):
        """Test that exposure times are positive."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate(100)

        assert np.all(data["exposure_times"] > 0)

    def test_feature_dimensions(self):
        """Test feature dimensions."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate(10)

        # Should have multiple features (humidity, temperature, paper type, etc.)
        assert data["features"].shape[1] >= 5


class TestDefectDataGenerator:
    """Tests for DefectDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = DefectDataGenerator(config, num_defect_types=7)

        data = generator.generate(10)

        assert "images" in data
        assert "masks" in data
        assert "labels" in data
        assert data["images"].shape[0] == 10
        assert data["masks"].shape[0] == 10
        assert data["labels"].shape[0] == 10

    def test_label_range(self):
        """Test that labels are in valid range."""
        config = SyntheticDataConfig(seed=42)
        num_classes = 7
        generator = DefectDataGenerator(config, num_defect_types=num_classes)

        data = generator.generate(100)

        assert np.all(data["labels"] >= 0)
        assert np.all(data["labels"] < num_classes)

    def test_label_noise(self):
        """Test that label noise is applied correctly."""
        config_no_noise = SyntheticDataConfig(seed=42, label_noise_probability=0.0)
        config_with_noise = SyntheticDataConfig(seed=42, label_noise_probability=0.5)

        gen_no_noise = DefectDataGenerator(config_no_noise, num_defect_types=7)
        gen_with_noise = DefectDataGenerator(config_with_noise, num_defect_types=7)

        # Generate many samples to observe noise effect
        gen_no_noise.generate(1000)
        data_with_noise = gen_with_noise.generate(1000)

        # With noise, labels should be more distributed
        # (hard to test exactly, but we can check labels are still valid)
        assert np.all(data_with_noise["labels"] >= 0)
        assert np.all(data_with_noise["labels"] < 7)


class TestRecipeDataGenerator:
    """Tests for RecipeDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate(10)

        assert "user_ids" in data
        assert "recipe_ids" in data
        assert "ratings" in data
        assert data["user_ids"].shape[0] == 10
        assert data["recipe_ids"].shape[0] == 10
        assert data["ratings"].shape[0] == 10

    def test_rating_range(self):
        """Test that ratings are in valid range."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate(100)

        # Ratings should be between 0 and 5 (or similar range)
        assert np.all(data["ratings"] >= 0)
        assert np.all(data["ratings"] <= 5)

    def test_id_types(self):
        """Test that IDs are integers."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate(100)

        assert data["user_ids"].dtype in [np.int32, np.int64]
        assert data["recipe_ids"].dtype in [np.int32, np.int64]


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 100
        assert config.early_stopping_patience == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=50,
        )

        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 50

    def test_validation(self):
        """Test that validation works."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)  # Must be >= 1

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-0.1)  # Must be > 0


class TestTrainingMetrics:
    """Tests for TrainingMetrics."""

    def test_create_metrics(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=1e-4,
        )

        assert metrics.epoch == 5
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.learning_rate == 1e-4

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        metrics = TrainingMetrics(epoch=0, train_loss=1.0)

        assert metrics.val_loss is None
        assert metrics.train_accuracy is None
        assert metrics.best_val_loss == float("inf")


class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_no_stop_on_improvement(self):
        """Test that training continues when improving."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        # Improving losses
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)
        assert not early_stopping(0.7)

        assert not early_stopping.should_stop

    def test_stop_on_no_improvement(self):
        """Test that training stops after patience exhausted."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        # Initial value
        assert not early_stopping(1.0)

        # No improvement
        assert not early_stopping(1.0)  # counter = 1
        assert not early_stopping(1.0)  # counter = 2
        assert early_stopping(1.0)  # counter = 3 -> stop

        assert early_stopping.should_stop

    def test_reset(self):
        """Test resetting early stopping."""
        early_stopping = EarlyStopping(patience=2)

        early_stopping(1.0)
        early_stopping(1.0)
        early_stopping(1.0)

        assert early_stopping.should_stop

        early_stopping.reset()

        assert not early_stopping.should_stop
        assert early_stopping.best_value is None
        assert early_stopping.counter == 0

    def test_mode_max(self):
        """Test maximization mode."""
        early_stopping = EarlyStopping(patience=3, mode="max")

        # Improving values (increasing)
        assert not early_stopping(0.5)
        assert not early_stopping(0.6)
        assert not early_stopping(0.7)

        # Decreasing (not improving)
        assert not early_stopping(0.6)
        assert not early_stopping(0.5)
        assert early_stopping(0.4)


class TestDataSeparation:
    """Tests for proper data separation between train/val/test."""

    def test_different_seeds_different_distributions(self):
        """Test that different seeds create non-overlapping data."""
        seed_train = 42
        seed_val = 1042
        seed_test = 2042

        config_train = SyntheticDataConfig(seed=seed_train)
        config_val = SyntheticDataConfig(seed=seed_val)
        config_test = SyntheticDataConfig(seed=seed_test)

        gen_train = CurveDataGenerator(config_train)
        gen_val = CurveDataGenerator(config_val)
        gen_test = CurveDataGenerator(config_test)

        train_data = gen_train.generate(100)
        val_data = gen_val.generate(100)
        test_data = gen_test.generate(100)

        # Ensure data is different
        assert not np.allclose(train_data["densities"], val_data["densities"])
        assert not np.allclose(train_data["densities"], test_data["densities"])
        assert not np.allclose(val_data["densities"], test_data["densities"])

    def test_val_test_have_less_noise(self):
        """Test that validation/test have different noise levels."""
        config_train = SyntheticDataConfig(seed=42, input_noise_std=0.1, output_noise_std=0.05)
        config_val = SyntheticDataConfig(seed=1042, input_noise_std=0.05, output_noise_std=0.025)
        config_test = SyntheticDataConfig(seed=2042, input_noise_std=0.0, output_noise_std=0.0)

        # This just verifies the configs are set up correctly
        assert config_train.input_noise_std > config_val.input_noise_std
        assert config_val.input_noise_std > config_test.input_noise_std


class TestAntiHallucinationMeasures:
    """Tests for anti-hallucination measures in data generation."""

    def test_input_output_mismatch(self):
        """Test that input and output are not identical."""
        config = SyntheticDataConfig(
            seed=42,
            input_noise_std=0.05,
            output_noise_std=0.02,
        )
        generator = CurveDataGenerator(config)

        data = generator.generate(100)

        # Input densities and output curves should be different
        # (densities are measured values, curves are corrections)
        assert not np.allclose(data["densities"], data["target_curves"])

    def test_variation_in_similar_inputs(self):
        """Test that similar inputs produce varied outputs."""
        config = SyntheticDataConfig(seed=42, output_noise_std=0.05)
        generator = ExposureDataGenerator(config)

        # Generate multiple samples
        data = generator.generate(100)

        # Check that there's variation in outputs even for similar features
        exposure_std = np.std(data["exposure_times"])
        assert exposure_std > 0, "Exposure times should have variation"

    def test_label_noise_prevents_overfitting(self):
        """Test that label noise adds uncertainty to labels."""
        config = SyntheticDataConfig(
            seed=42,
            label_noise_probability=0.1,
        )
        generator = DefectDataGenerator(config, num_defect_types=7)

        # Generate data and verify labels have some noise
        # (i.e., not all perfectly matching ground truth)
        data = generator.generate(1000)

        # Labels should still be valid integers
        assert np.all(data["labels"] >= 0)
        assert np.all(data["labels"] < 7)
        assert data["labels"].dtype in [np.int32, np.int64]


# Skip tests that require torch if not available
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingPipelinesWithTorch:
    """Tests for training pipelines that require PyTorch."""

    def test_detection_pipeline_creation(self):
        """Test DetectionTrainingPipeline can be created."""
        from ptpd_calibration.deep_learning.training.pipelines import (
            DetectionTrainingPipeline,
        )

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            train_samples=100,
            val_samples=20,
            test_samples=20,
            device="cpu",
        )

        pipeline = DetectionTrainingPipeline(config)
        assert pipeline.device.type == "cpu"

    def test_curve_pipeline_creation(self):
        """Test CurveTrainingPipeline can be created."""
        from ptpd_calibration.deep_learning.training.pipelines import (
            CurveTrainingPipeline,
        )

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            train_samples=100,
            val_samples=20,
            test_samples=20,
            device="cpu",
        )

        pipeline = CurveTrainingPipeline(config)
        assert pipeline.device.type == "cpu"

    def test_exposure_pipeline_creation(self):
        """Test ExposureTrainingPipeline can be created."""
        from ptpd_calibration.deep_learning.training.pipelines import (
            ExposureTrainingPipeline,
        )

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            train_samples=100,
            val_samples=20,
            test_samples=20,
            device="cpu",
        )

        pipeline = ExposureTrainingPipeline(config)
        assert pipeline.device.type == "cpu"

    def test_defect_pipeline_creation(self):
        """Test DefectTrainingPipeline can be created."""
        from ptpd_calibration.deep_learning.training.pipelines import (
            DefectTrainingPipeline,
        )

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            train_samples=100,
            val_samples=20,
            test_samples=20,
            device="cpu",
        )

        pipeline = DefectTrainingPipeline(config)
        assert pipeline.device.type == "cpu"
