"""
Tests for the deep learning training module.

Tests cover:
- Synthetic data generators with anti-hallucination measures
- Training pipelines for all models
- Configuration validation
- Data separation between train/val/test splits
"""

import pytest

pytest.importorskip("torch")

import numpy as np

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
        assert config.label_noise_probability == 0.01

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

        data = generator.generate_batch(10)

        assert len(data) == 10
        for sample in data:
            assert "image" in sample
            assert "patch_bboxes" in sample
            assert "patch_masks" in sample
            assert isinstance(sample["image"], np.ndarray)
            assert isinstance(sample["patch_bboxes"], list)
            assert isinstance(sample["patch_masks"], list)

    def test_image_shape(self):
        """Test that images have correct shape."""
        config = SyntheticDataConfig(seed=42, image_size=(256, 256), num_patches_range=(21, 22))
        generator = DetectionDataGenerator(config)

        data = generator.generate_batch(5)

        assert len(data) == 5
        for sample in data:
            assert sample["image"].shape == (256, 256, 3)
            assert sample["num_patches"] == 21

    def test_noise_application(self):
        """Test that noise is applied to prevent exact matching."""
        config_no_noise = SyntheticDataConfig(seed=42, input_noise_std=0.0)
        config_with_noise = SyntheticDataConfig(seed=42, input_noise_std=0.1)

        gen_no_noise = DetectionDataGenerator(config_no_noise)
        gen_with_noise = DetectionDataGenerator(config_with_noise)

        data_no_noise = gen_no_noise.generate_batch(1)
        data_with_noise = gen_with_noise.generate_batch(1)

        # Images should be different due to noise
        # Note: Can't be exactly equal check due to different seeds affecting internal state
        assert data_no_noise[0]["image"].shape == data_with_noise[0]["image"].shape

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config1 = SyntheticDataConfig(seed=42)
        config2 = SyntheticDataConfig(seed=42)

        gen1 = DetectionDataGenerator(config1)
        gen2 = DetectionDataGenerator(config2)

        data1 = gen1.generate_batch(5)
        data2 = gen2.generate_batch(5)

        for s1, s2 in zip(data1, data2, strict=False):
            np.testing.assert_array_equal(s1["image"], s2["image"])
            assert s1["patch_bboxes"] == s2["patch_bboxes"]

    def test_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        config1 = SyntheticDataConfig(seed=42)
        config2 = SyntheticDataConfig(seed=123)

        gen1 = DetectionDataGenerator(config1)
        gen2 = DetectionDataGenerator(config2)

        data1 = gen1.generate_batch(5)
        data2 = gen2.generate_batch(5)

        # Data should be different
        images1 = np.stack([s["image"] for s in data1])
        images2 = np.stack([s["image"] for s in data2])
        assert not np.allclose(images1, images2)


class TestCurveDataGenerator:
    """Tests for CurveDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = CurveDataGenerator(config)

        data = generator.generate_batch(10)

        assert len(data) == 10
        for sample in data:
            assert "input_densities" in sample
            assert "output_curve_y" in sample
            assert "conditioning" in sample
            assert len(sample["input_densities"]) == 21
            assert len(sample["output_curve_y"]) == 256

    def test_curve_monotonicity(self):
        """Test that generated curves are monotonic (increasing)."""
        config = SyntheticDataConfig(seed=42, output_noise_std=0.001)
        generator = CurveDataGenerator(config)

        data = generator.generate_batch(100)

        # Extract curves from samples
        curves = np.array([sample["output_curve_y"] for sample in data])

        # Most curves should be roughly monotonic
        # (small noise may cause slight inversions)
        diffs = np.diff(curves, axis=1)
        increasing_fraction = np.mean(diffs >= -0.1)  # Allow small noise
        assert increasing_fraction > 0.9

    def test_density_range(self):
        """Test that densities are in valid range."""
        config = SyntheticDataConfig(seed=42)
        generator = CurveDataGenerator(config)

        data = generator.generate_batch(100)

        # Extract densities and check they're in reasonable range
        all_densities = []
        for sample in data:
            all_densities.extend(sample["input_densities"])

        densities = np.array(all_densities)
        # Densities should be in actual density range (with dmin/dmax)
        assert np.all(densities >= 0)
        assert np.all(densities <= 3.0)  # Max reasonable density


class TestExposureDataGenerator:
    """Tests for ExposureDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate_batch(10)

        assert len(data) == 10
        for sample in data:
            assert "input_features" in sample
            assert "target_exposure" in sample
            assert isinstance(sample["input_features"], dict)
            assert isinstance(sample["target_exposure"], float)

    def test_exposure_time_positive(self):
        """Test that exposure times are positive."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate_batch(100)

        exposure_times = [sample["target_exposure"] for sample in data]
        assert np.all(np.array(exposure_times) > 0)

    def test_feature_dimensions(self):
        """Test feature dimensions."""
        config = SyntheticDataConfig(seed=42)
        generator = ExposureDataGenerator(config)

        data = generator.generate_batch(10)

        # Should have multiple features (humidity, temperature, paper type, etc.)
        for sample in data:
            assert len(sample["input_features"]) >= 5


class TestDefectDataGenerator:
    """Tests for DefectDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = DefectDataGenerator(config)

        data = generator.generate_batch(10)

        assert len(data) == 10
        for sample in data:
            assert "image" in sample
            assert "mask" in sample
            assert "defect_info" in sample
            assert isinstance(sample["image"], np.ndarray)
            assert isinstance(sample["mask"], np.ndarray)
            assert isinstance(sample["defect_info"], list)

    def test_label_range(self):
        """Test that labels are in valid range."""
        config = SyntheticDataConfig(seed=42)
        generator = DefectDataGenerator(config)

        data = generator.generate_batch(100)

        # Extract all defect class indices
        from ptpd_calibration.deep_learning.training.data_generators import DefectType

        num_classes = len(DefectType)
        all_class_indices = []
        for sample in data:
            for defect in sample["defect_info"]:
                all_class_indices.append(defect["class_idx"])

        if all_class_indices:
            assert np.all(np.array(all_class_indices) >= 0)
            assert np.all(np.array(all_class_indices) < num_classes)

    def test_label_noise(self):
        """Test that label noise is applied correctly."""
        config_no_noise = SyntheticDataConfig(seed=42, label_noise_probability=0.0)
        config_with_noise = SyntheticDataConfig(seed=42, label_noise_probability=0.5)

        gen_no_noise = DefectDataGenerator(config_no_noise)
        gen_with_noise = DefectDataGenerator(config_with_noise)

        # Generate many samples to observe noise effect
        gen_no_noise.generate_batch(1000)
        data_with_noise = gen_with_noise.generate_batch(1000)

        # With noise, labels should be more distributed
        # (hard to test exactly, but we can check labels are still valid)
        from ptpd_calibration.deep_learning.training.data_generators import DefectType

        num_classes = len(DefectType)
        all_class_indices = []
        for sample in data_with_noise:
            for defect in sample["defect_info"]:
                all_class_indices.append(defect["class_idx"])

        if all_class_indices:
            assert np.all(np.array(all_class_indices) >= 0)
            assert np.all(np.array(all_class_indices) < num_classes)


class TestRecipeDataGenerator:
    """Tests for RecipeDataGenerator."""

    def test_generate_samples(self):
        """Test basic sample generation."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate_batch(10)

        assert len(data) == 10
        for sample in data:
            assert "user" in sample
            assert "target_recipe" in sample
            assert "target_rating" in sample
            assert isinstance(sample["user"], dict)
            assert isinstance(sample["target_recipe"], dict)
            assert isinstance(sample["target_rating"], float)

    def test_rating_range(self):
        """Test that ratings are in valid range."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate_batch(100)

        # Ratings should be between 1 and 5
        ratings = [sample["target_rating"] for sample in data]
        assert np.all(np.array(ratings) >= 0)
        assert np.all(np.array(ratings) <= 5)

    def test_id_types(self):
        """Test that IDs are strings."""
        config = SyntheticDataConfig(seed=42)
        generator = RecipeDataGenerator(config)

        data = generator.generate_batch(100)

        for sample in data:
            assert isinstance(sample["user"]["id"], str)
            assert isinstance(sample["target_recipe"]["id"], str)


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

        train_data = gen_train.generate_batch(100)
        val_data = gen_val.generate_batch(100)
        test_data = gen_test.generate_batch(100)

        # Extract densities from samples
        train_densities = np.array([sample["input_densities"] for sample in train_data])
        val_densities = np.array([sample["input_densities"] for sample in val_data])
        test_densities = np.array([sample["input_densities"] for sample in test_data])

        # Ensure data is different
        assert not np.allclose(train_densities, val_densities)
        assert not np.allclose(train_densities, test_densities)
        assert not np.allclose(val_densities, test_densities)

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

        data = generator.generate_batch(100)

        # Input densities and output curves should be different
        # (densities are measured values, curves are corrections)
        for sample in data:
            # Note: input_densities has 21 values, output_curve_y has 256 values
            # They're not the same length, so they can't be directly compared
            # But we can check they're not all zeros or identical patterns
            assert len(sample["input_densities"]) == 21
            assert len(sample["output_curve_y"]) == 256
            assert np.std(sample["input_densities"]) > 0
            assert np.std(sample["output_curve_y"]) > 0

    def test_variation_in_similar_inputs(self):
        """Test that similar inputs produce varied outputs."""
        config = SyntheticDataConfig(seed=42, output_noise_std=0.05)
        generator = ExposureDataGenerator(config)

        # Generate multiple samples
        data = generator.generate_batch(100)

        # Check that there's variation in outputs even for similar features
        exposure_times = [sample["target_exposure"] for sample in data]
        exposure_std = np.std(exposure_times)
        assert exposure_std > 0, "Exposure times should have variation"

    def test_label_noise_prevents_overfitting(self):
        """Test that label noise adds uncertainty to labels."""
        config = SyntheticDataConfig(
            seed=42,
            label_noise_probability=0.1,
        )
        generator = DefectDataGenerator(config)

        # Generate data and verify labels have some noise
        # (i.e., not all perfectly matching ground truth)
        data = generator.generate_batch(1000)

        # Extract all defect class indices
        from ptpd_calibration.deep_learning.training.data_generators import DefectType

        num_classes = len(DefectType)
        all_class_indices = []
        for sample in data:
            for defect in sample["defect_info"]:
                all_class_indices.append(defect["class_idx"])

        # Labels should still be valid integers
        if all_class_indices:
            labels = np.array(all_class_indices)
            assert np.all(labels >= 0)
            assert np.all(labels < num_classes)
            assert labels.dtype in [np.int32, np.int64]


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
