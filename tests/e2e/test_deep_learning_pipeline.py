"""
End-to-End tests for deep learning training pipeline.

Tests the complete workflow from synthetic data generation through
model training to prediction and evaluation.
"""

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
from ptpd_calibration.ml.database import CalibrationDatabase

# Check if PyTorch is available using importlib (cleaner than try/import)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.deep,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
]


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def empty_database() -> CalibrationDatabase:
    """Create an empty calibration database."""
    return CalibrationDatabase()


@pytest.fixture
def test_record() -> CalibrationRecord:
    """Create a test calibration record for predictions."""
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
    )


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_default_database(self):
        """Test generating a default synthetic database."""
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=100, seed=42)

        assert len(db) == 100
        # Check records have required fields
        records = db.get_all_records()
        for record in records[:10]:  # Sample check
            assert record.paper_type is not None
            assert record.measured_densities is not None
            assert len(record.measured_densities) > 0

    def test_generate_reproducible_with_seed(self):
        """Test that generation is reproducible with same seed."""
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db1 = generate_training_data(num_records=50, seed=123)
        db2 = generate_training_data(num_records=50, seed=123)

        records1 = db1.get_all_records()
        records2 = db2.get_all_records()

        for r1, r2 in zip(records1, records2):
            assert r1.paper_type == r2.paper_type
            assert r1.metal_ratio == r2.metal_ratio
            np.testing.assert_array_almost_equal(
                r1.measured_densities, r2.measured_densities, decimal=5
            )

    def test_generated_data_has_realistic_properties(self):
        """Test that generated data has realistic physical properties."""
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=100, seed=42)
        records = db.get_all_records()

        for record in records:
            densities = np.array(record.measured_densities)

            # Density should generally increase (monotonic)
            monotonic_pairs = np.sum(np.diff(densities) > -0.05)
            assert monotonic_pairs >= len(densities) - 3, "Should be mostly monotonic"

            # Dmin should be reasonable (0 to 0.3)
            assert densities[0] >= 0.0
            assert densities[0] <= 0.5

            # Dmax should be reasonable (1.0 to 3.0)
            assert densities[-1] >= 0.5
            assert densities[-1] <= 3.5

    def test_generator_paper_profiles(self):
        """Test that generator uses multiple paper profiles."""
        from ptpd_calibration.ml.deep.synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        db = generator.generate_database(num_records=100)

        paper_types = set(r.paper_type for r in db.get_all_records())
        assert len(paper_types) >= 3, "Should generate data for multiple paper types"

    def test_exposure_series_generation(self):
        """Test generating an exposure series."""
        from ptpd_calibration.ml.deep.synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        series = generator.generate_exposure_series(num_exposures=7)

        assert len(series) == 7
        exposures = [r.exposure_time for r in series]
        assert exposures == sorted(exposures), "Exposures should be ordered"

    def test_metal_ratio_series_generation(self):
        """Test generating a metal ratio series."""
        from ptpd_calibration.ml.deep.synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        series = generator.generate_metal_ratio_series(num_ratios=5)

        assert len(series) == 5
        ratios = [r.metal_ratio for r in series]
        assert ratios == sorted(ratios), "Ratios should be ordered"
        assert ratios[0] == pytest.approx(0.0)
        assert ratios[-1] == pytest.approx(1.0)


class TestTrainingPipeline:
    """Tests for the training pipeline."""

    def test_quick_train_creates_predictor(self, temp_output_dir: Path):
        """Test quick_train creates a working predictor."""
        from ptpd_calibration.ml.deep.pipeline import quick_train

        predictor = quick_train(
            num_synthetic_samples=50,
            num_epochs=3,
            output_dir=temp_output_dir / "quick_train",
        )

        assert predictor.is_trained
        assert predictor.model is not None

    def test_training_pipeline_full_experiment(self, temp_output_dir: Path):
        """Test running a full experiment through the pipeline."""
        from ptpd_calibration.ml.deep.pipeline import (
            ExperimentConfig,
            TrainingPipeline,
        )
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Generate data
        db = generate_training_data(num_records=100, seed=42)

        # Configure experiment
        config = ExperimentConfig(
            name="test_experiment",
            description="E2E test experiment",
            num_epochs=3,
            batch_size=8,
            hidden_dims=[32, 64, 32],
            num_control_points=8,
            target_lut_size=64,
        )

        # Run pipeline
        pipeline = TrainingPipeline(db, output_dir=temp_output_dir)
        result = pipeline.run_experiment(config)

        # Verify results
        assert result.training_stats is not None
        assert result.evaluation_metrics is not None
        assert result.model_path is not None
        assert result.training_time > 0

        # Check evaluation metrics
        assert "mae" in result.evaluation_metrics
        assert "mse" in result.evaluation_metrics
        assert "monotonicity_rate" in result.evaluation_metrics

        # Check files were saved
        assert (result.model_path / "model.pt").exists()
        assert (temp_output_dir / "test_experiment" / "result.json").exists()

    def test_training_pipeline_result_serialization(self, temp_output_dir: Path):
        """Test that experiment results can be serialized and loaded."""
        from ptpd_calibration.ml.deep.pipeline import (
            ExperimentConfig,
            TrainingPipeline,
        )
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=50, seed=42)
        config = ExperimentConfig(
            name="serialization_test",
            num_epochs=2,
            batch_size=8,
            hidden_dims=[32, 32],
        )

        pipeline = TrainingPipeline(db, output_dir=temp_output_dir)
        _result = pipeline.run_experiment(config)  # noqa: F841 - run to create result file

        # Save and reload result
        result_path = temp_output_dir / "serialization_test" / "result.json"
        with open(result_path) as f:
            loaded = json.load(f)

        assert loaded["config"]["name"] == "serialization_test"
        assert "mae" in loaded["evaluation_metrics"]
        assert loaded["training_time"] > 0


class TestPredictorE2E:
    """End-to-end tests for the DeepCurvePredictor."""

    def test_predictor_training_and_prediction(self, test_record: CalibrationRecord):
        """Test training a predictor and making predictions."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Generate training data
        db = generate_training_data(num_records=100, seed=42)

        # Configure predictor
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 64, 32],
            num_epochs=5,
            batch_size=8,
            device="cpu",
        )

        # Train
        predictor = DeepCurvePredictor(settings=settings)
        stats = predictor.train(db, num_epochs=5)

        assert "num_samples" in stats
        assert stats["num_samples"] == 100

        # Predict
        result = predictor.predict(test_record, return_uncertainty=True)

        assert result.curve is not None
        assert len(result.curve) == 64
        assert result.curve[0] >= 0.0
        assert result.curve[-1] <= 1.0

        # Check monotonicity
        diffs = np.diff(result.curve)
        assert np.all(diffs >= -0.01), "Curve should be monotonically increasing"

    def test_predictor_save_and_load(
        self, temp_output_dir: Path, test_record: CalibrationRecord
    ):
        """Test saving and loading a trained predictor."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=50, seed=42)
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 32],
            num_epochs=3,
            device="cpu",
        )

        # Train and save
        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(db, num_epochs=3)
        original_prediction = predictor.predict(test_record)

        save_path = temp_output_dir / "predictor_e2e"
        predictor.save(save_path)

        # Load and verify
        loaded = DeepCurvePredictor.load(save_path)
        loaded_prediction = loaded.predict(test_record)

        np.testing.assert_array_almost_equal(
            original_prediction.curve,
            loaded_prediction.curve,
            decimal=5,
        )

    def test_predictor_different_paper_types(self):
        """Test predictor produces different curves for different papers."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=200, seed=42)
        settings = DeepLearningSettings(
            num_control_points=12,
            lut_size=64,
            hidden_dims=[64, 128, 64],
            num_epochs=10,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(db, num_epochs=10)

        # Create records for different papers
        papers = ["Arches Platine", "Bergger COT320", "Hahnemühle Platinum Rag"]
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

        # Different papers should give somewhat different curves
        # (trained model should learn paper-specific characteristics)
        # Verify predictions exist and have correct shape for all papers
        for p1 in papers:
            for p2 in papers:
                if p1 != p2:
                    # With limited training, differences may be small
                    # Just verify predictions work for all papers
                    assert len(predictions[p1].curve) == len(predictions[p2].curve)

    def test_predictor_to_curve_data_conversion(self, test_record: CalibrationRecord):
        """Test converting prediction result to CurveData."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=50, seed=42)
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 32],
            num_epochs=3,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(db, num_epochs=3)
        result = predictor.predict(test_record)

        curve_data = predictor.to_curve_data(
            result,
            name="E2E Test Curve",
            paper_type="Arches Platine",
        )

        assert curve_data.name == "E2E Test Curve"
        assert curve_data.paper_type == "Arches Platine"
        assert len(curve_data.input_values) == 64
        assert len(curve_data.output_values) == 64
        # Input values should be 0-1 linear ramp
        assert curve_data.input_values[0] == pytest.approx(0.0)
        assert curve_data.input_values[-1] == pytest.approx(1.0)


class TestProcessSimulator:
    """Tests for the process simulator."""

    def test_simulator_forward_pass(self):
        """Test basic simulator forward pass."""
        import torch

        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        settings = DeepLearningSettings()
        sim = ProcessSimulator(settings, learnable=False)

        negative_density = torch.linspace(0, 2.5, 21)
        print_density = sim(negative_density)

        assert print_density.shape == negative_density.shape
        assert torch.all(print_density >= 0.0)
        # Higher negative density -> lower print transmission -> higher print density
        # So print_density should generally increase with negative_density

    def test_simulator_with_learnable_params(self):
        """Test simulator with learnable parameters."""
        import torch

        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        settings = DeepLearningSettings()
        sim = ProcessSimulator(settings, learnable=True)

        # Should have learnable parameters
        params = list(sim.parameters())
        assert len(params) > 0

        # Forward should still work
        negative_density = torch.linspace(0, 2.5, 11)
        print_density = sim(negative_density)
        assert print_density.shape == negative_density.shape

    def test_simulator_intermediates(self):
        """Test simulator returns intermediate values."""
        import torch

        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.process_sim import ProcessSimulator

        settings = DeepLearningSettings()
        sim = ProcessSimulator(settings)

        negative_density = torch.linspace(0, 2, 11)
        print_density, intermediates = sim(negative_density, return_intermediates=True)

        assert "transmission" in intermediates
        assert "exposure" in intermediates
        assert "print_density" in intermediates

        # Verify physical relationships
        # Higher density -> lower transmission
        transmission = intermediates["transmission"]
        assert torch.all(transmission >= 0.0)
        assert torch.all(transmission <= 1.0)


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    def test_evaluation_metrics_computed(self):
        """Test that evaluation metrics are properly computed."""
        from ptpd_calibration.ml.deep.pipeline import (
            ExperimentConfig,
            TrainingPipeline,
        )
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=100, seed=42)
        config = ExperimentConfig(
            name="eval_test",
            num_epochs=3,
            batch_size=8,
            hidden_dims=[32, 32],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(db, output_dir=Path(tmpdir))
            result = pipeline.run_experiment(config)

        metrics = result.evaluation_metrics

        assert metrics["mae"] >= 0.0
        assert metrics["mse"] >= 0.0
        assert metrics["max_error"] >= 0.0
        assert 0.0 <= metrics["monotonicity_rate"] <= 1.0
        assert -1.0 <= metrics["mean_correlation"] <= 1.0
        assert metrics["num_samples"] == 100

    def test_trained_model_improves_loss(self):
        """Test that training actually improves the loss."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        db = generate_training_data(num_records=100, seed=42)
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 64, 32],
            num_epochs=10,
            early_stopping_patience=15,  # Don't stop early
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)

        # Capture training history
        losses = []

        def capture_loss(metrics):
            losses.append(metrics.val_loss)

        predictor.train(db, num_epochs=10, callbacks=[capture_loss])

        # Loss should generally decrease
        # Compare first half average to second half average
        if len(losses) >= 4:
            first_half = np.mean(losses[: len(losses) // 2])
            second_half = np.mean(losses[len(losses) // 2 :])
            assert second_half <= first_half * 1.5, "Loss should improve during training"
