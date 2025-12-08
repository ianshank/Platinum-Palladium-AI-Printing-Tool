"""
User Journey Tests for Deep Learning Workflow.

These tests simulate real user workflows for the deep learning curve prediction
feature, ensuring the complete user experience works as expected.
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

# Check if PyTorch is available using importlib (cleaner than try/import per Copilot feedback)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.deep,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
]


@pytest.fixture
def user_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for user files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestNewUserJourney:
    """
    User Journey: New user getting started with deep learning prediction.

    Scenario: A photographer new to the system wants to:
    1. Generate synthetic data to understand the system
    2. Train a basic model
    3. Make predictions for their paper
    4. Export the curve for use in printing
    """

    def test_new_user_complete_journey(self, user_workspace: Path):
        """Test complete new user journey."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Step 1: User generates synthetic training data
        # (In real life, this would be replaced with actual calibration data)
        print("Step 1: Generating synthetic training data...")
        training_db = generate_training_data(num_records=100, seed=42)
        assert len(training_db) == 100, "Training data should be generated"

        # Step 2: User configures and trains a model
        print("Step 2: Configuring and training model...")
        settings = DeepLearningSettings(
            model_type="curve_mlp",
            num_control_points=12,
            lut_size=256,
            hidden_dims=[64, 128, 64],
            num_epochs=5,  # Quick training for test
            batch_size=16,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        training_stats = predictor.train(training_db, num_epochs=5)

        assert predictor.is_trained, "Model should be trained"
        assert "best_val_loss" in training_stats, "Training stats should include loss"

        # Step 3: User makes a prediction for their paper
        print("Step 3: Making prediction for user's paper...")
        user_record = CalibrationRecord(
            paper_type="Arches Platine",  # User's paper
            exposure_time=180.0,
            metal_ratio=0.6,  # 60% platinum, 40% palladium
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=6.0,  # 6 drops
            developer=DeveloperType.POTASSIUM_OXALATE,
            humidity=50.0,
            temperature=21.0,
        )

        prediction = predictor.predict(user_record, return_uncertainty=True)

        assert prediction.curve is not None, "Prediction should return a curve"
        assert len(prediction.curve) == 256, "Curve should have correct LUT size"
        assert prediction.curve[0] >= 0.0, "Curve should start at or above 0"
        assert prediction.curve[-1] <= 1.0, "Curve should end at or below 1"

        # Step 4: User converts to CurveData for export
        print("Step 4: Converting to exportable format...")
        curve_data = predictor.to_curve_data(
            prediction,
            name="My Arches Platine Curve",
            paper_type="Arches Platine",
        )

        assert curve_data.name == "My Arches Platine Curve"
        assert len(curve_data.input_values) == 256
        assert len(curve_data.output_values) == 256

        # Step 5: User saves the model for future use
        print("Step 5: Saving model for future use...")
        model_path = user_workspace / "my_predictor"
        predictor.save(model_path)

        assert (model_path / "model.pt").exists(), "Model file should be saved"
        assert (model_path / "encoder.json").exists(), "Encoder should be saved"
        assert (model_path / "metadata.json").exists(), "Metadata should be saved"

        print("Journey complete! User has successfully trained and used the model.")

    def test_user_loads_saved_model(self, user_workspace: Path):
        """Test user loading a previously saved model."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Setup: Create and save a model
        training_db = generate_training_data(num_records=50, seed=42)
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 32],
            num_epochs=3,
            device="cpu",
        )

        original = DeepCurvePredictor(settings=settings)
        original.train(training_db, num_epochs=3)
        model_path = user_workspace / "saved_model"
        original.save(model_path)

        # User journey: Loading the saved model in a new session
        print("User opens new session and loads saved model...")
        loaded = DeepCurvePredictor.load(model_path)

        assert loaded.is_trained, "Loaded model should be trained"

        # Make prediction with loaded model
        test_record = CalibrationRecord(
            paper_type="Arches Platine",
            exposure_time=180.0,
            metal_ratio=0.5,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

        prediction = loaded.predict(test_record)
        assert prediction.curve is not None, "Should be able to predict with loaded model"


class TestExperiencedUserJourney:
    """
    User Journey: Experienced user fine-tuning predictions.

    Scenario: A photographer who has been using the system wants to:
    1. Add their own calibration data to improve predictions
    2. Train a custom model on their specific papers
    3. Compare predictions across different papers
    4. Get suggestions for adjustments
    """

    def test_experienced_user_custom_training(self, user_workspace: Path):
        """Test experienced user training on custom data."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import SyntheticDataGenerator

        # Step 1: User creates a database with their own calibration data
        print("Step 1: Creating database with user's calibration records...")
        user_db = CalibrationDatabase()

        # Simulate user's real calibration data (in practice, from actual measurements)
        generator = SyntheticDataGenerator()

        # Add records for papers the user actually uses
        user_papers = [
            ("Arches Platine", 0.6),  # User's go-to paper with 60% Pt
            ("Bergger COT320", 0.5),  # Secondary paper
        ]

        for paper_name, metal_ratio in user_papers:
            for exposure in [150, 180, 210]:
                for contrast in [4.0, 6.0, 8.0]:
                    record = generator.generate_record(
                        metal_ratio=metal_ratio,
                        exposure_time=float(exposure),
                        contrast_amount=contrast,
                    )
                    # Override paper type to match user's
                    record = CalibrationRecord(
                        paper_type=paper_name,
                        exposure_time=record.exposure_time,
                        metal_ratio=metal_ratio,
                        chemistry_type=record.chemistry_type,
                        contrast_agent=record.contrast_agent,
                        contrast_amount=contrast,
                        developer=record.developer,
                        humidity=record.humidity,
                        temperature=record.temperature,
                        measured_densities=record.measured_densities,
                    )
                    user_db.add_record(record)

        assert len(user_db) >= 18, "Should have user's custom records"

        # Step 2: Train on user's specific data
        print("Step 2: Training model on user's data...")
        settings = DeepLearningSettings(
            num_control_points=12,
            lut_size=128,
            hidden_dims=[64, 128, 64],
            num_epochs=10,
            batch_size=8,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        stats = predictor.train(user_db, num_epochs=10)
        assert "num_samples" in stats, "Training should return stats including num_samples"

        # Step 3: Compare predictions for different papers
        print("Step 3: Comparing predictions across papers...")
        predictions = {}

        for paper_name, default_ratio in user_papers:
            record = CalibrationRecord(
                paper_type=paper_name,
                exposure_time=180.0,
                metal_ratio=default_ratio,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=6.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            predictions[paper_name] = predictor.predict(record)

        # Verify predictions exist for all papers
        for paper_name in [p[0] for p in user_papers]:
            assert paper_name in predictions
            assert predictions[paper_name].curve is not None

        # Step 4: Get adjustment suggestions
        print("Step 4: Getting adjustment suggestions...")
        target_curve = np.linspace(0.1, 0.9, settings.lut_size)

        test_record = CalibrationRecord(
            paper_type="Arches Platine",
            exposure_time=180.0,
            metal_ratio=0.6,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=6.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

        suggestions = predictor.suggest_adjustments(test_record, target_curve)

        assert "suggestions" in suggestions, "Should return suggestions"
        assert "adjustments" in suggestions, "Should return adjustments"
        assert "current_curve" in suggestions, "Should include current curve"

        print("Experienced user journey complete!")


class TestExperimentalWorkflow:
    """
    User Journey: Exploring different process parameters.

    Scenario: A photographer experimenting with:
    1. Different metal ratios
    2. Different exposure times
    3. Different contrast agent amounts
    """

    def test_parameter_exploration_workflow(self):
        """Test exploring different parameters."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Setup: Train a model
        training_db = generate_training_data(num_records=200, seed=42)
        settings = DeepLearningSettings(
            num_control_points=12,
            lut_size=64,
            hidden_dims=[64, 128, 64],
            num_epochs=10,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(training_db, num_epochs=10)

        # Experiment 1: Explore metal ratios
        print("Experiment 1: Exploring metal ratios...")
        metal_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        ratio_curves = {}

        for ratio in metal_ratios:
            record = CalibrationRecord(
                paper_type="Arches Platine",
                exposure_time=180.0,
                metal_ratio=ratio,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            ratio_curves[ratio] = predictor.predict(record)

        # Verify all predictions work
        for ratio, prediction in ratio_curves.items():
            assert prediction.curve is not None
            assert len(prediction.curve) == 64

        # Experiment 2: Explore exposure times
        print("Experiment 2: Exploring exposure times...")
        exposures = [120, 150, 180, 210, 240]
        exposure_curves = {}

        for exposure in exposures:
            record = CalibrationRecord(
                paper_type="Arches Platine",
                exposure_time=float(exposure),
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            exposure_curves[exposure] = predictor.predict(record)

        # Experiment 3: Explore contrast amounts
        print("Experiment 3: Exploring contrast amounts...")
        contrast_amounts = [0.0, 3.0, 5.0, 7.0, 10.0]
        contrast_curves = {}

        for amount in contrast_amounts:
            agent = ContrastAgent.NA2 if amount > 0 else ContrastAgent.NONE
            record = CalibrationRecord(
                paper_type="Arches Platine",
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=agent,
                contrast_amount=amount,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
            contrast_curves[amount] = predictor.predict(record)

        # Verify all experiments produced valid results
        assert len(ratio_curves) == 5
        assert len(exposure_curves) == 5
        assert len(contrast_curves) == 5

        print("Parameter exploration complete!")


class TestBatchProcessingWorkflow:
    """
    User Journey: Processing multiple images with different curve needs.

    Scenario: A photographer needs to:
    1. Generate curves for multiple paper/chemistry combinations
    2. Apply consistent processing across a project
    """

    def test_batch_curve_generation(self, user_workspace: Path):
        """Test generating curves in batch."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Setup: Train model
        training_db = generate_training_data(num_records=100, seed=42)
        settings = DeepLearningSettings(
            num_control_points=12,
            lut_size=128,
            hidden_dims=[64, 64],
            num_epochs=5,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(training_db, num_epochs=5)

        # Define batch of curve requirements
        curve_requirements = [
            {
                "name": "Portrait Series - High Key",
                "paper": "Arches Platine",
                "metal_ratio": 0.7,
                "contrast": 4.0,
            },
            {
                "name": "Portrait Series - Low Key",
                "paper": "Arches Platine",
                "metal_ratio": 0.7,
                "contrast": 8.0,
            },
            {
                "name": "Landscape Series",
                "paper": "Bergger COT320",
                "metal_ratio": 0.5,
                "contrast": 6.0,
            },
            {
                "name": "Fine Art Series",
                "paper": "Hahnemühle Platinum Rag",
                "metal_ratio": 0.4,
                "contrast": 5.0,
            },
        ]

        # Generate curves in batch
        print("Generating curves for batch...")
        generated_curves = []

        for req in curve_requirements:
            record = CalibrationRecord(
                paper_type=req["paper"],
                exposure_time=180.0,
                metal_ratio=req["metal_ratio"],
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=req["contrast"],
                developer=DeveloperType.POTASSIUM_OXALATE,
            )

            prediction = predictor.predict(record)
            curve_data = predictor.to_curve_data(
                prediction,
                name=req["name"],
                paper_type=req["paper"],
            )

            generated_curves.append(
                {
                    "name": req["name"],
                    "curve_data": curve_data,
                    "prediction": prediction,
                }
            )

        # Verify all curves generated
        assert len(generated_curves) == len(curve_requirements)

        # Save batch results
        batch_output = user_workspace / "batch_curves"
        batch_output.mkdir(exist_ok=True)

        manifest = {"curves": []}
        for curve_info in generated_curves:
            curve_path = batch_output / f"{curve_info['name'].replace(' ', '_')}.json"
            curve_data = curve_info["curve_data"]

            # Save curve data
            with open(curve_path, "w") as f:
                json.dump(
                    {
                        "name": curve_data.name,
                        "paper_type": curve_data.paper_type,
                        "input_values": curve_data.input_values,
                        "output_values": curve_data.output_values,
                    },
                    f,
                    indent=2,
                )

            manifest["curves"].append(
                {
                    "name": curve_info["name"],
                    "file": str(curve_path.name),
                }
            )

        # Save manifest
        with open(batch_output / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Verify files exist
        assert (batch_output / "manifest.json").exists()
        assert len(list(batch_output.glob("*.json"))) == len(curve_requirements) + 1

        print(f"Batch processing complete! Generated {len(generated_curves)} curves.")


class TestErrorRecoveryJourney:
    """
    User Journey: Handling errors and edge cases gracefully.
    """

    def test_prediction_with_unknown_paper(self):
        """Test prediction with a paper type not in training data."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        # Train on limited paper types
        training_db = generate_training_data(num_records=100, seed=42)
        settings = DeepLearningSettings(
            num_control_points=8,
            lut_size=64,
            hidden_dims=[32, 32],
            num_epochs=3,
            device="cpu",
        )

        predictor = DeepCurvePredictor(settings=settings)
        predictor.train(training_db, num_epochs=3)

        # Try predicting for unknown paper
        unknown_record = CalibrationRecord(
            paper_type="Unknown Exotic Paper XYZ",  # Not in training data
            exposure_time=180.0,
            metal_ratio=0.5,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

        # Should still work (will use default encoding or similar paper)
        prediction = predictor.predict(unknown_record)
        assert prediction.curve is not None
        assert len(prediction.curve) == 64

    def test_model_not_trained_error(self):
        """Test proper error when trying to predict without training."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.exceptions import ModelNotTrainedError
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        settings = DeepLearningSettings()
        predictor = DeepCurvePredictor(settings=settings)

        record = CalibrationRecord(
            paper_type="Test Paper",
            exposure_time=180.0,
            metal_ratio=0.5,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

        with pytest.raises(ModelNotTrainedError):
            predictor.predict(record)

    def test_empty_database_training_error(self):
        """Test proper error when training with empty database."""
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.exceptions import DatasetError
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        empty_db = CalibrationDatabase()
        settings = DeepLearningSettings()
        predictor = DeepCurvePredictor(settings=settings)

        with pytest.raises(DatasetError):
            predictor.train(empty_db, num_epochs=3)
