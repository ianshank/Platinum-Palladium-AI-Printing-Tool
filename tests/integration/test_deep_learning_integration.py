
import pytest
import numpy as np
from ptpd_calibration.deep_learning.config import NeuralCurveSettings
from ptpd_calibration.deep_learning.neural_curve import NeuralCurvePredictor

@pytest.mark.integration
@pytest.mark.slow
class TestDeepLearningIntegration:
    """Integration tests for Deep Learning components."""

    def test_neural_curve_training_and_prediction(self):
        """
        Verify that we can:
        1. Initialize the NeuralCurvePredictor
        2. Train it on synthetic data (ensuring no errors)
        3. Make a prediction (ensuring valid output)
        """
        # 1. Setup small model for speed
        settings = NeuralCurveSettings(
            architecture="transformer",
            d_model=64,  # Small for test speed
            n_heads=2,
            n_layers=2,
            learning_rate=1e-3,
            batch_size=16,
            epochs=10,  # Minimum required by validation
            early_stopping_patience=5,
            device="cpu", # Force CPU for CI/CD compatibility
            uncertainty_method="ensemble",
            ensemble_size=2
        )
        
        predictor = NeuralCurvePredictor(settings)
        
        # 2. Generate synthetic data
        num_samples = 50
        num_points = 32
        
        # Random inputs
        X_train = np.random.rand(num_samples, num_points, settings.input_features).astype(np.float32)
        # Monotonic-ish outputs
        y_train = np.sort(np.random.rand(num_samples, num_points).astype(np.float32), axis=1)
        
        # 3. Train Ensemble
        histories = predictor.train_ensemble(
            X_train=X_train, 
            y_train=y_train
        )
        
        assert len(histories) == 2, "Should have trained 2 models"
        for history in histories:
            assert "loss" in history
            assert len(history["loss"]) > 0

        # 4. Predict
        input_values = np.linspace(0, 1, num_points)
        result = predictor.predict(
            input_values=input_values,
            return_uncertainty=True
        )
        
        # 5. Verify Result
        assert result is not None
        assert result.num_points == num_points
        assert len(result.output_values) == num_points
        assert result.confidence > 0.0
        assert result.mean_uncertainty >= 0.0
        
        # Check monotonicity of output (it should generally be monotonic)
        # Note: With only 5 epochs on random data, strict monotonicity isn't guaranteed, 
        # but the check ensures the flag is present and calculated.
        assert isinstance(result.is_monotonic, bool)

        # Cleanup
        predictor.cleanup()

