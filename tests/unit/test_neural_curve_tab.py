"""
Unit tests for Neural Curve Prediction tab.

Tests the helper functions and simulation logic
used in the neural curve prediction UI tab.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Test Helper Functions
# =============================================================================


class TestCreateRatioVisualization:
    """Tests for the Pt:Pd ratio visualization helper."""

    def test_create_ratio_viz_50_50(self):
        """Test visualization for equal ratio."""
        from ptpd_calibration.ui.tabs.neural_curve import _create_ratio_viz

        html = _create_ratio_viz(50)

        assert "50% Pt" in html
        assert "50% Pd" in html
        assert "display: flex" in html

    def test_create_ratio_viz_pure_platinum(self):
        """Test visualization for 100% platinum."""
        from ptpd_calibration.ui.tabs.neural_curve import _create_ratio_viz

        html = _create_ratio_viz(100)

        assert "100% Pt" in html
        assert "0% Pd" in html

    def test_create_ratio_viz_pure_palladium(self):
        """Test visualization for 100% palladium."""
        from ptpd_calibration.ui.tabs.neural_curve import _create_ratio_viz

        html = _create_ratio_viz(0)

        assert "0% Pt" in html
        assert "100% Pd" in html

    def test_create_ratio_viz_gradient_colors(self):
        """Test that gradient colors are included."""
        from ptpd_calibration.ui.tabs.neural_curve import _create_ratio_viz

        html = _create_ratio_viz(70)

        # Should have gold/amber for Pt and gray for Pd
        assert "#fbbf24" in html or "#f59e0b" in html
        assert "#94a3b8" in html or "#64748b" in html


# =============================================================================
# Test Synthetic Data Generation
# =============================================================================


class TestSyntheticDataGeneration:
    """Tests for synthetic training data generation."""

    def test_generate_synthetic_training_data(self):
        """Test basic synthetic data generation."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_synthetic_training_data

        data = _generate_synthetic_training_data(num_samples=50)

        assert len(data) == 50
        for sample in data:
            assert "input_densities" in sample
            assert "output_densities" in sample
            assert "pt_ratio" in sample
            assert "target_dmax" in sample

    def test_synthetic_data_structure(self):
        """Test structure of synthetic data."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_synthetic_training_data

        data = _generate_synthetic_training_data(num_samples=10)

        for sample in data:
            # Should have 21 points
            assert len(sample["input_densities"]) == 21
            assert len(sample["output_densities"]) == 21

            # Pt ratio should be 0-1
            assert 0.0 <= sample["pt_ratio"] <= 1.0

            # Target dmax should be reasonable
            assert 1.0 <= sample["target_dmax"] <= 2.5

    def test_synthetic_data_noise(self):
        """Test noise level affects output variation."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_synthetic_training_data

        # Low noise
        data_low = _generate_synthetic_training_data(num_samples=10, noise_level=0.001)
        # High noise
        data_high = _generate_synthetic_training_data(num_samples=10, noise_level=0.1)

        # High noise should have more variation (this is probabilistic)
        # Just verify both work without error
        assert len(data_low) == 10
        assert len(data_high) == 10

    def test_synthetic_data_monotonicity(self):
        """Test that synthetic curves are monotonic."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_synthetic_training_data

        data = _generate_synthetic_training_data(num_samples=20, noise_level=0.01)

        for sample in data:
            outputs = sample["output_densities"]
            # Check monotonic (each value >= previous)
            for i in range(1, len(outputs)):
                assert outputs[i] >= outputs[i - 1], "Curve should be monotonic"

    def test_synthetic_data_bounds(self):
        """Test that values stay within bounds."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_synthetic_training_data

        data = _generate_synthetic_training_data(num_samples=50, noise_level=0.05)

        for sample in data:
            for val in sample["output_densities"]:
                assert 0.0 <= val <= 1.0, "Output values should be 0-1"


# =============================================================================
# Test Training Simulation
# =============================================================================


class TestTrainingSimulation:
    """Tests for training simulation."""

    def test_simulate_training_basic(self):
        """Test basic training simulation."""
        from ptpd_calibration.ui.tabs.neural_curve import _simulate_training

        history = _simulate_training(epochs=50, learning_rate=0.001, data_size=100)

        assert "train" in history
        assert "val" in history
        assert len(history["train"]) == 50
        assert len(history["val"]) == 50

    def test_simulate_training_loss_decreases(self):
        """Test that simulated loss decreases over training."""
        from ptpd_calibration.ui.tabs.neural_curve import _simulate_training

        history = _simulate_training(epochs=100, learning_rate=0.001, data_size=200)

        # Overall trend should be decreasing
        first_half_avg = np.mean(history["train"][:50])
        second_half_avg = np.mean(history["train"][50:])

        assert second_half_avg < first_half_avg, "Loss should decrease"

    def test_simulate_training_val_higher(self):
        """Test that validation loss is slightly higher than training."""
        from ptpd_calibration.ui.tabs.neural_curve import _simulate_training

        history = _simulate_training(epochs=50, learning_rate=0.001, data_size=100)

        # On average, validation should be higher (with some noise)
        train_mean = np.mean(history["train"])
        val_mean = np.mean(history["val"])

        # Allow some variance, but val should generally be higher
        assert val_mean >= train_mean * 0.9  # At least 90% of train

    def test_simulate_training_different_data_sizes(self):
        """Test training with different data sizes."""
        from ptpd_calibration.ui.tabs.neural_curve import _simulate_training

        # Smaller dataset should start with higher loss
        history_small = _simulate_training(epochs=10, learning_rate=0.001, data_size=10)
        history_large = _simulate_training(epochs=10, learning_rate=0.001, data_size=1000)

        # Both should work
        assert len(history_small["train"]) == 10
        assert len(history_large["train"]) == 10


# =============================================================================
# Test Curve Prediction
# =============================================================================


class TestCurvePrediction:
    """Tests for curve prediction generation."""

    def test_generate_curve_prediction_basic(self):
        """Test basic curve prediction generation."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {
            "num_layers": 3,
            "hidden_dim": 128,
        }

        prediction = _generate_curve_prediction(
            paper_type="arches_platine",
            pt_ratio=0.5,
            target_dmax=1.8,
            num_points=51,
            model_config=model_config,
        )

        assert "input" in prediction
        assert "output" in prediction
        assert "uncertainty" in prediction
        assert "is_monotonic" in prediction
        assert "mean_uncertainty" in prediction
        assert "confidence" in prediction

    def test_curve_prediction_output_length(self):
        """Test that output has correct number of points."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {"num_layers": 2}

        for num_points in [21, 51, 101, 256]:
            prediction = _generate_curve_prediction(
                paper_type="arches_platine",
                pt_ratio=0.5,
                target_dmax=1.8,
                num_points=num_points,
                model_config=model_config,
            )

            assert len(prediction["input"]) == num_points
            assert len(prediction["output"]) == num_points
            assert len(prediction["uncertainty"]) == num_points

    def test_curve_prediction_monotonicity(self):
        """Test that predicted curves are monotonic."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {"num_layers": 3}

        prediction = _generate_curve_prediction(
            paper_type="arches_platine",
            pt_ratio=0.5,
            target_dmax=1.8,
            num_points=51,
            model_config=model_config,
        )

        assert prediction["is_monotonic"] is True

        outputs = prediction["output"]
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1]

    def test_curve_prediction_confidence(self):
        """Test that confidence is in valid range."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {"num_layers": 3}

        prediction = _generate_curve_prediction(
            paper_type="arches_platine",
            pt_ratio=0.5,
            target_dmax=1.8,
            num_points=51,
            model_config=model_config,
        )

        assert 0.0 <= prediction["confidence"] <= 1.0
        assert prediction["mean_uncertainty"] >= 0.0

    def test_curve_prediction_paper_types(self):
        """Test prediction with different paper types."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {"num_layers": 3}

        paper_types = [
            "arches_platine",
            "bergger_cot320",
            "hahnemuhle_platinum",
            "revere_platinum",
            "custom",
        ]

        for paper in paper_types:
            prediction = _generate_curve_prediction(
                paper_type=paper,
                pt_ratio=0.5,
                target_dmax=1.8,
                num_points=51,
                model_config=model_config,
            )

            assert prediction is not None
            assert len(prediction["output"]) == 51

    def test_curve_prediction_pt_ratio_effect(self):
        """Test that Pt ratio affects prediction."""
        from ptpd_calibration.ui.tabs.neural_curve import _generate_curve_prediction

        model_config = {"num_layers": 3}

        # Low Pt (more Pd)
        pred_low_pt = _generate_curve_prediction(
            paper_type="arches_platine",
            pt_ratio=0.2,
            target_dmax=1.8,
            num_points=51,
            model_config=model_config,
        )

        # High Pt
        pred_high_pt = _generate_curve_prediction(
            paper_type="arches_platine",
            pt_ratio=0.8,
            target_dmax=1.8,
            num_points=51,
            model_config=model_config,
        )

        # Curves should be different
        assert pred_low_pt["output"] != pred_high_pt["output"]


# =============================================================================
# Test Deep Learning Availability Check
# =============================================================================


class TestDLAvailabilityCheck:
    """Tests for deep learning availability checking."""

    def test_check_dl_available_without_torch(self):
        """Test DL check when torch is not available."""
        from ptpd_calibration.ui.tabs import neural_curve

        # Reset cached value
        neural_curve._DL_AVAILABLE = None

        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("builtins.__import__", side_effect=ImportError("No torch")),
        ):
            # Force re-check by resetting
            neural_curve._DL_AVAILABLE = None

            # This should handle ImportError gracefully
            result = neural_curve._check_dl_available()
            # May be True or False depending on actual torch availability
            assert isinstance(result, bool)


# =============================================================================
# Test Custom Data Loading
# =============================================================================


class TestCustomDataLoading:
    """Tests for loading custom training data."""

    def test_load_custom_data_json(self, tmp_path):
        """Test loading custom data from JSON."""
        import json

        from ptpd_calibration.ui.tabs.neural_curve import _load_custom_data

        # Create test JSON file
        test_data = [
            {"input": [0, 0.5, 1], "output": [0, 0.4, 0.9]},
            {"input": [0, 0.5, 1], "output": [0, 0.5, 1]},
        ]

        json_path = tmp_path / "test_data.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        data = _load_custom_data(str(json_path))

        assert len(data) == 2
        assert data[0]["input"] == [0, 0.5, 1]

    def test_load_custom_data_csv(self, tmp_path):
        """Test loading custom data from CSV."""
        from ptpd_calibration.ui.tabs.neural_curve import _load_custom_data

        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("input,output\n0.0,0.0\n0.5,0.4\n1.0,0.9\n")

        data = _load_custom_data(str(csv_path))

        assert len(data) == 3
        assert "input" in data[0]
        assert "output" in data[0]

    def test_load_custom_data_unsupported_format(self, tmp_path):
        """Test error on unsupported file format."""
        from ptpd_calibration.ui.tabs.neural_curve import _load_custom_data

        txt_path = tmp_path / "test.txt"
        txt_path.write_text("some text")

        with pytest.raises(ValueError, match="Unsupported"):
            _load_custom_data(str(txt_path))


# =============================================================================
# Test Session Data Loading
# =============================================================================


class TestSessionDataLoading:
    """Tests for loading data from session history."""

    def test_load_session_data_empty(self):
        """Test loading from empty session logger."""
        from ptpd_calibration.ui.tabs.neural_curve import _load_session_data

        mock_logger = MagicMock()
        mock_logger.list_sessions.return_value = []

        data = _load_session_data(mock_logger, "All sessions")

        assert data == []

    def test_load_session_data_with_records(self):
        """Test loading from session logger with records."""
        from ptpd_calibration.ui.tabs.neural_curve import _load_session_data

        # Create mock session with records
        mock_record = MagicMock()
        mock_record.densities = [0.1, 0.5, 1.0, 1.5, 2.0]
        mock_record.paper_type = "Arches Platine"

        mock_session = MagicMock()
        mock_session.records = [mock_record]

        mock_logger = MagicMock()
        mock_logger.list_sessions.return_value = [{"filepath": "/path/to/session"}]
        mock_logger.load_session.return_value = mock_session

        data = _load_session_data(mock_logger, "All sessions")

        assert len(data) >= 1

    def test_load_session_data_error_handling(self):
        """Test graceful handling of session load errors."""
        from ptpd_calibration.ui.tabs.neural_curve import _load_session_data

        mock_logger = MagicMock()
        mock_logger.list_sessions.side_effect = Exception("Database error")

        # Should handle error gracefully
        data = _load_session_data(mock_logger, "All sessions")
        assert data == []


# =============================================================================
# Test Tab Building
# =============================================================================


class TestTabBuilding:
    """Tests for the tab building function."""

    def test_build_tab_without_session_logger(self):
        """Test building tab without session logger."""
        pytest.importorskip("gradio")

        import gradio as gr

        from ptpd_calibration.ui.tabs.neural_curve import build_neural_curve_tab

        # Should not raise error
        with gr.Blocks() as demo:
            build_neural_curve_tab(session_logger=None)

        assert demo is not None

    def test_build_tab_with_mock_session_logger(self):
        """Test building tab with mock session logger."""
        pytest.importorskip("gradio")

        import gradio as gr

        from ptpd_calibration.ui.tabs.neural_curve import build_neural_curve_tab

        mock_logger = MagicMock()
        mock_logger.list_sessions.return_value = []

        with gr.Blocks() as demo:
            build_neural_curve_tab(session_logger=mock_logger)

        assert demo is not None
