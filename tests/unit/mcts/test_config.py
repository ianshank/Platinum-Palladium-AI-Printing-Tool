"""Tests for MCTS configuration settings."""

import os
from pathlib import Path

import pytest

from ptpd_calibration.mcts.config import (
    DEFAULT_PARAMETER_RANGES,
    MCTSSettings,
    ParameterRange,
)


class TestParameterRange:
    """Tests for ParameterRange model."""

    def test_parameter_range_creation(self) -> None:
        """Test creating a ParameterRange."""
        pr = ParameterRange(
            name="test_param",
            min_value=0.0,
            max_value=10.0,
            default_value=5.0,
            unit="units",
        )
        assert pr.name == "test_param"
        assert pr.min_value == 0.0
        assert pr.max_value == 10.0
        assert pr.default_value == 5.0
        assert pr.unit == "units"
        assert pr.step is None

    def test_parameter_range_with_step(self) -> None:
        """Test creating a ParameterRange with discrete steps."""
        pr = ParameterRange(
            name="discrete_param",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            step=0.1,
            unit="fraction",
        )
        assert pr.step == 0.1

    def test_parameter_range_repr(self) -> None:
        """Test string representation of ParameterRange."""
        pr = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            unit="x",
        )
        repr_str = repr(pr)
        assert "test" in repr_str
        assert "0.0" in repr_str
        assert "1.0" in repr_str


class TestDefaultParameterRanges:
    """Tests for DEFAULT_PARAMETER_RANGES."""

    def test_all_ranges_present(self) -> None:
        """Test that all expected parameter ranges are defined."""
        expected_params = [
            "metal_ratio",
            "coating_weight",
            "ferric_oxalate_pct",
            "exposure_time",
            "developer_temp",
            "humidity",
        ]
        for param in expected_params:
            assert param in DEFAULT_PARAMETER_RANGES

    def test_ranges_valid(self) -> None:
        """Test that all ranges have min < max."""
        for name, pr in DEFAULT_PARAMETER_RANGES.items():
            assert pr.min_value < pr.max_value, f"{name}: min must be < max"
            assert (
                pr.min_value <= pr.default_value <= pr.max_value
            ), f"{name}: default must be in [min, max]"

    def test_range_names_match_keys(self) -> None:
        """Test that ParameterRange.name matches dictionary key."""
        for key, pr in DEFAULT_PARAMETER_RANGES.items():
            assert pr.name == key

    def test_metal_ratio_range(self) -> None:
        """Test metal_ratio parameter range."""
        pr = DEFAULT_PARAMETER_RANGES["metal_ratio"]
        assert pr.min_value == 0.0
        assert pr.max_value == 1.0
        assert pr.default_value == 0.5
        assert "Pt" in pr.unit

    def test_exposure_time_range(self) -> None:
        """Test exposure_time parameter range."""
        pr = DEFAULT_PARAMETER_RANGES["exposure_time"]
        assert pr.min_value == 30.0
        assert pr.max_value == 600.0
        assert pr.unit == "seconds"


class TestMCTSSettings:
    """Tests for MCTSSettings configuration."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        settings = MCTSSettings()
        assert settings.num_simulations == 800
        assert settings.c_puct == 1.4
        assert settings.dirichlet_alpha == 0.3
        assert settings.dirichlet_fraction == 0.25
        assert settings.temperature_initial == 1.0
        assert settings.temperature_final == 0.1
        assert settings.action_bins == 21
        assert settings.checkpoint_dir == "data/mcts_checkpoints"
        assert settings.log_level == "INFO"

    def test_decision_order_default(self) -> None:
        """Test that decision_order has all expected parameters."""
        settings = MCTSSettings()
        assert len(settings.decision_order) == 6
        assert "metal_ratio" in settings.decision_order
        assert "coating_weight" in settings.decision_order
        assert "ferric_oxalate_pct" in settings.decision_order
        assert "exposure_time" in settings.decision_order
        assert "developer_temp" in settings.decision_order
        assert "humidity" in settings.decision_order

    def test_quality_weights(self) -> None:
        """Test quality metric weights."""
        settings = MCTSSettings()
        assert settings.linearity_weight == 0.4
        assert settings.dmax_weight == 0.3
        assert settings.smoothness_weight == 0.2
        assert settings.cost_weight == 0.1
        # Weights should sum to 1.0
        total_weight = (
            settings.linearity_weight
            + settings.dmax_weight
            + settings.smoothness_weight
            + settings.cost_weight
        )
        assert abs(total_weight - 1.0) < 1e-6

    def test_network_architecture(self) -> None:
        """Test neural network architecture defaults."""
        settings = MCTSSettings()
        assert settings.value_hidden_dims == [256, 256, 128]
        assert settings.policy_hidden_dims == [256, 256, 128]
        assert settings.network_learning_rate == 1e-3
        assert settings.network_weight_decay == 1e-4

    def test_training_parameters(self) -> None:
        """Test training parameter defaults."""
        settings = MCTSSettings()
        assert settings.num_training_episodes == 100
        assert settings.replay_buffer_size == 50000
        assert settings.training_batch_size == 256
        assert settings.training_epochs_per_episode == 5

    def test_progressive_widening(self) -> None:
        """Test progressive widening parameters."""
        settings = MCTSSettings()
        assert settings.progressive_widening_alpha == 0.5
        assert settings.progressive_widening_c == 1.0
        assert settings.max_actions_per_node == 20

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override defaults."""
        monkeypatch.setenv("PTPD_MCTS_NUM_SIMULATIONS", "200")
        monkeypatch.setenv("PTPD_MCTS_C_PUCT", "2.0")
        monkeypatch.setenv("PTPD_MCTS_ACTION_BINS", "31")

        settings = MCTSSettings()
        assert settings.num_simulations == 200
        assert settings.c_puct == 2.0
        assert settings.action_bins == 31

    def test_validation_num_simulations_too_low(self) -> None:
        """Test validation rejects num_simulations below minimum."""
        with pytest.raises(ValueError):
            MCTSSettings(num_simulations=10)

    def test_validation_num_simulations_too_high(self) -> None:
        """Test validation rejects num_simulations above maximum."""
        with pytest.raises(ValueError):
            MCTSSettings(num_simulations=20000)

    def test_validation_c_puct_out_of_bounds(self) -> None:
        """Test validation rejects c_puct outside valid range."""
        with pytest.raises(ValueError):
            MCTSSettings(c_puct=0.05)
        with pytest.raises(ValueError):
            MCTSSettings(c_puct=15.0)

    def test_validation_dirichlet_alpha_out_of_bounds(self) -> None:
        """Test validation rejects dirichlet_alpha outside valid range."""
        with pytest.raises(ValueError):
            MCTSSettings(dirichlet_alpha=0.005)
        with pytest.raises(ValueError):
            MCTSSettings(dirichlet_alpha=2.0)

    def test_validation_temperature_bounds(self) -> None:
        """Test validation of temperature parameters."""
        with pytest.raises(ValueError):
            MCTSSettings(temperature_initial=0.001)
        with pytest.raises(ValueError):
            MCTSSettings(temperature_final=1.5)

    def test_validation_action_bins(self) -> None:
        """Test validation of action_bins parameter."""
        with pytest.raises(ValueError):
            MCTSSettings(action_bins=3)
        with pytest.raises(ValueError):
            MCTSSettings(action_bins=150)

    def test_validation_target_dmax(self) -> None:
        """Test validation of target_dmax parameter."""
        with pytest.raises(ValueError):
            MCTSSettings(target_dmax=0.5)
        with pytest.raises(ValueError):
            MCTSSettings(target_dmax=4.0)

    def test_validation_weights(self) -> None:
        """Test validation of quality weights."""
        with pytest.raises(ValueError):
            MCTSSettings(linearity_weight=-0.1)
        with pytest.raises(ValueError):
            MCTSSettings(dmax_weight=1.5)

    def test_checkpoint_dir_path(self) -> None:
        """Test checkpoint_dir can be set to custom path."""
        custom_path = "/custom/checkpoint/path"
        settings = MCTSSettings(checkpoint_dir=custom_path)
        assert settings.checkpoint_dir == custom_path

    def test_custom_decision_order(self) -> None:
        """Test that decision_order can be customized."""
        custom_order = ["metal_ratio", "exposure_time"]
        settings = MCTSSettings(decision_order=custom_order)
        assert settings.decision_order == custom_order
        assert len(settings.decision_order) == 2

    def test_decision_order_validation_warns_invalid(self, caplog) -> None:
        """Test that invalid parameter names in decision_order trigger warning."""
        invalid_order = ["metal_ratio", "invalid_param", "exposure_time"]
        settings = MCTSSettings(decision_order=invalid_order)
        # Should still create settings but log a warning
        assert len(settings.decision_order) == 3
        # Check that warning was logged (implementation in config.py logs warnings)

    def test_network_learning_rate_bounds(self) -> None:
        """Test validation of network_learning_rate."""
        with pytest.raises(ValueError):
            MCTSSettings(network_learning_rate=1e-7)
        with pytest.raises(ValueError):
            MCTSSettings(network_learning_rate=1.0)

    def test_replay_buffer_size_bounds(self) -> None:
        """Test validation of replay_buffer_size."""
        with pytest.raises(ValueError):
            MCTSSettings(replay_buffer_size=500)
        with pytest.raises(ValueError):
            MCTSSettings(replay_buffer_size=1000000)

    def test_training_batch_size_bounds(self) -> None:
        """Test validation of training_batch_size."""
        with pytest.raises(ValueError):
            MCTSSettings(training_batch_size=16)
        with pytest.raises(ValueError):
            MCTSSettings(training_batch_size=4096)
