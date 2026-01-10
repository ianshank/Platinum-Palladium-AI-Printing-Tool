"""
Tests for the AlphaZero configuration module.

Verifies all config dataclasses work correctly.
"""

import tempfile
from pathlib import Path

import pytest

from ptpd_calibration.alphazero.config import (
    ActionConfig,
    ActionType,
    AlphaZeroConfig,
    MCTSConfig,
    NetworkConfig,
    PhysicsConfig,
    RewardConfig,
    StateConfig,
    TrainingConfig,
    _get_action_count,
)


class TestActionType:
    """Tests for the ActionType enum."""

    def test_action_type_count(self):
        """Test we have the expected number of actions."""
        actions = list(ActionType)
        # 4 metal ratio + 3 contrast + 4 exposure + 2 humidity + 2 special = 15
        assert len(actions) == 15

    def test_action_type_cached_count(self):
        """Test the cached action count function."""
        count = _get_action_count()
        assert count == 15
        # Second call should use cache
        assert _get_action_count() == count

    def test_action_types_unique(self):
        """Test all action types have unique values."""
        values = [a.value for a in ActionType]
        assert len(values) == len(set(values))

    def test_finish_action_exists(self):
        """Test FINISH action is defined."""
        assert ActionType.FINISH is not None
        assert ActionType.FINISH.value == "finish"

    def test_no_op_action_exists(self):
        """Test NO_OP action is defined."""
        assert ActionType.NO_OP is not None
        assert ActionType.NO_OP.value == "no_op"


class TestStateConfig:
    """Tests for StateConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StateConfig()
        assert config.num_density_steps == 21
        assert config.metal_ratio_idx == 0
        assert config.exposure_time_idx == 3

    def test_base_state_dim(self):
        """Test base_state_dim property."""
        config = StateConfig()
        assert config.base_state_dim == 6

    def test_full_state_dim(self):
        """Test full_state_dim property."""
        config = StateConfig()
        assert config.full_state_dim == 27  # 6 + 21

    def test_custom_density_steps(self):
        """Test with custom density steps."""
        config = StateConfig(num_density_steps=11)
        assert config.full_state_dim == 17  # 6 + 11

    def test_initial_values(self):
        """Test initial state values are set."""
        config = StateConfig()
        assert config.initial_metal_ratio == 0.5
        assert config.initial_exposure_time == 60.0
        assert config.initial_humidity == 50.0
        assert config.initial_temperature == 21.0

    def test_initial_ranges(self):
        """Test initial value ranges."""
        config = StateConfig()
        assert config.initial_metal_ratio_min < config.initial_metal_ratio
        assert config.initial_metal_ratio < config.initial_metal_ratio_max


class TestActionConfig:
    """Tests for ActionConfig."""

    def test_default_deltas(self):
        """Test default action deltas."""
        config = ActionConfig()
        assert config.metal_ratio_delta_small == 0.05
        assert config.metal_ratio_delta_large == 0.15
        assert config.contrast_delta == 0.5
        assert config.exposure_delta_small == 5.0
        assert config.exposure_delta_large == 20.0

    def test_num_actions(self):
        """Test num_actions property."""
        config = ActionConfig()
        assert config.num_actions == 15

    def test_get_action_delta_metal_ratio(self):
        """Test get_action_delta for metal ratio actions."""
        config = ActionConfig()

        idx, delta = config.get_action_delta(ActionType.METAL_RATIO_INCREASE_SMALL)
        assert idx == 0
        assert delta == 0.05

        idx, delta = config.get_action_delta(ActionType.METAL_RATIO_DECREASE_LARGE)
        assert idx == 0
        assert delta == -0.15

    def test_get_action_delta_exposure(self):
        """Test get_action_delta for exposure actions."""
        config = ActionConfig()

        idx, delta = config.get_action_delta(ActionType.EXPOSURE_INCREASE_LARGE)
        assert idx == 3
        assert delta == 20.0

    def test_get_action_delta_finish(self):
        """Test get_action_delta for terminal actions."""
        config = ActionConfig()

        idx, delta = config.get_action_delta(ActionType.FINISH)
        assert idx == -1
        assert delta == 0.0

    def test_min_moves_before_finish(self):
        """Test min_moves_before_finish is set."""
        config = ActionConfig()
        assert config.min_moves_before_finish == 3


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_architecture(self):
        """Test default network architecture."""
        config = NetworkConfig()
        assert config.input_dim == 27
        assert config.action_dim == 16
        assert config.hidden_dims == (256, 256, 128)
        assert config.dropout_rate == 0.1

    def test_custom_architecture(self):
        """Test custom architecture."""
        config = NetworkConfig(hidden_dims=(128, 64), dropout_rate=0.2)
        assert config.hidden_dims == (128, 64)
        assert config.dropout_rate == 0.2


class TestMCTSConfig:
    """Tests for MCTSConfig."""

    def test_default_values(self):
        """Test default MCTS values."""
        config = MCTSConfig()
        assert config.c_puct == 1.4
        assert config.n_playout == 100
        assert config.temperature == 1.0

    def test_epsilon_values(self):
        """Test numerical stability epsilon values."""
        config = MCTSConfig()
        assert config.policy_epsilon == 1e-10
        assert config.softmax_epsilon == 1e-10
        assert config.temperature_epsilon == 1e-6

    def test_temperature_settings(self):
        """Test temperature configuration."""
        config = MCTSConfig()
        assert config.temperature_threshold == 10
        assert config.greedy_temperature == 0.1


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training values."""
        config = TrainingConfig()
        assert config.num_iterations == 50
        assert config.games_per_iteration == 25
        assert config.batch_size == 64

    def test_buffer_sizes(self):
        """Test buffer size configuration."""
        config = TrainingConfig()
        assert config.buffer_size > config.min_buffer_size

    def test_evaluation_settings(self):
        """Test evaluation configuration."""
        config = TrainingConfig()
        assert config.num_evaluation_games == 5
        assert config.evaluation_temperature == 0.1


class TestRewardConfig:
    """Tests for RewardConfig."""

    def test_default_weights(self):
        """Test default reward weights."""
        config = RewardConfig()
        assert config.rmse_weight == 1.0
        assert config.monotonicity_weight == 0.3
        assert config.smoothness_weight == 0.2
        assert config.density_range_weight == 0.2

    def test_penalty_values(self):
        """Test penalty configuration."""
        config = RewardConfig()
        assert config.invalid_penalty < 0

    def test_scoring_parameters(self):
        """Test scoring parameters."""
        config = RewardConfig()
        assert config.linearity_decay_scale == 0.5
        assert config.smoothness_penalty_factor == 10.0


class TestPhysicsConfig:
    """Tests for PhysicsConfig."""

    def test_default_values(self):
        """Test default physics values."""
        config = PhysicsConfig()
        assert config.target_dmin == 0.1
        assert config.target_dmax == 2.0
        assert config.base_gamma == 1.6

    def test_gamma_bounds(self):
        """Test gamma bounds are valid."""
        config = PhysicsConfig()
        assert config.gamma_min < config.gamma_max
        assert config.gamma_min <= config.base_gamma <= config.gamma_max

    def test_curve_shape(self):
        """Test curve shape parameters."""
        config = PhysicsConfig()
        assert 0 < config.toe_position < config.shoulder_position < 1

    def test_humidity_bounds(self):
        """Test humidity bounds are valid."""
        config = PhysicsConfig()
        assert config.humidity_dmax_min < config.humidity_dmax_max


class TestAlphaZeroConfig:
    """Tests for AlphaZeroConfig."""

    def test_default_creation(self):
        """Test creating default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig(
                model_dir=Path(tmpdir) / "models",
                log_dir=Path(tmpdir) / "logs",
            )
            assert config.state is not None
            assert config.action is not None
            assert config.network is not None
            assert config.physics is not None

    def test_post_init_updates_network(self):
        """Test __post_init__ updates network dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig(
                model_dir=Path(tmpdir) / "models",
                log_dir=Path(tmpdir) / "logs",
            )
            assert config.network.input_dim == config.state.full_state_dim
            assert config.network.action_dim == config.action.num_actions

    def test_post_init_creates_directories(self):
        """Test __post_init__ creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models" / "test"
            log_dir = Path(tmpdir) / "logs" / "test"
            config = AlphaZeroConfig(model_dir=model_dir, log_dir=log_dir)
            assert config.model_dir.exists()
            assert config.log_dir.exists()

    def test_from_dict_minimal(self):
        """Test from_dict with minimal params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig.from_dict({
                "model_dir": str(Path(tmpdir) / "models"),
                "log_dir": str(Path(tmpdir) / "logs"),
            })
            assert config.seed == 42  # Default

    def test_from_dict_with_custom_values(self):
        """Test from_dict with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig.from_dict({
                "model_dir": str(Path(tmpdir) / "models"),
                "log_dir": str(Path(tmpdir) / "logs"),
                "seed": 123,
                "mcts": {"c_puct": 2.0},
            })
            assert config.seed == 123
            assert config.mcts.c_puct == 2.0

    def test_to_dict_contains_expected_keys(self):
        """Test to_dict contains expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig(
                model_dir=Path(tmpdir) / "models",
                log_dir=Path(tmpdir) / "logs",
            )
            d = config.to_dict()
            assert "state" in d
            assert "action" in d
            assert "network" in d
            assert "mcts" in d
            assert "training" in d
            assert "device" in d
            assert "seed" in d

    def test_physics_config_included(self):
        """Test physics config is properly included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig(
                model_dir=Path(tmpdir) / "models",
                log_dir=Path(tmpdir) / "logs",
            )
            assert config.physics is not None
            assert isinstance(config.physics, PhysicsConfig)
