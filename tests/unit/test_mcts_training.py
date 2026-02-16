"""
Tests for MCTS training infrastructure.

Tests ReplayBuffer and MCTSTrainer with proper PyTorch guards.
"""

from __future__ import annotations

import pytest

from ptpd_calibration.mcts.config import MCTSSettings
from ptpd_calibration.mcts.training import ReplayBuffer
from ptpd_calibration.mcts.types import TrainingExample

# PyTorch guard for tests that require it
torch = pytest.importorskip("torch", reason="PyTorch required for training tests")


# =============================================================================
# ReplayBuffer Tests
# =============================================================================


class TestReplayBuffer:
    """Test ReplayBuffer functionality."""

    def test_initialization_default_max_size(self):
        """Test ReplayBuffer initializes with default max_size from settings."""
        buffer = ReplayBuffer()

        assert buffer.size == 0
        assert buffer.max_size > 0
        assert buffer.max_size == MCTSSettings().replay_buffer_size

    def test_initialization_custom_max_size(self):
        """Test ReplayBuffer initializes with custom max_size."""
        custom_size = 500
        buffer = ReplayBuffer(max_size=custom_size)

        assert buffer.size == 0
        assert buffer.max_size == custom_size

    def test_initialization_with_settings(self):
        """Test ReplayBuffer uses settings for configuration."""
        settings = MCTSSettings(replay_buffer_size=750)
        buffer = ReplayBuffer(settings=settings)

        assert buffer.max_size == 750

    def test_initialization_max_size_overrides_settings(self):
        """Test explicit max_size parameter overrides settings."""
        settings = MCTSSettings(replay_buffer_size=500)
        buffer = ReplayBuffer(max_size=300, settings=settings)

        assert buffer.max_size == 300

    def test_add_single_example(self):
        """Test adding a single example to buffer."""
        buffer = ReplayBuffer(max_size=10)

        example = TrainingExample(
            state_features=[1.0, 2.0, 3.0],
            policy_target=[0.5, 0.5],
            value_target=0.75,
        )

        buffer.add(example)

        assert buffer.size == 1
        assert buffer.is_full is False

    def test_add_multiple_examples(self):
        """Test adding multiple examples individually."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(5):
            example = TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            buffer.add(example)

        assert buffer.size == 5

    def test_add_batch(self):
        """Test adding batch of examples."""
        buffer = ReplayBuffer(max_size=20)

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(10)
        ]

        buffer.add_batch(examples)

        assert buffer.size == 10

    def test_add_batch_empty_list(self):
        """Test adding empty batch doesn't change size."""
        buffer = ReplayBuffer(max_size=10)
        buffer.add_batch([])

        assert buffer.size == 0

    def test_sample_returns_correct_batch_size(self):
        """Test sample returns requested number of examples."""
        buffer = ReplayBuffer(max_size=20)

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(15)
        ]
        buffer.add_batch(examples)

        sampled = buffer.sample(5)

        assert len(sampled) == 5
        assert all(isinstance(ex, TrainingExample) for ex in sampled)

    def test_sample_raises_on_too_large_batch(self):
        """Test sample raises ValueError when batch_size > buffer size."""
        buffer = ReplayBuffer(max_size=20)

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(5)
        ]
        buffer.add_batch(examples)

        with pytest.raises(ValueError, match="exceeds buffer size"):
            buffer.sample(10)

    def test_sample_randomness(self):
        """Test sample returns different examples on multiple calls."""
        buffer = ReplayBuffer(max_size=50)

        # Add examples with distinct values
        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=float(i) / 100.0,
            )
            for i in range(50)
        ]
        buffer.add_batch(examples)

        # Sample twice
        sample1 = buffer.sample(10)
        sample2 = buffer.sample(10)

        # Extract value targets to compare
        values1 = [ex.value_target for ex in sample1]
        values2 = [ex.value_target for ex in sample2]

        # Very unlikely to be identical if truly random
        assert values1 != values2

    def test_is_full_when_at_capacity(self):
        """Test is_full property when buffer reaches max_size."""
        buffer = ReplayBuffer(max_size=5)

        assert buffer.is_full is False

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(5)
        ]
        buffer.add_batch(examples)

        assert buffer.is_full is True

    def test_is_full_before_capacity(self):
        """Test is_full property when buffer not full."""
        buffer = ReplayBuffer(max_size=10)

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(5)
        ]
        buffer.add_batch(examples)

        assert buffer.is_full is False

    def test_fifo_eviction_when_buffer_full(self):
        """Test FIFO eviction when adding beyond max_size."""
        buffer = ReplayBuffer(max_size=3)

        # Add 3 examples with distinct values
        for i in range(3):
            buffer.add(
                TrainingExample(
                    state_features=[float(i)],
                    policy_target=[0.5, 0.5],
                    value_target=float(i) / 10.0,
                )
            )

        assert buffer.size == 3
        assert buffer.is_full is True

        # Add one more - should evict the first
        buffer.add(
            TrainingExample(
                state_features=[99.0],
                policy_target=[0.5, 0.5],
                value_target=0.99,
            )
        )

        # Size should still be 3
        assert buffer.size == 3
        assert buffer.is_full is True

        # Sample all and check that first value (0.0) is gone
        sampled = buffer.sample(3)
        values = [ex.value_target for ex in sampled]
        assert 0.0 not in values
        assert 0.99 in values

    def test_get_statistics_empty_buffer(self):
        """Test get_statistics returns zeros for empty buffer."""
        buffer = ReplayBuffer(max_size=10)
        stats = buffer.get_statistics()

        assert stats["size"] == 0
        assert stats["mean_value"] == 0.0
        assert stats["std_value"] == 0.0
        assert stats["min_value"] == 0.0
        assert stats["max_value"] == 0.0

    def test_get_statistics_with_data(self):
        """Test get_statistics computes correct statistics."""
        buffer = ReplayBuffer(max_size=20)

        # Add examples with known value distribution
        examples = [
            TrainingExample(
                state_features=[1.0],
                policy_target=[0.5, 0.5],
                value_target=float(i) / 10.0,  # 0.0, 0.1, 0.2, 0.3, 0.4
            )
            for i in range(5)
        ]
        buffer.add_batch(examples)

        stats = buffer.get_statistics()

        assert stats["size"] == 5
        assert stats["mean_value"] == pytest.approx(0.2, abs=0.01)
        assert stats["min_value"] == 0.0
        assert stats["max_value"] == 0.4
        assert stats["std_value"] > 0.0

    def test_clear(self):
        """Test clear removes all examples."""
        buffer = ReplayBuffer(max_size=10)

        examples = [
            TrainingExample(
                state_features=[float(i)],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for i in range(5)
        ]
        buffer.add_batch(examples)

        assert buffer.size == 5

        buffer.clear()

        assert buffer.size == 0
        assert buffer.is_full is False

    def test_sample_tensors_returns_correct_shapes(self):
        """Test sample_tensors returns tensors with correct shapes."""
        buffer = ReplayBuffer(max_size=20)

        num_features = 10
        num_actions = 5

        examples = [
            TrainingExample(
                state_features=[float(i)] * num_features,
                policy_target=[0.1] * num_actions,
                value_target=0.5,
            )
            for i in range(15)
        ]
        buffer.add_batch(examples)

        batch_size = 8
        features, policies, values = buffer.sample_tensors(batch_size)

        assert features.shape == (batch_size, num_features)
        assert policies.shape == (batch_size, num_actions)
        assert values.shape == (batch_size, 1)

    def test_sample_tensors_correct_dtype(self):
        """Test sample_tensors returns float32 tensors."""
        buffer = ReplayBuffer(max_size=20)

        examples = [
            TrainingExample(
                state_features=[1.0, 2.0],
                policy_target=[0.5, 0.5],
                value_target=0.7,
            )
            for _ in range(10)
        ]
        buffer.add_batch(examples)

        features, policies, values = buffer.sample_tensors(5)

        assert features.dtype == torch.float32
        assert policies.dtype == torch.float32
        assert values.dtype == torch.float32

    def test_sample_tensors_correct_values(self):
        """Test sample_tensors preserves example values."""
        buffer = ReplayBuffer(max_size=10)

        # Add single example with known values
        example = TrainingExample(
            state_features=[1.0, 2.0, 3.0],
            policy_target=[0.3, 0.7],
            value_target=0.85,
        )
        buffer.add(example)

        features, policies, values = buffer.sample_tensors(1)

        assert torch.allclose(features[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(policies[0], torch.tensor([0.3, 0.7]))
        assert torch.allclose(values[0], torch.tensor([0.85]))

    def test_sample_tensors_raises_on_too_large_batch(self):
        """Test sample_tensors raises when batch_size > buffer size."""
        buffer = ReplayBuffer(max_size=20)

        examples = [
            TrainingExample(
                state_features=[1.0],
                policy_target=[0.5, 0.5],
                value_target=0.5,
            )
            for _ in range(5)
        ]
        buffer.add_batch(examples)

        with pytest.raises(ValueError, match="exceeds buffer size"):
            buffer.sample_tensors(10)


# =============================================================================
# MCTSTrainer Tests
# =============================================================================


class TestMCTSTrainer:
    """Test MCTSTrainer functionality."""

    @pytest.fixture
    def trainer(self):
        """Create MCTSTrainer instance with small settings for fast tests."""
        settings = MCTSSettings(
            replay_buffer_size=100,
            training_batch_size=8,
            training_epochs_per_episode=2,
            num_training_episodes=5,
        )
        from ptpd_calibration.mcts.training import MCTSTrainer

        return MCTSTrainer(settings=settings)

    def test_initialization(self, trainer):
        """Test MCTSTrainer initializes correctly."""
        assert trainer.network is not None
        assert trainer.optimizer is not None
        assert trainer.replay_buffer is not None
        assert trainer._episode_count == 0
        assert trainer._best_quality == 0.0

    def test_initialization_with_custom_settings(self):
        """Test MCTSTrainer uses custom settings."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        settings = MCTSSettings(
            replay_buffer_size=500,
            training_batch_size=16,
        )

        trainer = MCTSTrainer(settings=settings)

        assert trainer.replay_buffer.max_size == 500
        assert trainer.settings.training_batch_size == 16

    def test_train_episode_increments_episode_count(self, trainer):
        """Test train_episode increments episode count."""
        initial_count = trainer._episode_count

        metrics = trainer.train_episode()

        assert trainer._episode_count == initial_count + 1
        assert metrics.episode == trainer._episode_count

    def test_train_episode_returns_metrics(self, trainer):
        """Test train_episode returns TrainingMetrics."""
        from ptpd_calibration.mcts.types import TrainingMetrics

        metrics = trainer.train_episode()

        assert isinstance(metrics, TrainingMetrics)
        assert hasattr(metrics, "episode")
        assert hasattr(metrics, "value_loss")
        assert hasattr(metrics, "policy_loss")
        assert hasattr(metrics, "total_loss")
        assert hasattr(metrics, "best_quality")
        assert hasattr(metrics, "mean_quality")

    def test_train_episode_adds_to_replay_buffer(self, trainer):
        """Test train_episode adds examples to replay buffer."""
        initial_size = trainer.replay_buffer.size

        trainer.train_episode()

        assert trainer.replay_buffer.size > initial_size

    def test_train_episode_updates_best_quality(self, trainer):
        """Test train_episode tracks best quality."""
        # Run multiple episodes
        for _ in range(3):
            metrics = trainer.train_episode()

        # Best quality should be non-zero and valid
        assert trainer._best_quality >= 0.0
        assert trainer._best_quality <= 1.0
        assert metrics.best_quality == trainer._best_quality

    def test_train_episode_metrics_non_negative_losses(self, trainer):
        """Test episode metrics have non-negative losses."""
        # Fill buffer first
        for _ in range(2):
            trainer.train_episode()

        # Now we should have training
        metrics = trainer.train_episode()

        assert metrics.value_loss >= 0.0
        assert metrics.policy_loss >= 0.0
        assert metrics.total_loss >= 0.0

    def test_train_full_loop(self, trainer):
        """Test train runs full loop and returns metrics list."""
        num_episodes = 3
        metrics_list = trainer.train(num_episodes=num_episodes)

        assert len(metrics_list) == num_episodes
        assert trainer._episode_count == num_episodes

    def test_train_uses_settings_num_episodes(self, trainer):
        """Test train uses settings when num_episodes not specified."""
        # Settings has num_training_episodes=5
        metrics_list = trainer.train()

        assert len(metrics_list) == trainer.settings.num_training_episodes

    def test_train_callback_called(self, trainer):
        """Test train calls callback for each episode."""
        callback_calls = []

        def test_callback(episode_num, metrics):
            callback_calls.append((episode_num, metrics))

        num_episodes = 3
        trainer.train(num_episodes=num_episodes, callback=test_callback)

        assert len(callback_calls) == num_episodes
        # Check episode numbers are correct
        assert [call[0] for call in callback_calls] == [0, 1, 2]

    def test_metrics_history_property(self, trainer):
        """Test metrics_history property tracks all episodes."""
        num_episodes = 3
        trainer.train(num_episodes=num_episodes)

        history = trainer.metrics_history

        assert len(history) == num_episodes
        assert all(m.episode == i + 1 for i, m in enumerate(history))

    def test_sample_random_parameters(self, trainer):
        """Test _sample_random_parameters generates valid parameters."""
        from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

        params = trainer._sample_random_parameters()

        # Should have all parameter dimensions
        assert len(params) == len(DEFAULT_PARAMETER_RANGES)

        # All values should be within ranges
        for name, value in params.items():
            param_range = DEFAULT_PARAMETER_RANGES[name]
            assert param_range.min_value <= value <= param_range.max_value

    def test_generate_episode_data_returns_examples(self, trainer):
        """Test _generate_episode_data returns list of TrainingExample."""
        examples = trainer._generate_episode_data()

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)

    def test_generate_episode_data_example_structure(self, trainer):
        """Test generated examples have correct structure."""
        examples = trainer._generate_episode_data()

        for example in examples:
            assert len(example.state_features) > 0
            assert len(example.policy_target) > 0
            assert 0.0 <= example.value_target <= 1.0

    def test_encode_state_features_returns_list(self, trainer):
        """Test _encode_state_features returns list of floats."""
        decided = {"metal_ratio": 0.5}
        depth = 1

        features = trainer._encode_state_features(decided, depth)

        assert isinstance(features, list)
        assert all(isinstance(f, float) for f in features)
        assert len(features) > 0

    def test_encode_state_features_includes_decided_params(self, trainer):
        """Test encoded features represent decided parameters."""
        decided = {"metal_ratio": 0.5, "coating_weight": 1.5}
        depth = 2

        features = trainer._encode_state_features(decided, depth)

        # Features should be non-zero for decided parameters
        assert any(f != 0.0 for f in features)

    def test_create_policy_target_returns_distribution(self, trainer):
        """Test _create_policy_target returns probability distribution."""
        dimension = "metal_ratio"
        value = 0.5

        policy = trainer._create_policy_target(dimension, value)

        assert isinstance(policy, list)
        assert len(policy) == trainer.settings.action_bins
        assert all(isinstance(p, float) for p in policy)
        # Should be normalized (sum to ~1.0)
        assert 0.99 <= sum(policy) <= 1.01

    def test_create_policy_target_peaks_near_value(self, trainer):
        """Test policy target has highest probability near the chosen value."""
        dimension = "metal_ratio"
        value = 0.75  # High end of range

        policy = trainer._create_policy_target(dimension, value)

        # Maximum should be in the upper part of the distribution
        max_idx = policy.index(max(policy))
        num_bins = len(policy)

        # For value=0.75, expect peak in upper 30% of bins
        assert max_idx >= num_bins * 0.6

    def test_save_checkpoint_creates_file(self, trainer, tmp_path):
        """Test save_checkpoint creates checkpoint file."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        trainer.train_episode()  # Run one episode first
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

    def test_load_checkpoint_restores_state(self, trainer, tmp_path):
        """Test load_checkpoint restores trainer state."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Train and save
        trainer.train(num_episodes=3)
        original_count = trainer._episode_count
        original_best = trainer._best_quality
        trainer.save_checkpoint(str(checkpoint_path))

        # Create new trainer and load
        from ptpd_calibration.mcts.training import MCTSTrainer

        new_trainer = MCTSTrainer(settings=trainer.settings)
        assert new_trainer._episode_count == 0  # Fresh trainer

        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer._episode_count == original_count
        assert new_trainer._best_quality == original_best

    def test_train_step_returns_losses(self, trainer):
        """Test _train_step returns loss tuple."""
        # Fill buffer with enough data
        for _ in range(2):
            trainer.train_episode()

        total_loss, value_loss, policy_loss = trainer._train_step()

        assert isinstance(total_loss, float)
        assert isinstance(value_loss, float)
        assert isinstance(policy_loss, float)
        assert total_loss >= 0.0
        assert value_loss >= 0.0
        assert policy_loss >= 0.0

    def test_replay_buffer_size_limited(self, trainer):
        """Test replay buffer respects max_size during training."""
        max_size = trainer.replay_buffer.max_size

        # Run many episodes to overfill
        num_episodes = 10
        trainer.train(num_episodes=num_episodes)

        # Buffer should not exceed max_size
        assert trainer.replay_buffer.size <= max_size
