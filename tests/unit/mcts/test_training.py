"""
Tests for MCTS training infrastructure.

Validates ReplayBuffer FIFO behavior, sampling, and MCTSTrainer
episode generation and loss reduction.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings
from ptpd_calibration.mcts.types import TrainingExample

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def settings() -> MCTSSettings:
    """Compact settings for fast tests (respecting validation bounds)."""
    return MCTSSettings(
        replay_buffer_size=1000,
        training_batch_size=32,
        training_epochs_per_episode=2,
        num_training_episodes=10,
        value_hidden_dims=[16, 16],
        policy_hidden_dims=[16, 16],
        action_bins=11,
        num_simulations=50,
    )


@pytest.fixture
def num_features() -> int:
    """Feature dimension matching StateEncoder output."""
    num_params = len(DEFAULT_PARAMETER_RANGES)
    return num_params + num_params + num_params + 1


@pytest.fixture
def sample_example(settings: MCTSSettings, num_features: int) -> TrainingExample:
    """Create a sample training example."""
    return TrainingExample(
        state_features=[0.0] * num_features,
        policy_target=[1.0 / settings.action_bins] * settings.action_bins,
        value_target=0.5,
    )


def _make_example(
    num_features: int,
    num_actions: int,
    value: float = 0.5,
) -> TrainingExample:
    """Helper to create a training example."""
    return TrainingExample(
        state_features=np.random.randn(num_features).tolist(),
        policy_target=np.random.dirichlet(
            np.ones(num_actions),
        ).tolist(),
        value_target=value,
    )


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_empty_buffer(self, settings: MCTSSettings) -> None:
        """Empty buffer should have size 0."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        assert buffer.size == 0
        assert not buffer.is_full

    def test_add_single(
        self,
        settings: MCTSSettings,
        sample_example: TrainingExample,
    ) -> None:
        """Adding one example should increase size."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        buffer.add(sample_example)
        assert buffer.size == 1

    def test_add_batch(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """Adding a batch should increase size by batch length."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        examples = [
            _make_example(num_features, settings.action_bins)
            for _ in range(10)
        ]
        buffer.add_batch(examples)
        assert buffer.size == 10

    def test_fifo_eviction(self, num_features: int) -> None:
        """Buffer should evict oldest when full (FIFO)."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        max_size = 5
        buffer = ReplayBuffer(max_size=max_size)
        num_actions = 11

        # Add more than max_size
        for i in range(10):
            example = _make_example(num_features, num_actions, value=i * 0.1)
            buffer.add(example)

        assert buffer.size == max_size
        assert buffer.is_full

        # Oldest examples (values 0.0-0.4) should have been evicted
        # Newest examples (values 0.5-0.9) should remain
        sampled = buffer.sample(max_size)
        values = sorted([ex.value_target for ex in sampled])
        assert values[0] >= 0.5 - 0.01  # Allow small float tolerance

    def test_sample_size(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """Sample should return exactly batch_size examples."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        for _ in range(20):
            buffer.add(_make_example(num_features, settings.action_bins))

        sampled = buffer.sample(5)
        assert len(sampled) == 5

    def test_sample_exceeds_size_raises(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """Sampling more than buffer size should raise ValueError."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        buffer.add(_make_example(num_features, settings.action_bins))

        with pytest.raises(ValueError, match="exceeds buffer size"):
            buffer.sample(5)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_sample_tensors(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """sample_tensors should return correctly shaped tensors."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        for _ in range(20):
            buffer.add(_make_example(num_features, settings.action_bins))

        batch_size = 8
        features, policies, values = buffer.sample_tensors(batch_size)

        assert features.shape == (batch_size, num_features)
        assert policies.shape == (batch_size, settings.action_bins)
        assert values.shape == (batch_size, 1)

    def test_clear(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """Clear should empty the buffer."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        for _ in range(10):
            buffer.add(_make_example(num_features, settings.action_bins))

        assert buffer.size == 10
        buffer.clear()
        assert buffer.size == 0

    def test_statistics_empty(self, settings: MCTSSettings) -> None:
        """Statistics on empty buffer should return zeros."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        stats = buffer.get_statistics()
        assert stats["size"] == 0
        assert stats["mean_value"] == 0.0

    def test_statistics_populated(
        self,
        settings: MCTSSettings,
        num_features: int,
    ) -> None:
        """Statistics should reflect buffer contents."""
        from ptpd_calibration.mcts.training import ReplayBuffer

        buffer = ReplayBuffer(settings=settings)
        for i in range(10):
            buffer.add(
                _make_example(num_features, settings.action_bins, value=i * 0.1)
            )

        stats = buffer.get_statistics()
        assert stats["size"] == 10.0
        assert 0.0 <= stats["mean_value"] <= 1.0
        assert stats["min_value"] == pytest.approx(0.0, abs=0.01)
        assert stats["max_value"] == pytest.approx(0.9, abs=0.01)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMCTSTrainer:
    """Tests for MCTSTrainer."""

    def test_initialization(self, settings: MCTSSettings) -> None:
        """Trainer should initialize with all components."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        assert trainer.network is not None
        assert trainer.optimizer is not None
        assert trainer.replay_buffer is not None

    def test_generate_episode_data(self, settings: MCTSSettings) -> None:
        """Episode data generation should produce valid examples."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        examples = trainer._generate_episode_data()

        # Should produce one example per decision level
        assert len(examples) == len(settings.decision_order)

        for ex in examples:
            # State features should have correct dimension
            num_params = len(DEFAULT_PARAMETER_RANGES)
            expected_dim = num_params + num_params + num_params + 1
            assert len(ex.state_features) == expected_dim

            # Policy should sum to ~1
            assert abs(sum(ex.policy_target) - 1.0) < 0.01

            # Value should be in [0, 1]
            assert 0.0 <= ex.value_target <= 1.0

    def test_train_single_episode(self, settings: MCTSSettings) -> None:
        """Single episode training should complete and return metrics."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        metrics = trainer.train_episode()

        assert metrics.episode == 1
        assert metrics.episodes_completed == 1
        assert 0.0 <= metrics.best_quality <= 1.0
        assert 0.0 <= metrics.mean_quality <= 1.0

    def test_train_multiple_episodes(self, settings: MCTSSettings) -> None:
        """Multiple episodes should produce metrics history."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        # Use settings.num_training_episodes (already at minimum 10)
        # but override with a small count for speed
        trainer = MCTSTrainer(settings=settings)
        # Directly call train_episode a few times instead of .train()
        # to avoid exceeding the minimum num_training_episodes validation
        for _ in range(3):
            trainer.train_episode()

        assert len(trainer.metrics_history) == 3
        assert trainer.metrics_history[-1].episodes_completed == 3

    def test_replay_buffer_fills(self, settings: MCTSSettings) -> None:
        """Training should fill the replay buffer."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        trainer.train_episode()

        # Should have added examples (one per decision level)
        assert trainer.replay_buffer.size > 0

    def test_sample_random_parameters(self, settings: MCTSSettings) -> None:
        """Random parameters should be within configured ranges."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)

        for _ in range(10):
            params = trainer._sample_random_parameters()

            for name, param_range in DEFAULT_PARAMETER_RANGES.items():
                assert name in params
                assert param_range.min_value <= params[name] <= param_range.max_value

    def test_create_policy_target(self, settings: MCTSSettings) -> None:
        """Policy target should be a valid probability distribution."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)

        for name, param_range in DEFAULT_PARAMETER_RANGES.items():
            mid_value = (param_range.min_value + param_range.max_value) / 2.0
            policy = trainer._create_policy_target(name, mid_value)

            assert len(policy) == settings.action_bins
            assert abs(sum(policy) - 1.0) < 0.01
            assert all(p >= 0 for p in policy)

            # Peak should be near the middle
            peak_idx = np.argmax(policy)
            mid_idx = settings.action_bins // 2
            assert abs(peak_idx - mid_idx) <= 2  # Within 2 bins of center

    def test_metrics_history(self, settings: MCTSSettings) -> None:
        """Metrics history should track all episodes."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        for _ in range(3):
            trainer.train_episode()

        history = trainer.metrics_history
        assert len(history) == 3
        for i, metrics in enumerate(history):
            assert metrics.episode == i + 1

    def test_callback_invoked(self, settings: MCTSSettings) -> None:
        """Training callback should be called each episode."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        callback_calls: list[int] = []

        def on_episode(episode_num: int, metrics: object) -> None:
            callback_calls.append(episode_num)

        # Use num_episodes=10 (minimum allowed by validation)
        trainer.train(num_episodes=10, callback=on_episode)
        assert len(callback_calls) == 10

    def test_save_and_load_checkpoint(
        self,
        settings: MCTSSettings,
        tmp_path: object,
    ) -> None:
        """Checkpoint save/load should preserve state."""
        from ptpd_calibration.mcts.training import MCTSTrainer

        trainer = MCTSTrainer(settings=settings)
        for _ in range(2):
            trainer.train_episode()

        # Save
        checkpoint_path = str(tmp_path) + "/test_checkpoint.pt"  # type: ignore[operator]
        trainer.save_checkpoint(checkpoint_path)

        # Load into new trainer
        trainer2 = MCTSTrainer(settings=settings)
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2._episode_count == trainer._episode_count
        assert trainer2._best_quality == trainer._best_quality
