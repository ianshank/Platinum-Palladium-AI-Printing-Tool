"""
Training infrastructure for MCTS calibration neural networks.

Provides ReplayBuffer for experience storage and MCTSTrainer for
Expert Iteration training loop.

All neural network code is guarded behind TORCH_AVAILABLE.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque

import numpy as np

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings, PhysicsConstants
from ptpd_calibration.mcts.types import TrainingExample, TrainingMetrics

# PyTorch guard
try:
    import torch
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MCTS training. "
            "Install with: pip install ptpd-calibration[deep]"
        )


class ReplayBuffer:
    """Fixed-size replay buffer for training examples.

    Stores (state_features, policy_target, value_target) triples
    from MCTS self-play episodes with FIFO eviction.
    """

    def __init__(self, max_size: int | None = None, settings: MCTSSettings | None = None):
        """Initialize ReplayBuffer.

        Args:
            max_size: Maximum buffer size. If None, uses settings.
            settings: MCTS settings for buffer configuration.
        """
        self.settings = settings or MCTSSettings()
        self._max_size = max_size or self.settings.replay_buffer_size
        self._buffer: deque[TrainingExample] = deque(maxlen=self._max_size)

        logger.debug(f"ReplayBuffer initialized with max_size={self._max_size}")

    @property
    def size(self) -> int:
        """Current number of examples in buffer."""
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._buffer) >= self._max_size

    def add(self, example: TrainingExample) -> None:
        """Add a training example to the buffer.

        If buffer is full, oldest example is evicted (FIFO).

        Args:
            example: Training example to store
        """
        self._buffer.append(example)
        logger.debug(
            f"Added example to buffer (size={self.size}/{self._max_size}, "
            f"value={example.value_target:.3f})"
        )

    def add_batch(self, examples: list[TrainingExample]) -> None:
        """Add multiple training examples.

        Args:
            examples: List of training examples
        """
        for example in examples:
            self._buffer.append(example)
        logger.debug(
            f"Added {len(examples)} examples to buffer (size={self.size}/{self._max_size})"
        )

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of examples to sample

        Returns:
            List of sampled training examples

        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > self.size:
            raise ValueError(
                f"Requested batch_size={batch_size} exceeds buffer size={self.size}"
            )

        return random.sample(list(self._buffer), batch_size)

    def sample_tensors(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch and convert to tensors for training.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Tuple of (state_features, target_policies, target_values) tensors
        """
        _check_torch()

        examples = self.sample(batch_size)

        features = torch.tensor(
            [ex.state_features for ex in examples],
            dtype=torch.float32,
        )
        policies = torch.tensor(
            [ex.policy_target for ex in examples],
            dtype=torch.float32,
        )
        values = torch.tensor(
            [[ex.value_target] for ex in examples],
            dtype=torch.float32,
        )

        return features, policies, values

    def clear(self) -> None:
        """Clear all examples from the buffer."""
        self._buffer.clear()
        logger.debug("Replay buffer cleared")

    def get_statistics(self) -> dict[str, float]:
        """Get summary statistics of buffer contents.

        Returns:
            Dictionary with buffer statistics
        """
        if self.size == 0:
            return {
                "size": 0,
                "mean_value": 0.0,
                "std_value": 0.0,
                "min_value": 0.0,
                "max_value": 0.0,
            }

        values = [ex.value_target for ex in self._buffer]
        return {
            "size": float(self.size),
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
            "min_value": float(np.min(values)),
            "max_value": float(np.max(values)),
        }


class MCTSTrainer:
    """Expert Iteration trainer for MCTS neural networks.

    Runs self-play episodes using MCTS search, collects training data,
    and trains the DualNetwork on the collected experience.

    Training loop:
        1. Sample random starting conditions
        2. Run MCTS search with current networks -> trajectory
        3. Add trajectory to ReplayBuffer
        4. Sample batch, train DualNetwork
        5. Log metrics, checkpoint if improved
    """

    def __init__(
        self,
        settings: MCTSSettings | None = None,
        physics: PhysicsConstants | None = None,
    ):
        """Initialize MCTSTrainer.

        Args:
            settings: MCTS settings for training configuration.
            physics: Physics constants for simulator.
        """
        _check_torch()

        self.settings = settings or MCTSSettings()
        self.physics = physics or PhysicsConstants()

        # Lazy imports to avoid circular dependencies
        from ptpd_calibration.mcts.networks import DualNetwork

        self.network = DualNetwork(self.settings)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.settings.network_learning_rate,
            weight_decay=self.settings.network_weight_decay,
        )
        self.replay_buffer = ReplayBuffer(settings=self.settings)

        # Tracking
        self._best_quality = 0.0
        self._episode_count = 0
        self._metrics_history: list[TrainingMetrics] = []

        logger.info(
            f"MCTSTrainer initialized: "
            f"lr={self.settings.network_learning_rate}, "
            f"buffer_size={self.settings.replay_buffer_size}, "
            f"batch_size={self.settings.training_batch_size}"
        )

    def train_episode(self) -> TrainingMetrics:
        """Run one self-play episode and train on collected data.

        Returns:
            TrainingMetrics for this episode
        """
        episode_start = time.time()
        self._episode_count += 1

        # 1. Generate self-play data
        examples = self._generate_episode_data()
        self.replay_buffer.add_batch(examples)

        # 2. Train on sampled batches if buffer has enough data
        value_losses: list[float] = []
        policy_losses: list[float] = []
        total_losses: list[float] = []

        min_samples = min(self.settings.training_batch_size, self.replay_buffer.size)
        if min_samples >= self.settings.training_batch_size:
            for _ in range(self.settings.training_epochs_per_episode):
                loss_total, loss_value, loss_policy = self._train_step()
                total_losses.append(loss_total)
                value_losses.append(loss_value)
                policy_losses.append(loss_policy)

        # 3. Compute metrics
        episode_quality = max(
            (ex.value_target for ex in examples),
            default=0.0,
        )
        mean_quality = float(np.mean([ex.value_target for ex in examples])) if examples else 0.0

        if episode_quality > self._best_quality:
            self._best_quality = episode_quality
            logger.info(
                f"New best quality: {self._best_quality:.4f} "
                f"(episode {self._episode_count})"
            )

        metrics = TrainingMetrics(
            episode=self._episode_count,
            value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
            policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
            total_loss=float(np.mean(total_losses)) if total_losses else 0.0,
            best_quality=self._best_quality,
            mean_quality=mean_quality,
            episodes_completed=self._episode_count,
        )

        self._metrics_history.append(metrics)

        elapsed = time.time() - episode_start
        logger.info(
            f"Episode {self._episode_count}: "
            f"loss={metrics.total_loss:.4f}, "
            f"best_q={metrics.best_quality:.3f}, "
            f"mean_q={metrics.mean_quality:.3f}, "
            f"buffer={self.replay_buffer.size}, "
            f"time={elapsed:.1f}s"
        )

        return metrics

    def train(
        self,
        num_episodes: int | None = None,
        callback: object | None = None,
    ) -> list[TrainingMetrics]:
        """Run full training loop.

        Args:
            num_episodes: Number of episodes. If None, uses settings.
            callback: Optional callback called after each episode with
                     (episode_num, metrics) signature.

        Returns:
            List of TrainingMetrics for each episode
        """
        episodes = num_episodes or self.settings.num_training_episodes
        logger.info(f"Starting training for {episodes} episodes")

        all_metrics: list[TrainingMetrics] = []

        for i in range(episodes):
            metrics = self.train_episode()
            all_metrics.append(metrics)

            if callback is not None and callable(callback):
                callback(i, metrics)

        logger.info(
            f"Training complete: {episodes} episodes, "
            f"best_quality={self._best_quality:.4f}"
        )

        return all_metrics

    def _generate_episode_data(self) -> list[TrainingExample]:
        """Generate training data from one MCTS search episode.

        Performs a random rollout-based search and constructs
        training examples from the search tree visit distributions.

        Returns:
            List of TrainingExample from this episode
        """
        from ptpd_calibration.mcts.quality import QualityScorer
        from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator

        simulator = ExtendedProcessSimulator(
            physics=self.physics,
            settings=self.settings,
        )
        scorer = QualityScorer(settings=self.settings)

        # Generate random starting parameters
        params = self._sample_random_parameters()

        # Simulate to get quality score
        sim_result = simulator.simulate(params)
        quality = scorer.score(sim_result)

        # Create training examples for each decision level
        examples: list[TrainingExample] = []
        decision_order = self.settings.decision_order
        decided_so_far: dict[str, float] = {}

        for level, dim_name in enumerate(decision_order):
            # Encode state at this decision level
            features = self._encode_state_features(decided_so_far, level)

            # Create policy target: peak at the bin the parameter falls into
            param_value = params[dim_name]
            policy_target = self._create_policy_target(dim_name, param_value)

            example = TrainingExample(
                state_features=features,
                policy_target=policy_target,
                value_target=quality,
            )
            examples.append(example)

            # Add this parameter to the decided set for next level
            decided_so_far[dim_name] = param_value

        return examples

    def _sample_random_parameters(self) -> dict[str, float]:
        """Sample random parameters from configured ranges.

        Returns:
            Dictionary of parameter name -> sampled value
        """
        params: dict[str, float] = {}
        for name, param_range in DEFAULT_PARAMETER_RANGES.items():
            if param_range.step is not None:
                # Discrete: sample from step grid
                num_steps = int(
                    (param_range.max_value - param_range.min_value) / param_range.step
                ) + 1
                idx = random.randint(0, num_steps - 1)
                params[name] = param_range.min_value + idx * param_range.step
            else:
                # Continuous: uniform sample
                params[name] = random.uniform(
                    param_range.min_value,
                    param_range.max_value,
                )
        return params

    def _encode_state_features(
        self,
        decided_parameters: dict[str, float],
        depth: int,
    ) -> list[float]:
        """Encode state as a flat feature vector (numpy, for TrainingExample).

        Args:
            decided_parameters: Parameters decided so far
            depth: Current depth

        Returns:
            Feature list matching StateEncoder format
        """
        param_names = list(DEFAULT_PARAMETER_RANGES.keys())
        num_params = len(param_names)

        features = np.zeros(
            num_params + num_params + num_params + 1,
            dtype=np.float32,
        )

        # Continuous: normalized parameter values
        for i, name in enumerate(param_names):
            if name in decided_parameters:
                param_range = DEFAULT_PARAMETER_RANGES[name]
                raw = decided_parameters[name]
                span = param_range.max_value - param_range.min_value
                if span > 0:
                    features[i] = (raw - param_range.min_value) / span
                else:
                    features[i] = 0.5

        # Mask: 1 if decided
        mask_offset = num_params
        for i, name in enumerate(param_names):
            if name in decided_parameters:
                features[mask_offset + i] = 1.0

        # Depth: one-hot
        depth_offset = mask_offset + num_params
        clamped = min(depth, num_params)
        features[depth_offset + clamped] = 1.0

        return list(map(float, features))

    def _create_policy_target(
        self,
        dimension: str,
        value: float,
    ) -> list[float]:
        """Create soft policy target centered on the actual parameter value.

        Uses a Gaussian kernel centered on the bin corresponding to the
        chosen value, providing soft labels for training.

        Args:
            dimension: Parameter dimension name
            value: Chosen parameter value

        Returns:
            Policy distribution over action bins (sums to ~1)
        """
        num_bins = self.settings.action_bins
        param_range = DEFAULT_PARAMETER_RANGES.get(dimension)

        if param_range is None:
            # Uniform fallback
            return [1.0 / num_bins] * num_bins

        # Map value to bin position [0, 1]
        span = param_range.max_value - param_range.min_value
        if span <= 0:
            return [1.0 / num_bins] * num_bins

        normalized = (value - param_range.min_value) / span
        normalized = max(0.0, min(1.0, normalized))

        # Create Gaussian-shaped target centered on the bin
        bin_positions = np.linspace(0.0, 1.0, num_bins)
        sigma = 1.5 / num_bins  # Spread over ~3 bins
        target = np.exp(-0.5 * ((bin_positions - normalized) / sigma) ** 2)

        # Normalize to probability distribution
        total = target.sum()
        target = target / total if total > 0 else np.ones(num_bins) / num_bins

        return list(map(float, target))

    def _train_step(self) -> tuple[float, float, float]:
        """Perform one training step on a sampled batch.

        Returns:
            Tuple of (total_loss, value_loss, policy_loss)
        """
        features, policies, values = self.replay_buffer.sample_tensors(
            self.settings.training_batch_size
        )

        self.network.train()
        self.optimizer.zero_grad()

        total_loss, value_loss, policy_loss = self.network.compute_loss(
            features, values, policies,
        )

        total_loss.backward()
        self.optimizer.step()

        return (
            float(total_loss.item()),
            float(value_loss.item()),
            float(policy_loss.item()),
        )

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: File path for checkpoint
        """
        _check_torch()
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self._episode_count,
            "best_quality": self._best_quality,
            "buffer_size": self.replay_buffer.size,
            "settings": self.settings.model_dump(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (episode {self._episode_count})")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: File path for checkpoint
        """
        _check_torch()
        checkpoint = torch.load(path, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._episode_count = checkpoint.get("episode_count", 0)
        self._best_quality = checkpoint.get("best_quality", 0.0)
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(episode {self._episode_count}, best_q={self._best_quality:.4f})"
        )

    @property
    def metrics_history(self) -> list[TrainingMetrics]:
        """Get training metrics history."""
        return list(self._metrics_history)
