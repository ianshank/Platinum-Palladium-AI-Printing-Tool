"""
Configuration for AlphaZero-based calibration optimization.

Defines all hyperparameters and settings for the AlchemistZero system.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Pre-computed action count for efficiency (avoids repeated list() calls)
_ACTION_COUNT: int | None = None


def _get_action_count() -> int:
    """Get the number of actions (cached for efficiency)."""
    global _ACTION_COUNT
    if _ACTION_COUNT is None:
        _ACTION_COUNT = len(list(ActionType))
    return _ACTION_COUNT


class ActionType(str, Enum):
    """Types of discrete actions the agent can take."""

    # Metal ratio adjustments
    METAL_RATIO_DECREASE_LARGE = "metal_ratio_decrease_large"
    METAL_RATIO_DECREASE_SMALL = "metal_ratio_decrease_small"
    METAL_RATIO_INCREASE_SMALL = "metal_ratio_increase_small"
    METAL_RATIO_INCREASE_LARGE = "metal_ratio_increase_large"

    # Contrast adjustments
    CONTRAST_DECREASE = "contrast_decrease"
    CONTRAST_INCREASE = "contrast_increase"
    CONTRAST_TOGGLE = "contrast_toggle"

    # Exposure adjustments
    EXPOSURE_DECREASE_LARGE = "exposure_decrease_large"
    EXPOSURE_DECREASE_SMALL = "exposure_decrease_small"
    EXPOSURE_INCREASE_SMALL = "exposure_increase_small"
    EXPOSURE_INCREASE_LARGE = "exposure_increase_large"

    # Environmental adjustments (for completeness)
    HUMIDITY_DECREASE = "humidity_decrease"
    HUMIDITY_INCREASE = "humidity_increase"

    # Special actions
    FINISH = "finish"
    NO_OP = "no_op"


@dataclass
class StateConfig:
    """Configuration for the state vector."""

    # State dimensions
    metal_ratio_idx: int = 0
    contrast_active_idx: int = 1
    contrast_amount_idx: int = 2
    exposure_time_idx: int = 3
    humidity_idx: int = 4
    temperature_idx: int = 5

    # Observed densities (21 steps for standard step tablet)
    num_density_steps: int = 21

    # Initial state ranges
    initial_metal_ratio: float = 0.5
    initial_metal_ratio_min: float = 0.2
    initial_metal_ratio_max: float = 0.8
    initial_exposure_time: float = 60.0
    initial_exposure_min: float = 30.0
    initial_exposure_max: float = 120.0
    initial_humidity: float = 50.0
    initial_temperature: float = 21.0

    # Validation thresholds
    contrast_active_threshold: float = 0.5
    state_hash_decimals: int = 4

    @property
    def base_state_dim(self) -> int:
        """Base state dimension (parameters only)."""
        return 6

    @property
    def full_state_dim(self) -> int:
        """Full state dimension including observed densities."""
        return self.base_state_dim + self.num_density_steps


@dataclass
class ActionConfig:
    """Configuration for the action space."""

    # Action deltas
    metal_ratio_delta_small: float = 0.05
    metal_ratio_delta_large: float = 0.15
    contrast_delta: float = 0.5
    exposure_delta_small: float = 5.0  # seconds
    exposure_delta_large: float = 20.0  # seconds
    humidity_delta: float = 5.0  # percent

    # Game mechanics
    min_moves_before_finish: int = 3

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions."""
        return _get_action_count()

    def get_action_delta(self, action: ActionType) -> tuple[int, float]:
        """Get the state index and delta for an action.

        Returns:
            Tuple of (state_index, delta_value)
        """
        action_map = {
            ActionType.METAL_RATIO_DECREASE_LARGE: (0, -self.metal_ratio_delta_large),
            ActionType.METAL_RATIO_DECREASE_SMALL: (0, -self.metal_ratio_delta_small),
            ActionType.METAL_RATIO_INCREASE_SMALL: (0, self.metal_ratio_delta_small),
            ActionType.METAL_RATIO_INCREASE_LARGE: (0, self.metal_ratio_delta_large),
            ActionType.CONTRAST_DECREASE: (2, -self.contrast_delta),
            ActionType.CONTRAST_INCREASE: (2, self.contrast_delta),
            ActionType.CONTRAST_TOGGLE: (1, 1.0),  # Toggle, handled specially
            ActionType.EXPOSURE_DECREASE_LARGE: (3, -self.exposure_delta_large),
            ActionType.EXPOSURE_DECREASE_SMALL: (3, -self.exposure_delta_small),
            ActionType.EXPOSURE_INCREASE_SMALL: (3, self.exposure_delta_small),
            ActionType.EXPOSURE_INCREASE_LARGE: (3, self.exposure_delta_large),
            ActionType.HUMIDITY_DECREASE: (4, -self.humidity_delta),
            ActionType.HUMIDITY_INCREASE: (4, self.humidity_delta),
            ActionType.FINISH: (-1, 0.0),  # Terminal action
            ActionType.NO_OP: (-1, 0.0),  # No change
        }
        return action_map.get(action, (-1, 0.0))


@dataclass
class NetworkConfig:
    """Configuration for the neural network."""

    # Input/output dimensions (set dynamically)
    input_dim: int = 27  # 6 params + 21 densities
    action_dim: int = 16  # Number of actions

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    # Activation
    activation: str = "relu"


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""

    # MCTS hyperparameters
    c_puct: float = 1.4  # Exploration constant
    n_playout: int = 100  # Number of simulations per move

    # Temperature for action selection
    temperature: float = 1.0
    temperature_threshold: int = 10  # After this many moves, use greedy
    greedy_temperature: float = 0.1  # Temperature for greedy selection
    temperature_epsilon: float = 1e-6  # Threshold for greedy mode

    # Tree size limits
    max_tree_nodes: int = 10000

    # Dirichlet noise for exploration
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    # Numerical stability
    policy_epsilon: float = 1e-10  # For policy normalization
    softmax_epsilon: float = 1e-10  # For softmax stability


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training loop
    num_iterations: int = 50
    games_per_iteration: int = 25
    max_moves_per_game: int = 30

    # Neural network training
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 5

    # Replay buffer
    buffer_size: int = 10000
    min_buffer_size: int = 500

    # Checkpointing
    checkpoint_interval: int = 5

    # Early stopping
    target_reward: float = 0.95  # Stop if average reward exceeds this
    patience: int = 10  # Iterations without improvement

    # Logging
    games_log_interval: int = 5  # Log progress every N games

    # Evaluation
    num_evaluation_games: int = 5  # Games to play for final evaluation
    evaluation_temperature: float = 0.1  # Temperature during evaluation


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    # Linearity reward weights
    rmse_weight: float = 1.0
    monotonicity_weight: float = 0.3
    smoothness_weight: float = 0.2

    # Density range bonus
    density_range_target: float = 2.0  # Target Dmax - Dmin
    density_range_weight: float = 0.2

    # Penalty for invalid states
    invalid_penalty: float = -1.0

    # Reward scaling
    reward_scale: float = 1.0

    # Scoring parameters
    linearity_decay_scale: float = 0.5  # Exponential decay scale for RMSE
    smoothness_penalty_factor: float = 10.0  # Multiplier for smoothness penalty


@dataclass
class PhysicsConfig:
    """Physics-based simulator parameters.

    These values model the characteristic curve of Pt/Pd printing.
    """

    # Target density curve
    target_dmin: float = 0.1
    target_dmax: float = 2.0

    # Gamma (contrast) parameters
    base_gamma: float = 1.6
    gamma_metal_ratio_factor: float = 0.4
    contrast_gamma_factor: float = 0.1
    gamma_min: float = 1.0
    gamma_max: float = 3.0

    # Paper base density
    dmin_base: float = 0.08
    dmin_humidity_factor: float = 0.02

    # Maximum density
    dmax_base: float = 1.5
    dmax_metal_ratio_factor: float = 0.5
    exposure_saturation_time: float = 120.0  # seconds

    # Humidity effects
    humidity_center: float = 50.0
    humidity_factor: float = 0.1
    humidity_dmax_min: float = 0.9
    humidity_dmax_max: float = 1.1

    # Curve shape (shoulder/toe)
    shoulder_position: float = 0.85
    toe_position: float = 0.15
    shoulder_compression_factor: float = 0.3
    toe_expansion_factor: float = 0.2

    # Noise
    physics_noise_std: float = 0.01


@dataclass
class AlphaZeroConfig:
    """Master configuration for AlchemistZero system."""

    # Sub-configurations
    state: StateConfig = field(default_factory=StateConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Paths
    model_dir: Path = field(default_factory=lambda: Path("models/alphazero"))
    log_dir: Path = field(default_factory=lambda: Path("logs/alphazero"))

    # Device
    device: str = "cpu"  # or "cuda"

    # Random seed
    seed: int = 42

    # Logging
    log_interval: int = 10
    verbose: bool = True

    def __post_init__(self):
        """Initialize derived values."""
        # Update network dimensions based on state/action configs
        self.network.input_dim = self.state.full_state_dim
        self.network.action_dim = self.action.num_actions

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AlphaZeroConfig":
        """Create config from dictionary."""
        state = StateConfig(**config_dict.get("state", {}))
        action = ActionConfig(**config_dict.get("action", {}))
        network = NetworkConfig(**config_dict.get("network", {}))
        mcts = MCTSConfig(**config_dict.get("mcts", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        reward = RewardConfig(**config_dict.get("reward", {}))
        physics = PhysicsConfig(**config_dict.get("physics", {}))

        return cls(
            state=state,
            action=action,
            network=network,
            mcts=mcts,
            training=training,
            reward=reward,
            physics=physics,
            model_dir=Path(config_dict.get("model_dir", "models/alphazero")),
            log_dir=Path(config_dict.get("log_dir", "logs/alphazero")),
            device=config_dict.get("device", "cpu"),
            seed=config_dict.get("seed", 42),
            verbose=config_dict.get("verbose", True),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "state": {
                "num_density_steps": self.state.num_density_steps,
            },
            "action": {
                "metal_ratio_delta_small": self.action.metal_ratio_delta_small,
                "metal_ratio_delta_large": self.action.metal_ratio_delta_large,
                "contrast_delta": self.action.contrast_delta,
                "exposure_delta_small": self.action.exposure_delta_small,
                "exposure_delta_large": self.action.exposure_delta_large,
            },
            "network": {
                "hidden_dims": self.network.hidden_dims,
                "dropout_rate": self.network.dropout_rate,
            },
            "mcts": {
                "c_puct": self.mcts.c_puct,
                "n_playout": self.mcts.n_playout,
            },
            "training": {
                "num_iterations": self.training.num_iterations,
                "games_per_iteration": self.training.games_per_iteration,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
            },
            "device": self.device,
            "seed": self.seed,
        }
