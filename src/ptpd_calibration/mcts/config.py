"""
Configuration for MCTS-based calibration optimization.

Settings control search hyperparameters, neural network architecture,
training schedule, and quality metrics for AlphaZero-style calibration.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ParameterRange(BaseModel):
    """Definition of search bounds for a single calibration parameter."""

    name: str
    min_value: float
    max_value: float
    default_value: float
    step: float | None = None
    unit: str = ""

    def __repr__(self) -> str:
        """String representation of parameter range."""
        step_str = f", step={self.step}" if self.step is not None else ""
        return (
            f"ParameterRange({self.name}: "
            f"[{self.min_value}, {self.max_value}]{step_str} {self.unit})"
        )


DEFAULT_PARAMETER_RANGES: dict[str, ParameterRange] = {
    "metal_ratio": ParameterRange(
        name="metal_ratio",
        min_value=0.0,
        max_value=1.0,
        default_value=0.5,
        unit="Pt fraction",
    ),
    "coating_weight": ParameterRange(
        name="coating_weight",
        min_value=0.5,
        max_value=3.0,
        default_value=1.5,
        unit="ml/sq-inch",
    ),
    "ferric_oxalate_pct": ParameterRange(
        name="ferric_oxalate_pct",
        min_value=15.0,
        max_value=27.0,
        default_value=20.0,
        unit="%",
    ),
    "exposure_time": ParameterRange(
        name="exposure_time",
        min_value=30.0,
        max_value=600.0,
        default_value=180.0,
        unit="seconds",
    ),
    "developer_temp": ParameterRange(
        name="developer_temp",
        min_value=20.0,
        max_value=50.0,
        default_value=25.0,
        unit="°C",
    ),
    "humidity": ParameterRange(
        name="humidity",
        min_value=30.0,
        max_value=80.0,
        default_value=50.0,
        unit="% RH",
    ),
}


class MCTSSettings(BaseSettings):
    """Settings for Monte Carlo Tree Search calibration optimization.

    All settings can be overridden via environment variables with PTPD_MCTS_ prefix.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_MCTS_")

    # Search parameters
    num_simulations: int = Field(
        default=800,
        ge=50,
        le=10000,
        description="Number of MCTS simulations per search",
    )
    c_puct: float = Field(
        default=1.4,
        ge=0.1,
        le=10.0,
        description="Exploration constant for UCT formula",
    )
    dirichlet_alpha: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="Dirichlet noise alpha for root exploration",
    )
    dirichlet_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of root prior mixed with Dirichlet noise",
    )

    # Temperature schedule
    temperature_initial: float = Field(
        default=1.0,
        ge=0.01,
        le=5.0,
        description="Initial temperature for action selection",
    )
    temperature_final: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Final temperature for greedy selection",
    )
    temperature_decay_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of steps to decay temperature from initial to final",
    )

    # Decision order and discretization
    decision_order: list[str] = Field(
        default=[
            "metal_ratio",
            "coating_weight",
            "ferric_oxalate_pct",
            "exposure_time",
            "developer_temp",
            "humidity",
        ],
        description="Order of parameter decisions in tree search",
    )
    action_bins: int = Field(
        default=21,
        ge=5,
        le=101,
        description="Number of discretization bins per dimension",
    )

    # Progressive widening
    progressive_widening_alpha: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Progressive widening exponent: k(n) = c * n^alpha",
    )
    progressive_widening_c: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Progressive widening constant multiplier",
    )
    max_actions_per_node: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum number of child actions per node",
    )

    # Quality targets
    target_dmax: float = Field(
        default=2.0,
        ge=1.0,
        le=3.5,
        description="Target maximum density for quality scoring",
    )
    target_dmin: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Target minimum density for quality scoring",
    )

    # Quality metric weights
    linearity_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for linearity in quality score",
    )
    dmax_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for Dmax target matching in quality score",
    )
    smoothness_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for curve smoothness in quality score",
    )
    cost_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for chemistry cost in quality score",
    )

    # Quality scoring parameters
    linearity_decay_rate: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Exponential decay rate for linearity scoring",
    )
    dmax_scoring_sigma: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Gaussian sigma for Dmax target scoring",
    )
    smoothness_decay_rate: float = Field(
        default=20.0,
        ge=5.0,
        le=50.0,
        description="Exponential decay rate for smoothness scoring",
    )
    cost_metal_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight of metal cost vs coating cost",
    )

    # Neural network architecture
    value_hidden_dims: list[int] = Field(
        default=[256, 256, 128],
        description="Hidden layer dimensions for value network",
    )
    policy_hidden_dims: list[int] = Field(
        default=[256, 256, 128],
        description="Hidden layer dimensions for policy network",
    )

    # Training parameters
    network_learning_rate: float = Field(
        default=1e-3,
        ge=1e-6,
        le=1e-1,
        description="Learning rate for neural network optimizer",
    )
    network_weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=1e-1,
        description="L2 regularization weight decay",
    )
    num_training_episodes: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of self-play episodes for training",
    )
    replay_buffer_size: int = Field(
        default=50000,
        ge=1000,
        le=500000,
        description="Maximum size of experience replay buffer",
    )
    training_batch_size: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="Batch size for network training",
    )
    training_epochs_per_episode: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of training epochs after each self-play episode",
    )

    # Persistence
    checkpoint_dir: str = Field(
        default="data/mcts_checkpoints",
        description="Directory for saving model checkpoints",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for MCTS module",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize settings and validate decision order."""
        super().__init__(**data)
        logger.debug(f"Initialized MCTSSettings with {self.num_simulations} simulations")
        self._validate_decision_order()

    def _validate_decision_order(self) -> None:
        """Validate that decision_order contains only valid parameter names."""
        for param in self.decision_order:
            if param not in DEFAULT_PARAMETER_RANGES:
                logger.warning(
                    f"Parameter '{param}' in decision_order not found in DEFAULT_PARAMETER_RANGES"
                )


class PhysicsConstants(BaseModel):
    """Configurable physics model parameters for calibration simulation.

    All parameters are configurable to allow tuning the physics model
    based on empirical calibration data.
    """

    # Chemistry effects - metal ratio impact on gamma
    pt_gamma_base: float = Field(
        default=1.6,
        ge=0.5,
        le=3.0,
        description="Base gamma for pure platinum (metal_ratio=1.0)",
    )
    pd_gamma_base: float = Field(
        default=2.2,
        ge=0.5,
        le=3.5,
        description="Base gamma for pure palladium (metal_ratio=0.0)",
    )

    # Ferric oxalate impact on contrast
    fo_contrast_slope: float = Field(
        default=0.03,
        ge=0.0,
        le=0.1,
        description="Contrast change per FO% point deviation from center",
    )
    fo_contrast_center: float = Field(
        default=20.0,
        ge=15.0,
        le=27.0,
        description="FO% at neutral contrast (no adjustment)",
    )

    # Exposure effects on dmax
    exposure_dmax_rate: float = Field(
        default=0.005,
        ge=0.0,
        le=0.02,
        description="Rate of dmax increase with exposure time (used in quality scoring tests)",
    )
    exposure_dmax_ceiling: float = Field(
        default=2.5,
        ge=1.5,
        le=4.0,
        description="Maximum achievable dmax (asymptotic limit)",
    )
    exposure_dmax_halflife: float = Field(
        default=120.0,
        ge=30.0,
        le=600.0,
        description="Exposure time (seconds) to reach half of max dmax gain",
    )

    # Environment effects
    humidity_uniformity_slope: float = Field(
        default=-0.005,
        ge=-0.02,
        le=0.0,
        description="Uniformity loss per %RH deviation above optimal humidity",
    )
    humidity_optimal: float = Field(
        default=50.0,
        ge=30.0,
        le=80.0,
        description="Optimal humidity for coating uniformity (%RH)",
    )
    dev_temp_rate_slope: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Development rate change per degree C above reference",
    )
    dev_temp_reference: float = Field(
        default=25.0,
        ge=15.0,
        le=35.0,
        description="Reference temperature for development rate (°C)",
    )

    # Paper/coating effects
    coating_weight_dmax_slope: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Dmax increase per ml/sq-inch coating weight",
    )
    coating_weight_dmax_ceiling: float = Field(
        default=2.8,
        ge=2.0,
        le=4.0,
        description="Maximum dmax achievable from coating weight",
    )
    paper_dmin_base: float = Field(
        default=0.08,
        ge=0.0,
        le=0.5,
        description="Base paper Dmin (before coating)",
    )

    # Shoulder and toe adjustments
    shoulder_base: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Base shoulder position (high exposure region)",
    )
    shoulder_temp_sensitivity: float = Field(
        default=0.01,
        ge=0.0,
        le=0.05,
        description="Shoulder adjustment per degree C deviation from reference",
    )
    shoulder_compression_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Shoulder compression strength multiplier",
    )
    toe_base: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Base toe position (low exposure region)",
    )
    toe_expansion_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Toe expansion strength multiplier",
    )

    # Contrast clamping
    contrast_min: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Minimum contrast value (clamped)",
    )
    contrast_max: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Maximum contrast value (clamped)",
    )
