"""
Core types for MCTS calibration optimization.

Defines state space, actions, simulation results, and training data
for AlphaZero-style calibration parameter search.
"""

import logging
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalibrationAction(BaseModel):
    """A single decision in the calibration search tree."""

    dimension: str
    value: float
    bin_index: int

    def __repr__(self) -> str:
        """String representation of action."""
        return f"CalibrationAction({self.dimension}={self.value:.3f}, bin={self.bin_index})"


class CalibrationState(BaseModel):
    """State of a partially-configured calibration.

    Represents a node in the search tree with decisions made so far
    and remaining dimensions to explore.
    """

    id: UUID = Field(default_factory=uuid4)
    decided_parameters: dict[str, float] = Field(default_factory=dict)
    remaining_dimensions: list[str] = Field(default_factory=list)
    depth: int = Field(default=0, ge=0)
    paper_type: str | None = None
    uv_source: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (all dimensions decided)."""
        return len(self.remaining_dimensions) == 0

    @property
    def current_dimension(self) -> str | None:
        """Get the next dimension to decide, or None if terminal."""
        return self.remaining_dimensions[0] if self.remaining_dimensions else None

    def apply_action(self, action: CalibrationAction) -> "CalibrationState":
        """Create new state by applying an action.

        Args:
            action: The action to apply.

        Returns:
            New CalibrationState with action applied.

        Raises:
            ValueError: If the state is terminal or the action dimension is invalid
                for the current state.
        """
        if self.is_terminal:
            raise ValueError("Cannot apply action to terminal CalibrationState.")

        if action.dimension not in self.remaining_dimensions:
            raise ValueError(
                f"Invalid action dimension '{action.dimension}' for current "
                f"CalibrationState. Remaining: {self.remaining_dimensions}"
            )

        new_decided = {**self.decided_parameters, action.dimension: action.value}
        new_remaining = [d for d in self.remaining_dimensions if d != action.dimension]

        new_state = CalibrationState(
            decided_parameters=new_decided,
            remaining_dimensions=new_remaining,
            depth=self.depth + 1,
            paper_type=self.paper_type,
            uv_source=self.uv_source,
        )

        logger.debug(
            f"Applied action {action.dimension}={action.value:.3f}, "
            f"depth {self.depth} -> {new_state.depth}, "
            f"remaining {len(new_remaining)} dims"
        )

        return new_state

    def __repr__(self) -> str:
        """String representation of state."""
        return (
            f"CalibrationState(depth={self.depth}, "
            f"decided={len(self.decided_parameters)}, "
            f"remaining={len(self.remaining_dimensions)})"
        )


class SimulationResult(BaseModel):
    """Result from physics simulator evaluation of a complete calibration."""

    density_curve: list[float] = Field(min_length=2)
    dmin: float = Field(ge=0.0)
    dmax: float = Field(ge=0.0)
    density_range: float = Field(ge=0.0)
    gamma: float = Field(gt=0.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, float] = Field(default_factory=dict)
    constraint_violations: list[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        """String representation of simulation result."""
        violations_str = (
            f", {len(self.constraint_violations)} violations" if self.constraint_violations else ""
        )
        return (
            f"SimulationResult(quality={self.quality_score:.3f}, "
            f"dmax={self.dmax:.2f}, gamma={self.gamma:.2f}{violations_str})"
        )


class SearchResult(BaseModel):
    """Result of an MCTS search for optimal calibration parameters."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    best_parameters: dict[str, float]
    predicted_curve: list[float]
    quality_score: float = Field(ge=0.0, le=1.0)
    visit_distribution: dict[str, list[float]] = Field(default_factory=dict)
    num_simulations: int = Field(ge=0)
    search_time_seconds: float = Field(ge=0.0)
    constraint_violations: list[str] = Field(default_factory=list)
    alternatives: list[dict[str, float]] = Field(default_factory=list)
    paper_type: str | None = None
    uv_source: str | None = None

    def __repr__(self) -> str:
        """String representation of search result."""
        return (
            f"SearchResult(quality={self.quality_score:.3f}, "
            f"sims={self.num_simulations}, "
            f"time={self.search_time_seconds:.1f}s, "
            f"{len(self.alternatives)} alternatives)"
        )


class TrainingExample(BaseModel):
    """A single training example from self-play optimization.

    Used to train the neural network from self-play experience.
    """

    state_features: list[float]
    policy_target: list[float]
    value_target: float = Field(ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation of training example."""
        return (
            f"TrainingExample(features={len(self.state_features)}, "
            f"policy={len(self.policy_target)}, "
            f"value={self.value_target:.3f})"
        )


class TrainingMetrics(BaseModel):
    """Metrics from a training session."""

    episode: int
    value_loss: float
    policy_loss: float
    total_loss: float
    best_quality: float
    mean_quality: float
    episodes_completed: int
    timestamp: datetime = Field(default_factory=datetime.now)

    def __repr__(self) -> str:
        """String representation of training metrics."""
        return (
            f"TrainingMetrics(episode={self.episode}, "
            f"loss={self.total_loss:.4f}, "
            f"best_q={self.best_quality:.3f}, "
            f"mean_q={self.mean_quality:.3f})"
        )
