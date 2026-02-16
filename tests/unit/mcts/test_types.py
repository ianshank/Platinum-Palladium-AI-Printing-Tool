"""Tests for MCTS core types."""

from datetime import datetime
from uuid import UUID

import pytest

from ptpd_calibration.mcts.types import (
    CalibrationAction,
    CalibrationState,
    SearchResult,
    SimulationResult,
    TrainingExample,
    TrainingMetrics,
)


class TestCalibrationAction:
    """Tests for CalibrationAction model."""

    def test_action_creation(self) -> None:
        """Test creating a CalibrationAction."""
        action = CalibrationAction(
            dimension="metal_ratio",
            value=0.5,
            bin_index=10,
        )
        assert action.dimension == "metal_ratio"
        assert action.value == 0.5
        assert action.bin_index == 10

    def test_action_serialization_roundtrip(self) -> None:
        """Test serialization and deserialization of CalibrationAction."""
        action = CalibrationAction(
            dimension="exposure_time",
            value=180.0,
            bin_index=5,
        )
        json_data = action.model_dump()
        restored = CalibrationAction(**json_data)
        assert restored.dimension == action.dimension
        assert restored.value == action.value
        assert restored.bin_index == action.bin_index

    def test_action_repr(self) -> None:
        """Test string representation of CalibrationAction."""
        action = CalibrationAction(dimension="humidity", value=50.0, bin_index=8)
        repr_str = repr(action)
        assert "humidity" in repr_str
        assert "50" in repr_str


class TestCalibrationState:
    """Tests for CalibrationState model."""

    def test_state_creation_empty(self) -> None:
        """Test creating an empty CalibrationState."""
        state = CalibrationState()
        assert state.decided_parameters == {}
        assert state.remaining_dimensions == []
        assert state.depth == 0
        assert state.paper_type is None
        assert state.uv_source is None
        assert isinstance(state.id, UUID)

    def test_state_creation_with_data(self) -> None:
        """Test creating a CalibrationState with initial data."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["coating_weight", "exposure_time"],
            depth=1,
            paper_type="Arches Platine",
            uv_source="UV-LED",
        )
        assert state.decided_parameters == {"metal_ratio": 0.5}
        assert state.remaining_dimensions == ["coating_weight", "exposure_time"]
        assert state.depth == 1
        assert state.paper_type == "Arches Platine"
        assert state.uv_source == "UV-LED"

    def test_is_terminal_empty(self) -> None:
        """Test is_terminal on state with no remaining dimensions."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5, "exposure_time": 180.0},
            remaining_dimensions=[],
        )
        assert state.is_terminal is True

    def test_is_terminal_not_empty(self) -> None:
        """Test is_terminal on state with remaining dimensions."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["exposure_time", "humidity"],
        )
        assert state.is_terminal is False

    def test_current_dimension_none(self) -> None:
        """Test current_dimension returns None for terminal state."""
        state = CalibrationState(remaining_dimensions=[])
        assert state.current_dimension is None

    def test_current_dimension_first(self) -> None:
        """Test current_dimension returns first remaining dimension."""
        state = CalibrationState(
            remaining_dimensions=["coating_weight", "exposure_time", "humidity"]
        )
        assert state.current_dimension == "coating_weight"

    def test_apply_action_success(self) -> None:
        """Test applying an action to a state."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["coating_weight", "exposure_time"],
            depth=1,
        )
        action = CalibrationAction(
            dimension="coating_weight",
            value=1.5,
            bin_index=10,
        )
        new_state = state.apply_action(action)

        # Check new state has correct properties
        assert new_state.decided_parameters == {
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
        }
        assert new_state.remaining_dimensions == ["exposure_time"]
        assert new_state.depth == 2

        # Original state should be unchanged
        assert state.depth == 1
        assert len(state.decided_parameters) == 1

    def test_apply_action_removes_dimension(self) -> None:
        """Test that applying an action removes dimension from remaining list."""
        state = CalibrationState(
            remaining_dimensions=["metal_ratio", "coating_weight", "exposure_time"]
        )
        action = CalibrationAction(dimension="coating_weight", value=2.0, bin_index=15)
        new_state = state.apply_action(action)

        assert "coating_weight" not in new_state.remaining_dimensions
        assert "coating_weight" in new_state.decided_parameters
        assert len(new_state.remaining_dimensions) == 2

    def test_apply_action_to_terminal_state(self) -> None:
        """Test applying an action to a terminal state returns same state."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=[],
            depth=1,
        )
        action = CalibrationAction(dimension="exposure_time", value=180.0, bin_index=5)
        new_state = state.apply_action(action)

        # Should return original state since it's terminal
        assert new_state.is_terminal
        assert new_state.depth == state.depth

    def test_apply_action_invalid_dimension(self) -> None:
        """Test applying an action for dimension not in remaining list."""
        state = CalibrationState(
            remaining_dimensions=["metal_ratio", "coating_weight"]
        )
        action = CalibrationAction(dimension="exposure_time", value=180.0, bin_index=5)
        new_state = state.apply_action(action)

        # Should return original state
        assert "exposure_time" not in new_state.decided_parameters

    def test_state_preserves_metadata(self) -> None:
        """Test that paper_type and uv_source are preserved across actions."""
        state = CalibrationState(
            remaining_dimensions=["metal_ratio", "exposure_time"],
            paper_type="Arches Platine",
            uv_source="UV-LED",
        )
        action = CalibrationAction(dimension="metal_ratio", value=0.5, bin_index=10)
        new_state = state.apply_action(action)

        assert new_state.paper_type == "Arches Platine"
        assert new_state.uv_source == "UV-LED"

    def test_state_repr(self) -> None:
        """Test string representation of CalibrationState."""
        state = CalibrationState(
            decided_parameters={"metal_ratio": 0.5},
            remaining_dimensions=["exposure_time"],
            depth=1,
        )
        repr_str = repr(state)
        assert "depth=1" in repr_str
        assert "decided=1" in repr_str
        assert "remaining=1" in repr_str


class TestSimulationResult:
    """Tests for SimulationResult model."""

    def test_simulation_result_creation(self) -> None:
        """Test creating a SimulationResult."""
        result = SimulationResult(
            density_curve=[0.05, 0.5, 1.0, 1.5, 2.0],
            dmin=0.05,
            dmax=2.0,
            density_range=1.95,
            gamma=1.8,
            quality_score=0.85,
            parameters={"metal_ratio": 0.5, "exposure_time": 180.0},
            constraint_violations=[],
        )
        assert len(result.density_curve) == 5
        assert result.dmin == 0.05
        assert result.dmax == 2.0
        assert result.gamma == 1.8
        assert result.quality_score == 0.85

    def test_simulation_result_validation_quality_score(self) -> None:
        """Test validation of quality_score bounds."""
        with pytest.raises(ValueError):
            SimulationResult(
                density_curve=[0.0, 1.0],
                dmin=0.05,
                dmax=2.0,
                density_range=1.95,
                gamma=1.8,
                quality_score=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError):
            SimulationResult(
                density_curve=[0.0, 1.0],
                dmin=0.05,
                dmax=2.0,
                density_range=1.95,
                gamma=1.8,
                quality_score=-0.1,  # Invalid: < 0.0
            )

    def test_simulation_result_validation_gamma(self) -> None:
        """Test validation of gamma (must be positive)."""
        with pytest.raises(ValueError):
            SimulationResult(
                density_curve=[0.0, 1.0],
                dmin=0.05,
                dmax=2.0,
                density_range=1.95,
                gamma=0.0,  # Invalid: must be > 0
                quality_score=0.5,
            )

    def test_simulation_result_with_violations(self) -> None:
        """Test SimulationResult with constraint violations."""
        result = SimulationResult(
            density_curve=[0.0, 1.0],
            dmin=0.05,
            dmax=2.0,
            density_range=1.95,
            gamma=1.8,
            quality_score=0.5,
            constraint_violations=["Exposure time too short", "Humidity out of range"],
        )
        assert len(result.constraint_violations) == 2
        assert "Exposure time too short" in result.constraint_violations

    def test_simulation_result_repr(self) -> None:
        """Test string representation of SimulationResult."""
        result = SimulationResult(
            density_curve=[0.0, 1.0, 2.0],
            dmin=0.05,
            dmax=2.0,
            density_range=1.95,
            gamma=1.8,
            quality_score=0.85,
        )
        repr_str = repr(result)
        assert "0.85" in repr_str or "0.850" in repr_str
        assert "2.0" in repr_str or "2.00" in repr_str


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult."""
        result = SearchResult(
            best_parameters={"metal_ratio": 0.5, "exposure_time": 180.0},
            predicted_curve=[0.05, 0.5, 1.0, 1.5, 2.0],
            quality_score=0.92,
            num_simulations=800,
            search_time_seconds=45.3,
        )
        assert isinstance(result.id, UUID)
        assert isinstance(result.timestamp, datetime)
        assert result.quality_score == 0.92
        assert result.num_simulations == 800
        assert result.search_time_seconds == 45.3

    def test_search_result_with_alternatives(self) -> None:
        """Test SearchResult with alternative parameters."""
        result = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[0.0, 1.0, 2.0],
            quality_score=0.90,
            num_simulations=800,
            search_time_seconds=30.0,
            alternatives=[
                {"metal_ratio": 0.4},
                {"metal_ratio": 0.6},
            ],
        )
        assert len(result.alternatives) == 2

    def test_search_result_serialization(self) -> None:
        """Test serialization of SearchResult."""
        result = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[0.0, 1.0],
            quality_score=0.85,
            num_simulations=500,
            search_time_seconds=20.0,
            paper_type="Arches Platine",
            uv_source="UV-LED",
        )
        json_data = result.model_dump()

        assert json_data["quality_score"] == 0.85
        assert json_data["paper_type"] == "Arches Platine"
        assert json_data["uv_source"] == "UV-LED"

    def test_search_result_repr(self) -> None:
        """Test string representation of SearchResult."""
        result = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[0.0, 1.0],
            quality_score=0.88,
            num_simulations=800,
            search_time_seconds=42.5,
            alternatives=[{"metal_ratio": 0.4}, {"metal_ratio": 0.6}],
        )
        repr_str = repr(result)
        assert "0.88" in repr_str or "0.880" in repr_str
        assert "800" in repr_str
        assert "42.5" in repr_str
        assert "2 alternatives" in repr_str


class TestTrainingExample:
    """Tests for TrainingExample model."""

    def test_training_example_creation(self) -> None:
        """Test creating a TrainingExample."""
        example = TrainingExample(
            state_features=[0.5, 0.3, 0.7, 180.0, 25.0, 50.0],
            policy_target=[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            value_target=0.85,
        )
        assert len(example.state_features) == 6
        assert len(example.policy_target) == 7
        assert example.value_target == 0.85

    def test_training_example_validation(self) -> None:
        """Test validation of value_target bounds."""
        with pytest.raises(ValueError):
            TrainingExample(
                state_features=[0.5],
                policy_target=[1.0],
                value_target=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError):
            TrainingExample(
                state_features=[0.5],
                policy_target=[1.0],
                value_target=-0.1,  # Invalid: < 0.0
            )

    def test_training_example_repr(self) -> None:
        """Test string representation of TrainingExample."""
        example = TrainingExample(
            state_features=[0.5, 0.3],
            policy_target=[0.4, 0.6],
            value_target=0.75,
        )
        repr_str = repr(example)
        assert "features=2" in repr_str
        assert "policy=2" in repr_str
        assert "0.75" in repr_str or "0.750" in repr_str


class TestTrainingMetrics:
    """Tests for TrainingMetrics model."""

    def test_training_metrics_creation(self) -> None:
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            episode=10,
            value_loss=0.05,
            policy_loss=0.03,
            total_loss=0.08,
            best_quality=0.92,
            mean_quality=0.85,
            episodes_completed=10,
        )
        assert metrics.episode == 10
        assert metrics.value_loss == 0.05
        assert metrics.policy_loss == 0.03
        assert metrics.total_loss == 0.08
        assert metrics.best_quality == 0.92
        assert metrics.mean_quality == 0.85
        assert isinstance(metrics.timestamp, datetime)

    def test_training_metrics_repr(self) -> None:
        """Test string representation of TrainingMetrics."""
        metrics = TrainingMetrics(
            episode=5,
            value_loss=0.04,
            policy_loss=0.02,
            total_loss=0.06,
            best_quality=0.88,
            mean_quality=0.80,
            episodes_completed=5,
        )
        repr_str = repr(metrics)
        assert "episode=5" in repr_str
        assert "0.06" in repr_str or "0.060" in repr_str
        assert "0.88" in repr_str or "0.880" in repr_str
