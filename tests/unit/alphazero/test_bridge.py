"""
Tests for the printing simulator bridge.

Verifies the simulator wrapper returns valid densities
and correctly calculates rewards.
"""

import numpy as np
import pytest

from ptpd_calibration.alphazero.bridge.printing_env import (
    PrintingSimulator,
    PrintState,
    check_simulator,
)
from ptpd_calibration.alphazero.config import AlphaZeroConfig


class TestPrintState:
    """Tests for PrintState dataclass."""

    def test_create_state(self):
        """Test creating a print state."""
        densities = np.linspace(0.1, 2.0, 21)
        state = PrintState(
            metal_ratio=0.5,
            contrast_active=0.0,
            contrast_amount=0.0,
            exposure_time=60.0,
            humidity=50.0,
            temperature=21.0,
            densities=densities,
        )

        assert state.metal_ratio == 0.5
        assert state.exposure_time == 60.0
        assert len(state.densities) == 21

    def test_to_vector(self):
        """Test converting state to vector."""
        densities = np.linspace(0.1, 2.0, 21)
        state = PrintState(
            metal_ratio=0.5,
            contrast_active=1.0,
            contrast_amount=2.0,
            exposure_time=60.0,
            humidity=50.0,
            temperature=21.0,
            densities=densities,
        )

        vector = state.to_vector()

        assert len(vector) == 6 + 21  # 6 params + 21 densities
        assert vector[0] == 0.5  # metal_ratio
        assert vector[1] == 1.0  # contrast_active
        assert vector[3] == 60.0  # exposure_time

    def test_from_vector(self):
        """Test creating state from vector."""
        vector = np.zeros(27, dtype=np.float32)
        vector[0] = 0.75  # metal_ratio
        vector[3] = 90.0  # exposure_time
        vector[6:] = np.linspace(0.1, 2.0, 21)

        state = PrintState.from_vector(vector, num_density_steps=21)

        assert state.metal_ratio == 0.75
        assert state.exposure_time == 90.0
        assert len(state.densities) == 21

    def test_copy(self):
        """Test state copying."""
        densities = np.linspace(0.1, 2.0, 21)
        state = PrintState(
            metal_ratio=0.5,
            contrast_active=0.0,
            contrast_amount=0.0,
            exposure_time=60.0,
            humidity=50.0,
            temperature=21.0,
            densities=densities,
        )

        copy = state.copy()

        assert copy.metal_ratio == state.metal_ratio
        assert copy is not state
        assert copy.densities is not state.densities

    def test_is_valid(self):
        """Test state validation."""
        densities = np.linspace(0.1, 2.0, 21)

        # Valid state
        state = PrintState(
            metal_ratio=0.5,
            contrast_active=0.0,
            contrast_amount=0.0,
            exposure_time=60.0,
            humidity=50.0,
            temperature=21.0,
            densities=densities,
        )
        assert state.is_valid()

        # Invalid metal ratio
        invalid_state = state.copy()
        invalid_state.metal_ratio = 1.5
        assert not invalid_state.is_valid()

        # Invalid exposure
        invalid_state = state.copy()
        invalid_state.exposure_time = -10.0
        assert not invalid_state.is_valid()


class TestPrintingSimulator:
    """Tests for the PrintingSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance."""
        return PrintingSimulator()

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return AlphaZeroConfig()

    def test_create_simulator(self, simulator):
        """Test creating a simulator."""
        assert simulator is not None
        assert simulator.config is not None

    def test_get_initial_state(self, simulator):
        """Test getting initial state."""
        state = simulator.get_initial_state()

        assert isinstance(state, PrintState)
        assert state.is_valid()
        assert len(state.densities) == 21

    def test_predict_densities(self, simulator):
        """Test density prediction."""
        state = simulator.get_initial_state()
        densities = simulator.predict_densities(state)

        assert len(densities) == 21
        assert all(d >= 0 for d in densities)
        assert all(d < 5 for d in densities)  # Reasonable max

    def test_density_increases_with_exposure(self, simulator):
        """Test that density increases with exposure time."""
        state1 = simulator.get_initial_state(exposure_time=30.0)
        state2 = simulator.get_initial_state(exposure_time=120.0)

        # Higher exposure should give higher Dmax
        assert state2.densities.max() > state1.densities.max()

    def test_calculate_linearity_score(self, simulator):
        """Test linearity score calculation."""
        # Perfect linear response
        perfect = np.linspace(0.1, 2.0, 21)
        score = simulator.calculate_linearity_score(perfect)
        assert 0.9 < score <= 1.0

        # Non-linear response
        nonlinear = np.power(np.linspace(0, 1, 21), 2.5) * 2.0
        score = simulator.calculate_linearity_score(nonlinear)
        assert score < 0.9

    def test_calculate_reward(self, simulator):
        """Test reward calculation."""
        state = simulator.get_initial_state()
        reward = simulator.calculate_reward(state)

        assert 0.0 <= reward <= 1.0

    def test_invalid_state_reward(self, simulator):
        """Test reward for invalid state."""
        state = simulator.get_initial_state()
        state.metal_ratio = -1.0  # Invalid

        reward = simulator.calculate_reward(state)
        assert reward < 0  # Penalty

    def test_step(self, simulator):
        """Test taking a step."""
        state = simulator.get_initial_state()

        # Apply an action
        new_state, reward, done = simulator.step(
            state,
            action_idx=3,  # exposure increase
            action_delta=(3, 10.0),
        )

        assert isinstance(new_state, PrintState)
        assert new_state.exposure_time == state.exposure_time + 10.0
        assert not done  # Should not be terminal

    def test_check_simulator(self):
        """Run the built-in simulator check."""
        assert check_simulator()


class TestSimulatorPhysics:
    """Tests for the physics model in the simulator."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance."""
        return PrintingSimulator()

    def test_metal_ratio_effect(self, simulator):
        """Test that metal ratio affects tone."""
        # Pure palladium (warm)
        state_pd = simulator.get_initial_state(metal_ratio=0.0)
        # Pure platinum (cool)
        state_pt = simulator.get_initial_state(metal_ratio=1.0)

        # Platinum typically gives higher Dmax
        assert state_pt.densities.max() > state_pd.densities.max()

    def test_contrast_agent_effect(self, simulator):
        """Test that contrast agent affects gamma."""
        state = simulator.get_initial_state()

        # Without contrast
        state.contrast_active = 0.0
        densities_no_contrast = simulator.predict_densities(state)

        # With contrast
        state.contrast_active = 1.0
        state.contrast_amount = 2.0
        densities_contrast = simulator.predict_densities(state)

        # Contrast should increase the curve steepness
        # (midtone values should differ)
        mid_idx = 10
        assert densities_contrast[mid_idx] != densities_no_contrast[mid_idx]

    def test_humidity_effect(self, simulator):
        """Test that humidity affects the process."""
        state_dry = simulator.get_initial_state(humidity=30.0)
        state_humid = simulator.get_initial_state(humidity=70.0)

        # Different humidity should give different results
        assert not np.allclose(state_dry.densities, state_humid.densities)

    def test_reproducibility(self, simulator):
        """Test that predictions are deterministic."""
        state = simulator.get_initial_state(
            metal_ratio=0.5,
            exposure_time=60.0,
        )

        densities1 = simulator.predict_densities(state)
        densities2 = simulator.predict_densities(state)

        np.testing.assert_array_equal(densities1, densities2)
