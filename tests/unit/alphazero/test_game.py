"""
Tests for the PlatinumGame class.

Verifies the game interface works correctly for AlphaZero integration.
"""

import numpy as np
import pytest

from ptpd_calibration.alphazero.config import ActionType, AlphaZeroConfig
from ptpd_calibration.alphazero.game.platinum_game import PlatinumGame, check_game


class TestPlatinumGame:
    """Tests for the PlatinumGame class."""

    @pytest.fixture
    def game(self):
        """Create a game instance."""
        config = AlphaZeroConfig()
        config.training.max_moves_per_game = 20
        return PlatinumGame(config)

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return AlphaZeroConfig()

    def test_create_game(self, game):
        """Test creating a game."""
        assert game is not None
        assert game.num_actions == len(ActionType)

    def test_init_board(self, game):
        """Test initializing the game board."""
        state = game.init_board()

        assert isinstance(state, np.ndarray)
        assert state.shape == (game.state_dim,)
        assert game.current_move == 0

    def test_get_board_size(self, game):
        """Test getting board size."""
        size = game.get_board_size()

        assert size == (1, game.state_dim)

    def test_get_action_size(self, game):
        """Test getting action size."""
        size = game.get_action_size()

        assert size == len(ActionType)

    def test_get_next_state(self, game):
        """Test applying an action."""
        state = game.init_board()
        initial_exposure = state[3]  # exposure_time index

        # Apply exposure increase
        action = list(ActionType).index(ActionType.EXPOSURE_INCREASE_SMALL)
        next_state, player = game.get_next_state(state, action)

        assert next_state.shape == state.shape
        assert next_state[3] > initial_exposure  # Exposure increased
        assert player == 1

    def test_get_valid_moves(self, game):
        """Test getting valid moves."""
        state = game.init_board()
        valid = game.get_valid_moves(state)

        assert len(valid) == game.num_actions
        assert all(v in (0, 1) for v in valid)
        assert valid.sum() > 0  # At least some valid moves

    def test_get_reward(self, game):
        """Test getting reward."""
        state = game.init_board()
        reward = game.get_reward(state)

        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0

    def test_is_terminal_initial(self, game):
        """Test that initial state is not terminal."""
        state = game.init_board()

        assert not game.is_terminal(state)

    def test_is_terminal_max_moves(self, game):
        """Test terminal detection at max moves."""
        state = game.init_board()

        # Make max moves
        valid = game.get_valid_moves(state)
        action = np.where(valid > 0)[0][0]

        for _ in range(game.max_moves + 1):
            if game.is_terminal(state):
                break
            state, _ = game.get_next_state(state, action)
            valid = game.get_valid_moves(state)
            if valid.sum() > 0:
                action = np.where(valid > 0)[0][0]

        assert game.is_terminal(state)

    def test_finish_action_terminates(self, game):
        """Test that FINISH action terminates the game."""
        state = game.init_board()

        # Make a few moves first
        for _ in range(5):
            valid = game.get_valid_moves(state)
            action = np.where(valid > 0)[0][0]
            state, _ = game.get_next_state(state, action)

        # Apply FINISH action
        finish_action = list(ActionType).index(ActionType.FINISH)
        state, _ = game.get_next_state(state, finish_action)

        assert game.is_terminal(state)

    def test_canonical_form(self, game):
        """Test canonical form is identity for single player."""
        state = game.init_board()
        canonical = game.get_canonical_form(state)

        np.testing.assert_array_equal(state, canonical)

    def test_get_symmetries(self, game):
        """Test symmetries (should be just identity)."""
        state = game.init_board()
        policy = np.ones(game.num_actions) / game.num_actions

        symmetries = game.get_symmetries(state, policy)

        assert len(symmetries) == 1
        np.testing.assert_array_equal(symmetries[0][0], state)
        np.testing.assert_array_equal(symmetries[0][1], policy)

    def test_string_representation(self, game):
        """Test state to string conversion."""
        state = game.init_board()
        string = game.string_representation(state)

        assert isinstance(string, str)
        assert len(string) > 0

    def test_clone(self, game):
        """Test game cloning."""
        state = game.init_board()

        # Make some moves
        for _ in range(3):
            valid = game.get_valid_moves(state)
            action = np.where(valid > 0)[0][0]
            state, _ = game.get_next_state(state, action)

        # Clone
        cloned = game.clone()

        assert cloned.current_move == game.current_move
        assert len(cloned.history) == len(game.history)
        assert cloned is not game

    def test_action_description(self, game):
        """Test action descriptions."""
        for i, _action_type in enumerate(ActionType):
            desc = game.get_action_description(i)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_state_summary(self, game):
        """Test state summary."""
        state = game.init_board()
        summary = game.get_state_summary(state)

        assert "metal_ratio" in summary
        assert "exposure_time" in summary
        assert "linearity_score" in summary
        assert "reward" in summary

    def test_play_random_game(self, game):
        """Test playing a random game."""
        reward, actions = game.play_random_game(verbose=False)

        assert isinstance(reward, float)
        assert len(actions) > 0
        assert all(isinstance(a, (int, np.integer)) for a in actions)

    def test_check_game(self):
        """Run the built-in game check."""
        assert check_game()


class TestActionSpace:
    """Tests for the action space."""

    @pytest.fixture
    def game(self):
        """Create a game instance."""
        return PlatinumGame()

    def test_all_actions_have_descriptions(self, game):
        """Test that all actions have descriptions."""
        for i in range(game.num_actions):
            desc = game.get_action_description(i)
            assert desc != f"Action {i}"  # Not a fallback

    def test_metal_ratio_bounds(self, game):
        """Test metal ratio stays in bounds."""
        state = game.init_board()

        # Try to decrease below 0
        state[0] = 0.1  # Low metal ratio
        decrease = list(ActionType).index(ActionType.METAL_RATIO_DECREASE_LARGE)
        valid = game.get_valid_moves(state)

        # Either invalid or clamped
        if valid[decrease]:
            next_state, _ = game.get_next_state(state.copy(), decrease)
            assert next_state[0] >= 0.0

    def test_exposure_minimum(self, game):
        """Test exposure can't go below minimum."""
        state = game.init_board()
        state[3] = 10.0  # Low exposure

        decrease = list(ActionType).index(ActionType.EXPOSURE_DECREASE_LARGE)

        # Take the action
        for _ in range(10):
            next_state, _ = game.get_next_state(state.copy(), decrease)
            assert next_state[3] >= 1.0  # Minimum exposure
            state = next_state
