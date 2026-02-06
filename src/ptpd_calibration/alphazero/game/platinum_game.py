"""
PlatinumGame: AlphaZero Game interface for Pt/Pd calibration.

This module implements the Game interface required by AlphaZero,
treating the calibration process as a single-player strategy game
where the goal is to achieve linear density response.
"""


import numpy as np

from ptpd_calibration.alphazero.bridge.printing_env import PrintingSimulator, PrintState
from ptpd_calibration.alphazero.config import ActionType, AlphaZeroConfig


class PlatinumGame:
    """
    AlphaZero-compatible game interface for Pt/Pd calibration.

    The "game" treats calibration as a strategy where:
    - State: Current printing parameters + predicted densities
    - Actions: Discrete adjustments to parameters
    - Reward: Based on linearity of the density response
    - Terminal: When FINISH action is taken or max moves reached

    This class implements the interface expected by AlphaZero:
    - init_board(): Get initial state
    - get_next_state(state, action): Apply action, get new state
    - get_reward(state): Calculate reward for terminal state
    - get_valid_moves(state): Get mask of valid actions
    - is_terminal(state): Check if game is over
    """

    def __init__(self, config: AlphaZeroConfig | None = None):
        """
        Initialize the PlatinumGame.

        Args:
            config: AlphaZero configuration
        """
        self.config = config or AlphaZeroConfig()
        self.simulator = PrintingSimulator(config=self.config)

        # Action configuration
        self.action_config = self.config.action
        self.num_actions = self.action_config.num_actions
        self.action_types = list(ActionType)

        # State configuration
        self.state_config = self.config.state
        self.state_dim = self.state_config.full_state_dim

        # Game tracking
        self.current_move = 0
        self.max_moves = self.config.training.max_moves_per_game
        self.history: list[tuple[PrintState, int, float]] = []

    def init_board(self) -> np.ndarray:
        """
        Get the initial game state.

        Returns:
            Initial state as numpy array
        """
        self.current_move = 0
        self.history = []

        # Random initial conditions for variety in self-play
        rng = np.random.RandomState()
        metal_ratio = rng.uniform(0.2, 0.8)
        exposure_time = rng.uniform(30.0, 120.0)

        state = self.simulator.get_initial_state(
            metal_ratio=metal_ratio,
            exposure_time=exposure_time,
        )

        return state.to_vector()

    def get_board_size(self) -> tuple[int, int]:
        """
        Get the "board size" (state dimension).

        For compatibility with AlphaZero interface.
        Returns state dim as a 1D "board".

        Returns:
            Tuple of (1, state_dim) for 1D state
        """
        return (1, self.state_dim)

    def get_action_size(self) -> int:
        """
        Get the number of possible actions.

        Returns:
            Number of discrete actions
        """
        return self.num_actions

    def get_next_state(
        self,
        state: np.ndarray,
        action: int,
        player: int = 1,
    ) -> tuple[np.ndarray, int]:
        """
        Apply an action and get the next state.

        Args:
            state: Current state vector
            action: Action index
            player: Player indicator (always 1 for single-player)

        Returns:
            Tuple of (next_state, next_player)
        """
        # Convert to PrintState
        print_state = PrintState.from_vector(
            state,
            num_density_steps=self.state_config.num_density_steps,
        )

        # Get action delta
        if 0 <= action < len(self.action_types):
            action_type = self.action_types[action]
            action_delta = self.action_config.get_action_delta(action_type)
        else:
            action_delta = (-1, 0.0)

        # Apply action through simulator
        new_state, reward, done = self.simulator.step(
            print_state,
            action,
            action_delta,
        )

        # Track history
        self.history.append((print_state, action, reward))
        self.current_move += 1

        return new_state.to_vector(), player

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Get a mask of valid actions for the current state.

        Args:
            state: Current state vector

        Returns:
            Binary mask where 1 indicates valid action
        """
        valid = np.ones(self.num_actions, dtype=np.float32)

        # Convert to PrintState
        print_state = PrintState.from_vector(
            state,
            num_density_steps=self.state_config.num_density_steps,
        )

        # Check each action for validity
        for i, action_type in enumerate(self.action_types):
            if action_type == ActionType.NO_OP:
                continue  # Always valid

            if action_type == ActionType.FINISH:
                # Only valid after at least a few moves
                if self.current_move < 3:
                    valid[i] = 0
                continue

            # Get action delta and check if result would be valid
            state_idx, delta = self.action_config.get_action_delta(action_type)

            if state_idx >= 0:
                state_vec = print_state.to_vector()
                new_value = state_vec[state_idx] + delta

                # Check bounds for each parameter
                if state_idx == 0:  # metal_ratio
                    if not (0.0 <= new_value <= 1.0):
                        valid[i] = 0
                elif state_idx == 2:  # contrast_amount
                    if new_value < 0.0:
                        valid[i] = 0
                elif state_idx == 3:  # exposure_time
                    if new_value < 1.0:
                        valid[i] = 0
                elif state_idx == 4:  # humidity
                    if not (0.0 <= new_value <= 100.0):
                        valid[i] = 0

        return valid

    def get_reward(
        self, state: np.ndarray, player: int = 1  # noqa: ARG002
    ) -> float:
        """
        Get the reward for a terminal state.

        Args:
            state: Terminal state vector
            player: Player indicator (always 1, required for interface)

        Returns:
            Reward value (0.0 to 1.0)
        """
        print_state = PrintState.from_vector(
            state,
            num_density_steps=self.state_config.num_density_steps,
        )
        return self.simulator.calculate_reward(print_state)

    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Check if the game is over.

        Game ends when:
        - FINISH action was taken
        - Maximum moves reached
        - State is invalid

        Args:
            state: Current state vector

        Returns:
            True if game is over
        """
        # Check move limit
        if self.current_move >= self.max_moves:
            return True

        # Check if last action was FINISH
        if self.history:
            _, last_action, _ = self.history[-1]
            if last_action == self.action_types.index(ActionType.FINISH):
                return True

        # Check state validity
        print_state = PrintState.from_vector(
            state,
            num_density_steps=self.state_config.num_density_steps,
        )
        return bool(not print_state.is_valid())

    def get_canonical_form(
        self, state: np.ndarray, player: int = 1  # noqa: ARG002
    ) -> np.ndarray:
        """
        Get canonical form of the state.

        For single-player game, this is just the state itself.

        Args:
            state: Current state vector
            player: Player indicator (required for interface)

        Returns:
            Canonical state (unchanged for single-player)
        """
        return state

    def get_symmetries(
        self,
        state: np.ndarray,
        policy: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Get symmetric forms of state-policy pair.

        For this domain, there are no meaningful symmetries.

        Args:
            state: State vector
            policy: Policy vector

        Returns:
            List containing just the original state-policy pair
        """
        return [(state, policy)]

    def string_representation(self, state: np.ndarray) -> str:
        """
        Get a string representation of the state for hashing.

        Args:
            state: State vector

        Returns:
            String representation
        """
        # Round to reduce floating point variations
        rounded = np.round(state, decimals=4)
        return rounded.tobytes().hex()

    def clone(self) -> "PlatinumGame":
        """
        Create a clone of the game.

        Returns:
            Cloned game instance
        """
        new_game = PlatinumGame(config=self.config)
        new_game.current_move = self.current_move
        new_game.history = self.history.copy()
        return new_game

    def get_action_description(self, action: int) -> str:
        """
        Get a human-readable description of an action.

        Args:
            action: Action index

        Returns:
            Description string
        """
        if 0 <= action < len(self.action_types):
            action_type = self.action_types[action]
            state_idx, delta = self.action_config.get_action_delta(action_type)

            descriptions = {
                ActionType.METAL_RATIO_DECREASE_LARGE: f"Decrease Pt ratio by {abs(delta):.0%}",
                ActionType.METAL_RATIO_DECREASE_SMALL: f"Decrease Pt ratio by {abs(delta):.0%}",
                ActionType.METAL_RATIO_INCREASE_SMALL: f"Increase Pt ratio by {delta:.0%}",
                ActionType.METAL_RATIO_INCREASE_LARGE: f"Increase Pt ratio by {delta:.0%}",
                ActionType.CONTRAST_DECREASE: f"Decrease contrast by {abs(delta):.1f}",
                ActionType.CONTRAST_INCREASE: f"Increase contrast by {delta:.1f}",
                ActionType.CONTRAST_TOGGLE: "Toggle contrast agent",
                ActionType.EXPOSURE_DECREASE_LARGE: f"Decrease exposure by {abs(delta):.0f}s",
                ActionType.EXPOSURE_DECREASE_SMALL: f"Decrease exposure by {abs(delta):.0f}s",
                ActionType.EXPOSURE_INCREASE_SMALL: f"Increase exposure by {delta:.0f}s",
                ActionType.EXPOSURE_INCREASE_LARGE: f"Increase exposure by {delta:.0f}s",
                ActionType.HUMIDITY_DECREASE: f"Decrease humidity by {abs(delta):.0f}%",
                ActionType.HUMIDITY_INCREASE: f"Increase humidity by {delta:.0f}%",
                ActionType.FINISH: "Finish calibration",
                ActionType.NO_OP: "No change",
            }

            return descriptions.get(action_type, f"Action {action}")

        return f"Invalid action {action}"

    def get_state_summary(self, state: np.ndarray) -> dict:
        """
        Get a summary of the current state.

        Args:
            state: State vector

        Returns:
            Dictionary with state summary
        """
        print_state = PrintState.from_vector(
            state,
            num_density_steps=self.state_config.num_density_steps,
        )

        densities = print_state.densities
        linearity = self.simulator.calculate_linearity_score(densities)
        reward = self.simulator.calculate_reward(print_state)

        return {
            "metal_ratio": print_state.metal_ratio,
            "contrast_active": print_state.contrast_active > 0.5,
            "contrast_amount": print_state.contrast_amount,
            "exposure_time": print_state.exposure_time,
            "humidity": print_state.humidity,
            "temperature": print_state.temperature,
            "dmin": float(densities.min()),
            "dmax": float(densities.max()),
            "density_range": float(densities.max() - densities.min()),
            "linearity_score": linearity,
            "reward": reward,
            "move": self.current_move,
        }

    def play_random_game(self, verbose: bool = False) -> tuple[float, list[int]]:
        """
        Play a random game for testing.

        Args:
            verbose: Whether to print game progress

        Returns:
            Tuple of (final_reward, action_history)
        """
        state = self.init_board()
        actions = []

        while not self.is_terminal(state):
            valid_moves = self.get_valid_moves(state)
            valid_indices = np.where(valid_moves > 0)[0]

            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            actions.append(action)

            if verbose:
                print(f"Move {self.current_move}: {self.get_action_description(action)}")

            state, _ = self.get_next_state(state, action)

        reward = self.get_reward(state)

        if verbose:
            summary = self.get_state_summary(state)
            print("\nFinal state:")
            print(f"  Metal ratio: {summary['metal_ratio']:.2f}")
            print(f"  Exposure: {summary['exposure_time']:.1f}s")
            print(f"  Density range: {summary['dmin']:.2f} - {summary['dmax']:.2f}")
            print(f"  Linearity: {summary['linearity_score']:.3f}")
            print(f"  Final reward: {reward:.3f}")

        return reward, actions


def check_game() -> bool:
    """
    Verify game interface works correctly.

    Returns:
        True if all checks pass
    """
    game = PlatinumGame()

    # Check initialization
    state = game.init_board()
    if state.shape[0] != game.state_dim:
        print(f"ERROR: State dim mismatch: {state.shape[0]} vs {game.state_dim}")
        return False

    # Check valid moves
    valid = game.get_valid_moves(state)
    if len(valid) != game.num_actions:
        print(f"ERROR: Valid moves length: {len(valid)} vs {game.num_actions}")
        return False

    # Check action application
    valid_indices = np.where(valid > 0)[0]
    if len(valid_indices) == 0:
        print("ERROR: No valid moves from initial state")
        return False

    action = valid_indices[0]
    next_state, _ = game.get_next_state(state, action)
    if next_state.shape != state.shape:
        print("ERROR: State shape changed after action")
        return False

    # Check terminal detection
    if game.is_terminal(state):
        print("ERROR: Initial state should not be terminal")
        return False

    # Check reward calculation
    reward = game.get_reward(state)
    if not -1.0 <= reward <= 1.0:
        print(f"ERROR: Reward out of range: {reward}")
        return False

    # Play a random game
    print("\nPlaying random game...")
    final_reward, actions = game.play_random_game(verbose=True)

    print(f"\nGame check passed! Final reward: {final_reward:.3f}")
    return True


if __name__ == "__main__":
    check_game()
