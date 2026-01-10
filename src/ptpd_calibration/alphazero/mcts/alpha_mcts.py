"""
AlphaMCTS: Monte Carlo Tree Search with neural network guidance.

This module implements the MCTS component of AlphaZero, adapted
for the chemistry optimization domain. Uses the policy network
to guide tree exploration and the value network to evaluate
leaf nodes without full rollouts.
"""

import math
from typing import Optional

import numpy as np

from ptpd_calibration.alphazero.config import AlphaZeroConfig


class TreeNode:
    """
    Node in the MCTS tree.

    Each node represents a state in the calibration search space.
    Stores visit counts, value estimates, and prior probabilities.
    """

    def __init__(
        self,
        parent: Optional["TreeNode"] = None,
        prior_prob: float = 1.0,
        action: int = -1,
    ):
        """
        Initialize a tree node.

        Args:
            parent: Parent node (None for root)
            prior_prob: Prior probability from policy network
            action: Action that led to this node
        """
        self.parent = parent
        self.action = action

        # Prior probability from neural network
        self.P = prior_prob

        # Visit count
        self.N = 0

        # Total value (sum of backed up values)
        self.W = 0.0

        # Mean value (Q = W / N)
        self.Q = 0.0

        # Children: action -> TreeNode
        self.children: dict[int, TreeNode] = {}

        # State stored at this node (lazy evaluation)
        self._state: np.ndarray | None = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def expand(
        self,
        action_probs: np.ndarray,
        valid_actions: np.ndarray,
    ) -> None:
        """
        Expand this node with children for each valid action.

        Args:
            action_probs: Prior probabilities for each action
            valid_actions: Mask of valid actions (1 = valid)
        """
        for action in range(len(action_probs)):
            if valid_actions[action] > 0 and action not in self.children:
                self.children[action] = TreeNode(
                    parent=self,
                    prior_prob=action_probs[action],
                    action=action,
                )

    def select_child(self, c_puct: float) -> tuple[int, "TreeNode"]:
        """
        Select the best child according to PUCT formula.

        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Args:
            c_puct: Exploration constant

        Returns:
            Tuple of (action, child_node)
        """
        best_score = float("-inf")
        best_action = -1
        best_child = None

        sqrt_total = math.sqrt(self.N + 1)

        for action, child in self.children.items():
            # PUCT formula
            exploration = c_puct * child.P * sqrt_total / (1 + child.N)
            score = child.Q + exploration

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value: float) -> None:
        """
        Backup value through the tree.

        Updates visit counts and value estimates for all
        nodes on the path from this node to the root.

        Args:
            value: Value to backup (from neural network or terminal reward)
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def get_visit_counts(self) -> dict[int, int]:
        """Get visit counts for all children."""
        return {action: child.N for action, child in self.children.items()}

    def get_action_probs(
        self,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get action probabilities based on visit counts.

        Args:
            temperature: Temperature for action selection.
                temperature=1: proportional to visit counts
                temperature->0: greedy (max visit count)

        Returns:
            Tuple of (actions, probabilities)
        """
        if not self.children:
            return np.array([]), np.array([])

        actions = np.array(list(self.children.keys()))
        visits = np.array([self.children[a].N for a in actions], dtype=np.float64)

        if temperature < 1e-6:
            # Greedy selection
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature-scaled softmax
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / (visits_temp.sum() + 1e-10)

        return actions, probs


class AlphaMCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses the policy network to guide exploration and the
    value network to evaluate positions without rollouts.
    """

    def __init__(
        self,
        policy_value_net,
        game,
        config: AlphaZeroConfig | None = None,
    ):
        """
        Initialize MCTS.

        Args:
            policy_value_net: PolicyValueNet for evaluation
            game: PlatinumGame instance
            config: MCTS configuration
        """
        self.net = policy_value_net
        self.game = game
        self.config = config or AlphaZeroConfig()
        self.mcts_config = self.config.mcts

        # Tree root
        self.root: TreeNode | None = None

        # Statistics
        self.stats = {
            "simulations": 0,
            "max_depth": 0,
        }

    def get_action_probs(
        self,
        state: np.ndarray,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform MCTS simulations and return action probabilities.

        Args:
            state: Current game state
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of (full_policy, best_action)
            - full_policy: Probability for each action (action_dim,)
            - best_action: Index of selected action
        """
        # Initialize root
        self.root = TreeNode()
        self.root._state = state.copy()

        # Get initial policy and expand root
        policy, _ = self.net.predict(state)
        valid_actions = self.game.get_valid_moves(state)

        # Add Dirichlet noise for exploration during training
        if add_noise:
            noise = np.random.dirichlet(
                [self.mcts_config.dirichlet_alpha] * len(policy)
            )
            epsilon = self.mcts_config.exploration_fraction
            policy = (1 - epsilon) * policy + epsilon * noise

        # Mask invalid actions
        policy = policy * valid_actions
        policy = policy / (policy.sum() + 1e-10)

        self.root.expand(policy, valid_actions)

        # Run simulations
        for _ in range(self.mcts_config.n_playout):
            self._simulate(state.copy())

        # Get action probabilities from visit counts
        actions, probs = self.root.get_action_probs(temperature)

        # Create full policy vector
        full_policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
        for action, prob in zip(actions, probs, strict=False):
            full_policy[action] = prob

        # Select action
        if temperature < 1e-6:
            best_action = actions[np.argmax(probs)]
        else:
            best_action = np.random.choice(actions, p=probs)

        return full_policy, best_action

    def _simulate(self, state: np.ndarray) -> float:
        """
        Run a single MCTS simulation.

        Selection -> Expansion -> Evaluation -> Backup

        Args:
            state: Initial state for simulation

        Returns:
            Value from leaf evaluation
        """
        node = self.root
        depth = 0

        # Selection: traverse tree using PUCT
        while not node.is_leaf():
            action, node = node.select_child(self.mcts_config.c_puct)

            # Apply action to state
            state, _ = self.game.get_next_state(state, action)
            depth += 1

        # Update max depth statistic
        self.stats["max_depth"] = max(self.stats["max_depth"], depth)

        # Check if terminal
        if self.game.is_terminal(state):
            value = self.game.get_reward(state)
        else:
            # Expansion: get policy and value from network
            policy, value = self.net.predict(state)
            valid_actions = self.game.get_valid_moves(state)

            # Mask and normalize policy
            policy = policy * valid_actions
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # All actions invalid, use uniform over valid
                policy = valid_actions / (valid_actions.sum() + 1e-10)

            # Expand node
            node.expand(policy, valid_actions)

        # Backup
        node.backup(value)

        self.stats["simulations"] += 1

        return value

    def update_with_move(self, action: int) -> None:
        """
        Update the tree after a move is made.

        Reuses the subtree rooted at the taken action.

        Args:
            action: The action that was taken
        """
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    def reset(self) -> None:
        """Reset the tree."""
        self.root = None
        self.stats = {"simulations": 0, "max_depth": 0}


class MCTSPlayer:
    """
    Player that uses MCTS for action selection.

    Wrapper around AlphaMCTS for convenient game playing.
    """

    def __init__(
        self,
        policy_value_net,
        game,
        config: AlphaZeroConfig | None = None,
    ):
        """
        Initialize MCTS player.

        Args:
            policy_value_net: PolicyValueNet for evaluation
            game: PlatinumGame instance
            config: Configuration
        """
        self.net = policy_value_net
        self.game = game
        self.config = config or AlphaZeroConfig()
        self.mcts = AlphaMCTS(policy_value_net, game, config)

    def get_action(
        self,
        state: np.ndarray,
        temperature: float = 1.0,
        return_probs: bool = False,
    ) -> int | tuple[int, np.ndarray]:
        """
        Get action for a state using MCTS.

        Args:
            state: Current game state
            temperature: Temperature for action selection
            return_probs: Whether to return action probabilities

        Returns:
            Action index (and optionally probabilities)
        """
        policy, action = self.mcts.get_action_probs(state, temperature)

        if return_probs:
            return action, policy
        return action

    def self_play(
        self,
        temperature: float = 1.0,
        temperature_threshold: int = 10,
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Play a complete game via self-play.

        Returns training data in the form of (state, policy, value) tuples.

        Args:
            temperature: Initial temperature
            temperature_threshold: Move after which to use greedy selection

        Returns:
            List of (state, policy, value) training examples
        """
        # Reset game and MCTS
        self.game = self.game.clone()
        self.mcts.reset()

        state = self.game.init_board()
        training_data = []
        move = 0

        while not self.game.is_terminal(state):
            # Adjust temperature based on move number
            temp = temperature if move < temperature_threshold else 0.1

            # Get action and policy
            action, policy = self.get_action(state, temp, return_probs=True)

            # Store state and policy (value added later)
            training_data.append([state.copy(), policy, None])

            # Make move
            state, _ = self.game.get_next_state(state, action)
            self.mcts.update_with_move(action)
            move += 1

        # Get final reward
        reward = self.game.get_reward(state)

        # Update all training examples with final reward
        for example in training_data:
            example[2] = reward

        return [(s, p, v) for s, p, v in training_data]


def check_mcts() -> bool:
    """
    Verify MCTS works correctly.

    Returns:
        True if all checks pass
    """
    from ptpd_calibration.alphazero.game.platinum_game import PlatinumGame
    from ptpd_calibration.alphazero.nn.policy_value_net import PolicyValueNet

    config = AlphaZeroConfig()
    # Use fewer simulations for testing
    config.mcts.n_playout = 10

    game = PlatinumGame(config)
    net = PolicyValueNet(config)
    mcts = AlphaMCTS(net, game, config)

    # Get initial state
    state = game.init_board()

    # Run MCTS
    policy, action = mcts.get_action_probs(state, temperature=1.0)

    # Check policy shape
    if len(policy) != game.get_action_size():
        print(f"ERROR: Policy size {len(policy)} != {game.get_action_size()}")
        return False

    # Check policy sums to 1
    if not np.isclose(policy.sum(), 1.0, atol=1e-5):
        print(f"ERROR: Policy does not sum to 1: {policy.sum()}")
        return False

    # Check action is valid
    valid_actions = game.get_valid_moves(state)
    if valid_actions[action] == 0:
        print(f"ERROR: Selected invalid action {action}")
        return False

    print("MCTS check passed!")
    print(f"  Simulations: {mcts.stats['simulations']}")
    print(f"  Max depth: {mcts.stats['max_depth']}")
    print(f"  Selected action: {action} ({game.get_action_description(action)})")

    return True


if __name__ == "__main__":
    check_mcts()
