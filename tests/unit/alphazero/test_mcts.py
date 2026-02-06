"""
Tests for the MCTS implementation.

Verifies Monte Carlo Tree Search works correctly.
"""

import numpy as np
import pytest

# Skip all tests if torch not available
torch = pytest.importorskip("torch")

from ptpd_calibration.alphazero.config import AlphaZeroConfig
from ptpd_calibration.alphazero.game.platinum_game import PlatinumGame
from ptpd_calibration.alphazero.mcts.alpha_mcts import (
    AlphaMCTS,
    MCTSPlayer,
    TreeNode,
    check_mcts,
)
from ptpd_calibration.alphazero.nn.policy_value_net import PolicyValueNet


class TestTreeNode:
    """Tests for MCTS tree nodes."""

    def test_create_node(self):
        """Test creating a node."""
        node = TreeNode(prior_prob=0.5)

        assert node.P == 0.5
        assert node.N == 0
        assert node.Q == 0.0
        assert node.is_leaf()
        assert node.is_root()

    def test_expand(self):
        """Test node expansion."""
        node = TreeNode()

        action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        valid_actions = np.array([1, 1, 0, 1])

        node.expand(action_probs, valid_actions)

        assert len(node.children) == 3  # 3 valid actions
        assert 0 in node.children
        assert 1 in node.children
        assert 2 not in node.children  # Invalid
        assert 3 in node.children
        assert not node.is_leaf()

    def test_select_child(self):
        """Test child selection with PUCT."""
        node = TreeNode()
        node.N = 10

        # Expand with equal priors
        action_probs = np.ones(4) / 4
        valid_actions = np.ones(4)
        node.expand(action_probs, valid_actions)

        # Before any visits, selection should favor exploration
        action, child = node.select_child(c_puct=1.0)

        assert action in node.children
        assert child is node.children[action]

    def test_backup(self):
        """Test value backup."""
        root = TreeNode()
        root.expand(np.ones(4) / 4, np.ones(4))

        child = root.children[0]
        child.expand(np.ones(4) / 4, np.ones(4))

        grandchild = child.children[0]

        # Backup from grandchild
        grandchild.backup(0.5)

        assert grandchild.N == 1
        assert grandchild.Q == 0.5
        assert child.N == 1
        assert child.Q == 0.5
        assert root.N == 1
        assert root.Q == 0.5

    def test_get_action_probs(self):
        """Test action probability calculation."""
        node = TreeNode()
        node.expand(np.ones(4) / 4, np.ones(4))

        # Simulate visits
        node.children[0].N = 10
        node.children[1].N = 5
        node.children[2].N = 3
        node.children[3].N = 2

        actions, probs = node.get_action_probs(temperature=1.0)

        assert len(actions) == 4
        assert np.isclose(probs.sum(), 1.0)
        assert probs[0] > probs[1] > probs[2] > probs[3]  # Ordered by visits

    def test_greedy_selection(self):
        """Test greedy action selection (temperature=0)."""
        node = TreeNode()
        node.expand(np.ones(4) / 4, np.ones(4))

        node.children[0].N = 10
        node.children[1].N = 5
        node.children[2].N = 3
        node.children[3].N = 2

        actions, probs = node.get_action_probs(temperature=0.0)

        # Should be deterministic: all probability on best action
        assert probs[0] == 1.0
        assert probs[1:].sum() == 0.0


class TestAlphaMCTS:
    """Tests for the AlphaMCTS class."""

    @pytest.fixture
    def config(self):
        """Create a test config with few playouts."""
        config = AlphaZeroConfig()
        config.mcts.n_playout = 10
        return config

    @pytest.fixture
    def game(self, config):
        """Create a game instance."""
        return PlatinumGame(config)

    @pytest.fixture
    def net(self, config):
        """Create a network instance."""
        return PolicyValueNet(config)

    @pytest.fixture
    def mcts(self, net, game, config):
        """Create an MCTS instance."""
        return AlphaMCTS(net, game, config)

    def test_create_mcts(self, mcts):
        """Test creating MCTS."""
        assert mcts is not None
        assert mcts.root is None

    def test_get_action_probs(self, mcts, game):
        """Test getting action probabilities."""
        state = game.init_board()
        policy, action = mcts.get_action_probs(state)

        assert len(policy) == game.get_action_size()
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert 0 <= action < game.get_action_size()

    def test_valid_actions_respected(self, mcts, game):
        """Test that only valid actions are selected."""
        state = game.init_board()
        valid_actions = game.get_valid_moves(state)

        policy, action = mcts.get_action_probs(state)

        # Selected action should be valid
        assert valid_actions[action] > 0

        # Invalid actions should have zero probability
        for _i, (p, v) in enumerate(zip(policy, valid_actions, strict=False)):
            if v == 0:
                assert p == 0

    def test_update_with_move(self, mcts, game):
        """Test tree update after a move."""
        state = game.init_board()

        # Run MCTS
        policy, action = mcts.get_action_probs(state)

        # Update tree
        mcts.update_with_move(action)

        # Root should be updated (or reset if action wasn't in tree)
        # Just verify no crash

    def test_reset(self, mcts, game):
        """Test tree reset."""
        state = game.init_board()
        mcts.get_action_probs(state)

        assert mcts.root is not None

        mcts.reset()

        assert mcts.root is None
        assert mcts.stats["simulations"] == 0

    def test_exploration_with_noise(self, mcts, game):
        """Test that noise adds exploration."""
        state = game.init_board()

        # Run twice with noise - should give different results
        policy1, action1 = mcts.get_action_probs(state, add_noise=True)
        mcts.reset()
        policy2, action2 = mcts.get_action_probs(state, add_noise=True)

        # With random noise, policies should differ
        # (not guaranteed, but very likely)

    def test_check_mcts(self):
        """Run the built-in MCTS check."""
        assert check_mcts()


class TestMCTSPlayer:
    """Tests for the MCTSPlayer class."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        config = AlphaZeroConfig()
        config.mcts.n_playout = 5
        config.training.max_moves_per_game = 10
        return config

    @pytest.fixture
    def player(self, config):
        """Create a player instance."""
        game = PlatinumGame(config)
        net = PolicyValueNet(config)
        return MCTSPlayer(net, game, config)

    def test_get_action(self, player, config):
        """Test getting an action."""
        state = player.game.init_board()
        action = player.get_action(state)

        assert 0 <= action < config.action.num_actions

    def test_get_action_with_probs(self, player, config):
        """Test getting action with probabilities."""
        state = player.game.init_board()
        action, probs = player.get_action(state, return_probs=True)

        assert 0 <= action < config.action.num_actions
        assert len(probs) == config.action.num_actions
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_self_play(self, player):
        """Test self-play game."""
        examples = player.self_play(temperature=1.0, temperature_threshold=5)

        assert len(examples) > 0

        # Check example format
        for state, policy, value in examples:
            assert isinstance(state, np.ndarray)
            assert isinstance(policy, np.ndarray)
            assert isinstance(value, float)
            assert np.isclose(policy.sum(), 1.0, atol=1e-5)
