"""
Tests for the PolicyValueNet neural network.

Verifies the network architecture and training functionality.
"""

import numpy as np
import pytest

# Skip all tests if torch not available
torch = pytest.importorskip("torch")

from ptpd_calibration.alphazero.config import AlphaZeroConfig
from ptpd_calibration.alphazero.nn.policy_value_net import (
    PolicyValueNet,
    PolicyValueNetwork,
    check_net_dims,
)


class TestPolicyValueNetwork:
    """Tests for the low-level network module."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return AlphaZeroConfig()

    @pytest.fixture
    def network(self, config):
        """Create a network instance."""
        return PolicyValueNetwork(config.network)

    def test_create_network(self, network):
        """Test creating a network."""
        assert network is not None

    def test_forward_shape(self, network, config):
        """Test forward pass output shapes."""
        batch_size = 4
        input_dim = config.network.input_dim

        x = torch.randn(batch_size, input_dim)
        policy_logits, value = network(x)

        assert policy_logits.shape == (batch_size, config.network.action_dim)
        assert value.shape == (batch_size, 1)

    def test_predict(self, network, config):
        """Test prediction method."""
        batch_size = 4
        input_dim = config.network.input_dim

        x = torch.randn(batch_size, input_dim)
        policy_probs, value = network.predict(x)

        # Policy should be valid probability distribution
        assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert (policy_probs >= 0).all()

        # Value should be in [-1, 1]
        assert (value >= -1).all()
        assert (value <= 1).all()

    def test_predict_with_mask(self, network, config):
        """Test prediction with action mask."""
        batch_size = 4
        input_dim = config.network.input_dim
        action_dim = config.network.action_dim

        x = torch.randn(batch_size, input_dim)

        # Mask half the actions
        mask = torch.ones(batch_size, action_dim)
        mask[:, action_dim // 2:] = 0

        policy_probs, value = network.predict(x, valid_actions=mask)

        # Masked actions should have zero probability
        assert (policy_probs[:, action_dim // 2:] == 0).all()

        # Unmasked should still sum to 1
        assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestPolicyValueNet:
    """Tests for the high-level network wrapper."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return AlphaZeroConfig()

    @pytest.fixture
    def net(self, config):
        """Create a network instance."""
        return PolicyValueNet(config)

    def test_create_net(self, net):
        """Test creating the network wrapper."""
        assert net is not None
        assert net.get_num_parameters() > 0

    def test_predict_single(self, net, config):
        """Test prediction for single state."""
        state = np.random.randn(config.state.full_state_dim).astype(np.float32)

        policy, value = net.predict(state)

        assert policy.shape == (config.action.num_actions,)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1 <= value <= 1

    def test_predict_batch(self, net, config):
        """Test prediction for batch of states."""
        batch_size = 8
        states = np.random.randn(batch_size, config.state.full_state_dim).astype(np.float32)

        policy, value = net.predict(states)

        assert policy.shape == (batch_size, config.action.num_actions)
        assert value.shape == (batch_size,)

    def test_train_step(self, net, config):
        """Test a single training step."""
        batch_size = 8
        states = np.random.randn(batch_size, config.state.full_state_dim).astype(np.float32)
        policies = np.random.dirichlet(np.ones(config.action.num_actions), batch_size).astype(np.float32)
        values = np.random.uniform(-1, 1, batch_size).astype(np.float32)

        losses = net.train_step(states, policies, values)

        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"] > 0

    def test_train_on_batch(self, net, config):
        """Test training on a replay buffer."""
        # Create fake replay buffer
        buffer = []
        for _ in range(100):
            state = np.random.randn(config.state.full_state_dim).astype(np.float32)
            policy = np.random.dirichlet(np.ones(config.action.num_actions)).astype(np.float32)
            value = np.random.uniform(-1, 1)
            buffer.append((state, policy, value))

        stats = net.train_on_batch(buffer, batch_size=16, num_epochs=2)

        assert "final_policy_loss" in stats
        assert "final_value_loss" in stats
        assert "num_epochs" in stats
        assert "num_samples" in stats

    def test_save_load(self, net, config, tmp_path):
        """Test saving and loading the model."""
        # Get initial prediction
        state = np.random.randn(config.state.full_state_dim).astype(np.float32)
        policy1, value1 = net.predict(state)

        # Save
        save_path = tmp_path / "model.pth"
        net.save(save_path)

        assert save_path.exists()

        # Create new network and load
        net2 = PolicyValueNet(config)
        net2.load(save_path)

        # Should give same predictions
        policy2, value2 = net2.predict(state)

        np.testing.assert_allclose(policy1, policy2, atol=1e-5)
        np.testing.assert_allclose(value1, value2, atol=1e-5)

    def test_check_net_dims(self):
        """Run the built-in dimension check."""
        assert check_net_dims()


class TestNetworkArchitecture:
    """Tests for network architecture details."""

    def test_residual_connection(self):
        """Test residual blocks have skip connections."""
        from ptpd_calibration.alphazero.nn.policy_value_net import ResidualBlock

        block = ResidualBlock(dim=64)
        x = torch.randn(4, 64)
        y = block(x)

        assert y.shape == x.shape
        # Output should not be zero even with zero-initialized weights
        assert not torch.allclose(y, torch.zeros_like(y))

    def test_different_hidden_dims(self):
        """Test network with different hidden dimension configurations."""
        config = AlphaZeroConfig()

        # Test with different hidden dims
        for hidden_dims in [(128, 64), (256, 256, 128), (512,)]:
            config.network.hidden_dims = hidden_dims
            net = PolicyValueNet(config)

            state = np.random.randn(config.state.full_state_dim).astype(np.float32)
            policy, value = net.predict(state)

            assert policy.shape == (config.action.num_actions,)
            assert -1 <= value <= 1
