"""
Tests for MCTS neural network models.

Validates StateEncoder, ValueNetwork, PolicyNetwork, and DualNetwork
forward pass shapes, gradient flow, and output ranges.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available",
)


@pytest.fixture
def settings() -> MCTSSettings:
    """Default MCTS settings."""
    return MCTSSettings()


@pytest.fixture
def custom_settings() -> MCTSSettings:
    """Custom settings with smaller network for faster tests."""
    return MCTSSettings(
        value_hidden_dims=[32, 32],
        policy_hidden_dims=[32, 32],
        action_bins=11,
    )


class TestStateEncoder:
    """Tests for StateEncoder."""

    def test_output_dim_matches_expected(self, settings: MCTSSettings) -> None:
        """Output dimension should be computed correctly."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)
        num_params = len(DEFAULT_PARAMETER_RANGES)
        expected_dim = num_params + num_params + num_params + 1
        assert encoder.output_dim == expected_dim

    def test_encode_empty_state(self, settings: MCTSSettings) -> None:
        """Encoding with no decided parameters should work."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)
        features = encoder.encode_state({}, depth=0)
        assert features.shape == (encoder.output_dim,)

        # All continuous features should be 0
        num_params = len(DEFAULT_PARAMETER_RANGES)
        assert torch.all(features[:num_params] == 0.0)

        # All mask bits should be 0
        assert torch.all(features[num_params : 2 * num_params] == 0.0)

        # Depth 0 should be one-hot
        depth_offset = 2 * num_params
        assert features[depth_offset] == 1.0

    def test_encode_fully_decided_state(self, settings: MCTSSettings) -> None:
        """Encoding with all parameters decided should work."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)

        # Decide all parameters at midpoint
        params: dict[str, float] = {}
        for name, param_range in DEFAULT_PARAMETER_RANGES.items():
            params[name] = (param_range.min_value + param_range.max_value) / 2.0

        num_params = len(DEFAULT_PARAMETER_RANGES)
        features = encoder.encode_state(params, depth=num_params)

        # Continuous features should be ~0.5 (midpoint normalized)
        for i in range(num_params):
            assert abs(features[i].item() - 0.5) < 0.01

        # All mask bits should be 1
        mask_offset = num_params
        assert torch.all(features[mask_offset : mask_offset + num_params] == 1.0)

    def test_encode_partial_state(self, settings: MCTSSettings) -> None:
        """Encoding with partial parameters should have correct mask."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)
        params = {"metal_ratio": 0.5}
        features = encoder.encode_state(params, depth=1)

        param_names = list(DEFAULT_PARAMETER_RANGES.keys())
        num_params = len(param_names)
        mask_offset = num_params
        metal_idx = param_names.index("metal_ratio")

        # Only metal_ratio mask should be 1
        for i in range(num_params):
            if i == metal_idx:
                assert features[mask_offset + i] == 1.0
            else:
                assert features[mask_offset + i] == 0.0

    def test_encode_normalizes_to_01(self, settings: MCTSSettings) -> None:
        """Feature values should be normalized to [0, 1] range."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)

        # Use minimum values
        params_min = {
            name: param_range.min_value
            for name, param_range in DEFAULT_PARAMETER_RANGES.items()
        }
        features_min = encoder.encode_state(params_min, depth=0)

        # Use maximum values
        params_max = {
            name: param_range.max_value
            for name, param_range in DEFAULT_PARAMETER_RANGES.items()
        }
        features_max = encoder.encode_state(params_max, depth=0)

        num_params = len(DEFAULT_PARAMETER_RANGES)
        for i in range(num_params):
            assert features_min[i].item() == pytest.approx(0.0, abs=1e-6)
            assert features_max[i].item() == pytest.approx(1.0, abs=1e-6)

    def test_encode_batch(self, settings: MCTSSettings) -> None:
        """Batch encoding should stack correctly."""
        from ptpd_calibration.mcts.networks import StateEncoder

        encoder = StateEncoder(settings)
        states = [
            ({}, 0),
            ({"metal_ratio": 0.5}, 1),
            ({"metal_ratio": 0.3, "coating_weight": 1.5}, 2),
        ]
        batch = encoder.encode_batch(states)
        assert batch.shape == (3, encoder.output_dim)


class TestValueNetwork:
    """Tests for ValueNetwork."""

    def test_output_shape(self, custom_settings: MCTSSettings) -> None:
        """Output should be (batch, 1)."""
        from ptpd_calibration.mcts.networks import StateEncoder, ValueNetwork

        encoder = StateEncoder(custom_settings)
        value_net = ValueNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        batch_size = 8
        x = torch.randn(batch_size, encoder.output_dim)
        output = value_net(x)
        assert output.shape == (batch_size, 1)

    def test_output_range(self, custom_settings: MCTSSettings) -> None:
        """Output should be in [0, 1] (sigmoid)."""
        from ptpd_calibration.mcts.networks import StateEncoder, ValueNetwork

        encoder = StateEncoder(custom_settings)
        value_net = ValueNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        x = torch.randn(100, encoder.output_dim)
        output = value_net(x)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)

    def test_gradient_flow(self, custom_settings: MCTSSettings) -> None:
        """Gradients should flow through the network."""
        from ptpd_calibration.mcts.networks import StateEncoder, ValueNetwork

        encoder = StateEncoder(custom_settings)
        value_net = ValueNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        x = torch.randn(4, encoder.output_dim, requires_grad=True)
        output = value_net(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_output_shape(self, custom_settings: MCTSSettings) -> None:
        """Output should be (batch, num_actions)."""
        from ptpd_calibration.mcts.networks import PolicyNetwork, StateEncoder

        encoder = StateEncoder(custom_settings)
        policy_net = PolicyNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        batch_size = 8
        x = torch.randn(batch_size, encoder.output_dim)
        output = policy_net(x)
        assert output.shape == (batch_size, custom_settings.action_bins)

    def test_output_sums_to_one(self, custom_settings: MCTSSettings) -> None:
        """Policy output should sum to 1 (softmax)."""
        from ptpd_calibration.mcts.networks import PolicyNetwork, StateEncoder

        encoder = StateEncoder(custom_settings)
        policy_net = PolicyNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        x = torch.randn(16, encoder.output_dim)
        output = policy_net(x)

        row_sums = output.sum(dim=-1)
        for s in row_sums:
            assert s.item() == pytest.approx(1.0, abs=1e-5)

    def test_output_non_negative(self, custom_settings: MCTSSettings) -> None:
        """Policy probabilities should be non-negative."""
        from ptpd_calibration.mcts.networks import PolicyNetwork, StateEncoder

        encoder = StateEncoder(custom_settings)
        policy_net = PolicyNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        x = torch.randn(16, encoder.output_dim)
        output = policy_net(x)
        assert torch.all(output >= 0.0)

    def test_gradient_flow(self, custom_settings: MCTSSettings) -> None:
        """Gradients should flow through the policy network."""
        from ptpd_calibration.mcts.networks import PolicyNetwork, StateEncoder

        encoder = StateEncoder(custom_settings)
        policy_net = PolicyNetwork(
            input_dim=encoder.output_dim,
            settings=custom_settings,
        )

        x = torch.randn(4, encoder.output_dim, requires_grad=True)
        output = policy_net(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestDualNetwork:
    """Tests for DualNetwork."""

    def test_initialization(self, custom_settings: MCTSSettings) -> None:
        """DualNetwork should initialize without errors."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        assert network is not None
        assert hasattr(network, "encoder")
        assert hasattr(network, "value_head")
        assert hasattr(network, "policy_head")

    def test_forward_shapes(self, custom_settings: MCTSSettings) -> None:
        """Forward pass should return correct shapes."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        batch_size = 8
        feature_dim = network.encoder.output_dim

        x = torch.randn(batch_size, feature_dim)
        value, policy = network(x)

        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, custom_settings.action_bins)

    def test_predict_convenience(self, custom_settings: MCTSSettings) -> None:
        """Predict method should return scalar value and policy list."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        params = {"metal_ratio": 0.5, "coating_weight": 1.5}

        value, policy = network.predict(params, depth=2)

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
        assert isinstance(policy, list)
        assert len(policy) == custom_settings.action_bins
        assert abs(sum(policy) - 1.0) < 1e-5

    def test_compute_loss(self, custom_settings: MCTSSettings) -> None:
        """Loss computation should return valid tensors."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        feature_dim = network.encoder.output_dim
        batch_size = 4

        features = torch.randn(batch_size, feature_dim)
        target_values = torch.rand(batch_size, 1)
        target_policies = torch.softmax(
            torch.randn(batch_size, custom_settings.action_bins), dim=-1
        )

        total, value_loss, policy_loss = network.compute_loss(
            features, target_values, target_policies,
        )

        assert total.item() >= 0.0
        assert value_loss.item() >= 0.0
        assert policy_loss.item() >= 0.0
        assert total.requires_grad

    def test_loss_decreases_with_training(self, custom_settings: MCTSSettings) -> None:
        """Loss should decrease over a few training steps."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        feature_dim = network.encoder.output_dim
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Generate fixed training data
        batch_size = 16
        features = torch.randn(batch_size, feature_dim)
        target_values = torch.rand(batch_size, 1)
        target_policies = torch.softmax(
            torch.randn(batch_size, custom_settings.action_bins), dim=-1
        )

        # Track losses
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            total, _, _ = network.compute_loss(
                features, target_values, target_policies,
            )
            total.backward()
            optimizer.step()
            losses.append(total.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_parameter_count(self, custom_settings: MCTSSettings) -> None:
        """Network should have a reasonable number of parameters."""
        from ptpd_calibration.mcts.networks import DualNetwork

        network = DualNetwork(settings=custom_settings)
        total_params = sum(p.numel() for p in network.parameters())

        # With small hidden dims [32, 32], should be relatively small
        assert total_params > 0
        assert total_params < 1_000_000  # Sanity check


class TestBuildMLP:
    """Tests for the _build_mlp helper."""

    def test_simple_mlp(self) -> None:
        """Basic MLP should have correct structure."""
        from ptpd_calibration.mcts.networks import _build_mlp

        mlp = _build_mlp(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=1,
        )
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 1)

    def test_mlp_with_sigmoid(self) -> None:
        """MLP with sigmoid output should be in [0, 1]."""
        from ptpd_calibration.mcts.networks import _build_mlp

        mlp = _build_mlp(
            input_dim=10,
            hidden_dims=[16],
            output_dim=1,
            final_activation="sigmoid",
        )
        x = torch.randn(100, 10)
        out = mlp(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_mlp_with_tanh(self) -> None:
        """MLP with tanh output should be in [-1, 1]."""
        from ptpd_calibration.mcts.networks import _build_mlp

        mlp = _build_mlp(
            input_dim=10,
            hidden_dims=[16],
            output_dim=1,
            final_activation="tanh",
        )
        x = torch.randn(100, 10)
        out = mlp(x)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_mlp_with_dropout(self) -> None:
        """MLP with dropout should still produce output."""
        from ptpd_calibration.mcts.networks import _build_mlp

        mlp = _build_mlp(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            dropout_rate=0.5,
        )
        mlp.train()
        x = torch.randn(8, 10)
        out = mlp(x)
        assert out.shape == (8, 5)
