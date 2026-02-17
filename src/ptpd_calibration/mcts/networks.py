"""
Neural network models for MCTS calibration optimization.

Provides StateEncoder, ValueNetwork, PolicyNetwork, and DualNetwork
for guiding the MCTS search with learned value and policy estimates.

All neural network code is guarded behind TORCH_AVAILABLE. When PyTorch is
unavailable, the module still imports cleanly but instantiation raises ImportError.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ptpd_calibration.mcts.config import MCTSSettings

# PyTorch guard — matches pattern from ml/deep/process_sim.py
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MCTS neural networks. "
            "Install with: pip install ptpd-calibration[deep]"
        )


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "relu",
    final_activation: str | None = None,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Build a multi-layer perceptron from dimension specifications.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Hidden layer activation ('relu', 'elu', 'tanh')
        final_activation: Output activation (None, 'tanh', 'sigmoid', 'softmax')
        dropout_rate: Dropout rate between layers (0 = no dropout)

    Returns:
        nn.Sequential module
    """
    _check_torch()

    activation_map = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }
    act_cls = activation_map.get(activation, nn.ReLU)

    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_cls())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev_dim = hidden_dim

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    # Optional final activation
    if final_activation == "tanh":
        layers.append(nn.Tanh())
    elif final_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    # 'softmax' handled externally (needs dim argument)

    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """Encodes calibration state into a fixed-size feature vector.

    Handles:
    - Continuous parameter normalization (decided parameters)
    - Decision level encoding (one-hot for current depth)
    - Missing value masking (for partially decided states)
    """

    def __init__(self, settings: MCTSSettings | None = None):
        """Initialize StateEncoder.

        Args:
            settings: MCTS settings for architecture configuration.
                     If None, uses defaults.
        """
        _check_torch()
        super().__init__()

        self.settings = settings or MCTSSettings()
        self._param_names = list(DEFAULT_PARAMETER_RANGES.keys())
        self._num_params = len(self._param_names)

        # Feature dimensions:
        # - num_params continuous values (normalized)
        # - num_params mask bits (1 = decided, 0 = undecided)
        # - num_params + 1 one-hot depth encoding (0 to num_params inclusive)
        self._continuous_dim = self._num_params
        self._mask_dim = self._num_params
        self._depth_dim = self._num_params + 1
        self.feature_dim = self._continuous_dim + self._mask_dim + self._depth_dim

        logger.debug(
            f"StateEncoder: {self._num_params} params, "
            f"feature_dim={self.feature_dim}"
        )

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self.feature_dim

    def encode_state(
        self,
        decided_parameters: dict[str, float],
        depth: int,
    ) -> torch.Tensor:
        """Encode a single calibration state as a feature vector.

        Args:
            decided_parameters: Parameters decided so far
            depth: Current depth in decision tree

        Returns:
            Feature tensor of shape (feature_dim,)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)

        # Continuous features: normalize decided parameters to [0, 1]
        for i, name in enumerate(self._param_names):
            if name in decided_parameters:
                param_range = DEFAULT_PARAMETER_RANGES[name]
                raw = decided_parameters[name]
                span = param_range.max_value - param_range.min_value
                if span > 0:
                    features[i] = (raw - param_range.min_value) / span
                else:
                    features[i] = 0.5

        # Mask features: 1 if parameter is decided
        mask_offset = self._continuous_dim
        for i, name in enumerate(self._param_names):
            if name in decided_parameters:
                features[mask_offset + i] = 1.0

        # Depth encoding: one-hot
        depth_offset = mask_offset + self._mask_dim
        clamped_depth = min(depth, self._num_params)
        features[depth_offset + clamped_depth] = 1.0

        return torch.tensor(features, dtype=torch.float32)

    def encode_batch(
        self,
        states: list[tuple[dict[str, float], int]],
    ) -> torch.Tensor:
        """Encode a batch of states.

        Args:
            states: List of (decided_parameters, depth) tuples

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        encoded = [
            self.encode_state(params, depth) for params, depth in states
        ]
        return torch.stack(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through (identity) — encoding is done in encode_state.

        Args:
            x: Pre-encoded feature tensor

        Returns:
            Same tensor (identity operation)
        """
        return x


class ValueNetwork(nn.Module):
    """Predicts quality value for a calibration state.

    Input: state feature vector from StateEncoder
    Output: scalar quality estimate in [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        settings: MCTSSettings | None = None,
    ):
        """Initialize ValueNetwork.

        Args:
            input_dim: Input feature dimension from StateEncoder
            hidden_dims: Hidden layer dimensions. If None, uses settings.
            settings: MCTS settings for architecture.
        """
        _check_torch()
        super().__init__()

        self.settings = settings or MCTSSettings()
        dims = hidden_dims or self.settings.value_hidden_dims

        self.network = _build_mlp(
            input_dim=input_dim,
            hidden_dims=dims,
            output_dim=1,
            activation="relu",
            final_activation="sigmoid",
        )

        logger.debug(
            f"ValueNetwork: input={input_dim}, "
            f"hidden={dims}, output=1"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict value for state features.

        Args:
            x: State features of shape (batch, input_dim)

        Returns:
            Value predictions of shape (batch, 1)
        """
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Predicts action probabilities for a calibration state.

    Input: state feature vector from StateEncoder
    Output: probability distribution over action bins
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int | None = None,
        hidden_dims: list[int] | None = None,
        settings: MCTSSettings | None = None,
    ):
        """Initialize PolicyNetwork.

        Args:
            input_dim: Input feature dimension from StateEncoder
            num_actions: Number of action bins. If None, uses settings.action_bins.
            hidden_dims: Hidden layer dimensions. If None, uses settings.
            settings: MCTS settings for architecture.
        """
        _check_torch()
        super().__init__()

        self.settings = settings or MCTSSettings()
        self.num_actions = num_actions or self.settings.action_bins
        dims = hidden_dims or self.settings.policy_hidden_dims

        # No final activation — softmax applied in forward
        self.network = _build_mlp(
            input_dim=input_dim,
            hidden_dims=dims,
            output_dim=self.num_actions,
            activation="relu",
            final_activation=None,
        )

        logger.debug(
            f"PolicyNetwork: input={input_dim}, "
            f"hidden={dims}, actions={self.num_actions}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict action probabilities for state features.

        Args:
            x: State features of shape (batch, input_dim)

        Returns:
            Action probabilities of shape (batch, num_actions), sums to 1
        """
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class DualNetwork(nn.Module):
    """Combined value + policy network with shared encoder backbone.

    This is the main network used by the MCTS engine for both
    value estimation and policy guidance.
    """

    def __init__(self, settings: MCTSSettings | None = None):
        """Initialize DualNetwork.

        Args:
            settings: MCTS settings for architecture configuration.
        """
        _check_torch()
        super().__init__()

        self.settings = settings or MCTSSettings()

        # Encoder
        self.encoder = StateEncoder(self.settings)
        input_dim = self.encoder.output_dim

        # Value head
        self.value_head = ValueNetwork(
            input_dim=input_dim,
            hidden_dims=self.settings.value_hidden_dims,
            settings=self.settings,
        )

        # Policy head
        self.policy_head = PolicyNetwork(
            input_dim=input_dim,
            num_actions=self.settings.action_bins,
            hidden_dims=self.settings.policy_hidden_dims,
            settings=self.settings,
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualNetwork initialized: {total_params:,} parameters, "
            f"action_bins={self.settings.action_bins}"
        )

    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both heads.

        Args:
            x: State features of shape (batch, feature_dim)

        Returns:
            Tuple of (value, policy):
                - value: shape (batch, 1) in [0, 1]
                - policy: shape (batch, num_actions), sums to 1
        """
        features = self.encoder(x)
        value = self.value_head(features)
        policy = self.policy_head(features)
        return value, policy

    def predict(
        self,
        decided_parameters: dict[str, float],
        depth: int,
    ) -> tuple[float, list[float]]:
        """Predict value and policy for a single state (convenience method).

        Args:
            decided_parameters: Parameters decided so far
            depth: Current tree depth

        Returns:
            Tuple of (value_scalar, policy_list)
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder.encode_state(decided_parameters, depth)
            features = features.unsqueeze(0)  # Add batch dimension
            value, policy = self.forward(features)
            return (
                float(value.item()),
                policy.squeeze(0).tolist(),
            )

    def compute_loss(
        self,
        state_features: torch.Tensor,
        target_values: torch.Tensor,
        target_policies: torch.Tensor,
        weight_decay: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss for training.

        Loss = MSE(value, target) + CrossEntropy(policy, target) + λ * ||θ||²

        Args:
            state_features: Batch of encoded states (batch, feature_dim)
            target_values: Target value labels (batch, 1) in [0, 1]
            target_policies: Target policy distributions (batch, num_actions)
            weight_decay: L2 regularization weight. If None, uses settings.

        Returns:
            Tuple of (total_loss, value_loss, policy_loss)
        """
        wd = weight_decay if weight_decay is not None else self.settings.network_weight_decay

        # Forward pass
        pred_value, pred_policy = self.forward(state_features)

        # Value loss: MSE
        value_loss = F.mse_loss(pred_value, target_values)

        # Policy loss: cross-entropy (use log_softmax for numerical stability)
        # target_policies are probabilities, pred_policy is already softmax
        # Use KL divergence: sum(target * log(target / pred))
        # Equivalent to cross-entropy when target is fixed
        pred_log = torch.log(pred_policy + 1e-8)
        policy_loss = -torch.mean(torch.sum(target_policies * pred_log, dim=-1))

        # L2 regularization
        l2_reg = torch.tensor(0.0, device=state_features.device)
        if wd > 0:
            for param in self.parameters():
                l2_reg = l2_reg + torch.sum(param ** 2)
            l2_reg = wd * l2_reg

        total_loss = value_loss + policy_loss + l2_reg

        return total_loss, value_loss, policy_loss
