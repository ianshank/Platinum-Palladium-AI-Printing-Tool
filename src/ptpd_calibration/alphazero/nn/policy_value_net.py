"""
PolicyValueNet: 1D MLP neural network for AlphaZero.

This module implements the neural network component of AlphaZero,
refactored from a 2D CNN (for board games) to a 1D MLP suitable
for parameter vector inputs.

The network outputs:
- Policy: Probability distribution over actions
- Value: Expected reward from current state
"""

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ptpd_calibration.alphazero.config import AlphaZeroConfig, NetworkConfig


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the neural network. "
            "Install with: pip install torch"
        )


class ResidualBlock(nn.Module):
    """
    Residual block for the policy-value network.

    Uses skip connections for better gradient flow.
    """

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        """
        Initialize residual block.

        Args:
            dim: Hidden dimension
            dropout_rate: Dropout probability
        """
        _check_torch()
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class PolicyValueNetwork(nn.Module):
    """
    Neural network for policy and value prediction.

    Architecture:
    - Input projection layer
    - Stack of residual blocks
    - Separate policy and value heads
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize the network.

        Args:
            config: Network configuration
        """
        _check_torch()
        super().__init__()

        self.config = config
        input_dim = config.input_dim
        action_dim = config.action_dim
        hidden_dims = config.hidden_dims
        dropout_rate = config.dropout_rate

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Transition layer if dimensions change
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.res_blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.ReLU(),
                    )
                )
            else:
                self.res_blocks.append(ResidualBlock(hidden_dims[i], dropout_rate))

        # Shared representation
        final_hidden = hidden_dims[-1]

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.BatchNorm1d(final_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden // 2, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.BatchNorm1d(final_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden // 2, 1),
            nn.Tanh(),  # Value in [-1, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input state tensor (batch_size, input_dim)

        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch_size, action_dim)
            - value: (batch_size, 1)
        """
        # Input projection
        h = self.input_proj(x)

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)

        # Policy head
        policy_logits = self.policy_head(h)

        # Value head
        value = self.value_head(h)

        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        valid_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy probabilities and value.

        Args:
            x: Input state tensor
            valid_actions: Optional mask for valid actions

        Returns:
            Tuple of (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)

            # Apply mask if provided
            if valid_actions is not None:
                # Set invalid actions to very negative
                policy_logits = policy_logits.masked_fill(
                    valid_actions == 0,
                    float("-inf"),
                )

            # Softmax for probabilities
            policy_probs = F.softmax(policy_logits, dim=-1)

        return policy_probs, value


class PolicyValueNet:
    """
    High-level wrapper for the policy-value network.

    Provides training, inference, and checkpoint management.
    """

    def __init__(self, config: AlphaZeroConfig | None = None):
        """
        Initialize the policy-value network.

        Args:
            config: AlphaZero configuration
        """
        _check_torch()

        self.config = config or AlphaZeroConfig()
        self.network_config = self.config.network
        self.device = torch.device(self.config.device)

        # Create network
        self.net = PolicyValueNetwork(self.network_config).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Training history
        self.train_history: list[dict] = []

    def predict(
        self,
        state: np.ndarray,
        valid_actions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Get policy and value for a state.

        Args:
            state: State vector (state_dim,) or (batch, state_dim)
            valid_actions: Optional mask for valid actions

        Returns:
            Tuple of (policy_probs, value)
        """
        self.net.eval()

        # Handle single state
        single = state.ndim == 1
        if single:
            state = state[None, :]
            if valid_actions is not None:
                valid_actions = valid_actions[None, :]

        # Convert to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        valid_tensor = None
        if valid_actions is not None:
            valid_tensor = torch.FloatTensor(valid_actions).to(self.device)

        # Get predictions
        with torch.no_grad():
            policy_probs, value = self.net.predict(state_tensor, valid_tensor)

        policy = policy_probs.cpu().numpy()
        val = value.cpu().numpy()

        if single:
            return policy[0], float(val[0, 0])
        return policy, val[:, 0]

    def train_step(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
    ) -> dict:
        """
        Perform a single training step.

        Args:
            states: Batch of state vectors
            target_policies: Target policy distributions
            target_values: Target values

        Returns:
            Dictionary with loss values
        """
        self.net.train()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(target_policies).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)

        # Forward pass
        policy_logits, values = self.net(states)

        # Calculate losses
        # Policy loss: cross-entropy with target distribution
        policy_loss = -torch.mean(
            torch.sum(target_policies * F.log_softmax(policy_logits, dim=-1), dim=-1)
        )

        # Value loss: MSE
        value_loss = self.value_loss_fn(values.squeeze(-1), target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def train_on_batch(
        self,
        replay_buffer: list[tuple[np.ndarray, np.ndarray, float]],
        batch_size: int | None = None,
        num_epochs: int | None = None,
    ) -> dict:
        """
        Train on a batch of experience from replay buffer.

        Args:
            replay_buffer: List of (state, policy, value) tuples
            batch_size: Batch size for training
            num_epochs: Number of epochs

        Returns:
            Training statistics
        """
        if len(replay_buffer) < 10:
            return {"error": "Not enough samples"}

        batch_size = batch_size or self.config.training.batch_size
        num_epochs = num_epochs or self.config.training.num_epochs

        # Convert buffer to arrays
        states = np.array([x[0] for x in replay_buffer])
        policies = np.array([x[1] for x in replay_buffer])
        values = np.array([x[2] for x in replay_buffer])

        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(policies),
            torch.FloatTensor(values),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        epoch_losses = []
        for epoch in range(num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)

                losses = self.train_step(
                    batch_states.cpu().numpy(),
                    batch_policies.cpu().numpy(),
                    batch_values.cpu().numpy(),
                )

                epoch_policy_loss += losses["policy_loss"]
                epoch_value_loss += losses["value_loss"]
                num_batches += 1

            epoch_losses.append({
                "epoch": epoch,
                "policy_loss": epoch_policy_loss / num_batches,
                "value_loss": epoch_value_loss / num_batches,
            })

        # Record history
        self.train_history.extend(epoch_losses)

        return {
            "final_policy_loss": epoch_losses[-1]["policy_loss"],
            "final_value_loss": epoch_losses[-1]["value_loss"],
            "num_epochs": num_epochs,
            "num_samples": len(replay_buffer),
        }

    def save(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "train_history": self.train_history,
        }

        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "train_history" in checkpoint:
            self.train_history = checkpoint["train_history"]

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


def check_net_dims() -> bool:
    """
    Verify network dimensions are correct.

    Returns:
        True if dimensions check out
    """
    _check_torch()

    config = AlphaZeroConfig()
    net = PolicyValueNet(config)

    # Create dummy input
    batch_size = 4
    state_dim = config.state.full_state_dim
    action_dim = config.action.num_actions

    dummy_state = np.random.randn(batch_size, state_dim).astype(np.float32)

    # Forward pass
    policy, value = net.predict(dummy_state)

    # Check shapes
    if policy.shape != (batch_size, action_dim):
        print(f"ERROR: Policy shape {policy.shape} != expected {(batch_size, action_dim)}")
        return False

    if value.shape != (batch_size,):
        print(f"ERROR: Value shape {value.shape} != expected {(batch_size,)}")
        return False

    # Check policy is valid distribution
    if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-5):
        print(f"ERROR: Policy does not sum to 1: {policy.sum(axis=1)}")
        return False

    # Check value is in valid range
    if not np.all((value >= -1) & (value <= 1)):
        print(f"ERROR: Value out of range: {value}")
        return False

    print("Network dimension check passed!")
    print(f"  Input dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Total parameters: {net.get_num_parameters():,}")

    return True


if __name__ == "__main__":
    check_net_dims()
