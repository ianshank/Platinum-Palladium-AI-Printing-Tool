"""
Training loop and utilities for deep learning curve prediction.

Provides a flexible training framework with:
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Comprehensive metrics logging
- Custom loss functions for curve learning
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from ptpd_calibration.ml.deep.exceptions import CheckpointError, TrainingError

if TYPE_CHECKING:
    from ptpd_calibration.config import DeepLearningSettings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install ptpd-calibration[deep]"
        )


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_mse: float = 0.0
    val_mse: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    monotonicity_loss: float = 0.0
    smoothness_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "train_mae": self.train_mae,
            "val_mae": self.val_mae,
            "monotonicity_loss": self.monotonicity_loss,
            "smoothness_loss": self.smoothness_loss,
            "learning_rate": self.learning_rate,
            "epoch_time": self.epoch_time,
        }


@dataclass
class TrainingHistory:
    """Container for full training history."""

    metrics: list[TrainingMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time: float = 0.0

    def add(self, metrics: TrainingMetrics) -> None:
        """Add metrics for an epoch."""
        self.metrics.append(metrics)
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_epoch = metrics.epoch

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time": self.total_time,
        }


class CurveLoss(nn.Module):
    """
    Custom loss function for curve prediction.

    Combines MSE loss with regularization terms:
    - Monotonicity penalty: penalizes non-monotonic curves
    - Smoothness penalty: penalizes high-frequency variations
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        monotonicity_weight: float = 0.1,
        smoothness_weight: float = 0.05,
    ):
        """
        Initialize CurveLoss.

        Args:
            mse_weight: Weight for MSE loss.
            monotonicity_weight: Weight for monotonicity penalty.
            smoothness_weight: Weight for smoothness penalty.
        """
        _check_torch()
        super().__init__()
        self.mse_weight = mse_weight
        self.monotonicity_weight = monotonicity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss.

        Args:
            predicted: Predicted curve of shape (batch, length).
            target: Target curve of shape (batch, length).

        Returns:
            Tuple of (total_loss, loss_components).
        """
        # MSE loss
        mse_loss = nn.functional.mse_loss(predicted, target)

        # Monotonicity penalty: penalize decreasing values
        diffs = predicted[:, 1:] - predicted[:, :-1]
        monotonicity_loss = torch.mean(torch.relu(-diffs) ** 2)

        # Smoothness penalty: penalize high second derivatives
        second_diffs = diffs[:, 1:] - diffs[:, :-1]
        smoothness_loss = torch.mean(second_diffs**2)

        # Total loss
        total_loss = (
            self.mse_weight * mse_loss
            + self.monotonicity_weight * monotonicity_loss
            + self.smoothness_weight * smoothness_loss
        )

        components = {
            "mse": float(mse_loss.item()),
            "monotonicity": float(monotonicity_loss.item()),
            "smoothness": float(smoothness_loss.item()),
        }

        return total_loss, components


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait.
            min_delta: Minimum improvement.
            mode: 'min' or 'max' for loss or metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        PyTorch device.
    """
    _check_torch()

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


class CurveTrainer:
    """
    Trainer for curve prediction models.

    Handles the complete training loop including:
    - Training and validation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        settings: Optional[DeepLearningSettings] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize CurveTrainer.

        Args:
            model: PyTorch model to train.
            settings: Training settings.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps').
        """
        _check_torch()

        from ptpd_calibration.config import DeepLearningSettings, get_settings

        self.settings = settings or get_settings().deep_learning
        self.device = get_device(device or self.settings.device)
        self.model = model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        # Initialize loss function
        self.criterion = CurveLoss(
            mse_weight=self.settings.mse_weight,
            monotonicity_weight=self.settings.monotonicity_weight,
            smoothness_weight=self.settings.smoothness_weight,
        )

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.settings.early_stopping_patience,
            min_delta=self.settings.min_delta,
        )

        # Training state
        self.history = TrainingHistory()
        self.best_model_state = None
        self.callbacks: list[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """Add a callback function called after each epoch."""
        self.callbacks.append(callback)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs (uses settings if not provided).
            checkpoint_dir: Directory for checkpoints.

        Returns:
            Training history.
        """
        num_epochs = num_epochs or self.settings.num_epochs
        checkpoint_dir = checkpoint_dir or self.settings.checkpoint_dir

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()

                # Training phase
                train_metrics = self._train_epoch(train_loader)

                # Validation phase
                val_metrics = self._validate_epoch(val_loader)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Create metrics
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    train_mse=train_metrics["mse"],
                    val_mse=val_metrics["mse"],
                    train_mae=train_metrics["mae"],
                    val_mae=val_metrics["mae"],
                    monotonicity_loss=train_metrics["monotonicity"],
                    smoothness_loss=train_metrics["smoothness"],
                    learning_rate=current_lr,
                    epoch_time=time.time() - epoch_start,
                )
                self.history.add(metrics)

                # Log progress
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"train_loss: {metrics.train_loss:.6f}, "
                    f"val_loss: {metrics.val_loss:.6f}, "
                    f"val_mae: {metrics.val_mae:.6f}, "
                    f"lr: {current_lr:.2e}"
                )

                # Update scheduler
                self.scheduler.step(metrics.val_loss)

                # Save best model
                if metrics.val_loss <= self.history.best_val_loss:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

                    if checkpoint_dir and self.settings.save_best_only:
                        self._save_checkpoint(
                            checkpoint_dir / "best_model.pt",
                            epoch + 1,
                            metrics.val_loss,
                        )

                # Check early stopping
                if self.early_stopping(metrics.val_loss):
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best val_loss: {self.history.best_val_loss:.6f} "
                        f"at epoch {self.history.best_epoch}"
                    )
                    break

                # Run callbacks
                for callback in self.callbacks:
                    callback(metrics)

        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        self.history.total_time = time.time() - start_time
        logger.info(
            f"Training complete. Best val_loss: {self.history.best_val_loss:.6f} "
            f"at epoch {self.history.best_epoch}. "
            f"Total time: {self.history.total_time:.1f}s"
        )

        return self.history

    def _train_epoch(self, data_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_monotonicity = 0.0
        total_smoothness = 0.0
        num_batches = 0

        for features, targets in data_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(features)

            # Compute loss
            loss, components = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_mse += components["mse"]
            total_monotonicity += components["monotonicity"]
            total_smoothness += components["smoothness"]
            total_mae += float(torch.mean(torch.abs(predictions - targets)).item())
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches,
            "monotonicity": total_monotonicity / num_batches,
            "smoothness": total_smoothness / num_batches,
        }

    def _validate_epoch(self, data_loader: DataLoader) -> dict:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions, _ = self.model(features)

                # Compute loss
                loss, components = self.criterion(predictions, targets)

                # Accumulate metrics
                total_loss += loss.item()
                total_mse += components["mse"]
                total_mae += float(torch.mean(torch.abs(predictions - targets)).item())
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches,
        }

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save a model checkpoint."""
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "settings": self.settings.model_dump(),
            }
            torch.save(checkpoint, path)
            logger.debug(f"Saved checkpoint to {path}")
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, path: Path) -> dict:
        """
        Load a model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint metadata.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
            return {
                "epoch": checkpoint["epoch"],
                "val_loss": checkpoint["val_loss"],
            }
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> dict:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: Data loader for evaluation.

        Returns:
            Evaluation metrics.
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                predictions, _ = self.model(features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        mse = float(np.mean((predictions - targets) ** 2))
        mae = float(np.mean(np.abs(predictions - targets)))
        max_error = float(np.max(np.abs(predictions - targets)))

        # Check monotonicity
        diffs = np.diff(predictions, axis=1)
        monotonicity_violations = np.sum(diffs < -1e-6) / predictions.size

        # Correlation
        correlations = []
        for p, t in zip(predictions, targets):
            correlations.append(np.corrcoef(p, t)[0, 1])
        mean_correlation = float(np.mean(correlations))

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "monotonicity_violations": monotonicity_violations,
            "mean_correlation": mean_correlation,
            "num_samples": len(predictions),
        }
