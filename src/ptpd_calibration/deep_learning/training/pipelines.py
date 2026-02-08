"""
Training pipelines for deep learning models.

This module provides complete training pipelines for all neural network
models in the deep learning module. Each pipeline handles:
- Data loading from synthetic generators
- Model initialization and configuration
- Training loop with validation
- Early stopping and learning rate scheduling
- Checkpointing and logging
- Anti-hallucination measures through proper data separation

Key principle: Training, validation, and test sets use different seeds
to ensure no data leakage between splits.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    pass

# Try to import PyTorch, but make it optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from ptpd_calibration.deep_learning.training.data_generators import (
    CurveDataGenerator,
    DefectDataGenerator,
    DetectionDataGenerator,
    ExposureDataGenerator,
    SyntheticDataConfig,
)

logger = logging.getLogger(__name__)

# Type variable for models
ModelT = TypeVar("ModelT")


class TrainingConfig(BaseSettings):
    """Configuration for training pipelines.

    All values are configurable via environment variables with PTPD_TRAIN_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTPD_TRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Training hyperparameters
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0)
    num_epochs: int = Field(default=100, ge=1, le=10000)
    warmup_epochs: int = Field(default=5, ge=0)

    # Early stopping
    early_stopping_patience: int = Field(default=10, ge=1)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0)

    # Learning rate scheduling
    lr_scheduler: str = Field(default="cosine")  # cosine, step, plateau
    lr_step_size: int = Field(default=30, ge=1)
    lr_gamma: float = Field(default=0.1, gt=0, le=1.0)
    lr_min: float = Field(default=1e-7, gt=0)

    # Checkpointing
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    save_every_n_epochs: int = Field(default=5, ge=1)
    keep_n_checkpoints: int = Field(default=3, ge=1)

    # Logging
    log_every_n_steps: int = Field(default=10, ge=1)
    validate_every_n_epochs: int = Field(default=1, ge=1)

    # Data
    train_samples: int = Field(default=10000, ge=100)
    val_samples: int = Field(default=2000, ge=10)
    test_samples: int = Field(default=2000, ge=10)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)

    # Device
    device: str = Field(default="auto")  # auto, cuda, cpu, mps
    mixed_precision: bool = Field(default=True)

    # Reproducibility
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    # Anti-hallucination measures
    label_smoothing: float = Field(default=0.1, ge=0, le=0.5)
    dropout_rate: float = Field(default=0.1, ge=0, le=0.5)
    data_augmentation: bool = Field(default=True)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    additional_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of a training run."""

    best_model_path: Path | None
    final_metrics: TrainingMetrics
    history: list[TrainingMetrics]
    total_training_time: float
    early_stopped: bool
    best_epoch: int


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value to monitor

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class BaseTrainingPipeline(ABC, Generic[ModelT]):
    """Abstract base class for training pipelines.

    Provides common functionality for all model training:
    - Device management
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Early stopping
    - Metrics logging
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for training. Install with: pip install torch torchvision"
            )

        self.config = config or TrainingConfig()
        self.data_config = data_config or SyntheticDataConfig()
        self.device = self._get_device()
        self.model: ModelT | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.history: list[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Set up reproducibility
        self._set_seed(self.config.seed)

        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @abstractmethod
    def _create_model(self) -> ModelT:
        """Create the model to train."""
        pass

    @abstractmethod
    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        pass

    @abstractmethod
    def _compute_loss(
        self,
        model: ModelT,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for a batch.

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs,
                eta_min=self.config.lr_min,
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=self.config.early_stopping_patience // 2,
                min_lr=self.config.lr_min,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")

    def _warmup_lr(self, epoch: int, step: int, total_steps: int) -> None:
        """Apply learning rate warmup."""
        if epoch < self.config.warmup_epochs and self.optimizer is not None:
            warmup_progress = (epoch * total_steps + step) / (
                self.config.warmup_epochs * total_steps
            )
            lr = self.config.learning_rate * warmup_progress
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False,
    ) -> Path | None:
        """Save model checkpoint."""
        if self.model is None or self.optimizer is None:
            return None

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.model_dump(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )

        for checkpoint in checkpoints[self.config.keep_n_checkpoints :]:
            checkpoint.unlink()

    def _load_checkpoint(self, path: Path) -> int:
        """Load model from checkpoint.

        Returns:
            Starting epoch
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be created before loading")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"] + 1

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epoch: int,
    ) -> tuple[float, dict[str, float]]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_metrics: dict[str, list[float]] = {}

        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = tuple(t.to(self.device) for t in batch)

            # Apply warmup
            self._warmup_lr(epoch, step, len(train_loader))

            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler is not None:
                with torch.amp.autocast(device_type=str(self.device)):
                    loss, metrics = self._compute_loss(model, batch)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, metrics = self._compute_loss(model, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # Collect metrics
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            # Log progress
            if (step + 1) % self.config.log_every_n_steps == 0:
                logger.debug(
                    f"Epoch {epoch}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        all_metrics: dict[str, list[float]] = {}

        for batch in val_loader:
            batch = tuple(t.to(self.device) for t in batch)

            if self.config.mixed_precision:
                with torch.amp.autocast(device_type=str(self.device)):
                    loss, metrics = self._compute_loss(model, batch)
            else:
                loss, metrics = self._compute_loss(model, batch)

            total_loss += loss.item()

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def train(
        self,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """Run the full training pipeline.

        Args:
            resume_from: Path to checkpoint to resume from

        Returns:
            TrainingResult with final metrics and history
        """
        logger.info(f"Starting training on device: {self.device}")
        start_time = time.time()

        # Create model, optimizer, scheduler
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        self.optimizer = self._create_optimizer(self.model)
        self.scheduler = self._create_scheduler(self.optimizer)

        # Set up mixed precision
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()

        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from is not None and resume_from.exists():
            start_epoch = self._load_checkpoint(resume_from)
            logger.info(f"Resumed from epoch {start_epoch}")

        # Training loop
        early_stopped = False
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self._train_epoch(self.model, train_loader, epoch)

            # Validate
            val_loss, val_metrics = None, {}
            if (epoch + 1) % self.config.validate_every_n_epochs == 0:
                val_loss, val_metrics = self._validate(self.model, val_loader)

            # Update learning rate
            if epoch >= self.config.warmup_epochs:
                if self.config.lr_scheduler == "plateau" and val_loss is not None:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_time=time.time() - epoch_start,
                best_val_loss=self.best_val_loss,
                additional_metrics={**train_metrics, **val_metrics},
            )

            # Check for best model
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                is_best = True

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                self._save_checkpoint(epoch, metrics, is_best)

            # Update history
            self.history.append(metrics)

            # Log progress
            log_msg = f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
            log_msg += f", LR: {current_lr:.2e}"
            logger.info(log_msg)

            # Early stopping
            if val_loss is not None and self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopped = True
                break

        # Final evaluation on test set
        test_loss, test_metrics = self._validate(self.model, test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")

        # Create final metrics
        final_metrics = TrainingMetrics(
            epoch=self.best_epoch,
            train_loss=self.history[-1].train_loss if self.history else 0.0,
            val_loss=test_loss,
            best_val_loss=self.best_val_loss,
            additional_metrics=test_metrics,
        )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return TrainingResult(
            best_model_path=self.config.checkpoint_dir / "best_model.pt",
            final_metrics=final_metrics,
            history=self.history,
            total_training_time=total_time,
            early_stopped=early_stopped,
            best_epoch=self.best_epoch,
        )


class DetectionTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for step tablet detection model."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
        num_classes: int = 21,
    ):
        super().__init__(config, data_config)
        self.num_classes = num_classes

    def _create_model(self) -> nn.Module:
        """Create detection model (simplified backbone + detection head)."""
        from ptpd_calibration.deep_learning.detection import DeepTabletDetector

        # Create detector which initializes the YOLO model
        _detector = DeepTabletDetector()  # Initialize model weights

        # For training, we need the underlying model
        # This is a simplified version - real training would use the full YOLO training
        model = nn.Sequential(
            # Backbone (simplified CNN)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            # Detection head
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, self.num_classes * 5),  # 5 = x, y, w, h, confidence
        )

        return model

    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for detection training."""
        # Create generators with different seeds for each split
        train_config = SyntheticDataConfig(
            seed=self.config.seed,
            input_noise_std=self.data_config.input_noise_std,
            output_noise_std=self.data_config.output_noise_std,
            label_noise_probability=self.data_config.label_noise_probability,
        )
        val_config = SyntheticDataConfig(
            seed=self.config.seed + 1000,  # Different seed
            input_noise_std=self.data_config.input_noise_std * 0.5,  # Less noise
            output_noise_std=self.data_config.output_noise_std * 0.5,
            label_noise_probability=0.0,  # No label noise for validation
        )
        test_config = SyntheticDataConfig(
            seed=self.config.seed + 2000,  # Different seed
            input_noise_std=0.0,  # No noise for testing
            output_noise_std=0.0,
            label_noise_probability=0.0,
        )

        train_gen = DetectionDataGenerator(train_config)
        val_gen = DetectionDataGenerator(val_config)
        test_gen = DetectionDataGenerator(test_config)

        # Generate datasets
        train_data = train_gen.generate(self.config.train_samples)
        val_data = val_gen.generate(self.config.val_samples)
        test_data = test_gen.generate(self.config.test_samples)

        # Convert to tensors
        train_images = torch.from_numpy(train_data["images"]).permute(0, 3, 1, 2)
        train_boxes = torch.from_numpy(train_data["bboxes"])

        val_images = torch.from_numpy(val_data["images"]).permute(0, 3, 1, 2)
        val_boxes = torch.from_numpy(val_data["bboxes"])

        test_images = torch.from_numpy(test_data["images"]).permute(0, 3, 1, 2)
        test_boxes = torch.from_numpy(test_data["bboxes"])

        # Create datasets
        train_dataset = TensorDataset(train_images, train_boxes)
        val_dataset = TensorDataset(val_images, val_boxes)
        test_dataset = TensorDataset(test_images, test_boxes)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _compute_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute detection loss."""
        images, targets = batch

        # Forward pass
        predictions = model(images)

        # Reshape predictions to match targets
        batch_size = images.size(0)
        predictions = predictions.view(batch_size, self.num_classes, 5)

        # Compute losses
        # Box regression loss (smooth L1)
        box_pred = predictions[:, :, :4]
        box_target = targets[:, :, :4]
        box_loss = nn.functional.smooth_l1_loss(box_pred, box_target)

        # Confidence loss (BCE)
        conf_pred = predictions[:, :, 4]
        conf_target = targets[:, :, 4]
        conf_loss = nn.functional.binary_cross_entropy_with_logits(conf_pred, conf_target)

        # Total loss
        total_loss = box_loss + conf_loss

        metrics = {
            "box_loss": box_loss.item(),
            "conf_loss": conf_loss.item(),
        }

        return total_loss, metrics


class CurveTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for neural curve prediction model."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
        num_zones: int = 21,
    ):
        super().__init__(config, data_config)
        self.num_zones = num_zones

    def _create_model(self) -> nn.Module:
        """Create curve prediction model."""
        from ptpd_calibration.deep_learning.neural_curve import CurveTransformer

        return CurveTransformer(
            num_zones=self.num_zones,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=self.config.dropout_rate,
        )

    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for curve prediction training."""
        # Create generators with different seeds
        train_config = SyntheticDataConfig(
            seed=self.config.seed,
            input_noise_std=self.data_config.input_noise_std,
            output_noise_std=self.data_config.output_noise_std,
        )
        val_config = SyntheticDataConfig(
            seed=self.config.seed + 1000,
            input_noise_std=self.data_config.input_noise_std * 0.5,
            output_noise_std=self.data_config.output_noise_std * 0.5,
        )
        test_config = SyntheticDataConfig(
            seed=self.config.seed + 2000,
            input_noise_std=0.0,
            output_noise_std=0.0,
        )

        train_gen = CurveDataGenerator(train_config, num_zones=self.num_zones)
        val_gen = CurveDataGenerator(val_config, num_zones=self.num_zones)
        test_gen = CurveDataGenerator(test_config, num_zones=self.num_zones)

        # Generate data
        train_data = train_gen.generate(self.config.train_samples)
        val_data = val_gen.generate(self.config.val_samples)
        test_data = test_gen.generate(self.config.test_samples)

        # Convert to tensors
        def to_tensors(data: dict) -> tuple[torch.Tensor, ...]:
            densities = torch.from_numpy(data["densities"])
            conditions = torch.from_numpy(data["process_conditions"])
            curves = torch.from_numpy(data["target_curves"])
            return densities, conditions, curves

        train_tensors = to_tensors(train_data)
        val_tensors = to_tensors(val_data)
        test_tensors = to_tensors(test_data)

        # Create datasets
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        test_dataset = TensorDataset(*test_tensors)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _compute_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute curve prediction loss."""
        densities, conditions, targets = batch

        # Forward pass
        predictions = model(densities, conditions)

        # MSE loss for curve prediction
        mse_loss = nn.functional.mse_loss(predictions, targets)

        # Monotonicity constraint (curves should be monotonically increasing)
        diffs = predictions[:, 1:] - predictions[:, :-1]
        monotonicity_loss = torch.mean(torch.relu(-diffs))

        # Smoothness constraint
        second_diffs = diffs[:, 1:] - diffs[:, :-1]
        smoothness_loss = torch.mean(second_diffs**2)

        # Total loss
        total_loss = mse_loss + 0.1 * monotonicity_loss + 0.01 * smoothness_loss

        metrics = {
            "mse_loss": mse_loss.item(),
            "monotonicity_loss": monotonicity_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
        }

        return total_loss, metrics


class ExposureTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for UV exposure prediction model."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
    ):
        super().__init__(config, data_config)

    def _create_model(self) -> nn.Module:
        """Create exposure prediction model."""
        from ptpd_calibration.deep_learning.uv_exposure import ExposureNet

        return ExposureNet(
            input_dim=8,  # Matching data generator feature count
            hidden_dim=128,
            num_layers=4,
            dropout=self.config.dropout_rate,
        )

    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for exposure prediction training."""
        # Create generators
        train_config = SyntheticDataConfig(
            seed=self.config.seed,
            input_noise_std=self.data_config.input_noise_std,
            output_noise_std=self.data_config.output_noise_std,
        )
        val_config = SyntheticDataConfig(
            seed=self.config.seed + 1000,
            input_noise_std=self.data_config.input_noise_std * 0.5,
            output_noise_std=self.data_config.output_noise_std * 0.5,
        )
        test_config = SyntheticDataConfig(
            seed=self.config.seed + 2000,
            input_noise_std=0.0,
            output_noise_std=0.0,
        )

        train_gen = ExposureDataGenerator(train_config)
        val_gen = ExposureDataGenerator(val_config)
        test_gen = ExposureDataGenerator(test_config)

        # Generate data
        train_data = train_gen.generate(self.config.train_samples)
        val_data = val_gen.generate(self.config.val_samples)
        test_data = test_gen.generate(self.config.test_samples)

        # Convert to tensors
        def to_tensors(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
            features = torch.from_numpy(data["features"])
            times = torch.from_numpy(data["exposure_times"]).unsqueeze(-1)
            return features, times

        train_tensors = to_tensors(train_data)
        val_tensors = to_tensors(val_data)
        test_tensors = to_tensors(test_data)

        # Create datasets
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        test_dataset = TensorDataset(*test_tensors)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _compute_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute exposure prediction loss."""
        features, targets = batch

        # Forward pass
        predictions = model(features)

        # MSE loss for regression
        mse_loss = nn.functional.mse_loss(predictions, targets)

        # Relative error (for meaningful metrics)
        relative_error = torch.mean(torch.abs(predictions - targets) / (torch.abs(targets) + 1e-6))

        metrics = {
            "mse_loss": mse_loss.item(),
            "relative_error": relative_error.item(),
        }

        return mse_loss, metrics


class DefectTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for defect detection model."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
        num_classes: int = 7,
    ):
        super().__init__(config, data_config)
        self.num_classes = num_classes

    def _create_model(self) -> nn.Module:
        """Create defect detection model (segmentation + classification)."""
        from ptpd_calibration.deep_learning.defect_detection import (
            DefectClassifierNet,
            DefectSegmentationNet,
        )

        # Combined model for both segmentation and classification
        class CombinedDefectModel(nn.Module):
            def __init__(self, num_classes: int, dropout: float, image_size: int = 256):
                super().__init__()
                self.segmentation = DefectSegmentationNet(in_channels=3, out_channels=1)
                self.classifier = DefectClassifierNet(
                    num_classes=num_classes, dropout=dropout, image_size=image_size
                )

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                seg_output = self.segmentation(x)
                class_output = self.classifier(x)
                return seg_output, class_output

        return CombinedDefectModel(
            num_classes=self.num_classes,
            dropout=self.config.dropout_rate,
        )

    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for defect detection training."""
        # Create generators
        train_config = SyntheticDataConfig(
            seed=self.config.seed,
            input_noise_std=self.data_config.input_noise_std,
            output_noise_std=self.data_config.output_noise_std,
            label_noise_probability=self.data_config.label_noise_probability,
        )
        val_config = SyntheticDataConfig(
            seed=self.config.seed + 1000,
            input_noise_std=self.data_config.input_noise_std * 0.5,
            output_noise_std=self.data_config.output_noise_std * 0.5,
            label_noise_probability=0.0,
        )
        test_config = SyntheticDataConfig(
            seed=self.config.seed + 2000,
            input_noise_std=0.0,
            output_noise_std=0.0,
            label_noise_probability=0.0,
        )

        train_gen = DefectDataGenerator(train_config, num_defect_types=self.num_classes)
        val_gen = DefectDataGenerator(val_config, num_defect_types=self.num_classes)
        test_gen = DefectDataGenerator(test_config, num_defect_types=self.num_classes)

        # Generate data
        train_data = train_gen.generate(self.config.train_samples)
        val_data = val_gen.generate(self.config.val_samples)
        test_data = test_gen.generate(self.config.test_samples)

        # Convert to tensors
        def to_tensors(
            data: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2)
            masks = torch.from_numpy(data["masks"]).unsqueeze(1)
            labels = torch.from_numpy(data["labels"]).long()
            return images, masks, labels

        train_tensors = to_tensors(train_data)
        val_tensors = to_tensors(val_data)
        test_tensors = to_tensors(test_data)

        # Create datasets
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        test_dataset = TensorDataset(*test_tensors)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _compute_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute defect detection loss."""
        images, masks, labels = batch

        # Forward pass
        seg_pred, class_pred = model(images)

        # Segmentation loss (BCE with logits + Dice)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(seg_pred, masks)

        # Dice loss
        seg_sigmoid = torch.sigmoid(seg_pred)
        intersection = (seg_sigmoid * masks).sum(dim=(2, 3))
        union = seg_sigmoid.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        dice_loss = dice_loss.mean()

        # Classification loss with label smoothing
        class_loss = nn.functional.cross_entropy(
            class_pred,
            labels,
            label_smoothing=self.config.label_smoothing,
        )

        # Combined loss
        total_loss = bce_loss + dice_loss + class_loss

        metrics = {
            "bce_loss": bce_loss.item(),
            "dice_loss": dice_loss.item(),
            "class_loss": class_loss.item(),
        }

        return total_loss, metrics


class RecipeTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for recipe recommendation model."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_config: SyntheticDataConfig | None = None,
        embedding_dim: int = 64,
    ):
        super().__init__(config, data_config)
        self.embedding_dim = embedding_dim

    def _create_model(self) -> nn.Module:
        """Create recipe recommendation model."""
        from ptpd_calibration.deep_learning.recipe_recommendation import (
            CollaborativeFilter,
        )

        return CollaborativeFilter(
            num_users=1000,  # Will be updated based on data
            num_recipes=500,
            embedding_dim=self.embedding_dim,
            hidden_dims=[256, 128],
            dropout=self.config.dropout_rate,
        )

    def _create_data_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for recipe recommendation training."""
        from ptpd_calibration.deep_learning.training.data_generators import (
            RecipeDataGenerator,
        )

        # Create generators
        train_config = SyntheticDataConfig(
            seed=self.config.seed,
            input_noise_std=self.data_config.input_noise_std,
            output_noise_std=self.data_config.output_noise_std,
        )
        val_config = SyntheticDataConfig(
            seed=self.config.seed + 1000,
            input_noise_std=0.0,
            output_noise_std=0.0,
        )
        test_config = SyntheticDataConfig(
            seed=self.config.seed + 2000,
            input_noise_std=0.0,
            output_noise_std=0.0,
        )

        train_gen = RecipeDataGenerator(train_config)
        val_gen = RecipeDataGenerator(val_config)
        test_gen = RecipeDataGenerator(test_config)

        # Generate data
        train_data = train_gen.generate(self.config.train_samples)
        val_data = val_gen.generate(self.config.val_samples)
        test_data = test_gen.generate(self.config.test_samples)

        # Convert to tensors
        def to_tensors(data: dict) -> tuple[torch.Tensor, ...]:
            user_ids = torch.from_numpy(data["user_ids"]).long()
            recipe_ids = torch.from_numpy(data["recipe_ids"]).long()
            ratings = torch.from_numpy(data["ratings"]).unsqueeze(-1)
            return user_ids, recipe_ids, ratings

        train_tensors = to_tensors(train_data)
        val_tensors = to_tensors(val_data)
        test_tensors = to_tensors(test_data)

        # Create datasets
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        test_dataset = TensorDataset(*test_tensors)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _compute_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute recipe recommendation loss."""
        user_ids, recipe_ids, ratings = batch

        # Forward pass
        predictions = model(user_ids, recipe_ids)

        # MSE loss for rating prediction
        mse_loss = nn.functional.mse_loss(predictions, ratings)

        # RMSE for metrics
        rmse = torch.sqrt(mse_loss)

        metrics = {
            "mse_loss": mse_loss.item(),
            "rmse": rmse.item(),
        }

        return mse_loss, metrics


# Export all pipelines
__all__ = [
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResult",
    "EarlyStopping",
    "BaseTrainingPipeline",
    "DetectionTrainingPipeline",
    "CurveTrainingPipeline",
    "ExposureTrainingPipeline",
    "DefectTrainingPipeline",
    "RecipeTrainingPipeline",
]
