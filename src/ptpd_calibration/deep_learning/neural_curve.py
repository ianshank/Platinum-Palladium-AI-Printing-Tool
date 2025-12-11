"""
Transformer-based neural curve prediction for Platinum-Palladium printing.

This module provides deep learning-based curve prediction using Transformer
architectures with:
- Multi-head self-attention for curve modeling
- Positional encoding for sequence awareness
- Process parameter conditioning
- Uncertainty quantification (ensemble or MC dropout)
- Monotonicity constraints
- Model save/load functionality

All parameters are configuration-driven following 2025 best practices.
Dependencies are lazily loaded to avoid import errors.
"""

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ptpd_calibration.deep_learning.config import NeuralCurveSettings
from ptpd_calibration.deep_learning.models import CurvePredictionResult
from ptpd_calibration.deep_learning.types import (
    CurveLossFunction,
    CurvePredictorArchitecture,
    CurvePoints,
    UncertaintyMethod,
)


class CurveTransformer:
    """
    Transformer model for curve prediction (nn.Module wrapper).

    This class implements a Transformer architecture specifically designed for
    predicting tonal curves in platinum-palladium printing. Features include:
    - Positional encoding for curve point ordering
    - Multi-head self-attention for capturing curve relationships
    - Feed-forward networks for non-linear transformations
    - Support for process parameter conditioning

    The model is lazily initialized to avoid PyTorch import errors.

    Architecture:
        Input → Embedding → Positional Encoding → Transformer Layers → Output Head

    Args:
        settings: Neural curve configuration settings
    """

    def __init__(self, settings: NeuralCurveSettings):
        """Initialize transformer architecture."""
        self.settings = settings
        self._nn = None
        self._model = None
        self._torch = None

    def _lazy_load(self) -> None:
        """Lazy load PyTorch and build model."""
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for neural curve prediction. "
                "Install with: pip install torch>=2.0.0"
            ) from e

        # Build model
        self._model = self._build_model()

    def _build_model(self) -> Any:
        """Build the transformer model."""
        nn = self._nn
        torch = self._torch

        class TransformerCurvePredictor(nn.Module):
            """Transformer model for curve prediction."""

            def __init__(self, config: NeuralCurveSettings):
                super().__init__()
                self.config = config

                # Input projection
                self.input_projection = nn.Linear(
                    config.input_features, config.d_model
                )

                # Positional encoding
                self.positional_encoding = self._create_positional_encoding(
                    config.max_sequence_length, config.d_model
                )

                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,  # Pre-LN architecture (2025 best practice)
                )

                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=config.n_layers,
                )

                # Conditioning network (if enabled)
                if config.include_conditioning:
                    self.conditioning_net = nn.Sequential(
                        nn.Linear(config.input_features, config.d_model // 2),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_model // 2, config.d_model),
                    )

                # Output head
                self.output_head = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model // 2, 1),
                    nn.Sigmoid(),  # Output in [0, 1] range
                )

                # Initialize weights
                self.apply(self._init_weights)

            def _create_positional_encoding(
                self, max_len: int, d_model: int
            ) -> torch.Tensor:
                """Create sinusoidal positional encoding."""
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-np.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe

            def _init_weights(self, module):
                """Initialize weights (Kaiming for GELU, Xavier for linear)."""
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            def forward(
                self,
                x: torch.Tensor,
                conditioning: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                """
                Forward pass.

                Args:
                    x: Input tensor (batch_size, seq_len, input_features)
                    conditioning: Optional conditioning parameters (batch_size, input_features)

                Returns:
                    Predicted curve values (batch_size, seq_len, 1)
                """
                batch_size, seq_len, _ = x.shape

                # Project input
                x = self.input_projection(x)  # (B, S, d_model)

                # Add positional encoding
                pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
                pos_enc = pos_enc.to(x.device)
                x = x + pos_enc

                # Add conditioning if provided
                if conditioning is not None and self.config.include_conditioning:
                    cond_embed = self.conditioning_net(conditioning)  # (B, d_model)
                    cond_embed = cond_embed.unsqueeze(1)  # (B, 1, d_model)
                    x = x + cond_embed

                # Transformer encoding
                x = self.transformer_encoder(x)  # (B, S, d_model)

                # Output prediction
                output = self.output_head(x)  # (B, S, 1)

                return output.squeeze(-1)  # (B, S)

        return TransformerCurvePredictor(self.settings)

    def get_model(self) -> Any:
        """Get the PyTorch model."""
        self._lazy_load()
        return self._model

    def parameters(self):
        """Get model parameters for optimizer."""
        return self.get_model().parameters()

    def train(self):
        """Set model to training mode."""
        self.get_model().train()

    def eval(self):
        """Set model to evaluation mode."""
        self.get_model().eval()

    def to(self, device: str):
        """Move model to device."""
        self.get_model().to(device)
        return self

    def __call__(self, *args, **kwargs):
        """Forward pass through model."""
        return self.get_model()(*args, **kwargs)


class NeuralCurvePredictor:
    """
    Neural curve predictor using Transformer architecture.

    This class provides a complete training and inference pipeline for predicting
    tonal curves using deep learning. Key features:
    - Transformer-based architecture with attention mechanism
    - Uncertainty quantification via ensemble or MC dropout
    - Process parameter conditioning
    - Monotonicity and smoothness constraints
    - Model checkpointing and loading

    Examples:
        >>> from ptpd_calibration.deep_learning.config import NeuralCurveSettings
        >>> settings = NeuralCurveSettings()
        >>> predictor = NeuralCurvePredictor(settings)
        >>>
        >>> # Predict curve with conditioning
        >>> result = predictor.predict(
        ...     input_values=np.linspace(0, 1, 256),
        ...     conditioning={
        ...         "paper_type": "Arches Platine",
        ...         "metal_ratio": 0.5,
        ...         "exposure_time": 300
        ...     }
        ... )
        >>>
        >>> # Train on custom data
        >>> predictor.train(
        ...     X_train=training_features,
        ...     y_train=training_curves,
        ...     X_val=validation_features,
        ...     y_val=validation_curves
        ... )
        >>>
        >>> # Save and load
        >>> predictor.save_model("ptpd_curve_model.pt")
        >>> predictor.load_model("ptpd_curve_model.pt")

    Attributes:
        settings: Neural curve configuration
        device: Torch device (cuda, cpu, or mps)
        model: CurveTransformer instance
        optimizer: Optimizer (if training)
        ensemble: List of models (if using ensemble uncertainty)
    """

    def __init__(
        self,
        settings: Optional[NeuralCurveSettings] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the neural curve predictor.

        Args:
            settings: Neural curve settings. If None, uses defaults.
            device: Override device. If None, uses settings.device.
        """
        self.settings = settings or NeuralCurveSettings()
        self.device = device or self.settings.device
        self._resolve_device()

        # Lazy-loaded components
        self._torch = None
        self._nn = None
        self.model: Optional[CurveTransformer] = None
        self.optimizer = None
        self.ensemble: list[CurveTransformer] = []

    def _resolve_device(self) -> None:
        """Resolve device from 'auto' setting."""
        if self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

    def _lazy_load_dependencies(self) -> None:
        """Lazy load PyTorch dependencies."""
        if self._torch is not None:
            return

        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for neural curve prediction. "
                "Install with: pip install torch>=2.0.0"
            ) from e

    def _initialize_model(self) -> None:
        """Initialize the model."""
        if self.model is not None:
            return

        self._lazy_load_dependencies()

        # Create model
        self.model = CurveTransformer(self.settings)
        self.model.to(self.device)

        # Load pretrained if specified
        if (
            self.settings.pretrained_model_path
            and Path(self.settings.pretrained_model_path).exists()
        ):
            self.load_model(self.settings.pretrained_model_path)

    def _initialize_ensemble(self) -> None:
        """Initialize ensemble of models for uncertainty quantification."""
        if len(self.ensemble) > 0:
            return

        self._lazy_load_dependencies()

        for _ in range(self.settings.ensemble_size):
            model = CurveTransformer(self.settings)
            model.to(self.device)
            self.ensemble.append(model)

    def _compute_loss(
        self,
        predictions: Any,
        targets: Any,
        input_values: Any,
    ) -> Any:
        """
        Compute loss with optional monotonicity and smoothness penalties.

        Args:
            predictions: Predicted curve values
            targets: Ground truth curve values
            input_values: Input curve points (for monotonicity check)

        Returns:
            Total loss value
        """
        # Base loss
        if self.settings.loss_function == CurveLossFunction.MSE:
            base_loss = self._nn.functional.mse_loss(predictions, targets)
        elif self.settings.loss_function == CurveLossFunction.MAE:
            base_loss = self._nn.functional.l1_loss(predictions, targets)
        elif self.settings.loss_function == CurveLossFunction.HUBER:
            base_loss = self._nn.functional.huber_loss(predictions, targets)
        elif self.settings.loss_function == CurveLossFunction.SMOOTH_L1:
            base_loss = self._nn.functional.smooth_l1_loss(predictions, targets)
        else:
            base_loss = self._nn.functional.mse_loss(predictions, targets)

        total_loss = base_loss

        # Monotonicity penalty
        if self.settings.monotonicity_weight > 0:
            # Penalize negative slopes
            slopes = predictions[:, 1:] - predictions[:, :-1]
            monotonicity_penalty = self._torch.relu(-slopes).mean()
            total_loss = total_loss + self.settings.monotonicity_weight * monotonicity_penalty

        # Smoothness penalty
        if self.settings.smoothness_weight > 0:
            # Penalize second derivative (curvature)
            second_deriv = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            smoothness_penalty = (second_deriv ** 2).mean()
            total_loss = total_loss + self.settings.smoothness_weight * smoothness_penalty

        return total_loss

    def predict(
        self,
        input_values: CurvePoints,
        conditioning: Optional[dict[str, Any]] = None,
        return_uncertainty: bool = True,
    ) -> CurvePredictionResult:
        """
        Predict a tonal curve with uncertainty quantification.

        Args:
            input_values: Input curve points (x-axis values), array of shape (N,)
            conditioning: Optional conditioning parameters (e.g., paper type, metal ratio)
            return_uncertainty: Whether to compute and return uncertainty estimates

        Returns:
            CurvePredictionResult with predicted curve and uncertainty

        Raises:
            ValueError: If input is invalid
            RuntimeError: If prediction fails
        """
        start_time = time.time()

        # Initialize model
        self._initialize_model()

        # Validate input
        if len(input_values.shape) != 1:
            raise ValueError(f"input_values must be 1D array, got shape {input_values.shape}")

        num_points = len(input_values)

        # Prepare input tensor
        # Shape: (1, num_points, input_features)
        input_tensor = self._prepare_input_tensor(input_values, conditioning)

        # Predict
        self.model.eval()
        with self._torch.no_grad():
            if (
                return_uncertainty
                and self.settings.uncertainty_method == UncertaintyMethod.MC_DROPOUT
            ):
                # MC Dropout: keep dropout active during inference
                predictions = self._predict_mc_dropout(input_tensor, conditioning)
                output_values = predictions.mean(axis=0)
                uncertainty = predictions.std(axis=0)
            elif (
                return_uncertainty
                and self.settings.uncertainty_method == UncertaintyMethod.ENSEMBLE
            ):
                # Ensemble: use multiple models
                predictions = self._predict_ensemble(input_tensor, conditioning)
                output_values = predictions.mean(axis=0)
                uncertainty = predictions.std(axis=0)
            else:
                # Single prediction
                output = self.model(input_tensor)
                output_values = output[0].cpu().numpy()
                uncertainty = None

        # Calculate curve properties
        is_monotonic = self._check_monotonic(output_values)
        slopes = np.diff(output_values) / np.diff(input_values)
        max_slope = float(slopes.max()) if len(slopes) > 0 else 1.0
        min_slope = float(slopes.min()) if len(slopes) > 0 else 0.0

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000

        # Build result
        return CurvePredictionResult(
            input_values=input_values.tolist(),
            output_values=output_values.tolist(),
            num_points=num_points,
            uncertainty=uncertainty.tolist() if uncertainty is not None else None,
            mean_uncertainty=(
                float(uncertainty.mean()) if uncertainty is not None else 0.0
            ),
            confidence=1.0 - (uncertainty.mean() if uncertainty is not None else 0.0),
            is_monotonic=is_monotonic,
            max_slope=max_slope,
            min_slope=min_slope,
            conditioning_factors=conditioning or {},
            inference_time_ms=inference_time,
            device_used=self.device,
            model_version=self.settings.architecture.value,
        )

    def _prepare_input_tensor(
        self,
        input_values: CurvePoints,
        conditioning: Optional[dict[str, Any]],
    ) -> Any:
        """Prepare input tensor from curve points and conditioning."""
        num_points = len(input_values)

        # Create feature matrix
        # Features: [input_value, position_normalized, ...]
        features = np.zeros((1, num_points, self.settings.input_features))
        features[0, :, 0] = input_values  # Input curve points
        features[0, :, 1] = np.linspace(0, 1, num_points)  # Normalized position

        # Add conditioning features if provided
        if conditioning and self.settings.include_conditioning:
            # Simple encoding: repeat conditioning for all points
            # In practice, you'd have more sophisticated encoding
            if "metal_ratio" in conditioning:
                features[0, :, 2] = conditioning["metal_ratio"]
            if "exposure_time" in conditioning:
                # Normalize exposure time (assume max 600 seconds)
                features[0, :, 3] = conditioning["exposure_time"] / 600.0
            # Add more conditioning features as needed

        # Convert to tensor
        return self._torch.from_numpy(features).float().to(self.device)

    def _prepare_conditioning_tensor(
        self, conditioning: Optional[dict[str, Any]]
    ) -> Optional[Any]:
        """Prepare conditioning tensor for the model."""
        if not conditioning or not self.settings.include_conditioning:
            return None

        # Create conditioning feature vector
        cond_features = np.zeros((1, self.settings.input_features))

        if "metal_ratio" in conditioning:
            cond_features[0, 0] = conditioning["metal_ratio"]
        if "exposure_time" in conditioning:
            cond_features[0, 1] = conditioning["exposure_time"] / 600.0

        return self._torch.from_numpy(cond_features).float().to(self.device)

    def _predict_mc_dropout(
        self, input_tensor: Any, conditioning: Optional[dict[str, Any]]
    ) -> np.ndarray:
        """Predict with MC dropout for uncertainty estimation."""
        self.model.train()  # Keep dropout active

        predictions = []
        cond_tensor = self._prepare_conditioning_tensor(conditioning)

        for _ in range(self.settings.mc_dropout_samples):
            with self._torch.no_grad():
                output = self.model(input_tensor, cond_tensor)
                predictions.append(output[0].cpu().numpy())

        self.model.eval()
        return np.array(predictions)

    def _predict_ensemble(
        self, input_tensor: Any, conditioning: Optional[dict[str, Any]]
    ) -> np.ndarray:
        """Predict with ensemble for uncertainty estimation."""
        if len(self.ensemble) == 0:
            raise RuntimeError("Ensemble not initialized. Train ensemble first.")

        predictions = []
        cond_tensor = self._prepare_conditioning_tensor(conditioning)

        for model in self.ensemble:
            model.eval()
            with self._torch.no_grad():
                output = model(input_tensor, cond_tensor)
                predictions.append(output[0].cpu().numpy())

        return np.array(predictions)

    def _check_monotonic(self, values: np.ndarray) -> bool:
        """Check if curve is monotonically increasing."""
        return bool(np.all(np.diff(values) >= 0))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_best: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the neural curve predictor.

        Args:
            X_train: Training features, shape (N, num_points, input_features)
            y_train: Training targets, shape (N, num_points)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            save_best: Whether to save best model checkpoint

        Returns:
            Dictionary with training history (loss, val_loss, etc.)

        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If training fails
        """
        self._initialize_model()

        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same batch size")

        # Convert to tensors
        X_train_tensor = self._torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = self._torch.from_numpy(y_train).float().to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = self._torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = self._torch.from_numpy(y_val).float().to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = self._torch.optim.AdamW(
                self.model.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=self.settings.weight_decay,
            )

        # Training history
        history = {"loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.settings.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(X_train_tensor), self.settings.batch_size):
                batch_X = X_train_tensor[i : i + self.settings.batch_size]
                batch_y = y_train_tensor[i : i + self.settings.batch_size]

                # Forward pass
                predictions = self.model(batch_X)
                loss = self._compute_loss(predictions, batch_y, batch_X[:, :, 0])

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)

            # Validation phase
            if X_val_tensor is not None:
                self.model.eval()
                with self._torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self._compute_loss(
                        val_predictions, y_val_tensor, X_val_tensor[:, :, 0]
                    )
                    val_loss_value = val_loss.item()
                    history["val_loss"].append(val_loss_value)

                # Early stopping
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    patience_counter = 0

                    # Save best model
                    if save_best and self.settings.save_model_path:
                        self.save_model(self.settings.save_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= self.settings.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            if (epoch + 1) % 10 == 0:
                if X_val_tensor is not None:
                    print(
                        f"Epoch {epoch + 1}/{self.settings.epochs} - "
                        f"Loss: {avg_loss:.4f}, Val Loss: {val_loss_value:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{self.settings.epochs} - "
                        f"Loss: {avg_loss:.4f}"
                    )

        return history

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> list[dict[str, list[float]]]:
        """
        Train an ensemble of models for uncertainty quantification.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            List of training histories for each ensemble member
        """
        self._initialize_ensemble()

        histories = []

        for i, model in enumerate(self.ensemble):
            print(f"\nTraining ensemble member {i + 1}/{self.settings.ensemble_size}")

            # Temporarily set self.model to current ensemble member
            original_model = self.model
            self.model = model
            self.optimizer = None  # Reset optimizer for new model

            # Train
            history = self.train(X_train, y_train, X_val, y_val, save_best=False)
            histories.append(history)

            # Restore original model
            self.model = original_model

        return histories

    def save_model(self, path: Path | str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Cannot save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.get_model().state_dict(),
            "settings": self.settings.model_dump(),
            "architecture": self.settings.architecture.value,
        }

        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        self._torch.save(checkpoint, path)

    def load_model(self, path: Path | str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If loading fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self._lazy_load_dependencies()
        self._initialize_model()

        try:
            checkpoint = self._torch.load(path, map_location=self.device)
            self.model.get_model().load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    def cleanup(self) -> None:
        """Clean up models to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        for model in self.ensemble:
            del model
        self.ensemble = []

        if self._torch is not None and hasattr(self._torch.cuda, "empty_cache"):
            self._torch.cuda.empty_cache()
