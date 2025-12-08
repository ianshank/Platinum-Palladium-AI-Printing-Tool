"""
PyTorch neural network models for tone curve prediction.

This module provides various neural network architectures for learning
tone curves from calibration data:

- CurveMLP: Simple MLP that predicts curve control points from metadata
- CurveCNN: CNN-based model for LUT prediction
- ContentAwareCurveNet: U-Net style network for local curve adjustments
- UniformityCorrectionNet: Network for spatial non-uniformity correction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ptpd_calibration.config import DeepLearningSettings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder classes for type checking
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclass
class ModelOutput:
    """Output from curve prediction models."""

    curve: np.ndarray  # The predicted curve as numpy array
    control_points: Optional[np.ndarray] = None  # Optional control points
    uncertainty: Optional[float] = None  # Prediction uncertainty
    metadata: Optional[dict] = None  # Additional output metadata


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install ptpd-calibration[deep]"
        )


class MonotonicLayer(nn.Module):
    """Layer that enforces monotonicity via cumulative sum."""

    def __init__(self, normalize: bool = True):
        """
        Initialize monotonic layer.

        Args:
            normalize: Whether to normalize output to [0, 1] range.
        """
        _check_torch()
        super().__init__()
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply monotonicity constraint.

        Args:
            x: Input tensor of shape (batch, num_points).

        Returns:
            Monotonically increasing tensor.
        """
        # Apply softplus to ensure positive increments
        positive = F.softplus(x)
        # Cumulative sum ensures monotonicity
        cumsum = torch.cumsum(positive, dim=-1)

        if self.normalize:
            # Normalize to [0, 1]
            min_val = cumsum[..., :1]
            max_val = cumsum[..., -1:]
            # Avoid division by zero
            range_val = torch.clamp(max_val - min_val, min=1e-6)
            cumsum = (cumsum - min_val) / range_val

        return cumsum


class InterpolationLayer(nn.Module):
    """Layer that interpolates control points to full LUT."""

    def __init__(self, num_control_points: int, lut_size: int):
        """
        Initialize interpolation layer.

        Args:
            num_control_points: Number of input control points.
            lut_size: Size of output LUT.
        """
        _check_torch()
        super().__init__()
        self.num_control_points = num_control_points
        self.lut_size = lut_size

        # Pre-compute interpolation indices
        control_x = torch.linspace(0, 1, num_control_points)
        output_x = torch.linspace(0, 1, lut_size)
        self.register_buffer("control_x", control_x)
        self.register_buffer("output_x", output_x)

    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Interpolate control points to full LUT.

        Args:
            control_points: Shape (batch, num_control_points).

        Returns:
            Interpolated LUT of shape (batch, lut_size).
        """
        batch_size = control_points.shape[0]
        device = control_points.device

        # Use linear interpolation
        output = torch.zeros(batch_size, self.lut_size, device=device)

        for i in range(self.lut_size):
            x = self.output_x[i]
            # Find surrounding control points
            idx_right = torch.searchsorted(self.control_x, x)
            idx_right = torch.clamp(idx_right, 1, self.num_control_points - 1)
            idx_left = idx_right - 1

            # Linear interpolation
            x_left = self.control_x[idx_left]
            x_right = self.control_x[idx_right]
            t = (x - x_left) / (x_right - x_left + 1e-6)

            y_left = control_points[:, idx_left]
            y_right = control_points[:, idx_right]
            output[:, i] = y_left + t * (y_right - y_left)

        return output


class CurveMLP(nn.Module):
    """
    Multi-layer perceptron for curve prediction from process metadata.

    Predicts control points of a tone curve from process parameters
    (paper type, chemistry, exposure, humidity, etc.).
    """

    def __init__(
        self,
        num_features: int,
        num_control_points: int = 16,
        lut_size: int = 256,
        hidden_dims: Optional[list[int]] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Initialize CurveMLP.

        Args:
            num_features: Number of input features.
            num_control_points: Number of curve control points.
            lut_size: Size of output LUT.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
        """
        _check_torch()
        super().__init__()

        self.num_features = num_features
        self.num_control_points = num_control_points
        self.lut_size = lut_size

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        # Build encoder layers
        layers: list[nn.Module] = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Control point prediction head
        self.control_head = nn.Linear(in_dim, num_control_points)

        # Monotonicity constraint
        self.monotonic = MonotonicLayer(normalize=True)

        # Interpolation to full LUT
        self.interpolate = InterpolationLayer(num_control_points, lut_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_control_points: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, num_features).
            return_control_points: Whether to return control points.

        Returns:
            Tuple of (LUT, control_points or None).
        """
        # Encode features
        features = self.encoder(x)

        # Predict control points
        control_points = self.control_head(features)

        # Enforce monotonicity
        control_points = self.monotonic(control_points)

        # Interpolate to full LUT
        lut = self.interpolate(control_points)

        if return_control_points:
            return lut, control_points
        return lut, None

    @classmethod
    def from_settings(
        cls, num_features: int, settings: DeepLearningSettings
    ) -> "CurveMLP":
        """
        Create CurveMLP from settings.

        Args:
            num_features: Number of input features.
            settings: DeepLearningSettings configuration.

        Returns:
            Configured CurveMLP instance.
        """
        return cls(
            num_features=num_features,
            num_control_points=settings.num_control_points,
            lut_size=settings.lut_size,
            hidden_dims=settings.hidden_dims,
            dropout_rate=settings.dropout_rate,
            use_batch_norm=settings.use_batch_norm,
        )


class CurveCNN(nn.Module):
    """
    CNN-based model for LUT prediction.

    Uses 1D convolutions to capture local relationships in the curve.
    """

    def __init__(
        self,
        num_features: int,
        lut_size: int = 256,
        channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize CurveCNN.

        Args:
            num_features: Number of input features.
            lut_size: Size of output LUT.
            channels: List of channel dimensions for conv layers.
            kernel_size: Kernel size for convolutions.
            dropout_rate: Dropout rate.
        """
        _check_torch()
        super().__init__()

        self.num_features = num_features
        self.lut_size = lut_size

        if channels is None:
            channels = [64, 128, 64]

        # Feature projection to initial sequence
        self.feature_proj = nn.Linear(num_features, lut_size)

        # 1D conv layers
        conv_layers: list[nn.Module] = []
        in_channels = 1

        for out_channels in channels:
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Final projection
        self.output_proj = nn.Conv1d(in_channels, 1, 1)

        # Monotonicity
        self.monotonic = MonotonicLayer(normalize=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, num_features).

        Returns:
            Tuple of (LUT, None).
        """
        # Project features to sequence
        seq = self.feature_proj(x)  # (batch, lut_size)
        seq = seq.unsqueeze(1)  # (batch, 1, lut_size)

        # Apply convolutions
        seq = self.conv_layers(seq)

        # Project to single channel
        seq = self.output_proj(seq)  # (batch, 1, lut_size)
        seq = seq.squeeze(1)  # (batch, lut_size)

        # Enforce monotonicity
        lut = self.monotonic(seq)

        return lut, None


class ContentAwareCurveNet(nn.Module):
    """
    U-Net style network for content-aware local curve adjustments.

    Takes an image and produces a spatially-varying adjustment map
    that modifies the base curve per-pixel.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        adjustment_range: tuple[float, float] = (0.8, 1.2),
    ):
        """
        Initialize ContentAwareCurveNet.

        Args:
            in_channels: Number of input image channels.
            base_channels: Base number of channels.
            num_levels: Number of encoder/decoder levels.
            adjustment_range: Range for multiplicative adjustment.
        """
        _check_torch()
        super().__init__()

        self.adjustment_range = adjustment_range

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels

        for i in range(num_levels):
            out_ch = base_channels * (2**i)
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2**num_levels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )
        ch = bottleneck_ch

        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(num_levels - 1, -1, -1):
            skip_ch = base_channels * (2**i)
            self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(skip_ch * 2, skip_ch, 3, padding=1),
                    nn.BatchNorm2d(skip_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(skip_ch, skip_ch, 3, padding=1),
                    nn.BatchNorm2d(skip_ch),
                    nn.ReLU(inplace=True),
                )
            )
            ch = skip_ch

        # Output head
        self.output_head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image of shape (batch, channels, height, width).

        Returns:
            Adjustment map of shape (batch, 1, height, width).
        """
        # Encoder path
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skips = skips[::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            x = upconv(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output
        x = self.output_head(x)

        # Scale to adjustment range
        low, high = self.adjustment_range
        x = torch.sigmoid(x) * (high - low) + low

        return x


class UniformityCorrectionNet(nn.Module):
    """
    Network for spatial non-uniformity correction.

    Learns a smooth, low-frequency correction field to compensate
    for UV falloff, coating variations, and contact issues.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 31,
        sigma: float = 10.0,
        correction_range: tuple[float, float] = (0.8, 1.2),
    ):
        """
        Initialize UniformityCorrectionNet.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels.
            kernel_size: Kernel size for Gaussian smoothing.
            sigma: Sigma for Gaussian kernel.
            correction_range: Range for correction factor.
        """
        _check_torch()
        super().__init__()

        self.correction_range = correction_range

        # Small network for initial correction estimate
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

        # Gaussian smoothing for low-frequency constraint
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel())

    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        size = self.kernel_size
        x = torch.arange(size, dtype=torch.float32) - size // 2
        gauss_1d = torch.exp(-x**2 / (2 * self.sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image or chart of shape (batch, channels, height, width).

        Returns:
            Correction map of shape (batch, 1, height, width).
        """
        # Initial correction estimate
        correction = self.net(x)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        correction = F.pad(correction, (padding, padding, padding, padding), mode="reflect")
        correction = F.conv2d(correction, self.gaussian_kernel)

        # Scale to correction range
        low, high = self.correction_range
        correction = torch.sigmoid(correction) * (high - low) + low

        return correction

    def apply_correction(
        self, image: torch.Tensor, correction_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply correction map to image.

        Args:
            image: Input image tensor.
            correction_map: Correction map from forward pass.

        Returns:
            Corrected image.
        """
        return image * correction_map


def create_model(
    model_type: str,
    num_features: int,
    settings: DeepLearningSettings,
) -> nn.Module:
    """
    Factory function to create a model from settings.

    Args:
        model_type: Type of model to create.
        num_features: Number of input features.
        settings: DeepLearningSettings configuration.

    Returns:
        Configured model instance.
    """
    _check_torch()

    if model_type == "curve_mlp":
        return CurveMLP.from_settings(num_features, settings)
    elif model_type == "curve_cnn":
        return CurveCNN(
            num_features=num_features,
            lut_size=settings.lut_size,
            dropout_rate=settings.dropout_rate,
        )
    elif model_type == "content_aware":
        return ContentAwareCurveNet(
            adjustment_range=settings.uniformity_range,
        )
    elif model_type == "uniformity":
        return UniformityCorrectionNet(
            kernel_size=settings.uniformity_kernel_size,
            sigma=settings.uniformity_sigma,
            correction_range=settings.uniformity_range,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
