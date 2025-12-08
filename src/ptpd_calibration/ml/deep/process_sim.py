"""
Differentiable process simulator for alt-process printing.

Provides a learnable, differentiable model of the Pt/Pd (and other alt)
printing process, enabling end-to-end training of negative generation.

The simulator models:
- Paper characteristic curve (density vs. exposure)
- UV transmission through negative
- Process-specific color transforms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

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
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for process simulation. "
            "Install with: pip install ptpd-calibration[deep]"
        )


@dataclass
class ProcessParameters:
    """
    Physical parameters for alt-process printing.

    These represent the characteristic response of a specific
    paper/chemistry/UV combination.
    """

    gamma: float = 1.8
    dmin: float = 0.1
    dmax: float = 2.0
    shoulder_position: float = 0.85
    toe_position: float = 0.15
    contrast: float = 1.0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor."""
        _check_torch()
        tensor = torch.tensor(
            [
                self.gamma,
                self.dmin,
                self.dmax,
                self.shoulder_position,
                self.toe_position,
                self.contrast,
            ],
            dtype=torch.float32,
        )
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ProcessParameters":
        """Create from tensor."""
        _check_torch()
        values = tensor.detach().cpu().numpy()
        return cls(
            gamma=float(values[0]),
            dmin=float(values[1]),
            dmax=float(values[2]),
            shoulder_position=float(values[3]),
            toe_position=float(values[4]),
            contrast=float(values[5]),
        )


class CharacteristicCurve(nn.Module):
    """
    Differentiable characteristic curve model.

    Models the relationship between UV exposure and print density
    using a parametric curve with controllable shape.
    """

    def __init__(
        self,
        gamma: float = 1.8,
        dmin: float = 0.1,
        dmax: float = 2.0,
        learnable: bool = True,
    ):
        """
        Initialize CharacteristicCurve.

        Args:
            gamma: Initial gamma value.
            dmin: Minimum density (paper base).
            dmax: Maximum density.
            learnable: Whether parameters are learnable.
        """
        _check_torch()
        super().__init__()

        # Store initial values for reference
        self._initial_gamma = gamma
        self._initial_dmin = dmin
        self._initial_dmax = dmax

        if learnable:
            # Learnable parameters (in unconstrained space)
            self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma)))
            self.dmin_raw = nn.Parameter(torch.tensor(dmin))
            self.dmax_raw = nn.Parameter(torch.tensor(dmax))
            self.shoulder = nn.Parameter(torch.tensor(0.0))  # Sigmoid space
            self.toe = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed parameters
            self.register_buffer("log_gamma", torch.tensor(np.log(gamma)))
            self.register_buffer("dmin_raw", torch.tensor(dmin))
            self.register_buffer("dmax_raw", torch.tensor(dmax))
            self.register_buffer("shoulder", torch.tensor(0.0))
            self.register_buffer("toe", torch.tensor(0.0))

    @property
    def gamma(self) -> torch.Tensor:
        """Get gamma (always positive)."""
        return torch.exp(self.log_gamma)

    @property
    def dmin(self) -> torch.Tensor:
        """Get Dmin (clamped to valid range)."""
        return torch.clamp(self.dmin_raw, 0.0, 0.5)

    @property
    def dmax(self) -> torch.Tensor:
        """Get Dmax (clamped to valid range)."""
        return torch.clamp(self.dmax_raw, min=self.dmin + 0.5).clamp(max=4.0)

    def forward(self, exposure: torch.Tensor) -> torch.Tensor:
        """
        Compute density from exposure.

        Args:
            exposure: Relative exposure values (0-1).

        Returns:
            Print density values.
        """
        # Apply gamma
        response = torch.pow(torch.clamp(exposure, 1e-6, 1.0), self.gamma)

        # Apply shoulder compression (high values)
        shoulder_strength = torch.sigmoid(self.shoulder) * 0.5
        response = response - shoulder_strength * torch.pow(
            torch.clamp(response - 0.5, 0, 0.5), 2
        )

        # Apply toe expansion (low values)
        toe_strength = torch.sigmoid(self.toe) * 0.3
        response = response + toe_strength * torch.pow(
            torch.clamp(0.3 - response, 0, 0.3), 2
        )

        # Scale to density range
        density = self.dmin + (self.dmax - self.dmin) * response

        return density


class ProcessSimulator(nn.Module):
    """
    Differentiable simulator for alt-process printing.

    Models the complete chain from digital negative to print density:
    1. Negative density to UV transmission
    2. Paper exposure from UV transmission
    3. Paper characteristic curve response
    4. Final print density
    """

    def __init__(
        self,
        settings: Optional[DeepLearningSettings] = None,
        learnable: bool = True,
    ):
        """
        Initialize ProcessSimulator.

        Args:
            settings: Deep learning settings with process parameters.
            learnable: Whether process parameters are learnable.
        """
        _check_torch()
        super().__init__()

        from ptpd_calibration.config import get_settings

        if settings is None:
            settings = get_settings().deep_learning

        # Paper characteristic curve
        self.characteristic = CharacteristicCurve(
            gamma=settings.process_gamma_init,
            dmin=settings.process_dmin_init,
            dmax=settings.process_dmax_init,
            learnable=learnable,
        )

        # UV source non-linearity (optional)
        if learnable:
            self.uv_gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("uv_gamma", torch.tensor(1.0))

        # Store settings for reference
        self._settings = settings

    def negative_to_transmission(self, negative_density: torch.Tensor) -> torch.Tensor:
        """
        Convert negative density to UV transmission.

        Uses Beer-Lambert law: T = 10^(-D)

        Args:
            negative_density: Density values of the negative.

        Returns:
            Transmission values (0-1).
        """
        return torch.pow(torch.tensor(10.0), -negative_density)

    def transmission_to_exposure(self, transmission: torch.Tensor) -> torch.Tensor:
        """
        Convert transmission to relative paper exposure.

        Args:
            transmission: UV transmission through negative.

        Returns:
            Relative exposure (0-1).
        """
        # Apply UV source gamma
        uv_gamma = torch.clamp(self.uv_gamma, 0.5, 2.0)
        return torch.pow(torch.clamp(transmission, 1e-6, 1.0), uv_gamma)

    def forward(
        self,
        negative_density: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Simulate the complete printing process.

        Args:
            negative_density: Density values of the digital negative.
            return_intermediates: Whether to return intermediate values.

        Returns:
            Print density (and optionally intermediate values).
        """
        # Step 1: Negative density to transmission
        transmission = self.negative_to_transmission(negative_density)

        # Step 2: Transmission to exposure
        exposure = self.transmission_to_exposure(transmission)

        # Step 3: Exposure to print density
        print_density = cast(torch.Tensor, self.characteristic(exposure))

        if return_intermediates:
            intermediates = {
                "transmission": transmission,
                "exposure": exposure,
                "print_density": print_density,
            }
            return print_density, intermediates

        return print_density

    def invert(
        self,
        target_density: torch.Tensor,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
    ) -> torch.Tensor:
        """
        Find negative density that produces target print density.

        Uses gradient descent to invert the process.

        Args:
            target_density: Desired print density.
            num_iterations: Number of optimization iterations.
            learning_rate: Learning rate for optimization.

        Returns:
            Negative density values.
        """
        # Initialize with a reasonable starting point
        # Invert the target through approximate characteristic
        dmin = self.characteristic.dmin.detach()
        dmax = self.characteristic.dmax.detach()
        gamma = self.characteristic.gamma.detach()

        # Normalize target to 0-1
        target_norm = (target_density - dmin) / (dmax - dmin + 1e-6)
        target_norm = torch.clamp(target_norm, 0.0, 1.0)

        # Initial estimate (inverse gamma approximation)
        init_exposure = torch.pow(target_norm + 1e-6, 1.0 / gamma)
        init_transmission = init_exposure  # Simplified
        init_negative = -torch.log10(init_transmission + 1e-6)

        # Optimize
        negative = init_negative.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([negative], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            simulated = cast(torch.Tensor, self.forward(negative))
            loss = F.mse_loss(simulated, target_density)
            loss.backward()
            optimizer.step()

            # Clamp to valid range
            with torch.no_grad():
                negative.clamp_(0.0, 4.0)

        return negative.detach()

    def get_parameters(self) -> ProcessParameters:
        """Get current process parameters."""
        return ProcessParameters(
            gamma=float(self.characteristic.gamma.item()),
            dmin=float(self.characteristic.dmin.item()),
            dmax=float(self.characteristic.dmax.item()),
            shoulder_position=float(torch.sigmoid(self.characteristic.shoulder).item()),
            toe_position=float(torch.sigmoid(self.characteristic.toe).item()),
        )

    def set_parameters(self, params: ProcessParameters) -> None:
        """Set process parameters."""
        with torch.no_grad():
            self.characteristic.log_gamma.fill_(np.log(params.gamma))
            self.characteristic.dmin_raw.fill_(params.dmin)
            self.characteristic.dmax_raw.fill_(params.dmax)
            # Convert shoulder/toe from 0-1 to logit space
            self.characteristic.shoulder.fill_(
                np.log(params.shoulder_position / (1 - params.shoulder_position + 1e-6))
            )
            self.characteristic.toe.fill_(
                np.log(params.toe_position / (1 - params.toe_position + 1e-6))
            )


class NegativeGenerator(nn.Module):
    """
    End-to-end negative generator using process simulation.

    Takes a desired print appearance and generates the digital negative
    needed to achieve it, using a differentiable process model.
    """

    def __init__(
        self,
        curve_model: nn.Module,
        process_sim: ProcessSimulator,
    ):
        """
        Initialize NegativeGenerator.

        Args:
            curve_model: Model that predicts tone curves.
            process_sim: Differentiable process simulator.
        """
        _check_torch()
        super().__init__()
        self.curve_model = curve_model
        self.process_sim = process_sim

    def forward(
        self,
        target_image: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative for target print appearance.

        Args:
            target_image: Desired print appearance (density values).
            metadata: Process metadata features.

        Returns:
            Tuple of (negative_density, predicted_curve).
        """
        # Get curve from metadata
        curve, _ = self.curve_model(metadata)

        # Apply curve to get negative density
        # curve maps input (0-1) to output (0-1)
        # We need to map target density to negative density

        # Normalize target to 0-1
        target_min = target_image.min(dim=-1, keepdim=True)[0]
        target_max = target_image.max(dim=-1, keepdim=True)[0]
        target_norm = (target_image - target_min) / (target_max - target_min + 1e-6)

        # Look up in curve (interpolate)
        # curve is (batch, lut_size), target_norm is (batch, image_size)
        # Use grid_sample for differentiable lookup
        curve_expanded = curve.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, lut_size)
        grid_x = target_norm * 2 - 1  # Convert to [-1, 1]
        grid_x = grid_x.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, image_size)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        negative_norm = F.grid_sample(
            curve_expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        negative_norm = negative_norm.squeeze(1).squeeze(1)

        # Scale to density range (typical negative is 0-2.5)
        negative_density = negative_norm * 2.5

        return negative_density, curve

    def compute_loss(
        self,
        target_image: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute end-to-end loss.

        Args:
            target_image: Desired print appearance.
            metadata: Process metadata.

        Returns:
            Tuple of (loss, loss_components).
        """
        # Generate negative
        negative, curve = self.forward(target_image, metadata)

        # Simulate print
        simulated_print = cast(torch.Tensor, self.process_sim(negative))

        # Compute loss
        reconstruction_loss = F.mse_loss(simulated_print, target_image)

        # Curve smoothness
        curve_diffs = curve[:, 1:] - curve[:, :-1]
        smoothness_loss = torch.mean(torch.abs(curve_diffs[:, 1:] - curve_diffs[:, :-1]))

        # Total loss
        total_loss = reconstruction_loss + 0.01 * smoothness_loss

        components = {
            "reconstruction": float(reconstruction_loss.item()),
            "smoothness": float(smoothness_loss.item()),
        }

        return total_loss, components
