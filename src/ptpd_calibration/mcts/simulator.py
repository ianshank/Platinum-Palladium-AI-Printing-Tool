"""
Extended process simulator for MCTS calibration search.

Wraps the existing ProcessSimulator/CharacteristicCurve and adds
parameter-to-process-characteristics mapping for the full calibration space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ptpd_calibration.mcts.config import MCTSSettings, PhysicsConstants
from ptpd_calibration.mcts.types import SimulationResult

# Try to import torch, but provide numpy fallback
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


# ProcessParameters is a simple dataclass - define locally to avoid circular import issues
# when torch is not available (process_sim.py tries to use nn.Module which is None)
@dataclass
class ProcessParameters:
    """Physical parameters for alt-process printing."""

    gamma: float = 1.8
    dmin: float = 0.1
    dmax: float = 2.0
    shoulder_position: float = 0.85
    toe_position: float = 0.15
    contrast: float = 1.0


# CharacteristicCurve requires PyTorch — import conditionally
_CharacteristicCurve: type | None = None
if TORCH_AVAILABLE:
    try:
        from ptpd_calibration.ml.deep.process_sim import (
            CharacteristicCurve as _CharacteristicCurve,  # type: ignore[assignment, misc, no-redef]
        )
    except (ImportError, AttributeError):
        # If import fails (e.g., due to nn being None), stay with None
        _CharacteristicCurve = None

logger = logging.getLogger(__name__)


class ExtendedProcessSimulator:
    """Extended process simulator mapping calibration parameters to density curves.

    This class bridges the gap between high-level calibration parameters
    (metal_ratio, coating_weight, etc.) and the low-level process parameters
    (gamma, dmin, dmax, etc.) used by the existing ProcessSimulator.

    Uses configurable physics models to compute the mapping, with both
    PyTorch (differentiable) and NumPy (fallback) implementations.
    """

    def __init__(
        self,
        physics: PhysicsConstants | None = None,
        settings: MCTSSettings | None = None,
    ):
        """Initialize ExtendedProcessSimulator.

        Args:
            physics: Physics model parameters. If None, uses defaults.
            settings: MCTS settings. If None, uses defaults.
        """
        self.physics = physics or PhysicsConstants()
        self.settings = settings or MCTSSettings()
        self.use_torch = TORCH_AVAILABLE

        logger.debug(
            f"Initialized ExtendedProcessSimulator (torch={'available' if self.use_torch else 'unavailable'})"
        )

    def compute_process_parameters(
        self,
        params: dict[str, float],
    ) -> ProcessParameters:
        """Map calibration parameters to ProcessParameters using physics models.

        Args:
            params: Dictionary with keys: metal_ratio, coating_weight,
                   ferric_oxalate_pct, exposure_time, developer_temp, humidity

        Returns:
            ProcessParameters for use with CharacteristicCurve
        """
        # Extract parameters with defaults
        metal_ratio = params.get("metal_ratio", 0.5)
        coating_weight = params.get("coating_weight", 1.5)
        ferric_oxalate_pct = params.get("ferric_oxalate_pct", 20.0)
        exposure_time = params.get("exposure_time", 180.0)
        developer_temp = params.get("developer_temp", 25.0)
        humidity = params.get("humidity", 50.0)

        # 1. Gamma: interpolate between Pt and Pd gamma based on metal_ratio
        # metal_ratio = 1.0 -> pure Pt -> pt_gamma_base
        # metal_ratio = 0.0 -> pure Pd -> pd_gamma_base
        gamma = (
            metal_ratio * self.physics.pt_gamma_base
            + (1.0 - metal_ratio) * self.physics.pd_gamma_base
        )

        # 2. Dmin: base paper dmin (configurable)
        dmin = self.physics.paper_dmin_base

        # 3. Dmax: function of coating_weight, exposure_time, chemistry
        # Both coating weight and exposure contribute multiplicatively
        # Coating weight provides the base capacity
        coating_factor = min(
            self.physics.coating_weight_dmax_slope * coating_weight,
            self.physics.coating_weight_dmax_ceiling - dmin,
        )

        # Exposure determines how much of that capacity is realized
        # Using saturating exponential: factor = 1 - exp(-t/halflife)
        exposure_factor = 1.0 - np.exp(-exposure_time / self.physics.exposure_dmax_halflife)

        # Combined: exposure_factor scales the coating capacity
        # This ensures both matter: low coating OR low exposure -> low dmax
        dmax_gain = coating_factor * exposure_factor

        # Final dmax
        dmax = dmin + dmax_gain
        dmax = min(dmax, self.physics.exposure_dmax_ceiling)

        # 4. Contrast: function of FO% deviation from optimal
        # More FO -> higher contrast (up to a point)
        fo_deviation = ferric_oxalate_pct - self.physics.fo_contrast_center
        contrast_adjustment = 1.0 + self.physics.fo_contrast_slope * fo_deviation
        contrast = max(
            self.physics.contrast_min, min(self.physics.contrast_max, contrast_adjustment)
        )

        # 5. Shoulder position: affected by exposure and development
        # Higher exposure -> more defined shoulder
        # Temperature affects development rate, which impacts shoulder
        temp_deviation = developer_temp - self.physics.dev_temp_reference
        shoulder_adjustment = temp_deviation * self.physics.shoulder_temp_sensitivity
        shoulder_position = np.clip(
            self.physics.shoulder_base + shoulder_adjustment,
            0.5,
            1.0,
        )

        # 6. Toe position: affected by humidity and coating uniformity
        # Humidity away from optimal reduces uniformity, affects toe
        humidity_deviation = abs(humidity - self.physics.humidity_optimal)
        toe_adjustment = humidity_deviation * self.physics.humidity_uniformity_slope
        toe_position = np.clip(
            self.physics.toe_base + toe_adjustment,
            0.0,
            0.5,
        )

        logger.debug(
            f"Computed ProcessParameters: gamma={gamma:.3f}, dmin={dmin:.3f}, "
            f"dmax={dmax:.3f}, contrast={contrast:.3f}, "
            f"shoulder={shoulder_position:.3f}, toe={toe_position:.3f}"
        )

        return ProcessParameters(
            gamma=float(gamma),
            dmin=float(dmin),
            dmax=float(dmax),
            shoulder_position=float(shoulder_position),
            toe_position=float(toe_position),
            contrast=float(contrast),
        )

    def simulate(
        self,
        params: dict[str, float],
        num_steps: int = 21,
    ) -> SimulationResult:
        """Run full simulation and return density curve.

        Args:
            params: Calibration parameters dictionary
            num_steps: Number of steps in the density curve

        Returns:
            SimulationResult with density curve and metrics
        """
        if self.use_torch:
            return self.simulate_with_torch(params, num_steps)
        else:
            return self.simulate_with_numpy(params, num_steps)

    def simulate_with_torch(
        self,
        params: dict[str, float],
        num_steps: int = 21,
    ) -> SimulationResult:
        """PyTorch-based simulation (differentiable).

        Args:
            params: Calibration parameters dictionary
            num_steps: Number of steps in the density curve

        Returns:
            SimulationResult with density curve and metrics
        """
        if not self.use_torch or _CharacteristicCurve is None:
            raise RuntimeError("PyTorch not available, use simulate_with_numpy instead")

        # 1. Compute ProcessParameters from params
        process_params = self.compute_process_parameters(params)

        # 2. Create characteristic curve
        curve = _CharacteristicCurve(
            gamma=process_params.gamma,
            dmin=process_params.dmin,
            dmax=process_params.dmax,
            learnable=False,
        )

        # Set shoulder and toe
        with torch.no_grad():
            curve.shoulder.fill_(
                np.log(
                    process_params.shoulder_position
                    / (1.0 - process_params.shoulder_position + 1e-6)
                )
            )
            curve.toe.fill_(
                np.log(process_params.toe_position / (1.0 - process_params.toe_position + 1e-6))
            )

        # 3. Generate exposure values (0 to 1, num_steps points)
        exposure = torch.linspace(0.0, 1.0, num_steps)

        # 4. Apply characteristic curve to get densities
        with torch.no_grad():
            densities = curve(exposure)
            density_list = densities.cpu().numpy().tolist()

        # 5. Extract metrics
        dmin_actual = float(min(density_list))
        dmax_actual = float(max(density_list))
        density_range = dmax_actual - dmin_actual

        logger.debug(
            f"Torch simulation: {num_steps} steps, "
            f"dmin={dmin_actual:.3f}, dmax={dmax_actual:.3f}, "
            f"gamma={process_params.gamma:.3f}"
        )

        return SimulationResult(
            density_curve=density_list,
            dmin=dmin_actual,
            dmax=dmax_actual,
            density_range=density_range,
            gamma=process_params.gamma,
            quality_score=0.0,  # Will be computed by QualityScorer
            parameters=params,
            constraint_violations=[],
        )

    def simulate_with_numpy(
        self,
        params: dict[str, float],
        num_steps: int = 21,
    ) -> SimulationResult:
        """NumPy fallback simulation (not differentiable).

        Args:
            params: Calibration parameters dictionary
            num_steps: Number of steps in the density curve

        Returns:
            SimulationResult with density curve and metrics
        """
        # 1. Compute ProcessParameters from params
        process_params = self.compute_process_parameters(params)

        # 2. Generate exposure values (0 to 1, num_steps points)
        exposure = np.linspace(0.0, 1.0, num_steps)

        # 3. Apply characteristic curve logic (replicate CharacteristicCurve.forward)
        densities = self._numpy_characteristic_curve(
            exposure,
            gamma=process_params.gamma,
            dmin=process_params.dmin,
            dmax=process_params.dmax,
            shoulder=process_params.shoulder_position,
            toe=process_params.toe_position,
        )

        density_list = densities.tolist()

        # 4. Extract metrics
        dmin_actual = float(min(density_list))
        dmax_actual = float(max(density_list))
        density_range = dmax_actual - dmin_actual

        logger.debug(
            f"NumPy simulation: {num_steps} steps, "
            f"dmin={dmin_actual:.3f}, dmax={dmax_actual:.3f}, "
            f"gamma={process_params.gamma:.3f}"
        )

        return SimulationResult(
            density_curve=density_list,
            dmin=dmin_actual,
            dmax=dmax_actual,
            density_range=density_range,
            gamma=process_params.gamma,
            quality_score=0.0,  # Will be computed by QualityScorer
            parameters=params,
            constraint_violations=[],
        )

    def _numpy_characteristic_curve(
        self,
        exposure: np.ndarray,
        gamma: float,
        dmin: float,
        dmax: float,
        shoulder: float,
        toe: float,
    ) -> np.ndarray:
        """NumPy implementation of CharacteristicCurve.forward logic.

        Replicates the exact math of CharacteristicCurve.forward() including
        the dmin/dmax property clamping and shoulder/toe sigmoid scaling.

        Args:
            exposure: Exposure values (0-1)
            gamma: Gamma value
            dmin: Minimum density
            dmax: Maximum density
            shoulder: Shoulder position (0-1)
            toe: Toe position (0-1)

        Returns:
            Density values
        """
        # Apply same dmin/dmax clamping as CharacteristicCurve properties
        # See CharacteristicCurve.dmin and CharacteristicCurve.dmax properties
        dmin = float(np.clip(dmin, 0.0, 0.5))
        dmax = float(np.clip(max(dmax, dmin + 0.5), None, 4.0))

        # Apply gamma (power law response)
        response = np.power(np.clip(exposure, 1e-6, 1.0), gamma)

        # Apply shoulder compression (high values)
        # PyTorch path: simulate_with_torch fills buffer with logit(shoulder_position),
        # then forward() computes sigmoid(buffer) * 0.5.
        # sigmoid(logit(x)) = x, so shoulder_strength = shoulder * 0.5
        shoulder_raw = np.log(shoulder / (1.0 - shoulder + 1e-6))
        shoulder_strength = 1.0 / (1.0 + np.exp(-shoulder_raw)) * 0.5
        response = response - shoulder_strength * np.power(np.clip(response - 0.5, 0, 0.5), 2)

        # Apply toe expansion (low values)
        # Same pattern: sigmoid(logit(toe_position)) * 0.3
        toe_raw = np.log(toe / (1.0 - toe + 1e-6))
        toe_strength = 1.0 / (1.0 + np.exp(-toe_raw)) * 0.3
        response = response + toe_strength * np.power(np.clip(0.3 - response, 0, 0.3), 2)

        # Scale to density range
        density: np.ndarray = dmin + (dmax - dmin) * response

        return density
