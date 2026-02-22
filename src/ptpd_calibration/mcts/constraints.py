"""
Neuro-symbolic constraints for MCTS calibration pruning.

This module provides photochemistry-aware constraints for validating and pruning
actions in the MCTS search tree. Unlike the base constraints that operate on
density curve values, these constraints operate on calibration parameters
(metal_ratio, coating_weight, ferric_oxalate_pct, etc.).

Physics background:
- Metal ratio (Pt/Pd) affects contrast and tone
- Ferric oxalate concentration affects sensitivity
- Exposure time affects maximum density
- Developer temperature affects development rate
- Coating weight affects maximum density
- Humidity affects coating uniformity

Each constraint encodes domain knowledge from sensitometry, photographic
chemistry, and empirical calibration data to guide the MCTS search toward
physically plausible parameter combinations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, PhysicsConstants
from ptpd_calibration.neuro_symbolic.constraints import (
    ConstraintResult,
    ConstraintType,
    ConstraintViolation,
    SymbolicConstraint,
)

if TYPE_CHECKING:
    from ptpd_calibration.mcts.config import MCTSSettings, ParameterRange
    from ptpd_calibration.mcts.types import CalibrationAction, CalibrationState

logger = logging.getLogger(__name__)


class ParameterBoundsConstraint(SymbolicConstraint):
    """Hard bounds constraint for a single calibration parameter.

    Validates that a parameter value is within its defined ParameterRange.
    This is a hard constraint - values outside bounds are invalid.

    Physics rationale: Parameter ranges are defined by physical limits
    (e.g., cannot have negative concentrations, equipment has min/max values).
    """

    def __init__(
        self,
        parameter_name: str,
        parameter_range: ParameterRange | None = None,
        weight: float = 10.0,
    ):
        """Initialize parameter bounds constraint.

        Args:
            parameter_name: Name of parameter to constrain
            parameter_range: Range definition (defaults to DEFAULT_PARAMETER_RANGES)
            weight: Constraint weight (high for hard bounds)
        """
        super().__init__(
            name=f"Parameter Bounds: {parameter_name}",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.parameter_name = parameter_name

        if parameter_range is not None:
            self.range: ParameterRange = parameter_range
        elif parameter_name in DEFAULT_PARAMETER_RANGES:
            self.range = DEFAULT_PARAMETER_RANGES[parameter_name]
        else:
            logger.warning(
                f"No parameter range found for '{parameter_name}' in DEFAULT_PARAMETER_RANGES"
            )
            # Create a wide default range to prevent crashes
            from ptpd_calibration.mcts.config import ParameterRange

            self.range = ParameterRange(
                name=parameter_name,
                min_value=0.0,
                max_value=1000.0,
                default_value=0.5,
            )

        logger.debug(
            f"Initialized bounds constraint for {parameter_name}: "
            f"[{self.range.min_value}, {self.range.max_value}]"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate parameter bounds constraint.

        Args:
            values: Single-element array with parameter value

        Returns:
            ConstraintResult with hard bounds check
        """
        if len(values) != 1:
            logger.warning(f"Expected single value for parameter constraint, got {len(values)}")
            return ConstraintResult(
                is_satisfied=False,
                loss_value=1000.0,
                explanation="Invalid input: expected single parameter value",
            )

        value = float(values[0])
        is_satisfied = self.range.min_value <= value <= self.range.max_value
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            violation_magnitude = 0.0
            description = ""

            if value < self.range.min_value:
                violation_magnitude = self.range.min_value - value
                description = (
                    f"{self.parameter_name}={value:.3f} below minimum "
                    f"{self.range.min_value:.3f} {self.range.unit}"
                )
            elif value > self.range.max_value:
                violation_magnitude = value - self.range.max_value
                description = (
                    f"{self.parameter_name}={value:.3f} above maximum "
                    f"{self.range.max_value:.3f} {self.range.unit}"
                )

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=violation_magnitude,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Use value in range [{self.range.min_value}, {self.range.max_value}]",
                )
            )

        explanation = (
            f"{self.parameter_name}={value:.3f} {self.range.unit} within bounds"
            if is_satisfied
            else f"{self.parameter_name}={value:.3f} {self.range.unit} violates bounds [{self.range.min_value}, {self.range.max_value}]"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute hard bounds penalty using squared distance.

        Args:
            values: Single-element array with parameter value

        Returns:
            Loss value (0 if within bounds, quadratic penalty outside)
        """
        if len(values) != 1:
            return 1000.0

        value = float(values[0])

        # Squared penalty for out-of-bounds values
        if value < self.range.min_value:
            return float((self.range.min_value - value) ** 2)
        elif value > self.range.max_value:
            return float((value - self.range.max_value) ** 2)
        else:
            return 0.0


class FOConcentrationConstraint(SymbolicConstraint):
    """Soft constraint for ferric oxalate concentration sweet spot.

    Ferric oxalate (FO) is the light-sensitive component. While the hard
    bounds are 15-27%, the practical sweet spot is narrower (18-24%).
    Values outside this range are valid but produce less predictable results.

    Physics rationale: Very low FO = low sensitivity, long exposures.
    Very high FO = uneven coating, higher cost, clearing issues.
    """

    def __init__(
        self,
        sweet_spot_min: float = 18.0,
        sweet_spot_max: float = 24.0,
        weight: float = 1.0,
    ):
        """Initialize FO concentration constraint.

        Args:
            sweet_spot_min: Lower bound of preferred range (%)
            sweet_spot_max: Upper bound of preferred range (%)
            weight: Constraint weight
        """
        super().__init__(
            name="FO Concentration Sweet Spot",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.sweet_spot_min = sweet_spot_min
        self.sweet_spot_max = sweet_spot_max
        self.hard_range = DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"]

        logger.debug(
            f"Initialized FO constraint: sweet spot [{sweet_spot_min}, {sweet_spot_max}]%, "
            f"hard bounds [{self.hard_range.min_value}, {self.hard_range.max_value}]%"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate FO concentration constraint.

        Args:
            values: Single-element array with FO percentage

        Returns:
            ConstraintResult with soft penalty for suboptimal values
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single FO% value",
            )

        fo_pct = float(values[0])
        is_satisfied = self.sweet_spot_min <= fo_pct <= self.sweet_spot_max
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if fo_pct < self.sweet_spot_min:
                description = (
                    f"FO {fo_pct:.1f}% below sweet spot minimum {self.sweet_spot_min}% "
                    f"(low sensitivity, longer exposures)"
                )
            else:
                description = (
                    f"FO {fo_pct:.1f}% above sweet spot maximum {self.sweet_spot_max}% "
                    f"(coating issues, higher cost)"
                )

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=loss,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Use FO% in range [{self.sweet_spot_min}, {self.sweet_spot_max}] for most predictable results",
                )
            )

        explanation = (
            f"FO {fo_pct:.1f}% within sweet spot [{self.sweet_spot_min}, {self.sweet_spot_max}]%"
            if is_satisfied
            else f"FO {fo_pct:.1f}% outside sweet spot (soft penalty)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute soft penalty for deviation from sweet spot.

        Args:
            values: Single-element array with FO percentage

        Returns:
            Loss value (0 in sweet spot, quadratic penalty outside)
        """
        if len(values) != 1:
            return 100.0

        fo_pct = float(values[0])

        # No penalty in sweet spot
        if self.sweet_spot_min <= fo_pct <= self.sweet_spot_max:
            return 0.0

        # Soft quadratic penalty outside sweet spot
        if fo_pct < self.sweet_spot_min:
            deviation = self.sweet_spot_min - fo_pct
        else:
            deviation = fo_pct - self.sweet_spot_max

        return float(deviation**2 * 0.1)  # Scale factor for soft penalty


class MetalRatioConstraint(SymbolicConstraint):
    """Soft constraint for metal ratio (Pt/Pd) extremes.

    Metal ratio affects tone and contrast:
    - Low ratio (more Pd): warmer tones, lower contrast
    - High ratio (more Pt): cooler tones, higher contrast
    - Extreme values (near 0 or 1) can be unpredictable

    Physics rationale: Pure metals (ratio=0 or ratio=1) are valid but
    less commonly used due to cost and working characteristics.
    Blends (0.3-0.7) are more typical.
    """

    def __init__(
        self,
        blend_min: float = 0.25,
        blend_max: float = 0.75,
        weight: float = 0.5,
    ):
        """Initialize metal ratio constraint.

        Args:
            blend_min: Lower bound of typical blend range
            blend_max: Upper bound of typical blend range
            weight: Constraint weight (low for soft guidance)
        """
        super().__init__(
            name="Metal Ratio Blending",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.blend_min = blend_min
        self.blend_max = blend_max
        self.hard_range = DEFAULT_PARAMETER_RANGES["metal_ratio"]

        logger.debug(
            f"Initialized metal ratio constraint: typical blend [{blend_min}, {blend_max}]"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate metal ratio constraint.

        Args:
            values: Single-element array with metal ratio (0=Pd, 1=Pt)

        Returns:
            ConstraintResult with soft penalty for extreme ratios
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single metal ratio value",
            )

        ratio = float(values[0])
        is_satisfied = self.blend_min <= ratio <= self.blend_max
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if ratio < self.blend_min:
                description = f"Metal ratio {ratio:.2f} is very Pd-heavy (warmer, lower contrast, less common)"
            else:
                description = f"Metal ratio {ratio:.2f} is very Pt-heavy (cooler, higher contrast, more expensive)"

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=loss,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Consider blend in range [{self.blend_min}, {self.blend_max}] for typical working characteristics",
                )
            )

        explanation = (
            f"Metal ratio {ratio:.2f} within typical blend range"
            if is_satisfied
            else f"Metal ratio {ratio:.2f} is extreme (soft penalty)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute soft penalty for extreme metal ratios.

        Args:
            values: Single-element array with metal ratio

        Returns:
            Loss value (0 in blend range, quadratic penalty for extremes)
        """
        if len(values) != 1:
            return 100.0

        ratio = float(values[0])

        # No penalty in typical blend range
        if self.blend_min <= ratio <= self.blend_max:
            return 0.0

        # Soft quadratic penalty for extreme ratios
        deviation = self.blend_min - ratio if ratio < self.blend_min else ratio - self.blend_max
        return float(deviation**2 * 0.5)  # Scale factor for soft penalty


class ExposureTimeConstraint(SymbolicConstraint):
    """Soft constraint for exposure time practicality.

    Exposure time affects maximum density and throughput:
    - Very short (<60s): may not reach full Dmax
    - Very long (>300s): diminishing returns, reduced throughput

    Physics rationale: UV exposure follows logarithmic response.
    Initial seconds give most density gain, later seconds give less.
    Extremely long exposures waste time for minimal density gain.
    """

    def __init__(
        self,
        practical_min: float = 60.0,
        practical_max: float = 300.0,
        weight: float = 0.3,
    ):
        """Initialize exposure time constraint.

        Args:
            practical_min: Minimum for reliable results (seconds)
            practical_max: Maximum before diminishing returns (seconds)
            weight: Constraint weight
        """
        super().__init__(
            name="Practical Exposure Time",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.practical_min = practical_min
        self.practical_max = practical_max
        self.hard_range = DEFAULT_PARAMETER_RANGES["exposure_time"]

        logger.debug(
            f"Initialized exposure constraint: practical [{practical_min}, {practical_max}]s"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate exposure time constraint.

        Args:
            values: Single-element array with exposure time (seconds)

        Returns:
            ConstraintResult with soft penalty for impractical times
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single exposure time value",
            )

        exposure_time = float(values[0])
        is_satisfied = self.practical_min <= exposure_time <= self.practical_max
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if exposure_time < self.practical_min:
                description = (
                    f"Exposure {exposure_time:.0f}s may not reach full Dmax (short for UV sources)"
                )
            else:
                description = f"Exposure {exposure_time:.0f}s has diminishing returns (long wait for small density gain)"

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=loss,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Use exposure in range [{self.practical_min}, {self.practical_max}]s for efficient workflow",
                )
            )

        explanation = (
            f"Exposure {exposure_time:.0f}s within practical range"
            if is_satisfied
            else f"Exposure {exposure_time:.0f}s is impractical (soft penalty)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute soft penalty for impractical exposure times.

        Args:
            values: Single-element array with exposure time

        Returns:
            Loss value (0 in practical range, quadratic penalty outside)
        """
        if len(values) != 1:
            return 100.0

        exposure_time = float(values[0])

        # No penalty in practical range
        if self.practical_min <= exposure_time <= self.practical_max:
            return 0.0

        # Soft quadratic penalty for impractical times
        if exposure_time < self.practical_min:
            deviation = (self.practical_min - exposure_time) / self.practical_min
        else:
            deviation = (exposure_time - self.practical_max) / self.practical_max

        return float(deviation**2 * 0.2)  # Scale factor for soft penalty


class DeveloperTemperatureConstraint(SymbolicConstraint):
    """Soft constraint for developer temperature deviation from reference.

    Developer temperature affects reaction rate:
    - Lower temps: slower development, more control
    - Higher temps: faster development, less control
    - Far from reference (25°C): less predictable results

    Physics rationale: Chemical reaction rates approximately double
    per 10°C increase. Deviation from calibration reference reduces
    predictability of results.
    """

    def __init__(
        self,
        physics_constants: PhysicsConstants | None = None,
        tolerance: float = 5.0,
        weight: float = 0.4,
    ):
        """Initialize developer temperature constraint.

        Args:
            physics_constants: Physics model (provides reference temp)
            tolerance: Acceptable deviation from reference (°C)
            weight: Constraint weight
        """
        super().__init__(
            name="Developer Temperature Reference",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.physics = physics_constants or PhysicsConstants()
        self.reference_temp = self.physics.dev_temp_reference
        self.tolerance = tolerance
        self.hard_range = DEFAULT_PARAMETER_RANGES["developer_temp"]

        logger.debug(
            f"Initialized developer temp constraint: reference {self.reference_temp}°C ± {tolerance}°C"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate developer temperature constraint.

        Args:
            values: Single-element array with temperature (°C)

        Returns:
            ConstraintResult with soft penalty for deviation
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single temperature value",
            )

        temp = float(values[0])
        deviation = abs(temp - self.reference_temp)
        is_satisfied = deviation <= self.tolerance
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if temp < self.reference_temp:
                description = f"Developer temp {temp:.1f}°C is {deviation:.1f}°C below reference (slower development)"
            else:
                description = f"Developer temp {temp:.1f}°C is {deviation:.1f}°C above reference (faster development)"

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=loss,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Use temperature within {self.tolerance}°C of reference ({self.reference_temp}°C) for predictable results",
                )
            )

        explanation = (
            f"Developer temp {temp:.1f}°C within {self.tolerance}°C of reference"
            if is_satisfied
            else f"Developer temp {temp:.1f}°C deviates {deviation:.1f}°C from reference (soft penalty)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute soft penalty for temperature deviation.

        Args:
            values: Single-element array with temperature

        Returns:
            Loss value (0 near reference, quadratic penalty for deviation)
        """
        if len(values) != 1:
            return 100.0

        temp = float(values[0])
        deviation = abs(temp - self.reference_temp)

        # No penalty within tolerance
        if deviation <= self.tolerance:
            return 0.0

        # Soft quadratic penalty for deviation beyond tolerance
        excess_deviation = deviation - self.tolerance
        return float((excess_deviation / self.tolerance) ** 2 * 0.3)


class CoatingWeightConstraint(SymbolicConstraint):
    """Hard constraint for coating weight (metal solution volume).

    Coating weight affects maximum density and evenness:
    - Too little: incomplete coverage, low Dmax
    - Too much: pooling, uneven drying, waste

    Physics rationale: Minimum needed for full paper coverage,
    maximum before surface tension breaks down and pooling occurs.
    """

    def __init__(
        self,
        weight: float = 5.0,
    ):
        """Initialize coating weight constraint.

        Args:
            weight: Constraint weight (high for important bounds)
        """
        super().__init__(
            name="Coating Weight Coverage",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.hard_range = DEFAULT_PARAMETER_RANGES["coating_weight"]

        logger.debug(
            f"Initialized coating weight constraint: "
            f"[{self.hard_range.min_value}, {self.hard_range.max_value}] {self.hard_range.unit}"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate coating weight constraint.

        Args:
            values: Single-element array with coating weight (ml/sq-inch)

        Returns:
            ConstraintResult with bounds check
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single coating weight value",
            )

        weight = float(values[0])
        is_satisfied = self.hard_range.min_value <= weight <= self.hard_range.max_value
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if weight < self.hard_range.min_value:
                description = f"Coating weight {weight:.2f} {self.hard_range.unit} insufficient for full coverage"
            else:
                description = (
                    f"Coating weight {weight:.2f} {self.hard_range.unit} excessive (pooling risk)"
                )

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=abs(loss),
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Use coating weight in range [{self.hard_range.min_value}, {self.hard_range.max_value}] {self.hard_range.unit}",
                )
            )

        explanation = (
            f"Coating weight {weight:.2f} {self.hard_range.unit} within bounds"
            if is_satisfied
            else f"Coating weight {weight:.2f} {self.hard_range.unit} violates bounds"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute hard penalty for coating weight out of bounds.

        Args:
            values: Single-element array with coating weight

        Returns:
            Loss value (0 within bounds, quadratic penalty outside)
        """
        if len(values) != 1:
            return 100.0

        weight = float(values[0])

        # Squared penalty for out-of-bounds values
        if weight < self.hard_range.min_value:
            return float((self.hard_range.min_value - weight) ** 2)
        elif weight > self.hard_range.max_value:
            return float((weight - self.hard_range.max_value) ** 2)
        else:
            return 0.0


class HumidityConstraint(SymbolicConstraint):
    """Soft constraint for humidity deviation from optimal.

    Humidity affects coating uniformity and drying:
    - Too low (<40%): fast drying, uneven coating
    - Too high (>60%): slow drying, mold risk
    - Optimal (50%): even coating, predictable drying

    Physics rationale: Surface tension and evaporation rate
    are humidity-dependent. Deviation from optimal reduces
    coating uniformity and result consistency.
    """

    def __init__(
        self,
        physics_constants: PhysicsConstants | None = None,
        tolerance: float = 10.0,
        weight: float = 0.5,
    ):
        """Initialize humidity constraint.

        Args:
            physics_constants: Physics model (provides optimal humidity)
            tolerance: Acceptable deviation from optimal (% RH)
            weight: Constraint weight
        """
        super().__init__(
            name="Optimal Humidity",
            constraint_type=ConstraintType.CUSTOM,
            weight=weight,
        )
        self.physics = physics_constants or PhysicsConstants()
        self.optimal_humidity = self.physics.humidity_optimal
        self.tolerance = tolerance
        self.hard_range = DEFAULT_PARAMETER_RANGES["humidity"]

        logger.debug(
            f"Initialized humidity constraint: optimal {self.optimal_humidity}% RH ± {tolerance}%"
        )

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate humidity constraint.

        Args:
            values: Single-element array with humidity (% RH)

        Returns:
            ConstraintResult with soft penalty for deviation
        """
        if len(values) != 1:
            return ConstraintResult(
                is_satisfied=False,
                loss_value=100.0,
                explanation="Invalid input: expected single humidity value",
            )

        humidity = float(values[0])
        deviation = abs(humidity - self.optimal_humidity)
        is_satisfied = deviation <= self.tolerance
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            if humidity < self.optimal_humidity:
                description = f"Humidity {humidity:.0f}% RH is {deviation:.0f}% below optimal (fast drying, uneven coating)"
            else:
                description = f"Humidity {humidity:.0f}% RH is {deviation:.0f}% above optimal (slow drying, mold risk)"

            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=loss,
                    violation_indices=[0],
                    description=description,
                    suggested_fix=f"Control humidity within {self.tolerance}% RH of optimal ({self.optimal_humidity}% RH)",
                )
            )

        explanation = (
            f"Humidity {humidity:.0f}% RH within {self.tolerance}% of optimal"
            if is_satisfied
            else f"Humidity {humidity:.0f}% RH deviates {deviation:.0f}% from optimal (soft penalty)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute soft penalty for humidity deviation.

        Args:
            values: Single-element array with humidity

        Returns:
            Loss value (0 near optimal, quadratic penalty for deviation)
        """
        if len(values) != 1:
            return 100.0

        humidity = float(values[0])
        deviation = abs(humidity - self.optimal_humidity)

        # No penalty within tolerance
        if deviation <= self.tolerance:
            return 0.0

        # Soft quadratic penalty for deviation beyond tolerance
        excess_deviation = deviation - self.tolerance
        return float((excess_deviation / self.tolerance) ** 2 * 0.4)


class ActionPruner:
    """Prunes and scores MCTS actions based on constraint satisfaction.

    The ActionPruner uses a set of constraints to filter out invalid
    actions (hard constraint violations) and score remaining actions
    by how well they satisfy soft constraints.

    This dramatically reduces the MCTS search space by eliminating
    physically implausible parameter combinations early.
    """

    def __init__(
        self,
        constraints: list[SymbolicConstraint] | None = None,
        settings: MCTSSettings | None = None,
    ):
        """Initialize action pruner.

        Args:
            constraints: List of constraints to apply (defaults to standard set)
            settings: MCTS settings
        """
        from ptpd_calibration.mcts.config import MCTSSettings

        self.settings = settings or MCTSSettings()
        self.constraints = constraints or self._create_default_constraints()

        logger.info(f"Initialized ActionPruner with {len(self.constraints)} constraints")

    def _create_default_constraints(self) -> list[SymbolicConstraint]:
        """Create default constraint set for MCTS parameter pruning.

        Returns:
            List of standard parameter constraints
        """
        physics = PhysicsConstants()

        constraints: list[SymbolicConstraint] = [
            # Hard bounds for all parameters
            ParameterBoundsConstraint("metal_ratio", weight=10.0),
            ParameterBoundsConstraint("coating_weight", weight=10.0),
            ParameterBoundsConstraint("ferric_oxalate_pct", weight=10.0),
            ParameterBoundsConstraint("exposure_time", weight=10.0),
            ParameterBoundsConstraint("developer_temp", weight=10.0),
            ParameterBoundsConstraint("humidity", weight=10.0),
            # Soft constraints for practical ranges
            FOConcentrationConstraint(weight=1.0),
            MetalRatioConstraint(weight=0.5),
            ExposureTimeConstraint(weight=0.3),
            DeveloperTemperatureConstraint(physics_constants=physics, weight=0.4),
            HumidityConstraint(physics_constants=physics, weight=0.5),
        ]

        logger.debug(f"Created {len(constraints)} default constraints for pruning")
        return constraints

    def prune_actions(
        self,
        state: CalibrationState,
        actions: list[CalibrationAction],
    ) -> list[CalibrationAction]:
        """Remove actions that violate hard constraints.

        Args:
            state: Current calibration state
            actions: List of candidate actions

        Returns:
            Filtered list with invalid actions removed
        """
        if not actions:
            logger.debug("No actions to prune")
            return []

        valid_actions = []
        pruned_count = 0

        for action in actions:
            # Check if action violates any hard constraint
            is_valid = True
            value_array = np.array([action.value])

            # Find constraints applicable to this action's dimension
            for constraint in self.constraints:
                # Only check parameter-specific constraints for this dimension
                if isinstance(constraint, ParameterBoundsConstraint):
                    if constraint.parameter_name == action.dimension:
                        result = constraint.evaluate(value_array)
                        if not result.is_satisfied:
                            logger.debug(
                                f"Pruned action {action.dimension}={action.value:.3f}: "
                                f"{result.explanation}"
                            )
                            is_valid = False
                            pruned_count += 1
                            break

            if is_valid:
                valid_actions.append(action)

        logger.debug(
            f"Pruned {pruned_count}/{len(actions)} actions, "
            f"kept {len(valid_actions)} valid actions"
        )

        return valid_actions

    def score_action(
        self,
        state: CalibrationState,
        action: CalibrationAction,
    ) -> float:
        """Compute constraint satisfaction score for an action.

        Higher scores indicate better constraint satisfaction.
        Score of 1.0 = all constraints satisfied.
        Score < 1.0 = some soft constraints violated.

        Args:
            state: Current calibration state
            action: Action to score

        Returns:
            Constraint satisfaction score in [0, 1]
        """
        value_array = np.array([action.value])

        total_weight = 0.0
        weighted_satisfaction = 0.0

        # Evaluate all constraints applicable to this dimension
        for constraint in self.constraints:
            # Check if constraint applies to this action's dimension
            applies = False

            if isinstance(constraint, ParameterBoundsConstraint):
                applies = constraint.parameter_name == action.dimension
            elif isinstance(constraint, FOConcentrationConstraint):
                applies = action.dimension == "ferric_oxalate_pct"
            elif isinstance(constraint, MetalRatioConstraint):
                applies = action.dimension == "metal_ratio"
            elif isinstance(constraint, ExposureTimeConstraint):
                applies = action.dimension == "exposure_time"
            elif isinstance(constraint, DeveloperTemperatureConstraint):
                applies = action.dimension == "developer_temp"
            elif isinstance(constraint, HumidityConstraint):
                applies = action.dimension == "humidity"
            elif isinstance(constraint, CoatingWeightConstraint):
                applies = action.dimension == "coating_weight"

            if not applies:
                continue

            # Evaluate constraint
            result = constraint.evaluate(value_array)

            # Convert satisfaction to score: 1.0 if satisfied, penalized if violated
            # Use exponential decay for soft penalties to differentiate violations
            # Small losses (~0.01) -> score ~0.99, Medium (~0.1) -> ~0.90
            # Large (~1.0) -> ~0.37, Very large (>2.0) -> near 0
            satisfaction_score = 1.0 if result.is_satisfied else float(np.exp(-result.loss_value))

            weighted_satisfaction += constraint.weight * satisfaction_score
            total_weight += constraint.weight

        # Normalize by total weight
        score = weighted_satisfaction / total_weight if total_weight > 0 else 1.0

        logger.debug(f"Action {action.dimension}={action.value:.3f} score: {score:.3f}")

        return float(np.clip(score, 0.0, 1.0))

    @classmethod
    def create_default_pruner(cls) -> ActionPruner:
        """Create ActionPruner with default constraints.

        Returns:
            ActionPruner with standard constraint set
        """
        return cls()
