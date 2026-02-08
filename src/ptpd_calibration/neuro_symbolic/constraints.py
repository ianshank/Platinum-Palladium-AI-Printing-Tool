"""
Symbolic Constraints with Differentiable Loss Terms.

This module implements physics-informed constraints for curve generation
that can be used as differentiable loss terms in optimization.

The key insight is that platinum/palladium printing follows well-understood
physical laws that can be encoded as soft constraints:
- Monotonicity: Density should increase with exposure
- Bounded range: Densities must be within physical limits
- H&D curve characteristics: Toe, linear, shoulder regions
- Smoothness: No abrupt discontinuities in physical systems

References:
- Hurter-Driffield (H&D) curve theory
- Sensitometry principles from Mees & James
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from ptpd_calibration.config import NeuroSymbolicSettings, get_settings


class ConstraintType(str, Enum):
    """Types of symbolic constraints."""

    MONOTONICITY = "monotonicity"
    DENSITY_BOUNDS = "density_bounds"
    PHYSICS_HD_CURVE = "physics_hd_curve"
    SMOOTHNESS = "smoothness"
    TOE_REGION = "toe_region"
    SHOULDER_REGION = "shoulder_region"
    CUSTOM = "custom"


class ConstraintViolation(BaseModel):
    """Record of a constraint violation."""

    constraint_type: ConstraintType
    constraint_name: str
    violation_magnitude: float = Field(ge=0.0)
    violation_indices: list[int] = Field(default_factory=list)
    description: str = ""
    suggested_fix: str | None = None


class ConstraintResult(BaseModel):
    """Result of constraint evaluation."""

    is_satisfied: bool
    loss_value: float = Field(ge=0.0)
    gradient: list[float] | None = None
    violations: list[ConstraintViolation] = Field(default_factory=list)
    explanation: str = ""


class SymbolicConstraint(ABC):
    """Abstract base class for symbolic constraints.

    Symbolic constraints encode domain knowledge as differentiable
    loss terms that can be used in optimization.
    """

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        weight: float,
        settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize constraint.

        Args:
            name: Human-readable constraint name
            constraint_type: Type of constraint
            weight: Weight for this constraint in combined loss
            settings: Configuration settings
        """
        self.name = name
        self.constraint_type = constraint_type
        self.weight = weight
        self.settings = settings or get_settings().neuro_symbolic

    @abstractmethod
    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate constraint on curve values.

        Args:
            values: Array of curve output values

        Returns:
            ConstraintResult with loss and violation details
        """
        pass

    @abstractmethod
    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute differentiable loss for this constraint.

        Args:
            values: Array of curve output values

        Returns:
            Loss value (0 if constraint fully satisfied)
        """
        pass

    def compute_gradient(
        self, values: NDArray[np.float64], epsilon: float = 1e-7
    ) -> NDArray[np.float64]:
        """Compute numerical gradient of constraint loss.

        Args:
            values: Array of curve output values
            epsilon: Perturbation for numerical differentiation

        Returns:
            Gradient array
        """
        gradient = np.zeros_like(values)
        for i in range(len(values)):
            values_plus = values.copy()
            values_plus[i] += epsilon
            values_minus = values.copy()
            values_minus[i] -= epsilon
            gradient[i] = (self.compute_loss(values_plus) - self.compute_loss(values_minus)) / (
                2 * epsilon
            )
        return gradient

    def weighted_loss(self, values: NDArray[np.float64]) -> float:
        """Compute weighted loss.

        Args:
            values: Array of curve output values

        Returns:
            Loss multiplied by weight
        """
        return self.weight * self.compute_loss(values)


class MonotonicityConstraint(SymbolicConstraint):
    """Constraint enforcing monotonically increasing curve.

    Physics rationale: Density must increase with exposure.
    A reversal would indicate non-physical behavior (except for
    solarization at extreme exposures, which is rare in Pt/Pd).

    Loss function: Sum of squared negative differences
        L = sum(max(0, y[i] - y[i+1])^2)
    """

    def __init__(
        self,
        weight: float | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        settings = settings or get_settings().neuro_symbolic
        super().__init__(
            name="Monotonicity",
            constraint_type=ConstraintType.MONOTONICITY,
            weight=weight if weight is not None else settings.monotonicity_weight,
            settings=settings,
        )
        self._tolerance = 1e-8  # Small tolerance for numerical precision

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate monotonicity constraint."""
        diffs = np.diff(values)
        violations_mask = diffs < -self._tolerance
        violation_indices = np.where(violations_mask)[0].tolist()
        violation_magnitudes = np.abs(diffs[violations_mask])

        is_satisfied = len(violation_indices) == 0
        loss = self.compute_loss(values)

        violations = []
        if not is_satisfied:
            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=float(np.sum(violation_magnitudes)),
                    violation_indices=violation_indices,
                    description=f"Curve decreases at {len(violation_indices)} point(s)",
                    suggested_fix="Apply monotonicity enforcement or increase smoothing",
                )
            )

        explanation = (
            "Monotonicity satisfied: curve increases throughout"
            if is_satisfied
            else f"Monotonicity violated at indices {violation_indices[:5]}{'...' if len(violation_indices) > 5 else ''}"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            gradient=self.compute_gradient(values).tolist(),
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute monotonicity loss using smooth penalty."""
        diffs = np.diff(values)
        # Soft penalty: squared negative differences
        negative_diffs = np.minimum(diffs, 0)
        return float(np.sum(negative_diffs**2))


class DensityBoundsConstraint(SymbolicConstraint):
    """Constraint enforcing density values within physical bounds.

    Physics rationale: Density cannot be negative and has a
    maximum determined by the medium (paper + chemistry).

    Loss function: Sum of squared out-of-bounds values
        L = sum(max(0, min_d - y)^2) + sum(max(0, y - max_d)^2)
    """

    def __init__(
        self,
        min_density: float | None = None,
        max_density: float | None = None,
        weight: float | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        settings = settings or get_settings().neuro_symbolic
        super().__init__(
            name="Density Bounds",
            constraint_type=ConstraintType.DENSITY_BOUNDS,
            weight=weight if weight is not None else settings.density_bounds_weight,
            settings=settings,
        )
        self.min_density = min_density if min_density is not None else settings.min_density
        self.max_density = max_density if max_density is not None else settings.max_density

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate density bounds constraint."""
        below_min = values < self.min_density
        above_max = values > self.max_density

        below_indices = np.where(below_min)[0].tolist()
        above_indices = np.where(above_max)[0].tolist()

        is_satisfied = len(below_indices) == 0 and len(above_indices) == 0
        loss = self.compute_loss(values)

        violations = []
        if below_indices:
            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=f"{self.name} (lower)",
                    violation_magnitude=float(np.sum(self.min_density - values[below_min])),
                    violation_indices=below_indices,
                    description=f"{len(below_indices)} values below minimum density {self.min_density}",
                    suggested_fix="Check for calculation errors or increase exposure",
                )
            )
        if above_indices:
            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=f"{self.name} (upper)",
                    violation_magnitude=float(np.sum(values[above_max] - self.max_density)),
                    violation_indices=above_indices,
                    description=f"{len(above_indices)} values above maximum density {self.max_density}",
                    suggested_fix="Check for measurement errors or reduce exposure",
                )
            )

        explanation = (
            f"All values within bounds [{self.min_density}, {self.max_density}]"
            if is_satisfied
            else f"Bounds violated: {len(below_indices)} below min, {len(above_indices)} above max"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            gradient=self.compute_gradient(values).tolist(),
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute bounds loss using soft penalty."""
        # Lower bound penalty
        below_min = np.maximum(self.min_density - values, 0)
        lower_loss = np.sum(below_min**2)

        # Upper bound penalty
        above_max = np.maximum(values - self.max_density, 0)
        upper_loss = np.sum(above_max**2)

        return float(lower_loss + upper_loss)


class PhysicsConstraint(SymbolicConstraint):
    """Constraint enforcing H&D curve physics.

    The Hurter-Driffield (H&D) characteristic curve has three regions:
    1. Toe: Low exposure region with sub-linear response (sqrt-like)
    2. Linear: Middle region with approximately linear response
    3. Shoulder: High exposure region with saturation

    This constraint penalizes curves that deviate significantly
    from expected physical behavior in these regions.
    """

    def __init__(
        self,
        toe_fraction: float = 0.2,
        shoulder_fraction: float = 0.2,
        weight: float | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize physics constraint.

        Args:
            toe_fraction: Fraction of curve considered toe region
            shoulder_fraction: Fraction of curve considered shoulder region
            weight: Constraint weight
            settings: Configuration settings
        """
        settings = settings or get_settings().neuro_symbolic
        super().__init__(
            name="H&D Curve Physics",
            constraint_type=ConstraintType.PHYSICS_HD_CURVE,
            weight=weight if weight is not None else settings.physics_constraint_weight,
            settings=settings,
        )
        self.toe_fraction = toe_fraction
        self.shoulder_fraction = shoulder_fraction

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate physics constraint."""
        n = len(values)
        toe_end = int(n * self.toe_fraction)
        shoulder_start = int(n * (1 - self.shoulder_fraction))

        violations = []
        total_loss = 0.0

        # Check toe region (should be concave/sqrt-like)
        toe_loss = self._evaluate_toe_region(values[:toe_end])
        if toe_loss > 0.01:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.TOE_REGION,
                    constraint_name="Toe Region Physics",
                    violation_magnitude=toe_loss,
                    violation_indices=list(range(toe_end)),
                    description="Toe region deviates from expected concave shape",
                    suggested_fix="Check for underexposure or coating issues",
                )
            )
        total_loss += toe_loss

        # Check shoulder region (should show saturation)
        shoulder_loss = self._evaluate_shoulder_region(values[shoulder_start:])
        if shoulder_loss > 0.01:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.SHOULDER_REGION,
                    constraint_name="Shoulder Region Physics",
                    violation_magnitude=shoulder_loss,
                    violation_indices=list(range(shoulder_start, n)),
                    description="Shoulder region deviates from expected saturation",
                    suggested_fix="Check for overexposure or chemistry exhaustion",
                )
            )
        total_loss += shoulder_loss

        is_satisfied = len(violations) == 0
        loss = self.compute_loss(values)

        explanation = (
            "Curve follows expected H&D physics"
            if is_satisfied
            else f"Physics violations in {len(violations)} region(s)"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            gradient=self.compute_gradient(values).tolist(),
            violations=violations,
            explanation=explanation,
        )

    def _evaluate_toe_region(self, toe_values: NDArray[np.float64]) -> float:
        """Evaluate toe region physics (should be concave/sqrt-like)."""
        if len(toe_values) < 3:
            return 0.0

        # Second derivative should be negative (concave) in toe region
        second_deriv = np.diff(np.diff(toe_values))
        # Penalize positive second derivatives (convex behavior)
        positive_second = np.maximum(second_deriv, 0)
        return float(np.mean(positive_second**2))

    def _evaluate_shoulder_region(self, shoulder_values: NDArray[np.float64]) -> float:
        """Evaluate shoulder region physics (should show saturation)."""
        if len(shoulder_values) < 3:
            return 0.0

        # First derivative should decrease (saturation)
        first_deriv = np.diff(shoulder_values)
        if len(first_deriv) < 2:
            return 0.0

        # Rate of change of derivative should be negative
        deriv_change = np.diff(first_deriv)
        # Penalize increasing derivatives (no saturation)
        positive_change = np.maximum(deriv_change, 0)
        return float(np.mean(positive_change**2))

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute physics loss."""
        n = len(values)
        toe_end = int(n * self.toe_fraction)
        shoulder_start = int(n * (1 - self.shoulder_fraction))

        toe_loss = self._evaluate_toe_region(values[:toe_end])
        shoulder_loss = self._evaluate_shoulder_region(values[shoulder_start:])

        return toe_loss + shoulder_loss


class SmoothnessConstraint(SymbolicConstraint):
    """Constraint enforcing smooth curve transitions.

    Physics rationale: Physical systems don't have abrupt
    discontinuities. The curve should be smooth (continuous
    first and second derivatives).

    Loss function: Total variation or second derivative penalty
        L = sum((y[i+2] - 2*y[i+1] + y[i])^2)
    """

    def __init__(
        self,
        order: int = 2,
        weight: float | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize smoothness constraint.

        Args:
            order: Order of derivative to penalize (1=first, 2=second)
            weight: Constraint weight
            settings: Configuration settings
        """
        settings = settings or get_settings().neuro_symbolic
        super().__init__(
            name="Smoothness",
            constraint_type=ConstraintType.SMOOTHNESS,
            weight=weight if weight is not None else settings.smoothness_weight,
            settings=settings,
        )
        self.order = order

    def evaluate(self, values: NDArray[np.float64]) -> ConstraintResult:
        """Evaluate smoothness constraint."""
        loss = self.compute_loss(values)

        # Compute roughness metric
        roughness = self._compute_roughness(values)
        is_satisfied = roughness < 0.1  # Configurable threshold

        violations = []
        if not is_satisfied:
            rough_indices = self._find_rough_regions(values)
            violations.append(
                ConstraintViolation(
                    constraint_type=self.constraint_type,
                    constraint_name=self.name,
                    violation_magnitude=roughness,
                    violation_indices=rough_indices,
                    description=f"Curve roughness {roughness:.3f} exceeds threshold",
                    suggested_fix="Apply smoothing or check for measurement noise",
                )
            )

        explanation = (
            f"Curve is smooth (roughness={roughness:.4f})"
            if is_satisfied
            else f"Curve is rough (roughness={roughness:.4f})"
        )

        return ConstraintResult(
            is_satisfied=is_satisfied,
            loss_value=loss,
            gradient=self.compute_gradient(values).tolist(),
            violations=violations,
            explanation=explanation,
        )

    def compute_loss(self, values: NDArray[np.float64]) -> float:
        """Compute smoothness loss."""
        if len(values) < self.order + 1:
            return 0.0

        # Compute finite differences
        deriv = values.copy()
        for _ in range(self.order):
            deriv = np.diff(deriv)

        return float(np.sum(deriv**2))

    def _compute_roughness(self, values: NDArray[np.float64]) -> float:
        """Compute normalized roughness metric."""
        if len(values) < 3:
            return 0.0

        second_deriv = np.diff(np.diff(values))
        roughness = np.std(second_deriv)
        return float(roughness)

    def _find_rough_regions(
        self, values: NDArray[np.float64], threshold_factor: float = 2.0
    ) -> list[int]:
        """Find indices where curve is rough."""
        if len(values) < 3:
            return []

        second_deriv = np.diff(np.diff(values))
        threshold = threshold_factor * np.std(second_deriv)
        rough_mask = np.abs(second_deriv) > threshold
        return (np.where(rough_mask)[0] + 1).tolist()  # +1 for diff offset


class ConstraintSet:
    """Collection of constraints to apply together."""

    def __init__(
        self,
        constraints: list[SymbolicConstraint] | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize constraint set.

        Args:
            constraints: List of constraints to include
            settings: Configuration settings
        """
        self.settings = settings or get_settings().neuro_symbolic
        self.constraints: list[SymbolicConstraint] = constraints or []

    @classmethod
    def default_set(cls, settings: NeuroSymbolicSettings | None = None) -> "ConstraintSet":
        """Create default constraint set for Pt/Pd calibration.

        Args:
            settings: Configuration settings

        Returns:
            ConstraintSet with standard physics constraints
        """
        settings = settings or get_settings().neuro_symbolic
        return cls(
            constraints=[
                MonotonicityConstraint(settings=settings),
                DensityBoundsConstraint(settings=settings),
                PhysicsConstraint(settings=settings),
                SmoothnessConstraint(settings=settings),
            ],
            settings=settings,
        )

    def add_constraint(self, constraint: SymbolicConstraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)

    def remove_constraint(self, name: str) -> bool:
        """Remove constraint by name."""
        for i, c in enumerate(self.constraints):
            if c.name == name:
                self.constraints.pop(i)
                return True
        return False

    def evaluate_all(self, values: NDArray[np.float64]) -> dict[str, ConstraintResult]:
        """Evaluate all constraints.

        Args:
            values: Curve output values

        Returns:
            Dictionary mapping constraint name to result
        """
        return {c.name: c.evaluate(values) for c in self.constraints}

    def compute_total_loss(self, values: NDArray[np.float64]) -> float:
        """Compute total weighted loss from all constraints.

        Args:
            values: Curve output values

        Returns:
            Total loss value
        """
        return sum(c.weighted_loss(values) for c in self.constraints)

    def compute_total_gradient(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute total gradient from all constraints.

        Args:
            values: Curve output values

        Returns:
            Combined gradient array
        """
        gradient = np.zeros_like(values)
        for c in self.constraints:
            gradient += c.weight * c.compute_gradient(values)
        return gradient

    def all_satisfied(self, values: NDArray[np.float64]) -> bool:
        """Check if all constraints are satisfied."""
        return all(c.evaluate(values).is_satisfied for c in self.constraints)

    def get_violations(self, values: NDArray[np.float64]) -> list[ConstraintViolation]:
        """Get all constraint violations."""
        violations = []
        for c in self.constraints:
            result = c.evaluate(values)
            violations.extend(result.violations)
        return violations

    def generate_report(self, values: NDArray[np.float64]) -> str:
        """Generate human-readable constraint report.

        Args:
            values: Curve output values

        Returns:
            Formatted report string
        """
        lines = ["=" * 50, "CONSTRAINT EVALUATION REPORT", "=" * 50, ""]

        results = self.evaluate_all(values)
        total_loss = self.compute_total_loss(values)

        for name, result in results.items():
            status = "✓ SATISFIED" if result.is_satisfied else "✗ VIOLATED"
            lines.append(f"{name}: {status}")
            lines.append(f"  Loss: {result.loss_value:.6f}")
            lines.append(f"  {result.explanation}")
            if result.violations:
                for v in result.violations:
                    lines.append(f"  - {v.description}")
                    if v.suggested_fix:
                        lines.append(f"    Fix: {v.suggested_fix}")
            lines.append("")

        lines.append("-" * 50)
        lines.append(f"TOTAL WEIGHTED LOSS: {total_loss:.6f}")
        lines.append("=" * 50)

        return "\n".join(lines)


class DifferentiableLoss:
    """Differentiable loss function combining data fit and constraints.

    This class provides a combined loss function suitable for optimization:
        L_total = L_data + sum(w_i * L_constraint_i)

    where L_data is the data fitting loss (e.g., MSE to target)
    and L_constraint_i are the symbolic constraint losses.
    """

    def __init__(
        self,
        constraint_set: ConstraintSet,
        target_values: NDArray[np.float64] | None = None,
        data_loss_weight: float = 1.0,
        data_loss_type: str = "mse",
    ):
        """Initialize differentiable loss.

        Args:
            constraint_set: Set of constraints to enforce
            target_values: Optional target curve to fit
            data_loss_weight: Weight for data fitting loss
            data_loss_type: Type of data loss ('mse', 'mae', 'huber')
        """
        self.constraint_set = constraint_set
        self.target_values = target_values
        self.data_loss_weight = data_loss_weight
        self.data_loss_type = data_loss_type

    def compute_data_loss(self, values: NDArray[np.float64]) -> float:
        """Compute data fitting loss."""
        if self.target_values is None:
            return 0.0

        diff = values - self.target_values

        if self.data_loss_type == "mse":
            return float(np.mean(diff**2))
        elif self.data_loss_type == "mae":
            return float(np.mean(np.abs(diff)))
        elif self.data_loss_type == "huber":
            delta = 0.1  # Huber delta
            abs_diff = np.abs(diff)
            quadratic = np.minimum(abs_diff, delta)
            linear = abs_diff - quadratic
            return float(np.mean(0.5 * quadratic**2 + delta * linear))
        else:
            return float(np.mean(diff**2))

    def __call__(self, values: NDArray[np.float64]) -> float:
        """Compute total loss.

        Args:
            values: Curve output values

        Returns:
            Total loss value
        """
        data_loss = self.data_loss_weight * self.compute_data_loss(values)
        constraint_loss = self.constraint_set.compute_total_loss(values)
        return data_loss + constraint_loss

    def gradient(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute total gradient.

        Args:
            values: Curve output values

        Returns:
            Gradient array
        """
        # Data loss gradient
        data_grad = np.zeros_like(values)
        if self.target_values is not None:
            if self.data_loss_type == "mse":
                data_grad = 2 * self.data_loss_weight * (values - self.target_values) / len(values)
            elif self.data_loss_type == "mae":
                data_grad = (
                    self.data_loss_weight * np.sign(values - self.target_values) / len(values)
                )
            else:
                data_grad = 2 * self.data_loss_weight * (values - self.target_values) / len(values)

        # Constraint gradient
        constraint_grad = self.constraint_set.compute_total_gradient(values)

        return data_grad + constraint_grad


class ConstrainedCurveOptimizer:
    """Optimizer for generating curves that satisfy symbolic constraints.

    This optimizer uses gradient-based methods to find curves that:
    1. Fit the measured data
    2. Satisfy physical constraints

    The optimization is physics-informed: the loss function encodes
    domain knowledge from sensitometry and photographic science.
    """

    def __init__(
        self,
        constraint_set: ConstraintSet | None = None,
        settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize optimizer.

        Args:
            constraint_set: Constraints to enforce (default: physics set)
            settings: Configuration settings
        """
        self.settings = settings or get_settings().neuro_symbolic
        self.constraint_set = constraint_set or ConstraintSet.default_set(self.settings)

    def optimize(
        self,
        initial_values: NDArray[np.float64],
        target_values: NDArray[np.float64] | None = None,
        bounds: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Optimize curve to satisfy constraints.

        Args:
            initial_values: Starting curve values
            target_values: Optional target curve to fit
            bounds: (min, max) bounds for curve values

        Returns:
            Dictionary with optimized values and optimization info
        """
        # Set up loss function
        loss_fn = DifferentiableLoss(
            constraint_set=self.constraint_set,
            target_values=target_values,
            data_loss_weight=1.0,
        )

        # Set up bounds
        if bounds is None:
            bounds = (self.settings.min_density, 1.0)
        scipy_bounds = [(bounds[0], bounds[1]) for _ in initial_values]

        # Optimize
        result = minimize(
            loss_fn,
            initial_values,
            method=self.settings.optimizer_method,
            jac=loss_fn.gradient,
            bounds=scipy_bounds,
            options={
                "maxiter": self.settings.optimizer_max_iterations,
                "ftol": self.settings.optimizer_tolerance,
            },
        )

        # Evaluate final constraints
        final_values = result.x
        constraint_results = self.constraint_set.evaluate_all(final_values)
        violations = self.constraint_set.get_violations(final_values)

        return {
            "optimized_values": final_values,
            "initial_values": initial_values,
            "target_values": target_values,
            "success": result.success,
            "message": result.message,
            "iterations": result.nit,
            "final_loss": result.fun,
            "constraint_results": constraint_results,
            "violations": violations,
            "all_constraints_satisfied": len(violations) == 0,
            "report": self.constraint_set.generate_report(final_values),
        }

    def project_to_constraints(
        self, values: NDArray[np.float64], max_iterations: int = 100
    ) -> NDArray[np.float64]:
        """Project curve values to satisfy constraints.

        This performs constrained optimization starting from the given
        values, adjusting only enough to satisfy constraints.

        Args:
            values: Input curve values
            max_iterations: Maximum projection iterations

        Returns:
            Adjusted values satisfying constraints
        """
        result = self.optimize(
            initial_values=values.copy(),
            target_values=values,  # Stay close to original
        )
        return result["optimized_values"]

    def validate_curve(self, values: NDArray[np.float64]) -> tuple[bool, list[ConstraintViolation]]:
        """Validate curve against all constraints.

        Args:
            values: Curve values to validate

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = self.constraint_set.get_violations(values)
        return len(violations) == 0, violations
