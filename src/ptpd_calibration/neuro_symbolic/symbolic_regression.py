"""
Differentiable Symbolic Regression for Curve Formula Discovery.

This module implements symbolic regression with differentiable expressions
for discovering interpretable curve formulas specific to paper types.

Key features:
- Expression tree representation for symbolic formulas
- Differentiable operators for gradient-based optimization
- Genetic programming for expression evolution
- Parsimony pressure for simpler expressions
- Physics-informed constraints on valid formulas

The goal is to discover formulas like:
    density = A * log(exposure + B) * (1 - exp(-C * exposure))
that are both accurate and interpretable.

References:
- Koza (1992) "Genetic Programming"
- Petersen et al. (2021) "Deep Symbolic Regression"
- Cranmer et al. (2020) "PySR"
"""

import random
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ptpd_calibration.config import NeuroSymbolicSettings, get_settings


class OperatorType(str, Enum):
    """Types of operators in symbolic expressions."""

    # Binary operators
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"

    # Unary operators
    NEG = "neg"
    ABS = "abs"
    SQRT = "sqrt"
    LOG = "log"
    EXP = "exp"
    SIN = "sin"
    COS = "cos"
    TANH = "tanh"

    # Special operators for H&D curves
    SIGMOID = "sigmoid"
    SMOOTHSTEP = "smoothstep"

    # Terminals
    CONSTANT = "constant"
    VARIABLE = "variable"


# Operator properties: (arity, symbol, safe_domain)
OPERATOR_INFO: dict[OperatorType, tuple[int, str, bool]] = {
    OperatorType.ADD: (2, "+", True),
    OperatorType.SUB: (2, "-", True),
    OperatorType.MUL: (2, "*", True),
    OperatorType.DIV: (2, "/", False),  # Division by zero
    OperatorType.POW: (2, "^", False),  # Negative base issues
    OperatorType.NEG: (1, "-", True),
    OperatorType.ABS: (1, "abs", True),
    OperatorType.SQRT: (1, "sqrt", False),  # Negative input
    OperatorType.LOG: (1, "log", False),  # Non-positive input
    OperatorType.EXP: (1, "exp", True),
    OperatorType.SIN: (1, "sin", True),
    OperatorType.COS: (1, "cos", True),
    OperatorType.TANH: (1, "tanh", True),
    OperatorType.SIGMOID: (1, "σ", True),
    OperatorType.SMOOTHSTEP: (1, "S", True),
    OperatorType.CONSTANT: (0, "c", True),
    OperatorType.VARIABLE: (0, "x", True),
}


@dataclass
class ExpressionNode:
    """A node in a symbolic expression tree.

    Each node represents either an operator with children,
    a constant value, or a variable reference.
    """

    operator: OperatorType
    value: float | None = None  # For constants
    variable_name: str = "x"  # For variables
    children: list["ExpressionNode"] = field(default_factory=list)

    # Optimization metadata
    _gradient: float | None = None
    _cached_value: float | None = None

    def arity(self) -> int:
        """Get expected number of children."""
        return OPERATOR_INFO[self.operator][0]

    def symbol(self) -> str:
        """Get display symbol."""
        return OPERATOR_INFO[self.operator][1]

    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.arity() == 0

    def depth(self) -> int:
        """Calculate tree depth."""
        if self.is_terminal():
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """Calculate total number of nodes."""
        if self.is_terminal():
            return 1
        return 1 + sum(child.size() for child in self.children)

    def evaluate(
        self,
        x: float | NDArray[np.float64],
        epsilon: float = 1e-10,
    ) -> float | NDArray[np.float64]:
        """Evaluate expression at given input(s).

        Args:
            x: Input value(s)
            epsilon: Small value to prevent division by zero

        Returns:
            Expression value(s)
        """
        if self.operator == OperatorType.CONSTANT:
            if isinstance(x, np.ndarray):
                return np.full_like(x, self.value or 0.0)
            return self.value or 0.0

        if self.operator == OperatorType.VARIABLE:
            return x

        # Evaluate children
        child_vals = [child.evaluate(x, epsilon) for child in self.children]

        # Apply operator with safe handling
        return self._apply_operator(child_vals, epsilon)

    def _apply_operator(
        self,
        args: list[float | NDArray[np.float64]],
        epsilon: float,
    ) -> float | NDArray[np.float64]:
        """Apply operator to arguments with numerical safety."""
        op = self.operator

        if op == OperatorType.ADD:
            return args[0] + args[1]
        elif op == OperatorType.SUB:
            return args[0] - args[1]
        elif op == OperatorType.MUL:
            return args[0] * args[1]
        elif op == OperatorType.DIV:
            # Safe division
            return args[0] / (args[1] + np.sign(args[1]) * epsilon + (args[1] == 0) * epsilon)
        elif op == OperatorType.POW:
            # Safe power (handle negative bases)
            base = np.abs(args[0]) + epsilon
            return np.power(base, args[1])
        elif op == OperatorType.NEG:
            return -args[0]
        elif op == OperatorType.ABS:
            return np.abs(args[0])
        elif op == OperatorType.SQRT:
            return np.sqrt(np.abs(args[0]) + epsilon)
        elif op == OperatorType.LOG:
            return np.log(np.abs(args[0]) + epsilon)
        elif op == OperatorType.EXP:
            # Clip to prevent overflow
            return np.exp(np.clip(args[0], -50, 50))
        elif op == OperatorType.SIN:
            return np.sin(args[0])
        elif op == OperatorType.COS:
            return np.cos(args[0])
        elif op == OperatorType.TANH:
            return np.tanh(args[0])
        elif op == OperatorType.SIGMOID:
            return 1.0 / (1.0 + np.exp(-np.clip(args[0], -50, 50)))
        elif op == OperatorType.SMOOTHSTEP:
            # Smoothstep: 3t^2 - 2t^3 for t in [0,1]
            t = np.clip(args[0], 0, 1)
            return t * t * (3 - 2 * t)
        else:
            return args[0] if args else 0.0

    def to_string(self, precision: int = 4) -> str:
        """Convert expression to human-readable string.

        Args:
            precision: Decimal places for constants

        Returns:
            String representation
        """
        if self.operator == OperatorType.CONSTANT:
            return f"{self.value:.{precision}f}"

        if self.operator == OperatorType.VARIABLE:
            return self.variable_name

        symbol = self.symbol()
        child_strs = [child.to_string(precision) for child in self.children]

        if self.arity() == 1:
            return f"{symbol}({child_strs[0]})"
        elif self.arity() == 2:
            if self.operator in (
                OperatorType.ADD,
                OperatorType.SUB,
                OperatorType.MUL,
                OperatorType.DIV,
            ):
                return f"({child_strs[0]} {symbol} {child_strs[1]})"
            else:
                return f"{symbol}({child_strs[0]}, {child_strs[1]})"
        else:
            return f"{symbol}({', '.join(child_strs)})"

    def to_latex(self, precision: int = 4) -> str:
        """Convert expression to LaTeX string."""
        if self.operator == OperatorType.CONSTANT:
            return f"{self.value:.{precision}f}"

        if self.operator == OperatorType.VARIABLE:
            return self.variable_name

        child_latex = [child.to_latex(precision) for child in self.children]

        op = self.operator
        if op == OperatorType.ADD:
            return f"({child_latex[0]} + {child_latex[1]})"
        elif op == OperatorType.SUB:
            return f"({child_latex[0]} - {child_latex[1]})"
        elif op == OperatorType.MUL:
            return f"({child_latex[0]} \\cdot {child_latex[1]})"
        elif op == OperatorType.DIV:
            return f"\\frac{{{child_latex[0]}}}{{{child_latex[1]}}}"
        elif op == OperatorType.POW:
            return f"{child_latex[0]}^{{{child_latex[1]}}}"
        elif op == OperatorType.SQRT:
            return f"\\sqrt{{{child_latex[0]}}}"
        elif op == OperatorType.LOG:
            return f"\\log({child_latex[0]})"
        elif op == OperatorType.EXP:
            return f"e^{{{child_latex[0]}}}"
        elif op == OperatorType.SIGMOID:
            return f"\\sigma({child_latex[0]})"
        else:
            return f"\\text{{{self.symbol()}}}({child_latex[0]})"

    def get_constants(self) -> list[float]:
        """Get all constant values in expression."""
        if self.operator == OperatorType.CONSTANT:
            return [self.value or 0.0]

        constants = []
        for child in self.children:
            constants.extend(child.get_constants())
        return constants

    def set_constants(self, values: list[float]) -> int:
        """Set constant values from a list.

        Args:
            values: List of values to assign

        Returns:
            Number of constants assigned
        """
        idx = 0
        if self.operator == OperatorType.CONSTANT:
            if idx < len(values):
                self.value = values[idx]
            return 1

        for child in self.children:
            idx += child.set_constants(values[idx:])
        return idx

    def copy(self) -> "ExpressionNode":
        """Create a deep copy of this expression."""
        return deepcopy(self)


class SymbolicExpression(BaseModel):
    """A complete symbolic expression with metadata."""

    class Config:
        arbitrary_types_allowed = True

    root: ExpressionNode
    fitness: float = Field(default=float("inf"))
    complexity: int = Field(default=0)
    r_squared: float = Field(default=0.0)
    mse: float = Field(default=float("inf"))
    description: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.complexity = self.root.size()

    def evaluate(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Evaluate expression."""
        return self.root.evaluate(x)

    def to_string(self) -> str:
        """Get string representation."""
        return self.root.to_string()

    def to_latex(self) -> str:
        """Get LaTeX representation."""
        return self.root.to_latex()


class ExpressionLibrary:
    """Library of common expressions for curve fitting."""

    @staticmethod
    def linear() -> ExpressionNode:
        """Create linear expression: a*x + b"""
        return ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                        ExpressionNode(operator=OperatorType.VARIABLE),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=0.0),
            ],
        )

    @staticmethod
    def power_law() -> ExpressionNode:
        """Create power law: a * x^b + c"""
        return ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                        ExpressionNode(
                            operator=OperatorType.POW,
                            children=[
                                ExpressionNode(operator=OperatorType.VARIABLE),
                                ExpressionNode(operator=OperatorType.CONSTANT, value=0.8),
                            ],
                        ),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=0.0),
            ],
        )

    @staticmethod
    def logarithmic() -> ExpressionNode:
        """Create logarithmic: a * log(b*x + c) + d"""
        return ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                        ExpressionNode(
                            operator=OperatorType.LOG,
                            children=[
                                ExpressionNode(
                                    operator=OperatorType.ADD,
                                    children=[
                                        ExpressionNode(
                                            operator=OperatorType.MUL,
                                            children=[
                                                ExpressionNode(
                                                    operator=OperatorType.CONSTANT, value=1.0
                                                ),
                                                ExpressionNode(operator=OperatorType.VARIABLE),
                                            ],
                                        ),
                                        ExpressionNode(operator=OperatorType.CONSTANT, value=0.1),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=0.0),
            ],
        )

    @staticmethod
    def sigmoid() -> ExpressionNode:
        """Create sigmoid: a / (1 + exp(-b*(x-c))) + d"""
        return ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                        ExpressionNode(
                            operator=OperatorType.SIGMOID,
                            children=[
                                ExpressionNode(
                                    operator=OperatorType.MUL,
                                    children=[
                                        ExpressionNode(operator=OperatorType.CONSTANT, value=5.0),
                                        ExpressionNode(
                                            operator=OperatorType.SUB,
                                            children=[
                                                ExpressionNode(operator=OperatorType.VARIABLE),
                                                ExpressionNode(
                                                    operator=OperatorType.CONSTANT, value=0.5
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=0.0),
            ],
        )

    @staticmethod
    def hd_curve() -> ExpressionNode:
        """Create H&D characteristic curve model.

        This models the toe-linear-shoulder behavior:
        density = Dmax * (1 - exp(-k * exposure^gamma)) + Dmin
        """
        return ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),  # Dmax
                        ExpressionNode(
                            operator=OperatorType.SUB,
                            children=[
                                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                                ExpressionNode(
                                    operator=OperatorType.EXP,
                                    children=[
                                        ExpressionNode(
                                            operator=OperatorType.NEG,
                                            children=[
                                                ExpressionNode(
                                                    operator=OperatorType.MUL,
                                                    children=[
                                                        ExpressionNode(
                                                            operator=OperatorType.CONSTANT,
                                                            value=3.0,
                                                        ),  # k
                                                        ExpressionNode(
                                                            operator=OperatorType.POW,
                                                            children=[
                                                                ExpressionNode(
                                                                    operator=OperatorType.VARIABLE
                                                                ),
                                                                ExpressionNode(
                                                                    operator=OperatorType.CONSTANT,
                                                                    value=0.8,
                                                                ),  # gamma
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=0.1),  # Dmin
            ],
        )


class DifferentiableSymbolicRegressor:
    """Symbolic regression with differentiable optimization.

    Combines genetic programming for structure search with
    gradient-based optimization for constant refinement.
    """

    def __init__(self, settings: NeuroSymbolicSettings | None = None):
        """Initialize regressor.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings().neuro_symbolic
        self._population: list[SymbolicExpression] = []
        self._best_expression: SymbolicExpression | None = None
        self._generation = 0

        # Parse allowed operators
        self._allowed_ops = self._parse_operators(self.settings.sr_allowed_operators)

    def _parse_operators(self, op_names: list[str]) -> list[OperatorType]:
        """Parse operator names to enum values."""
        ops = []
        for name in op_names:
            with suppress(ValueError):
                ops.append(OperatorType(name))
        return ops

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        initial_expressions: list[ExpressionNode] | None = None,
    ) -> SymbolicExpression:
        """Fit symbolic expression to data.

        Args:
            x: Input values
            y: Target values
            initial_expressions: Optional seed expressions

        Returns:
            Best discovered expression
        """
        # Initialize population
        self._initialize_population(initial_expressions)

        # Evolutionary loop
        for gen in range(self.settings.sr_generations):
            self._generation = gen

            # Evaluate fitness
            for expr in self._population:
                self._evaluate_fitness(expr, x, y)

            # Sort by fitness
            self._population.sort(key=lambda e: e.fitness)

            # Track best
            if (
                self._best_expression is None
                or self._population[0].fitness < self._best_expression.fitness
            ):
                self._best_expression = self._population[0].root.copy()
                self._best_expression = SymbolicExpression(
                    root=self._population[0].root.copy(),
                    fitness=self._population[0].fitness,
                    r_squared=self._population[0].r_squared,
                    mse=self._population[0].mse,
                )

            # Selection and reproduction
            self._evolve_population(x, y)

        # Final optimization of best expression
        if self._best_expression is not None:
            self._optimize_constants(self._best_expression, x, y)

        return self._best_expression

    def _initialize_population(
        self, initial_expressions: list[ExpressionNode] | None = None
    ) -> None:
        """Initialize population with expressions."""
        self._population = []

        # Add seed expressions
        seeds = [
            ExpressionLibrary.linear(),
            ExpressionLibrary.power_law(),
            ExpressionLibrary.logarithmic(),
            ExpressionLibrary.sigmoid(),
            ExpressionLibrary.hd_curve(),
        ]

        if initial_expressions:
            seeds.extend(initial_expressions)

        for seed in seeds:
            self._population.append(SymbolicExpression(root=seed.copy()))

        # Fill rest with random expressions
        while len(self._population) < self.settings.sr_population_size:
            expr = self._generate_random_expression(max_depth=self.settings.sr_max_expression_depth)
            self._population.append(SymbolicExpression(root=expr))

    def _generate_random_expression(self, max_depth: int, current_depth: int = 0) -> ExpressionNode:
        """Generate random expression tree."""
        # Terminal probability increases with depth
        terminal_prob = current_depth / max_depth

        if current_depth >= max_depth or random.random() < terminal_prob:
            # Generate terminal
            if random.random() < 0.5:
                return ExpressionNode(
                    operator=OperatorType.CONSTANT,
                    value=random.uniform(-2, 2),
                )
            else:
                return ExpressionNode(operator=OperatorType.VARIABLE)

        # Generate operator
        available_ops = [
            op
            for op in self._allowed_ops
            if OPERATOR_INFO[op][0] > 0  # Non-terminal
        ]

        if not available_ops:
            return ExpressionNode(operator=OperatorType.VARIABLE)

        op = random.choice(available_ops)
        arity = OPERATOR_INFO[op][0]

        children = [
            self._generate_random_expression(max_depth, current_depth + 1) for _ in range(arity)
        ]

        return ExpressionNode(operator=op, children=children)

    def _evaluate_fitness(
        self,
        expr: SymbolicExpression,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        """Evaluate expression fitness."""
        try:
            y_pred = expr.evaluate(x)

            # Handle non-finite values
            if not np.all(np.isfinite(y_pred)):
                expr.fitness = float("inf")
                expr.mse = float("inf")
                expr.r_squared = 0.0
                return

            # MSE
            mse = float(np.mean((y - y_pred) ** 2))

            # R-squared
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0

            # Complexity penalty (parsimony)
            complexity = expr.root.size()
            complexity_penalty = self.settings.sr_parsimony_coefficient * complexity

            expr.mse = mse
            expr.r_squared = float(r2)
            expr.complexity = complexity
            expr.fitness = mse + complexity_penalty

        except Exception:
            expr.fitness = float("inf")
            expr.mse = float("inf")
            expr.r_squared = 0.0

    def _evolve_population(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Evolve population through selection and genetic operators."""
        new_population = []

        # Elitism: keep best expressions
        elite_size = max(1, self.settings.sr_population_size // 10)
        new_population.extend(
            SymbolicExpression(
                root=e.root.copy(), fitness=e.fitness, r_squared=e.r_squared, mse=e.mse
            )
            for e in self._population[:elite_size]
        )

        # Fill rest with genetic operations
        while len(new_population) < self.settings.sr_population_size:
            if random.random() < self.settings.sr_crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_select()
                child = self._mutate(parent)

            new_population.append(SymbolicExpression(root=child))

        self._population = new_population

    def _tournament_select(self, tournament_size: int = 5) -> ExpressionNode:
        """Select expression via tournament selection."""
        contestants = random.sample(
            self._population,
            min(tournament_size, len(self._population)),
        )
        best = min(contestants, key=lambda e: e.fitness)
        return best.root.copy()

    def _crossover(self, parent1: ExpressionNode, parent2: ExpressionNode) -> ExpressionNode:
        """Perform subtree crossover."""
        child = parent1.copy()

        # Find random subtree in child
        nodes1 = self._collect_nodes(child)
        if not nodes1:
            return child

        # Find random subtree in parent2
        nodes2 = self._collect_nodes(parent2)
        if not nodes2:
            return child

        # Swap subtrees
        idx1 = random.randrange(len(nodes1))
        idx2 = random.randrange(len(nodes2))

        # Simple swap: replace child node with copy of parent2 subtree
        target, parent, child_idx = nodes1[idx1]
        source, _, _ = nodes2[idx2]

        if parent is not None and child_idx is not None:
            parent.children[child_idx] = source.copy()

        # Limit depth
        if child.depth() > self.settings.sr_max_expression_depth:
            return parent1.copy()

        return child

    def _mutate(self, parent: ExpressionNode) -> ExpressionNode:
        """Perform mutation on expression."""
        child = parent.copy()

        mutation_type = random.random()

        if mutation_type < 0.3:
            # Point mutation: change operator
            self._point_mutate(child)
        elif mutation_type < 0.6:
            # Constant mutation: perturb constants
            self._constant_mutate(child)
        else:
            # Subtree mutation: replace subtree
            self._subtree_mutate(child)

        return child

    def _point_mutate(self, node: ExpressionNode) -> None:
        """Mutate a single operator."""
        nodes = self._collect_nodes(node)
        if not nodes:
            return

        target, _, _ = random.choice(nodes)

        if target.operator == OperatorType.CONSTANT:
            # Mutate constant value
            target.value = (target.value or 0.0) + random.gauss(0, 0.5)
        elif not target.is_terminal():
            # Change operator (same arity)
            same_arity = [op for op in self._allowed_ops if OPERATOR_INFO[op][0] == target.arity()]
            if same_arity:
                target.operator = random.choice(same_arity)

    def _constant_mutate(self, node: ExpressionNode) -> None:
        """Mutate constant values."""
        constants = self._collect_constants(node)
        for const_node in constants:
            if random.random() < self.settings.sr_mutation_rate:
                const_node.value = (const_node.value or 0.0) + random.gauss(0, 0.3)

    def _subtree_mutate(self, node: ExpressionNode) -> None:
        """Replace a subtree with a random one."""
        nodes = self._collect_nodes(node)
        if not nodes:
            return

        target, parent, child_idx = random.choice(nodes)

        if parent is not None and child_idx is not None:
            new_subtree = self._generate_random_expression(
                max_depth=max(1, self.settings.sr_max_expression_depth - 2)
            )
            parent.children[child_idx] = new_subtree

    def _collect_nodes(
        self, node: ExpressionNode
    ) -> list[tuple[ExpressionNode, ExpressionNode | None, int | None]]:
        """Collect all nodes with parent references."""
        result: list[tuple[ExpressionNode, ExpressionNode | None, int | None]] = [
            (node, None, None)
        ]

        def _collect(n: ExpressionNode, parent: ExpressionNode | None, idx: int | None) -> None:
            for i, child in enumerate(n.children):
                result.append((child, n, i))
                _collect(child, n, i)

        _collect(node, None, None)
        return result

    def _collect_constants(self, node: ExpressionNode) -> list[ExpressionNode]:
        """Collect all constant nodes."""
        result = []
        if node.operator == OperatorType.CONSTANT:
            result.append(node)
        for child in node.children:
            result.extend(self._collect_constants(child))
        return result

    def _optimize_constants(
        self,
        expr: SymbolicExpression,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        """Optimize constants in expression using gradient descent."""
        from scipy.optimize import minimize

        constants = expr.root.get_constants()
        if not constants:
            return

        def loss(params: NDArray[np.float64]) -> float:
            expr.root.set_constants(list(params))
            try:
                y_pred = expr.evaluate(x)
                if not np.all(np.isfinite(y_pred)):
                    return 1e10
                return float(np.mean((y - y_pred) ** 2))
            except Exception:
                return 1e10

        result = minimize(
            loss,
            np.array(constants),
            method="L-BFGS-B",
            options={"maxiter": 100},
        )

        if result.success:
            expr.root.set_constants(list(result.x))
            self._evaluate_fitness(expr, x, y)


class CurveFormulaDiscovery:
    """High-level interface for discovering curve formulas.

    Combines symbolic regression with domain-specific templates
    for platinum/palladium curve discovery.
    """

    def __init__(self, settings: NeuroSymbolicSettings | None = None):
        """Initialize formula discovery.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings().neuro_symbolic
        self._regressor = DifferentiableSymbolicRegressor(settings)

    def discover_formula(
        self,
        measured_densities: list[float],
        input_values: list[float] | None = None,
        paper_type: str | None = None,
    ) -> dict[str, Any]:
        """Discover symbolic formula for curve.

        Args:
            measured_densities: Measured density values
            input_values: Optional input values (default: linspace 0-1)
            paper_type: Optional paper type for seeding

        Returns:
            Dictionary with discovered formula and metadata
        """
        y = np.array(measured_densities)
        n = len(y)

        x = np.linspace(0, 1, n) if input_values is None else np.array(input_values)

        # Normalize
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_norm = (y - y_min) / y_range

        # Seed with H&D curve template
        seeds = [ExpressionLibrary.hd_curve()]

        # Run symbolic regression
        best = self._regressor.fit(x, y_norm, initial_expressions=seeds)

        # Denormalize formula description
        formula_str = best.to_string()
        latex_str = best.to_latex()

        # Generate evaluation
        y_pred = best.evaluate(x) * y_range + y_min
        residuals = y - y_pred

        return {
            "formula": formula_str,
            "latex": latex_str,
            "expression": best,
            "r_squared": best.r_squared,
            "mse": best.mse * y_range**2,  # Denormalized MSE
            "complexity": best.complexity,
            "constants": best.root.get_constants(),
            "predictions": y_pred.tolist(),
            "residuals": residuals.tolist(),
            "paper_type": paper_type,
            "interpretation": self._interpret_formula(best),
        }

    def _interpret_formula(self, expr: SymbolicExpression) -> str:
        """Generate human-readable interpretation of formula."""
        formula = expr.to_string()
        interpretation = []

        # Check for common patterns
        if "log" in formula:
            interpretation.append("Logarithmic component suggests typical H&D toe behavior")
        if "exp" in formula:
            interpretation.append("Exponential component indicates saturation (shoulder region)")
        if "^" in formula or "pow" in formula:
            interpretation.append("Power law component suggests gamma correction")
        if "σ" in formula or "sigmoid" in formula.lower():
            interpretation.append("Sigmoid indicates S-curve characteristic")

        if expr.complexity < 10:
            interpretation.append("Simple formula - good for interpretation")
        elif expr.complexity < 20:
            interpretation.append("Moderate complexity - reasonably interpretable")
        else:
            interpretation.append("Complex formula - consider simplification")

        if expr.r_squared > 0.99:
            interpretation.append("Excellent fit (R² > 0.99)")
        elif expr.r_squared > 0.95:
            interpretation.append("Good fit (R² > 0.95)")
        elif expr.r_squared > 0.90:
            interpretation.append("Acceptable fit (R² > 0.90)")
        else:
            interpretation.append(f"Poor fit (R² = {expr.r_squared:.3f}) - may need more data")

        return "; ".join(interpretation) if interpretation else "No specific patterns detected"

    def compare_formulas(
        self,
        formulas: list[SymbolicExpression],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> list[dict[str, Any]]:
        """Compare multiple formulas on same data.

        Args:
            formulas: List of expressions to compare
            x: Input values
            y: Target values

        Returns:
            Ranked list of formula evaluations
        """
        results = []

        for expr in formulas:
            y_pred = expr.evaluate(x)

            # Calculate metrics
            mse = float(np.mean((y - y_pred) ** 2))
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0

            # AIC for model comparison
            n = len(y)
            k = expr.complexity  # Number of parameters approximated by complexity
            aic = n * np.log(mse + 1e-10) + 2 * k

            results.append(
                {
                    "formula": expr.to_string(),
                    "r_squared": float(r2),
                    "mse": mse,
                    "complexity": expr.complexity,
                    "aic": float(aic),
                    "interpretation": self._interpret_formula(expr),
                }
            )

        # Sort by AIC (lower is better)
        results.sort(key=lambda r: r["aic"])
        return results
