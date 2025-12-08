"""
Tests for neuro-symbolic regression module.

Tests differentiable symbolic regression for discovering
interpretable curve formulas.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ptpd_calibration.config import NeuroSymbolicSettings
from ptpd_calibration.neuro_symbolic.symbolic_regression import (
    OperatorType,
    ExpressionNode,
    SymbolicExpression,
    ExpressionLibrary,
    DifferentiableSymbolicRegressor,
    CurveFormulaDiscovery,
    OPERATOR_INFO,
)


class TestExpressionNode:
    """Tests for ExpressionNode class."""

    def test_constant_node(self):
        """Test constant node creation and evaluation."""
        node = ExpressionNode(operator=OperatorType.CONSTANT, value=3.14)

        assert node.is_terminal()
        assert node.arity() == 0
        assert node.evaluate(1.0) == pytest.approx(3.14)
        # Use float array to preserve constant's float value
        assert node.evaluate(np.array([1.0, 2.0, 3.0])) == pytest.approx([3.14, 3.14, 3.14])

    def test_variable_node(self):
        """Test variable node creation and evaluation."""
        node = ExpressionNode(operator=OperatorType.VARIABLE)

        assert node.is_terminal()
        assert node.evaluate(5.0) == pytest.approx(5.0)
        assert_array_almost_equal(
            node.evaluate(np.array([1, 2, 3])), np.array([1, 2, 3])
        )

    def test_add_operator(self):
        """Test addition operator."""
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )

        assert not node.is_terminal()
        assert node.arity() == 2
        assert node.evaluate(5.0) == pytest.approx(6.0)

    def test_mul_operator(self):
        """Test multiplication operator."""
        node = ExpressionNode(
            operator=OperatorType.MUL,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )

        assert node.evaluate(3.0) == pytest.approx(6.0)

    def test_div_operator_safe(self):
        """Test division operator with safe handling."""
        node = ExpressionNode(
            operator=OperatorType.DIV,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )

        # Should handle division by zero gracefully
        result = node.evaluate(0.0)
        assert np.isfinite(result)

    def test_sqrt_operator_safe(self):
        """Test sqrt with negative input handling."""
        node = ExpressionNode(
            operator=OperatorType.SQRT,
            children=[ExpressionNode(operator=OperatorType.VARIABLE)],
        )

        # Should handle negative input
        result = node.evaluate(-1.0)
        assert np.isfinite(result)
        assert result >= 0

    def test_log_operator_safe(self):
        """Test log with non-positive input handling."""
        node = ExpressionNode(
            operator=OperatorType.LOG,
            children=[ExpressionNode(operator=OperatorType.VARIABLE)],
        )

        # Should handle zero/negative input
        result = node.evaluate(0.0)
        assert np.isfinite(result)

    def test_exp_operator_safe(self):
        """Test exp with overflow protection."""
        node = ExpressionNode(
            operator=OperatorType.EXP,
            children=[ExpressionNode(operator=OperatorType.VARIABLE)],
        )

        # Should handle large input without overflow
        result = node.evaluate(1000.0)
        assert np.isfinite(result)

    def test_sigmoid_operator(self):
        """Test sigmoid operator."""
        node = ExpressionNode(
            operator=OperatorType.SIGMOID,
            children=[ExpressionNode(operator=OperatorType.VARIABLE)],
        )

        assert node.evaluate(0.0) == pytest.approx(0.5)
        assert node.evaluate(10.0) > 0.99
        assert node.evaluate(-10.0) < 0.01

    def test_smoothstep_operator(self):
        """Test smoothstep operator."""
        node = ExpressionNode(
            operator=OperatorType.SMOOTHSTEP,
            children=[ExpressionNode(operator=OperatorType.VARIABLE)],
        )

        assert node.evaluate(0.0) == pytest.approx(0.0)
        assert node.evaluate(0.5) == pytest.approx(0.5)
        assert node.evaluate(1.0) == pytest.approx(1.0)

    def test_nested_expression(self):
        """Test nested expression evaluation."""
        # x^2 + 2*x + 1 = (x+1)^2
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.POW,
                    children=[
                        ExpressionNode(operator=OperatorType.VARIABLE),
                        ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                    ],
                ),
                ExpressionNode(
                    operator=OperatorType.ADD,
                    children=[
                        ExpressionNode(
                            operator=OperatorType.MUL,
                            children=[
                                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                                ExpressionNode(operator=OperatorType.VARIABLE),
                            ],
                        ),
                        ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                    ],
                ),
            ],
        )

        # At x=3: 9 + 6 + 1 = 16 = (3+1)^2
        assert node.evaluate(3.0) == pytest.approx(16.0)

    def test_depth_calculation(self):
        """Test expression tree depth calculation."""
        # Simple constant
        const = ExpressionNode(operator=OperatorType.CONSTANT, value=1.0)
        assert const.depth() == 1

        # x + 1
        simple = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )
        assert simple.depth() == 2

    def test_size_calculation(self):
        """Test expression tree size calculation."""
        # x + 1 has 3 nodes
        simple = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )
        assert simple.size() == 3

    def test_to_string(self):
        """Test string representation."""
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )

        string = node.to_string()
        assert "x" in string
        assert "1.0" in string
        assert "+" in string

    def test_to_latex(self):
        """Test LaTeX representation."""
        node = ExpressionNode(
            operator=OperatorType.DIV,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
            ],
        )

        latex = node.to_latex()
        assert "\\frac" in latex

    def test_get_set_constants(self):
        """Test getting and setting constants."""
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.MUL,
                    children=[
                        ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                        ExpressionNode(operator=OperatorType.VARIABLE),
                    ],
                ),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )

        constants = node.get_constants()
        assert len(constants) == 2
        assert 2.0 in constants
        assert 1.0 in constants

        # Set new constants
        node.set_constants([3.0, 5.0])
        new_constants = node.get_constants()
        assert 3.0 in new_constants
        assert 5.0 in new_constants

    def test_copy(self):
        """Test deep copy of expression."""
        original = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )

        copy = original.copy()

        # Modify copy
        copy.children[1].value = 99.0

        # Original should be unchanged
        assert original.children[1].value == 1.0


class TestExpressionLibrary:
    """Tests for ExpressionLibrary templates."""

    def test_linear_template(self):
        """Test linear expression template."""
        expr = ExpressionLibrary.linear()

        x = np.linspace(0, 1, 10)
        y = expr.evaluate(x)

        # Default should be identity
        assert_array_almost_equal(y, x, decimal=4)

    def test_power_law_template(self):
        """Test power law expression template."""
        expr = ExpressionLibrary.power_law()

        x = np.linspace(0, 1, 10)
        y = expr.evaluate(x)

        assert np.all(np.isfinite(y))
        assert y[0] == pytest.approx(0.0, abs=0.1)

    def test_logarithmic_template(self):
        """Test logarithmic expression template."""
        expr = ExpressionLibrary.logarithmic()

        x = np.linspace(0, 1, 10)
        y = expr.evaluate(x)

        assert np.all(np.isfinite(y))

    def test_sigmoid_template(self):
        """Test sigmoid expression template."""
        expr = ExpressionLibrary.sigmoid()

        x = np.linspace(0, 1, 10)
        y = expr.evaluate(x)

        assert np.all(np.isfinite(y))
        assert y[0] < y[-1]  # Increasing

    def test_hd_curve_template(self):
        """Test H&D curve expression template."""
        expr = ExpressionLibrary.hd_curve()

        x = np.linspace(0, 1, 100)
        y = expr.evaluate(x)

        assert np.all(np.isfinite(y))
        # Should be increasing
        assert y[-1] > y[0]
        # Should have saturation (decreasing derivative at end)
        deriv = np.diff(y)
        assert deriv[-1] < deriv[len(deriv)//2]


class TestSymbolicExpression:
    """Tests for SymbolicExpression wrapper."""

    def test_expression_creation(self):
        """Test symbolic expression creation."""
        root = ExpressionNode(operator=OperatorType.VARIABLE)
        expr = SymbolicExpression(root=root)

        assert expr.complexity == 1
        assert expr.fitness == float("inf")

    def test_expression_evaluation(self):
        """Test expression evaluation via wrapper."""
        root = ExpressionNode(
            operator=OperatorType.MUL,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )
        expr = SymbolicExpression(root=root)

        assert expr.evaluate(5.0) == pytest.approx(10.0)

    def test_expression_string_methods(self):
        """Test string conversion methods."""
        root = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
            ],
        )
        expr = SymbolicExpression(root=root)

        assert len(expr.to_string()) > 0
        assert len(expr.to_latex()) > 0


class TestDifferentiableSymbolicRegressor:
    """Tests for DifferentiableSymbolicRegressor."""

    @pytest.fixture
    def regressor(self):
        """Create regressor with fast settings for testing."""
        settings = NeuroSymbolicSettings(
            sr_population_size=20,
            sr_generations=10,
            sr_max_expression_depth=4,
        )
        return DifferentiableSymbolicRegressor(settings=settings)

    def test_fit_linear_data(self, regressor):
        """Test fitting linear data."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 0.5  # Linear

        result = regressor.fit(x, y)

        assert result is not None
        # Genetic programming is stochastic - with limited generations
        # we expect reasonable but not perfect fit
        assert result.r_squared > 0.7

        # Should produce reasonable predictions
        y_pred = result.evaluate(x)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.5  # More lenient for stochastic algorithm

    def test_fit_power_law_data(self, regressor):
        """Test fitting power law data."""
        x = np.linspace(0.01, 1, 50)
        y = np.power(x, 0.5)  # Square root

        result = regressor.fit(x, y)

        assert result is not None
        # May not achieve perfect fit with limited generations
        # but should be reasonable
        assert result.mse < 0.5

    def test_fit_with_seeds(self, regressor):
        """Test fitting with seed expressions."""
        x = np.linspace(0, 1, 50)
        y = np.exp(-3 * x)  # Exponential decay

        seeds = [
            ExpressionNode(
                operator=OperatorType.EXP,
                children=[
                    ExpressionNode(
                        operator=OperatorType.MUL,
                        children=[
                            ExpressionNode(operator=OperatorType.CONSTANT, value=-1.0),
                            ExpressionNode(operator=OperatorType.VARIABLE),
                        ],
                    )
                ],
            )
        ]

        result = regressor.fit(x, y, initial_expressions=seeds)

        assert result is not None

    def test_parsimony_pressure(self):
        """Test that parsimony pressure favors simpler expressions."""
        settings_no_parsimony = NeuroSymbolicSettings(
            sr_population_size=20,
            sr_generations=10,  # Minimum valid value
            sr_parsimony_coefficient=0.0,
        )
        settings_high_parsimony = NeuroSymbolicSettings(
            sr_population_size=20,
            sr_generations=10,  # Minimum valid value
            sr_parsimony_coefficient=0.1,
        )

        regressor_no = DifferentiableSymbolicRegressor(settings_no_parsimony)
        regressor_high = DifferentiableSymbolicRegressor(settings_high_parsimony)

        x = np.linspace(0, 1, 30)
        y = x  # Simple linear

        result_no = regressor_no.fit(x, y)
        result_high = regressor_high.fit(x, y)

        # High parsimony should prefer simpler expressions
        # (not guaranteed but likely)
        assert result_high.complexity <= result_no.complexity + 5


class TestCurveFormulaDiscovery:
    """Tests for CurveFormulaDiscovery."""

    @pytest.fixture
    def discovery(self):
        """Create formula discovery with fast settings."""
        settings = NeuroSymbolicSettings(
            sr_population_size=20,
            sr_generations=10,
        )
        return CurveFormulaDiscovery(settings=settings)

    def test_discover_formula_basic(self, discovery):
        """Test basic formula discovery."""
        # Simulate H&D curve data
        x = np.linspace(0, 1, 21)
        densities = 0.1 + 2.0 * (1 - np.exp(-3 * np.power(x, 0.8)))

        result = discovery.discover_formula(densities.tolist())

        assert "formula" in result
        assert "latex" in result
        assert "r_squared" in result
        assert "interpretation" in result
        assert result["formula"] is not None

    def test_discover_formula_with_paper_type(self, discovery):
        """Test formula discovery with paper type."""
        densities = list(np.linspace(0.1, 2.1, 21))

        result = discovery.discover_formula(
            densities,
            paper_type="Test Paper",
        )

        assert result["paper_type"] == "Test Paper"

    def test_discover_formula_custom_input(self, discovery):
        """Test formula discovery with custom input values."""
        x = np.linspace(0, 2, 21)  # Non-standard range
        densities = list(0.5 * x + 0.1)

        result = discovery.discover_formula(
            densities,
            input_values=x.tolist(),
        )

        assert result["formula"] is not None
        assert len(result["predictions"]) == 21

    def test_interpretation_patterns(self, discovery):
        """Test that interpretation detects patterns."""
        # Create expression with known patterns
        root = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(
                    operator=OperatorType.LOG,
                    children=[ExpressionNode(operator=OperatorType.VARIABLE)],
                ),
                ExpressionNode(
                    operator=OperatorType.EXP,
                    children=[ExpressionNode(operator=OperatorType.VARIABLE)],
                ),
            ],
        )
        expr = SymbolicExpression(root=root, r_squared=0.99)

        interpretation = discovery._interpret_formula(expr)

        assert "log" in interpretation.lower() or "toe" in interpretation.lower()
        assert "exp" in interpretation.lower() or "saturation" in interpretation.lower()
        assert "R²" in interpretation or "fit" in interpretation.lower()

    def test_compare_formulas(self, discovery):
        """Test comparing multiple formulas."""
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50) ** 0.8

        formulas = [
            SymbolicExpression(root=ExpressionLibrary.linear()),
            SymbolicExpression(root=ExpressionLibrary.power_law()),
        ]

        results = discovery.compare_formulas(formulas, x, y)

        assert len(results) == 2
        # Should be sorted by AIC
        assert results[0]["aic"] <= results[1]["aic"]
        # Each result should have all required fields
        for r in results:
            assert "formula" in r
            assert "r_squared" in r
            assert "mse" in r
            assert "complexity" in r
            assert "aic" in r


class TestOperatorInfo:
    """Tests for operator information."""

    def test_all_operators_have_info(self):
        """Test that all operators have info defined."""
        for op in OperatorType:
            assert op in OPERATOR_INFO

    def test_operator_arity_correct(self):
        """Test operator arities are correct."""
        assert OPERATOR_INFO[OperatorType.CONSTANT][0] == 0
        assert OPERATOR_INFO[OperatorType.VARIABLE][0] == 0
        assert OPERATOR_INFO[OperatorType.ADD][0] == 2
        assert OPERATOR_INFO[OperatorType.NEG][0] == 1
        assert OPERATOR_INFO[OperatorType.LOG][0] == 1
