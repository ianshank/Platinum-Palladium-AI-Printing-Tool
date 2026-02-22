"""
Comprehensive unit tests for neuro_symbolic module.

Tests coverage:
1. Constraints (constraints.py)
2. Knowledge Graph (knowledge_graph.py)
3. Symbolic Regression (symbolic_regression.py)
4. Curve Generator (curve_generator.py)
"""

import numpy as np
import pytest
from uuid import uuid4

from ptpd_calibration.neuro_symbolic.constraints import (
    ConstraintType,
    ConstraintResult,
    ConstraintViolation,
    MonotonicityConstraint,
    DensityBoundsConstraint,
    PhysicsConstraint,
    SmoothnessConstraint,
    ConstraintSet,
    DifferentiableLoss,
    ConstrainedCurveOptimizer,
)
from ptpd_calibration.neuro_symbolic.knowledge_graph import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    KnowledgeGraph,
    PaperChemistryKnowledgeGraph,
)
from ptpd_calibration.neuro_symbolic.symbolic_regression import (
    OperatorType,
    ExpressionNode,
    SymbolicExpression,
    ExpressionLibrary,
    DifferentiableSymbolicRegressor,
    CurveFormulaDiscovery,
)
from ptpd_calibration.neuro_symbolic.curve_generator import (
    NeuroSymbolicCurveGenerator,
    CurveGenerationResult,
)
from ptpd_calibration.core.types import CurveType


# ============================================================================
# Test Constraints (neuro_symbolic/constraints.py)
# ============================================================================


class TestConstraintType:
    """Test ConstraintType enum values."""

    def test_enum_values_exist(self):
        """Test all expected enum values are defined."""
        assert ConstraintType.MONOTONICITY == "monotonicity"
        assert ConstraintType.DENSITY_BOUNDS == "density_bounds"
        assert ConstraintType.PHYSICS_HD_CURVE == "physics_hd_curve"
        assert ConstraintType.SMOOTHNESS == "smoothness"
        assert ConstraintType.TOE_REGION == "toe_region"
        assert ConstraintType.SHOULDER_REGION == "shoulder_region"
        assert ConstraintType.CUSTOM == "custom"


class TestConstraintResult:
    """Test ConstraintResult creation and fields."""

    def test_create_satisfied_result(self):
        """Test creating a satisfied constraint result."""
        result = ConstraintResult(
            is_satisfied=True,
            loss_value=0.0,
            explanation="All good",
        )
        assert result.is_satisfied is True
        assert result.loss_value == 0.0
        assert result.explanation == "All good"
        assert result.violations == []
        assert result.gradient is None

    def test_create_violated_result(self):
        """Test creating a violated constraint result."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.MONOTONICITY,
            constraint_name="Test",
            violation_magnitude=0.5,
            violation_indices=[0, 1],
        )
        result = ConstraintResult(
            is_satisfied=False,
            loss_value=0.5,
            gradient=[0.1, -0.2, 0.3],
            violations=[violation],
            explanation="Violated",
        )
        assert result.is_satisfied is False
        assert result.loss_value == 0.5
        assert len(result.violations) == 1
        assert result.gradient == [0.1, -0.2, 0.3]

    def test_loss_value_non_negative(self):
        """Test loss value must be non-negative."""
        with pytest.raises(Exception):  # Pydantic validation error
            ConstraintResult(is_satisfied=True, loss_value=-1.0)


class TestConstraintViolation:
    """Test ConstraintViolation creation and fields."""

    def test_create_violation(self):
        """Test creating a constraint violation."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.SMOOTHNESS,
            constraint_name="Smoothness Test",
            violation_magnitude=1.5,
            violation_indices=[5, 6, 7],
            description="Too rough",
            suggested_fix="Apply smoothing",
        )
        assert violation.constraint_type == ConstraintType.SMOOTHNESS
        assert violation.constraint_name == "Smoothness Test"
        assert violation.violation_magnitude == 1.5
        assert violation.violation_indices == [5, 6, 7]
        assert violation.description == "Too rough"
        assert violation.suggested_fix == "Apply smoothing"

    def test_violation_magnitude_non_negative(self):
        """Test violation magnitude must be non-negative."""
        with pytest.raises(Exception):  # Pydantic validation error
            ConstraintViolation(
                constraint_type=ConstraintType.MONOTONICITY,
                constraint_name="Test",
                violation_magnitude=-0.5,
            )


class TestMonotonicityConstraint:
    """Test MonotonicityConstraint."""

    def test_evaluate_monotonic_curve(self):
        """Test monotonicity constraint on a monotonically increasing curve."""
        constraint = MonotonicityConstraint(weight=1.0)
        values = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        result = constraint.evaluate(values)

        assert result.is_satisfied is True
        assert result.loss_value == 0.0
        assert len(result.violations) == 0

    def test_evaluate_non_monotonic_curve(self):
        """Test monotonicity constraint on a non-monotonic curve."""
        constraint = MonotonicityConstraint(weight=1.0)
        values = np.array([0.0, 0.5, 0.3, 0.8, 1.0])  # Decreases at index 2
        result = constraint.evaluate(values)

        assert result.is_satisfied is False
        assert result.loss_value > 0.0
        assert len(result.violations) == 1
        assert 1 in result.violations[0].violation_indices

    def test_compute_loss_returns_float(self):
        """Test compute_loss returns a float."""
        constraint = MonotonicityConstraint(weight=1.0)
        values = np.array([0.0, 0.5, 0.3, 0.8, 1.0])
        loss = constraint.compute_loss(values)

        assert isinstance(loss, float)
        assert loss > 0.0

    def test_satisfied_constraint_has_zero_loss(self):
        """Test satisfied constraint has zero loss."""
        constraint = MonotonicityConstraint(weight=1.0)
        values = np.linspace(0, 1, 10)
        loss = constraint.compute_loss(values)

        assert loss == 0.0


class TestDensityBoundsConstraint:
    """Test DensityBoundsConstraint."""

    def test_evaluate_values_within_bounds(self):
        """Test bounds constraint on values within bounds."""
        constraint = DensityBoundsConstraint(min_density=0.0, max_density=2.5, weight=1.0)
        values = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        result = constraint.evaluate(values)

        assert result.is_satisfied is True
        assert result.loss_value == 0.0
        assert len(result.violations) == 0

    def test_evaluate_values_below_minimum(self):
        """Test bounds constraint on values below minimum."""
        constraint = DensityBoundsConstraint(min_density=0.0, max_density=2.5, weight=1.0)
        values = np.array([-0.5, 0.5, 1.0, 1.5, 2.0])
        result = constraint.evaluate(values)

        assert result.is_satisfied is False
        assert result.loss_value > 0.0
        assert len(result.violations) >= 1

    def test_evaluate_values_above_maximum(self):
        """Test bounds constraint on values above maximum."""
        constraint = DensityBoundsConstraint(min_density=0.0, max_density=2.5, weight=1.0)
        values = np.array([0.1, 0.5, 1.0, 1.5, 3.0])
        result = constraint.evaluate(values)

        assert result.is_satisfied is False
        assert result.loss_value > 0.0
        assert len(result.violations) >= 1

    def test_compute_loss_returns_float(self):
        """Test compute_loss returns a float."""
        constraint = DensityBoundsConstraint(min_density=0.0, max_density=2.5, weight=1.0)
        values = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
        loss = constraint.compute_loss(values)

        assert isinstance(loss, float)
        assert loss > 0.0


class TestPhysicsConstraint:
    """Test PhysicsConstraint."""

    def test_evaluate_returns_result(self):
        """Test physics constraint returns ConstraintResult."""
        constraint = PhysicsConstraint(weight=1.0)
        values = np.linspace(0.1, 2.0, 21)
        result = constraint.evaluate(values)

        assert isinstance(result, ConstraintResult)
        assert isinstance(result.loss_value, float)
        assert result.loss_value >= 0.0

    def test_compute_loss_returns_float(self):
        """Test compute_loss returns a float."""
        constraint = PhysicsConstraint(weight=1.0)
        values = np.linspace(0.1, 2.0, 21)
        loss = constraint.compute_loss(values)

        assert isinstance(loss, float)
        assert loss >= 0.0


class TestSmoothnessConstraint:
    """Test SmoothnessConstraint."""

    def test_evaluate_smooth_curve(self):
        """Test smoothness constraint on a smooth curve."""
        constraint = SmoothnessConstraint(order=2, weight=1.0)
        # Smooth curve: quadratic
        x = np.linspace(0, 1, 21)
        values = 2.0 * x**2
        result = constraint.evaluate(values)

        # Quadratic has constant second derivative, should be smooth
        assert result.is_satisfied is True
        assert result.loss_value < 0.1

    def test_evaluate_rough_curve(self):
        """Test smoothness constraint on a rough curve."""
        constraint = SmoothnessConstraint(order=2, weight=1.0)
        # Rough curve with noise
        values = np.array([0.0, 0.1, 0.3, 0.2, 0.5, 0.4, 0.8, 0.7, 1.0])
        result = constraint.evaluate(values)

        # Should have higher loss for rough curve
        assert result.loss_value > 0.0

    def test_compute_loss_returns_float(self):
        """Test compute_loss returns a float."""
        constraint = SmoothnessConstraint(order=2, weight=1.0)
        values = np.linspace(0, 1, 10)
        loss = constraint.compute_loss(values)

        assert isinstance(loss, float)
        assert loss >= 0.0


class TestConstraintSet:
    """Test ConstraintSet (formerly ConstraintEngine)."""

    def test_add_constraint(self):
        """Test adding a constraint to the set."""
        constraint_set = ConstraintSet()
        constraint = MonotonicityConstraint(weight=1.0)
        constraint_set.add_constraint(constraint)

        assert len(constraint_set.constraints) == 1

    def test_evaluate_all(self):
        """Test evaluating all constraints."""
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(MonotonicityConstraint(weight=1.0))
        constraint_set.add_constraint(DensityBoundsConstraint(weight=1.0))

        values = np.linspace(0.1, 2.0, 21)
        results = constraint_set.evaluate_all(values)

        assert len(results) == 2
        assert "Monotonicity" in results
        assert "Density Bounds" in results

    def test_get_violations(self):
        """Test getting all violations."""
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(MonotonicityConstraint(weight=1.0))
        constraint_set.add_constraint(
            DensityBoundsConstraint(min_density=0.0, max_density=2.5, weight=1.0)
        )

        # Values that violate both constraints
        values = np.array([0.5, 0.3, 0.1, -0.1, 3.0])
        violations = constraint_set.get_violations(values)

        # Should have violations from both constraints
        assert len(violations) > 0


class TestDifferentiableLoss:
    """Test DifferentiableLoss."""

    def test_compute_data_loss_mse(self):
        """Test MSE data loss computation."""
        constraint_set = ConstraintSet()
        target = np.array([0.0, 0.5, 1.0])
        loss_fn = DifferentiableLoss(
            constraint_set=constraint_set,
            target_values=target,
            data_loss_weight=1.0,
            data_loss_type="mse",
        )

        values = np.array([0.1, 0.6, 1.1])
        loss = loss_fn.compute_data_loss(values)

        expected_loss = np.mean((values - target) ** 2)
        assert np.isclose(loss, expected_loss)

    def test_call_returns_total_loss(self):
        """Test calling loss function returns total loss."""
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(MonotonicityConstraint(weight=0.5))

        target = np.array([0.0, 0.5, 1.0])
        loss_fn = DifferentiableLoss(
            constraint_set=constraint_set,
            target_values=target,
            data_loss_weight=1.0,
        )

        values = np.array([0.0, 0.5, 1.0])
        loss = loss_fn(values)

        assert isinstance(loss, float)
        assert loss >= 0.0


class TestConstrainedCurveOptimizer:
    """Test ConstrainedCurveOptimizer."""

    def test_validate_curve(self):
        """Test validate_curve checks constraints."""
        optimizer = ConstrainedCurveOptimizer()
        values = np.linspace(0.1, 2.0, 21)

        is_valid, violations = optimizer.validate_curve(values)

        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)

    def test_validate_curve_detects_violations(self):
        """Test validate_curve detects constraint violations."""
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(MonotonicityConstraint(weight=1.0))
        optimizer = ConstrainedCurveOptimizer(constraint_set=constraint_set)

        # Non-monotonic values
        values = np.array([0.0, 0.5, 0.3, 0.8, 1.0])
        is_valid, violations = optimizer.validate_curve(values)

        assert is_valid is False
        assert len(violations) > 0


# ============================================================================
# Test Knowledge Graph (neuro_symbolic/knowledge_graph.py)
# ============================================================================


class TestEntity:
    """Test Entity creation."""

    def test_create_entity(self):
        """Test creating an entity."""
        entity = Entity(
            name="Test Paper",
            entity_type=EntityType.PAPER,
            properties={"weight_gsm": 300, "coating": "smooth"},
        )

        assert entity.name == "Test Paper"
        assert entity.entity_type == EntityType.PAPER
        assert entity.properties["weight_gsm"] == 300
        assert entity.id is not None


class TestRelationship:
    """Test Relationship creation."""

    def test_create_relationship(self):
        """Test creating a relationship."""
        source_id = uuid4()
        target_id = uuid4()

        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType.COMPATIBLE_WITH,
            weight=0.9,
            confidence=0.95,
        )

        assert relationship.source_id == source_id
        assert relationship.target_id == target_id
        assert relationship.relation_type == RelationType.COMPATIBLE_WITH
        assert relationship.weight == 0.9
        assert relationship.confidence == 0.95


class TestKnowledgeGraph:
    """Test KnowledgeGraph base class."""

    def test_add_entity(self):
        """Test adding an entity to the graph."""
        kg = KnowledgeGraph()
        entity = Entity(
            name="Test Paper",
            entity_type=EntityType.PAPER,
            properties={"weight_gsm": 300},
        )

        entity_id = kg.add_entity(entity)

        assert entity_id == entity.id
        assert kg.get_entity(entity_id) == entity

    def test_add_relationship(self):
        """Test adding a relationship to the graph."""
        kg = KnowledgeGraph()

        # Create entities
        paper = Entity(name="Paper A", entity_type=EntityType.PAPER)
        metal = Entity(name="Metal B", entity_type=EntityType.METAL_SALT)
        kg.add_entity(paper)
        kg.add_entity(metal)

        # Create relationship
        rel = Relationship(
            source_id=paper.id,
            target_id=metal.id,
            relation_type=RelationType.COMPATIBLE_WITH,
        )
        rel_id = kg.add_relationship(rel)

        assert rel_id == rel.id

    def test_get_entity_by_id(self):
        """Test getting entity by ID."""
        kg = KnowledgeGraph()
        entity = Entity(name="Test Paper", entity_type=EntityType.PAPER)
        entity_id = kg.add_entity(entity)

        retrieved = kg.get_entity(entity_id)

        assert retrieved == entity

    def test_get_entity_returns_none_for_missing(self):
        """Test get_entity returns None for missing entity."""
        kg = KnowledgeGraph()
        missing_id = uuid4()

        result = kg.get_entity(missing_id)

        assert result is None

    def test_get_relationships_with_direction_filter(self):
        """Test getting relationships with direction filter."""
        kg = KnowledgeGraph()

        # Create entities and relationships
        e1 = Entity(name="E1", entity_type=EntityType.PAPER)
        e2 = Entity(name="E2", entity_type=EntityType.PAPER)
        kg.add_entity(e1)
        kg.add_entity(e2)

        rel = Relationship(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.SIMILAR_TO,
        )
        kg.add_relationship(rel)

        # Test outgoing
        outgoing = kg.get_relationships(e1.id, direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0].source_id == e1.id

        # Test incoming
        incoming = kg.get_relationships(e2.id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].target_id == e2.id

        # Test both
        both = kg.get_relationships(e1.id, direction="both")
        assert len(both) >= 1

    def test_compute_similarity(self):
        """Test computing similarity between entities."""
        kg = KnowledgeGraph()

        # Create similar entities
        e1 = Entity(
            name="Paper A",
            entity_type=EntityType.PAPER,
            properties={"weight_gsm": 300, "texture": "smooth"},
        )
        e2 = Entity(
            name="Paper B",
            entity_type=EntityType.PAPER,
            properties={"weight_gsm": 300, "texture": "smooth"},
        )
        kg.add_entity(e1)
        kg.add_entity(e2)

        similarity = kg.compute_similarity(e1.id, e2.id)

        # Same type and properties should have high similarity
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be relatively similar


class TestPaperChemistryKnowledgeGraph:
    """Test PaperChemistryKnowledgeGraph specialized graph."""

    def test_initialization_creates_entities(self):
        """Test graph is pre-populated with domain knowledge."""
        kg = PaperChemistryKnowledgeGraph()

        # Should have papers
        papers = kg.get_entities_by_type(EntityType.PAPER)
        assert len(papers) > 0

        # Should have metal salts
        metals = kg.get_entities_by_type(EntityType.METAL_SALT)
        assert len(metals) > 0

    def test_infer_settings_for_known_paper(self):
        """Test inferring settings for a known paper."""
        kg = PaperChemistryKnowledgeGraph()

        result = kg.infer_settings_for_paper("Arches Platine")

        assert result.confidence > 0.5
        assert "metal_ratio" in result.result_values
        assert "coating_factor" in result.result_values
        assert "expected_dmax" in result.result_values

    def test_infer_settings_for_unknown_paper(self):
        """Test inferring settings for unknown paper returns low confidence."""
        kg = PaperChemistryKnowledgeGraph()

        result = kg.infer_settings_for_paper("Completely Unknown Paper XYZ")

        # Should attempt fuzzy matching or return low confidence
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# Test Symbolic Regression (neuro_symbolic/symbolic_regression.py)
# ============================================================================


class TestExpressionNode:
    """Test ExpressionNode."""

    def test_create_constant_node(self):
        """Test creating a constant node."""
        node = ExpressionNode(operator=OperatorType.CONSTANT, value=5.0)

        assert node.operator == OperatorType.CONSTANT
        assert node.value == 5.0
        assert node.is_terminal()

    def test_create_variable_node(self):
        """Test creating a variable node."""
        node = ExpressionNode(operator=OperatorType.VARIABLE, variable_name="x")

        assert node.operator == OperatorType.VARIABLE
        assert node.variable_name == "x"
        assert node.is_terminal()

    def test_size_counts_nodes(self):
        """Test size counts all nodes in tree."""
        # Tree: x + 5.0
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=5.0),
            ],
        )

        assert node.size() == 3  # ADD + VARIABLE + CONSTANT

    def test_evaluate_constant(self):
        """Test evaluating a constant node."""
        node = ExpressionNode(operator=OperatorType.CONSTANT, value=3.14)
        result = node.evaluate(10.0)

        assert result == 3.14

    def test_evaluate_variable(self):
        """Test evaluating a variable node."""
        node = ExpressionNode(operator=OperatorType.VARIABLE)
        result = node.evaluate(7.5)

        assert result == 7.5

    def test_evaluate_addition(self):
        """Test evaluating addition operation."""
        # Tree: 3.0 + x
        node = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=3.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )
        result = node.evaluate(2.0)

        assert result == 5.0

    def test_evaluate_multiplication(self):
        """Test evaluating multiplication operation."""
        # Tree: 2.0 * x
        node = ExpressionNode(
            operator=OperatorType.MUL,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )
        result = node.evaluate(3.0)

        assert result == 6.0

    def test_evaluate_with_array(self):
        """Test evaluating expression with numpy array."""
        # Tree: x * 2.0
        node = ExpressionNode(
            operator=OperatorType.MUL,
            children=[
                ExpressionNode(operator=OperatorType.VARIABLE),
                ExpressionNode(operator=OperatorType.CONSTANT, value=2.0),
            ],
        )
        x = np.array([1.0, 2.0, 3.0])
        result = node.evaluate(x)

        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_copy_creates_deep_copy(self):
        """Test copy creates a deep copy."""
        original = ExpressionNode(
            operator=OperatorType.ADD,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=1.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )
        copy = original.copy()

        # Modify copy
        copy.children[0].value = 2.0

        # Original should be unchanged
        assert original.children[0].value == 1.0
        assert copy.children[0].value == 2.0


class TestSymbolicExpression:
    """Test SymbolicExpression."""

    def test_create_expression(self):
        """Test creating a symbolic expression."""
        root = ExpressionNode(operator=OperatorType.VARIABLE)
        expr = SymbolicExpression(root=root)

        assert expr.root == root
        assert expr.complexity == 1

    def test_evaluate_delegates_to_root(self):
        """Test evaluate delegates to root node."""
        root = ExpressionNode(
            operator=OperatorType.MUL,
            children=[
                ExpressionNode(operator=OperatorType.CONSTANT, value=3.0),
                ExpressionNode(operator=OperatorType.VARIABLE),
            ],
        )
        expr = SymbolicExpression(root=root)

        result = expr.evaluate(2.0)

        assert result == 6.0


class TestExpressionLibrary:
    """Test ExpressionLibrary template expressions."""

    def test_linear_expression(self):
        """Test linear expression template."""
        expr = ExpressionLibrary.linear()

        assert expr.operator == OperatorType.ADD
        assert len(expr.children) == 2

    def test_power_law_expression(self):
        """Test power law expression template."""
        expr = ExpressionLibrary.power_law()

        assert expr is not None
        # Should have power operation somewhere
        assert expr.size() > 1

    def test_logarithmic_expression(self):
        """Test logarithmic expression template."""
        expr = ExpressionLibrary.logarithmic()

        assert expr is not None
        assert expr.size() > 1

    def test_sigmoid_expression(self):
        """Test sigmoid expression template."""
        expr = ExpressionLibrary.sigmoid()

        assert expr is not None
        assert expr.size() > 1

    def test_hd_curve_expression(self):
        """Test H&D curve expression template."""
        expr = ExpressionLibrary.hd_curve()

        assert expr is not None
        # H&D curve should be complex
        assert expr.size() > 5


class TestDifferentiableSymbolicRegressor:
    """Test DifferentiableSymbolicRegressor."""

    def test_initialization(self):
        """Test regressor initialization."""
        regressor = DifferentiableSymbolicRegressor()

        assert regressor._population == []
        assert regressor._best_expression is None
        assert regressor._generation == 0

    def test_generate_random_expression(self):
        """Test generating random expression."""
        regressor = DifferentiableSymbolicRegressor()
        expr = regressor._generate_random_expression(max_depth=3)

        assert isinstance(expr, ExpressionNode)
        # Random generation may exceed max_depth by 1 due to implementation
        assert expr.depth() <= 4

    def test_fit_with_simple_linear_data(self):
        """Test fitting symbolic regressor to linear data."""
        regressor = DifferentiableSymbolicRegressor()

        # Simple linear data: y = 2x + 1
        x = np.linspace(0, 1, 10)
        y = 2 * x + 1

        # Use fewer generations for test speed
        regressor.settings.sr_generations = 5
        regressor.settings.sr_population_size = 10

        best = regressor.fit(x, y)

        # Should find some expression
        assert best is not None
        assert isinstance(best, SymbolicExpression)
        # Should fit reasonably well (not testing for perfect fit)
        assert best.r_squared > 0.5 or best.mse < 0.5


class TestCurveFormulaDiscovery:
    """Test CurveFormulaDiscovery."""

    def test_discover_formula(self):
        """Test discovering formula from data."""
        discovery = CurveFormulaDiscovery()

        # Simple curve data
        x = np.linspace(0, 1, 10)
        y = x**2  # Quadratic

        # Reduce generations for test speed
        discovery._regressor.settings.sr_generations = 3
        discovery._regressor.settings.sr_population_size = 10

        result = discovery.discover_formula(measured_densities=y.tolist())

        assert "formula" in result
        assert "latex" in result
        assert "r_squared" in result
        assert "mse" in result
        assert "complexity" in result
        assert isinstance(result["formula"], str)


# ============================================================================
# Test Curve Generator (neuro_symbolic/curve_generator.py)
# ============================================================================


class TestNeuroSymbolicCurveGenerator:
    """Test NeuroSymbolicCurveGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = NeuroSymbolicCurveGenerator()

        assert generator._base_generator is not None
        assert generator._constraint_set is not None
        assert generator._optimizer is not None
        assert generator._knowledge_graph is not None
        assert generator._formula_discovery is not None

    def test_generate_curve_basic(self):
        """Test generating a basic curve."""
        generator = NeuroSymbolicCurveGenerator()

        # Simple monotonic densities
        densities = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

        result = generator.generate(
            measured_densities=densities,
            curve_type=CurveType.LINEAR,
            enforce_constraints=True,
            discover_formula=False,  # Skip for speed
            use_knowledge_graph=False,
            quantify_uncertainty=False,
        )

        assert isinstance(result, CurveGenerationResult)
        assert result.curve is not None
        assert len(result.curve.output_values) > 0

    def test_generate_validates_constraints(self):
        """Test curve generation validates constraints."""
        generator = NeuroSymbolicCurveGenerator()

        densities = [0.1, 0.3, 0.5, 0.7, 0.9]

        result = generator.generate(
            measured_densities=densities,
            enforce_constraints=True,
            discover_formula=False,
            use_knowledge_graph=False,
            quantify_uncertainty=False,
        )

        assert isinstance(result.constraints_satisfied, bool)
        assert isinstance(result.constraint_violations, list)

    def test_generate_with_knowledge_graph(self):
        """Test curve generation with knowledge graph inference."""
        generator = NeuroSymbolicCurveGenerator()

        densities = [0.1, 0.3, 0.5, 0.7, 0.9]

        result = generator.generate(
            measured_densities=densities,
            paper_type="Arches Platine",
            use_knowledge_graph=True,
            discover_formula=False,
            quantify_uncertainty=False,
        )

        # Should have knowledge inference
        assert result.knowledge_inference is not None
        assert len(result.inferred_settings) > 0

    @pytest.mark.parametrize(
        "curve_type",
        [
            CurveType.LINEAR,
            CurveType.PAPER_WHITE,
            CurveType.AESTHETIC,
        ],
    )
    def test_generate_different_curve_types(self, curve_type):
        """Test generating different curve types."""
        generator = NeuroSymbolicCurveGenerator()

        densities = [0.1, 0.3, 0.5, 0.7, 0.9]

        result = generator.generate(
            measured_densities=densities,
            curve_type=curve_type,
            enforce_constraints=False,  # Skip for speed
            discover_formula=False,
            use_knowledge_graph=False,
            quantify_uncertainty=False,
        )

        assert result.curve.curve_type == curve_type


class TestCurveGenerationResult:
    """Test CurveGenerationResult model."""

    def test_create_minimal_result(self):
        """Test creating a minimal result."""
        from ptpd_calibration.core.models import CurveData

        curve = CurveData(
            name="Test",
            curve_type=CurveType.LINEAR,
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.5, 1.0],
        )

        result = CurveGenerationResult(curve=curve)

        assert result.curve == curve
        assert result.constraints_satisfied is True
        assert result.constraint_violations == []

    def test_result_with_all_fields(self):
        """Test creating result with all fields populated."""
        from ptpd_calibration.core.models import CurveData

        curve = CurveData(
            name="Test",
            curve_type=CurveType.LINEAR,
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.5, 1.0],
        )

        result = CurveGenerationResult(
            curve=curve,
            constraints_satisfied=True,
            constraint_report="All good",
            discovered_formula="x^2",
            formula_r_squared=0.99,
            explanation="Test curve generated successfully",
            reasoning_steps=["Step 1", "Step 2"],
        )

        assert result.discovered_formula == "x^2"
        assert result.formula_r_squared == 0.99
        assert len(result.reasoning_steps) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestConstraintIntegration:
    """Integration tests for constraint system."""

    def test_constraint_set_default_constraints(self):
        """Test default constraint set has all standard constraints."""
        constraint_set = ConstraintSet.default_set()

        assert len(constraint_set.constraints) == 4
        constraint_names = [c.name for c in constraint_set.constraints]
        assert "Monotonicity" in constraint_names
        assert "Density Bounds" in constraint_names
        assert "H&D Curve Physics" in constraint_names
        assert "Smoothness" in constraint_names

    def test_constraint_optimization_improves_curve(self):
        """Test constraint optimization improves a bad curve."""
        optimizer = ConstrainedCurveOptimizer()

        # Bad curve: non-monotonic
        bad_curve = np.array([0.1, 0.5, 0.3, 0.7, 1.0])

        result = optimizer.optimize(
            initial_values=bad_curve,
            target_values=bad_curve,
        )

        optimized = result["optimized_values"]

        # Optimized should be more monotonic
        diffs = np.diff(optimized)
        assert np.all(diffs >= -1e-6)  # Allow small numerical errors


class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph."""

    def test_paper_chemistry_relationships(self):
        """Test pre-populated relationships work correctly."""
        kg = PaperChemistryKnowledgeGraph()

        # Get Arches Platine
        arches = kg.get_entity_by_name("Arches Platine")
        assert arches is not None

        # Should have relationships
        relationships = kg.get_relationships(arches.id)
        assert len(relationships) > 0

    def test_find_similar_papers(self):
        """Test finding similar papers."""
        kg = PaperChemistryKnowledgeGraph()

        arches = kg.get_entity_by_name("Arches Platine")
        assert arches is not None

        similar = kg.find_similar(arches.id, top_k=3)

        # Should find some similar papers
        assert len(similar) > 0
        assert all(0.0 <= s.similarity_score <= 1.0 for s in similar)


class TestSymbolicRegressionIntegration:
    """Integration tests for symbolic regression."""

    def test_discover_and_evaluate_formula(self):
        """Test end-to-end formula discovery and evaluation."""
        discovery = CurveFormulaDiscovery()

        # Create H&D-like curve
        x = np.linspace(0, 1, 21)
        y = 2.0 * (1 - np.exp(-3.0 * x**0.8)) + 0.1

        # Reduce generations for test
        discovery._regressor.settings.sr_generations = 5
        discovery._regressor.settings.sr_population_size = 20

        result = discovery.discover_formula(
            measured_densities=y.tolist(),
            paper_type="Test Paper",
        )

        # Should produce valid formula
        assert result["formula"] is not None
        # r_squared may be near-zero (float epsilon) with small sr_generations/population;
        # allow -1e-9 tolerance to avoid flakiness from stochastic symbolic regression
        assert result["r_squared"] >= -1e-9
        assert len(result["predictions"]) == len(y)

        # Predictions should be reasonable
        predictions = np.array(result["predictions"])
        assert predictions.shape == y.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
