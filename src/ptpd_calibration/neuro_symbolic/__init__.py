"""
Neuro-Symbolic AI Module for Platinum/Palladium Calibration.

This module provides advanced neuro-symbolic AI capabilities for curve generation,
including:
- Physics-informed symbolic constraints with differentiable loss terms
- Knowledge graph for paper/chemistry relationship reasoning
- Differentiable symbolic regression for discovering interpretable curve formulas

The neuro-symbolic approach combines:
- Neural networks for learning complex patterns from calibration data
- Symbolic reasoning for encoding physical constraints and expert knowledge
- Explainable outputs for interpretable predictions

References:
- Garcez et al. (2023) "Neural-Symbolic AI: The Third Wave"
- Karniadakis et al. (2021) "Physics-Informed Machine Learning"
"""

from ptpd_calibration.neuro_symbolic.constraints import (
    ConstrainedCurveOptimizer,
    ConstraintSet,
    ConstraintType,
    DensityBoundsConstraint,
    DifferentiableLoss,
    MonotonicityConstraint,
    PhysicsConstraint,
    SymbolicConstraint,
)
from ptpd_calibration.neuro_symbolic.curve_generator import (
    NeuroSymbolicCurveGenerator,
)
from ptpd_calibration.neuro_symbolic.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    PaperChemistryKnowledgeGraph,
    Relationship,
    RelationType,
)
from ptpd_calibration.neuro_symbolic.symbolic_regression import (
    CurveFormulaDiscovery,
    DifferentiableSymbolicRegressor,
    ExpressionLibrary,
    ExpressionNode,
    OperatorType,
    SymbolicExpression,
)

__all__ = [
    # Constraints
    "ConstraintType",
    "SymbolicConstraint",
    "MonotonicityConstraint",
    "DensityBoundsConstraint",
    "PhysicsConstraint",
    "ConstraintSet",
    "DifferentiableLoss",
    "ConstrainedCurveOptimizer",
    # Knowledge Graph
    "EntityType",
    "RelationType",
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "PaperChemistryKnowledgeGraph",
    # Symbolic Regression
    "OperatorType",
    "ExpressionNode",
    "SymbolicExpression",
    "ExpressionLibrary",
    "DifferentiableSymbolicRegressor",
    "CurveFormulaDiscovery",
    # Curve Generator
    "NeuroSymbolicCurveGenerator",
]
