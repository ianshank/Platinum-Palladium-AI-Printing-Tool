"""
Monte Carlo Tree Search (MCTS) module for calibration optimization.

Provides an AlphaZero-style search engine for finding optimal
platinum/palladium printing parameters through Expert Iteration
and neural network-guided exploration.

Modules:
    config: MCTSSettings, ParameterRange, PhysicsConstants
    types: CalibrationState, CalibrationAction, SearchResult, etc.
    simulator: ExtendedProcessSimulator
    quality: QualityScorer
    constraints: Photochemistry constraints + ActionPruner
    tree: TreeNode for MCTS search tree
    engine: MCTSEngine search algorithm
    networks: DualNetwork, StateEncoder (requires PyTorch)
    training: ReplayBuffer, MCTSTrainer (requires PyTorch)
    export: MCTSResultExporter
    agents: ChemistrySubagent, ExposureSubagent, CalibrationCoordinatorSubagent
"""

from ptpd_calibration.mcts.config import (
    DEFAULT_PARAMETER_RANGES,
    MCTSSettings,
    ParameterRange,
    PhysicsConstants,
)
from ptpd_calibration.mcts.engine import MCTSEngine
from ptpd_calibration.mcts.export import MCTSResultExporter
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator
from ptpd_calibration.mcts.tree import TreeNode
from ptpd_calibration.mcts.types import (
    CalibrationAction,
    CalibrationState,
    SearchResult,
    SimulationResult,
    TrainingExample,
    TrainingMetrics,
)

__all__ = [
    # Config
    "MCTSSettings",
    "ParameterRange",
    "PhysicsConstants",
    "DEFAULT_PARAMETER_RANGES",
    # Types
    "CalibrationAction",
    "CalibrationState",
    "SearchResult",
    "SimulationResult",
    "TrainingExample",
    "TrainingMetrics",
    # Core
    "ExtendedProcessSimulator",
    "QualityScorer",
    "TreeNode",
    "MCTSEngine",
    # Export
    "MCTSResultExporter",
]
