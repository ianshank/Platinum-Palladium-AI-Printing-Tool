"""
MCTS engine for calibration parameter optimization.

Implements AlphaZero-style Monte Carlo Tree Search with progressive widening
for continuous action spaces.
"""

from __future__ import annotations

import logging
import random
import time

from ptpd_calibration.mcts.config import (
    DEFAULT_PARAMETER_RANGES,
    MCTSSettings,
    PhysicsConstants,
)
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator
from ptpd_calibration.mcts.tree import TreeNode
from ptpd_calibration.mcts.types import (
    CalibrationAction,
    CalibrationState,
    SearchResult,
)

logger = logging.getLogger(__name__)


class MCTSEngine:
    """Monte Carlo Tree Search engine for calibration optimization.

    Implements SELECT → EXPAND → EVALUATE → BACKPROPAGATE loop
    with progressive widening for continuous action spaces.
    """

    def __init__(
        self,
        settings: MCTSSettings | None = None,
        physics: PhysicsConstants | None = None,
        simulator: ExtendedProcessSimulator | None = None,
        scorer: QualityScorer | None = None,
    ):
        """Initialize MCTS engine.

        Args:
            settings: MCTS configuration. If None, uses defaults.
            physics: Physics model parameters. If None, uses defaults.
            simulator: Process simulator. If None, creates one with physics.
            scorer: Quality scorer. If None, creates one with settings.
        """
        self.settings = settings or MCTSSettings()
        self.physics = physics or PhysicsConstants()
        self.simulator = simulator or ExtendedProcessSimulator(
            physics=self.physics,
            settings=self.settings,
        )
        self.scorer = scorer or QualityScorer(settings=self.settings)

        logger.info(
            f"Initialized MCTSEngine: {self.settings.num_simulations} simulations, "
            f"c_puct={self.settings.c_puct:.2f}, "
            f"progressive_widening_alpha={self.settings.progressive_widening_alpha:.2f}"
        )

    def search(
        self,
        target_curve: list[float] | None = None,
        fixed_parameters: dict[str, float] | None = None,
        paper_type: str | None = None,
        uv_source: str | None = None,
    ) -> SearchResult:
        """Run MCTS search for optimal calibration parameters.

        1. Create root state from decision_order (minus fixed params)
        2. For each simulation:
           a. SELECT: Traverse using PUCT to find leaf
           b. EXPAND: If not terminal and progressive widening allows, add child
           c. EVALUATE: Terminal → simulator + quality scorer. Non-terminal → rollout or value network
           d. BACKPROPAGATE: Update values up to root
        3. Extract best parameters from root visit distribution

        Args:
            target_curve: Optional target density curve for quality scoring
            fixed_parameters: Parameters to fix (not search over)
            paper_type: Paper type for context
            uv_source: UV source type for context

        Returns:
            SearchResult with best parameters and quality metrics
        """
        start_time = time.time()

        # 1. Create initial state
        initial_state = self._create_initial_state(
            fixed_parameters=fixed_parameters,
            paper_type=paper_type,
            uv_source=uv_source,
        )

        root = TreeNode(state=initial_state)

        logger.info(
            f"Starting MCTS search: {self.settings.num_simulations} simulations, "
            f"{len(initial_state.remaining_dimensions)} dimensions to explore"
        )

        # 2. Run simulations
        for sim_idx in range(self.settings.num_simulations):
            logger.debug(f"Simulation {sim_idx + 1}/{self.settings.num_simulations}")

            # a. SELECT
            leaf = self._select(root)

            # b. EXPAND (if not terminal and progressive widening allows)
            if not leaf.is_terminal and self._should_expand(leaf):
                leaf = self._expand(leaf)

            # c. EVALUATE
            value = self._evaluate(leaf, target_curve)

            # d. BACKPROPAGATE
            leaf.backpropagate(value)

            # Log progress periodically
            if (sim_idx + 1) % 100 == 0:
                logger.info(
                    f"Progress: {sim_idx + 1}/{self.settings.num_simulations} simulations, "
                    f"root visits={root.visit_count}, "
                    f"mean_value={root.mean_value:.3f}"
                )

        # 3. Extract results
        search_time = time.time() - start_time
        result = self._extract_result(
            root=root,
            target_curve=target_curve,
            search_time=search_time,
            paper_type=paper_type,
            uv_source=uv_source,
        )

        logger.info(
            f"Search complete: quality={result.quality_score:.3f}, "
            f"time={search_time:.2f}s, "
            f"{len(result.alternatives)} alternatives"
        )

        return result

    def _create_initial_state(
        self,
        fixed_parameters: dict[str, float] | None,
        paper_type: str | None,
        uv_source: str | None,
    ) -> CalibrationState:
        """Create initial state with fixed params pre-decided.

        Args:
            fixed_parameters: Parameters to fix (not search over)
            paper_type: Paper type for context
            uv_source: UV source type for context

        Returns:
            Initial CalibrationState for root node
        """
        fixed_params = fixed_parameters or {}

        # Start with all dimensions from decision_order
        remaining_dimensions = [
            dim for dim in self.settings.decision_order if dim not in fixed_params
        ]

        # Fixed parameters are already decided
        decided_parameters = dict(fixed_params)

        initial_state = CalibrationState(
            decided_parameters=decided_parameters,
            remaining_dimensions=remaining_dimensions,
            depth=0,
            paper_type=paper_type,
            uv_source=uv_source,
        )

        logger.debug(
            f"Initial state: {len(decided_parameters)} fixed, {len(remaining_dimensions)} to search"
        )

        return initial_state

    def _select(self, node: TreeNode) -> TreeNode:
        """SELECT phase: traverse tree using UCB to find leaf.

        Args:
            node: Starting node (typically root)

        Returns:
            Leaf node to expand or evaluate
        """
        current = node

        while not current.is_leaf and not current.is_terminal:
            # Select child with highest UCB score
            current = current.select_child(self.settings.c_puct)

        logger.debug(
            f"Selected leaf at depth {current.state.depth}, terminal={current.is_terminal}"
        )

        return current

    def _should_expand(self, node: TreeNode) -> bool:
        """Check progressive widening condition.

        Expand when N(node) > c * k^alpha
        where k = number of existing children, N = visit count

        Args:
            node: Node to check for expansion

        Returns:
            True if node should be expanded
        """
        if node.is_terminal:
            return False

        k = len(node.children)
        n = node.visit_count

        # Progressive widening formula
        threshold = self.settings.progressive_widening_c * (
            k**self.settings.progressive_widening_alpha
        )

        should_expand = n > threshold and k < self.settings.max_actions_per_node

        logger.debug(
            f"Progressive widening: N={n}, k={k}, threshold={threshold:.1f}, expand={should_expand}"
        )

        return should_expand

    def _expand(self, node: TreeNode) -> TreeNode:
        """EXPAND phase: add new child to node.

        Generate action from policy (or uniform if no network).
        Use progressive widening to limit branching.

        Args:
            node: Node to expand

        Returns:
            New child node
        """
        # Generate action for current dimension
        action = self._generate_action(node.state)

        # Uniform prior (no neural network yet)
        prior = 1.0 / self.settings.action_bins

        # Create and return child
        child = node.expand(action, prior=prior)

        logger.debug(
            f"Expanded: {action.dimension}={action.value:.3f} "
            f"(bin {action.bin_index}/{self.settings.action_bins})"
        )

        return child

    def _generate_action(self, state: CalibrationState) -> CalibrationAction:
        """Generate action for current dimension.

        Without neural network: sample uniformly from parameter range.
        With network: use policy output (Phase 4 integration point).

        Args:
            state: Current state

        Returns:
            CalibrationAction for next dimension
        """
        # Get current dimension to decide
        dimension = state.current_dimension
        if dimension is None:
            raise ValueError("Cannot generate action for terminal state")

        # Get parameter range
        param_range = DEFAULT_PARAMETER_RANGES.get(dimension)
        if param_range is None:
            raise ValueError(f"Unknown parameter dimension: {dimension}")

        # Sample bin index uniformly
        bin_index = random.randint(0, self.settings.action_bins - 1)

        # Convert bin to value
        bin_fraction = bin_index / (self.settings.action_bins - 1)
        value = param_range.min_value + bin_fraction * (
            param_range.max_value - param_range.min_value
        )

        action = CalibrationAction(
            dimension=dimension,
            value=value,
            bin_index=bin_index,
        )

        logger.debug(
            f"Generated action: {dimension}={value:.3f} "
            f"(bin {bin_index}, range [{param_range.min_value}, {param_range.max_value}])"
        )

        return action

    def _evaluate(self, node: TreeNode, target_curve: list[float] | None) -> float:
        """EVALUATE phase: get quality score for node.

        Terminal: simulate and score.
        Non-terminal: rollout to terminal with random actions, then score.

        Args:
            node: Node to evaluate
            target_curve: Optional target curve for quality scoring

        Returns:
            Quality score in [0, 1]
        """
        state = node.state

        # Get complete parameter set
        parameters = state.decided_parameters if state.is_terminal else self._rollout(state)

        # Simulate to get density curve
        sim_result = self.simulator.simulate(parameters)

        # Score quality
        quality = self.scorer.score(sim_result, target_curve=target_curve)

        # Update simulation result with quality score
        sim_result.quality_score = quality

        logger.debug(
            f"Evaluated: quality={quality:.3f}, "
            f"dmax={sim_result.dmax:.2f}, "
            f"terminal={state.is_terminal}"
        )

        return quality

    def _rollout(self, state: CalibrationState) -> dict[str, float]:
        """Random rollout to complete a partial state.

        Fill remaining dimensions with uniform random values
        from their parameter ranges.

        Args:
            state: Partial state to complete

        Returns:
            Complete parameter dictionary
        """
        # Start with decided parameters
        parameters = dict(state.decided_parameters)

        # Fill in remaining dimensions with random values
        for dimension in state.remaining_dimensions:
            param_range = DEFAULT_PARAMETER_RANGES.get(dimension)
            if param_range is None:
                logger.warning(f"Unknown parameter dimension: {dimension}")
                continue

            # Sample uniformly
            value = random.uniform(param_range.min_value, param_range.max_value)
            parameters[dimension] = value

            logger.debug(
                f"Rollout: {dimension}={value:.3f} "
                f"(range [{param_range.min_value}, {param_range.max_value}])"
            )

        return parameters

    def _get_temperature(self, step: int) -> float:
        """Get temperature for current search step.

        Linear decay from temperature_initial to temperature_final
        over temperature_decay_steps.

        Args:
            step: Current simulation step

        Returns:
            Temperature value
        """
        if step >= self.settings.temperature_decay_steps:
            return self.settings.temperature_final

        # Linear interpolation
        progress = step / self.settings.temperature_decay_steps
        temperature = self.settings.temperature_initial + progress * (
            self.settings.temperature_final - self.settings.temperature_initial
        )

        return temperature

    def _extract_result(
        self,
        root: TreeNode,
        target_curve: list[float] | None,
        search_time: float,
        paper_type: str | None,
        uv_source: str | None,
    ) -> SearchResult:
        """Extract SearchResult from completed search tree.

        Follow best path from root to get best_parameters.
        Collect visit distribution for each dimension.
        Find top-N alternatives.

        Args:
            root: Root node of search tree
            target_curve: Optional target curve used in search
            search_time: Time taken for search (seconds)
            paper_type: Paper type for context
            uv_source: UV source type for context

        Returns:
            SearchResult with best parameters and alternatives
        """
        # Follow best path (greedy by visit count) to get best parameters
        best_params: dict[str, float] = dict(root.state.decided_parameters)
        current = root

        visit_distribution: dict[str, list[float]] = {}

        while not current.is_terminal:
            if current.is_leaf:
                # Incomplete search - complete with defaults
                logger.warning("Incomplete search tree, using defaults for remaining dimensions")
                for dim in current.state.remaining_dimensions:
                    param_range = DEFAULT_PARAMETER_RANGES.get(dim)
                    if param_range:
                        best_params[dim] = param_range.default_value
                break

            # Get visit distribution for this dimension
            dimension = current.state.current_dimension
            if dimension is not None:
                # Collect visit counts across bins
                visit_dist = current.get_visit_distribution()
                # Convert to list indexed by bin
                dist_list = [0.0] * self.settings.action_bins
                total_visits = sum(visit_dist.values())
                for bin_idx, count in visit_dist.items():
                    if 0 <= bin_idx < len(dist_list) and total_visits > 0:
                        dist_list[bin_idx] = count / total_visits
                visit_distribution[dimension] = dist_list

            # Move to best child
            current = current.best_child(temperature=0.0)

            # Add to best parameters
            if current.action is not None:
                best_params[current.action.dimension] = current.action.value

        # Simulate best parameters to get predicted curve
        sim_result = self.simulator.simulate(best_params)
        quality_score = self.scorer.score(sim_result, target_curve=target_curve)

        # Find alternatives (other high-quality parameter sets)
        alternatives = self._extract_alternatives(root, target_curve, top_n=5)

        result = SearchResult(
            best_parameters=best_params,
            predicted_curve=sim_result.density_curve,
            quality_score=quality_score,
            visit_distribution=visit_distribution,
            num_simulations=self.settings.num_simulations,
            search_time_seconds=search_time,
            constraint_violations=sim_result.constraint_violations,
            alternatives=alternatives,
            paper_type=paper_type,
            uv_source=uv_source,
        )

        return result

    def _extract_alternatives(
        self,
        root: TreeNode,
        target_curve: list[float] | None,
        top_n: int = 5,
    ) -> list[dict[str, float]]:
        """Extract top-N alternative parameter sets from search tree.

        Args:
            root: Root node of search tree
            target_curve: Optional target curve for quality scoring
            top_n: Number of alternatives to extract

        Returns:
            List of parameter dictionaries, sorted by quality (best first)
        """
        # Collect all terminal nodes via DFS
        terminal_nodes: list[tuple[TreeNode, dict[str, float]]] = []

        def collect_terminals(node: TreeNode, params: dict[str, float]) -> None:
            """Recursively collect terminal nodes and their parameters."""
            if node.is_terminal:
                terminal_nodes.append((node, dict(params)))
                return

            for child in node.children:
                if child.action is not None:
                    child_params = {**params, child.action.dimension: child.action.value}
                    collect_terminals(child, child_params)

        collect_terminals(root, dict(root.state.decided_parameters))

        # If we have fewer terminals than top_n, generate random rollouts
        while len(terminal_nodes) < top_n and len(terminal_nodes) < 20:
            # Random rollout from root
            rollout_params = self._rollout(root.state)
            # Create a dummy node for consistency
            dummy_node = TreeNode(
                state=CalibrationState(
                    decided_parameters=rollout_params,
                    remaining_dimensions=[],
                    depth=len(self.settings.decision_order),
                )
            )
            terminal_nodes.append((dummy_node, rollout_params))

        # Score each terminal node
        scored_alternatives: list[tuple[float, dict[str, float]]] = []
        for node, params in terminal_nodes:
            sim_result = self.simulator.simulate(params)
            quality = self.scorer.score(sim_result, target_curve=target_curve)
            scored_alternatives.append((quality, params))

        # Sort by quality (descending)
        scored_alternatives.sort(key=lambda x: x[0], reverse=True)

        # Return top-N parameter dicts (excluding the best, which is already in best_parameters)
        alternatives = [params for _, params in scored_alternatives[1 : top_n + 1]]

        logger.debug(
            f"Extracted {len(alternatives)} alternatives from {len(terminal_nodes)} terminals"
        )

        return alternatives
