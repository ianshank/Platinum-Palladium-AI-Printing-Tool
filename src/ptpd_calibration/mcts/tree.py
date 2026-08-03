"""
MCTS tree node implementation for calibration optimization.

Provides mutable, performance-critical tree nodes for Monte Carlo Tree Search
using __slots__ for memory efficiency.
"""

from __future__ import annotations

import logging
import math

from ptpd_calibration.mcts.types import CalibrationAction, CalibrationState

logger = logging.getLogger(__name__)


class TreeNode:
    """Mutable tree node for MCTS search.

    Uses __slots__ for memory efficiency. Not a Pydantic model
    because tree operations are performance-critical.
    """

    __slots__ = (
        "state",
        "parent",
        "children",
        "action",
        "visit_count",
        "value_sum",
        "prior",
    )

    def __init__(
        self,
        state: CalibrationState,
        parent: TreeNode | None = None,
        action: CalibrationAction | None = None,
        prior: float = 0.0,
    ):
        """Initialize tree node.

        Args:
            state: CalibrationState for this node
            parent: Parent node, or None if root
            action: Action that led to this node, or None if root
            prior: Prior probability from policy network (0-1)
        """
        self.state = state
        self.parent = parent
        self.children: list[TreeNode] = []
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (all dimensions decided)."""
        return self.state.is_terminal

    @property
    def mean_value(self) -> float:
        """Get mean value of this node.

        Returns:
            Mean value (value_sum / visit_count), or 0 if unvisited
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """Upper Confidence Bound for Trees with PUCT.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
        where Q = mean value, P = prior, N = visit count

        Args:
            c_puct: Exploration constant (higher = more exploration)

        Returns:
            UCB score for this node
        """
        q_value = self.mean_value

        # Exploration term
        parent_visits = self.parent.visit_count if self.parent is not None else 1

        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1.0 + self.visit_count)

        ucb = q_value + exploration

        logger.debug(
            f"UCB: Q={q_value:.3f}, exploration={exploration:.3f}, "
            f"prior={self.prior:.3f}, visits={self.visit_count}, ucb={ucb:.3f}"
        )

        return ucb

    def select_child(self, c_puct: float) -> TreeNode:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant for UCB calculation

        Returns:
            Child node with highest UCB score

        Raises:
            ValueError: If node has no children
        """
        if self.is_leaf:
            raise ValueError("Cannot select child from leaf node")

        # Find child with max UCB
        best_child = max(self.children, key=lambda child: child.ucb_score(c_puct))

        logger.debug(
            f"Selected child with UCB {best_child.ucb_score(c_puct):.3f} "
            f"(visits={best_child.visit_count}, value={best_child.mean_value:.3f})"
        )

        return best_child

    def expand(self, action: CalibrationAction, prior: float = 0.0) -> TreeNode:
        """Create and add child node from action.

        Args:
            action: Action to apply to create child state
            prior: Prior probability for the new child

        Returns:
            New child node
        """
        # Create new state by applying action
        new_state = self.state.apply_action(action)

        # Create child node
        child = TreeNode(
            state=new_state,
            parent=self,
            action=action,
            prior=prior,
        )

        # Add to children list
        self.children.append(child)

        logger.debug(
            f"Expanded node: action={action.dimension}={action.value:.3f}, "
            f"prior={prior:.3f}, new_depth={new_state.depth}"
        )

        return child

    def backpropagate(self, value: float) -> None:
        """Backpropagate value from this node up to root.

        Updates visit count and value sum for this node and all ancestors.

        Args:
            value: Value to backpropagate (typically quality score in [0, 1])
        """
        node: TreeNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            logger.debug(
                f"Backprop: depth={node.state.depth}, "
                f"visits={node.visit_count}, "
                f"mean_value={node.mean_value:.3f}"
            )
            node = node.parent

    def get_visit_distribution(self) -> dict[int, int]:
        """Return mapping of child action bin_index -> visit count.

        Returns:
            Dictionary mapping bin indices to visit counts
        """
        distribution: dict[int, int] = {}
        for child in self.children:
            if child.action is not None:
                distribution[child.action.bin_index] = child.visit_count
        return distribution

    def best_child(self, temperature: float = 0.0) -> TreeNode:
        """Select best child by visit count (with temperature for exploration).

        Args:
            temperature: Temperature for selection. 0 = greedy (highest visit count),
                        higher values = more stochastic sampling by visit distribution.

        Returns:
            Best child node

        Raises:
            ValueError: If node has no children
        """
        if self.is_leaf:
            raise ValueError("Cannot select best child from leaf node")

        if temperature == 0.0:
            # Greedy: pick child with most visits
            best = max(self.children, key=lambda child: child.visit_count)
            logger.debug(
                f"Best child (greedy): visits={best.visit_count}, value={best.mean_value:.3f}"
            )
            return best
        else:
            # Stochastic: sample proportional to visits^(1/temperature)
            import random

            visits = [child.visit_count for child in self.children]
            # Apply temperature
            weights = [v ** (1.0 / temperature) for v in visits]
            total_weight = sum(weights)

            if total_weight == 0:
                # All children unvisited, pick randomly
                best = random.choice(self.children)
                logger.debug("Best child (random): all unvisited")
                return best

            # Normalize and sample
            probabilities = [w / total_weight for w in weights]
            best = random.choices(self.children, weights=probabilities, k=1)[0]
            logger.debug(
                f"Best child (temp={temperature:.2f}): "
                f"visits={best.visit_count}, "
                f"value={best.mean_value:.3f}"
            )
            return best
