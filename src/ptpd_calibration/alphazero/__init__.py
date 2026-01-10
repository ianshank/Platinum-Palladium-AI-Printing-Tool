"""
AlchemistZero: AlphaZero-based optimization for Pt/Pd printing calibration.

This module implements an AlphaZero-style reinforcement learning system
that treats platinum/palladium printing calibration as a strategy game.
The agent learns to optimize printing parameters through self-play,
using Monte Carlo Tree Search (MCTS) guided by a neural network.

Key Components:
- PlatinumGame: Game interface treating calibration as a strategy game
- PolicyValueNet: 1D MLP neural network for policy and value prediction
- AlphaMCTS: Monte Carlo Tree Search for action exploration
- PrintingSimulator: Environment wrapper for the printing process
"""

from ptpd_calibration.alphazero.config import AlphaZeroConfig

__all__ = [
    "AlphaZeroConfig",
]

# Lazy imports to avoid loading PyTorch at module import time
def get_platinum_game():
    """Get the PlatinumGame class."""
    from ptpd_calibration.alphazero.game.platinum_game import PlatinumGame
    return PlatinumGame


def get_policy_value_net():
    """Get the PolicyValueNet class."""
    from ptpd_calibration.alphazero.nn.policy_value_net import PolicyValueNet
    return PolicyValueNet


def get_alpha_mcts():
    """Get the AlphaMCTS class."""
    from ptpd_calibration.alphazero.mcts.alpha_mcts import AlphaMCTS
    return AlphaMCTS


def get_printing_simulator():
    """Get the PrintingSimulator class."""
    from ptpd_calibration.alphazero.bridge.printing_env import PrintingSimulator
    return PrintingSimulator
