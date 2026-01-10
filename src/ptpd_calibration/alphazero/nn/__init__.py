"""
Neural network module for AlphaZero.

Provides the PolicyValueNet - a 1D MLP that takes state vectors
and outputs action probabilities and state values.
"""

from ptpd_calibration.alphazero.nn.policy_value_net import PolicyValueNet

__all__ = ["PolicyValueNet"]
