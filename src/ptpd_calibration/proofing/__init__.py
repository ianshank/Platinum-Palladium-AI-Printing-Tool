"""
Soft proofing simulation for visualizing final print appearance.

Simulate how prints will look on paper, accounting for Dmax and paper white.
"""

from ptpd_calibration.proofing.simulation import (
    SoftProofer,
    ProofSettings,
    ProofResult,
    PaperSimulation,
    PAPER_PRESETS,
)

__all__ = [
    "SoftProofer",
    "ProofSettings",
    "ProofResult",
    "PaperSimulation",
    "PAPER_PRESETS",
]
