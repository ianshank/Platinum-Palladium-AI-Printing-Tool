"""
SCI-05 golden fixtures for the MCTS search and the process simulator.

``search_seed{N}_sims200_bins11.json`` snapshot a full six-dimension search
with default settings for seeds 0, 1 and 2: ``best_parameters`` and
``quality_score`` (9 dp), the root visit count, the integer per-dimension visit
counts along the greedy path, the total node count and the alternatives.

``simulator_numpy_5x21.json`` snapshots ``ExtendedProcessSimulator.simulate``
on the NumPy path for five fixed parameter sets at 21 steps, so the physics
can be regression-tested without PyTorch (and independently of the search).

Both use the NumPy simulator path explicitly: the float32 torch path is
covered by the torch-vs-NumPy differential (TST tier ``dl``), not by these
fixtures.

Regenerate after an intentional change to the search or the physics with::

    PTPD_UPDATE_GOLDENS=1 .venv/bin/python -m pytest tests/golden/mcts -q -o addopts=""
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ptpd_calibration.mcts.config import MCTSSettings
from ptpd_calibration.mcts.engine import MCTSEngine
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator
from ptpd_calibration.mcts.tree import TreeNode
from ptpd_calibration.mcts.types import SearchResult
from tests.golden.conftest import GoldenFile

GOLDEN_SEEDS: tuple[int, ...] = (0, 1, 2)
NUM_SIMULATIONS = 200
ACTION_BINS = 11
SIMULATOR_STEPS = 21

# Settings fields that influence a search result. They are stored in the
# fixture so a changed default shows up as a named diff, not a bare float mismatch.
_SETTINGS_FIELDS = (
    "num_simulations",
    "action_bins",
    "c_puct",
    "decision_order",
    "progressive_widening_alpha",
    "progressive_widening_c",
    "max_actions_per_node",
    "target_dmax",
    "target_dmin",
    "linearity_weight",
    "dmax_weight",
    "smoothness_weight",
    "cost_weight",
    "linearity_decay_rate",
    "dmax_scoring_sigma",
    "smoothness_decay_rate",
    "cost_metal_weight",
)

SIMULATOR_CASES: list[dict[str, Any]] = [
    {
        "name": "defaults",
        "params": {
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        },
    },
    {
        "name": "pure_pd_thin_short",
        "params": {
            "metal_ratio": 0.0,
            "coating_weight": 0.5,
            "ferric_oxalate_pct": 15.0,
            "exposure_time": 30.0,
            "developer_temp": 20.0,
            "humidity": 30.0,
        },
    },
    {
        "name": "pure_pt_heavy_long",
        "params": {
            "metal_ratio": 1.0,
            "coating_weight": 3.0,
            "ferric_oxalate_pct": 27.0,
            "exposure_time": 600.0,
            "developer_temp": 50.0,
            "humidity": 75.0,
        },
    },
    {
        "name": "pd_leaning_mid",
        "params": {
            "metal_ratio": 0.25,
            "coating_weight": 2.0,
            "ferric_oxalate_pct": 18.0,
            "exposure_time": 120.0,
            "developer_temp": 30.0,
            "humidity": 40.0,
        },
    },
    {
        "name": "pt_leaning_warm",
        "params": {
            "metal_ratio": 0.75,
            "coating_weight": 1.0,
            "ferric_oxalate_pct": 24.0,
            "exposure_time": 300.0,
            "developer_temp": 40.0,
            "humidity": 65.0,
        },
    },
]


def golden_settings(seed: int) -> MCTSSettings:
    """Default settings except the documented simulation budget, bins and seed."""
    return MCTSSettings(num_simulations=NUM_SIMULATIONS, action_bins=ACTION_BINS, seed=seed)


def _count_nodes(node: TreeNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def greedy_path_visit_counts(root: TreeNode, action_bins: int) -> dict[str, list[int]]:
    """Integer visit counts per bin for every dimension along the greedy path."""
    counts: dict[str, list[int]] = {}
    node = root
    while not node.is_terminal and not node.is_leaf:
        dimension = node.state.current_dimension
        assert dimension is not None
        per_bin = [0] * action_bins
        for bin_index, visits in node.get_visit_distribution().items():
            per_bin[bin_index] = visits
        counts[dimension] = per_bin
        node = node.best_child(temperature=0.0)
    return counts


def snapshot_search(engine: MCTSEngine, result: SearchResult) -> dict[str, Any]:
    """JSON-serialisable snapshot of everything a seed must determine."""
    root = engine.last_root
    assert root is not None
    settings = engine.settings
    return {
        "seed": engine.seed,
        "settings": {name: getattr(settings, name) for name in _SETTINGS_FIELDS},
        "best_parameters": dict(result.best_parameters),
        "quality_score": result.quality_score,
        "root_visit_count": root.visit_count,
        "total_nodes": _count_nodes(root),
        "visit_counts": greedy_path_visit_counts(root, settings.action_bins),
        "alternatives": [dict(alt) for alt in result.alternatives],
    }


@pytest.mark.parametrize("seed", GOLDEN_SEEDS)
def test_search_golden(seed: int, golden: Callable[[str], GoldenFile]) -> None:
    """A seeded search reproduces its committed fixture (ints exact, floats 1e-9)."""
    engine = MCTSEngine(settings=golden_settings(seed))
    engine.simulator.use_torch = False  # goldens are for the NumPy physics path
    result = engine.search()

    snapshot = snapshot_search(engine, result)
    assert snapshot["root_visit_count"] == NUM_SIMULATIONS
    assert set(snapshot["visit_counts"]) == set(engine.settings.decision_order)

    golden(f"search_seed{seed}_sims{NUM_SIMULATIONS}_bins{ACTION_BINS}.json").check(snapshot)


def test_search_goldens_are_distinct() -> None:
    """The three fixtures must not collapse onto one result (seed actually used)."""
    results = []
    for seed in GOLDEN_SEEDS:
        engine = MCTSEngine(settings=golden_settings(seed))
        engine.simulator.use_torch = False
        results.append(engine.search().best_parameters)
    assert len({tuple(sorted(r.items())) for r in results}) == len(GOLDEN_SEEDS)


def snapshot_simulator(simulator: ExtendedProcessSimulator) -> dict[str, Any]:
    """Simulate every case on the NumPy path and collect curve plus metrics."""
    cases = []
    for case in SIMULATOR_CASES:
        sim = simulator.simulate(dict(case["params"]), num_steps=SIMULATOR_STEPS)
        cases.append(
            {
                "name": case["name"],
                "params": dict(case["params"]),
                "density_curve": list(sim.density_curve),
                "dmin": sim.dmin,
                "dmax": sim.dmax,
                "density_range": sim.density_range,
                "gamma": sim.gamma,
            }
        )
    return {"num_steps": SIMULATOR_STEPS, "cases": cases}


def test_simulator_numpy_golden(golden: Callable[[str], GoldenFile]) -> None:
    """The NumPy simulator reproduces 5 fixed parameter sets x 21 steps to 1e-9."""
    simulator = ExtendedProcessSimulator()
    simulator.use_torch = False
    snapshot = snapshot_simulator(simulator)

    assert len(snapshot["cases"]) == len(SIMULATOR_CASES)
    for case in snapshot["cases"]:
        assert len(case["density_curve"]) == SIMULATOR_STEPS
        assert case["density_curve"] == sorted(case["density_curve"]), case["name"]

    golden(f"simulator_numpy_{len(SIMULATOR_CASES)}x{SIMULATOR_STEPS}.json").check(snapshot)
