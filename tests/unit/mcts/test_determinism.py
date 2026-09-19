"""
SCI-05: seeded determinism for the MCTS engine, tree, replay buffer and trainer.

Every stochastic call site in the MCTS path draws from an injectable
``RandomSource``. With a seed, two independent runs must be bit-identical;
with different seeds they must differ; with no seed the module-level ``random``
generator is used exactly as before.
"""

from __future__ import annotations

import random

import pytest

from ptpd_calibration.mcts.config import MCTSSettings
from ptpd_calibration.mcts.engine import MCTSEngine
from ptpd_calibration.mcts.training import TORCH_AVAILABLE, ReplayBuffer
from ptpd_calibration.mcts.tree import TreeNode
from ptpd_calibration.mcts.types import (
    CalibrationAction,
    CalibrationState,
    SearchResult,
    TrainingExample,
    make_rng,
)

NUM_SIMULATIONS = 100
ACTION_BINS = 11


def _settings(**overrides: object) -> MCTSSettings:
    """Fast settings over all six decision dimensions."""
    return MCTSSettings(num_simulations=NUM_SIMULATIONS, action_bins=ACTION_BINS, **overrides)


def _snapshot(result: SearchResult) -> tuple:
    """The parts of a SearchResult that a seed must fully determine."""
    return (
        result.best_parameters,
        result.quality_score,
        result.visit_distribution,
        result.alternatives,
        result.predicted_curve,
    )


def _search(
    seed: int | None, settings: MCTSSettings | None = None
) -> tuple[MCTSEngine, SearchResult]:
    engine = MCTSEngine(settings=settings or _settings(), seed=seed)
    return engine, engine.search()


class TestMakeRng:
    """The shared factory behind every seed parameter."""

    def test_none_returns_module_level_random(self) -> None:
        assert make_rng(None) is random

    def test_seed_returns_private_generator(self) -> None:
        rng = make_rng(3)
        assert isinstance(rng, random.Random)
        assert rng is not random

    def test_same_seed_same_stream(self) -> None:
        a, b = make_rng(11), make_rng(11)
        assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]

    def test_private_generator_is_isolated_from_global_state(self) -> None:
        random.seed(0)
        before = make_rng(7).random()
        random.seed(12345)
        after = make_rng(7).random()
        assert before == after


class TestEngineSeed:
    """MCTSEngine.search() is a pure function of its seed."""

    def test_same_seed_twice_identical(self) -> None:
        _, first = _search(0)
        _, second = _search(0)
        assert _snapshot(first) == _snapshot(second)

    def test_same_seed_identical_integer_visit_counts(self) -> None:
        engine_a, _ = _search(0)
        engine_b, _ = _search(0)
        assert engine_a.last_root is not None and engine_b.last_root is not None
        assert engine_a.last_root.visit_count == NUM_SIMULATIONS
        assert engine_a.last_root.get_visit_distribution() == (
            engine_b.last_root.get_visit_distribution()
        )

    def test_different_seeds_differ_in_at_least_one_dimension(self) -> None:
        _, a = _search(0)
        _, b = _search(1)
        differing = [
            dim for dim, value in a.best_parameters.items() if b.best_parameters[dim] != value
        ]
        assert differing, "seed is being ignored: identical best_parameters for seeds 0 and 1"

    def test_seed_none_still_works(self) -> None:
        engine, result = _search(None)
        assert engine.seed is None
        assert isinstance(result, SearchResult)
        assert set(result.best_parameters) == set(engine.settings.decision_order)
        assert 0.0 <= result.quality_score <= 1.0

    def test_engine_seed_overrides_settings_seed(self) -> None:
        settings = _settings(seed=5)
        engine = MCTSEngine(settings=settings, seed=9)
        assert engine.seed == 9

    def test_settings_seed_used_when_engine_seed_absent(self) -> None:
        _, from_settings = _search(None, _settings(seed=4))
        _, explicit = _search(4)
        assert _snapshot(from_settings) == _snapshot(explicit)

    def test_env_var_seed_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_MCTS_SEED", "7")
        settings = MCTSSettings(num_simulations=NUM_SIMULATIONS, action_bins=ACTION_BINS)
        assert settings.seed == 7

        engine_a = MCTSEngine(settings=settings)
        engine_b = MCTSEngine(settings=settings)
        assert engine_a.seed == engine_b.seed == 7
        assert _snapshot(engine_a.search()) == _snapshot(engine_b.search())

    def test_seeded_engine_ignores_global_random_state(self) -> None:
        random.seed(100)
        _, a = _search(2)
        random.seed(200)
        _, b = _search(2)
        assert _snapshot(a) == _snapshot(b)

    def test_unseeded_engine_follows_global_random_state(self) -> None:
        """Backward compatibility: callers that seed ``random`` still get repeatability."""
        random.seed(31)
        _, a = _search(None)
        random.seed(31)
        _, b = _search(None)
        assert _snapshot(a) == _snapshot(b)

    def test_second_search_on_same_engine_continues_stream(self) -> None:
        """A single engine's generator advances; repeatability is per fresh engine."""
        engine = MCTSEngine(settings=_settings(), seed=0)
        first = engine.search()
        second = engine.search()
        _, fresh = _search(0)
        assert _snapshot(first) == _snapshot(fresh)
        assert _snapshot(second) != _snapshot(first)

    def test_last_root_is_none_before_search(self) -> None:
        assert MCTSEngine(settings=_settings(), seed=0).last_root is None


def _node_with_children(visit_counts: list[int]) -> TreeNode:
    """Root over one dimension with one child per entry of ``visit_counts``."""
    root = TreeNode(state=CalibrationState(remaining_dimensions=["metal_ratio"]))
    for bin_index, visits in enumerate(visit_counts):
        child = root.expand(
            CalibrationAction(dimension="metal_ratio", value=bin_index / 10, bin_index=bin_index)
        )
        child.visit_count = visits
    return root


class TestTreeBestChildRng:
    """TreeNode.best_child's stochastic branch honours an injected generator."""

    @staticmethod
    def _draws(root: TreeNode, rng: random.Random | None, temperature: float) -> list[int]:
        picks = []
        for _ in range(25):
            child = root.best_child(temperature=temperature, rng=rng)
            assert child.action is not None
            picks.append(child.action.bin_index)
        return picks

    def test_temperature_sampling_is_deterministic_with_rng(self) -> None:
        root = _node_with_children([1, 5, 20, 3, 8])
        a = self._draws(root, random.Random(3), temperature=1.0)
        b = self._draws(root, random.Random(3), temperature=1.0)
        assert a == b
        assert len(set(a)) > 1, "temperature sampling should not be degenerate"

    def test_all_unvisited_choice_is_deterministic_with_rng(self) -> None:
        root = _node_with_children([0, 0, 0, 0])
        a = self._draws(root, random.Random(9), temperature=1.0)
        b = self._draws(root, random.Random(9), temperature=1.0)
        assert a == b

    def test_greedy_ignores_rng(self) -> None:
        root = _node_with_children([1, 5, 20, 3])
        assert root.best_child(temperature=0.0, rng=random.Random(1)).action.bin_index == 2
        assert root.best_child(temperature=0.0).action.bin_index == 2

    def test_no_rng_falls_back_to_global_random(self) -> None:
        root = _node_with_children([1, 5, 20, 3, 8])
        random.seed(17)
        a = self._draws(root, None, temperature=1.0)
        random.seed(17)
        b = self._draws(root, None, temperature=1.0)
        assert a == b


def _examples(n: int) -> list[TrainingExample]:
    return [
        TrainingExample(state_features=[float(i)], policy_target=[0.5, 0.5], value_target=i / n)
        for i in range(n)
    ]


class TestReplayBufferSeed:
    """ReplayBuffer.sample honours its seed."""

    def test_same_seed_same_sample(self) -> None:
        a = ReplayBuffer(max_size=50, seed=5)
        b = ReplayBuffer(max_size=50, seed=5)
        a.add_batch(_examples(50))
        b.add_batch(_examples(50))
        assert [e.value_target for e in a.sample(10)] == [e.value_target for e in b.sample(10)]

    def test_different_seeds_differ(self) -> None:
        a = ReplayBuffer(max_size=50, seed=5)
        b = ReplayBuffer(max_size=50, seed=6)
        a.add_batch(_examples(50))
        b.add_batch(_examples(50))
        assert [e.value_target for e in a.sample(10)] != [e.value_target for e in b.sample(10)]

    def test_settings_seed_is_used(self) -> None:
        settings = MCTSSettings(seed=8)
        a = ReplayBuffer(max_size=50, settings=settings)
        b = ReplayBuffer(max_size=50, seed=8)
        assert a.seed == 8
        a.add_batch(_examples(50))
        b.add_batch(_examples(50))
        assert [e.value_target for e in a.sample(10)] == [e.value_target for e in b.sample(10)]

    def test_seed_none_still_samples(self) -> None:
        buffer = ReplayBuffer(max_size=20)
        assert buffer.seed is None
        buffer.add_batch(_examples(20))
        assert len(buffer.sample(5)) == 5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainerSeed:
    """MCTSTrainer seeds parameter sampling, replay sampling and torch init."""

    @staticmethod
    def _trainer(seed: int | None):
        from ptpd_calibration.mcts.training import MCTSTrainer

        settings = MCTSSettings(
            num_simulations=50,
            action_bins=ACTION_BINS,
            value_hidden_dims=[16, 8],
            policy_hidden_dims=[16, 8],
            replay_buffer_size=1000,
            training_batch_size=32,
        )
        return MCTSTrainer(settings=settings, seed=seed)

    @staticmethod
    def _weights(trainer) -> list[float]:
        import torch

        return torch.cat([p.detach().flatten() for p in trainer.network.parameters()]).tolist()

    def test_same_seed_same_parameters_and_weights(self) -> None:
        a, b = self._trainer(0), self._trainer(0)
        assert a.seed == b.seed == 0
        assert a.replay_buffer.seed == b.replay_buffer.seed
        assert a.replay_buffer.seed != 0, "buffer stream must be derived, not the raw seed"
        assert a._sample_random_parameters() == b._sample_random_parameters()
        assert self._weights(a) == self._weights(b)

    def test_different_seeds_differ(self) -> None:
        a, b = self._trainer(0), self._trainer(1)
        assert a._sample_random_parameters() != b._sample_random_parameters()
        assert self._weights(a) != self._weights(b)

    def test_seed_none_still_trains(self) -> None:
        trainer = self._trainer(None)
        assert trainer.seed is None
        assert trainer.replay_buffer.seed is None
        params = trainer._sample_random_parameters()
        assert set(params) == set(trainer.settings.decision_order)
