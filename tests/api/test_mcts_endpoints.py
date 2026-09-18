"""
SCI-06 endpoint tests: ``/api/mcts/search`` runs the real engine and
``/api/mcts/feedback`` persists measured records.

Before this change the search endpoint ran one heuristic simulation and
reported ``num_simulations=800``, and the target curve was silently dropped.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ptpd_calibration.api.mcts_router import (  # noqa: E402
    SEARCH_BACKEND_ENGINE,
    SEARCH_BACKEND_HEURISTIC,
    create_mcts_router,
)
from ptpd_calibration.mcts.agents import CalibrationCoordinatorSubagent  # noqa: E402
from ptpd_calibration.mcts.config import MCTSSettings  # noqa: E402
from ptpd_calibration.mcts.engine import MCTSEngine  # noqa: E402
from ptpd_calibration.mcts.feedback import (  # noqa: E402
    FEEDBACK_FILENAME,
    FeedbackRecord,
    FeedbackStore,
    resolve_feedback_path,
)

pytestmark = pytest.mark.api

CURVE_POINTS = 21  # ExtendedProcessSimulator default num_steps
LEGACY_RESPONSE_FIELDS = {
    "search_id",
    "best_parameters",
    "predicted_curve",
    "quality_score",
    "alternatives",
    "search_time_seconds",
    "num_simulations",
}
DEFAULT_PARAMS = {
    "metal_ratio": 0.5,
    "coating_weight": 1.5,
    "ferric_oxalate_pct": 20.0,
    "exposure_time": 180.0,
    "developer_temp": 25.0,
    "humidity": 50.0,
}


def linear_target() -> list[float]:
    return [0.1 + 1.8 * i / (CURVE_POINTS - 1) for i in range(CURVE_POINTS)]


def flat_target() -> list[float]:
    return [2.0] * CURVE_POINTS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def feedback_store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.jsonl")


@pytest.fixture
def mcts_client(feedback_store: FeedbackStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_mcts_router(feedback_store=feedback_store))
    return TestClient(app)


# =============================================================================
# /api/mcts/search runs MCTSEngine
# =============================================================================


class TestSearchRunsEngine:
    def test_response_is_backwards_compatible_and_truthful(self, mcts_client: TestClient) -> None:
        response = mcts_client.post("/api/mcts/search", json={"num_simulations": 60})
        assert response.status_code == 200
        data = response.json()

        assert data.keys() >= LEGACY_RESPONSE_FIELDS
        assert data["engine_used"] is True
        assert data["search_backend"] == SEARCH_BACKEND_ENGINE
        assert data["num_simulations"] == 60
        assert data["target_curve_used"] is False
        assert len(data["predicted_curve"]) == CURVE_POINTS
        assert 0.0 <= data["quality_score"] <= 1.0
        assert set(data["best_parameters"]) == set(DEFAULT_PARAMS)
        assert isinstance(data["alternatives"], list)
        assert all(isinstance(alt, dict) for alt in data["alternatives"])
        assert data["search_time_seconds"] >= 0.0

    def test_num_simulations_equals_evaluations_performed(self, mcts_client: TestClient) -> None:
        calls: list[object] = []
        original = MCTSEngine._evaluate

        def counting(self: MCTSEngine, node, target_curve):  # type: ignore[no-untyped-def]
            calls.append(target_curve)
            return original(self, node, target_curve)

        with patch.object(MCTSEngine, "_evaluate", counting):
            response = mcts_client.post("/api/mcts/search", json={"num_simulations": 75})

        assert response.status_code == 200
        assert response.json()["num_simulations"] == 75
        assert len(calls) == 75

    def test_target_curve_reaches_engine_and_scorer(self, mcts_client: TestClient) -> None:
        target = linear_target()
        with patch.object(
            MCTSEngine, "search", autospec=True, side_effect=MCTSEngine.search
        ) as spy:
            response = mcts_client.post(
                "/api/mcts/search", json={"num_simulations": 50, "target_curve": target}
            )

        assert response.status_code == 200
        spy.assert_called_once()
        assert spy.call_args.kwargs["target_curve"] == target
        assert response.json()["target_curve_used"] is True

    def test_different_target_curves_give_different_quality(self, mcts_client: TestClient) -> None:
        scores: list[float] = []
        for target in (linear_target(), flat_target()):
            random.seed(1234)  # same search randomness; only the target differs
            response = mcts_client.post(
                "/api/mcts/search",
                json={"num_simulations": 50, "target_curve": target},
            )
            assert response.status_code == 200
            scores.append(response.json()["quality_score"])
        assert scores[0] != scores[1]

    def test_mismatched_target_length_is_reported_not_silently_used(
        self, mcts_client: TestClient
    ) -> None:
        response = mcts_client.post(
            "/api/mcts/search",
            json={"num_simulations": 50, "target_curve": [0.1, 0.5, 1.0, 1.5, 2.0]},
        )
        assert response.status_code == 200
        assert response.json()["target_curve_used"] is False

    def test_fixed_parameters_are_honoured_by_engine(self, mcts_client: TestClient) -> None:
        response = mcts_client.post(
            "/api/mcts/search",
            json={"num_simulations": 50, "fixed_parameters": {"metal_ratio": 0.5}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["engine_used"] is True
        assert data["best_parameters"]["metal_ratio"] == 0.5
        assert all(alt["metal_ratio"] == 0.5 for alt in data["alternatives"])

    def test_aesthetics_pin_coordinator_chemistry(self, mcts_client: TestClient) -> None:
        # warmth=1.0 -> pure palladium (metal_ratio at range minimum);
        # contrast/tonal_range default to 0.5 -> FO% at centre, coating mid-range.
        response = mcts_client.post(
            "/api/mcts/search",
            json={"num_simulations": 50, "target_aesthetics": {"warmth": 1.0}},
        )
        assert response.status_code == 200
        best = response.json()["best_parameters"]
        assert best["metal_ratio"] == pytest.approx(0.0)
        assert best["ferric_oxalate_pct"] == pytest.approx(20.0)
        assert best["coating_weight"] == pytest.approx(1.75)

    def test_user_fixed_parameters_override_aesthetics(self, mcts_client: TestClient) -> None:
        response = mcts_client.post(
            "/api/mcts/search",
            json={
                "num_simulations": 50,
                "target_aesthetics": {"warmth": 1.0},
                "fixed_parameters": {"metal_ratio": 0.9},
            },
        )
        assert response.json()["best_parameters"]["metal_ratio"] == pytest.approx(0.9)

    def test_num_simulations_clamped_to_settings_cap(
        self, mcts_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PTPD_MCTS_MAX_SIMULATIONS_PER_REQUEST", "50")
        response = mcts_client.post("/api/mcts/search", json={"num_simulations": 100})
        assert response.status_code == 200
        assert response.json()["num_simulations"] == 50

    def test_search_runs_off_the_event_loop(self, mcts_client: TestClient) -> None:
        with patch(
            "ptpd_calibration.api.mcts_router.run_in_threadpool",
            autospec=True,
            side_effect=_run_inline,
        ) as run_in_threadpool:
            response = mcts_client.post("/api/mcts/search", json={"num_simulations": 50})
        assert response.status_code == 200
        assert response.json()["engine_used"] is True
        run_in_threadpool.assert_called_once()
        assert run_in_threadpool.call_args.args[0].__name__ == "search"
        assert run_in_threadpool.call_args.kwargs["target_curve"] is None


async def _run_inline(func, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Stand-in for run_in_threadpool that calls the function on the loop thread."""
    return func(*args, **kwargs)


# =============================================================================
# /api/mcts/search falls back truthfully
# =============================================================================


class TestSearchFallback:
    def test_engine_failure_falls_back_to_coordinator(self, mcts_client: TestClient) -> None:
        with patch.object(MCTSEngine, "search", side_effect=RuntimeError("engine exploded")):
            response = mcts_client.post(
                "/api/mcts/search",
                json={"num_simulations": 100, "target_curve": linear_target()},
            )
        assert response.status_code == 200
        data = response.json()
        assert data.keys() >= LEGACY_RESPONSE_FIELDS
        assert data["engine_used"] is False
        assert data["search_backend"] == SEARCH_BACKEND_HEURISTIC
        assert data["num_simulations"] == 1
        assert data["target_curve_used"] is True
        assert data["alternatives"] == [data["best_parameters"]]

    def test_fallback_scores_against_target_curve(self, mcts_client: TestClient) -> None:
        with patch.object(MCTSEngine, "search", side_effect=RuntimeError("engine exploded")):
            linear = mcts_client.post(
                "/api/mcts/search", json={"target_curve": linear_target()}
            ).json()
            flat = mcts_client.post("/api/mcts/search", json={"target_curve": flat_target()}).json()
            none = mcts_client.post("/api/mcts/search", json={}).json()
        # coordinator path is deterministic, so only the target curve can move the score
        assert linear["quality_score"] != flat["quality_score"]
        assert none["target_curve_used"] is False
        assert linear["best_parameters"] == flat["best_parameters"]

    def test_engine_construction_failure_also_falls_back(self, mcts_client: TestClient) -> None:
        with patch(
            "ptpd_calibration.mcts.engine.MCTSEngine.__init__",
            side_effect=ValueError("bad settings"),
        ):
            response = mcts_client.post("/api/mcts/search", json={"num_simulations": 50})
        assert response.status_code == 200
        assert response.json()["engine_used"] is False


# =============================================================================
# Coordinator consumes target_curve (agents.py)
# =============================================================================


class TestCoordinatorTargetCurve:
    async def test_coordinate_search_scores_against_target_curve(self) -> None:
        coordinator = CalibrationCoordinatorSubagent()
        linear = await coordinator._coordinate_search({"target_curve": linear_target()})
        flat = await coordinator._coordinate_search({"target_curve": flat_target()})
        none = await coordinator._coordinate_search({})

        assert linear["evaluation"]["target_curve_used"] is True
        assert flat["evaluation"]["target_curve_used"] is True
        assert none["evaluation"]["target_curve_used"] is False
        assert linear["evaluation"]["quality_score"] != flat["evaluation"]["quality_score"]
        assert linear["full_parameters"] == flat["full_parameters"]

    def test_evaluate_parameters_reports_length_mismatch(self) -> None:
        coordinator = CalibrationCoordinatorSubagent()
        baseline = coordinator._evaluate_parameters(DEFAULT_PARAMS)
        mismatched = coordinator._evaluate_parameters(DEFAULT_PARAMS, target_curve=[1.0, 2.0])
        assert baseline["target_curve_used"] is False
        assert mismatched["target_curve_used"] is False
        assert mismatched["quality_score"] == baseline["quality_score"]

    async def test_run_evaluate_parameters_task_passes_target(self) -> None:
        coordinator = CalibrationCoordinatorSubagent()
        result = await coordinator.run(
            "evaluate_parameters",
            context={"parameters": DEFAULT_PARAMS, "target_curve": flat_target()},
        )
        assert result.success is True
        assert result.result["target_curve_used"] is True


# =============================================================================
# /api/mcts/feedback persists measured records
# =============================================================================


class TestFeedbackPersistence:
    def test_feedback_is_persisted_and_readable(
        self, mcts_client: TestClient, feedback_store: FeedbackStore
    ) -> None:
        payload = {
            "parameters": {"metal_ratio": 0.5, "exposure_time": 180.0},
            "measured_curve": [0.1, 0.5, 1.0, 1.5, 2.0],
            "quality_rating": 0.8,
            "notes": "first print on Platine",
        }
        response = mcts_client.post("/api/mcts/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recorded" in data["message"].lower()
        assert data["provenance"] == "measured"
        assert isinstance(data["record_id"], str) and len(data["record_id"]) == 32
        assert "timestamp" in data

        records = feedback_store.list()
        assert len(records) == 1
        record = records[0]
        assert record.id == data["record_id"]
        assert record.parameters == payload["parameters"]
        assert record.measured_curve == payload["measured_curve"]
        assert record.quality_rating == 0.8
        assert record.notes == payload["notes"]
        assert record.provenance == "measured"
        assert feedback_store.path.exists()

    def test_records_append_in_order_and_limit(
        self, mcts_client: TestClient, feedback_store: FeedbackStore
    ) -> None:
        for rating in (0.1, 0.2, 0.3):
            payload = {
                "parameters": {"metal_ratio": 0.5},
                "measured_curve": [0.1, 1.0],
                "quality_rating": rating,
            }
            assert mcts_client.post("/api/mcts/feedback", json=payload).status_code == 200

        assert [r.quality_rating for r in feedback_store.list()] == [0.1, 0.2, 0.3]
        assert [r.quality_rating for r in feedback_store.list(limit=2)] == [0.2, 0.3]
        assert feedback_store.list(limit=0) == []
        assert feedback_store.count() == 3
        assert len({r.id for r in feedback_store.list()}) == 3

    @pytest.mark.parametrize(
        "payload",
        [
            {"parameters": {}, "quality_rating": 0.5},  # missing curve
            {"parameters": {}, "measured_curve": [], "quality_rating": 0.5},  # empty curve
            {"parameters": {}, "measured_curve": [1.0], "quality_rating": 1.5},  # rating > 1
            {"parameters": {}, "measured_curve": [1.0], "quality_rating": -0.1},  # rating < 0
            {"parameters": {}, "measured_curve": ["a"], "quality_rating": 0.5},  # non-numeric
            {"measured_curve": [1.0], "quality_rating": 0.5},  # missing parameters
            {"parameters": {"x": "y"}, "measured_curve": [1.0], "quality_rating": 0.5},
        ],
    )
    def test_malformed_feedback_is_422_and_not_stored(
        self, mcts_client: TestClient, feedback_store: FeedbackStore, payload: dict
    ) -> None:
        assert mcts_client.post("/api/mcts/feedback", json=payload).status_code == 422
        assert feedback_store.list() == []

    def test_store_skips_malformed_lines(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "feedback.jsonl"
        good = FeedbackRecord(parameters={"a": 1.0}, measured_curve=[0.1], quality_rating=0.5)
        path.write_text(
            good.model_dump_json()
            + "\nnot json at all\n"
            + '{"parameters": {}, "measured_curve": [0.1], "quality_rating": 5}\n\n'
        )
        with caplog.at_level("WARNING"):
            records = FeedbackStore(path).list()
        assert [r.id for r in records] == [good.id]
        assert "malformed feedback line" in caplog.text

    def test_list_on_missing_file_is_empty(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "nope.jsonl")
        assert store.list() == []
        assert store.count() == 0

    def test_resolve_path_defaults_under_checkpoint_dir(self) -> None:
        settings = MCTSSettings(checkpoint_dir="some/ckpt")
        assert resolve_feedback_path(settings) == Path("some/ckpt") / FEEDBACK_FILENAME

    def test_resolve_path_honours_explicit_setting(self, tmp_path: Path) -> None:
        settings = MCTSSettings(feedback_path=str(tmp_path / "custom.jsonl"))
        assert resolve_feedback_path(settings) == tmp_path / "custom.jsonl"

    def test_default_router_store_uses_env_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "env" / "feedback.jsonl"
        monkeypatch.setenv("PTPD_MCTS_FEEDBACK_PATH", str(target))
        app = FastAPI()
        app.include_router(create_mcts_router())
        client = TestClient(app)

        payload = {
            "parameters": {"metal_ratio": 0.4},
            "measured_curve": [0.2],
            "quality_rating": 0.9,
        }
        response = client.post("/api/mcts/feedback", json=payload)
        assert response.status_code == 200
        records = FeedbackStore(target).list()
        assert len(records) == 1
        assert records[0].id == response.json()["record_id"]
