"""
SCI-08 provenance guard tests for the ML database and predictor.

Simulator-generated ``CalibrationRecord``s (``provenance == "simulated"``) must be
excluded from every ``CalibrationDatabase`` query helper and from
``CurvePredictor.train`` unless the caller passes ``include_simulated=True``.
"""

from __future__ import annotations

import logging

import pytest

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
from ptpd_calibration.ml.database import CalibrationDatabase, filter_by_provenance


def _record(
    paper: str = "Arches Platine",
    *,
    simulated: bool = False,
    exposure: float = 180.0,
    metal_ratio: float = 0.5,
    chemistry: ChemistryType = ChemistryType.PLATINUM_PALLADIUM,
    tags: list[str] | None = None,
) -> CalibrationRecord:
    """Build a record with a full 21-step density list."""
    return CalibrationRecord(
        paper_type=paper,
        exposure_time=exposure,
        metal_ratio=metal_ratio,
        chemistry_type=chemistry,
        contrast_agent=ContrastAgent.NONE,
        developer=DeveloperType.POTASSIUM_OXALATE,
        measured_densities=[0.1 + 0.09 * k for k in range(21)],
        provenance="simulated" if simulated else "measured",
        developer_temp_c=25.0 if simulated else None,
        tags=tags or [],
    )


@pytest.fixture
def mixed_db() -> tuple[CalibrationDatabase, list[CalibrationRecord], list[CalibrationRecord]]:
    """Database with 6 measured and 4 simulated records across two papers/chemistries."""
    measured = [
        _record("Arches Platine", exposure=150.0 + 10 * i, metal_ratio=0.2 + 0.1 * i)
        for i in range(3)
    ] + [
        _record(
            "Bergger COT320",
            exposure=200.0 + 10 * i,
            metal_ratio=0.9 + 0.03 * i,
            chemistry=ChemistryType.PLATINUM,
        )
        for i in range(3)
    ]
    simulated = [
        _record("Arches Platine", simulated=True, exposure=160.0, metal_ratio=0.3, tags=["mcts"])
        for _ in range(2)
    ] + [
        _record(
            "Bergger COT320",
            simulated=True,
            exposure=210.0,
            metal_ratio=0.95,
            chemistry=ChemistryType.PLATINUM,
            tags=["mcts"],
        )
        for _ in range(2)
    ]
    db = CalibrationDatabase()
    for record in measured + simulated:
        db.add_record(record)
    return db, measured, simulated


class TestFilterByProvenance:
    """Tests for the shared helper."""

    def test_excludes_simulated_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [_record(), _record(simulated=True), _record()]
        with caplog.at_level(logging.DEBUG, logger="ptpd_calibration.ml.database"):
            kept = filter_by_provenance(records, context="unit")
        assert [r.provenance for r in kept] == ["measured", "measured"]
        assert any("Excluded 1 simulated record(s) from unit" in m for m in caplog.messages)

    def test_include_simulated_keeps_everything(self) -> None:
        records = [_record(), _record(simulated=True)]
        assert filter_by_provenance(records, include_simulated=True) == records

    def test_no_log_when_nothing_excluded(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="ptpd_calibration.ml.database"):
            filter_by_provenance([_record(), _record()])
        assert not any("Excluded" in m for m in caplog.messages)

    def test_accepts_any_iterable(self) -> None:
        records = {_record().id: _record()}
        assert len(filter_by_provenance(records.values())) == 1


class TestCalibrationDatabaseProvenance:
    """Every query helper applies the guard, and every one can opt out."""

    def test_get_all_records(self, mixed_db) -> None:
        db, measured, simulated = mixed_db
        assert len(db) == 10  # raw store still holds everything
        assert {r.id for r in db.get_all_records()} == {r.id for r in measured}
        assert len(db.get_all_records(include_simulated=True)) == 10

    def test_get_records_for_paper(self, mixed_db) -> None:
        db, _, _ = mixed_db
        assert len(db.get_records_for_paper("Arches Platine")) == 3
        assert len(db.get_records_for_paper("Arches Platine", include_simulated=True)) == 5

    def test_get_records_for_chemistry(self, mixed_db) -> None:
        db, _, _ = mixed_db
        assert len(db.get_records_for_chemistry(ChemistryType.PLATINUM.value)) == 3
        assert (
            len(db.get_records_for_chemistry(ChemistryType.PLATINUM.value, include_simulated=True))
            == 5
        )

    def test_query(self, mixed_db) -> None:
        db, _, _ = mixed_db
        assert db.query(tags=["mcts"]) == []
        assert len(db.query(tags=["mcts"], include_simulated=True)) == 4
        assert len(db.query(paper_type="arches platine")) == 3
        assert len(db.query(paper_type="arches platine", include_simulated=True)) == 5

    def test_get_similar_records(self, mixed_db) -> None:
        db, measured, simulated = mixed_db
        reference = measured[0]
        similar_ids = {r.id for r, _ in db.get_similar_records(reference, min_similarity=0.0)}
        assert similar_ids.isdisjoint({r.id for r in simulated})
        similar_ids_all = {
            r.id
            for r, _ in db.get_similar_records(
                reference, min_similarity=0.0, max_records=50, include_simulated=True
            )
        }
        assert {r.id for r in simulated} <= similar_ids_all

    def test_get_record_by_id_is_not_guarded(self, mixed_db) -> None:
        """Direct lookup by id is an explicit request and is never filtered."""
        db, _, simulated = mixed_db
        assert db.get_record(simulated[0].id) is simulated[0]

    def test_save_and_load_preserve_provenance(self, mixed_db, tmp_path) -> None:
        db, _, _ = mixed_db
        path = tmp_path / "db.json"
        db.save(path)
        loaded = CalibrationDatabase.load(path)
        assert len(loaded) == 10
        assert len(loaded.get_all_records()) == 6
        simulated = loaded.get_all_records(include_simulated=True)
        assert sum(r.is_simulated for r in simulated) == 4
        assert all(r.developer_temp_c == 25.0 for r in simulated if r.is_simulated)

    def test_legacy_database_json_loads_as_measured(self, tmp_path) -> None:
        """A database file written before SCI-08 has no provenance key."""
        import json

        legacy_record = _record().model_dump(mode="json")
        legacy_record.pop("provenance")
        legacy_record.pop("developer_temp_c")
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"version": "1.0", "records": [legacy_record]}))

        loaded = CalibrationDatabase.load(path)
        records = loaded.get_all_records()
        assert len(records) == 1
        assert records[0].provenance == "measured"
        assert records[0].developer_temp_c is None


class TestCurvePredictorProvenance:
    """``CurvePredictor.train`` never learns from simulated densities by default."""

    @pytest.fixture
    def predictor_cls(self):
        pytest.importorskip("sklearn")
        from ptpd_calibration.ml.predictor import CurvePredictor

        return CurvePredictor

    def test_train_excludes_simulated_by_default(
        self, predictor_cls, mixed_db, caplog: pytest.LogCaptureFixture
    ) -> None:
        db, measured, _ = mixed_db
        predictor = predictor_cls(model_type="random_forest")
        with caplog.at_level(logging.DEBUG):
            stats = predictor.train(db)
        assert stats["num_samples"] == len(measured)
        assert any("4 simulated excluded" in m for m in caplog.messages)

    def test_train_can_opt_into_simulated(self, predictor_cls, mixed_db) -> None:
        db, _, _ = mixed_db
        predictor = predictor_cls(model_type="random_forest")
        stats = predictor.train(db, include_simulated=True)
        assert stats["num_samples"] == 10

    def test_train_fails_when_only_simulated_records_exist(self, predictor_cls) -> None:
        """Simulated-only databases do not silently satisfy the minimum sample count."""
        db = CalibrationDatabase()
        for _ in range(10):
            db.add_record(_record(simulated=True))
        predictor = predictor_cls(model_type="random_forest")
        with pytest.raises(ValueError, match="Need at least"):
            predictor.train(db)
        assert predictor.train(db, include_simulated=True)["num_samples"] == 10
