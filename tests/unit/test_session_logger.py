from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ptpd_calibration.session.logger import (
    ChemistryUsed,
    PrintRecord,
    PrintResult,
    PrintSession,
    SessionLogger,
)


class TestPrintResult:
    """Tests for PrintResult enum."""

    def test_all_results_exist(self):
        """All result types should exist."""
        assert PrintResult.EXCELLENT.value == "excellent"
        assert PrintResult.GOOD.value == "good"
        assert PrintResult.ACCEPTABLE.value == "acceptable"
        assert PrintResult.POOR.value == "poor"
        assert PrintResult.FAILED.value == "failed"


class TestChemistryUsed:
    """Tests for ChemistryUsed dataclass."""

    def test_default_chemistry(self):
        """Default chemistry should have zero drops."""
        chem = ChemistryUsed()
        assert chem.ferric_oxalate_drops == 0.0
        assert chem.palladium_drops == 0.0
        assert chem.platinum_drops == 0.0
        assert chem.na2_drops == 0.0

    def test_total_drops(self):
        """Total drops should sum all drops."""
        chem = ChemistryUsed(
            ferric_oxalate_drops=10.0,
            ferric_oxalate_contrast_drops=2.0,
            palladium_drops=5.0,
            platinum_drops=3.0,
            na2_drops=1.0,
        )
        assert chem.total_drops == 21.0

    def test_platinum_ratio_all_palladium(self):
        """Platinum ratio should be 0 for all palladium."""
        chem = ChemistryUsed(palladium_drops=10.0, platinum_drops=0.0)
        assert chem.platinum_ratio == 0.0

    def test_platinum_ratio_mixed(self):
        """Platinum ratio should calculate correctly."""
        chem = ChemistryUsed(palladium_drops=8.0, platinum_drops=2.0)
        assert chem.platinum_ratio == pytest.approx(0.2)

    def test_platinum_ratio_no_metal(self):
        """Platinum ratio should be 0 with no metal drops."""
        chem = ChemistryUsed()
        assert chem.platinum_ratio == 0.0

    def test_to_dict(self):
        """Should serialize to dictionary."""
        chem = ChemistryUsed(
            palladium_drops=5.0,
            platinum_drops=5.0,
            developer="Ammonium Citrate",
        )
        d = chem.to_dict()
        assert d["palladium_drops"] == 5.0
        assert d["platinum_drops"] == 5.0
        assert d["platinum_ratio"] == "50%"
        assert d["developer"] == "Ammonium Citrate"

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "ferric_oxalate_drops": 10.0,
            "palladium_drops": 6.0,
            "platinum_drops": 4.0,
            "developer": "Hot Potassium Oxalate",
            "developer_temperature_f": 100.0,
        }
        chem = ChemistryUsed.from_dict(data)
        assert chem.ferric_oxalate_drops == 10.0
        assert chem.palladium_drops == 6.0
        assert chem.developer == "Hot Potassium Oxalate"
        assert chem.developer_temperature_f == 100.0


class TestPrintRecord:
    """Tests for PrintRecord dataclass."""

    def test_default_record(self):
        """Default record should have sensible defaults."""
        record = PrintRecord()
        assert record.id is not None
        assert record.timestamp is not None
        assert record.result == PrintResult.ACCEPTABLE

    def test_custom_record(self):
        """Custom record should store all fields."""
        record = PrintRecord(
            image_name="landscape.tif",
            paper_type="Arches Platine",
            paper_size="8x10",
            exposure_time_minutes=12.5,
            result=PrintResult.EXCELLENT,
            dmax_achieved=1.65,
            notes="Beautiful print",
            tags=["portfolio", "landscape"],
        )
        assert record.image_name == "landscape.tif"
        assert record.paper_type == "Arches Platine"
        assert record.exposure_time_minutes == 12.5
        assert record.result == PrintResult.EXCELLENT
        assert record.dmax_achieved == 1.65
        assert "portfolio" in record.tags

    def test_record_to_dict(self):
        """Should serialize to dictionary."""
        record = PrintRecord(
            image_name="test.tif",
            paper_type="COT 320",
            result=PrintResult.GOOD,
        )
        d = record.to_dict()
        assert d["image_name"] == "test.tif"
        assert d["paper_type"] == "COT 320"
        assert d["result"] == "good"
        assert "id" in d
        assert "timestamp" in d

    def test_record_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "image_name": "portrait.tif",
            "paper_type": "Hahnemuhle Platinum Rag",
            "result": "excellent",
            "dmax_achieved": 1.7,
            "tags": ["portrait", "commission"],
        }
        record = PrintRecord.from_dict(data)
        assert record.image_name == "portrait.tif"
        assert record.paper_type == "Hahnemuhle Platinum Rag"
        assert record.result == PrintResult.EXCELLENT
        assert record.dmax_achieved == 1.7

    def test_record_from_dict_minimal(self):
        """Should handle minimal dictionary."""
        data = {}
        record = PrintRecord.from_dict(data)
        assert record.id is not None
        assert record.result == PrintResult.ACCEPTABLE


class TestPrintSession:
    """Tests for PrintSession dataclass."""

    def test_default_session(self):
        """Default session should be created."""
        session = PrintSession()
        assert session.id is not None
        assert session.started_at is not None
        assert session.ended_at is None
        assert len(session.records) == 0

    def test_add_record(self):
        """Should add records to session."""
        session = PrintSession(name="Test Session")
        record1 = PrintRecord(image_name="img1.tif")
        record2 = PrintRecord(image_name="img2.tif")

        session.add_record(record1)
        session.add_record(record2)

        assert len(session.records) == 2
        assert session.records[0].image_name == "img1.tif"

    def test_end_session(self):
        """Should mark session as ended."""
        session = PrintSession()
        assert session.ended_at is None

        session.end_session()
        assert session.ended_at is not None

    def test_duration_hours(self):
        """Should calculate duration."""
        session = PrintSession()
        session.started_at = datetime.now() - timedelta(hours=2)
        session.end_session()

        assert session.duration_hours is not None
        assert session.duration_hours >= 2.0

    def test_duration_hours_ongoing(self):
        """Should calculate ongoing duration."""
        session = PrintSession()
        session.started_at = datetime.now() - timedelta(hours=1)

        # No end time, should use now
        assert session.duration_hours is not None
        assert session.duration_hours >= 1.0

    def test_success_rate_empty(self):
        """Empty session should have 0% success rate."""
        session = PrintSession()
        assert session.success_rate == 0.0

    def test_success_rate_all_successful(self):
        """All successful prints should be 100%."""
        session = PrintSession()
        session.add_record(PrintRecord(result=PrintResult.EXCELLENT))
        session.add_record(PrintRecord(result=PrintResult.GOOD))
        session.add_record(PrintRecord(result=PrintResult.ACCEPTABLE))

        assert session.success_rate == 100.0

    def test_success_rate_mixed(self):
        """Mixed results should calculate correctly."""
        session = PrintSession()
        session.add_record(PrintRecord(result=PrintResult.EXCELLENT))
        session.add_record(PrintRecord(result=PrintResult.GOOD))
        session.add_record(PrintRecord(result=PrintResult.POOR))
        session.add_record(PrintRecord(result=PrintResult.FAILED))

        # 2 successful (excellent, good) out of 4 = 50%
        assert session.success_rate == 50.0

    def test_get_statistics_empty(self):
        """Empty session should return minimal stats."""
        session = PrintSession()
        stats = session.get_statistics()
        assert stats["total_prints"] == 0

    def test_get_statistics_with_records(self):
        """Should calculate statistics correctly."""
        session = PrintSession()
        session.add_record(PrintRecord(
            paper_type="Arches Platine",
            result=PrintResult.EXCELLENT,
            exposure_time_minutes=10.0,
        ))
        session.add_record(PrintRecord(
            paper_type="COT 320",
            result=PrintResult.GOOD,
            exposure_time_minutes=12.0,
        ))

        stats = session.get_statistics()
        assert stats["total_prints"] == 2
        assert "excellent" in stats["results"]
        assert stats["avg_exposure_minutes"] == 11.0
        assert "Arches Platine" in stats["papers_used"]

    def test_session_to_dict(self):
        """Should serialize session."""
        session = PrintSession(name="Test")
        session.add_record(PrintRecord(image_name="test.tif"))

        d = session.to_dict()
        assert d["name"] == "Test"
        assert len(d["records"]) == 1
        assert "statistics" in d

    def test_session_from_dict(self):
        """Should deserialize session."""
        record_id = str(uuid4())
        data = {
            "id": str(uuid4()),
            "name": "Loaded Session",
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "records": [
                {
                    "id": record_id,
                    "timestamp": datetime.now().isoformat(),
                    "image_name": "loaded.tif",
                    "paper_type": "",
                    "result": "good",
                }
            ],
            "notes": "Test notes",
        }
        session = PrintSession.from_dict(data)
        assert session.name == "Loaded Session"
        assert len(session.records) == 1
        assert session.records[0].image_name == "loaded.tif"


class TestSessionLogger:
    """Tests for SessionLogger class."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create session logger with temp storage."""
        return SessionLogger(storage_dir=tmp_path)

    def test_logger_creates_storage_dir(self, tmp_path):
        """Logger should create storage directory."""
        storage_dir = tmp_path / "new_sessions"
        _ = SessionLogger(storage_dir=storage_dir)
        assert storage_dir.exists()

    def test_start_session(self, logger):
        """Should start new session."""
        session = logger.start_session("Test Session")
        assert session.name == "Test Session"
        assert logger.get_current_session() == session

    def test_start_session_auto_name(self, logger):
        """Should auto-generate session name."""
        session = logger.start_session()
        assert "Session" in session.name
        assert logger.get_current_session() is not None

    def test_get_current_session_none(self, logger):
        """Should return None when no session active."""
        assert logger.get_current_session() is None

    def test_log_print_creates_session(self, logger):
        """Should create session if none active."""
        record = PrintRecord(image_name="auto_session.tif")
        logger.log_print(record)

        session = logger.get_current_session()
        assert session is not None
        assert len(session.records) == 1

    def test_log_print_to_existing_session(self, logger):
        """Should add record to existing session."""
        logger.start_session("Existing")
        record1 = PrintRecord(image_name="first.tif")
        record2 = PrintRecord(image_name="second.tif")

        logger.log_print(record1)
        logger.log_print(record2)

        session = logger.get_current_session()
        assert len(session.records) == 2

    def test_end_session(self, logger):
        """Should end and save session."""
        logger.start_session("End Test")
        logger.log_print(PrintRecord(image_name="test.tif"))

        ended = logger.end_session()
        assert ended is not None
        assert ended.ended_at is not None
        assert logger.get_current_session() is None

    def test_end_session_no_active(self, logger):
        """Should return None if no active session."""
        result = logger.end_session()
        assert result is None

    def test_save_and_load_session(self, logger, tmp_path):
        """Should save and load session."""
        session = PrintSession(name="Save Load Test")
        session.add_record(PrintRecord(
            image_name="save_load.tif",
            paper_type="Arches Platine",
            result=PrintResult.GOOD,
        ))

        filepath = logger.save_session(session)
        assert filepath.exists()

        loaded = logger.load_session(filepath)
        assert loaded.name == "Save Load Test"
        assert len(loaded.records) == 1
        assert loaded.records[0].image_name == "save_load.tif"

    def test_list_sessions(self, logger):
        """Should list saved sessions."""
        # Create and save multiple sessions
        for i in range(3):
            session = PrintSession(name=f"List Test {i}")
            session.add_record(PrintRecord(image_name=f"img_{i}.tif"))
            logger.save_session(session)

        sessions = logger.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_limit(self, logger):
        """Should respect limit parameter."""
        for i in range(5):
            session = PrintSession(name=f"Limit Test {i}")
            logger.save_session(session)

        sessions = logger.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_search_records_by_paper(self, logger):
        """Should search records by paper type."""
        session = PrintSession(name="Search Test")
        session.add_record(PrintRecord(paper_type="Arches Platine"))
        session.add_record(PrintRecord(paper_type="COT 320"))
        session.add_record(PrintRecord(paper_type="Arches Platine"))
        logger.save_session(session)

        results = logger.search_records(paper_type="Arches Platine")
        assert len(results) == 2

    def test_search_records_by_result(self, logger):
        """Should search records by result."""
        session = PrintSession(name="Result Search")
        session.add_record(PrintRecord(result=PrintResult.EXCELLENT))
        session.add_record(PrintRecord(result=PrintResult.FAILED))
        session.add_record(PrintRecord(result=PrintResult.EXCELLENT))
        logger.save_session(session)

        results = logger.search_records(result=PrintResult.EXCELLENT)
        assert len(results) == 2

    def test_search_records_by_tags(self, logger):
        """Should search records by tags."""
        session = PrintSession(name="Tag Search")
        session.add_record(PrintRecord(tags=["landscape", "portfolio"]))
        session.add_record(PrintRecord(tags=["portrait"]))
        session.add_record(PrintRecord(tags=["landscape"]))
        logger.save_session(session)

        results = logger.search_records(tags=["landscape"])
        assert len(results) == 2

    def test_search_records_limit(self, logger):
        """Should respect search limit."""
        session = PrintSession(name="Search Limit")
        for _i in range(10):
            session.add_record(PrintRecord(paper_type="Test Paper"))
        logger.save_session(session)

        results = logger.search_records(paper_type="Test Paper", limit=3)
        assert len(results) == 3

    def test_get_paper_statistics(self, logger):
        """Should calculate paper statistics."""
        session = PrintSession(name="Stats Test")
        session.add_record(PrintRecord(
            paper_type="Arches Platine",
            result=PrintResult.EXCELLENT,
            exposure_time_minutes=10.0,
        ))
        session.add_record(PrintRecord(
            paper_type="Arches Platine",
            result=PrintResult.GOOD,
            exposure_time_minutes=12.0,
        ))
        session.add_record(PrintRecord(
            paper_type="COT 320",
            result=PrintResult.FAILED,
            exposure_time_minutes=15.0,
        ))
        logger.save_session(session)

        stats = logger.get_paper_statistics()

        assert "Arches Platine" in stats
        assert stats["Arches Platine"]["total_prints"] == 2
        assert stats["Arches Platine"]["excellent"] == 1
        assert stats["Arches Platine"]["good"] == 1
        assert stats["Arches Platine"]["avg_exposure"] == 11.0

        assert "COT 320" in stats
        assert stats["COT 320"]["failed"] == 1

    def test_get_paper_statistics_empty_paper(self, logger):
        """Should skip records with empty paper type."""
        session = PrintSession(name="Empty Paper")
        session.add_record(PrintRecord(paper_type=""))  # Empty
        session.add_record(PrintRecord(paper_type="Valid Paper"))
        logger.save_session(session)

        stats = logger.get_paper_statistics()
        assert "" not in stats
        assert "Valid Paper" in stats
