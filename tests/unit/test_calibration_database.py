"""
Tests for calibration database module.

Tests calibration session and record storage, retrieval, and statistics.
"""

import pytest
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from ptpd_calibration.calibration.database import (
    CalibrationDatabase,
    CalibrationRecord,
    CalibrationSessionRecord,
)


# =============================================================================
# CalibrationRecord Tests
# =============================================================================


class TestCalibrationRecord:
    """Tests for CalibrationRecord Pydantic model."""

    def test_default_record(self):
        """Should create record with defaults."""
        record = CalibrationRecord(
            session_id=uuid4(),
            paper_type="Arches Platine",
            exposure_time="3:00"
        )
        assert record.id is not None
        assert record.iteration_number == 1
        assert record.paper_type == "Arches Platine"
        assert record.highlight_adj == 0.0
        assert record.midtone_adj == 0.0
        assert record.shadow_adj == 0.0

    def test_full_record(self):
        """Should create record with all fields."""
        session_id = uuid4()
        record = CalibrationRecord(
            session_id=session_id,
            iteration_number=2,
            paper_type="Arches Platine",
            chemistry_ratio="6Pd:2Pt",
            fo_ratio="7:1",
            exposure_time="3:15",
            uv_source="UV LED",
            base_curve_name="Base Curve v1",
            curve_version="v2",
            highlight_density=0.12,
            midtone_density=0.65,
            shadow_density=1.55,
            tonal_range=1.43,
            midtone_separation=0.18,
            highlight_adj=0.02,
            midtone_adj=0.08,
            shadow_adj=0.05,
            recommended_highlight_adj=0.01,
            recommended_midtone_adj=0.03,
            recommended_shadow_adj=0.02,
            notes="Test iteration",
        )

        assert record.session_id == session_id
        assert record.iteration_number == 2
        assert record.midtone_density == 0.65
        assert record.midtone_adj == 0.08

    def test_record_to_dict(self):
        """Should serialize to dictionary."""
        record = CalibrationRecord(
            session_id=uuid4(),
            paper_type="Test Paper",
            exposure_time="3:00",
            midtone_adj=0.08
        )
        d = record.to_dict()

        assert d['paper_type'] == "Test Paper"
        assert d['exposure_time'] == "3:00"
        assert d['midtone_adj'] == 0.08
        assert 'id' in d
        assert 'session_id' in d

    def test_record_uuid_validation(self):
        """Should accept string UUIDs and convert to UUID objects."""
        session_id = str(uuid4())
        record = CalibrationRecord(
            session_id=session_id,
            paper_type="Test Paper",
            exposure_time="3:00"
        )
        assert isinstance(record.session_id, UUID)

    def test_record_adjustment_bounds(self):
        """Adjustments should be within valid bounds."""
        # Valid adjustments
        record = CalibrationRecord(
            session_id=uuid4(),
            paper_type="Test Paper",
            exposure_time="3:00",
            highlight_adj=0.15,
            midtone_adj=-0.10,
            shadow_adj=0.20
        )
        assert record.highlight_adj == 0.15
        assert record.midtone_adj == -0.10

        # Out of bounds should fail
        with pytest.raises(Exception):
            CalibrationRecord(
                session_id=uuid4(),
                paper_type="Test Paper",
                exposure_time="3:00",
                midtone_adj=0.50  # Too high
            )


# =============================================================================
# CalibrationSessionRecord Tests
# =============================================================================


class TestCalibrationSessionRecord:
    """Tests for CalibrationSessionRecord Pydantic model."""

    def test_default_session(self):
        """Should create session with defaults."""
        session = CalibrationSessionRecord(
            paper_type="Arches Platine"
        )
        assert session.id is not None
        assert session.paper_type == "Arches Platine"
        assert session.is_complete is False
        assert session.iteration_count == 0
        assert session.total_highlight_adj == 0.0
        assert session.total_midtone_adj == 0.0
        assert session.total_shadow_adj == 0.0

    def test_full_session(self):
        """Should create session with all fields."""
        session = CalibrationSessionRecord(
            name="Arches Platine Calibration",
            paper_type="Arches Platine",
            chemistry="6Pd:2Pt",
            target_curve_name="Arches_Platine_v1",
            is_complete=True,
            iteration_count=3,
            total_highlight_adj=0.04,
            total_midtone_adj=0.13,
            total_shadow_adj=0.08,
            final_dmax=1.58,
            final_dmin=0.08,
            notes="Successful calibration",
            tags=["arches", "complete"],
        )

        assert session.name == "Arches Platine Calibration"
        assert session.is_complete is True
        assert session.iteration_count == 3
        assert session.total_midtone_adj == 0.13

    def test_session_to_dict(self):
        """Should serialize to dictionary."""
        session = CalibrationSessionRecord(
            paper_type="Test Paper",
            chemistry="50/50 Pt/Pd"
        )
        d = session.to_dict()

        assert d['paper_type'] == "Test Paper"
        assert d['chemistry'] == "50/50 Pt/Pd"
        assert 'id' in d
        assert 'created_at' in d
        assert 'updated_at' in d


# =============================================================================
# CalibrationDatabase Tests
# =============================================================================


class TestCalibrationDatabase:
    """Tests for CalibrationDatabase class."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_calibration.db"
        return CalibrationDatabase(db_path)

    @pytest.fixture
    def populated_db(self, db):
        """Create database with sample data."""
        # Create sessions
        session1 = db.create_session(
            paper_type="Arches Platine",
            chemistry="6Pd:2Pt",
            name="Arches Calibration"
        )
        session2 = db.create_session(
            paper_type="Bergger COT320",
            chemistry="Pure Pd"
        )

        # Add records to first session
        for i in range(3):
            record = CalibrationRecord(
                session_id=session1.id,
                iteration_number=i + 1,
                paper_type="Arches Platine",
                exposure_time=f"3:{i*5:02d}",
                curve_version=f"v{i+1}",
                highlight_adj=0.02,
                midtone_adj=0.05,
                shadow_adj=0.03
            )
            db.add_record(record)

        return db

    # Session Tests

    def test_create_session(self, db):
        """Should create a new session."""
        session = db.create_session(
            paper_type="Arches Platine",
            chemistry="6Pd:2Pt",
            name="Test Session"
        )

        assert isinstance(session, CalibrationSessionRecord)
        assert session.paper_type == "Arches Platine"
        assert session.chemistry == "6Pd:2Pt"
        assert session.name == "Test Session"

    def test_get_session(self, db):
        """Should retrieve session by ID."""
        created = db.create_session(
            paper_type="Arches Platine"
        )

        retrieved = db.get_session(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.paper_type == "Arches Platine"

    def test_get_session_not_found(self, db):
        """Should return None for non-existent session."""
        result = db.get_session(uuid4())
        assert result is None

    def test_update_session(self, db):
        """Should update session fields."""
        session = db.create_session(paper_type="Test Paper")

        result = db.update_session(session.id, {
            'is_complete': True,
            'notes': 'Updated notes'
        })

        assert result is True

        updated = db.get_session(session.id)
        assert updated.is_complete is True
        assert updated.notes == 'Updated notes'

    def test_update_session_not_found(self, db):
        """Should return False for non-existent session."""
        result = db.update_session(uuid4(), {'notes': 'test'})
        assert result is False

    def test_list_sessions(self, populated_db):
        """Should list all sessions."""
        sessions = populated_db.list_sessions()

        assert len(sessions) >= 2
        assert any(s.paper_type == "Arches Platine" for s in sessions)
        assert any(s.paper_type == "Bergger COT320" for s in sessions)

    def test_list_sessions_by_paper(self, populated_db):
        """Should filter sessions by paper type."""
        sessions = populated_db.list_sessions(paper_type="Arches Platine")

        assert len(sessions) == 1
        assert sessions[0].paper_type == "Arches Platine"

    def test_list_sessions_by_completion(self, populated_db):
        """Should filter sessions by completion status."""
        # Mark one as complete
        sessions = populated_db.list_sessions()
        populated_db.update_session(sessions[0].id, {'is_complete': True})

        complete = populated_db.list_sessions(is_complete=True)
        incomplete = populated_db.list_sessions(is_complete=False)

        assert len(complete) == 1
        assert len(incomplete) >= 1

    def test_delete_session(self, populated_db):
        """Should delete session and its records."""
        sessions = populated_db.list_sessions()
        session_to_delete = sessions[0]

        result = populated_db.delete_session(session_to_delete.id)

        assert result is True
        assert populated_db.get_session(session_to_delete.id) is None

        # Records should also be deleted
        records = populated_db.get_session_records(session_to_delete.id)
        assert len(records) == 0

    def test_delete_session_not_found(self, db):
        """Should return False for non-existent session."""
        result = db.delete_session(uuid4())
        assert result is False

    # Record Tests

    def test_add_record(self, db):
        """Should add record and update session."""
        session = db.create_session(paper_type="Test Paper")

        record = CalibrationRecord(
            session_id=session.id,
            paper_type="Test Paper",
            exposure_time="3:00",
            highlight_adj=0.02,
            midtone_adj=0.08,
            shadow_adj=0.05
        )
        record_id = db.add_record(record)

        assert isinstance(record_id, UUID)

        # Session should be updated
        updated_session = db.get_session(session.id)
        assert updated_session.iteration_count == 1
        assert updated_session.total_midtone_adj == 0.08

    def test_get_record(self, db):
        """Should retrieve record by ID."""
        session = db.create_session(paper_type="Test Paper")
        record = CalibrationRecord(
            session_id=session.id,
            paper_type="Test Paper",
            exposure_time="3:00"
        )
        record_id = db.add_record(record)

        retrieved = db.get_record(record_id)

        assert retrieved is not None
        assert retrieved.id == record_id
        assert retrieved.paper_type == "Test Paper"

    def test_get_record_not_found(self, db):
        """Should return None for non-existent record."""
        result = db.get_record(uuid4())
        assert result is None

    def test_get_session_records(self, populated_db):
        """Should get all records for a session."""
        sessions = populated_db.list_sessions(paper_type="Arches Platine")
        session = sessions[0]

        records = populated_db.get_session_records(session.id)

        assert len(records) == 3
        assert all(r.paper_type == "Arches Platine" for r in records)
        # Should be ordered by iteration number
        assert records[0].iteration_number == 1
        assert records[1].iteration_number == 2
        assert records[2].iteration_number == 3

    def test_get_records_by_paper(self, populated_db):
        """Should get records filtered by paper type."""
        records = populated_db.get_records_by_paper("Arches Platine")

        assert len(records) >= 3
        assert all(r.paper_type == "Arches Platine" for r in records)

    def test_cumulative_adjustments(self, db):
        """Should accumulate adjustments correctly."""
        session = db.create_session(paper_type="Test Paper")

        for i in range(3):
            record = CalibrationRecord(
                session_id=session.id,
                iteration_number=i + 1,
                paper_type="Test Paper",
                exposure_time="3:00",
                highlight_adj=0.02,
                midtone_adj=0.05,
                shadow_adj=0.03
            )
            db.add_record(record)

        updated = db.get_session(session.id)
        assert updated.iteration_count == 3
        assert updated.total_highlight_adj == pytest.approx(0.06)
        assert updated.total_midtone_adj == pytest.approx(0.15)
        assert updated.total_shadow_adj == pytest.approx(0.09)

    # Statistics Tests

    def test_get_statistics(self, populated_db):
        """Should return database statistics."""
        stats = populated_db.get_statistics()

        assert 'total_sessions' in stats
        assert 'total_records' in stats
        assert 'complete_sessions' in stats
        assert 'incomplete_sessions' in stats
        assert 'papers' in stats

        assert stats['total_sessions'] >= 2
        assert stats['total_records'] >= 3

    def test_get_statistics_empty_db(self, db):
        """Should handle empty database."""
        stats = db.get_statistics()

        assert stats['total_sessions'] == 0
        assert stats['total_records'] == 0

    # Context Manager Tests

    def test_context_manager(self, tmp_path):
        """Should work as context manager."""
        db_path = tmp_path / "context_test.db"

        with CalibrationDatabase(db_path) as db:
            session = db.create_session(paper_type="Test Paper")
            assert session is not None

        # Should be able to reopen
        with CalibrationDatabase(db_path) as db:
            sessions = db.list_sessions()
            assert len(sessions) == 1

    def test_close(self, db):
        """Should close database connection."""
        db.create_session(paper_type="Test Paper")
        db.close()

        # Connection should be closed
        # Further operations might fail or need reconnection


# =============================================================================
# Database Persistence Tests
# =============================================================================


class TestDatabasePersistence:
    """Tests for database persistence across instances."""

    def test_data_persists(self, tmp_path):
        """Data should persist across database instances."""
        db_path = tmp_path / "persist_test.db"

        # Create and populate first instance
        db1 = CalibrationDatabase(db_path)
        session = db1.create_session(
            paper_type="Arches Platine",
            chemistry="Test"
        )
        session_id = session.id

        record = CalibrationRecord(
            session_id=session_id,
            paper_type="Arches Platine",
            exposure_time="3:00",
            midtone_adj=0.08
        )
        db1.add_record(record)
        db1.close()

        # Reopen with new instance
        db2 = CalibrationDatabase(db_path)
        retrieved_session = db2.get_session(session_id)

        assert retrieved_session is not None
        assert retrieved_session.paper_type == "Arches Platine"
        assert retrieved_session.iteration_count == 1
        assert retrieved_session.total_midtone_adj == 0.08

        records = db2.get_session_records(session_id)
        assert len(records) == 1
        db2.close()

    def test_default_path(self, monkeypatch, tmp_path):
        """Should use default path when none specified."""
        # Mock home directory
        monkeypatch.setenv("HOME", str(tmp_path))

        db = CalibrationDatabase()
        expected_path = tmp_path / ".ptpd" / "calibration.db"

        # Should create directory if needed
        assert db.db_path == expected_path
        db.close()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def db(self, tmp_path):
        db_path = tmp_path / "edge_test.db"
        return CalibrationDatabase(db_path)

    def test_empty_paper_type(self, db):
        """Should handle empty paper type validation."""
        with pytest.raises(Exception):
            CalibrationSessionRecord(paper_type="")

    def test_special_characters_in_notes(self, db):
        """Should handle special characters in notes."""
        session = db.create_session(paper_type="Test Paper")

        # Update with special characters in notes
        notes = "Test with 'quotes' and \"double quotes\" and special chars: <>{}[]"
        db.update_session(session.id, {"notes": notes})

        retrieved = db.get_session(session.id)
        assert retrieved.notes is not None
        assert "quotes" in retrieved.notes
        assert "double quotes" in retrieved.notes

    def test_unicode_in_paper_type(self, db):
        """Should handle Unicode in paper type."""
        session = db.create_session(
            paper_type="Hahnemühle Platinum Rag"
        )

        retrieved = db.get_session(session.id)
        assert retrieved.paper_type == "Hahnemühle Platinum Rag"

    def test_large_metadata(self, db):
        """Should handle large metadata dictionaries."""
        session = db.create_session(paper_type="Test Paper")
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        record = CalibrationRecord(
            session_id=session.id,
            paper_type="Test Paper",
            exposure_time="3:00",
            metadata=large_metadata
        )
        record_id = db.add_record(record)

        retrieved = db.get_record(record_id)
        assert len(retrieved.metadata) == 100

    def test_concurrent_writes(self, tmp_path):
        """Database should handle multiple sequential writes."""
        db_path = tmp_path / "concurrent_test.db"
        db = CalibrationDatabase(db_path)

        session = db.create_session(paper_type="Test Paper")

        # Rapid sequential writes
        for i in range(20):
            record = CalibrationRecord(
                session_id=session.id,
                iteration_number=i + 1,
                paper_type="Test Paper",
                exposure_time=f"3:{i:02d}",
                midtone_adj=0.01
            )
            db.add_record(record)

        updated = db.get_session(session.id)
        assert updated.iteration_count == 20
        assert updated.total_midtone_adj == pytest.approx(0.20, rel=0.01)
        db.close()

    def test_many_sessions(self, db):
        """Should handle many sessions efficiently."""
        for i in range(50):
            db.create_session(
                paper_type=f"Paper Type {i}",
                name=f"Session {i}"
            )

        sessions = db.list_sessions(limit=100)
        assert len(sessions) == 50

        stats = db.get_statistics()
        assert stats['total_sessions'] == 50
