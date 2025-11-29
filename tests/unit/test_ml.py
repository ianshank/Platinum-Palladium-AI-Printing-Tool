"""
Tests for ML prediction and refinement.
"""

import tempfile
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
from ptpd_calibration.ml.database import CalibrationDatabase
from ptpd_calibration.ml.active_learning import ActiveLearner


class TestCalibrationDatabase:
    """Tests for CalibrationDatabase."""

    @pytest.fixture
    def sample_records(self):
        """Create sample calibration records."""
        records = []
        papers = ["Arches Platine", "Bergger COT320", "Hahnemühle Platinum Rag"]

        for i, paper in enumerate(papers):
            for j in range(3):
                record = CalibrationRecord(
                    paper_type=paper,
                    exposure_time=150.0 + j * 30,
                    metal_ratio=0.3 + j * 0.2,
                    chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                    contrast_agent=ContrastAgent.NA2 if j > 0 else ContrastAgent.NONE,
                    contrast_amount=j * 3.0,
                    developer=DeveloperType.POTASSIUM_OXALATE,
                    measured_densities=[0.1 + k * 0.1 for k in range(21)],
                )
                records.append(record)

        return records

    def test_add_and_get_record(self, sample_records):
        """Test adding and retrieving records."""
        db = CalibrationDatabase()
        record = sample_records[0]

        db.add_record(record)

        retrieved = db.get_record(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.paper_type == record.paper_type

    def test_get_all_records(self, sample_records):
        """Test getting all records."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        all_records = db.get_all_records()
        assert len(all_records) == len(sample_records)

    def test_get_records_for_paper(self, sample_records):
        """Test filtering by paper type."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        arches_records = db.get_records_for_paper("Arches Platine")
        assert len(arches_records) == 3
        assert all(r.paper_type == "Arches Platine" for r in arches_records)

    def test_get_similar_records(self, sample_records):
        """Test finding similar records."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        reference = sample_records[0]
        similar = db.get_similar_records(reference, max_records=5)

        # Should find similar records
        assert len(similar) > 0
        # First result should be most similar
        _, score = similar[0]
        assert score > 0.3

    def test_query_with_filters(self, sample_records):
        """Test querying with filters."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        # Filter by paper type
        results = db.query(paper_type="Arches Platine")
        assert len(results) == 3

        # Filter by exposure range
        results = db.query(min_exposure=170, max_exposure=200)
        assert all(170 <= r.exposure_time <= 200 for r in results)

    def test_save_and_load(self, sample_records, tmp_path):
        """Test saving and loading database."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        save_path = tmp_path / "calibrations.json"
        db.save(save_path)

        assert save_path.exists()

        loaded_db = CalibrationDatabase.load(save_path)
        assert len(loaded_db) == len(sample_records)

        # Check a specific record
        original = sample_records[0]
        loaded = loaded_db.get_record(original.id)
        assert loaded is not None
        assert loaded.paper_type == original.paper_type

    def test_statistics(self, sample_records):
        """Test database statistics."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        stats = db.get_statistics()

        assert stats["total_records"] == len(sample_records)
        assert len(stats["paper_types"]) == 3
        assert "exposure_range" in stats

    def test_summary(self, sample_records):
        """Test summary generation."""
        db = CalibrationDatabase()
        for record in sample_records:
            db.add_record(record)

        summary = db.summary()
        assert "9 records" in summary
        assert "3 paper types" in summary


class TestActiveLearner:
    """Tests for ActiveLearner."""

    @pytest.fixture
    def base_record(self):
        """Create a base calibration record."""
        return CalibrationRecord(
            paper_type="Arches Platine",
            exposure_time=180.0,
            metal_ratio=0.5,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

    def test_suggest_exposure_bracket(self, base_record):
        """Test exposure bracket suggestion."""
        learner = ActiveLearner()

        brackets = learner.suggest_exposure_bracket(
            base_record,
            bracket_stops=0.5,
            num_brackets=5,
        )

        assert len(brackets) == 5
        # Should be centered around base exposure
        assert any(abs(b - 180.0) < 1 for b in brackets)
        # Should span range
        assert min(brackets) < 180.0
        assert max(brackets) > 180.0

    def test_suggest_metal_ratio_series(self, base_record):
        """Test metal ratio series suggestion."""
        learner = ActiveLearner()

        series = learner.suggest_metal_ratio_series(base_record, num_steps=5)

        assert len(series) == 5
        assert 0.0 in series
        assert 1.0 in series
        assert 0.5 in series

    def test_suggest_contrast_series(self, base_record):
        """Test contrast agent series suggestion."""
        learner = ActiveLearner()

        series = learner.suggest_contrast_series(
            base_record,
            agent_type="na2",
            num_steps=5,
        )

        assert len(series) == 5
        assert 0.0 in series
        assert max(series) > 0

    def test_evaluate_calibration_quality_good(self, base_record):
        """Test quality evaluation for good calibration."""
        learner = ActiveLearner()

        # Add good density measurements
        base_record.measured_densities = list(np.linspace(0.1, 2.0, 21))

        quality = learner.evaluate_calibration_quality(base_record)

        assert "quality_score" in quality
        assert quality["quality_score"] > 0.5
        assert "metrics" in quality
        assert "recommendations" in quality

    def test_evaluate_calibration_quality_poor(self, base_record):
        """Test quality evaluation for poor calibration."""
        learner = ActiveLearner()

        # Add poor density measurements (low range)
        base_record.measured_densities = list(np.linspace(0.1, 0.8, 21))

        quality = learner.evaluate_calibration_quality(base_record)

        assert quality["quality_score"] < 0.8
        assert any("range" in r.lower() for r in quality["recommendations"])

    def test_suggest_next_experiment(self, base_record):
        """Test experiment suggestion."""
        learner = ActiveLearner()

        variations = [
            {"metal_ratio": 0.3},
            {"metal_ratio": 0.7},
            {"exposure_time": 150},
            {"exposure_time": 210},
        ]

        suggestion = learner.suggest_next_experiment(
            base_record,
            variations,
            strategy="diversity",
        )

        assert "variation" in suggestion
        assert "rationale" in suggestion
        assert "score" in suggestion


class TestTransferLearner:
    """Tests for TransferLearner."""

    @pytest.fixture
    def populated_db(self):
        """Create a populated database."""
        db = CalibrationDatabase()

        # Add some records for Arches Platine
        for i in range(3):
            record = CalibrationRecord(
                paper_type="Arches Platine",
                exposure_time=150.0 + i * 30,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                developer=DeveloperType.POTASSIUM_OXALATE,
                measured_densities=[0.1 + k * 0.1 for k in range(21)],
            )
            db.add_record(record)

        return db

    def test_find_similar_papers(self, populated_db):
        """Test finding similar papers."""
        from ptpd_calibration.ml.transfer import TransferLearner

        transfer = TransferLearner(populated_db)

        # Should find Arches Platine as similar to new cotton paper
        similar = transfer.find_similar_papers(
            "New Cotton Paper",
            paper_weight=300,
        )

        assert len(similar) > 0
        # First result should be the known paper
        paper_name, score = similar[0]
        assert paper_name == "Arches Platine"

    def test_transfer_curve(self, populated_db):
        """Test curve transfer."""
        from ptpd_calibration.ml.transfer import TransferLearner

        transfer = TransferLearner(populated_db)

        source_records = populated_db.get_records_for_paper("Arches Platine")

        transferred = transfer.transfer_curve(
            source_records,
            target_paper="New Paper",
            adjustment_factor=0.9,
        )

        assert len(transferred) == 21
        assert transferred[0] >= 0

    def test_suggest_starting_parameters(self, populated_db):
        """Test parameter suggestion."""
        from ptpd_calibration.ml.transfer import TransferLearner

        transfer = TransferLearner(populated_db)

        params = transfer.suggest_starting_parameters(
            "Arches Platine",  # Known paper
        )

        assert "exposure_time" in params
        assert "metal_ratio" in params
        assert params["source_paper"] == "Arches Platine"

    def test_estimate_exposure_adjustment(self, populated_db):
        """Test exposure adjustment estimation."""
        from ptpd_calibration.ml.transfer import TransferLearner

        transfer = TransferLearner(populated_db)

        adjusted = transfer.estimate_exposure_adjustment(
            source_paper="Arches Platine",
            target_paper="Gampi Torinoko",  # Different paper type
            source_exposure=180.0,
        )

        # Should be different from source
        assert adjusted != 180.0
        assert adjusted > 0
