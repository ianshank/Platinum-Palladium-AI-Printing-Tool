"""
Calibration database for storing and querying historical records.
"""

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.gcp.storage import StorageBackend


class CalibrationDatabase:
    """
    Database for storing and querying calibration records.

    Provides efficient storage, retrieval, and similarity search
    for building ML training datasets.
    """

    def __init__(
        self, storage_backend: StorageBackend | None = None, db_path: str = "database.json"
    ):
        """Initialize database with optional storage backend."""
        self.records: dict[UUID, CalibrationRecord] = {}
        self._paper_index: dict[str, list[UUID]] = {}
        self._chemistry_index: dict[str, list[UUID]] = {}
        self.storage = storage_backend
        self.db_path = db_path

    def add_record(self, record: CalibrationRecord) -> None:
        """
        Add a calibration record to the database.

        Args:
            record: CalibrationRecord to add.
        """
        self.records[record.id] = record

        # Update indices
        if record.paper_type not in self._paper_index:
            self._paper_index[record.paper_type] = []
        self._paper_index[record.paper_type].append(record.id)

        chem_key = record.chemistry_type.value
        if chem_key not in self._chemistry_index:
            self._chemistry_index[chem_key] = []
        self._chemistry_index[chem_key].append(record.id)

        # Auto-save if storage is configured
        if self.storage:
            self.save()

    def get_record(self, record_id: UUID) -> CalibrationRecord | None:
        """Get a record by ID."""
        return self.records.get(record_id)

    def get_all_records(self) -> list[CalibrationRecord]:
        """Get all records."""
        return list(self.records.values())

    def get_records_for_paper(self, paper_type: str) -> list[CalibrationRecord]:
        """Get all records for a specific paper type."""
        ids = self._paper_index.get(paper_type, [])
        return [self.records[rid] for rid in ids if rid in self.records]

    def get_records_for_chemistry(self, chemistry_type: str) -> list[CalibrationRecord]:
        """Get all records for a specific chemistry type."""
        ids = self._chemistry_index.get(chemistry_type, [])
        return [self.records[rid] for rid in ids if rid in self.records]

    def get_similar_records(
        self,
        reference: CalibrationRecord,
        max_records: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[CalibrationRecord, float]]:
        """
        Find records similar to a reference record.

        Args:
            reference: Reference record to compare against.
            max_records: Maximum number of records to return.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of (record, similarity) tuples, sorted by similarity.
        """
        similarities = []

        for record in self.records.values():
            if record.id == reference.id:
                continue

            similarity = self._calculate_similarity(reference, record)
            if similarity >= min_similarity:
                similarities.append((record, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:max_records]

    def _calculate_similarity(
        self, record1: CalibrationRecord, record2: CalibrationRecord
    ) -> float:
        """
        Calculate similarity between two records.

        Uses weighted combination of categorical and numerical features.
        """
        score = 0.0
        weights = 0.0

        # Paper type match (high weight)
        if record1.paper_type.lower() == record2.paper_type.lower():
            score += 0.3
        weights += 0.3

        # Chemistry type match
        if record1.chemistry_type == record2.chemistry_type:
            score += 0.15
        weights += 0.15

        # Metal ratio similarity
        ratio_diff = abs(record1.metal_ratio - record2.metal_ratio)
        score += 0.15 * max(0, 1 - ratio_diff)
        weights += 0.15

        # Contrast agent match
        if record1.contrast_agent == record2.contrast_agent:
            score += 0.1
            # Also compare amount if same agent
            if record1.contrast_amount > 0 and record2.contrast_amount > 0:
                amount_ratio = min(record1.contrast_amount, record2.contrast_amount) / max(
                    record1.contrast_amount, record2.contrast_amount
                )
                score += 0.05 * amount_ratio
        weights += 0.15

        # Developer match
        if record1.developer == record2.developer:
            score += 0.1
        weights += 0.1

        # Exposure time similarity (log scale)
        if record1.exposure_time > 0 and record2.exposure_time > 0:
            log_ratio = abs(np.log(record1.exposure_time) - np.log(record2.exposure_time))
            score += 0.15 * max(0, 1 - log_ratio / 2)
        weights += 0.15

        return score / weights if weights > 0 else 0.0

    def query(
        self,
        paper_type: str | None = None,
        chemistry_type: str | None = None,
        min_exposure: float | None = None,
        max_exposure: float | None = None,
        min_metal_ratio: float | None = None,
        max_metal_ratio: float | None = None,
        tags: list[str] | None = None,
    ) -> list[CalibrationRecord]:
        """
        Query records with filters.

        Args:
            paper_type: Filter by paper type (case-insensitive).
            chemistry_type: Filter by chemistry type.
            min_exposure: Minimum exposure time.
            max_exposure: Maximum exposure time.
            min_metal_ratio: Minimum Pt ratio.
            max_metal_ratio: Maximum Pt ratio.
            tags: Required tags (all must match).

        Returns:
            List of matching records.
        """
        results = list(self.records.values())

        if paper_type:
            results = [r for r in results if r.paper_type.lower() == paper_type.lower()]

        if chemistry_type:
            results = [r for r in results if r.chemistry_type.value == chemistry_type]

        if min_exposure is not None:
            results = [r for r in results if r.exposure_time >= min_exposure]

        if max_exposure is not None:
            results = [r for r in results if r.exposure_time <= max_exposure]

        if min_metal_ratio is not None:
            results = [r for r in results if r.metal_ratio >= min_metal_ratio]

        if max_metal_ratio is not None:
            results = [r for r in results if r.metal_ratio <= max_metal_ratio]

        if tags:
            results = [r for r in results if all(t in r.tags for t in tags)]

        return results

    def get_statistics(self) -> dict:
        """Get database statistics."""
        if not self.records:
            return {
                "total_records": 0,
                "paper_types": [],
                "chemistry_types": [],
            }

        records = list(self.records.values())

        return {
            "total_records": len(records),
            "paper_types": list(self._paper_index.keys()),
            "chemistry_types": list(self._chemistry_index.keys()),
            "date_range": (
                min(r.timestamp for r in records).isoformat(),
                max(r.timestamp for r in records).isoformat(),
            ),
            "exposure_range": (
                min(r.exposure_time for r in records),
                max(r.exposure_time for r in records),
            ),
        }

    def summary(self) -> str:
        """Generate a summary string."""
        stats = self.get_statistics()
        return (
            f"Calibration Database: {stats['total_records']} records, "
            f"{len(stats.get('paper_types', []))} paper types, "
            f"{len(stats.get('chemistry_types', []))} chemistry types"
        )

    def save(self, path: Path | None = None) -> None:
        """Save database to JSON file."""
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "records": [r.model_dump(mode="json") for r in self.records.values()],
        }

        # Convert UUIDs to strings
        for record in data["records"]:
            record["id"] = str(record["id"])
            if record.get("extraction_id"):
                record["extraction_id"] = str(record["extraction_id"])
            if record.get("curve_id"):
                record["curve_id"] = str(record["curve_id"])

            if record.get("curve_id"):
                record["curve_id"] = str(record["curve_id"])

        json_str = json.dumps(data, indent=2, default=str)

        if self.storage:
            self.storage.save(self.db_path, json_str)
        else:
            # Fallback to local file if path provided (legacy)
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(json_str)

    def load_from_storage(self) -> None:
        """Load database from storage."""
        if not self.storage:
            return

        try:
            data_bytes = self.storage.load(self.db_path)
            data = json.loads(data_bytes)

            self.records = {}
            self._paper_index = {}
            self._chemistry_index = {}

            for record_data in data.get("records", []):
                try:
                    record = CalibrationRecord(**record_data)
                    self.add_record(record)
                except Exception:
                    continue
        except Exception:
            # If load fails (e.g. file doesn't exist), start empty
            pass

    @classmethod
    def load(cls, path: Path) -> "CalibrationDatabase":
        """Load database from JSON file."""
        with open(path) as f:
            data = json.load(f)

        db = cls()
        for record_data in data.get("records", []):
            record = CalibrationRecord(**record_data)
            db.add_record(record)

        return db

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())
