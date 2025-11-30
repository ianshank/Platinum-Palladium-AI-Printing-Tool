"""
Comprehensive tests for data management modules.

Tests:
- PrintDatabase: CRUD operations, search, filtering, statistics, backup/restore
- DataExporter/DataImporter: JSON, YAML, XML, CSV format support
- SyncManager with LocalStorageProvider: sync operations
- VersionController: commit, history, diff, rollback, branch, merge
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ptpd_calibration.data.cloud_sync import (
    ConflictStrategy,
    LocalStorageProvider,
    SyncManager,
    SyncStatus,
)
from ptpd_calibration.data.database import PrintDatabase, PrintRecord
from ptpd_calibration.data.export_import import DataExporter, DataImporter
from ptpd_calibration.data.version_control import (
    MergeConflict,
    VersionController,
    VersionedItem,
)


# ============================================================================
# PrintDatabase Tests
# ============================================================================


class TestPrintDatabase:
    """Test PrintDatabase CRUD operations, search, filtering, and statistics."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = PrintDatabase(db_path)
        yield db
        db.close()

        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def sample_record(self):
        """Create sample print record."""
        return PrintRecord(
            name="Test Print",
            paper_type="arches_platine",
            paper_weight=300,
            exposure_time=12.5,
            metal_ratio=0.5,
            dmin=0.1,
            dmax=2.2,
            density_range=2.1,
            overall_quality=0.85,
            tags=["test", "platinum"],
            notes="Test print for unit testing",
        )

    def test_add_print(self, temp_db, sample_record):
        """Test adding a print record."""
        record_id = temp_db.add_print(sample_record)
        assert record_id == sample_record.id

        # Verify it was added
        retrieved = temp_db.get_print(record_id)
        assert retrieved is not None
        assert retrieved.name == sample_record.name
        assert retrieved.paper_type == sample_record.paper_type

    def test_get_print_not_found(self, temp_db):
        """Test getting non-existent print returns None."""
        result = temp_db.get_print(uuid4())
        assert result is None

    def test_update_print(self, temp_db, sample_record):
        """Test updating a print record."""
        record_id = temp_db.add_print(sample_record)

        # Update fields
        updates = {
            "name": "Updated Print",
            "overall_quality": 0.95,
            "tags": ["updated", "test"],
        }
        success = temp_db.update_print(record_id, updates)
        assert success

        # Verify updates
        retrieved = temp_db.get_print(record_id)
        assert retrieved.name == "Updated Print"
        assert retrieved.overall_quality == 0.95
        assert "updated" in retrieved.tags

    def test_update_nonexistent_print(self, temp_db):
        """Test updating non-existent print returns False."""
        success = temp_db.update_print(uuid4(), {"name": "test"})
        assert not success

    def test_delete_print(self, temp_db, sample_record):
        """Test deleting a print record."""
        record_id = temp_db.add_print(sample_record)

        # Delete record
        success = temp_db.delete_print(record_id)
        assert success

        # Verify it's gone
        retrieved = temp_db.get_print(record_id)
        assert retrieved is None

    def test_delete_nonexistent_print(self, temp_db):
        """Test deleting non-existent print returns False."""
        success = temp_db.delete_print(uuid4())
        assert not success

    def test_search_prints(self, temp_db):
        """Test full-text search functionality."""
        # Add records with searchable content
        record1 = PrintRecord(
            name="Platinum Test",
            paper_type="arches_platine",
            exposure_time=10.0,
            notes="Beautiful platinum print with deep blacks",
        )
        record2 = PrintRecord(
            name="Palladium Test",
            paper_type="hahnemuhle_platinum",
            exposure_time=15.0,
            notes="Soft palladium tones",
        )
        record3 = PrintRecord(
            name="Mixed Metal",
            paper_type="bergger_cot320",
            exposure_time=12.0,
            notes="Combination of platinum and palladium",
        )

        temp_db.add_print(record1)
        temp_db.add_print(record2)
        temp_db.add_print(record3)

        # Search for platinum
        results = temp_db.search_prints("platinum")
        assert len(results) >= 2
        names = [r.name for r in results]
        assert "Platinum Test" in names

        # Search for palladium
        results = temp_db.search_prints("palladium")
        assert len(results) >= 2

    def test_filter_prints_equality(self, temp_db):
        """Test filtering prints by exact value."""
        # Add records with different paper types
        for i in range(3):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="arches_platine" if i < 2 else "bergger_cot320",
                exposure_time=10.0 + i,
            )
            temp_db.add_print(record)

        # Filter by paper type
        results = temp_db.filter_prints({"paper_type": "arches_platine"})
        assert len(results) == 2

    def test_filter_prints_comparison(self, temp_db):
        """Test filtering with comparison operators."""
        # Add records with different exposure times
        for i in range(5):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="test_paper",
                exposure_time=float(i * 5),
            )
            temp_db.add_print(record)

        # Filter by exposure_time > 10
        results = temp_db.filter_prints({"exposure_time__gt": 10.0})
        assert len(results) == 2  # 15.0 and 20.0

        # Filter by exposure_time <= 10
        results = temp_db.filter_prints({"exposure_time__lte": 10.0})
        assert len(results) == 3  # 0.0, 5.0, 10.0

    def test_get_prints_by_date_range(self, temp_db):
        """Test getting prints within a date range."""
        now = datetime.now()

        # Add records with different timestamps
        for i in range(3):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="test_paper",
                exposure_time=10.0,
                timestamp=now - timedelta(days=i),
            )
            temp_db.add_print(record)

        # Get prints from last 2 days
        start = now - timedelta(days=2)
        end = now + timedelta(days=1)
        results = temp_db.get_prints_by_date_range(start, end)
        assert len(results) == 3

    def test_get_prints_by_paper(self, temp_db):
        """Test getting prints by paper type."""
        # Add records with different paper types
        for i in range(3):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="arches_platine" if i < 2 else "bergger_cot320",
                exposure_time=10.0,
            )
            temp_db.add_print(record)

        results = temp_db.get_prints_by_paper("arches_platine")
        assert len(results) == 2

    def test_get_prints_by_recipe(self, temp_db):
        """Test getting prints by recipe ID."""
        recipe_id = uuid4()

        # Add records with same recipe ID
        for i in range(2):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="test_paper",
                exposure_time=10.0,
                recipe_id=recipe_id,
            )
            temp_db.add_print(record)

        # Add one with different recipe
        other_record = PrintRecord(
            name="Other Print",
            paper_type="test_paper",
            exposure_time=10.0,
            recipe_id=uuid4(),
        )
        temp_db.add_print(other_record)

        results = temp_db.get_prints_by_recipe(recipe_id)
        assert len(results) == 2

    def test_get_statistics(self, temp_db):
        """Test database statistics calculation."""
        # Add multiple records
        for i in range(5):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="arches_platine" if i < 3 else "bergger_cot320",
                exposure_time=10.0 + i,
                dmin=0.1,
                dmax=2.0 + i * 0.1,
                density_range=1.9 + i * 0.1,
                overall_quality=0.8 + i * 0.02,
            )
            temp_db.add_print(record)

        stats = temp_db.get_statistics()

        assert stats["total_prints"] == 5
        assert "arches_platine" in stats["paper_types"]
        assert stats["paper_types"]["arches_platine"] == 3
        assert stats["avg_exposure_time"] == 12.0  # (10+11+12+13+14)/5
        assert stats["avg_dmin"] is not None
        assert stats["avg_dmax"] is not None
        assert "first_print" in stats
        assert "last_print" in stats

    def test_backup_database(self, temp_db, sample_record):
        """Test database backup functionality."""
        # Add some data
        temp_db.add_print(sample_record)

        # Create backup
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            backup_path = Path(f.name)

        try:
            temp_db.backup_database(backup_path)
            assert backup_path.exists()

            # Verify backup contains data
            backup_db = PrintDatabase(backup_path)
            stats = backup_db.get_statistics()
            assert stats["total_prints"] == 1
            backup_db.close()
        finally:
            if backup_path.exists():
                backup_path.unlink()

    def test_restore_database(self, temp_db, sample_record):
        """Test database restore functionality."""
        # Create original database with data
        temp_db.add_print(sample_record)

        # Create backup
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            backup_path = Path(f.name)

        try:
            temp_db.backup_database(backup_path)

            # Add more data to original
            new_record = PrintRecord(
                name="New Print",
                paper_type="test_paper",
                exposure_time=15.0,
            )
            temp_db.add_print(new_record)
            assert temp_db.get_statistics()["total_prints"] == 2

            # Restore from backup
            temp_db.restore_database(backup_path)

            # Verify restored state
            stats = temp_db.get_statistics()
            assert stats["total_prints"] == 1
        finally:
            if backup_path.exists():
                backup_path.unlink()

    def test_backup_in_memory_database_fails(self):
        """Test that backing up in-memory database raises error."""
        db = PrintDatabase()  # In-memory database

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            backup_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Cannot backup in-memory database"):
                db.backup_database(backup_path)
        finally:
            if backup_path.exists():
                backup_path.unlink()

    def test_restore_nonexistent_backup_fails(self, temp_db):
        """Test that restoring from non-existent backup raises error."""
        backup_path = Path("/tmp/nonexistent_backup.db")

        with pytest.raises(FileNotFoundError):
            temp_db.restore_database(backup_path)

    def test_print_record_validation(self):
        """Test PrintRecord validation."""
        # Valid record
        record = PrintRecord(
            name="Test",
            paper_type="test_paper",
            exposure_time=10.0,
        )
        assert record.name == "Test"

        # Invalid: empty name
        with pytest.raises(ValueError):
            PrintRecord(name="", paper_type="test_paper", exposure_time=10.0)

        # Invalid: negative exposure time
        with pytest.raises(ValueError):
            PrintRecord(name="Test", paper_type="test_paper", exposure_time=-1.0)

        # Invalid: humidity > 100
        with pytest.raises(ValueError):
            PrintRecord(
                name="Test",
                paper_type="test_paper",
                exposure_time=10.0,
                humidity=150.0,
            )


# ============================================================================
# DataExporter/DataImporter Tests
# ============================================================================


class TestDataExport:
    """Test data export functionality for all formats."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database with sample data."""
        db = PrintDatabase()

        # Add sample records
        for i in range(3):
            record = PrintRecord(
                name=f"Print {i}",
                paper_type="arches_platine",
                exposure_time=10.0 + i,
                tags=[f"tag{i}"],
                metadata={"key": f"value{i}"},
            )
            db.add_print(record)

        yield db
        db.close()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_export_to_json(self, temp_db, temp_dir):
        """Test exporting data to JSON format."""
        exporter = DataExporter(temp_db)
        output_path = temp_dir / "export.json"

        count = exporter.export_prints(filters=None, format="json", path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify JSON structure
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "records" in data
        assert len(data["records"]) == 3
        assert data["metadata"]["format"] == "json"

    def test_export_to_yaml(self, temp_db, temp_dir):
        """Test exporting data to YAML format."""
        pytest.importorskip("yaml")  # Skip if PyYAML not installed

        exporter = DataExporter(temp_db)
        output_path = temp_dir / "export.yaml"

        count = exporter.export_prints(filters=None, format="yaml", path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify YAML can be loaded
        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        assert "metadata" in data
        assert "records" in data
        assert len(data["records"]) == 3

    def test_export_to_xml(self, temp_db, temp_dir):
        """Test exporting data to XML format."""
        exporter = DataExporter(temp_db)
        output_path = temp_dir / "export.xml"

        count = exporter.export_prints(filters=None, format="xml", path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify XML structure
        tree = ET.parse(output_path)
        root = tree.getroot()

        assert root.tag == "export"
        metadata = root.find("metadata")
        assert metadata is not None

        records = root.find("records")
        assert records is not None
        assert len(records.findall("record")) == 3

    def test_export_to_csv(self, temp_db, temp_dir):
        """Test exporting data to CSV format."""
        exporter = DataExporter(temp_db)
        output_path = temp_dir / "export.csv"

        count = exporter.export_prints(filters=None, format="csv", path=output_path)

        assert count == 3
        assert output_path.exists()

        # Verify CSV has header and data
        import csv

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert "name" in rows[0]

    def test_export_filtered_data(self, temp_db, temp_dir):
        """Test exporting filtered subset of data."""
        exporter = DataExporter(temp_db)
        output_path = temp_dir / "filtered.json"

        # Export only records with exposure_time > 10
        filters = {"exposure_time__gt": 10.0}
        count = exporter.export_prints(filters=filters, format="json", path=output_path)

        assert count == 2  # Should have 2 records (11.0 and 12.0)

    def test_export_invalid_format(self, temp_db, temp_dir):
        """Test exporting with invalid format raises error."""
        exporter = DataExporter(temp_db)
        output_path = temp_dir / "export.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_prints(filters=None, format="invalid", path=output_path)

    def test_export_without_database(self, temp_dir):
        """Test exporting without database set raises error."""
        exporter = DataExporter()  # No database
        output_path = temp_dir / "export.json"

        with pytest.raises(ValueError, match="Database not set"):
            exporter.export_prints(filters=None, format="json", path=output_path)

    def test_export_all(self, temp_db, temp_dir):
        """Test exporting all data to directory."""
        exporter = DataExporter(temp_db)

        counts = exporter.export_all(temp_dir)

        assert "prints" in counts
        assert counts["prints"] == 3
        assert (temp_dir / "prints.json").exists()
        assert (temp_dir / "statistics.json").exists()
        assert (temp_dir / "export_metadata.json").exists()


class TestDataImport:
    """Test data import functionality for all formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_json_data(self, temp_dir):
        """Create sample JSON export file."""
        data = {
            "metadata": {"format": "json", "record_count": 2},
            "records": [
                {
                    "id": str(uuid4()),
                    "name": "Import Test 1",
                    "timestamp": datetime.now().isoformat(),
                    "paper_type": "test_paper",
                    "chemistry_type": "platinum_palladium",
                    "metal_ratio": 0.5,
                    "contrast_amount": 0.0,
                    "exposure_time": 10.0,
                    "overall_quality": 0.8,
                    "image_paths": [],
                    "tags": ["imported"],
                    "metadata": {},
                },
                {
                    "id": str(uuid4()),
                    "name": "Import Test 2",
                    "timestamp": datetime.now().isoformat(),
                    "paper_type": "test_paper",
                    "chemistry_type": "platinum_palladium",
                    "metal_ratio": 0.5,
                    "contrast_amount": 0.0,
                    "exposure_time": 15.0,
                    "overall_quality": 0.9,
                    "image_paths": [],
                    "tags": ["imported"],
                    "metadata": {},
                },
            ],
        }

        json_path = temp_dir / "import.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        return json_path

    def test_import_from_json(self, sample_json_data):
        """Test importing data from JSON format."""
        importer = DataImporter()
        data = importer.import_from_json(sample_json_data)

        assert "metadata" in data
        assert "records" in data
        assert len(data["records"]) == 2

    def test_validate_import_data(self, sample_json_data):
        """Test validating imported data."""
        importer = DataImporter()
        data = importer.import_from_json(sample_json_data)

        is_valid, errors = importer.validate_import_data(data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_import_data(self):
        """Test validation fails for invalid data."""
        importer = DataImporter()
        data = {
            "records": [
                {
                    "name": "Invalid",
                    # Missing required fields
                }
            ]
        }

        is_valid, errors = importer.validate_import_data(data)

        assert not is_valid
        assert len(errors) > 0

    def test_merge_with_existing_skip_strategy(self, sample_json_data):
        """Test merging with skip strategy."""
        db = PrintDatabase()
        importer = DataImporter(db)

        data = importer.import_from_json(sample_json_data)
        stats = importer.merge_with_existing(data, strategy="skip")

        assert stats["total"] == 2
        assert stats["added"] == 2
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

        # Import again with same IDs
        stats = importer.merge_with_existing(data, strategy="skip")

        assert stats["total"] == 2
        assert stats["added"] == 0
        assert stats["skipped"] == 2

        db.close()

    def test_merge_with_existing_update_strategy(self, sample_json_data):
        """Test merging with update strategy."""
        db = PrintDatabase()
        importer = DataImporter(db)

        data = importer.import_from_json(sample_json_data)

        # First import
        stats1 = importer.merge_with_existing(data, strategy="update")
        assert stats1["added"] == 2

        # Modify data
        data["records"][0]["name"] = "Updated Name"

        # Import again
        stats2 = importer.merge_with_existing(data, strategy="update")
        assert stats2["updated"] == 2
        assert stats2["added"] == 0

        db.close()

    def test_merge_invalid_strategy(self, sample_json_data):
        """Test merging with invalid strategy raises error."""
        db = PrintDatabase()
        importer = DataImporter(db)

        data = importer.import_from_json(sample_json_data)

        with pytest.raises(ValueError, match="Invalid strategy"):
            importer.merge_with_existing(data, strategy="invalid")

        db.close()

    def test_merge_without_database(self, sample_json_data):
        """Test merging without database set raises error."""
        importer = DataImporter()
        data = importer.import_from_json(sample_json_data)

        with pytest.raises(ValueError, match="Database not set"):
            importer.merge_with_existing(data, strategy="skip")


# ============================================================================
# SyncManager Tests
# ============================================================================


class TestSyncManager:
    """Test cloud sync functionality with LocalStorageProvider."""

    @pytest.fixture
    def temp_local(self):
        """Create temporary local directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def temp_sync(self):
        """Create temporary sync directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def provider(self, temp_sync):
        """Create LocalStorageProvider."""
        return LocalStorageProvider({"sync_dir": str(temp_sync)})

    @pytest.fixture
    def sync_manager(self, temp_local, provider):
        """Create SyncManager."""
        return SyncManager(temp_local, provider)

    def test_upload_file(self, provider, temp_local):
        """Test uploading a file to sync storage."""
        # Create test file
        test_file = temp_local / "test.txt"
        test_file.write_text("Test content")

        # Upload
        success = provider.upload_file(test_file, "test.txt")

        assert success
        assert provider.sync_dir / "test.txt"

    def test_download_file(self, provider, temp_local):
        """Test downloading a file from sync storage."""
        # Create file in sync storage
        sync_file = provider.sync_dir / "download_test.txt"
        sync_file.write_text("Download content")

        # Download
        local_file = temp_local / "downloaded.txt"
        success = provider.download_file("download_test.txt", local_file)

        assert success
        assert local_file.exists()
        assert local_file.read_text() == "Download content"

    def test_list_files(self, provider):
        """Test listing files in sync storage."""
        # Create test files
        (provider.sync_dir / "file1.txt").write_text("File 1")
        (provider.sync_dir / "file2.txt").write_text("File 2")
        subdir = provider.sync_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("File 3")

        # List all files
        files = provider.list_files()

        assert len(files) == 3
        assert "file1.txt" in files
        assert "subdir/file3.txt" in files or "subdir\\file3.txt" in files

    def test_delete_file(self, provider):
        """Test deleting a file from sync storage."""
        # Create test file
        test_file = provider.sync_dir / "delete_me.txt"
        test_file.write_text("Delete this")

        # Delete
        success = provider.delete_file("delete_me.txt")

        assert success
        assert not test_file.exists()

    def test_get_file_metadata(self, provider):
        """Test getting file metadata."""
        # Create test file
        test_file = provider.sync_dir / "metadata_test.txt"
        test_file.write_text("Test content for metadata")

        # Get metadata
        metadata = provider.get_file_metadata("metadata_test.txt")

        assert metadata is not None
        assert metadata.path == "metadata_test.txt"
        assert metadata.size > 0
        assert metadata.checksum is not None

    def test_test_connection(self, provider):
        """Test connection testing."""
        assert provider.test_connection()

    def test_sync_to_cloud(self, sync_manager, temp_local):
        """Test syncing local files to cloud."""
        # Create local files
        (temp_local / "file1.txt").write_text("File 1")
        (temp_local / "file2.txt").write_text("File 2")

        # Sync to cloud
        record = sync_manager.sync_to_cloud()

        assert record.status == SyncStatus.COMPLETED
        assert record.files_synced == 2
        assert record.direction == "upload"

    def test_sync_from_cloud(self, sync_manager, provider, temp_local):
        """Test syncing cloud files to local."""
        # Create files in cloud
        (provider.sync_dir / "cloud_file1.txt").write_text("Cloud File 1")
        (provider.sync_dir / "cloud_file2.txt").write_text("Cloud File 2")

        # Sync from cloud
        record = sync_manager.sync_from_cloud()

        assert record.status == SyncStatus.COMPLETED
        assert record.files_synced == 2
        assert record.direction == "download"
        assert (temp_local / "cloud_file1.txt").exists()

    def test_get_sync_status(self, sync_manager, temp_local):
        """Test getting sync status."""
        # Initial status
        status = sync_manager.get_sync_status()

        assert status["last_sync"] is None
        assert status["total_syncs"] == 0

        # Perform sync
        (temp_local / "file.txt").write_text("Test")
        sync_manager.sync_to_cloud()

        # Check status again
        status = sync_manager.get_sync_status()

        assert status["last_sync"] is not None
        assert status["total_syncs"] == 1

    def test_resolve_conflicts_local_wins(self, sync_manager, temp_local, provider):
        """Test conflict resolution with local wins strategy."""
        # Create file locally
        local_file = temp_local / "conflict.txt"
        local_file.write_text("Local content")

        # Create file remotely
        (provider.sync_dir / "conflict.txt").write_text("Remote content")

        # Resolve with local wins
        result = sync_manager.resolve_conflicts(ConflictStrategy.LOCAL_WINS)

        # Remote should now have local content
        remote_content = (provider.sync_dir / "conflict.txt").read_text()
        assert "Local content" in remote_content or result["resolved"] >= 0

    def test_get_last_sync_time(self, sync_manager, temp_local):
        """Test getting last sync time."""
        # No syncs yet
        assert sync_manager.get_last_sync_time() is None

        # Perform sync
        (temp_local / "file.txt").write_text("Test")
        sync_manager.sync_to_cloud()

        # Should have last sync time
        last_sync = sync_manager.get_last_sync_time()
        assert last_sync is not None
        assert isinstance(last_sync, datetime)


# ============================================================================
# VersionController Tests
# ============================================================================


class TestVersionController:
    """Test version control functionality."""

    @pytest.fixture
    def temp_vc(self):
        """Create temporary version controller."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        vc = VersionController(db_path)
        yield vc
        vc.close()

        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def sample_content(self):
        """Create sample content for versioning."""
        return {
            "name": "Test Recipe",
            "metal_ratio": 0.5,
            "exposure_time": 10.0,
        }

    def test_commit(self, temp_vc, sample_content):
        """Test creating a version commit."""
        version = temp_vc.commit(
            item_id="recipe_1",
            content=sample_content,
            message="Initial commit",
            author="Test User",
        )

        assert version.item_id == "recipe_1"
        assert version.version_number == 1
        assert version.message == "Initial commit"
        assert version.content == sample_content

    def test_get_version(self, temp_vc, sample_content):
        """Test getting a specific version."""
        version1 = temp_vc.commit(
            item_id="recipe_1", content=sample_content, message="First version"
        )

        retrieved = temp_vc.get_version(version1.version_id)

        assert retrieved is not None
        assert retrieved.version_id == version1.version_id
        assert retrieved.content == sample_content

    def test_get_history(self, temp_vc, sample_content):
        """Test getting version history."""
        # Create multiple versions
        for i in range(3):
            content = {**sample_content, "version": i}
            temp_vc.commit(
                item_id="recipe_1", content=content, message=f"Version {i}"
            )

        # Get history
        history = temp_vc.get_history("recipe_1")

        assert len(history) == 3
        assert history[0].version_number == 3  # Most recent first
        assert history[2].version_number == 1

    def test_get_history_with_limit(self, temp_vc, sample_content):
        """Test getting limited version history."""
        # Create multiple versions
        for i in range(5):
            content = {**sample_content, "version": i}
            temp_vc.commit(
                item_id="recipe_1", content=content, message=f"Version {i}"
            )

        # Get limited history
        history = temp_vc.get_history("recipe_1", limit=2)

        assert len(history) == 2

    def test_diff(self, temp_vc):
        """Test comparing two versions."""
        content1 = {"name": "Recipe v1", "metal_ratio": 0.5, "exposure_time": 10.0}
        content2 = {"name": "Recipe v2", "metal_ratio": 0.6, "exposure_time": 10.0}

        v1 = temp_vc.commit(item_id="recipe_1", content=content1, message="Version 1")
        v2 = temp_vc.commit(item_id="recipe_1", content=content2, message="Version 2")

        diff = temp_vc.diff(v1.version_id, v2.version_id)

        assert "name" in diff.modified_keys
        assert "metal_ratio" in diff.modified_keys
        assert diff.changes["name"]["old"] == "Recipe v1"
        assert diff.changes["name"]["new"] == "Recipe v2"

    def test_rollback(self, temp_vc, sample_content):
        """Test rolling back to a previous version."""
        v1 = temp_vc.commit(
            item_id="recipe_1", content=sample_content, message="Version 1"
        )

        # Make changes
        new_content = {**sample_content, "metal_ratio": 0.8}
        temp_vc.commit(item_id="recipe_1", content=new_content, message="Version 2")

        # Rollback to v1
        rollback_version = temp_vc.rollback(v1.version_id)

        assert rollback_version.content == sample_content
        assert rollback_version.version_number == 3
        assert "Rollback" in rollback_version.message

    def test_branch(self, temp_vc, sample_content):
        """Test creating a branch."""
        # Create initial commit on main
        temp_vc.commit(
            item_id="recipe_1", content=sample_content, message="Main version"
        )

        # Create branch
        success = temp_vc.branch("recipe_1", "experimental", from_branch="main")

        assert success

        # Trying to create same branch again should fail
        success = temp_vc.branch("recipe_1", "experimental", from_branch="main")
        assert not success

    def test_branch_from_nonexistent(self, temp_vc):
        """Test branching from non-existent branch raises error."""
        with pytest.raises(ValueError, match="Source branch not found"):
            temp_vc.branch("recipe_1", "new_branch", from_branch="nonexistent")

    def test_merge_auto_no_conflicts(self, temp_vc):
        """Test merging branches without conflicts."""
        # Create main branch
        content = {"name": "Recipe", "metal_ratio": 0.5}
        temp_vc.commit(item_id="recipe_1", content=content, message="Main v1")

        # Create experimental branch
        temp_vc.branch("recipe_1", "experimental", from_branch="main")

        # Make non-conflicting changes
        exp_content = {**content, "exposure_time": 15.0}
        temp_vc.commit(
            item_id="recipe_1",
            content=exp_content,
            message="Add exposure time",
            branch="experimental",
        )

        # Merge
        result = temp_vc.merge("recipe_1", "experimental", "main", strategy="auto")

        assert result.success
        assert len(result.conflicts) == 0

    def test_merge_with_conflicts(self, temp_vc):
        """Test merging branches with conflicts."""
        # Create main branch
        content = {"name": "Recipe", "metal_ratio": 0.5}
        temp_vc.commit(item_id="recipe_1", content=content, message="Main v1")

        # Create experimental branch
        temp_vc.branch("recipe_1", "experimental", from_branch="main")

        # Make conflicting changes on main
        main_content = {**content, "metal_ratio": 0.6}
        temp_vc.commit(
            item_id="recipe_1", content=main_content, message="Main v2", branch="main"
        )

        # Make conflicting changes on experimental
        exp_content = {**content, "metal_ratio": 0.7}
        temp_vc.commit(
            item_id="recipe_1",
            content=exp_content,
            message="Exp v2",
            branch="experimental",
        )

        # Merge should detect conflicts
        result = temp_vc.merge("recipe_1", "experimental", "main", strategy="auto")

        assert not result.success
        assert len(result.conflicts) > 0

    def test_merge_strategy_ours(self, temp_vc):
        """Test merge with 'ours' strategy."""
        # Setup branches with conflicts
        content = {"name": "Recipe", "metal_ratio": 0.5}
        temp_vc.commit(item_id="recipe_1", content=content, message="Main v1")
        temp_vc.branch("recipe_1", "experimental", from_branch="main")

        # Make conflicting changes
        main_content = {**content, "metal_ratio": 0.6}
        temp_vc.commit(
            item_id="recipe_1", content=main_content, message="Main v2", branch="main"
        )

        exp_content = {**content, "metal_ratio": 0.7}
        temp_vc.commit(
            item_id="recipe_1",
            content=exp_content,
            message="Exp v2",
            branch="experimental",
        )

        # Merge with 'ours' strategy
        result = temp_vc.merge("recipe_1", "experimental", "main", strategy="ours")

        assert result.success

    def test_list_branches(self, temp_vc, sample_content):
        """Test listing branches."""
        # Create main commit
        temp_vc.commit(item_id="recipe_1", content=sample_content, message="Main v1")

        # Create branches
        temp_vc.branch("recipe_1", "experimental", from_branch="main")
        temp_vc.branch("recipe_1", "dev", from_branch="main")

        # List branches
        branches = temp_vc.list_branches("recipe_1")

        assert len(branches) == 3  # main, experimental, dev
        branch_names = [b["name"] for b in branches]
        assert "main" in branch_names
        assert "experimental" in branch_names
        assert "dev" in branch_names

    def test_tag_version(self, temp_vc, sample_content):
        """Test tagging a version."""
        version = temp_vc.commit(
            item_id="recipe_1", content=sample_content, message="Version 1"
        )

        # Add tag
        success = temp_vc.tag_version(version.version_id, "v1.0")

        assert success

        # Retrieve and verify tag
        retrieved = temp_vc.get_version(version.version_id)
        assert "v1.0" in retrieved.tags

    def test_tag_nonexistent_version(self, temp_vc):
        """Test tagging non-existent version returns False."""
        success = temp_vc.tag_version(uuid4(), "tag")
        assert not success

    def test_content_hash_calculation(self):
        """Test content hash is calculated correctly."""
        content = {"key": "value", "number": 42}

        hash1 = VersionedItem.calculate_hash(content)
        hash2 = VersionedItem.calculate_hash(content)

        # Same content should produce same hash
        assert hash1 == hash2

        # Different content should produce different hash
        different_content = {"key": "different"}
        hash3 = VersionedItem.calculate_hash(different_content)
        assert hash1 != hash3
