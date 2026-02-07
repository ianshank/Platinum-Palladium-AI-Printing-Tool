"""
Data management module for PTPD Calibration System.

Provides comprehensive data management capabilities including:
- SQLite database for print records
- Export/import in multiple formats (JSON, YAML, XML, CSV)
- Cloud synchronization with pluggable providers
- Git-like version control for recipes and settings
"""

# Database
# Cloud Sync
from ptpd_calibration.data.cloud_sync import (
    CloudProvider,
    ConflictStrategy,
    LocalStorageProvider,
    S3Provider,
    SyncManager,
    SyncRecord,
    SyncStatus,
)
from ptpd_calibration.data.database import PrintDatabase, PrintRecord

# Export/Import
from ptpd_calibration.data.export_import import (
    DataExporter,
    DataImporter,
    ExportMetadata,
)

# Version Control
from ptpd_calibration.data.version_control import (
    MergeConflict,
    MergeResult,
    VersionController,
    VersionDiff,
    VersionedItem,
)

__all__ = [
    # Database
    "PrintRecord",
    "PrintDatabase",
    # Export/Import
    "DataExporter",
    "DataImporter",
    "ExportMetadata",
    # Cloud Sync
    "CloudProvider",
    "LocalStorageProvider",
    "S3Provider",
    "SyncManager",
    "SyncStatus",
    "SyncRecord",
    "ConflictStrategy",
    # Version Control
    "VersionedItem",
    "VersionController",
    "VersionDiff",
    "MergeConflict",
    "MergeResult",
]
