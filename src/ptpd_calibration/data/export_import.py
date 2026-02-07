"""
Data export and import functionality.

Supports multiple formats: JSON, YAML, XML, CSV for print records,
recipes, and other application data.
"""

import csv
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ptpd_calibration.data.database import PrintDatabase, PrintRecord


class ExportMetadata(BaseModel):
    """Metadata for exported data."""

    export_date: datetime = Field(default_factory=datetime.now)
    export_version: str = Field(default="1.0.0")
    format: str = Field(...)
    record_count: int = Field(default=0)
    notes: str | None = Field(default=None)


class DataExporter:
    """Export data to various formats."""

    def __init__(self, database: PrintDatabase | None = None) -> None:
        """
        Initialize the data exporter.

        Args:
            database: PrintDatabase instance to export from
        """
        self.database = database

    def export_to_json(self, data: list[dict[str, Any]], path: Path) -> None:
        """
        Export data to JSON format.

        Args:
            data: List of dictionaries to export
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = ExportMetadata(
            format="json",
            record_count=len(data),
        )

        export_data = {
            "metadata": metadata.model_dump(),
            "records": data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)

    def export_to_yaml(self, data: list[dict[str, Any]], path: Path) -> None:
        """
        Export data to YAML format.

        Args:
            data: List of dictionaries to export
            path: Output file path

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML export. Install with: pip install pyyaml"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = ExportMetadata(
            format="yaml",
            record_count=len(data),
        )

        export_data = {
            "metadata": metadata.model_dump(),
            "records": data,
        }

        # Convert any non-serializable types
        export_data = json.loads(json.dumps(export_data, default=self._json_serializer))

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

    def export_to_xml(self, data: list[dict[str, Any]], path: Path) -> None:
        """
        Export data to XML format.

        Args:
            data: List of dictionaries to export
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = ExportMetadata(
            format="xml",
            record_count=len(data),
        )

        # Create root element
        root = ET.Element("export")

        # Add metadata
        meta_elem = ET.SubElement(root, "metadata")
        for key, value in metadata.model_dump().items():
            elem = ET.SubElement(meta_elem, key)
            elem.text = str(value)

        # Add records
        records_elem = ET.SubElement(root, "records")
        for record in data:
            record_elem = ET.SubElement(records_elem, "record")
            self._dict_to_xml(record, record_elem)

        # Write to file with pretty formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def export_to_csv(self, data: list[dict[str, Any]], path: Path) -> None:
        """
        Export data to CSV format.

        Args:
            data: List of dictionaries to export
            path: Output file path

        Note:
            Nested structures are converted to JSON strings
        """
        if not data:
            # Create empty CSV
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Get all unique keys from all records
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())

        fieldnames = sorted(all_keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in data:
                # Convert complex types to JSON strings
                row = {}
                for key in fieldnames:
                    value = record.get(key)
                    if value is None:
                        row[key] = ""
                    elif isinstance(value, (list, dict)):
                        row[key] = json.dumps(value)
                    elif isinstance(value, (datetime, UUID)):
                        row[key] = str(value)
                    else:
                        row[key] = value
                writer.writerow(row)

    def export_prints(
        self, filters: dict[str, Any] | None, format: str, path: Path
    ) -> int:
        """
        Export print records from database.

        Args:
            filters: Optional filters to apply (None = export all)
            format: Export format (json, yaml, xml, csv)
            path: Output file path

        Returns:
            Number of records exported

        Raises:
            ValueError: If database is not set or format is invalid
        """
        if self.database is None:
            raise ValueError("Database not set")

        # Get prints
        if filters:
            prints = self.database.filter_prints(filters)
        else:
            prints = self.database.filter_prints({})

        # Convert to dicts
        data = [p.to_dict() for p in prints]

        # Export based on format
        format_lower = format.lower()
        if format_lower == "json":
            self.export_to_json(data, path)
        elif format_lower == "yaml":
            self.export_to_yaml(data, path)
        elif format_lower == "xml":
            self.export_to_xml(data, path)
        elif format_lower == "csv":
            self.export_to_csv(data, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return len(data)

    def export_recipes(
        self, recipe_ids: list[UUID], format: str, path: Path
    ) -> int:
        """
        Export specific recipes.

        Args:
            recipe_ids: List of recipe UUIDs to export
            format: Export format (json, yaml, xml, csv)
            path: Output file path

        Returns:
            Number of recipes exported

        Note:
            This is a placeholder. Actual implementation would need
            a recipe database/storage system.
        """
        # Placeholder - would need actual recipe storage
        recipes = []
        for recipe_id in recipe_ids:
            recipes.append(
                {
                    "id": str(recipe_id),
                    "name": f"Recipe {recipe_id}",
                    "created_at": datetime.now().isoformat(),
                }
            )

        format_lower = format.lower()
        if format_lower == "json":
            self.export_to_json(recipes, path)
        elif format_lower == "yaml":
            self.export_to_yaml(recipes, path)
        elif format_lower == "xml":
            self.export_to_xml(recipes, path)
        elif format_lower == "csv":
            self.export_to_csv(recipes, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return len(recipes)

    def export_all(self, path: Path) -> dict[str, int]:
        """
        Export all data to a directory structure.

        Args:
            path: Output directory path

        Returns:
            Dictionary with counts of exported items per category
        """
        if self.database is None:
            raise ValueError("Database not set")

        path.mkdir(parents=True, exist_ok=True)

        counts = {}

        # Export all prints
        prints = self.database.filter_prints({})
        self.export_to_json(
            [p.to_dict() for p in prints],
            path / "prints.json",
        )
        counts["prints"] = len(prints)

        # Export statistics
        stats = self.database.get_statistics()
        with open(path / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        counts["statistics"] = 1

        # Create export metadata
        metadata = {
            "export_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "counts": counts,
        }
        with open(path / "export_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return counts

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (UUID, Path)):
            return str(obj)
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Type {type(obj)} not serializable")

    def _dict_to_xml(self, data: dict[str, Any], parent: ET.Element) -> None:
        """Convert dictionary to XML elements recursively."""
        for key, value in data.items():
            # Sanitize key for XML
            key_clean = key.replace(" ", "_")

            if value is None:
                elem = ET.SubElement(parent, key_clean)
                elem.set("null", "true")
            elif isinstance(value, dict):
                elem = ET.SubElement(parent, key_clean)
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                elem = ET.SubElement(parent, key_clean)
                for item in value:
                    item_elem = ET.SubElement(elem, "item")
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                elem = ET.SubElement(parent, key_clean)
                elem.text = str(value)


class DataImporter:
    """Import data from various formats."""

    def __init__(self, database: PrintDatabase | None = None) -> None:
        """
        Initialize the data importer.

        Args:
            database: PrintDatabase instance to import into
        """
        self.database = database

    def import_from_json(self, path: Path) -> dict[str, Any]:
        """
        Import data from JSON format.

        Args:
            path: Path to JSON file

        Returns:
            Imported data dictionary
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return data

    def import_from_yaml(self, path: Path) -> dict[str, Any]:
        """
        Import data from YAML format.

        Args:
            path: Path to YAML file

        Returns:
            Imported data dictionary

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML import. Install with: pip install pyyaml"
            )

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data

    def import_from_xml(self, path: Path) -> dict[str, Any]:
        """
        Import data from XML format.

        Args:
            path: Path to XML file

        Returns:
            Imported data dictionary
        """
        tree = ET.parse(path)
        root = tree.getroot()

        data = {"metadata": {}, "records": []}

        # Parse metadata
        meta_elem = root.find("metadata")
        if meta_elem is not None:
            data["metadata"] = self._xml_to_dict(meta_elem)

        # Parse records
        records_elem = root.find("records")
        if records_elem is not None:
            for record_elem in records_elem.findall("record"):
                data["records"].append(self._xml_to_dict(record_elem))

        return data

    def validate_import_data(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate imported data before merging.

        Args:
            data: Imported data dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check structure
        if "records" not in data:
            errors.append("Missing 'records' key in import data")
            return False, errors

        # Validate each record
        records = data.get("records", [])
        for i, record in enumerate(records):
            try:
                # Try to create PrintRecord to validate
                PrintRecord(**record)
            except Exception as e:
                errors.append(f"Record {i}: {str(e)}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def merge_with_existing(
        self, imported_data: dict[str, Any], strategy: str = "skip"
    ) -> dict[str, int]:
        """
        Merge imported data with existing database.

        Args:
            imported_data: Data to import
            strategy: Conflict resolution strategy:
                     'skip' - Skip records with existing IDs
                     'update' - Update existing records
                     'replace' - Replace existing records

        Returns:
            Dictionary with import statistics

        Raises:
            ValueError: If database is not set or strategy is invalid
        """
        if self.database is None:
            raise ValueError("Database not set")

        valid_strategies = ["skip", "update", "replace"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")

        stats = {
            "total": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
        }

        records = imported_data.get("records", [])
        stats["total"] = len(records)

        for record_data in records:
            try:
                record = PrintRecord(**record_data)

                # Check if record exists
                existing = self.database.get_print(record.id)

                if existing is None:
                    # New record
                    self.database.add_print(record)
                    stats["added"] += 1
                else:
                    # Existing record
                    if strategy == "skip":
                        stats["skipped"] += 1
                    elif strategy in ("update", "replace"):
                        self.database.update_print(record.id, record.to_dict())
                        stats["updated"] += 1

            except Exception:
                stats["errors"] += 1

        return stats

    def import_all(self, path: Path) -> dict[str, int]:
        """
        Import all data from an export directory.

        Args:
            path: Path to export directory

        Returns:
            Dictionary with import statistics

        Raises:
            ValueError: If database is not set
        """
        if self.database is None:
            raise ValueError("Database not set")

        stats = {}

        # Import prints
        prints_path = path / "prints.json"
        if prints_path.exists():
            data = self.import_from_json(prints_path)
            import_stats = self.merge_with_existing(data, strategy="skip")
            stats["prints"] = import_stats

        return stats

    def _xml_to_dict(self, element: ET.Element) -> dict[str, Any]:
        """Convert XML element to dictionary recursively."""
        result = {}

        for child in element:
            # Check if null
            if child.get("null") == "true":
                result[child.tag] = None
            # Check if has children (nested dict or list)
            elif len(child) > 0:
                # Check if it's a list (all children are 'item')
                if all(subchild.tag == "item" for subchild in child):
                    result[child.tag] = [
                        self._xml_to_dict(item) if len(item) > 0 else item.text
                        for item in child
                    ]
                else:
                    result[child.tag] = self._xml_to_dict(child)
            else:
                result[child.tag] = child.text

        return result
