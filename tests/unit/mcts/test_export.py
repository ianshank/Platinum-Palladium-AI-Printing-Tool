"""
Tests for MCTS result export module.

Verifies conversion of SearchResult to various export formats:
- CurveData
- CalibrationRecord
- Recipe JSON
- CSV
- QTR curves
- File export
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ptpd_calibration.mcts.config import MCTSSettings
from ptpd_calibration.mcts.export import MCTSResultExporter
from ptpd_calibration.mcts.types import SearchResult


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Create a sample SearchResult for testing."""
    # Generate a realistic density curve (sigmoid-like)
    num_points = 31
    x = np.linspace(0, 1, num_points)
    # Sigmoid curve: dmin=0.05, dmax=2.0, gamma=1.8
    densities = 0.05 + 1.95 / (1 + np.exp(-10 * (x - 0.5))) ** (1 / 1.8)

    return SearchResult(
        id=uuid4(),
        timestamp=datetime(2026, 2, 16, 10, 30, 0),
        best_parameters={
            "metal_ratio": 0.6,
            "coating_weight": 1.8,
            "ferric_oxalate_pct": 22.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        },
        predicted_curve=densities.tolist(),
        quality_score=0.87,
        visit_distribution={
            "metal_ratio": [0.1, 0.2, 0.4, 0.2, 0.1],
            "exposure_time": [0.05, 0.15, 0.6, 0.15, 0.05],
        },
        num_simulations=800,
        search_time_seconds=12.5,
        constraint_violations=[],
        alternatives=[
            {
                "metal_ratio": 0.55,
                "coating_weight": 1.7,
                "ferric_oxalate_pct": 21.0,
                "exposure_time": 170.0,
                "developer_temp": 24.0,
                "humidity": 48.0,
                "quality_score": 0.85,
            },
            {
                "metal_ratio": 0.65,
                "coating_weight": 1.9,
                "ferric_oxalate_pct": 23.0,
                "exposure_time": 190.0,
                "developer_temp": 26.0,
                "humidity": 52.0,
                "quality_score": 0.84,
            },
        ],
        paper_type="Arches Platine",
        uv_source="LED 365nm",
    )


@pytest.fixture
def minimal_search_result() -> SearchResult:
    """Create a minimal SearchResult with edge cases."""
    return SearchResult(
        id=uuid4(),
        timestamp=datetime.now(),
        best_parameters={"metal_ratio": 0.5},
        predicted_curve=[0.1, 0.5, 1.0, 1.5, 2.0],
        quality_score=0.5,
        visit_distribution={},
        num_simulations=100,
        search_time_seconds=1.0,
        constraint_violations=["low_dmax"],
        alternatives=[],
        paper_type=None,
        uv_source=None,
    )


@pytest.fixture
def exporter() -> MCTSResultExporter:
    """Create exporter instance."""
    return MCTSResultExporter()


@pytest.fixture
def exporter_with_settings() -> MCTSResultExporter:
    """Create exporter with custom settings."""
    settings = MCTSSettings(num_simulations=1000, c_puct=1.5)
    return MCTSResultExporter(settings=settings)


class TestMCTSResultExporter:
    """Tests for MCTSResultExporter class."""

    def test_init_default(self):
        """Test initialization with default settings."""
        exporter = MCTSResultExporter()
        assert exporter.settings is not None
        assert isinstance(exporter.settings, MCTSSettings)
        assert exporter.settings.num_simulations == 800  # Default

    def test_init_custom_settings(self, exporter_with_settings):
        """Test initialization with custom settings."""
        assert exporter_with_settings.settings.num_simulations == 1000
        assert exporter_with_settings.settings.c_puct == 1.5

    def test_to_curve_data_success(self, exporter, sample_search_result):
        """Test conversion to CurveData returns valid data."""
        try:
            from ptpd_calibration.core.models import CurveData
        except ImportError:
            pytest.skip("CurveData not available")

        curve_data = exporter.to_curve_data(sample_search_result)

        # Verify type
        assert isinstance(curve_data, CurveData)

        # Verify fields
        assert curve_data.id == sample_search_result.id
        assert curve_data.paper_type == "Arches Platine"
        assert "Pt60Pd40" in curve_data.name
        assert curve_data.created_at == sample_search_result.timestamp

        # Verify curve data
        assert len(curve_data.input_values) == len(sample_search_result.predicted_curve)
        assert len(curve_data.output_values) == len(sample_search_result.predicted_curve)
        assert curve_data.input_values[0] == 0.0
        assert curve_data.input_values[-1] == 1.0
        assert curve_data.output_values == sample_search_result.predicted_curve

        # Verify metadata
        assert "MCTS" in curve_data.notes
        assert "Quality Score: 0.87" in curve_data.notes

    def test_to_curve_data_correct_number_of_points(
        self, exporter, sample_search_result
    ):
        """Test that CurveData has correct number of points."""
        try:
            curve_data = exporter.to_curve_data(sample_search_result)
        except ImportError:
            pytest.skip("CurveData not available")

        expected_points = len(sample_search_result.predicted_curve)
        assert len(curve_data.input_values) == expected_points
        assert len(curve_data.output_values) == expected_points

    def test_to_calibration_record_success(self, exporter, sample_search_result):
        """Test conversion to CalibrationRecord maps parameters correctly."""
        try:
            from ptpd_calibration.core.models import CalibrationRecord
            from ptpd_calibration.core.types import ChemistryType
        except ImportError:
            pytest.skip("CalibrationRecord not available")

        record = exporter.to_calibration_record(sample_search_result)

        # Verify type
        assert isinstance(record, CalibrationRecord)

        # Verify ID and timestamp
        assert record.id == sample_search_result.id
        assert record.timestamp == sample_search_result.timestamp

        # Verify parameters
        assert record.metal_ratio == 0.6
        assert record.exposure_time == 180.0
        assert record.humidity == 50.0
        assert record.temperature == 25.0

        # Verify chemistry type inference
        assert record.chemistry_type == ChemistryType.PLATINUM_PALLADIUM

        # Verify paper info
        assert record.paper_type == "Arches Platine"
        assert record.uv_source == "LED 365nm"

        # Verify measured densities
        assert len(record.measured_densities) > 0
        assert len(record.measured_densities) <= 21  # Standard step wedge

        # Verify tags
        assert "mcts" in record.tags
        assert "optimized" in record.tags

    def test_to_calibration_record_chemistry_types(self, exporter):
        """Test chemistry type inference from metal ratio."""
        try:
            from ptpd_calibration.core.types import ChemistryType
        except ImportError:
            pytest.skip("ChemistryType not available")

        # Pure platinum (>0.9)
        result_pt = SearchResult(
            best_parameters={"metal_ratio": 0.95},
            predicted_curve=[0.1, 1.0, 2.0],
            quality_score=0.8,
            num_simulations=100,
            search_time_seconds=1.0,
        )
        record_pt = exporter.to_calibration_record(result_pt)
        assert record_pt.chemistry_type == ChemistryType.PLATINUM

        # Pure palladium (<0.1)
        result_pd = SearchResult(
            best_parameters={"metal_ratio": 0.05},
            predicted_curve=[0.1, 1.0, 2.0],
            quality_score=0.8,
            num_simulations=100,
            search_time_seconds=1.0,
        )
        record_pd = exporter.to_calibration_record(result_pd)
        assert record_pd.chemistry_type == ChemistryType.PALLADIUM

        # Mixed (0.1-0.9)
        result_mix = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[0.1, 1.0, 2.0],
            quality_score=0.8,
            num_simulations=100,
            search_time_seconds=1.0,
        )
        record_mix = exporter.to_calibration_record(result_mix)
        assert record_mix.chemistry_type == ChemistryType.PLATINUM_PALLADIUM

    def test_to_recipe_json_contains_all_keys(self, exporter, sample_search_result):
        """Test recipe JSON contains all required keys."""
        recipe = exporter.to_recipe_json(sample_search_result)

        # Verify all required keys
        required_keys = [
            "id",
            "timestamp",
            "name",
            "parameters",
            "predicted_curve",
            "quality_score",
            "alternatives",
            "metadata",
            "chemistry_info",
            "process_summary",
        ]

        for key in required_keys:
            assert key in recipe, f"Missing key: {key}"

        # Verify nested structures
        assert "input_values" in recipe["predicted_curve"]
        assert "output_values" in recipe["predicted_curve"]
        assert "num_points" in recipe["predicted_curve"]

        assert "num_simulations" in recipe["metadata"]
        assert "search_time_seconds" in recipe["metadata"]

        # Verify alternatives have rank
        assert len(recipe["alternatives"]) == 2
        assert recipe["alternatives"][0]["rank"] == 1
        assert recipe["alternatives"][1]["rank"] == 2

    def test_to_recipe_json_quality_score_matches(self, exporter, sample_search_result):
        """Test quality score in recipe matches source."""
        recipe = exporter.to_recipe_json(sample_search_result)

        assert recipe["quality_score"] == sample_search_result.quality_score
        assert recipe["parameters"]["quality_score"] == sample_search_result.quality_score

    def test_to_csv_produces_valid_csv(self, exporter, sample_search_result):
        """Test CSV output has correct columns."""
        csv_output = exporter.to_csv(sample_search_result)

        lines = csv_output.split("\n")

        # Verify header
        assert lines[0] == "step,exposure,density"

        # Verify data rows
        num_points = len(sample_search_result.predicted_curve)
        assert len(lines) == num_points + 1  # Header + data

        # Verify first data row
        parts = lines[1].split(",")
        assert len(parts) == 3
        assert parts[0] == "0"  # Step 0
        assert float(parts[1]) == 0.0  # Exposure starts at 0
        assert float(parts[2]) >= 0.0  # Density is non-negative

        # Verify last data row
        parts_last = lines[-1].split(",")
        assert len(parts_last) == 3
        assert parts_last[0] == str(num_points - 1)  # Last step
        assert float(parts_last[1]) == pytest.approx(1.0, abs=1e-5)  # Exposure ends at 1

    def test_to_qtr_curve_returns_256_values(self, exporter, sample_search_result):
        """Test QTR curve returns 256 values."""
        qtr_values = exporter.to_qtr_curve(sample_search_result)

        assert len(qtr_values) == 256
        assert all(isinstance(v, int) for v in qtr_values)

    def test_to_qtr_curve_values_in_range(self, exporter, sample_search_result):
        """Test QTR curve values are in 0-255 range."""
        qtr_values = exporter.to_qtr_curve(sample_search_result)

        assert all(0 <= v <= 255 for v in qtr_values)
        assert min(qtr_values) >= 0
        assert max(qtr_values) <= 255

    def test_to_qtr_curve_monotonic(self, exporter, sample_search_result):
        """Test QTR curve is generally monotonic (allowing small deviations)."""
        qtr_values = exporter.to_qtr_curve(sample_search_result)

        # Count inversions (where value decreases)
        inversions = sum(
            1 for i in range(len(qtr_values) - 1) if qtr_values[i] > qtr_values[i + 1]
        )

        # Allow up to 10% inversions due to noise
        max_inversions = len(qtr_values) * 0.1
        assert inversions <= max_inversions

    def test_to_qtr_curve_flat_curve_edge_case(self, exporter):
        """Test QTR curve with flat input (all same density)."""
        flat_result = SearchResult(
            best_parameters={"metal_ratio": 0.5},
            predicted_curve=[1.0] * 10,  # Flat curve
            quality_score=0.5,
            num_simulations=100,
            search_time_seconds=1.0,
        )

        qtr_values = exporter.to_qtr_curve(flat_result)

        # Should produce midpoint value
        assert len(qtr_values) == 256
        assert all(v == 128 for v in qtr_values)

    def test_export_to_file_json(self, exporter, sample_search_result, tmp_path):
        """Test export to JSON file."""
        output_path = exporter.export_to_file(
            sample_search_result,
            str(tmp_path),
            format="json",
        )

        # Verify file exists
        assert Path(output_path).exists()
        assert output_path.endswith(".json")

        # Verify contents
        with open(output_path) as f:
            data = json.load(f)

        assert data["quality_score"] == 0.87
        assert data["parameters"]["metal_ratio"] == 0.6
        assert len(data["alternatives"]) == 2

    def test_export_to_file_csv(self, exporter, sample_search_result, tmp_path):
        """Test export to CSV file."""
        output_path = exporter.export_to_file(
            sample_search_result,
            str(tmp_path),
            format="csv",
        )

        # Verify file exists
        assert Path(output_path).exists()
        assert output_path.endswith(".csv")

        # Verify contents
        with open(output_path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "step,exposure,density"
        assert len(lines) > 1

    def test_export_to_file_recipe(self, exporter, sample_search_result, tmp_path):
        """Test export to recipe bundle (multiple files)."""
        base_path = exporter.export_to_file(
            sample_search_result,
            str(tmp_path),
            format="recipe",
        )

        # Verify all files exist
        assert Path(f"{base_path}_recipe.json").exists()
        assert Path(f"{base_path}_curve.csv").exists()
        assert Path(f"{base_path}_qtr.txt").exists()

        # Verify QTR file contents
        with open(f"{base_path}_qtr.txt") as f:
            lines = f.readlines()

        assert "# QuadTone RIP Curve" in lines[0]
        assert "# Quality Score: 0.870" in lines[2]

    def test_export_to_file_creates_directory(
        self, exporter, sample_search_result, tmp_path
    ):
        """Test export creates directory if it doesn't exist."""
        nested_path = tmp_path / "nested" / "dir"
        assert not nested_path.exists()

        output_path = exporter.export_to_file(
            sample_search_result,
            str(nested_path),
            format="json",
        )

        assert Path(output_path).exists()
        assert nested_path.exists()

    def test_export_to_file_invalid_format(self, exporter, sample_search_result, tmp_path):
        """Test export with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_to_file(
                sample_search_result,
                str(tmp_path),
                format="invalid",
            )

    def test_minimal_result_export(self, exporter, minimal_search_result):
        """Test export with minimal SearchResult (edge case)."""
        # Should not raise
        csv_output = exporter.to_csv(minimal_search_result)
        assert "step,exposure,density" in csv_output

        qtr_values = exporter.to_qtr_curve(minimal_search_result)
        assert len(qtr_values) == 256

        recipe = exporter.to_recipe_json(minimal_search_result)
        assert recipe["quality_score"] == 0.5
        assert len(recipe["alternatives"]) == 0

    def test_minimal_result_with_violations(self, exporter, minimal_search_result):
        """Test export with constraint violations."""
        recipe = exporter.to_recipe_json(minimal_search_result)

        assert "low_dmax" in recipe["metadata"]["constraint_violations"]

        # Check notes include violations
        try:
            record = exporter.to_calibration_record(minimal_search_result)
            assert "Constraint Violations" in record.notes
            assert "low_dmax" in record.notes
        except ImportError:
            pytest.skip("CalibrationRecord not available")

    def test_roundtrip_recipe_json(self, exporter, sample_search_result):
        """Test roundtrip: export to JSON and verify contents match."""
        recipe = exporter.to_recipe_json(sample_search_result)

        # Verify all parameters are preserved
        assert recipe["parameters"]["metal_ratio"] == 0.6
        assert recipe["parameters"]["coating_weight"] == 1.8
        assert recipe["parameters"]["ferric_oxalate_pct"] == 22.0
        assert recipe["parameters"]["exposure_time"] == 180.0

        # Verify curve is preserved
        assert recipe["predicted_curve"]["output_values"] == sample_search_result.predicted_curve

        # Verify metadata
        assert recipe["metadata"]["num_simulations"] == 800
        assert recipe["metadata"]["search_time_seconds"] == 12.5
        assert recipe["metadata"]["paper_type"] == "Arches Platine"

    def test_curve_name_generation(self, exporter, sample_search_result):
        """Test curve name generation format."""
        name = exporter._generate_curve_name(sample_search_result)

        # Should contain chemistry info
        assert "Pt60Pd40" in name

        # Should contain paper type
        assert "Arches" in name or "Platine" in name

        # Should contain quality indicator
        assert "Q87" in name

        # Should contain date
        assert "20260216" in name

    def test_chemistry_info_formatting(self, exporter):
        """Test chemistry info string formatting."""
        params = {
            "metal_ratio": 0.6,
            "coating_weight": 1.8,
            "ferric_oxalate_pct": 22.0,
        }

        chemistry_info = exporter._format_chemistry_info(params)

        assert "Pt:60%" in chemistry_info
        assert "Pd:40%" in chemistry_info
        assert "Coating:1.80ml/sq-in" in chemistry_info
        assert "FO:22.0%" in chemistry_info

    def test_process_summary_generation(self, exporter):
        """Test process summary generation."""
        params = {
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        }

        summary = exporter._generate_process_summary(params)

        assert summary["exposure_time_seconds"] == 180.0
        assert summary["developer_temperature_celsius"] == 25.0
        assert summary["ambient_humidity_percent"] == 50.0
        assert "developer_type" in summary

    def test_density_measurements_generation(self, exporter):
        """Test density measurements sampling."""
        curve = list(np.linspace(0.1, 2.0, 50))

        measurements = exporter._generate_density_measurements(curve)

        # Should sample at most 21 points
        assert len(measurements) <= 21

        # Should be within curve range
        assert min(measurements) >= min(curve)
        assert max(measurements) <= max(curve)

    def test_short_curve_measurements(self, exporter):
        """Test density measurements with curve shorter than 21 points."""
        curve = [0.1, 0.5, 1.0, 1.5, 2.0]  # Only 5 points

        measurements = exporter._generate_density_measurements(curve)

        # Should return all points
        assert len(measurements) == 5
        assert set(measurements) == set(curve)


class TestExportIntegration:
    """Integration tests for export pipeline."""

    def test_full_export_pipeline(self, exporter, sample_search_result, tmp_path):
        """Test complete export pipeline for all formats."""
        # Export as recipe bundle
        base_path = exporter.export_to_file(
            sample_search_result,
            str(tmp_path),
            format="recipe",
        )

        # Verify all outputs exist
        json_path = f"{base_path}_recipe.json"
        csv_path = f"{base_path}_curve.csv"
        qtr_path = f"{base_path}_qtr.txt"

        assert Path(json_path).exists()
        assert Path(csv_path).exists()
        assert Path(qtr_path).exists()

        # Load and verify JSON
        with open(json_path) as f:
            recipe = json.load(f)
        assert recipe["quality_score"] == 0.87

        # Load and verify CSV
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 32  # Header + 31 points

        # Load and verify QTR
        with open(qtr_path) as f:
            lines = f.readlines()
        # Header comments + 256 values
        qtr_values = [
            int(line.strip())
            for line in lines
            if not line.startswith("#") and line.strip()
        ]
        assert len(qtr_values) == 256

    def test_export_with_missing_optional_fields(self, exporter, tmp_path):
        """Test export with minimal SearchResult (optional fields missing)."""
        minimal = SearchResult(
            best_parameters={},  # Empty parameters
            predicted_curve=[0.1, 1.0, 2.0],
            quality_score=0.5,
            num_simulations=100,
            search_time_seconds=1.0,
        )

        # Should not raise
        output_path = exporter.export_to_file(
            minimal,
            str(tmp_path),
            format="json",
        )

        # Verify file was created
        assert Path(output_path).exists()

        # Verify contents have defaults
        with open(output_path) as f:
            recipe = json.load(f)

        # Should have default parameter values
        assert "metal_ratio" in recipe["parameters"] or len(recipe["parameters"]) == 1  # quality_score
