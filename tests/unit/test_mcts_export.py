"""Tests for the MCTS result export module.

Covers MCTSResultExporter export formats (recipe JSON, CSV, QTR curve).
"""

import csv
import io
import json
from datetime import datetime
from uuid import uuid4

import pytest

from ptpd_calibration.mcts.export import MCTSResultExporter
from ptpd_calibration.mcts.types import SearchResult


@pytest.fixture()
def sample_result() -> SearchResult:
    """Create a sample SearchResult for testing."""
    return SearchResult(
        id=uuid4(),
        best_parameters={
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        },
        predicted_curve=[0.08 + i * 0.09 for i in range(21)],
        quality_score=0.85,
        num_simulations=800,
        search_time_seconds=5.2,
        alternatives=[],
        constraint_violations=[],
        visit_distribution={},
        paper_type="Arches Platine",
        uv_source="UV LED",
        timestamp=datetime.now(),
    )


@pytest.fixture()
def exporter() -> MCTSResultExporter:
    return MCTSResultExporter()


class TestRecipeJsonExport:
    def test_returns_valid_dict(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert isinstance(result, dict)

    def test_contains_parameters(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert "parameters" in result
        assert result["parameters"]["metal_ratio"] == 0.5

    def test_contains_quality_info(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert "quality_score" in result
        assert result["quality_score"] == pytest.approx(0.85)

    def test_contains_predicted_curve(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert "predicted_curve" in result
        assert len(result["predicted_curve"]["output_values"]) == 21

    def test_contains_metadata(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert "metadata" in result
        assert result["metadata"]["num_simulations"] == 800

    def test_contains_chemistry_info(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        assert "chemistry_info" in result
        assert "Pt:50%" in result["chemistry_info"]

    def test_json_serializable(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_recipe_json(sample_result)
        # Should not raise
        serialized = json.dumps(result, default=str)
        assert len(serialized) > 0


class TestCsvExport:
    def test_returns_valid_csv(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_csv(sample_result)
        assert isinstance(result, str)

    def test_csv_parseable(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_csv(sample_result)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 22  # header + 21 data rows

    def test_csv_has_header(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_csv(sample_result)
        first_line = result.strip().split("\n")[0]
        assert "step" in first_line.lower()
        assert "density" in first_line.lower()

    def test_csv_data_values(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_csv(sample_result)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        # First density should be close to dmin
        assert float(rows[0]["density"]) == pytest.approx(0.08, abs=0.01)


class TestQtrCurveExport:
    def test_returns_list(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_qtr_curve(sample_result)
        assert isinstance(result, list)

    def test_256_values(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_qtr_curve(sample_result)
        assert len(result) == 256

    def test_values_in_range(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_qtr_curve(sample_result)
        assert all(0 <= v <= 255 for v in result)

    def test_integers(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        result = exporter.to_qtr_curve(sample_result)
        assert all(isinstance(v, int) for v in result)

    def test_monotonic(
        self, exporter: MCTSResultExporter, sample_result: SearchResult
    ) -> None:
        """QTR curves should be monotonically non-decreasing or non-increasing."""
        result = exporter.to_qtr_curve(sample_result)
        diffs = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        assert all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)


class TestExportToFile:
    def test_export_json_file(
        self, exporter: MCTSResultExporter, sample_result: SearchResult, tmp_path: str
    ) -> None:
        path = exporter.export_to_file(sample_result, str(tmp_path), format="json")
        assert path.endswith(".json")

    def test_export_csv_file(
        self, exporter: MCTSResultExporter, sample_result: SearchResult, tmp_path: str
    ) -> None:
        path = exporter.export_to_file(sample_result, str(tmp_path), format="csv")
        assert path.endswith(".csv")

    def test_export_recipe_bundle(
        self, exporter: MCTSResultExporter, sample_result: SearchResult, tmp_path: str
    ) -> None:
        path = exporter.export_to_file(sample_result, str(tmp_path), format="recipe")
        assert isinstance(path, str)

    def test_export_invalid_format(
        self, exporter: MCTSResultExporter, sample_result: SearchResult, tmp_path: str
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_to_file(sample_result, str(tmp_path), format="invalid")
