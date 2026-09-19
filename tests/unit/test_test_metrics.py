"""Tests for scripts/test_metrics.py (ADR-0007: README numbers come from artifacts)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test_metrics.py"


@pytest.fixture(scope="module")
def metrics_module():
    spec = importlib.util.spec_from_file_location("test_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["test_metrics"] = module
    spec.loader.exec_module(module)
    return module


JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="5" failures="1" errors="0" skipped="2" time="1.5">
    <testcase classname="a" name="t1" time="0.1"/>
    <testcase classname="a" name="t2" time="0.1"><failure message="boom">boom</failure></testcase>
    <testcase classname="a" name="t3" time="0.1"><skipped message="Skipped: PyTorch not available"/></testcase>
    <testcase classname="a" name="t4" time="0.1"><skipped message="Skipped: PyTorch not available"/></testcase>
    <testcase classname="a" name="t5" time="0.1"/>
  </testsuite>
</testsuites>
"""

COVERAGE_XML = """<?xml version="1.0" ?>
<coverage version="7.0" line-rate="0.64" branch-rate="0.55">
  <packages>
    <package name="src.ptpd_calibration.chemistry" line-rate="0.95" branch-rate="0.86"/>
    <package name="src.ptpd_calibration.curves" line-rate="0.90" branch-rate="0.72"/>
    <package name="src.ptpd_calibration.curves.sub" line-rate="0.80" branch-rate="0.60"/>
    <package name="src.ptpd_calibration.other" line-rate="0.10" branch-rate="0.05"/>
  </packages>
</coverage>
"""

VITEST_JSON = {
    "numTotalTests": 831,
    "numPassedTests": 826,
    "numFailedTests": 5,
    "numPendingTests": 0,
    "numTodoTests": 0,
    "startTime": 1000,
    "testResults": [{"endTime": 11200}],
}


def test_parse_junit_counts_and_skip_reasons(metrics_module, tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT_XML)
    suite = metrics_module.parse_junit(path)
    assert (suite.collected, suite.passed, suite.failed, suite.errors, suite.skipped) == (
        5,
        2,
        1,
        0,
        2,
    )
    assert suite.skip_reasons == {"PyTorch not available": 2}
    assert suite.duration_seconds == pytest.approx(1.5)


def test_parse_coverage_tiers_average_subpackages(metrics_module, tmp_path: Path) -> None:
    path = tmp_path / "coverage.xml"
    path.write_text(COVERAGE_XML)
    cov = metrics_module.parse_coverage_xml(path)
    assert cov.line_rate == pytest.approx(64.0)
    assert cov.branch_rate == pytest.approx(55.0)
    assert cov.packages["ptpd_calibration.chemistry"] == {"line_rate": 95.0, "branch_rate": 86.0}
    # curves and curves.sub are averaged into the curves tier
    assert cov.packages["ptpd_calibration.curves"] == {"line_rate": 85.0, "branch_rate": 66.0}
    assert "ptpd_calibration.other" not in cov.packages


def test_parse_vitest_json(metrics_module, tmp_path: Path) -> None:
    path = tmp_path / "vitest.json"
    path.write_text(json.dumps(VITEST_JSON))
    suite = metrics_module.parse_vitest_json(path)
    assert (suite.collected, suite.passed, suite.failed, suite.skipped) == (831, 826, 5, 0)
    assert suite.duration_seconds == pytest.approx(10.2)


def test_collect_and_render_end_to_end(metrics_module, tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT_XML)
    cov = tmp_path / "coverage.xml"
    cov.write_text(COVERAGE_XML)
    out = tmp_path / "reports" / "metrics.json"
    rc = metrics_module.main(
        [
            "collect",
            "--junit",
            str(junit),
            "--coverage",
            str(cov),
            "--vitest",
            str(tmp_path / "missing.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    assert [s["name"] for s in doc["suites"]] == ["backend"]
    assert doc["missing"] == [str(tmp_path / "missing.json")]
    markdown = metrics_module.render(doc)
    assert "| backend | 5 | 2 | 1 | 0 | 2 |" in markdown
    assert "| ptpd_calibration.chemistry | 95.0 | 86.0 |" in markdown
    assert "Missing artifacts" in markdown
