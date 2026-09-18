#!/usr/bin/env python3
"""Collect and render test/coverage metrics from CI artifacts.

README numbers are never typed by hand (ADR-0007). CI runs::

    python scripts/test_metrics.py collect --junit reports/junit.xml \
        --coverage coverage.xml --vitest frontend/coverage/vitest-results.json \
        --out reports/metrics.json
    python scripts/test_metrics.py render reports/metrics.json >> "$GITHUB_STEP_SUMMARY"

Only the standard library is used so the script runs before any dependency is
installed. Every input is optional; missing inputs are reported as absent
rather than silently zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("test_metrics")

# Coverage tiers reported per package (mirrors the CI floor steps).
DEFAULT_TIERS: tuple[str, ...] = (
    "ptpd_calibration.chemistry",
    "ptpd_calibration.curves",
    "ptpd_calibration.mcts",
    "ptpd_calibration.api",
)


@dataclass
class SuiteMetrics:
    """Pass/fail/skip counts for one suite, with skip reasons grouped."""

    name: str
    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    duration_seconds: float | None = None
    source: str | None = None


@dataclass
class CoverageMetrics:
    """Line and branch coverage overall and per tier package."""

    line_rate: float | None = None
    branch_rate: float | None = None
    packages: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str | None = None


def _normalise_skip_reason(message: str | None) -> str:
    """Collapse skip messages into a short, groupable reason."""
    if not message:
        return "unspecified"
    text = message.strip()
    # pytest writes "Skipped: <reason>" or "could not import 'x': ..." etc.
    for prefix in ("Skipped: ", "skipped: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.splitlines()[0] if text else "unspecified"
    return text[:80]


def parse_junit(path: Path, name: str = "backend") -> SuiteMetrics:
    """Parse a JUnit XML report produced by pytest ``--junitxml``."""
    tree = ET.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    metrics = SuiteMetrics(name=name, source=str(path))
    reasons: Counter[str] = Counter()
    duration = 0.0
    for suite in suites:
        metrics.collected += int(suite.get("tests", 0))
        metrics.failed += int(suite.get("failures", 0))
        metrics.errors += int(suite.get("errors", 0))
        metrics.skipped += int(suite.get("skipped", 0))
        duration += float(suite.get("time", 0.0) or 0.0)
        for case in suite.iter("testcase"):
            skipped = case.find("skipped")
            if skipped is not None:
                reasons[_normalise_skip_reason(skipped.get("message"))] += 1
    metrics.passed = metrics.collected - metrics.failed - metrics.errors - metrics.skipped
    metrics.skip_reasons = dict(reasons.most_common())
    metrics.duration_seconds = round(duration, 2)
    logger.debug("junit %s: %s", path, metrics)
    return metrics


def parse_coverage_xml(path: Path, tiers: tuple[str, ...] = DEFAULT_TIERS) -> CoverageMetrics:
    """Parse a Cobertura-style coverage.xml produced by coverage.py."""
    tree = ET.parse(path)
    root = tree.getroot()
    metrics = CoverageMetrics(source=str(path))
    metrics.line_rate = round(float(root.get("line-rate", 0.0)) * 100, 1)
    branch_rate = root.get("branch-rate")
    metrics.branch_rate = round(float(branch_rate) * 100, 1) if branch_rate is not None else None
    for package in root.iter("package"):
        pkg_name = package.get("name", "")
        # coverage.py names packages by dotted path relative to the source root,
        # e.g. "src.ptpd_calibration.curves" or "ptpd_calibration.curves".
        normalised = pkg_name.replace("src.", "", 1)
        for tier in tiers:
            if normalised == tier or normalised.startswith(tier + "."):
                entry = metrics.packages.setdefault(
                    tier, {"line_rate": 0.0, "branch_rate": 0.0, "_n": 0}
                )
                entry["line_rate"] += float(package.get("line-rate", 0.0)) * 100
                entry["branch_rate"] += float(package.get("branch-rate", 0.0)) * 100
                entry["_n"] += 1
    for tier, entry in metrics.packages.items():
        n = max(int(entry.pop("_n")), 1)
        entry["line_rate"] = round(entry["line_rate"] / n, 1)
        entry["branch_rate"] = round(entry["branch_rate"] / n, 1)
        logger.debug("coverage tier %s: %s", tier, entry)
    return metrics


def parse_vitest_json(path: Path) -> SuiteMetrics:
    """Parse the JSON produced by ``vitest run --reporter=json --outputFile``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = SuiteMetrics(name="frontend", source=str(path))
    metrics.collected = int(data.get("numTotalTests", 0))
    metrics.passed = int(data.get("numPassedTests", 0))
    metrics.failed = int(data.get("numFailedTests", 0))
    metrics.skipped = int(data.get("numPendingTests", 0)) + int(data.get("numTodoTests", 0))
    start = data.get("startTime")
    end = max((r.get("endTime", 0) for r in data.get("testResults", [])), default=None)
    if start and end:
        metrics.duration_seconds = round((end - start) / 1000, 2)
    logger.debug("vitest %s: %s", path, metrics)
    return metrics


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Build the metrics document from whichever artifacts exist."""
    result: dict[str, Any] = {"suites": [], "coverage": None, "missing": []}
    for label, path, parser in (
        ("junit", args.junit, lambda p: asdict(parse_junit(p))),
        ("vitest", args.vitest, lambda p: asdict(parse_vitest_json(p))),
    ):
        if path is None:
            continue
        p = Path(path)
        if p.is_file():
            result["suites"].append(parser(p))
        else:
            logger.warning("%s artifact not found: %s", label, p)
            result["missing"].append(str(p))
    if args.coverage is not None:
        p = Path(args.coverage)
        if p.is_file():
            result["coverage"] = asdict(parse_coverage_xml(p))
        else:
            logger.warning("coverage artifact not found: %s", p)
            result["missing"].append(str(p))
    return result


def render(doc: dict[str, Any]) -> str:
    """Render the metrics document as GitHub-flavoured markdown."""
    lines = [
        "## Test metrics",
        "",
        "| Suite | Collected | Passed | Failed | Errors | Skipped | Duration |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suite in doc.get("suites", []):
        duration = (
            f"{suite['duration_seconds']} s" if suite.get("duration_seconds") is not None else "n/a"
        )
        lines.append(
            f"| {suite['name']} | {suite['collected']} | {suite['passed']} | {suite['failed']} | "
            f"{suite['errors']} | {suite['skipped']} | {duration} |"
        )
    for suite in doc.get("suites", []):
        if suite.get("skip_reasons"):
            lines += ["", f"Skip reasons ({suite['name']}):", ""]
            lines += [f"- {reason}: {count}" for reason, count in suite["skip_reasons"].items()]
    coverage = doc.get("coverage")
    if coverage:
        lines += ["", "| Coverage | Line % | Branch % |", "|---|---:|---:|"]
        branch = coverage["branch_rate"] if coverage.get("branch_rate") is not None else "n/a"
        lines.append(f"| whole package | {coverage['line_rate']} | {branch} |")
        for tier, entry in coverage.get("packages", {}).items():
            lines.append(f"| {tier} | {entry['line_rate']} | {entry['branch_rate']} |")
    if doc.get("missing"):
        lines += ["", "Missing artifacts: " + ", ".join(doc["missing"])]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="parse artifacts into a metrics JSON document")
    c.add_argument("--junit", help="pytest --junitxml report")
    c.add_argument("--coverage", help="coverage.py XML report")
    c.add_argument("--vitest", help="vitest JSON reporter output")
    c.add_argument("--out", required=True, help="where to write metrics.json")
    r = sub.add_parser("render", help="render metrics.json as markdown")
    r.add_argument("metrics", help="metrics.json produced by collect")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    if args.command == "collect":
        doc = collect(args)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)
        return 0
    doc = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    sys.stdout.write(render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
