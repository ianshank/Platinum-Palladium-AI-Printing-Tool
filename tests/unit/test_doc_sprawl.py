"""Tests for scripts/check-doc-sprawl.sh (AGENTS.md placement rules, ADR-0008)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-doc-sprawl.sh"
CANONICAL = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "CLAUDE.md",
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def _run(root: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), str(root)], capture_output=True, text=True, env=merged, check=False
    )


def _canonical_repo(root: Path) -> Path:
    for name in CANONICAL:
        (root / name).write_text(f"# {name}\n")
    (root / "docs" / "adr").mkdir(parents=True)
    (root / "docs" / "adr" / "0001-example.md").write_text("# ADR\n")
    return root


def test_canonical_layout_passes(tmp_path: Path) -> None:
    result = _run(_canonical_repo(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "doc-sprawl: ok" in result.stdout


@pytest.mark.parametrize(
    "name", ["NEXT_STEPS.md", "plan.md", "IMPLEMENTATION_SUMMARY.md", "notes.md"]
)
def test_extra_root_markdown_fails(tmp_path: Path, name: str) -> None:
    root = _canonical_repo(tmp_path)
    (root / name).write_text("stray\n")
    result = _run(root)
    assert result.returncode == 1
    assert name in result.stdout


@pytest.mark.parametrize(
    "relative",
    [
        "src/pkg/AGENT.md",
        "docs/QUICK_REFERENCE.md",
        "src/AGENTIC_NEXT_STEPS_V3.md",
        "tests/INVESTIGATION_SUMMARY.md",
    ],
)
def test_forbidden_patterns_anywhere_fail(tmp_path: Path, relative: str) -> None:
    root = _canonical_repo(tmp_path)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("stray\n")
    result = _run(root)
    assert result.returncode == 1
    assert relative in result.stdout


def test_archive_and_plans_are_exempt(tmp_path: Path) -> None:
    root = _canonical_repo(tmp_path)
    for relative in (
        "docs/archive/2026-02-NEXT_STEPS.md",
        "docs/plans/2026-09-review/IMPLEMENTATION_SUMMARY.md",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("history\n")
    result = _run(root)
    assert result.returncode == 0, result.stdout


def test_allowlist_is_configurable(tmp_path: Path) -> None:
    root = _canonical_repo(tmp_path)
    (root / "HANDBOOK.md").write_text("allowed here\n")
    assert _run(root).returncode == 1
    env = {"DOC_SPRAWL_ROOT_ALLOWLIST": " ".join((*CANONICAL, "HANDBOOK.md"))}
    assert _run(root, env).returncode == 0


def test_repository_itself_complies() -> None:
    """The live repository must satisfy its own rules."""
    result = _run(SCRIPT.parents[1])
    assert result.returncode == 0, result.stdout
