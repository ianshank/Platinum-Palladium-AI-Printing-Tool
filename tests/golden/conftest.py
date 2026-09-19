"""
Shared helpers for golden-file tests.

A golden test computes a JSON-serialisable snapshot and compares it with the
committed fixture next to the test module. Comparison rules (SCI-05):

* ``int`` and ``bool`` values must match exactly,
* ``float`` values must agree to ``FLOAT_ABS_TOL`` (1e-9),
* strings, ``None``, dict keys and list lengths must match exactly.

Regenerating fixtures
---------------------
Set ``PTPD_UPDATE_GOLDENS=1`` and run the golden suite; every fixture the run
touches is rewritten (floats rounded to ``FLOAT_DECIMALS`` places) and the test
passes. Review the resulting diff before committing it::

    PTPD_UPDATE_GOLDENS=1 .venv/bin/python -m pytest tests/golden -q -o addopts=""
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

UPDATE_ENV_VAR = "PTPD_UPDATE_GOLDENS"
FLOAT_DECIMALS = 9
FLOAT_ABS_TOL = 1e-9
_TRUTHY = {"1", "true", "yes", "on"}


def update_goldens_requested() -> bool:
    """True when the environment asks for fixtures to be rewritten."""
    return os.environ.get(UPDATE_ENV_VAR, "").strip().lower() in _TRUTHY


def round_floats(value: Any, decimals: int = FLOAT_DECIMALS) -> Any:
    """Recursively round floats (ints, bools and strings are untouched)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {str(k): round_floats(v, decimals) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v, decimals) for v in value]
    return value


def assert_golden_equal(
    actual: Any, expected: Any, path: str = "$", abs_tol: float = FLOAT_ABS_TOL
) -> None:
    """Recursively compare a snapshot with its golden fixture.

    Raises:
        AssertionError: naming the first JSON path that differs.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected object, got {type(actual).__name__}"
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        assert not missing and not extra, f"{path}: key mismatch missing={missing} extra={extra}"
        for key in expected:
            assert_golden_equal(actual[key], expected[key], f"{path}.{key}", abs_tol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_golden_equal(a, e, f"{path}[{index}]", abs_tol)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
        return
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), (
            f"{path}: expected number, got {actual!r} vs {expected!r}"
        )
        assert abs(actual - expected) <= abs_tol, (
            f"{path}: {actual!r} != {expected!r} (|diff|={abs(actual - expected):.3e} > {abs_tol})"
        )
        return
    if isinstance(expected, int):
        assert isinstance(actual, int) and actual == expected, (
            f"{path}: int {actual!r} != {expected!r} (ints must match exactly)"
        )
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


class GoldenFile:
    """One fixture file: ``check`` compares, or rewrites when updating."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Any:
        return json.loads(self.path.read_text())

    def write(self, snapshot: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(round_floats(snapshot), indent=2, sort_keys=True) + "\n")
        logger.info("Wrote golden fixture %s", self.path)

    def check(self, snapshot: Any) -> None:
        """Compare ``snapshot`` with the fixture, or rewrite it when updating."""
        rounded = round_floats(snapshot)
        if update_goldens_requested():
            self.write(rounded)
            # Guard against non-JSON-serialisable snapshots producing a fixture
            # that could never match on the next run.
            assert_golden_equal(rounded, self.load(), path=f"$ ({self.path.name} after write)")
            return
        if not self.path.exists():
            pytest.fail(
                f"Golden fixture {self.path} is missing. Generate it with "
                f"{UPDATE_ENV_VAR}=1 and commit the file."
            )
        assert_golden_equal(rounded, self.load(), path=f"$ ({self.path.name})")


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> Callable[[str], GoldenFile]:
    """Factory for fixtures stored next to the requesting test module."""
    directory = Path(request.path).parent

    def _factory(name: str) -> GoldenFile:
        return GoldenFile(directory / name)

    return _factory
