"""Tests for the UTC time helpers and the ban on ``datetime.utcnow``."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ptpd_calibration.core.time import UTC_SUFFIX, utc_now, utc_timestamp

pytestmark = pytest.mark.unit

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "ptpd_calibration"
UTCNOW_CALL = re.compile(r"\butcnow\s*\(")
# The helper module explains the deprecation in prose, so it is allowed to name it.
DOCUMENTING_MODULE = SOURCE_ROOT / "core" / "time.py"
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


class TestUtcNow:
    def test_is_timezone_aware_utc(self) -> None:
        now = utc_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_tracks_wall_clock(self) -> None:
        assert abs(utc_now() - datetime.now(UTC)) < timedelta(seconds=5)

    def test_usable_as_default_factory(self) -> None:
        """pydantic calls the factory with no arguments."""
        assert isinstance(utc_now(), datetime)


class TestUtcTimestamp:
    def test_iso8601_with_z_suffix(self) -> None:
        stamp = utc_timestamp()
        assert stamp.endswith(UTC_SUFFIX)
        assert ISO_Z.match(stamp), stamp
        assert "+00:00" not in stamp

    def test_round_trips_through_fromisoformat(self) -> None:
        stamp = utc_timestamp()
        parsed = datetime.fromisoformat(stamp.replace(UTC_SUFFIX, "+00:00"))
        assert parsed.utcoffset() == timedelta(0)
        assert abs(parsed - datetime.now(UTC)) < timedelta(seconds=5)

    def test_matches_the_legacy_wire_format(self) -> None:
        """The old payloads were ``datetime.utcnow().isoformat() + "Z"``."""
        legacy = datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"
        assert ISO_Z.match(legacy)
        assert len(utc_timestamp()) == len(legacy)


def test_no_module_calls_datetime_utcnow() -> None:
    """``datetime.utcnow()`` warns on Python 3.12+ and is removed later.

    Warnings are errors in this suite, so a reintroduced call would fail some
    unrelated test with a confusing message; this test names the real problem.
    """
    offenders = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*.py")
        if path != DOCUMENTING_MODULE and UTCNOW_CALL.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "use ptpd_calibration.core.time.utc_now()/utc_timestamp() instead of "
        f"datetime.utcnow() in: {', '.join(offenders)}"
    )
