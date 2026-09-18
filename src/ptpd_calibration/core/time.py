"""UTC time helpers.

``datetime.datetime.utcnow()`` is deprecated from Python 3.12 and scheduled for
removal, because it returns a naive datetime that silently claims to be local
time. These helpers give the two shapes the code base actually needs:

* :func:`utc_now` — an aware datetime in UTC, for arithmetic and for model
  fields;
* :func:`utc_timestamp` — the ISO 8601 string with a trailing ``Z`` that the
  agent log and metric payloads have always emitted, so consumers parsing those
  records keep working unchanged.

Both are module-level functions rather than lambdas so they can be used
directly as a pydantic ``default_factory``.
"""

from __future__ import annotations

from datetime import datetime, timezone

# ``datetime.UTC`` is a 3.11 alias; the package floor is 3.10, so use the
# spelling that works on every supported version.
UTC = timezone.utc

# Suffix used by the JSON payloads; ``datetime.isoformat`` writes "+00:00" for
# an aware UTC datetime, which is equivalent but not what consumers expect.
UTC_SUFFIX = "Z"
_ISO_UTC_OFFSET = "+00:00"


def utc_now() -> datetime:
    """Return the current time as an aware datetime in UTC."""
    return datetime.now(UTC)


def utc_timestamp() -> str:
    """Return the current UTC time as ISO 8601 with a trailing ``Z``.

    Equivalent to the old ``datetime.utcnow().isoformat() + "Z"`` output, to
    the microsecond, but without the deprecated call.
    """
    return utc_now().isoformat().replace(_ISO_UTC_OFFSET, UTC_SUFFIX)
