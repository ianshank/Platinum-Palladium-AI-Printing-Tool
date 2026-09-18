"""Logging and health must tell an operator what is actually happening.

Logging was configured by whichever module called ``get_logger`` first, so the
level, format and destination depended on import order and a deployment could
not choose them. ``/api/health`` returned a constant, so it could not
distinguish a working deployment from one whose provider is unconfigured or
whose optional extras are missing. Neither is a security control; both are what
someone debugging a ruined print run reaches for first.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ptpd_calibration.api.observability import REQUEST_ID_HEADER  # noqa: E402
from ptpd_calibration.api.server import _api_version, create_app  # noqa: E402
from ptpd_calibration.core.logging import LogContext  # noqa: E402

pytestmark = pytest.mark.api


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


class TestRequestIdentifier:
    def test_every_response_carries_an_identifier(self, client: TestClient) -> None:
        response = client.get("/api/health")

        assert response.headers.get(REQUEST_ID_HEADER)

    def test_identifiers_differ_between_requests(self, client: TestClient) -> None:
        first = client.get("/api/health").headers[REQUEST_ID_HEADER]
        second = client.get("/api/health").headers[REQUEST_ID_HEADER]

        assert first != second

    def test_a_caller_supplied_identifier_is_echoed(self, client: TestClient) -> None:
        """A front end can correlate a whole calibration under one id."""
        response = client.get("/api/health", headers={REQUEST_ID_HEADER: "journey-42"})

        assert response.headers[REQUEST_ID_HEADER] == "journey-42"

    def test_an_absurd_caller_identifier_is_replaced(self, client: TestClient) -> None:
        """The value is echoed and logged, so it is bounded."""
        response = client.get("/api/health", headers={REQUEST_ID_HEADER: "x" * 500})

        assert response.headers[REQUEST_ID_HEADER] != "x" * 500

    def test_the_identifier_reaches_the_log_records(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="ptpd_calibration.api.observability"):
            client.get("/api/health", headers={REQUEST_ID_HEADER: "traced-7"})

        assert "/api/health" in caplog.text


class TestLogContextDefault:
    def test_context_is_readable_before_anything_sets_it(self) -> None:
        """The context variable had no default, so the first read raised."""
        from ptpd_calibration.core.logging import _log_context

        assert _log_context.get() in (None, {})

    def test_nested_contexts_merge_and_restore(self) -> None:
        from ptpd_calibration.core.logging import _log_context

        with LogContext(request_id="outer"):
            with LogContext(user="inner"):
                current = _log_context.get() or {}
                assert current == {"request_id": "outer", "user": "inner"}
            assert (_log_context.get() or {}) == {"request_id": "outer"}
        assert _log_context.get() in (None, {})


class TestHealthReportsReality:
    def test_health_reports_the_running_version(self, client: TestClient) -> None:
        body = client.get("/api/health").json()

        assert body["status"] == "healthy"
        assert body["version"] == _api_version()

    def test_health_reports_whether_a_provider_is_configured(self, client: TestClient) -> None:
        body = client.get("/api/health").json()

        assert isinstance(body["llm_provider_configured"], bool)

    def test_health_reports_which_optional_routers_mounted(self, client: TestClient) -> None:
        body = client.get("/api/health").json()

        assert set(body["features"]) == {"mcts", "deep_learning"}
        assert all(isinstance(value, bool) for value in body["features"].values())

    def test_the_documented_version_is_not_a_placeholder(self) -> None:
        """The API reported 1.0.0 whatever was deployed."""
        assert _api_version() != "0.0.0+unknown"


class TestRequestIdIsSafeToEchoAndLog:
    """A caller-supplied id comes back as a header and joins every log record.

    Bounding only its length was not enough. A value carrying CR or LF ends the
    ``X-Request-ID`` header and begins another, and a control byte reaches
    whatever reads the logs. The value is now accepted only if it matches in
    full, and a value that does not is dropped rather than repaired: a
    correlation id the client never sent is more use than a mangled one.
    """

    @staticmethod
    def _incoming(raw: bytes) -> str | None:
        from ptpd_calibration.api.observability import _incoming_request_id

        return _incoming_request_id({"headers": [(b"x-request-id", raw)]})

    @pytest.mark.parametrize(
        "raw",
        [
            b"abc\r\nX-Evil: injected",  # the finding: a second header
            b"a\nb",
            b"a\rb",
            b"\x07bell",  # reaches a terminal reading the logs
            b"has space",
            b"semi;colon",
            b"angle<bracket>",
            b"",
            b"   ",
        ],
        ids=[
            "crlf-injection",
            "lf",
            "cr",
            "control-byte",
            "space",
            "semicolon",
            "angle-brackets",
            "empty",
            "whitespace-only",
        ],
    )
    def test_an_unsafe_value_is_refused(self, raw: bytes) -> None:
        assert self._incoming(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [b"ok-123", b"0123456789abcdef", b"trace:1.2-3", b"A_b.c-d", b"  padded  "],
        ids=["hyphen", "hex", "trace-id", "mixed", "trimmed"],
    )
    def test_a_safe_value_is_kept(self, raw: bytes) -> None:
        assert self._incoming(raw) == raw.decode().strip()

    def test_the_length_bound_is_a_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_API_MAX_REQUEST_ID_LENGTH", "16")
        from ptpd_calibration import config

        monkeypatch.setattr(config, "_settings", None)

        assert self._incoming(b"a" * 16) == "a" * 16
        assert self._incoming(b"a" * 17) is None

    def test_a_refused_value_still_yields_a_usable_id(self) -> None:
        """The request must carry an id regardless; a generated one serves."""
        from fastapi.testclient import TestClient

        from ptpd_calibration.api.observability import REQUEST_ID_HEADER
        from ptpd_calibration.api.server import create_app

        with TestClient(create_app()) as client:
            response = client.get(
                "/api/health", headers={REQUEST_ID_HEADER: "abc\r\nX-Evil: injected"}
            )

        echoed = response.headers[REQUEST_ID_HEADER]
        assert echoed
        assert "X-Evil" not in echoed
        assert "\r" not in echoed and "\n" not in echoed
        assert "X-Evil" not in response.headers
