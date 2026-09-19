"""Request-scoped logging context for the API.

An operator debugging a bad print run needs to tie log records to the request
that produced them. This middleware gives every request an identifier, binds it
to the logging context so each record carries it, and echoes it in a response
header so a user can quote it in a report.

It is pure ASGI rather than a ``BaseHTTPMiddleware`` subclass on purpose.
``BaseHTTPMiddleware`` catches exceptions raised while the body is read and
re-raises them in a context where FastAPI turns them into a generic 400, which
silently downgraded the 413 that :class:`~ptpd_calibration.api.security.
RequestBodyLimitMiddleware` raises for an oversized upload.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from ptpd_calibration.core.logging import LogContext, sanitize_log_text

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Header carrying the identifier, on the way in and on the way out.
REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")

#: What a request id may contain. Deliberately narrow: this covers a uuid, a
#: hex digest and the dotted or colon-separated trace ids tracing systems emit,
#: and excludes everything that could terminate a header or reach a terminal.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]+")

#: Used only when settings cannot be loaded; the real value is a settings field.
_FALLBACK_REQUEST_ID_LENGTH = 128


def _incoming_request_id(scope: Scope) -> str | None:
    """Return a caller-supplied request id, if it is safe to echo and to log.

    The value comes back to the caller as a response header and is bound to
    every log record for the request, so it is accepted only when it matches
    :data:`_SAFE_REQUEST_ID` in full. Bounding the length is not enough on its
    own: a value carrying CR or LF ends the header and starts another, and a
    control byte reaches a terminal reading the logs. A value that does not
    match is discarded rather than repaired, and the caller gets the generated
    id instead, because a correlation id the client did not send is more useful
    than a mangled version of one it did.
    """
    limit = _max_request_id_length()
    for name, value in scope.get("headers", ()):
        if name.lower() != _HEADER_BYTES:
            continue
        candidate = value.decode("latin-1").strip()
        if not candidate or len(candidate) > limit:
            logger.debug("Ignoring request id of length %d (limit %d)", len(candidate), limit)
            return None
        if not _SAFE_REQUEST_ID.fullmatch(candidate):
            logger.debug("Ignoring request id with unsafe characters: %r", candidate)
            return None
        return candidate
    return None


def _max_request_id_length() -> int:
    """Read the bound from settings, tolerating an unloadable config."""
    try:
        from ptpd_calibration.config import get_settings

        return int(get_settings().api.max_request_id_length)
    except Exception:  # pragma: no cover - configuration is broken, still serve
        logger.debug("Falling back to the built-in request id limit", exc_info=True)
        return _FALLBACK_REQUEST_ID_LENGTH


class RequestContextMiddleware:
    """Bind a request id to the logging context and echo it to the caller."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid4().hex

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((_HEADER_BYTES, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        with LogContext(request_id=request_id):
            # The path arrives percent-decoded, so an encoded newline in the
            # URL reached the log verbatim.
            logger.debug("%s %s", scope.get("method"), sanitize_log_text(scope.get("path")))
            await self.app(scope, receive, send_with_request_id)
