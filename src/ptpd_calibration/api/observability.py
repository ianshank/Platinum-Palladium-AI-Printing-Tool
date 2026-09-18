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
from typing import TYPE_CHECKING
from uuid import uuid4

from ptpd_calibration.core.logging import LogContext

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Header carrying the identifier, on the way in and on the way out.
REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")


def _incoming_request_id(scope: Scope) -> str | None:
    """Return a caller-supplied request id, so a client can correlate a journey."""
    for name, value in scope.get("headers", ()):
        if name.lower() == _HEADER_BYTES:
            candidate = value.decode("latin-1").strip()
            # Bound it: the value is echoed back and written to every record.
            if candidate and len(candidate) <= 128:
                return candidate
    return None


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
            logger.debug("%s %s", scope.get("method"), scope.get("path"))
            await self.app(scope, receive, send_with_request_id)
