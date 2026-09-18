"""
Request-hardening helpers shared by every file-handling API endpoint.

Implements the fix-level items SEC-01 (filename sanitisation), SEC-02
(size-bounded streamed uploads with cleanup) and SEC-03 (global request
body cap) from the validation SDLC plan. Every limit is supplied by the
caller from ``APISettings`` so nothing here is hard-coded.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException
from starlette.responses import JSONResponse

from ptpd_calibration.core.logging import sanitize_log_text

if TYPE_CHECKING:
    from fastapi import UploadFile
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

_PATH_SEPARATORS: tuple[str, ...] = ("/", "\\")
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
# Conservative default for record identifiers that come from a URL: letters,
# digits, dash and underscore only. Callers with a known id format (a UUID,
# say) pass a stricter pattern.
RECORD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_KB = 1024


def _body_too_large_detail(max_bytes: int) -> str:
    return f"Request body exceeds maximum size of {max_bytes // _BYTES_PER_MB} MB"


class RequestBodyTooLarge(HTTPException):
    """Raised by :class:`RequestBodyLimitMiddleware` when a streamed body exceeds the cap.

    It is an ``HTTPException`` (413) on purpose: FastAPI wraps any other
    exception raised while it reads the request body into a generic 400, but
    re-raises ``HTTPException`` so the client sees the correct status.
    """

    def __init__(self, received: int, max_bytes: int) -> None:
        self.received = received
        self.max_bytes = max_bytes
        super().__init__(status_code=413, detail=_body_too_large_detail(max_bytes))


def mb_to_bytes(megabytes: int) -> int:
    """Convert a megabyte setting to bytes."""
    return megabytes * _BYTES_PER_MB


def kb_to_bytes(kilobytes: int) -> int:
    """Convert a kilobyte setting to bytes."""
    return kilobytes * _BYTES_PER_KB


def is_safe_basename(filename: str | None) -> bool:
    """Return True when ``filename`` is a plain basename with no traversal potential.

    Rejects empty names, NUL bytes, any path separator, and names that start
    with a dot (which covers ``.`` and ``..``). Browsers only ever send a
    basename, so anything else is either a bug or an attack.
    """
    if not filename or "\x00" in filename:
        return False
    if any(sep in filename for sep in _PATH_SEPARATORS):
        return False
    if filename.startswith("."):
        return False
    return Path(filename).name == filename


def safe_suffix(filename: str | None, allowed: Iterable[str]) -> str:
    """Validate a client-supplied filename and return its lower-cased extension.

    Args:
        filename: The name the client sent (``UploadFile.filename``).
        allowed: Allowlist of extensions including the leading dot.

    Returns:
        The lower-cased extension, guaranteed to be in ``allowed``.

    Raises:
        HTTPException: 400 when the name is unsafe or the extension is not allowed.
    """
    allowed_set = {ext.lower() for ext in allowed}
    if not is_safe_basename(filename):
        logger.debug("Rejected unsafe upload filename: %r", filename)
        raise HTTPException(
            status_code=400,
            detail="Invalid filename: must be a plain file name without path components",
        )
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in allowed_set:
        logger.debug(
            "Rejected upload with disallowed extension %r (allowed=%s)", suffix, allowed_set
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed_set))}",
        )
    return suffix


def server_upload_path(upload_dir: Path, suffix: str) -> Path:
    """Return a server-generated, collision-free path inside ``upload_dir``.

    The client never influences the path: only the validated ``suffix`` is used.
    """
    return upload_dir / f"{uuid4().hex}{suffix}"


def stored_record_path(
    directory: Path,
    identifier: str,
    suffix: str,
    *,
    pattern: re.Pattern[str] = RECORD_ID_PATTERN,
) -> Path | None:
    """Return the path of a stored record inside ``directory``, or None if unsafe.

    ``identifier`` reaches this function straight from a URL path parameter, so
    it is never trusted. Three independent checks apply, in order:

    1. it must match ``pattern`` in full, which admits no separator, dot or NUL
       and so cannot express traversal;
    2. it must still be a plain basename;
    3. the resolved path must sit inside ``directory``, which holds even if a
       future caller passes a looser pattern.

    Returning ``None`` rather than raising lets callers answer 404, which tells
    an attacker nothing about what does or does not exist on disk.

    Args:
        directory: Directory the record must live in.
        identifier: Untrusted record id from the request.
        suffix: Extension to append, including the leading dot.
        pattern: Full-match pattern the identifier must satisfy.

    Returns:
        The resolved path, or ``None`` when the identifier is not safe.
    """
    if not identifier or not pattern.fullmatch(identifier):
        logger.debug("Rejected record identifier not matching %s: %r", pattern.pattern, identifier)
        return None
    if not is_safe_basename(identifier):  # pragma: no cover - implied by the pattern
        logger.debug("Rejected unsafe record identifier: %r", identifier)
        return None

    root = directory.resolve()
    candidate = (root / f"{identifier}{suffix}").resolve()
    if not candidate.is_relative_to(root):  # pragma: no cover - implied by the pattern
        logger.warning("Rejected record identifier escaping %s: %r", root, identifier)
        return None
    return candidate


async def stream_upload_to_path(
    upload_file: UploadFile,
    dest: Path,
    max_bytes: int,
    chunk_size: int = 64 * _BYTES_PER_KB,
) -> int:
    """Stream an upload to ``dest`` in chunks, aborting once ``max_bytes`` is exceeded.

    A partially written file is removed before the error is raised so the
    upload directory never accumulates rejected data.

    Returns:
        Number of bytes written.

    Raises:
        HTTPException: 413 when the body exceeds ``max_bytes``; 500 on I/O failure.
    """
    bytes_written = 0
    try:
        with open(dest, "wb") as handle:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    handle.close()
                    unlink_quietly(dest)
                    logger.debug(
                        "Upload to %s aborted at %d bytes (cap %d)",
                        dest.name,
                        bytes_written,
                        max_bytes,
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {max_bytes // _BYTES_PER_MB} MB",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        unlink_quietly(dest)
        logger.warning("Failed to persist upload to %s: %s", dest, exc)
        raise HTTPException(status_code=500, detail="Failed to save upload") from None

    logger.debug("Streamed %d bytes to %s", bytes_written, dest.name)
    return bytes_written


def safe_export_name(name: str | None, *, default: str = "curve", max_length: int = 64) -> str:
    """Reduce a client-supplied display name to a safe download filename stem.

    Directory components are dropped, anything outside ``[A-Za-z0-9._-]`` is
    replaced with ``_``, leading/trailing punctuation is stripped and the result
    is truncated to ``max_length``. Falls back to ``default`` when nothing is left.
    The result is only ever used in a ``Content-Disposition`` header, never as a
    filesystem path.
    """
    base = Path(name or "").name
    cleaned = _UNSAFE_NAME_CHARS.sub("_", base).strip("._-")[:max_length].strip("._-")
    if not cleaned:
        logger.debug("Export name %r sanitised to default %r", name, default)
        return default
    if cleaned != name:
        logger.debug("Export name %r sanitised to %r", name, cleaned)
    return cleaned


def unlink_quietly(path: Path) -> None:
    """Remove ``path`` if it exists, swallowing filesystem errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove temporary file %s: %s", path, exc)


class RequestBodyLimitMiddleware:
    """Pure-ASGI middleware that rejects request bodies larger than ``max_bytes``.

    Requests declaring a ``Content-Length`` above the cap are answered with 413
    before the application sees them. Bodies streamed without a length header
    are counted as they arrive and aborted at the cap (SEC-03).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_content_length(scope)
        if declared is not None and declared > self.max_bytes:
            logger.debug(
                "Rejected %s %s: Content-Length %d exceeds cap %d",
                scope.get("method"),
                sanitize_log_text(scope.get("path")),
                declared,
                self.max_bytes,
            )
            await self._send_413(scope, receive, send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge(received, self.max_bytes)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except RequestBodyTooLarge as exc:
            logger.debug(
                "Aborted streamed body for %s %s at %d bytes (cap %d)",
                scope.get("method"),
                sanitize_log_text(scope.get("path")),
                exc.received,
                exc.max_bytes,
            )
            if response_started:
                raise
            await self._send_413(scope, receive, send)

    async def _send_413(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": _body_too_large_detail(self.max_bytes)},
            status_code=413,
        )
        await response(scope, receive, send)


def _declared_content_length(scope: Scope) -> int | None:
    """Return the integer ``Content-Length`` from an ASGI scope, or None."""
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value.decode("latin-1").strip())
            except ValueError:
                return None
    return None
