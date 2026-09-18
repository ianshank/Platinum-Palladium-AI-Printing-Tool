"""
Unit tests for the shared request-hardening helpers in ``ptpd_calibration.api.security``.

Covers SEC-01 (filename sanitisation), SEC-02 (bounded streamed uploads with
cleanup) and SEC-03 (request body cap middleware) at the helper level; the
endpoint-level behaviour lives in ``tests/api/test_security_endpoints.py``.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from ptpd_calibration.api.security import (  # noqa: E402
    RequestBodyLimitMiddleware,
    RequestBodyTooLarge,
    is_safe_basename,
    kb_to_bytes,
    mb_to_bytes,
    safe_export_name,
    safe_suffix,
    server_upload_path,
    stream_upload_to_path,
    unlink_quietly,
)

ALLOWED = (".quad", ".txt")


# =============================================================================
# Filename sanitisation (SEC-01)
# =============================================================================


class TestIsSafeBasename:
    @pytest.mark.parametrize(
        "filename",
        ["scan.png", "profile.quad", "my..scan.png", "UPPER.TIF", "a-b_c.d.txt"],
    )
    def test_plain_basenames_are_safe(self, filename: str) -> None:
        assert is_safe_basename(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "../scan.png",
            "..\\scan.png",
            "dir/scan.png",
            "dir\\scan.png",
            "/etc/passwd",
            ".hidden.png",
            "..",
            ".",
            "",
            None,
            "sc\x00an.png",
        ],
    )
    def test_traversal_and_hidden_names_are_unsafe(self, filename: str | None) -> None:
        assert is_safe_basename(filename) is False


class TestSafeSuffix:
    def test_returns_lowercased_allowed_extension(self) -> None:
        assert safe_suffix("PROFILE.QUAD", ALLOWED) == ".quad"
        assert safe_suffix("notes.txt", ALLOWED) == ".txt"

    def test_allowlist_is_case_insensitive(self) -> None:
        assert safe_suffix("profile.quad", (".QUAD",)) == ".quad"

    @pytest.mark.parametrize("filename", ["../profile.quad", "..\\profile.quad", "a/b.quad"])
    def test_traversal_filename_is_400(self, filename: str) -> None:
        with pytest.raises(HTTPException) as excinfo:
            safe_suffix(filename, ALLOWED)
        assert excinfo.value.status_code == 400
        assert "Invalid filename" in excinfo.value.detail

    @pytest.mark.parametrize("filename", ["profile.exe", "profile", "profile.quad.sh"])
    def test_disallowed_extension_is_400(self, filename: str) -> None:
        with pytest.raises(HTTPException) as excinfo:
            safe_suffix(filename, ALLOWED)
        assert excinfo.value.status_code == 400
        assert "Unsupported file type" in excinfo.value.detail
        assert ".quad" in excinfo.value.detail

    def test_missing_filename_is_400(self) -> None:
        with pytest.raises(HTTPException) as excinfo:
            safe_suffix(None, ALLOWED)
        assert excinfo.value.status_code == 400


class TestServerUploadPath:
    def test_path_is_inside_upload_dir_with_suffix(self, tmp_path: Path) -> None:
        path = server_upload_path(tmp_path, ".quad")
        assert path.parent == tmp_path
        assert path.suffix == ".quad"
        assert len(path.stem) == 32  # uuid4().hex

    def test_paths_are_unique(self, tmp_path: Path) -> None:
        assert server_upload_path(tmp_path, ".png") != server_upload_path(tmp_path, ".png")


class TestSafeExportName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("curve", "curve"),
            ("My Curve", "My_Curve"),
            ("../../etc/passwd", "passwd"),
            ("..\\..\\boot.ini", "boot.ini"),
            ("my curve (v2)", "my_curve_v2"),
            ("..", "curve"),
            ("...", "curve"),
            ("", "curve"),
            (None, "curve"),
            ("émile", "mile"),
            ("a.b-c_d", "a.b-c_d"),
        ],
    )
    def test_sanitises_to_safe_stem(self, raw: str | None, expected: str) -> None:
        assert safe_export_name(raw) == expected

    def test_custom_default(self) -> None:
        assert safe_export_name("///", default="export") == "export"

    def test_truncates_to_max_length(self) -> None:
        assert safe_export_name("a" * 100, max_length=16) == "a" * 16

    def test_truncation_does_not_leave_trailing_punctuation(self) -> None:
        assert safe_export_name("abc-" + "d" * 40, max_length=4) == "abc"


# =============================================================================
# Bounded streamed uploads (SEC-02)
# =============================================================================


def _upload(data: bytes, filename: str = "blob.bin") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename, size=len(data))


class TestStreamUploadToPath:
    async def test_writes_body_under_cap(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        data = b"0123456789" * 100
        written = await stream_upload_to_path(_upload(data), dest, max_bytes=2000, chunk_size=64)
        assert written == len(data)
        assert dest.read_bytes() == data

    async def test_exactly_at_cap_is_accepted(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        data = b"x" * 512
        assert (
            await stream_upload_to_path(_upload(data), dest, max_bytes=512, chunk_size=100) == 512
        )

    async def test_over_cap_is_413_and_partial_file_removed(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        data = b"x" * 513
        with pytest.raises(HTTPException) as excinfo:
            await stream_upload_to_path(_upload(data), dest, max_bytes=512, chunk_size=100)
        assert excinfo.value.status_code == 413
        assert "maximum size" in excinfo.value.detail
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    async def test_unwritable_destination_is_500(self, tmp_path: Path) -> None:
        dest = tmp_path / "missing-dir" / "out.bin"
        with pytest.raises(HTTPException) as excinfo:
            await stream_upload_to_path(_upload(b"abc"), dest, max_bytes=10)
        assert excinfo.value.status_code == 500


class TestUnlinkQuietly:
    def test_removes_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("x")
        unlink_quietly(target)
        assert not target.exists()

    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        unlink_quietly(tmp_path / "never-existed.txt")


class TestUnitConversions:
    def test_mb_and_kb(self) -> None:
        assert mb_to_bytes(1) == 1_048_576
        assert mb_to_bytes(50) == 50 * 1_048_576
        assert kb_to_bytes(64) == 65_536


# =============================================================================
# Request body cap middleware (SEC-03)
# =============================================================================


async def _echo_length_app(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
    """Minimal ASGI app that reads the whole body and answers with its length."""
    if scope["type"] != "http":
        # lifespan etc.
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": str(len(body)).encode()})


@pytest.fixture
def limited_client() -> TestClient:
    return TestClient(RequestBodyLimitMiddleware(_echo_length_app, max_bytes=100))


class TestRequestBodyLimitMiddleware:
    def test_body_under_cap_passes_through(self, limited_client: TestClient) -> None:
        response = limited_client.post("/", content=b"x" * 100)
        assert response.status_code == 200
        assert response.text == "100"

    def test_declared_content_length_over_cap_is_413(self, limited_client: TestClient) -> None:
        response = limited_client.post("/", content=b"x" * 101)
        assert response.status_code == 413
        assert "maximum size" in response.json()["detail"]

    def test_streamed_body_without_content_length_is_capped(
        self, limited_client: TestClient
    ) -> None:
        def chunks():  # type: ignore[no-untyped-def]
            for _ in range(5):
                yield b"x" * 40  # 200 bytes total, no Content-Length header

        response = limited_client.post("/", content=chunks())
        assert response.status_code == 413

    def test_streamed_body_under_cap_passes(self, limited_client: TestClient) -> None:
        def chunks():  # type: ignore[no-untyped-def]
            yield b"x" * 40
            yield b"x" * 40

        response = limited_client.post("/", content=chunks())
        assert response.status_code == 200
        assert response.text == "80"

    def test_get_without_body_passes(self, limited_client: TestClient) -> None:
        assert limited_client.get("/").status_code == 200

    def test_garbage_content_length_is_ignored_by_precheck(
        self, limited_client: TestClient
    ) -> None:
        # The pre-check must not crash on a malformed header; the counted path still applies.
        response = limited_client.post("/", content=b"x" * 10, headers={"content-length": "abc"})
        assert response.status_code in (200, 400, 413)


class TestRequestBodyTooLarge:
    def test_is_a_413_http_exception(self) -> None:
        exc = RequestBodyTooLarge(received=200, max_bytes=100)
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 413
        assert exc.received == 200
        assert exc.max_bytes == 100
