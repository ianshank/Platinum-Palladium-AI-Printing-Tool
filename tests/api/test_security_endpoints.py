"""
Security hardening endpoint tests (SEC-01, SEC-02, SEC-03, SEC-07).

Uses a dedicated application built from explicit ``Settings`` with small
limits so the negative paths are cheap to exercise, plus the shared default
``client`` fixture for module-level bounds and CORS defaults.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.api

MB = 1024 * 1024
SMALL_DENSITIES = [0.1, 0.4, 0.8, 1.2, 1.6, 2.0]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def secure_settings(tmp_path_factory: pytest.TempPathFactory):
    """Settings with tight limits and an isolated upload directory."""
    from ptpd_calibration.config import APISettings, Settings

    upload_dir = tmp_path_factory.mktemp("uploads")
    return Settings(
        api=APISettings(
            upload_dir=upload_dir,
            max_upload_size_mb=1,
            max_request_body_mb=2,
            max_list_length=16,
            max_string_length=32,
            max_export_name_length=24,
        )
    )


@pytest.fixture(scope="module")
def secure_client(secure_settings) -> Generator:
    from fastapi.testclient import TestClient

    from ptpd_calibration.api.server import create_app

    with TestClient(create_app(settings=secure_settings)) as client:
        yield client


@pytest.fixture
def upload_dir(secure_settings) -> Path:
    return secure_settings.api.upload_dir


def _quad_upload(content: bytes, filename: str) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (filename, content, "application/octet-stream")}


# =============================================================================
# SEC-01 / SEC-02: /api/curves/upload-quad
# =============================================================================


class TestUploadQuadHardening:
    def test_traversal_filename_rejected_and_sentinel_untouched(
        self, secure_client, upload_dir: Path, sample_quad_content: str
    ) -> None:
        # The pre-fix code wrote to upload_dir / file.filename and then unlinked it,
        # so a "../" name would have clobbered a file outside the upload directory.
        sentinel = upload_dir.parent / "sentinel.quad"
        sentinel.write_text("SENTINEL")

        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(sample_quad_content.encode(), "../sentinel.quad"),
            data={"channel": "K"},
        )

        assert response.status_code == 400
        assert "Invalid filename" in response.json()["detail"]
        assert sentinel.read_text() == "SENTINEL"
        assert list(upload_dir.iterdir()) == []

    @pytest.mark.parametrize("filename", ["..\\sentinel.quad", "sub/dir.quad", ".hidden.quad"])
    def test_other_unsafe_names_rejected(
        self, secure_client, filename: str, sample_quad_content: str
    ) -> None:
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(sample_quad_content.encode(), filename),
            data={"channel": "K"},
        )
        assert response.status_code == 400

    def test_disallowed_extension_rejected(self, secure_client, sample_quad_content: str) -> None:
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(sample_quad_content.encode(), "profile.exe"),
            data={"channel": "K"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Unsupported file type" in detail
        assert ".quad" in detail

    def test_upload_over_cap_returns_413_and_leaves_nothing(
        self, secure_client, upload_dir: Path
    ) -> None:
        oversized = b"0" * (MB + 1)  # over the 1 MB upload cap, under the 2 MB body cap
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(oversized, "big.quad"),
            data={"channel": "K"},
        )
        assert response.status_code == 413
        assert "maximum size" in response.json()["detail"]
        assert list(upload_dir.iterdir()) == []

    def test_valid_upload_round_trip(
        self, secure_client, upload_dir: Path, sample_quad_content: str
    ) -> None:
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(sample_quad_content.encode(), "test.quad"),
            data={"channel": "K"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "profile_name" in data
        assert "K" in data["active_channels"]
        assert data["curve_id"] is not None
        # server-generated temp file is cleaned up after parsing
        assert list(upload_dir.iterdir()) == []

    def test_txt_extension_is_allowed_by_default(
        self, secure_client, sample_quad_content: str
    ) -> None:
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(sample_quad_content.encode(), "profile.TXT"),
            data={"channel": "K"},
        )
        assert response.status_code == 200


# =============================================================================
# SEC-01 / SEC-02: /api/scan/upload (shares the same helpers)
# =============================================================================


class TestScanUploadHardening:
    def test_traversal_filename_rejected(
        self, secure_client, upload_dir: Path, sample_step_tablet_image: Path
    ) -> None:
        sentinel = upload_dir.parent / "sentinel.png"
        sentinel.write_bytes(b"SENTINEL")
        response = secure_client.post(
            "/api/scan/upload",
            files={"file": ("../sentinel.png", sample_step_tablet_image.read_bytes(), "image/png")},
            data={"tablet_type": "stouffer_21"},
        )
        assert response.status_code == 400
        assert sentinel.read_bytes() == b"SENTINEL"
        assert list(upload_dir.iterdir()) == []

    def test_disallowed_extension_rejected(self, secure_client) -> None:
        response = secure_client.post(
            "/api/scan/upload",
            files={"file": ("scan.txt", b"not an image", "text/plain")},
            data={"tablet_type": "stouffer_21"},
        )
        assert response.status_code == 400

    def test_scan_over_cap_returns_413(self, secure_client, upload_dir: Path) -> None:
        response = secure_client.post(
            "/api/scan/upload",
            files={"file": ("big.png", b"0" * (MB + 1), "image/png")},
            data={"tablet_type": "stouffer_21"},
        )
        assert response.status_code == 413
        assert list(upload_dir.iterdir()) == []

    def test_valid_scan_still_processes(
        self, secure_client, upload_dir: Path, sample_step_tablet_image: Path
    ) -> None:
        response = secure_client.post(
            "/api/scan/upload",
            files={"file": ("step_tablet.png", sample_step_tablet_image.read_bytes(), "image/png")},
            data={"tablet_type": "stouffer_21"},
        )
        assert response.status_code == 200
        assert response.json()["original_filename"] == "step_tablet.png"
        assert list(upload_dir.iterdir()) == []


# =============================================================================
# SEC-01: /api/curves/export and /api/curves/{id}/export
# =============================================================================


class TestCurveExportHardening:
    def test_client_name_only_reaches_content_disposition(
        self, secure_client, upload_dir: Path
    ) -> None:
        response = secure_client.post(
            "/api/curves/export",
            data={"densities": SMALL_DENSITIES, "name": "../../etc/passwd", "format": "csv"},
        )
        assert response.status_code == 200
        disposition = response.headers["content-disposition"]
        assert "passwd.csv" in disposition
        assert "../" not in disposition and "etc" not in disposition
        assert len(response.content) > 0
        # nothing was written outside the upload dir and the temp file is gone
        assert not (upload_dir.parent / "etc").exists()
        assert not (upload_dir / "passwd.csv").exists()
        assert list(upload_dir.iterdir()) == []

    def test_name_with_only_traversal_falls_back_to_default(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/export",
            data={"densities": SMALL_DENSITIES, "name": "..", "format": "json"},
        )
        assert response.status_code == 200
        assert "curve.json" in response.headers["content-disposition"]

    def test_invalid_format_is_422(self, secure_client, upload_dir: Path) -> None:
        response = secure_client.post(
            "/api/curves/export",
            data={"densities": SMALL_DENSITIES, "name": "curve", "format": "exe"},
        )
        assert response.status_code == 422
        assert "Unsupported export format" in response.json()["detail"]
        assert list(upload_dir.iterdir()) == []

    @pytest.mark.parametrize(
        ("fmt", "ext"),
        [("qtr", ".txt"), ("piezography", ".ppt"), ("csv", ".csv"), ("json", ".json")],
    )
    def test_supported_formats_round_trip(
        self, secure_client, upload_dir: Path, fmt: str, ext: str
    ) -> None:
        response = secure_client.post(
            "/api/curves/export",
            data={"densities": SMALL_DENSITIES, "name": "Round Trip", "format": fmt},
        )
        assert response.status_code == 200
        assert f"Round_Trip{ext}" in response.headers["content-disposition"]
        assert len(response.content) > 0
        assert list(upload_dir.iterdir()) == []

    def test_long_name_is_truncated(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/export",
            data={"densities": SMALL_DENSITIES, "name": "n" * 30, "format": "csv"},
        )
        assert response.status_code == 200
        assert ("n" * 24) + ".csv" in response.headers["content-disposition"]
        assert ("n" * 25) not in response.headers["content-disposition"]

    def test_stored_curve_export_validates_format_and_cleans_up(
        self, secure_client, upload_dir: Path
    ) -> None:
        create = secure_client.post(
            "/api/curves/modify",
            json={
                "input_values": [0.0, 0.5, 1.0],
                "output_values": [0.0, 0.5, 1.0],
                "name": "Stored ../x",
                "adjustment_type": "brightness",
                "amount": 0.0,
            },
        )
        assert create.status_code == 200
        curve_id = create.json()["curve_id"]

        bad = secure_client.post(f"/api/curves/{curve_id}/export?format=exe")
        assert bad.status_code == 422

        good = secure_client.post(f"/api/curves/{curve_id}/export?format=json")
        assert good.status_code == 200
        disposition = good.headers["content-disposition"]
        assert "../" not in disposition
        assert ".json" in disposition
        assert list(upload_dir.iterdir()) == []


# =============================================================================
# SEC-03: request body cap and field bounds
# =============================================================================


class TestRequestBodyCap:
    def test_declared_body_over_cap_is_413(self, secure_client) -> None:
        body = b"[" + b"0," * (2 * MB) + b"0]"  # > 2 MB with Content-Length set
        response = secure_client.post(
            "/api/analyze", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        assert "maximum size" in response.json()["detail"]

    def test_streamed_body_over_cap_is_413(self, secure_client) -> None:
        def chunks() -> Iterator[bytes]:
            for _ in range(40):  # 2.5 MB, sent chunked without Content-Length
                yield b"0" * 65536

        response = secure_client.post(
            "/api/analyze", content=chunks(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 413

    def test_multipart_over_cap_is_413(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/upload-quad",
            files=_quad_upload(b"0" * (3 * MB), "big.quad"),
            data={"channel": "K"},
        )
        assert response.status_code == 413

    def test_small_body_passes(self, secure_client) -> None:
        response = secure_client.post("/api/analyze", json={"densities": SMALL_DENSITIES})
        assert response.status_code == 200

    def test_413_response_carries_cors_headers(self, secure_client) -> None:
        body = b"[" + b"0," * (2 * MB) + b"0]"
        response = secure_client.post(
            "/api/analyze",
            content=body,
            headers={"content-type": "application/json", "origin": "https://app.example"},
        )
        assert response.status_code == 413
        assert response.headers.get("access-control-allow-origin") == "*"


class TestFieldBounds:
    def test_list_over_max_length_is_422(self, secure_client) -> None:
        response = secure_client.post("/api/analyze", json={"densities": [0.5] * 17})
        assert response.status_code == 422

    def test_list_at_max_length_is_accepted(self, secure_client) -> None:
        densities = [0.1 + 0.1 * i for i in range(16)]
        response = secure_client.post("/api/analyze", json={"densities": densities})
        assert response.status_code == 200

    def test_string_over_max_length_is_422(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/generate",
            json={"densities": SMALL_DENSITIES, "name": "n" * 33, "curve_type": "linear"},
        )
        assert response.status_code == 422

    def test_calibration_lists_and_strings_bounded(self, secure_client) -> None:
        base = {"paper_type": "Paper", "exposure_time": 120.0, "metal_ratio": 0.5}
        assert (
            secure_client.post(
                "/api/calibrations", json={**base, "densities": [0.1] * 17}
            ).status_code
            == 422
        )
        assert (
            secure_client.post("/api/calibrations", json={**base, "notes": "x" * 33}).status_code
            == 422
        )
        assert secure_client.post("/api/calibrations", json=base).status_code == 200

    def test_curve_modify_lists_bounded(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/modify",
            json={
                "input_values": [0.0] * 17,
                "output_values": [0.0] * 17,
                "adjustment_type": "brightness",
            },
        )
        assert response.status_code == 422

    def test_blend_lists_bounded(self, secure_client) -> None:
        response = secure_client.post(
            "/api/curves/blend",
            json={
                "curve1_inputs": [0.0] * 17,
                "curve1_outputs": [0.0] * 17,
                "curve2_inputs": [0.0] * 3,
                "curve2_outputs": [0.0] * 3,
            },
        )
        assert response.status_code == 422

    def test_form_string_bounded(self, secure_client, sample_quad_content: str) -> None:
        response = secure_client.post(
            "/api/curves/parse-quad",
            data={"content": sample_quad_content, "name": "n" * 33, "channel": "K"},
        )
        assert response.status_code == 422


class TestModuleLevelBounds:
    """The MCTS and deep-learning routers use APISettings defaults (4096)."""

    def test_mcts_search_target_curve_bounded(self, client) -> None:
        response = client.post("/api/mcts/search", json={"target_curve": [0.0] * 4097})
        assert response.status_code == 422

    def test_mcts_search_string_bounded(self, client) -> None:
        response = client.post("/api/mcts/search", json={"paper_type": "p" * 4097})
        assert response.status_code == 422

    def test_mcts_feedback_curve_bounded(self, client) -> None:
        payload = {"parameters": {}, "measured_curve": [0.0] * 4097, "quality_rating": 0.5}
        assert client.post("/api/mcts/feedback", json=payload).status_code == 422

    def test_mcts_feedback_empty_curve_rejected(self, client) -> None:
        payload = {"parameters": {}, "measured_curve": [], "quality_rating": 0.5}
        assert client.post("/api/mcts/feedback", json=payload).status_code == 422

    def test_mcts_recommendations_limit_bounded(self, client) -> None:
        assert client.get("/api/mcts/recommendations?limit=99999").status_code == 422
        assert client.get("/api/mcts/recommendations?limit=-1").status_code == 422
        assert client.get("/api/mcts/recommendations?limit=2").status_code == 200

    def test_mcts_export_parameters_bounded(self, client) -> None:
        params = {f"p{i}": 0.5 for i in range(4097)}
        assert client.post("/api/mcts/export?format=json", json=params).status_code == 422

    def test_deep_generate_synthetic_bounded(self, client) -> None:
        assert client.post("/api/deep/generate-synthetic?num_samples=999999").status_code == 422
        assert client.post("/api/deep/generate-synthetic?num_samples=0").status_code == 422

    def test_deep_train_hidden_dims_bounded(self, client) -> None:
        too_many = {"model_name": "m", "hidden_dims": [8] * 17}
        assert client.post("/api/deep/train", json=too_many).status_code == 422
        too_wide = {"model_name": "m", "hidden_dims": [99999]}
        assert client.post("/api/deep/train", json=too_wide).status_code == 422
        empty = {"model_name": "m", "hidden_dims": []}
        assert client.post("/api/deep/train", json=empty).status_code == 422

    def test_deep_suggest_target_curve_bounded(self, client) -> None:
        payload = {"paper_type": "p", "target_curve": [0.0] * 4097}
        assert client.post("/api/deep/suggest-adjustments", json=payload).status_code == 422


# =============================================================================
# SEC-07: CORS defaults
# =============================================================================


class TestCORSDefaults:
    def test_foreign_origin_preflight_gets_no_credentials_header(self, client) -> None:
        response = client.options(
            "/api/health",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-credentials") != "true"

    def test_foreign_origin_simple_request_gets_no_credentials_header(self, client) -> None:
        response = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-credentials") != "true"

    def test_explicit_origins_with_credentials(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from ptpd_calibration.api.server import create_app
        from ptpd_calibration.config import APISettings, Settings

        settings = Settings(
            api=APISettings(
                upload_dir=tmp_path / "uploads",
                cors_origins=["https://app.example"],
                cors_allow_credentials=True,
            )
        )
        with TestClient(create_app(settings=settings)) as strict_client:
            allowed = strict_client.options(
                "/api/health",
                headers={
                    "Origin": "https://app.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert allowed.headers.get("access-control-allow-origin") == "https://app.example"
            assert allowed.headers.get("access-control-allow-credentials") == "true"

            # Preflight from a foreign origin is rejected outright and never
            # names that origin, so the browser blocks the credentialed request.
            foreign_preflight = strict_client.options(
                "/api/health",
                headers={
                    "Origin": "https://evil.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert foreign_preflight.status_code == 400
            assert "access-control-allow-origin" not in foreign_preflight.headers

            # A simple request from a foreign origin is never granted an
            # Access-Control-Allow-Origin, which is what makes the browser block
            # it (Starlette emits allow-credentials unconditionally once enabled).
            foreign_simple = strict_client.get(
                "/api/health", headers={"Origin": "https://evil.example"}
            )
            assert foreign_simple.status_code == 200
            assert "access-control-allow-origin" not in foreign_simple.headers

    def test_wildcard_with_credentials_cannot_build_settings(self) -> None:
        from ptpd_calibration.config import APISettings

        with pytest.raises(ValueError, match="wildcard"):
            APISettings(cors_origins=["*"], cors_allow_credentials=True)


def test_scan_upload_rejects_oversized_image_before_decode(client, tmp_path):
    """A tiny file declaring a huge canvas is refused with 413 before any pixel decode (SEC-04)."""
    from PIL import Image

    huge = tmp_path / "huge.png"
    # A 1-bit 20000x20000 PNG compresses to a few kilobytes but would decode to 400 MP.
    Image.new("1", (20000, 20000)).save(huge, format="PNG", optimize=True)
    assert huge.stat().st_size < 200_000

    with open(huge, "rb") as fh:
        response = client.post("/api/scan/upload", files={"file": ("huge.png", fh, "image/png")})

    assert response.status_code == 413, response.text
    assert (
        "pixel" in response.json()["detail"].lower() or "large" in response.json()["detail"].lower()
    )
