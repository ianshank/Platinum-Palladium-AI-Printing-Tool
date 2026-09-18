"""The digital-negative export endpoint (plan item ARC-07).

The React app had no way to produce a negative, which is the artefact the whole
toolkit exists to make; only the frozen Gradio UI could, and ADR-0004 cannot
retire it until that gap closes. These tests cover the contract the client will
generate against and the bit-depth guarantee of ADR-0016, so a 16-bit request
that quietly returns 8 bits fails here rather than in someone's print.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")

pytestmark = pytest.mark.api

ENDPOINT = "/api/export/negative"
SIDE = 48
SCALE_8_TO_16 = 257
EIGHT_BIT_LEVELS = 256
SIXTEEN_BIT_MAX = 65535
DENSITIES = [0.08, 0.22, 0.41, 0.63, 0.88, 1.10, 1.28, 1.41, 1.48, 1.52]


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _eight_bit_source() -> bytes:
    ramp = np.linspace(0, 255, SIDE * SIDE).astype(np.uint8).reshape(SIDE, SIDE)
    return _png_bytes(ramp)


def _sixteen_bit_source() -> bytes:
    ramp = np.linspace(0, SIXTEEN_BIT_MAX, SIDE * SIDE).astype(np.uint16).reshape(SIDE, SIDE)
    return _png_bytes(ramp)


def _rgb_source() -> bytes:
    ramp = np.linspace(0, 255, SIDE * SIDE).astype(np.uint8).reshape(SIDE, SIDE)
    return _png_bytes(np.stack([ramp, ramp, ramp], axis=-1))


def _upload(payload: bytes, filename: str = "scan.png") -> dict:
    return {"file": (filename, payload, "image/png")}


def _opened(response) -> tuple[str, np.ndarray]:
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        return image.mode, np.asarray(image)


class TestNegativeExportContract:
    """Format, curve source and colour mode are all caller decisions."""

    def test_a_negative_is_returned_for_a_plain_upload(self, client) -> None:
        response = client.post(ENDPOINT, files=_upload(_eight_bit_source()))

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"
        assert "negative.tiff" in response.headers["content-disposition"]

    def test_inversion_is_what_makes_it_a_negative(self, client) -> None:
        source = np.full((8, 8), 200, dtype=np.uint8)

        inverted = client.post(
            ENDPOINT,
            files=_upload(_png_bytes(source)),
            data={"format": "png", "invert": "true"},
        )
        straight = client.post(
            ENDPOINT,
            files=_upload(_png_bytes(source)),
            data={"format": "png", "invert": "false"},
        )

        _, dark = _opened(inverted)
        _, light = _opened(straight)
        assert int(dark.mean()) < int(light.mean())
        assert int(light.mean()) == 200

    def test_a_stored_curve_can_be_named_by_id(self, client) -> None:
        generated = client.post(
            "/api/curves/generate", json={"densities": DENSITIES, "curve_type": "linear"}
        )
        assert generated.status_code == 200
        curve_id = generated.json()["curve_id"]

        response = client.post(
            ENDPOINT,
            files=_upload(_eight_bit_source()),
            data={"curve_id": curve_id, "format": "png"},
        )

        assert response.status_code == 200
        mode, data = _opened(response)
        assert mode == "L"
        assert np.unique(data).size > 1

    def test_densities_generate_a_curve_on_the_spot(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_eight_bit_source()),
            data={"densities": [str(d) for d in DENSITIES], "format": "png"},
        )

        assert response.status_code == 200
        assert _opened(response)[0] == "L"

    def test_a_download_name_is_sanitised(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_eight_bit_source()),
            data={"name": "../../etc/passwd", "format": "png"},
        )

        assert response.status_code == 200
        disposition = response.headers["content-disposition"]
        assert "/" not in disposition.split("filename=")[-1]
        assert ".." not in disposition


class TestNegativeExportDepth:
    """ADR-0016: the requested format decides the depth, and it is honoured."""

    def test_a_sixteen_bit_request_returns_a_sixteen_bit_file(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_sixteen_bit_source()),
            data={"densities": [str(d) for d in DENSITIES], "format": "tiff_16bit"},
        )

        assert response.status_code == 200
        mode, data = _opened(response)
        assert mode == "I;16"
        assert data.dtype == np.uint16
        assert np.unique(data).size > EIGHT_BIT_LEVELS
        assert not np.all(data % SCALE_8_TO_16 == 0), "an 8-bit image widened after the fact"

    def test_an_eight_bit_request_returns_an_eight_bit_file(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_sixteen_bit_source()),
            data={"format": "png"},
        )

        assert response.status_code == 200
        mode, data = _opened(response)
        assert mode == "L"
        # Clipping instead of scaling would have blown the frame to white.
        assert float((data == 255).mean()) < 0.1

    @pytest.mark.parametrize(
        ("fmt", "media_type"),
        [
            ("tiff", "image/tiff"),
            ("tiff_16bit", "image/tiff"),
            ("png", "image/png"),
            ("png_16bit", "image/png"),
            ("jpeg", "image/jpeg"),
            ("jpeg_high", "image/jpeg"),
        ],
    )
    def test_every_offered_format_round_trips(self, client, fmt: str, media_type: str) -> None:
        response = client.post(ENDPOINT, files=_upload(_sixteen_bit_source()), data={"format": fmt})

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == media_type
        assert len(response.content) > 0


class TestNegativeExportRejections:
    """Bad input is refused with the status that says why."""

    def test_an_unknown_format_is_rejected(self, client) -> None:
        response = client.post(ENDPOINT, files=_upload(_eight_bit_source()), data={"format": "bmp"})

        assert response.status_code == 422
        assert "bmp" in response.json()["detail"]

    def test_an_unknown_color_mode_is_rejected(self, client) -> None:
        response = client.post(
            ENDPOINT, files=_upload(_eight_bit_source()), data={"color_mode": "cmyk"}
        )

        assert response.status_code == 422
        assert "cmyk" in response.json()["detail"]

    def test_a_missing_curve_is_a_404(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_eight_bit_source()),
            data={"curve_id": "00000000-0000-4000-8000-000000000000"},
        )

        assert response.status_code == 404

    def test_a_traversal_curve_id_is_not_a_server_error(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_eight_bit_source()),
            data={"curve_id": "../../../etc/passwd"},
        )

        assert response.status_code == 404

    def test_a_file_that_is_not_an_image_is_refused(self, client) -> None:
        response = client.post(ENDPOINT, files={"file": ("x.png", b"not an image", "image/png")})

        assert response.status_code in (400, 415)

    def test_a_decompression_bomb_is_refused_before_decode(self, client) -> None:
        bomb = io.BytesIO()
        Image.new("1", (30_000, 30_000)).save(bomb, format="PNG")

        response = client.post(ENDPOINT, files=_upload(bomb.getvalue(), "bomb.png"))

        assert response.status_code == 413

    def test_temporary_files_do_not_accumulate(self, tmp_path) -> None:
        """Both the upload and the rendered negative are server-owned.

        The source is removed once it is decoded and the negative once the
        response has been sent, so a busy server does not fill its disk with
        other people's scans.
        """
        from fastapi.testclient import TestClient

        from ptpd_calibration.api.server import create_app
        from ptpd_calibration.config import Settings

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        settings = Settings()
        settings.api.upload_dir = upload_dir

        with TestClient(create_app(settings)) as isolated:
            ok = isolated.post(ENDPOINT, files=_upload(_eight_bit_source()), data={"format": "png"})
            assert ok.status_code == 200
            rejected = isolated.post(
                ENDPOINT, files=_upload(_eight_bit_source()), data={"format": "bmp"}
            )
            assert rejected.status_code == 422

        assert list(upload_dir.iterdir()) == []


class TestExportFailuresAreClientErrors:
    """A refused format must not escape as a 500 or strand a file.

    The export call sat outside the handler's error boundary, so the one
    combination the writers deliberately refuse, 16-bit colour as PNG, became
    an unhandled 500 and left the rendered negative on disk.
    """

    def test_sixteen_bit_colour_png_is_a_client_error(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_rgb_source()),
            data={"format": "png_16bit", "color_mode": "rgb"},
        )

        assert response.status_code == 422, response.text
        assert "16-bit colour" in response.json()["detail"]

    def test_sixteen_bit_colour_tiff_still_succeeds(self, client) -> None:
        response = client.post(
            ENDPOINT,
            files=_upload(_rgb_source()),
            data={"format": "tiff_16bit", "color_mode": "rgb"},
        )

        assert response.status_code == 200, response.text

    def test_a_refused_export_leaves_no_file_behind(self, tmp_path) -> None:
        from fastapi.testclient import TestClient

        from ptpd_calibration.api.server import create_app
        from ptpd_calibration.config import Settings

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        settings = Settings()
        settings.api.upload_dir = upload_dir

        with TestClient(create_app(settings)) as isolated:
            refused = isolated.post(
                ENDPOINT,
                files=_upload(_rgb_source()),
                data={"format": "png_16bit", "color_mode": "rgb"},
            )
            assert refused.status_code == 422

        assert list(upload_dir.iterdir()) == []
