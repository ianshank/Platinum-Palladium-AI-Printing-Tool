"""Regression tests for the Pillow ``mode`` deprecation fix in ``imaging/processor.py``.

``Image.fromarray(arr, mode="I;16")`` is deprecated (removed in Pillow 13); the
processor now lets Pillow infer ``I;16`` from the ``uint16`` dtype. These tests
pin the produced image mode, dtype and pixel values so the change is verified
to be behaviour-preserving.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.imaging.processor import ExportSettings, ImageFormat, ImageProcessor

EXPECTED_16BIT_MODE = "I;16"
EXPECTED_16BIT_DTYPE = np.uint16
SCALE_8_TO_16 = 257
"""8-bit -> 16-bit scaling used by ImageProcessor.export (255 * 257 == 65535)."""
GRADIENT_SIDE = 10


def _gradient() -> np.ndarray:
    return np.arange(GRADIENT_SIDE * GRADIENT_SIDE, dtype=np.uint8).reshape(
        GRADIENT_SIDE, GRADIENT_SIDE
    )


def _export(fmt: ImageFormat, path: Path) -> np.ndarray:
    gradient = _gradient()
    processor = ImageProcessor()
    result = processor.load_image(Image.fromarray(gradient))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        processor.export(result, path, ExportSettings(format=fmt))
    return gradient


@pytest.mark.parametrize(
    ("fmt", "suffix"),
    [(ImageFormat.TIFF_16BIT, ".tiff"), (ImageFormat.PNG_16BIT, ".png")],
    ids=["tiff", "png"],
)
def test_16bit_grayscale_export_mode_dtype_and_values(
    tmp_path: Path, fmt: ImageFormat, suffix: str
) -> None:
    path = tmp_path / f"gradient{suffix}"
    gradient = _export(fmt, path)

    with Image.open(path) as loaded:
        loaded.load()
        assert loaded.mode == EXPECTED_16BIT_MODE
        data = np.array(loaded)

    assert data.dtype == EXPECTED_16BIT_DTYPE
    assert np.array_equal(data, gradient.astype(np.uint16) * SCALE_8_TO_16)


def test_fromarray_infers_16bit_mode_without_mode_argument() -> None:
    """Pillow maps a uint16 2-D array to "I;16" on its own, no ``mode`` needed."""
    arr = _gradient().astype(np.uint16) * SCALE_8_TO_16

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = Image.fromarray(arr)

    assert image.mode == EXPECTED_16BIT_MODE
    assert np.array_equal(np.array(image), arr)


def test_save_16bit_tiff_helper_emits_no_deprecation_warning(tmp_path: Path) -> None:
    arr = _gradient().astype(np.uint16) * SCALE_8_TO_16
    path = tmp_path / "helper.tiff"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ImageProcessor()._save_16bit_tiff(arr, path, {})

    with Image.open(path) as loaded:
        loaded.load()
        assert loaded.mode == EXPECTED_16BIT_MODE
        assert np.array_equal(np.array(loaded), arr)


class TestLoadImageGuards:
    """``ImageProcessor.load_image`` must apply the same guards as the API path.

    Before this, the path and bytes branches called ``PIL.Image.open`` directly,
    so a decompression bomb reached the decoder even though
    ``open_image_safely`` existed for exactly that reason.
    """

    BOMB_SIDE = 20_000  # 400 MP declared, tiny on disk as a 1-bit PNG

    def test_path_source_rejects_a_decompression_bomb(self, tmp_path: Path) -> None:
        from ptpd_calibration.imaging.safe_image import ImageTooLargeError

        bomb = tmp_path / "bomb.png"
        Image.new("1", (self.BOMB_SIDE, self.BOMB_SIDE)).save(bomb, format="PNG")

        with pytest.raises(ImageTooLargeError):
            ImageProcessor().load_image(bomb)

    def test_bytes_source_rejects_a_decompression_bomb(self, tmp_path: Path) -> None:
        from ptpd_calibration.imaging.safe_image import ImageTooLargeError

        bomb = tmp_path / "bomb.png"
        Image.new("1", (self.BOMB_SIDE, self.BOMB_SIDE)).save(bomb, format="PNG")

        with pytest.raises(ImageTooLargeError):
            ImageProcessor().load_image(bomb.read_bytes())

    def test_ordinary_image_still_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.png"
        Image.new("L", (16, 12), 128).save(path)

        result = ImageProcessor().load_image(path)

        assert result.image.size == (16, 12)
        assert result.original_size == (16, 12)
        assert result.original_format == "PNG"
