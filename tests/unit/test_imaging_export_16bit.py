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

from ptpd_calibration.core.models import CurveData
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


class TestLutCache:
    """The lookup-table cache must identify a curve by its values.

    The key used to be name plus point count. Generated curves default to the
    same name and always carry 256 points, so the second curve of a session
    silently received the first one's table: a user who rescanned, regenerated
    and exported again got the previous calibration with no warning.
    """

    @staticmethod
    def _curve(outputs: list[float], name: str = "Calibration Curve") -> CurveData:
        return CurveData(
            name=name,
            input_values=list(np.linspace(0.0, 1.0, len(outputs))),
            output_values=outputs,
        )

    def test_same_name_and_length_do_not_share_a_table(self) -> None:
        axis = list(np.linspace(0.0, 1.0, 256))
        identity = self._curve(axis)
        squared = self._curve([v**2 for v in axis])
        processor = ImageProcessor()

        first = processor._create_lut(identity).copy()
        second = processor._create_lut(squared)

        assert not np.array_equal(first, second)
        assert first[128] == 128
        assert second[128] == 64

    def test_an_identical_curve_is_still_cached(self) -> None:
        axis = list(np.linspace(0.0, 1.0, 256))
        processor = ImageProcessor()

        first = processor._create_lut(self._curve(axis)).copy()
        again = processor._create_lut(self._curve(axis, name="a different label"))

        assert np.array_equal(first, again)

    def test_values_are_rounded_not_truncated(self) -> None:
        """Truncation biases every entry by up to half a code value."""
        axis = list(np.linspace(0.0, 1.0, 256))
        lut = ImageProcessor()._create_lut(self._curve(axis))

        assert lut[128] == 128  # 0.50196 * 255 = 128.0, truncation gives 127
        assert lut[0] == 0
        assert lut[-1] == 255
