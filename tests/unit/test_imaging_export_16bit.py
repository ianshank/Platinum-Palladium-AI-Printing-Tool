"""Regression tests for the Pillow ``mode`` deprecation fix in ``imaging/processor.py``.

``Image.fromarray(arr, mode="I;16")`` is deprecated (removed in Pillow 13); the
processor now lets Pillow infer ``I;16`` from the ``uint16`` dtype. These tests
pin the produced image mode, dtype and pixel values so the change is verified
to be behaviour-preserving.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.config import ImagingSettings
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging.processor import (
    EIGHT_BIT_LEVELS,
    EIGHT_BIT_MAX,
    SIXTEEN_BIT_LEVELS,
    SIXTEEN_BIT_MAX,
    ColorMode,
    ExportSettings,
    ImageFormat,
    ImageProcessor,
    to_eight_bit_gray,
    to_uint16,
)
from ptpd_calibration.imaging.safe_image import (
    HIGH_DEPTH_GRAY_MODES,
    ImageDecodeSettings,
    open_image_safely,
    resize_to_fit,
)

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


class TestHighDepthPipeline:
    """A 16-bit source must stay 16-bit through curve, inversion and export.

    Before this, ``apply_curve`` dispatched on the Pillow mode and ``I;16``
    matched no branch, so a 16-bit scan was converted to 8-bit RGB. ``export``
    then multiplied the result by 257 and wrote a file that declared 16 bits
    while holding fewer distinct levels than the scanner produced. The tell is
    that every sample in such a file divides by 257.
    """

    SIDE = 64
    GAMMA = 1.5

    @staticmethod
    def _curve() -> CurveData:
        axis = np.linspace(0.0, 1.0, 32)
        return CurveData(
            name="high-depth",
            input_values=list(axis),
            output_values=list(axis**TestHighDepthPipeline.GAMMA),
        )

    @classmethod
    def _ramp(cls) -> np.ndarray:
        return (
            np.linspace(0, SIXTEEN_BIT_MAX, cls.SIDE * cls.SIDE)
            .astype(np.uint16)
            .reshape(cls.SIDE, cls.SIDE)
        )

    @classmethod
    def _scan(cls, tmp_path: Path) -> Path:
        path = tmp_path / "scan.tiff"
        Image.fromarray(cls._ramp()).save(path, format="TIFF")
        return path

    def test_load_keeps_the_scanner_depth(self, tmp_path: Path) -> None:
        result = ImageProcessor().load_image(self._scan(tmp_path))

        assert result.image.mode == EXPECTED_16BIT_MODE
        assert np.asarray(result.image).dtype == EXPECTED_16BIT_DTYPE

    def test_apply_curve_keeps_the_scanner_depth(self, tmp_path: Path) -> None:
        processor = ImageProcessor()
        loaded = processor.load_image(self._scan(tmp_path))

        curved = processor.apply_curve(loaded, self._curve())

        assert curved.image.mode == EXPECTED_16BIT_MODE
        levels = np.unique(np.asarray(curved.image)).size
        assert levels > EIGHT_BIT_LEVELS, f"curve quantised the scan to {levels} levels"

    def test_invert_uses_the_sixteen_bit_maximum(self) -> None:
        """Inverting against 255 would clamp every 16-bit sample to black."""
        mid = np.full((4, 4), SIXTEEN_BIT_MAX // 2, dtype=np.uint16)
        processor = ImageProcessor()

        inverted = processor.invert(processor.load_image(Image.fromarray(mid)))

        assert inverted.image.mode == EXPECTED_16BIT_MODE
        assert np.array_equal(np.asarray(inverted.image), SIXTEEN_BIT_MAX - mid)

    @pytest.mark.parametrize(
        ("fmt", "suffix"),
        [(ImageFormat.TIFF_16BIT, ".tiff"), (ImageFormat.PNG_16BIT, ".png")],
        ids=["tiff", "png"],
    )
    def test_exported_negative_is_genuinely_sixteen_bit(
        self, tmp_path: Path, fmt: ImageFormat, suffix: str
    ) -> None:
        processor = ImageProcessor()
        negative = processor.create_digital_negative(self._scan(tmp_path), curve=self._curve())
        path = tmp_path / f"negative{suffix}"

        processor.export(negative, path, ExportSettings(format=fmt))

        with Image.open(path) as loaded:
            loaded.load()
            assert loaded.mode == EXPECTED_16BIT_MODE, "the negative lost its single channel"
            data = np.asarray(loaded)

        assert data.ndim == 2
        assert data.dtype == EXPECTED_16BIT_DTYPE
        assert np.unique(data).size > EIGHT_BIT_LEVELS
        assert not np.all(data % SCALE_8_TO_16 == 0), "every sample divides by 257: 8-bit upconvert"

    def test_an_eight_bit_source_still_upconverts_by_257(self, tmp_path: Path) -> None:
        """The honest 8-bit path is unchanged; only the 16-bit path was wrong."""
        gradient = _gradient()
        processor = ImageProcessor()
        path = tmp_path / "eight.tiff"

        processor.export(
            processor.load_image(Image.fromarray(gradient)),
            path,
            ExportSettings(format=ImageFormat.TIFF_16BIT),
        )

        with Image.open(path) as loaded:
            loaded.load()
            data = np.asarray(loaded)

        assert np.array_equal(data, gradient.astype(np.uint16) * SCALE_8_TO_16)

    def test_a_uint16_array_source_is_not_truncated(self) -> None:
        """``astype(np.uint8)`` wraps modulo 256, so level 256 arrived as 0."""
        ramp = self._ramp()

        result = ImageProcessor().load_image(ramp)

        assert result.image.mode == EXPECTED_16BIT_MODE
        assert np.array_equal(np.asarray(result.image), ramp)

    def test_explicit_rgb_request_scales_rather_than_clips(self, tmp_path: Path) -> None:
        """Asking for colour is a caller decision, not an accident of dispatch.

        Pillow's convert() clips a high-depth image at 255, so asserting only
        the mode would pass on a frame that is almost entirely white.
        """
        processor = ImageProcessor()
        loaded = processor.load_image(self._scan(tmp_path))

        curved = processor.apply_curve(loaded, self._curve(), ColorMode.RGB)

        assert curved.image.mode == "RGB"
        data = np.asarray(curved.image)
        assert float((data == EIGHT_BIT_MAX).mean()) < 0.1, "the frame was clipped to white"
        assert np.unique(data).size > 2

    def test_preview_thumbnails_a_high_depth_source(self, tmp_path: Path) -> None:
        """Pillow cannot resample "I;16", so this raised for every 16-bit scan."""
        processor = ImageProcessor()

        original, processed = processor.preview_curve_effect(
            self._scan(tmp_path), self._curve(), thumbnail_size=(16, 16)
        )

        assert max(original.size) <= 16
        assert max(processed.size) <= 16
        assert np.unique(np.asarray(processed)).size > 2

    def test_preserve_bit_depth_off_coarsens_rather_than_wraps(self) -> None:
        """Opting out must mean fewer levels, not scrambled ones.

        The escape hatch used to route the array back through
        ``astype(np.uint8)``, which wraps modulo 256: level 256 became 0.
        """
        processor = ImageProcessor(ImagingSettings(preserve_bit_depth=False))
        source = np.array(
            [[0, SCALE_8_TO_16, 128 * SCALE_8_TO_16, SIXTEEN_BIT_MAX]], dtype=np.uint16
        )

        loaded = processor.load_image(source)

        assert loaded.image.mode == "L"
        assert np.array_equal(
            np.asarray(loaded.image), np.array([[0, 1, 128, 255]], dtype=np.uint8)
        )

    def test_both_depths_describe_the_same_transfer_function(self) -> None:
        """One builder serves both tables, so they cannot drift apart."""
        processor = ImageProcessor()
        curve = self._curve()

        lut8 = processor._create_lut(curve)
        lut16 = processor._create_lut_16(curve)

        assert lut8.size == EIGHT_BIT_LEVELS
        assert lut16.size == SIXTEEN_BIT_LEVELS
        sampled = np.rint(lut16[np.arange(EIGHT_BIT_LEVELS) * SCALE_8_TO_16] / SCALE_8_TO_16)
        assert np.all(np.abs(sampled - lut8) <= 1)

    def test_sixteen_bit_scaling_does_not_overflow(self) -> None:
        """Multiplying an already-16-bit array by 257 would wrap it."""
        full_scale = np.full((2, 2), SIXTEEN_BIT_MAX, dtype=np.uint16)

        assert np.array_equal(ImageProcessor._as_16bit(full_scale), full_scale)


class TestLutCacheBounds:
    """The lookup-table cache must not grow for the life of the process.

    ``BatchProcessor`` holds one ``ImageProcessor`` and applies a different
    curve per job. A 16-bit table is 128 KB, so an unbounded cache was a slow
    leak proportional to the number of curves a batch touched.
    """

    @staticmethod
    def _curve(gamma: float) -> CurveData:
        axis = np.linspace(0.0, 1.0, 16)
        return CurveData(name="c", input_values=list(axis), output_values=list(axis**gamma))

    def test_the_cache_is_bounded(self) -> None:
        limit = 3
        processor = ImageProcessor(ImagingSettings(lut_cache_entries=limit))

        for index in range(limit * 3):
            processor._create_lut(self._curve(1.0 + index / 10))

        assert len(processor._lut_cache) == limit

    def test_a_re_read_table_outlives_a_newer_one(self) -> None:
        """Eviction must be least-recently-*used*, not first-in-first-out.

        A bound alone is satisfied by FIFO or by clearing the cache, so this
        reads the oldest entry back before overflowing it.
        """
        processor = ImageProcessor(ImagingSettings(lut_cache_entries=2))
        oldest, newer, newest = (self._curve(g) for g in (1.1, 1.2, 1.3))

        processor._create_lut(oldest)
        processor._create_lut(newer)
        processor._create_lut(oldest)  # touch the oldest: it is now most recent
        processor._create_lut(newest)  # evicts `newer`, not `oldest`

        keys = list(processor._lut_cache)
        assert processor._cache_key_for(oldest) in keys
        assert processor._cache_key_for(newer) not in keys

    def test_concurrent_callers_do_not_race(self) -> None:
        """BatchProcessor shares one ImageProcessor across a thread pool.

        An interleaved read and eviction raised KeyError inside move_to_end.
        """
        processor = ImageProcessor(ImagingSettings(lut_cache_entries=2))
        curves = [self._curve(1.0 + index / 10) for index in range(8)]
        errors: list[str] = []

        def hammer() -> None:
            try:
                for _ in range(40):
                    for curve in curves:
                        processor._create_lut(curve)
                        processor._create_lut_16(curve)
            except Exception as exc:  # pragma: no cover - the bug being pinned
                errors.append(f"{type(exc).__name__}: {exc}")

        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=hammer) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)

        assert errors == []

    def test_the_two_depths_do_not_share_a_cache_entry(self) -> None:
        processor = ImageProcessor()
        curve = self._curve(2.0)

        lut8 = processor._create_lut(curve)
        lut16 = processor._create_lut_16(curve)

        assert lut8.dtype == np.uint8
        assert lut16.dtype == np.uint16
        assert np.array_equal(processor._create_lut(curve), lut8)


class TestEightBitDestinations:
    """Formats that cannot carry 16 bits must scale down, not clip or raise.

    Keeping the depth through the pipeline means the image reaching ``export``
    is now "I;16". JPEG is 8-bit only, and Pillow's own ``convert("L")`` clips
    ``I;16`` rather than scaling it, so every sample above 255 would have come
    out white. Both are silent failures in the print.
    """

    @staticmethod
    def _sixteen_bit(values: list[int]) -> Image.Image:
        return Image.fromarray(np.array([values], dtype=np.uint16))

    def test_pillow_convert_clips_rather_than_scales(self) -> None:
        """The premise of ``to_eight_bit_gray``, pinned so it cannot regress."""
        image = self._sixteen_bit([0, SCALE_8_TO_16, SIXTEEN_BIT_MAX])

        assert np.array_equal(np.asarray(image.convert("L")), np.array([[0, 255, 255]]))

    def test_to_eight_bit_gray_scales_the_full_range(self) -> None:
        image = self._sixteen_bit([0, SCALE_8_TO_16, 128 * SCALE_8_TO_16, SIXTEEN_BIT_MAX])

        reduced = to_eight_bit_gray(image)

        assert reduced.mode == "L"
        assert np.array_equal(np.asarray(reduced), np.array([[0, 1, 128, 255]], dtype=np.uint8))

    @pytest.mark.parametrize(
        "fmt", [ImageFormat.JPEG, ImageFormat.JPEG_HIGH], ids=["jpeg", "jpeg_high"]
    )
    def test_jpeg_export_of_a_high_depth_image_succeeds(
        self, tmp_path: Path, fmt: ImageFormat
    ) -> None:
        ramp = np.linspace(0, SIXTEEN_BIT_MAX, 64 * 64).astype(np.uint16).reshape(64, 64)
        processor = ImageProcessor()
        loaded = processor.load_image(Image.fromarray(ramp))
        path = tmp_path / "negative.jpg"

        processor.export(loaded, path, ExportSettings(format=fmt))

        with Image.open(path) as saved:
            saved.load()
            assert saved.mode == "L"
            # A clipping conversion would have left almost the whole frame white.
            assert np.asarray(saved).mean() < 200

    @pytest.mark.parametrize(
        "fmt", [ImageFormat.JPEG, ImageFormat.JPEG_HIGH], ids=["jpeg", "jpeg_high"]
    )
    def test_jpeg_bytes_export_of_a_high_depth_image_succeeds(self, fmt: ImageFormat) -> None:
        ramp = np.linspace(0, SIXTEEN_BIT_MAX, 32 * 32).astype(np.uint16).reshape(32, 32)
        processor = ImageProcessor()

        payload, extension = processor.export_to_bytes(
            processor.load_image(Image.fromarray(ramp)), ExportSettings(format=fmt)
        )

        assert extension == ".jpg"
        assert payload[:2] == b"\xff\xd8"  # JPEG start-of-image marker

    def test_a_loaded_array_is_not_aliased_to_the_caller(self) -> None:
        """``Image.fromarray`` shares the buffer unless the array is copied."""
        source = np.zeros((2, 2), dtype=np.uint16)
        result = ImageProcessor().load_image(source)

        source[0, 0] = SIXTEEN_BIT_MAX

        assert np.asarray(result.image)[0, 0] == 0

    def test_narrowing_a_wider_source_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A 32-bit "I" source loses highlights; that must not be silent."""
        wider = np.array([[0, SIXTEEN_BIT_MAX + 1000]], dtype=np.int32)

        with caplog.at_level(logging.WARNING, logger="ptpd_calibration.imaging.processor"):
            narrowed = to_uint16(wider)

        assert narrowed[0, 1] == SIXTEEN_BIT_MAX
        assert "Clamped 1 sample(s) above" in caplog.text

    def test_clamping_a_negative_sample_is_logged_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ "I" is signed, so a source can carry values below zero."""
        signed = np.array([[-5, 100]], dtype=np.int32)

        with caplog.at_level(logging.WARNING, logger="ptpd_calibration.imaging.processor"):
            narrowed = to_uint16(signed)

        assert narrowed[0, 0] == 0
        assert "1 below zero" in caplog.text

    def test_an_in_range_source_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        in_range = np.array([[0, SIXTEEN_BIT_MAX]], dtype=np.int32)

        with caplog.at_level(logging.WARNING, logger="ptpd_calibration.imaging.processor"):
            to_uint16(in_range)

        assert caplog.text == ""


class TestHighDepthDecode:
    """A 16-bit scan larger than the decode limit must open, not raise.

    ``ImageDecodeSettings.downsample_max_side`` defaults to 4096 and
    ``Image.thumbnail`` cannot resample any ``I;16`` mode, so every real
    scanner file above that size raised ``ValueError: image has wrong mode``
    from inside the decode guard. That made the whole 16-bit path unreachable
    for the input it exists to serve, and ``BatchProcessor`` reported it as a
    per-file failure with an opaque message.
    """

    WIDE = 200
    TALL = 100
    LIMIT = 64

    @classmethod
    def _wide_scan(cls, tmp_path: Path) -> Path:
        path = tmp_path / "wide.tiff"
        ramp = (
            np.linspace(0, SIXTEEN_BIT_MAX, cls.WIDE * cls.TALL)
            .astype(np.uint16)
            .reshape(cls.TALL, cls.WIDE)
        )
        Image.fromarray(ramp).save(path, format="TIFF")
        return path

    def test_an_oversized_high_depth_scan_decodes(self, tmp_path: Path) -> None:
        image = open_image_safely(
            self._wide_scan(tmp_path), ImageDecodeSettings(downsample_max_side=self.LIMIT)
        )

        assert max(image.size) <= self.LIMIT
        assert image.mode in HIGH_DEPTH_GRAY_MODES
        # Clipping to "L" would have capped the ramp at 255.
        assert int(np.asarray(image).max()) > EIGHT_BIT_MAX

    def test_an_eight_bit_scan_is_unaffected(self, tmp_path: Path) -> None:
        path = tmp_path / "wide.png"
        Image.new("L", (self.WIDE, self.TALL), 128).save(path)

        image = open_image_safely(path, ImageDecodeSettings(downsample_max_side=self.LIMIT))

        assert image.mode == "L"
        assert max(image.size) <= self.LIMIT

    def test_resize_to_fit_leaves_a_small_image_alone(self) -> None:
        image = Image.fromarray(np.zeros((4, 4), dtype=np.uint16))

        assert resize_to_fit(image, 64) is image

    @pytest.mark.parametrize("mode", ["I", "I;16B"], ids=["int32", "big-endian"])
    def test_other_high_depth_modes_carry_through(self, mode: str) -> None:
        """ "I" and the big-endian variants are decode modes the guard allows."""
        values = np.array([[0, SCALE_8_TO_16, 128 * SCALE_8_TO_16, SIXTEEN_BIT_MAX]], np.uint16)
        image = Image.fromarray(values)
        if mode != "I;16B":
            image = image.convert(mode)

        assert np.array_equal(to_uint16(np.asarray(image)), values)


class TestExportDepthMatchesTheRequestedFormat:
    """The eight-bit and sixteen-bit menu entries must produce different files.

    ``ImageFormat.TIFF`` sits beside ``ImageFormat.TIFF_16BIT`` in the format
    list, and ``BatchSettings.export_format`` defaults to the plain one. With
    depth carried through the pipeline, both wrote a 16-bit file and the two
    choices became indistinguishable.
    """

    @staticmethod
    def _high_depth_result() -> object:
        ramp = np.linspace(0, SIXTEEN_BIT_MAX, 32 * 32).astype(np.uint16).reshape(32, 32)
        return ImageProcessor().load_image(Image.fromarray(ramp))

    @pytest.mark.parametrize(
        ("fmt", "suffix", "expected_mode"),
        [
            (ImageFormat.TIFF, ".tiff", "L"),
            (ImageFormat.PNG, ".png", "L"),
            (ImageFormat.TIFF_16BIT, ".tiff", EXPECTED_16BIT_MODE),
            (ImageFormat.PNG_16BIT, ".png", EXPECTED_16BIT_MODE),
        ],
        ids=["tiff-8", "png-8", "tiff-16", "png-16"],
    )
    def test_requested_format_decides_the_depth(
        self, tmp_path: Path, fmt: ImageFormat, suffix: str, expected_mode: str
    ) -> None:
        processor = ImageProcessor()
        path = tmp_path / f"out{suffix}"

        processor.export(self._high_depth_result(), path, ExportSettings(format=fmt))

        with Image.open(path) as saved:
            saved.load()
            assert saved.mode == expected_mode

    def test_original_keeps_the_source_depth(self, tmp_path: Path) -> None:
        """ "Same as input" must not quietly mean "eight bits"."""
        processor = ImageProcessor()
        path = tmp_path / "out.tiff"

        processor.export(
            self._high_depth_result(), path, ExportSettings(format=ImageFormat.ORIGINAL)
        )

        with Image.open(path) as saved:
            saved.load()
            assert saved.mode == EXPECTED_16BIT_MODE

    @pytest.mark.parametrize(
        ("fmt", "suffix"),
        [(ImageFormat.TIFF_16BIT, ".tiff"), (ImageFormat.PNG_16BIT, ".png")],
        ids=["tiff", "png"],
    )
    def test_export_to_bytes_honours_a_sixteen_bit_request(
        self, fmt: ImageFormat, suffix: str
    ) -> None:
        """Only the file path widened an 8-bit image; the bytes path did not.

        An HTTP export endpoint returns bytes, so it silently served 8-bit
        files to callers who asked for 16.
        """
        gradient = _gradient()
        processor = ImageProcessor()

        payload, extension = processor.export_to_bytes(
            processor.load_image(Image.fromarray(gradient)), ExportSettings(format=fmt)
        )

        assert extension == suffix
        with Image.open(io.BytesIO(payload)) as saved:
            saved.load()
            assert saved.mode == EXPECTED_16BIT_MODE
            assert np.array_equal(np.asarray(saved), gradient.astype(np.uint16) * SCALE_8_TO_16)

    def test_export_to_bytes_keeps_a_high_depth_source(self) -> None:
        processor = ImageProcessor()

        payload, _ = processor.export_to_bytes(
            self._high_depth_result(), ExportSettings(format=ImageFormat.TIFF_16BIT)
        )

        with Image.open(io.BytesIO(payload)) as saved:
            saved.load()
            data = np.asarray(saved)
        assert saved.mode == EXPECTED_16BIT_MODE
        assert not np.all(data % SCALE_8_TO_16 == 0)

    @pytest.mark.parametrize(
        ("array", "expected"),
        [
            (np.array([[EIGHT_BIT_MAX]], dtype=np.uint8), SIXTEEN_BIT_MAX),
            (np.array([[True]], dtype=bool), SIXTEEN_BIT_MAX),
            (np.array([[float(EIGHT_BIT_MAX)]], dtype=np.float32), SIXTEEN_BIT_MAX),
            (np.array([[float(SIXTEEN_BIT_MAX)]], dtype=np.float32), SIXTEEN_BIT_MAX),
            (np.array([[SIXTEEN_BIT_MAX]], dtype=np.uint16), SIXTEEN_BIT_MAX),
        ],
        ids=["uint8", "bool", "float-8bit-range", "float-16bit-range", "uint16"],
    )
    def test_full_scale_stays_full_scale(self, array: np.ndarray, expected: int) -> None:
        """Multiplying by 257 wrapped float and bool full-scale to near black."""
        assert int(ImageProcessor._as_16bit(array)[0, 0]) == expected


class TestAIFacadeKeepsTheDepth:
    """``PlatinumPalladiumAI`` builds negatives without ``ImageProcessor``.

    It carried its own ``convert("L")`` block, which clips rather than scales,
    so the public AI entry point produced an almost entirely white negative
    from a 16-bit scan even after the processor was fixed. A guard in one
    place is not a guard.
    """

    SIDE = 32

    def test_a_sixteen_bit_scan_survives_the_ai_entry_point(self, tmp_path: Path) -> None:
        from ptpd_calibration.ai.platinum_palladium_ai import PlatinumPalladiumAI

        ramp = (
            np.linspace(0, SIXTEEN_BIT_MAX, self.SIDE * self.SIDE)
            .astype(np.uint16)
            .reshape(self.SIDE, self.SIDE)
        )
        source = tmp_path / "scan.tiff"
        Image.fromarray(ramp).save(source, format="TIFF")
        axis = np.linspace(0.0, 1.0, 32)
        curve = CurveData(name="ai", input_values=list(axis), output_values=list(axis**1.5))

        result = PlatinumPalladiumAI().generate_digital_negative(
            source,
            curve=curve,
            output_path=tmp_path / "negative.tiff",
            output_format=ImageFormat.TIFF_16BIT,
        )

        with Image.open(result.output_path) as saved:
            saved.load()
            data = np.asarray(saved)
            assert saved.mode == EXPECTED_16BIT_MODE
        assert np.unique(data).size > EIGHT_BIT_LEVELS
        assert not np.all(data % SCALE_8_TO_16 == 0)
        # Clipping blew 99.6% of the frame to white.
        assert float((data >= SIXTEEN_BIT_MAX - SCALE_8_TO_16).mean()) < 0.5
