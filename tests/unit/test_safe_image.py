"""Tests for the hardened image decoder (SEC-04)."""

from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFile

from ptpd_calibration.imaging.safe_image import (
    ImageDecodeError,
    ImageDecodeSettings,
    ImageTooLargeError,
    UnsupportedImageError,
    load_image_array,
    open_image_safely,
)

pytestmark = pytest.mark.unit

BOMB_SIDE = 20_000  # 400 MP declared, ~50 KB on disk as a 1-bit PNG


@pytest.fixture(scope="module")
def bomb_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny file whose header declares a 400 MP image."""
    path = tmp_path_factory.mktemp("bomb") / "bomb.png"
    Image.new("1", (BOMB_SIDE, BOMB_SIDE)).save(path, format="PNG")
    return path


@pytest.fixture
def small_rgb(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    path = tmp_path / "small.png"
    Image.fromarray(arr).save(path)
    return path, arr


@pytest.fixture
def no_pixel_decode(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any call to the pixel decoder so rejected images can prove they never decoded."""
    calls: list[str] = []
    original = ImageFile.ImageFile.load

    def spy(self: ImageFile.ImageFile) -> object:
        calls.append(self.format or "?")
        return original(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", spy)
    return calls


class TestImageDecodeSettings:
    def test_defaults(self) -> None:
        settings = ImageDecodeSettings()
        assert settings.max_pixels == 40_000_000
        assert settings.max_frames == 1
        assert settings.allowed_formats == ["PNG", "JPEG", "TIFF", "BMP"]
        assert settings.downsample_max_side == 4096
        assert settings.treat_bomb_warning_as_error is True
        assert {"L", "RGB", "I;16", "RGBA"} <= set(settings.allowed_modes)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_IMAGE_MAX_PIXELS", "100")
        monkeypatch.setenv("PTPD_IMAGE_ALLOWED_FORMATS", '["PNG"]')
        monkeypatch.setenv("PTPD_IMAGE_DOWNSAMPLE_MAX_SIDE", "8")
        settings = ImageDecodeSettings()
        assert settings.max_pixels == 100
        assert settings.allowed_formats == ["PNG"]
        assert settings.downsample_max_side == 8

    def test_env_override_is_enforced_by_default_helper(
        self, monkeypatch: pytest.MonkeyPatch, small_rgb: tuple[Path, np.ndarray]
    ) -> None:
        monkeypatch.setenv("PTPD_IMAGE_MAX_PIXELS", "100")
        path, _ = small_rgb  # 600 pixels
        with pytest.raises(ImageTooLargeError):
            open_image_safely(path)


class TestDecompressionBomb:
    def test_bomb_file_is_small_and_rejected_before_decode(
        self, bomb_png: Path, no_pixel_decode: list[str]
    ) -> None:
        assert bomb_png.stat().st_size < 512 * 1024, "fixture must be a real bomb: tiny file"

        started = time.perf_counter()
        with pytest.raises(ImageTooLargeError, match="pixels"):
            open_image_safely(bomb_png)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert no_pixel_decode == []

    def test_bomb_rejected_via_load_image_array(self, bomb_png: Path) -> None:
        with pytest.raises(ImageTooLargeError):
            load_image_array(bomb_png)

    def test_pillow_global_limit_is_restored(self, bomb_png: Path) -> None:
        before = Image.MAX_IMAGE_PIXELS
        with pytest.raises(ImageTooLargeError):
            open_image_safely(bomb_png, ImageDecodeSettings(max_pixels=10))
        assert before == Image.MAX_IMAGE_PIXELS

    def test_explicit_check_catches_range_between_warning_and_error(
        self, small_rgb: tuple[Path, np.ndarray]
    ) -> None:
        """600 px lies between max_pixels and 2*max_pixels: Pillow only warns there."""
        path, _ = small_rgb
        settings = ImageDecodeSettings(max_pixels=400, treat_bomb_warning_as_error=False)
        with pytest.warns(Image.DecompressionBombWarning), pytest.raises(ImageTooLargeError):
            open_image_safely(path, settings)

    def test_warning_promoted_to_error_when_configured(
        self, small_rgb: tuple[Path, np.ndarray]
    ) -> None:
        path, _ = small_rgb
        settings = ImageDecodeSettings(max_pixels=400, treat_bomb_warning_as_error=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any leaked warning would fail the test
            with pytest.raises(ImageTooLargeError):
                open_image_safely(path, settings)


class TestFormatAndFrameGuards:
    def test_multi_frame_tiff_rejected(self, tmp_path: Path, no_pixel_decode: list[str]) -> None:
        path = tmp_path / "pages.tif"
        first, second = Image.new("L", (8, 8), 10), Image.new("L", (8, 8), 200)
        first.save(path, format="TIFF", save_all=True, append_images=[second])

        with pytest.raises(UnsupportedImageError, match="frames"):
            open_image_safely(path)
        assert no_pixel_decode == []

    def test_multi_frame_allowed_when_limit_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "pages.tif"
        Image.new("L", (8, 8)).save(
            path, format="TIFF", save_all=True, append_images=[Image.new("L", (8, 8))]
        )
        with open_image_safely(path, ImageDecodeSettings(max_frames=2)) as im:
            assert im.size == (8, 8)

    def test_gif_renamed_to_png_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "actually.png"
        Image.new("P", (8, 8)).save(path, format="GIF")
        with pytest.raises(UnsupportedImageError, match="allowed formats"):
            open_image_safely(path)

    def test_eps_renamed_to_png_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "vector.png"
        path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n")
        with pytest.raises(UnsupportedImageError):
            open_image_safely(path)

    def test_garbage_bytes_rejected(self) -> None:
        with pytest.raises(UnsupportedImageError):
            open_image_safely(b"\x00\x01\x02 not an image")

    def test_mode_outside_allowlist_rejected(
        self, small_rgb: tuple[Path, np.ndarray], no_pixel_decode: list[str]
    ) -> None:
        path, _ = small_rgb
        with pytest.raises(UnsupportedImageError, match="pixel mode RGB"):
            open_image_safely(path, ImageDecodeSettings(allowed_modes=["L"]))
        assert no_pixel_decode == []

    def test_empty_mode_allowlist_disables_mode_check(
        self, small_rgb: tuple[Path, np.ndarray]
    ) -> None:
        path, _ = small_rgb
        assert open_image_safely(path, ImageDecodeSettings(allowed_modes=[])).mode == "RGB"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_image_safely(tmp_path / "missing.png")

    def test_unsupported_source_type(self) -> None:
        with pytest.raises(TypeError):
            open_image_safely(12345)  # type: ignore[arg-type]

    def test_errors_are_value_errors(self) -> None:
        assert issubclass(ImageTooLargeError, ValueError)
        assert issubclass(UnsupportedImageError, ValueError)
        assert issubclass(ImageTooLargeError, ImageDecodeError)


class TestRoundTrip:
    def test_path_round_trip(self, small_rgb: tuple[Path, np.ndarray]) -> None:
        path, arr = small_rgb
        im = open_image_safely(path)
        assert im.format == "PNG"
        assert np.array_equal(np.array(im), arr)

    def test_string_path_bytes_and_stream_sources(self, small_rgb: tuple[Path, np.ndarray]) -> None:
        path, arr = small_rgb
        raw = path.read_bytes()
        assert np.array_equal(np.array(open_image_safely(str(path))), arr)
        assert np.array_equal(np.array(open_image_safely(raw)), arr)
        assert np.array_equal(np.array(open_image_safely(io.BytesIO(raw))), arr)

    def test_sixteen_bit_grayscale(self, tmp_path: Path) -> None:
        arr = (np.arange(16 * 8, dtype=np.uint16).reshape(8, 16) * 257).astype(np.uint16)
        path = tmp_path / "gray16.png"
        Image.fromarray(arr).save(
            path
        )  # uint16 arrays map to I;16 without the deprecated mode kwarg
        loaded = load_image_array(path)
        assert loaded.dtype == np.uint16
        assert np.array_equal(loaded, arr)

    def test_load_image_array_passes_arrays_and_pil_images_through(self) -> None:
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        assert load_image_array(arr) is arr
        pil = Image.new("L", (5, 4), 7)
        out = load_image_array(pil)
        assert out.shape == (4, 5)
        assert int(out[0, 0]) == 7

    def test_load_image_array_rejects_unknown_types(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type"):
            load_image_array(3.14)  # type: ignore[arg-type]


class TestDownsampling:
    def test_downsample_preserves_aspect_ratio(self, tmp_path: Path) -> None:
        path = tmp_path / "wide.png"
        Image.new("RGB", (200, 100), (10, 20, 30)).save(path)
        im = open_image_safely(path, ImageDecodeSettings(downsample_max_side=64))
        assert im.size == (64, 32)
        assert im.mode == "RGB"

    def test_downsample_tall_image(self, tmp_path: Path) -> None:
        path = tmp_path / "tall.png"
        Image.new("L", (50, 200)).save(path)
        im = open_image_safely(path, ImageDecodeSettings(downsample_max_side=100))
        assert im.size == (25, 100)

    def test_no_downsample_when_within_limit(self, small_rgb: tuple[Path, np.ndarray]) -> None:
        path, arr = small_rgb
        im = open_image_safely(path, ImageDecodeSettings(downsample_max_side=30))
        assert np.array_equal(np.array(im), arr)

    def test_downsample_can_be_disabled(self, tmp_path: Path) -> None:
        path = tmp_path / "wide.png"
        Image.new("RGB", (200, 100)).save(path)
        im = open_image_safely(path, ImageDecodeSettings(downsample_max_side=None))
        assert im.size == (200, 100)

    def test_jpeg_downsample(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), (200, 100, 50)).save(path, format="JPEG", quality=90)
        im = open_image_safely(path, ImageDecodeSettings(downsample_max_side=100))
        assert max(im.size) == 100
        assert im.size[0] / im.size[1] == pytest.approx(4 / 3, abs=0.02)


class TestCallSites:
    """The detection and histogram entry points reject bombs and accept arrays."""

    def test_scanner_calibration_rejects_bomb(self, bomb_png: Path) -> None:
        from ptpd_calibration.detection.scanner import ScannerCalibration

        with pytest.raises(ImageTooLargeError):
            ScannerCalibration()._load_image(bomb_png)

    def test_detector_extractor_reader_reject_bomb(self, bomb_png: Path) -> None:
        from ptpd_calibration.detection.detector import StepTabletDetector
        from ptpd_calibration.detection.extractor import DensityExtractor
        from ptpd_calibration.detection.reader import StepTabletReader

        for loader in (StepTabletDetector(), DensityExtractor(), StepTabletReader()):
            with pytest.raises(ImageTooLargeError):
                loader._load_image(bomb_png)

    def test_histogram_analyzer_rejects_bomb_and_accepts_array(self, bomb_png: Path) -> None:
        from ptpd_calibration.imaging.histogram import HistogramAnalyzer

        analyzer = HistogramAnalyzer()
        with pytest.raises(ImageTooLargeError):
            analyzer.analyze(bomb_png)
        result = analyzer.analyze(np.full((8, 8), 128, dtype=np.uint8))
        assert result.image_size == (8, 8)

    def test_histogram_analyzer_loads_path(self, small_rgb: tuple[Path, np.ndarray]) -> None:
        from ptpd_calibration.imaging.histogram import HistogramAnalyzer

        path, _ = small_rgb
        result = HistogramAnalyzer().analyze(path)
        assert result.image_size == (30, 20)
        assert result.image_mode == "RGB"
