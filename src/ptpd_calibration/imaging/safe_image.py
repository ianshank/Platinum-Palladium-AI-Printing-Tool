"""
Hardened image decoding (plan item SEC-04).

Every path-, bytes- or stream-based image open in the detection and imaging
packages goes through :func:`open_image_safely`, which

* restricts the decoder plugins Pillow may try (``allowed_formats``);
* rejects images whose *declared* size, frame count or pixel mode exceeds the
  configured limits **before any pixel data is decoded** (decompression-bomb
  defence: a 20000x20000 1-bit PNG is a few kilobytes on disk but 400 MP once
  decoded);
* pins ``PIL.Image.MAX_IMAGE_PIXELS`` to the same limit for the duration of the
  header read and turns Pillow's ``DecompressionBombWarning`` into an error;
* optionally down-samples very large images (aspect ratio preserved) so the
  NumPy work downstream stays bounded.

All limits live on :class:`ImageDecodeSettings` and can be overridden with
``PTPD_IMAGE_*`` environment variables (``PTPD_IMAGE_MAX_PIXELS`` etc.).
NumPy arrays and already-open ``PIL.Image.Image`` objects are passed through
unchanged by :func:`load_image_array`, so in-memory callers are unaffected.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ImageSource = Path | str | bytes | bytearray | memoryview | BinaryIO
"""Sources :func:`open_image_safely` accepts (a path, raw bytes or a binary stream)."""

#: Pillow modes for single-channel images carrying more than eight bits per
#: sample. ``I`` is 32-bit signed integer; the ``I;16*`` family is unsigned
#: 16-bit in the byte orders a scanner may produce. Defined here, at the
#: decode boundary, because both the decoder and the processor need them and
#: the processor is the higher layer.
HIGH_DEPTH_GRAY_MODES: frozenset[str] = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})

#: The widest integer mode Pillow can actually resample. ``resize`` raises
#: "image has wrong mode" for every ``I;16*`` mode, so a high-depth image is
#: converted to this first. The conversion is lossless: ``I`` is wider.
RESAMPLE_INTEGER_MODE = "I"

# ``Image.MAX_IMAGE_PIXELS`` and the warnings registry are process-global; the
# header read that depends on them is serialised so concurrent callers with
# different settings cannot observe each other's limit.
_OPEN_LOCK = threading.Lock()


class ImageDecodeError(ValueError):
    """Base class for images refused by :func:`open_image_safely`."""


class ImageTooLargeError(ImageDecodeError):
    """The declared pixel count exceeds ``ImageDecodeSettings.max_pixels``."""


class UnsupportedImageError(ImageDecodeError):
    """The file is not one of the allowed formats, modes or frame counts."""


class ImageDecodeSettings(BaseSettings):
    """Limits applied when decoding untrusted image files.

    Every field can be overridden with a ``PTPD_IMAGE_`` environment variable,
    e.g. ``PTPD_IMAGE_MAX_PIXELS=10000000`` or
    ``PTPD_IMAGE_ALLOWED_FORMATS='["PNG","TIFF"]'``.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_IMAGE_")

    max_pixels: int = Field(
        default=40_000_000,
        ge=1,
        description="Maximum width*height accepted; checked from the header before decoding",
    )
    max_frames: int = Field(
        default=1,
        ge=1,
        description="Maximum number of frames/pages (multi-page TIFF, APNG, MPO are rejected)",
    )
    allowed_formats: list[str] = Field(
        default_factory=lambda: ["PNG", "JPEG", "TIFF", "BMP"],
        description="Pillow plugin identifiers that may be used to decode a file",
    )
    allowed_modes: list[str] = Field(
        # Everything the detection/histogram code handles today: 8-bit gray and
        # colour (with alpha), palette and bilevel scans, and the 16/32-bit
        # integer and float modes produced by scanner TIFF/PNG output.
        default_factory=lambda: [
            "1",
            "L",
            "LA",
            "P",
            "PA",
            "RGB",
            "RGBA",
            "RGB;16",
            "CMYK",
            "I",
            "I;16",
            "I;16B",
            "I;16L",
            "F",
        ],
        description="Pillow pixel modes accepted; an empty list disables the mode check",
    )
    downsample_max_side: int | None = Field(
        default=4096,
        ge=1,
        description="Longest side after decoding; larger images are shrunk in place (None disables)",
    )
    treat_bomb_warning_as_error: bool = Field(
        default=True,
        description="Turn Pillow's DecompressionBombWarning into ImageTooLargeError",
    )


def _coerce_source(source: ImageSource) -> tuple[Path | BinaryIO, str]:
    """Normalise ``source`` into something ``Image.open`` accepts plus a log label."""
    if isinstance(source, Path | str):
        path = Path(source)
        return path, str(path)
    if isinstance(source, bytes | bytearray | memoryview):
        return io.BytesIO(bytes(source)), f"<{len(source)} bytes>"
    if hasattr(source, "read"):
        return source, str(getattr(source, "name", "<stream>"))
    raise TypeError(f"Unsupported image source type: {type(source)}")


def _open_header(fp: Path | BinaryIO, settings: ImageDecodeSettings, label: str) -> Image.Image:
    """Run ``Image.open`` (header parse only) under the configured pixel cap."""
    with _OPEN_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = settings.max_pixels
        try:
            with warnings.catch_warnings():
                if settings.treat_bomb_warning_as_error:
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                return Image.open(fp, formats=list(settings.allowed_formats))
        except Image.DecompressionBombError as exc:
            logger.debug("Rejected %s: Pillow decompression bomb error (%s)", label, exc)
            raise ImageTooLargeError(
                f"Image {label} declares more than {settings.max_pixels} pixels "
                "and was refused before decoding"
            ) from exc
        except Image.DecompressionBombWarning as exc:
            logger.debug("Rejected %s: Pillow decompression bomb warning (%s)", label, exc)
            raise ImageTooLargeError(
                f"Image {label} declares more than {settings.max_pixels} pixels "
                "and was refused before decoding"
            ) from exc
        except UnidentifiedImageError as exc:
            logger.debug("Rejected %s: not one of %s", label, settings.allowed_formats)
            raise UnsupportedImageError(
                f"Image {label} is not a supported format; allowed formats are "
                f"{', '.join(settings.allowed_formats)}"
            ) from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit


def _check_header(im: Image.Image, settings: ImageDecodeSettings, label: str) -> None:
    """Validate format, size, frame count and mode from header metadata only."""
    fmt = im.format or "unknown"
    if fmt not in settings.allowed_formats:
        # ``formats=`` limits the plugins tried, but a plugin may hand back a
        # sibling format (the JPEG plugin returns MPO for multi-picture files).
        logger.debug("Rejected %s: format %s not in %s", label, fmt, settings.allowed_formats)
        raise UnsupportedImageError(
            f"Image {label} has format {fmt}; allowed formats are "
            f"{', '.join(settings.allowed_formats)}"
        )

    width, height = im.size
    pixels = width * height
    if pixels > settings.max_pixels:
        logger.debug(
            "Rejected %s: %dx%d = %d pixels exceeds max_pixels=%d",
            label,
            width,
            height,
            pixels,
            settings.max_pixels,
        )
        raise ImageTooLargeError(
            f"Image {label} is {width}x{height} = {pixels} pixels, "
            f"which exceeds the limit of {settings.max_pixels} pixels"
        )

    try:
        n_frames = int(getattr(im, "n_frames", 1))
    except (OSError, ValueError, TypeError, EOFError) as exc:
        logger.debug("Rejected %s: could not determine frame count (%s)", label, exc)
        raise UnsupportedImageError(
            f"Image {label} has an unreadable frame table and was refused"
        ) from exc
    if n_frames > settings.max_frames:
        logger.debug(
            "Rejected %s: %d frames exceeds max_frames=%d", label, n_frames, settings.max_frames
        )
        raise UnsupportedImageError(
            f"Image {label} has {n_frames} frames; at most {settings.max_frames} "
            "frame(s) are accepted"
        )

    if settings.allowed_modes and im.mode not in settings.allowed_modes:
        logger.debug("Rejected %s: mode %s not in %s", label, im.mode, settings.allowed_modes)
        raise UnsupportedImageError(
            f"Image {label} uses pixel mode {im.mode}; allowed modes are "
            f"{', '.join(settings.allowed_modes)}"
        )


def resize_to_fit(im: Image.Image, max_side: int, label: str = "image") -> Image.Image:
    """Shrink ``im`` so its longest side is at most ``max_side``.

    Returns the same object when no work is needed or when it could be resized
    in place, and a new image when the mode had to change first. Callers must
    use the return value.

    Pillow cannot resample the ``I;16*`` modes at all: ``resize`` raises
    ``ValueError: image has wrong mode``. Every 16-bit scan wider than the
    configured limit therefore failed to open, which is most real scanner
    output. Converting to :data:`RESAMPLE_INTEGER_MODE` first is lossless and
    keeps the depth the scan was made at, where dropping to "L" would clip
    every sample above 255 to white.
    """
    original = im.size
    if max(original) <= max_side:
        return im
    if im.mode in HIGH_DEPTH_GRAY_MODES and im.mode != RESAMPLE_INTEGER_MODE:
        im = im.convert(RESAMPLE_INTEGER_MODE)
        logger.debug("Converted %s to %s so it can be resampled", label, RESAMPLE_INTEGER_MODE)
    # ``thumbnail`` keeps the aspect ratio, only ever shrinks, and uses the JPEG
    # DCT-scaling ``draft`` mode internally so oversized JPEGs never fully decode.
    im.thumbnail((max_side, max_side))
    logger.debug("Downsampled %s from %sx%s to %sx%s", label, *original, *im.size)
    return im


def _maybe_downsample(im: Image.Image, settings: ImageDecodeSettings, label: str) -> Image.Image:
    """Apply :func:`resize_to_fit` when ``downsample_max_side`` is configured."""
    max_side = settings.downsample_max_side
    if max_side is None:
        return im
    return resize_to_fit(im, max_side, label)


def open_image_safely(
    source: ImageSource,
    settings: ImageDecodeSettings | None = None,
) -> Image.Image:
    """Open an untrusted image with format, size, frame and mode guards.

    Args:
        source: A filesystem path, raw bytes or a binary file object.
        settings: Limits to apply; ``None`` reads :class:`ImageDecodeSettings`
            from the environment.

    Returns:
        A fully decoded ``PIL.Image.Image`` (down-sampled when configured).

    Raises:
        ImageTooLargeError: The declared pixel count exceeds ``max_pixels``.
        UnsupportedImageError: The format, frame count or mode is not allowed.
        FileNotFoundError: ``source`` is a path that does not exist.
        TypeError: ``source`` is not a path, bytes or binary stream.
    """
    settings = settings or ImageDecodeSettings()
    fp, label = _coerce_source(source)
    started = time.perf_counter()

    im = _open_header(fp, settings, label)
    try:
        _check_header(im, settings, label)
    except ImageDecodeError:
        # Nothing has been decoded yet; release the file handle and propagate.
        im.close()
        raise

    im = _maybe_downsample(im, settings, label)
    # Decode now (bounded by the checks above). For path sources this also
    # closes the underlying file handle, matching the old eager ``np.array``.
    im.load()
    logger.debug(
        "Decoded %s: format=%s mode=%s size=%sx%s in %.1f ms",
        label,
        im.format,
        im.mode,
        im.size[0],
        im.size[1],
        (time.perf_counter() - started) * 1000.0,
    )
    return im


def load_image_array(
    image: np.ndarray | Image.Image | ImageSource,
    settings: ImageDecodeSettings | None = None,
) -> np.ndarray:
    """Return ``image`` as a NumPy array, decoding files through the safe path.

    In-memory inputs keep their previous behaviour: NumPy arrays are returned
    as-is and ``PIL.Image.Image`` objects are converted with ``np.array``.
    Paths, bytes and binary streams go through :func:`open_image_safely`.

    Raises:
        TypeError: ``image`` is none of the supported types.
    """
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, Path | str | bytes | bytearray | memoryview) or hasattr(image, "read"):
        return np.array(open_image_safely(image, settings))
    raise TypeError(f"Unsupported image type: {type(image)}")


__all__ = [
    "HIGH_DEPTH_GRAY_MODES",
    "RESAMPLE_INTEGER_MODE",
    "ImageDecodeError",
    "ImageDecodeSettings",
    "ImageSource",
    "ImageTooLargeError",
    "UnsupportedImageError",
    "load_image_array",
    "open_image_safely",
    "resize_to_fit",
]
