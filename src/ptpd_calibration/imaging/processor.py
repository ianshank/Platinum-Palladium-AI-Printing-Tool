"""
Image processor for digital negative creation.

Applies calibration curves to images, creates inverted negatives,
and exports in various formats while preserving resolution.
"""

import hashlib
import io
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from ptpd_calibration.config import ImagingSettings, get_settings
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging.safe_image import open_image_safely

logger = logging.getLogger(__name__)

# Bit depths are fixed by the file formats, not by taste, so they are constants
# here rather than settings. The scale factor is exact: 255 * 257 == 65535, so
# an 8-bit value maps onto the 16-bit range without rounding drift.
EIGHT_BIT_LEVELS = 256
SIXTEEN_BIT_LEVELS = 65536
EIGHT_BIT_MAX = EIGHT_BIT_LEVELS - 1
SIXTEEN_BIT_MAX = SIXTEEN_BIT_LEVELS - 1
EIGHT_TO_SIXTEEN = SIXTEEN_BIT_MAX // EIGHT_BIT_MAX

#: Pillow modes for single-channel images carrying more than eight bits per
#: sample. ``I`` is 32-bit signed integer; the ``I;16*`` family is unsigned
#: 16-bit in the various byte orders a scanner may produce.
HIGH_DEPTH_GRAY_MODES: frozenset[str] = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})


def is_high_depth_gray(image: Image.Image) -> bool:
    """Report whether ``image`` is single-channel with more than eight bits."""
    return image.mode in HIGH_DEPTH_GRAY_MODES


def to_uint16(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as ``uint16`` without rescaling its values.

    Pillow decodes ``I;16`` to ``uint16`` and ``I`` to ``int32``. Both already
    hold values on the 0-65535 scale, so the conversion clips rather than
    scales; only a genuinely 8-bit array needs :data:`EIGHT_TO_SIXTEEN`.

    ``I`` is 32 bits wide, so a source really carrying more than 16 bits would
    lose its highlights here. That is logged rather than papered over, because
    rescaling by the observed maximum would change the tone curve silently.
    """
    if array.dtype == np.uint16:
        return array
    over = int(np.count_nonzero(array > SIXTEEN_BIT_MAX))
    if over:
        logger.warning(
            "Clipped %d sample(s) above %d while narrowing %s to 16 bits",
            over,
            SIXTEEN_BIT_MAX,
            array.dtype,
        )
    return np.clip(array, 0, SIXTEEN_BIT_MAX).astype(np.uint16)


def to_eight_bit_gray(image: Image.Image) -> Image.Image:
    """Scale a high-depth grayscale image down to 8-bit "L".

    Pillow's own ``convert("L")`` *clips* ``I;16`` instead of scaling it, so
    every sample above 255 came out white. Divide by :data:`EIGHT_TO_SIXTEEN`
    instead, which is exact at both ends of the range.
    """
    scaled = np.rint(to_uint16(np.asarray(image)) / EIGHT_TO_SIXTEEN)
    return Image.fromarray(np.clip(scaled, 0, EIGHT_BIT_MAX).astype(np.uint8), mode="L")


class ImageFormat(str, Enum):
    """Supported image export formats."""

    TIFF = "tiff"
    TIFF_16BIT = "tiff_16bit"
    PNG = "png"
    PNG_16BIT = "png_16bit"
    JPEG = "jpeg"
    JPEG_HIGH = "jpeg_high"
    ORIGINAL = "original"  # Same as input


class ColorMode(str, Enum):
    """Image color modes for processing."""

    GRAYSCALE = "grayscale"
    RGB = "rgb"
    PRESERVE = "preserve"  # Keep original mode


@dataclass
class ExportSettings:
    """Settings for image export."""

    format: ImageFormat = ImageFormat.ORIGINAL
    jpeg_quality: int = 95  # 0-100
    preserve_metadata: bool = True
    preserve_resolution: bool = True
    target_dpi: int | None = None  # Override DPI if set
    compression: str | None = None  # Format-specific compression


@dataclass
class ProcessingResult:
    """Result of image processing operation."""

    image: Image.Image
    original_size: tuple[int, int]
    original_mode: str
    original_format: str | None
    original_dpi: tuple[int, int] | None
    curve_applied: bool
    inverted: bool
    processing_notes: list[str] = field(default_factory=list)

    def get_info(self) -> dict:
        """Get processing info as dictionary."""
        return {
            "size": f"{self.image.size[0]}x{self.image.size[1]}",
            "original_size": f"{self.original_size[0]}x{self.original_size[1]}",
            "mode": self.image.mode,
            "original_mode": self.original_mode,
            "original_format": self.original_format,
            "dpi": self.original_dpi,
            "curve_applied": self.curve_applied,
            "inverted": self.inverted,
            "notes": self.processing_notes,
        }


class ImageProcessor:
    """Process images with calibration curves for digital negative creation.

    Supports:
    - Loading images in various formats
    - Applying calibration curves using lookup tables
    - Creating inverted negatives
    - Exporting in various formats while preserving quality
    """

    def __init__(self, settings: ImagingSettings | None = None) -> None:
        """Initialize the image processor.

        Args:
            settings: Imaging settings; ``None`` reads them from the
                environment. Injecting them keeps the processor testable
                without mutating global configuration.
        """
        self._settings = settings or get_settings().imaging
        # An LRU rather than a plain dict: a long-lived batch processor applies
        # one curve per job, and a 16-bit table is 128 KB, so an unbounded
        # cache grew without limit for the lifetime of the process.
        self._lut_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def load_image(
        self,
        source: str | Path | Image.Image | np.ndarray | bytes,
    ) -> ProcessingResult:
        """Load an image from various sources.

        Args:
            source: Image path, PIL Image, numpy array, or bytes

        Returns:
            ProcessingResult with loaded image and metadata
        """
        if isinstance(source, str | Path | bytes):
            # Paths and byte strings are the two untrusted sources: they arrive
            # from an upload or a user-chosen file. open_image_safely applies
            # the format, pixel-count, frame-count and mode guards from
            # ImageDecodeSettings and fully decodes, so no file handle is left
            # open for the garbage collector to close.
            img = open_image_safely(source)
            original_format = img.format
        elif isinstance(source, Image.Image):
            img = source.copy()
            original_format = getattr(source, "format", None)
        elif isinstance(source, np.ndarray):
            if source.ndim == 2 and source.dtype == np.uint16 and self._settings.preserve_bit_depth:
                # ``astype(np.uint8)`` truncates modulo 256, so level 256 became
                # 0 and a 16-bit array arrived scrambled rather than merely
                # coarsened. Keep the depth instead; Pillow infers "I;16".
                # Copy for the same reason the PIL branch does: ``fromarray``
                # shares the buffer, so a later write to the caller's array
                # would silently change the image already loaded from it.
                img = Image.fromarray(source.copy())
            elif source.ndim == 2:
                img = Image.fromarray(source.astype(np.uint8), mode="L")
            elif source.ndim == 3 and source.shape[2] == 3:
                img = Image.fromarray(source.astype(np.uint8), mode="RGB")
            elif source.ndim == 3 and source.shape[2] == 4:
                img = Image.fromarray(source.astype(np.uint8), mode="RGBA")
            else:
                raise ValueError(f"Unsupported array shape: {source.shape}")
            original_format = None
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Get DPI if available
        dpi = img.info.get("dpi")

        return ProcessingResult(
            image=img,
            original_size=img.size,
            original_mode=img.mode,
            original_format=original_format,
            original_dpi=dpi,
            curve_applied=False,
            inverted=False,
        )

    def apply_curve(
        self,
        result: ProcessingResult,
        curve: CurveData,
        color_mode: ColorMode = ColorMode.PRESERVE,
    ) -> ProcessingResult:
        """Apply a calibration curve to an image.

        Args:
            result: ProcessingResult with image to process
            curve: CurveData with calibration curve
            color_mode: How to handle color (grayscale, rgb, preserve)

        Returns:
            New ProcessingResult with curve applied
        """
        img = result.image

        # Convert to appropriate mode for processing. A high-depth grayscale
        # source is already grayscale, so ColorMode.GRAYSCALE must not push it
        # through "L" and throw away eight bits; ColorMode.RGB is an explicit
        # request for 8-bit colour and is honoured as such.
        if (
            color_mode == ColorMode.GRAYSCALE
            and img.mode not in ("L", "LA")
            and not self._keeps_high_depth(img)
        ):
            img = img.convert("L")
        elif color_mode == ColorMode.RGB and img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Apply the LUT based on image mode. Each table is built inside the
        # branch that uses it, so a 16-bit workload does not also fill the
        # cache with 8-bit tables it never reads.
        if self._keeps_high_depth(img):
            processed = self._apply_lut_grayscale_16(img, self._create_lut_16(curve))
        elif img.mode == "L":
            processed = self._apply_lut_grayscale(img, self._create_lut(curve))
        elif img.mode == "LA":
            # Grayscale with alpha
            l_channel = img.split()[0]
            a_channel = img.split()[1]
            processed_l = self._apply_lut_grayscale(l_channel, self._create_lut(curve))
            processed = Image.merge("LA", (processed_l, a_channel))
        elif img.mode == "RGB":
            processed = self._apply_lut_rgb(img, self._create_lut(curve))
        elif img.mode == "RGBA":
            # RGB with alpha
            rgb = img.convert("RGB")
            a_channel = img.split()[3]
            processed_rgb = self._apply_lut_rgb(rgb, self._create_lut(curve))
            processed = processed_rgb.copy()
            processed.putalpha(a_channel)
        else:
            # Try to convert to RGB first
            try:
                rgb = img.convert("RGB")
                processed = self._apply_lut_rgb(rgb, self._create_lut(curve))
            except Exception as e:
                raise ValueError(f"Unsupported image mode: {img.mode}") from e

        notes = list(result.processing_notes)
        notes.append(f"Applied curve: {curve.name}")

        return ProcessingResult(
            image=processed,
            original_size=result.original_size,
            original_mode=result.original_mode,
            original_format=result.original_format,
            original_dpi=result.original_dpi,
            curve_applied=True,
            inverted=result.inverted,
            processing_notes=notes,
        )

    def apply_curves_per_channel(
        self,
        result: ProcessingResult,
        curves: dict[str, CurveData],
    ) -> ProcessingResult:
        """Apply different curves to each RGB channel.

        This method allows independent curve application to R, G, B channels,
        useful for scanner calibration or color correction workflows.

        Args:
            result: ProcessingResult with RGB/RGBA image to process
            curves: Dictionary with keys 'R', 'G', 'B' mapping to CurveData.
                    Missing channels use identity (no modification).

        Returns:
            New ProcessingResult with per-channel curves applied

        Raises:
            ValueError: If image is not RGB or RGBA mode
        """
        img = result.image

        # Validate input mode
        if img.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Per-channel curves require RGB/RGBA image, got {img.mode}")

        # Extract alpha if present
        has_alpha = img.mode == "RGBA"
        if has_alpha:
            a_channel = img.split()[3]
            img = img.convert("RGB")

        # Convert to numpy array for processing
        arr = np.array(img)

        # Validate channel dimensions
        self._validate_channel_dimensions(arr, expected_channels=3)

        # Create LUTs for each channel
        luts = {}
        for channel in ("R", "G", "B"):
            if channel in curves and curves[channel] is not None:
                luts[channel] = self._create_lut(curves[channel])
            else:
                # Identity LUT (no change)
                luts[channel] = np.arange(256, dtype=np.uint8)

        # Apply LUTs per channel
        processed = np.zeros_like(arr)
        channel_map = {"R": 0, "G": 1, "B": 2}
        for channel, idx in channel_map.items():
            processed[:, :, idx] = luts[channel][arr[:, :, idx]]

        # Convert back to PIL Image
        processed_img = Image.fromarray(processed, mode="RGB")

        # Restore alpha if present
        if has_alpha:
            processed_img.putalpha(a_channel)

        notes = list(result.processing_notes)
        applied_channels = [ch for ch in ("R", "G", "B") if ch in curves and curves[ch] is not None]
        if applied_channels:
            notes.append(f"Applied per-channel curves: {', '.join(applied_channels)}")
        else:
            notes.append("Applied per-channel curves: none")

        return ProcessingResult(
            image=processed_img,
            original_size=result.original_size,
            original_mode=result.original_mode,
            original_format=result.original_format,
            original_dpi=result.original_dpi,
            curve_applied=True,
            inverted=result.inverted,
            processing_notes=notes,
        )

    def _validate_channel_dimensions(
        self,
        arr: np.ndarray,
        expected_channels: int | None = None,
    ) -> None:
        """Validate that all channels have consistent dimensions.

        Args:
            arr: Image array to validate (H, W) or (H, W, C)
            expected_channels: Expected number of channels (None to skip check)

        Raises:
            ValueError: If validation fails
        """
        if arr.ndim == 2:
            # Grayscale - single channel, nothing to validate
            return

        if arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

        height, width, channels = arr.shape

        if expected_channels is not None and channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {channels}")

        # Validate no channel is empty (all zeros)
        for c in range(channels):
            channel_data = arr[:, :, c]
            if channel_data.max() == 0 and channel_data.min() == 0:
                # Channel is all zeros - might be intentional, just log
                pass

        # Validate all channels have same dimensions (already guaranteed by numpy)
        # This check is mostly for documentation/clarity

    def validate_image_channels(
        self,
        result: ProcessingResult,
        require_uniform: bool = True,
    ) -> dict:
        """Validate image channel consistency and return diagnostics.

        Args:
            result: ProcessingResult to validate
            require_uniform: If True, raises error on inconsistent channels

        Returns:
            Dictionary with channel statistics:
            - mode: Image mode (L, RGB, RGBA, etc.)
            - channels: Number of channels
            - shape: Image dimensions (width, height)
            - per_channel_stats: List of (min, max, mean, std) per channel
            - all_channels_processed: True if no channel is constant/flat
        """
        img = result.image
        arr = np.array(img)

        stats = {
            "mode": img.mode,
            "shape": img.size,  # (width, height)
        }

        if arr.ndim == 2:
            stats["channels"] = 1
            stats["per_channel_stats"] = [
                {
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                }
            ]
            stats["all_channels_processed"] = arr.max() != arr.min()
        else:
            stats["channels"] = arr.shape[2]
            stats["per_channel_stats"] = []
            all_processed = True

            for c in range(arr.shape[2]):
                channel = arr[:, :, c]
                channel_stats = {
                    "min": int(channel.min()),
                    "max": int(channel.max()),
                    "mean": float(channel.mean()),
                    "std": float(channel.std()),
                }
                stats["per_channel_stats"].append(channel_stats)

                # Check if channel appears unprocessed (all same value)
                if channel.max() == channel.min():
                    all_processed = False

            stats["all_channels_processed"] = all_processed

        if require_uniform and not stats["all_channels_processed"]:
            # Check if any channel is constant (might indicate unprocessed)
            constant_channels = [
                i for i, s in enumerate(stats["per_channel_stats"]) if s["min"] == s["max"]
            ]
            if constant_channels:
                raise ValueError(
                    f"Channels {constant_channels} appear unprocessed (constant value)"
                )

        return stats

    def invert(self, result: ProcessingResult) -> ProcessingResult:
        """Invert an image (create negative).

        Args:
            result: ProcessingResult with image to invert

        Returns:
            New ProcessingResult with inverted image
        """
        img = result.image

        # Handle different modes
        if self._keeps_high_depth(img):
            # Inverting against 255 here would have clamped every 16-bit sample
            # to black; the mode was silently converted to RGB instead, which
            # cost the depth. Invert against the 16-bit maximum in place.
            high_depth_arr = to_uint16(np.asarray(img))
            inverted = Image.fromarray((SIXTEEN_BIT_MAX - high_depth_arr).astype(np.uint16))
        elif img.mode == "L":
            arr = np.array(img)
            inverted_arr = 255 - arr
            inverted = Image.fromarray(inverted_arr.astype(np.uint8), mode="L")
        elif img.mode == "LA":
            l_channel, a_channel = img.split()
            l_arr = np.array(l_channel)
            inverted_l = Image.fromarray((255 - l_arr).astype(np.uint8), mode="L")
            inverted = Image.merge("LA", (inverted_l, a_channel))
        elif img.mode == "RGB":
            arr = np.array(img)
            inverted_arr = 255 - arr
            inverted = Image.fromarray(inverted_arr.astype(np.uint8), mode="RGB")
        elif img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb_arr = np.array(rgb)
            inverted_rgb = Image.fromarray((255 - rgb_arr).astype(np.uint8), mode="RGB")
            inverted = inverted_rgb.copy()
            inverted.putalpha(a)
        else:
            # Try to handle other modes
            try:
                rgb = img.convert("RGB")
                arr = np.array(rgb)
                inverted_arr = 255 - arr
                inverted = Image.fromarray(inverted_arr.astype(np.uint8), mode="RGB")
            except Exception as e:
                raise ValueError(f"Cannot invert image mode: {img.mode}") from e

        notes = list(result.processing_notes)
        notes.append("Image inverted (negative created)")

        return ProcessingResult(
            image=inverted,
            original_size=result.original_size,
            original_mode=result.original_mode,
            original_format=result.original_format,
            original_dpi=result.original_dpi,
            curve_applied=result.curve_applied,
            inverted=not result.inverted,  # Toggle inversion state
            processing_notes=notes,
        )

    def create_digital_negative(
        self,
        source: str | Path | Image.Image | np.ndarray | bytes,
        curve: CurveData | None = None,
        invert: bool = True,
        color_mode: ColorMode = ColorMode.GRAYSCALE,
    ) -> ProcessingResult:
        """Create a digital negative from an image.

        Complete workflow: load → apply curve → invert

        Args:
            source: Image source
            curve: Optional calibration curve to apply
            invert: Whether to invert the image
            color_mode: Color mode for processing

        Returns:
            ProcessingResult with digital negative
        """
        result = self.load_image(source)

        # Convert to appropriate color mode. High-depth grayscale is grayscale
        # already, so converting it here would defeat the 16-bit path below.
        if (
            color_mode == ColorMode.GRAYSCALE
            and result.image.mode not in ("L", "LA")
            and not self._keeps_high_depth(result.image)
        ):
            result = ProcessingResult(
                image=result.image.convert("L"),
                original_size=result.original_size,
                original_mode=result.original_mode,
                original_format=result.original_format,
                original_dpi=result.original_dpi,
                curve_applied=result.curve_applied,
                inverted=result.inverted,
                processing_notes=result.processing_notes + ["Converted to grayscale"],
            )

        # Apply curve if provided
        if curve is not None:
            result = self.apply_curve(result, curve, color_mode)

        # Invert if requested
        if invert:
            result = self.invert(result)

        return result

    def preview_curve_effect(
        self,
        source: str | Path | Image.Image | np.ndarray | bytes,
        curve: CurveData,
        color_mode: ColorMode = ColorMode.PRESERVE,
        thumbnail_size: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        """Preview the effect of a curve on an image.

        Args:
            source: Image source
            curve: Calibration curve to preview
            color_mode: Color mode for processing
            thumbnail_size: Optional size to resize for faster preview

        Returns:
            Tuple of (original_image, processed_image)
        """
        result = self.load_image(source)

        original = result.image.copy()
        if thumbnail_size:
            original.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            # Resize the processing image too
            result = ProcessingResult(
                image=result.image.copy(),
                original_size=result.original_size,
                original_mode=result.original_mode,
                original_format=result.original_format,
                original_dpi=result.original_dpi,
                curve_applied=False,
                inverted=False,
            )
            result.image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

        processed_result = self.apply_curve(result, curve, color_mode)

        return original, processed_result.image

    def export(
        self,
        result: ProcessingResult,
        output_path: str | Path,
        settings: ExportSettings | None = None,
    ) -> Path:
        """Export processed image to file.

        Args:
            result: ProcessingResult to export
            output_path: Output file path
            settings: Export settings

        Returns:
            Path to exported file
        """
        settings = settings or ExportSettings()
        output_path = Path(output_path)

        img = result.image

        # Determine format
        if settings.format == ImageFormat.ORIGINAL:
            # Use original format or infer from path
            fmt = result.original_format or output_path.suffix.lstrip(".").upper()
            if fmt.upper() == "JPG":
                fmt = "JPEG"
        else:
            fmt = settings.format.value.upper()
            if fmt.endswith("_16BIT"):
                fmt = fmt.replace("_16BIT", "")
            if fmt == "JPEG_HIGH":
                fmt = "JPEG"

        # Handle 16-bit export
        is_16bit = settings.format in (ImageFormat.TIFF_16BIT, ImageFormat.PNG_16BIT)

        # Build save kwargs
        save_kwargs: dict[str, Any] = {}

        # Set DPI
        if settings.preserve_resolution and result.original_dpi:
            save_kwargs["dpi"] = result.original_dpi
        elif settings.target_dpi:
            save_kwargs["dpi"] = (settings.target_dpi, settings.target_dpi)

        # Format-specific settings
        if fmt == "JPEG":
            quality = 98 if settings.format == ImageFormat.JPEG_HIGH else settings.jpeg_quality
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0  # Best quality
            # JPEG carries neither alpha nor more than eight bits per sample.
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB" if img.mode == "RGBA" else "L")
            elif is_high_depth_gray(img):
                img = to_eight_bit_gray(img)

        elif fmt == "TIFF":
            if settings.compression:
                save_kwargs["compression"] = settings.compression
            else:
                save_kwargs["compression"] = "tiff_lzw"

        elif fmt == "PNG":
            save_kwargs["compress_level"] = 6  # Balanced compression

        # Handle 16-bit conversion
        if is_16bit:
            arr = self._as_16bit(np.asarray(img))
            if fmt == "TIFF":
                # Save 16-bit TIFF
                self._save_16bit_tiff(arr, output_path, save_kwargs)
                return output_path
            elif fmt == "PNG":
                # PIL can handle 16-bit PNG for grayscale. The test is the array
                # shape, not the mode: a curve applied at 16 bits leaves "I;16",
                # which is just as single-channel as "L".
                if arr.ndim == 2:
                    # Pillow infers "I;16" from the uint16 dtype; the explicit
                    # ``mode`` argument is deprecated and removed in Pillow 13.
                    img = Image.fromarray(arr)
                else:
                    # For RGB, need to use array directly
                    pass  # Fall through to standard save

        # Standard save
        img.save(output_path, format=fmt, **save_kwargs)

        return output_path

    def export_to_bytes(
        self,
        result: ProcessingResult,
        settings: ExportSettings | None = None,
    ) -> tuple[bytes, str]:
        """Export processed image to bytes.

        Args:
            result: ProcessingResult to export
            settings: Export settings

        Returns:
            Tuple of (image_bytes, format_extension)
        """
        settings = settings or ExportSettings()
        img = result.image

        # Determine format
        if settings.format == ImageFormat.ORIGINAL:
            fmt = result.original_format or "PNG"
        else:
            fmt = settings.format.value.upper()
            if fmt.endswith("_16BIT"):
                fmt = fmt.replace("_16BIT", "")
            if fmt == "JPEG_HIGH":
                fmt = "JPEG"

        # Build extension map
        ext_map = {
            "TIFF": ".tiff",
            "PNG": ".png",
            "JPEG": ".jpg",
        }
        ext = ext_map.get(fmt, ".png")

        # Build save kwargs
        save_kwargs: dict[str, Any] = {}

        if fmt == "JPEG":
            quality = 98 if settings.format == ImageFormat.JPEG_HIGH else settings.jpeg_quality
            save_kwargs["quality"] = quality
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB" if img.mode == "RGBA" else "L")
            elif is_high_depth_gray(img):
                img = to_eight_bit_gray(img)

        elif fmt == "TIFF":
            save_kwargs["compression"] = "tiff_lzw"

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format=fmt, **save_kwargs)
        buffer.seek(0)

        return buffer.read(), ext

    @staticmethod
    def _lut_cache_key(curve: CurveData) -> str:
        """Identify a curve by its contents, not by its label.

        The key used to be the name and point count. Generated curves default
        to the same name and always carry 256 points, so a second curve
        silently reused the first one's table: the user exported a negative
        with the previous calibration and had no way to tell. Hashing the
        values makes two curves share an entry only when they are equal.
        """
        digest = hashlib.blake2b(digest_size=16)
        for values in (curve.input_values, curve.output_values):
            digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
            digest.update(b"|")
        return digest.hexdigest()

    def _cached_lut(self, cache_key: str) -> np.ndarray | None:
        """Return a cached table, marking it as most recently used."""
        lut = self._lut_cache.get(cache_key)
        if lut is not None:
            self._lut_cache.move_to_end(cache_key)
        return lut

    def _store_lut(self, cache_key: str, lut: np.ndarray) -> None:
        """Cache ``lut``, evicting the least recently used table when full."""
        self._lut_cache[cache_key] = lut
        self._lut_cache.move_to_end(cache_key)
        while len(self._lut_cache) > self._settings.lut_cache_entries:
            evicted, _ = self._lut_cache.popitem(last=False)
            logger.debug(
                "Evicted LUT %s (cache limit %d)", evicted[:16], self._settings.lut_cache_entries
            )

    def _build_lut(self, curve: CurveData, levels: int, dtype: np.dtype | type) -> np.ndarray:
        """Interpolate ``curve`` onto a table of ``levels`` entries.

        One builder serves both depths so the 8-bit and 16-bit tables cannot
        drift apart in rounding or clipping behaviour.

        Args:
            curve: CurveData with input/output values in 0..1.
            levels: Number of table entries (256 for 8-bit, 65536 for 16-bit).
            dtype: Integer type of the table entries.

        Returns:
            NumPy array of ``levels`` output values.
        """
        cache_key = f"{levels}:{self._lut_cache_key(curve)}"
        cached = self._cached_lut(cache_key)
        if cached is not None:
            logger.debug("LUT cache hit for curve %s at %d levels", curve.name, levels)
            return cached

        input_vals = np.asarray(curve.input_values, dtype=np.float64)
        output_vals = np.asarray(curve.output_values, dtype=np.float64)

        x_lut = np.linspace(0.0, 1.0, levels)
        y_lut = np.interp(x_lut, input_vals, output_vals)

        # Round rather than truncate: casting discards the fraction, which
        # biases every entry by up to half a code value and maps 0.5 to 127
        # instead of 128.
        lut = np.rint(np.clip(y_lut, 0.0, 1.0) * (levels - 1)).astype(dtype)

        self._store_lut(cache_key, lut)
        logger.debug("Built %d-entry LUT for curve %s", levels, curve.name)
        return lut

    def _create_lut(self, curve: CurveData) -> np.ndarray:
        """Create the 256-entry 8-bit lookup table for ``curve``."""
        return self._build_lut(curve, EIGHT_BIT_LEVELS, np.uint8)

    def _create_lut_16(self, curve: CurveData) -> np.ndarray:
        """Create the 65536-entry 16-bit lookup table for ``curve``.

        A 16-bit scan carries 65536 levels. Passing it through the 256-entry
        table quantises it to 8 bits, which the export then scaled back up, so
        a "16-bit" negative held fewer distinct levels than the source did.
        Highlight banding is precisely what a 16-bit negative exists to avoid.
        """
        return self._build_lut(curve, SIXTEEN_BIT_LEVELS, np.uint16)

    def _keeps_high_depth(self, img: Image.Image) -> bool:
        """Report whether ``img`` should be processed at more than eight bits."""
        return self._settings.preserve_bit_depth and is_high_depth_gray(img)

    @staticmethod
    def _as_16bit(arr: np.ndarray) -> np.ndarray:
        """Put ``arr`` on the 16-bit scale, scaling only genuinely 8-bit data.

        An array that already holds 0-65535 values must pass through untouched:
        multiplying it by :data:`EIGHT_TO_SIXTEEN` would overflow and wrap.
        """
        if arr.dtype.itemsize > 1 and np.issubdtype(arr.dtype, np.integer):
            return to_uint16(arr)
        return arr.astype(np.uint16) * EIGHT_TO_SIXTEEN

    def _apply_lut_grayscale(self, img: Image.Image, lut: np.ndarray) -> Image.Image:
        """Apply LUT to grayscale image.

        Args:
            img: Grayscale PIL Image
            lut: 256-entry lookup table

        Returns:
            Processed grayscale image
        """
        arr = np.array(img)
        processed = lut[arr]
        return Image.fromarray(processed, mode="L")

    def _apply_lut_grayscale_16(self, img: Image.Image, lut: np.ndarray) -> Image.Image:
        """Apply a 16-bit LUT to a high-depth grayscale image.

        Args:
            img: PIL Image in one of :data:`HIGH_DEPTH_GRAY_MODES`.
            lut: 65536-entry lookup table.

        Returns:
            Processed image in "I;16", inferred by Pillow from the dtype.
        """
        arr = to_uint16(np.asarray(img))
        return Image.fromarray(lut[arr])

    def _apply_lut_rgb(self, img: Image.Image, lut: np.ndarray) -> Image.Image:
        """Apply LUT to RGB image (same curve to all channels).

        Args:
            img: RGB PIL Image
            lut: 256-entry lookup table

        Returns:
            Processed RGB image
        """
        arr = np.array(img)
        processed = lut[arr]
        return Image.fromarray(processed, mode="RGB")

    def _save_16bit_tiff(
        self,
        arr: np.ndarray,
        path: Path,
        kwargs: dict,
    ) -> None:
        """Save 16-bit TIFF using tifffile (if available) or PIL fallback.

        Args:
            arr: 16-bit numpy array (H, W) for grayscale or (H, W, 3) for RGB
            path: Output path
            kwargs: Additional save arguments (dpi, compression, etc.)
        """
        if arr.ndim == 2:
            # Grayscale - PIL handles this fine ("I;16" is inferred from uint16;
            # the explicit ``mode`` argument is deprecated and removed in Pillow 13)
            img = Image.fromarray(arr)
            img.save(path, format="TIFF", **kwargs)
        elif HAS_TIFFFILE:
            # RGB 16-bit - use tifffile for proper support
            # Extract DPI for resolution tags
            dpi = kwargs.get("dpi")
            resolution = None
            if dpi:
                # tifffile expects resolution as (value, value) in pixels per cm or inch
                resolution = (dpi[0], dpi[1])

            compression = kwargs.get("compression", "lzw")
            # Map PIL compression names to tifffile
            compression_map = {
                "tiff_lzw": "lzw",
                "tiff_deflate": "deflate",
                "tiff_adobe_deflate": "deflate",
                None: None,
            }
            tiff_compression = compression_map.get(compression, compression)

            # Try with compression, fallback to no compression if imagecodecs unavailable
            try:
                tifffile.imwrite(
                    path,
                    arr,
                    photometric="rgb",
                    compression=tiff_compression,
                    resolution=resolution,
                    resolutionunit=2 if resolution else None,  # 2 = inch
                )
            except KeyError:
                # imagecodecs not available for compression, use no compression
                tifffile.imwrite(
                    path,
                    arr,
                    photometric="rgb",
                    compression=None,
                    resolution=resolution,
                    resolutionunit=2 if resolution else None,
                )
        else:
            # Fallback: save as 8-bit if tifffile not available
            arr_8bit = (arr / 257).astype(np.uint8)
            img = Image.fromarray(arr_8bit, mode="RGB")
            img.save(path, format="TIFF", **kwargs)

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported image formats for import."""
        return [
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".tif",
            ".bmp",
            ".gif",
            ".webp",
            ".ppm",
            ".pgm",
        ]

    @staticmethod
    def get_export_formats() -> list[tuple[str, str]]:
        """Get list of export formats with descriptions."""
        return [
            ("tiff", "TIFF (Lossless)"),
            ("tiff_16bit", "TIFF 16-bit (High Quality)"),
            ("png", "PNG (Lossless)"),
            ("png_16bit", "PNG 16-bit"),
            ("jpeg", "JPEG (Standard Quality)"),
            ("jpeg_high", "JPEG (High Quality)"),
            ("original", "Same as Original"),
        ]
