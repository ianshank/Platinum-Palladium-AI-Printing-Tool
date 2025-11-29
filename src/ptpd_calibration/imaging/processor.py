"""
Image processor for digital negative creation.

Applies calibration curves to images, creates inverted negatives,
and exports in various formats while preserving resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import io

import numpy as np
from PIL import Image

from ptpd_calibration.core.models import CurveData


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
    target_dpi: Optional[int] = None  # Override DPI if set
    compression: Optional[str] = None  # Format-specific compression


@dataclass
class ProcessingResult:
    """Result of image processing operation."""

    image: Image.Image
    original_size: tuple[int, int]
    original_mode: str
    original_format: Optional[str]
    original_dpi: Optional[tuple[int, int]]
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

    def __init__(self):
        """Initialize the image processor."""
        self._lut_cache: dict[str, np.ndarray] = {}

    def load_image(
        self,
        source: Union[str, Path, Image.Image, np.ndarray, bytes],
    ) -> ProcessingResult:
        """Load an image from various sources.

        Args:
            source: Image path, PIL Image, numpy array, or bytes

        Returns:
            ProcessingResult with loaded image and metadata
        """
        if isinstance(source, (str, Path)):
            img = Image.open(source)
            original_format = img.format
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
            original_format = img.format
        elif isinstance(source, Image.Image):
            img = source.copy()
            original_format = getattr(source, "format", None)
        elif isinstance(source, np.ndarray):
            if source.ndim == 2:
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

        # Convert to appropriate mode for processing
        if color_mode == ColorMode.GRAYSCALE:
            if img.mode not in ("L", "LA"):
                img = img.convert("L")
        elif color_mode == ColorMode.RGB:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

        # Create lookup table from curve
        lut = self._create_lut(curve)

        # Apply LUT based on image mode
        if img.mode == "L":
            processed = self._apply_lut_grayscale(img, lut)
        elif img.mode == "LA":
            # Grayscale with alpha
            l_channel = img.split()[0]
            a_channel = img.split()[1]
            processed_l = self._apply_lut_grayscale(l_channel, lut)
            processed = Image.merge("LA", (processed_l, a_channel))
        elif img.mode == "RGB":
            processed = self._apply_lut_rgb(img, lut)
        elif img.mode == "RGBA":
            # RGB with alpha
            rgb = img.convert("RGB")
            a_channel = img.split()[3]
            processed_rgb = self._apply_lut_rgb(rgb, lut)
            processed = processed_rgb.copy()
            processed.putalpha(a_channel)
        else:
            # Try to convert to RGB first
            try:
                rgb = img.convert("RGB")
                processed = self._apply_lut_rgb(rgb, lut)
            except Exception:
                raise ValueError(f"Unsupported image mode: {img.mode}")

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

    def invert(self, result: ProcessingResult) -> ProcessingResult:
        """Invert an image (create negative).

        Args:
            result: ProcessingResult with image to invert

        Returns:
            New ProcessingResult with inverted image
        """
        img = result.image

        # Handle different modes
        if img.mode == "L":
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
            except Exception:
                raise ValueError(f"Cannot invert image mode: {img.mode}")

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
        source: Union[str, Path, Image.Image, np.ndarray, bytes],
        curve: Optional[CurveData] = None,
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

        # Convert to appropriate color mode
        if color_mode == ColorMode.GRAYSCALE:
            if result.image.mode not in ("L", "LA"):
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
        source: Union[str, Path, Image.Image, np.ndarray, bytes],
        curve: CurveData,
        color_mode: ColorMode = ColorMode.PRESERVE,
        thumbnail_size: Optional[tuple[int, int]] = None,
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
        output_path: Union[str, Path],
        settings: Optional[ExportSettings] = None,
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
        save_kwargs = {}

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
            # JPEG doesn't support alpha
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB" if img.mode == "RGBA" else "L")

        elif fmt == "TIFF":
            if settings.compression:
                save_kwargs["compression"] = settings.compression
            else:
                save_kwargs["compression"] = "tiff_lzw"

        elif fmt == "PNG":
            save_kwargs["compress_level"] = 6  # Balanced compression

        # Handle 16-bit conversion
        if is_16bit:
            arr = np.array(img).astype(np.uint16) * 257  # Scale 8-bit to 16-bit
            if fmt == "TIFF":
                # Save 16-bit TIFF
                self._save_16bit_tiff(arr, output_path, save_kwargs)
                return output_path
            elif fmt == "PNG":
                # PIL can handle 16-bit PNG for grayscale
                if img.mode == "L":
                    img = Image.fromarray(arr, mode="I;16")
                else:
                    # For RGB, need to use array directly
                    pass  # Fall through to standard save

        # Standard save
        img.save(output_path, format=fmt, **save_kwargs)

        return output_path

    def export_to_bytes(
        self,
        result: ProcessingResult,
        settings: Optional[ExportSettings] = None,
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
        save_kwargs = {}

        if fmt == "JPEG":
            quality = 98 if settings.format == ImageFormat.JPEG_HIGH else settings.jpeg_quality
            save_kwargs["quality"] = quality
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB" if img.mode == "RGBA" else "L")

        elif fmt == "TIFF":
            save_kwargs["compression"] = "tiff_lzw"

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format=fmt, **save_kwargs)
        buffer.seek(0)

        return buffer.read(), ext

    def _create_lut(self, curve: CurveData) -> np.ndarray:
        """Create 256-entry lookup table from curve.

        Args:
            curve: CurveData with input/output values

        Returns:
            NumPy array of 256 output values
        """
        # Check cache
        cache_key = f"{curve.name}_{len(curve.input_values)}"
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        # Interpolate curve to 256 points
        input_vals = np.array(curve.input_values)
        output_vals = np.array(curve.output_values)

        # Create LUT for all 256 possible input values
        x_lut = np.linspace(0, 1, 256)
        y_lut = np.interp(x_lut, input_vals, output_vals)

        # Convert to 0-255 range
        lut = (np.clip(y_lut, 0, 1) * 255).astype(np.uint8)

        # Cache and return
        self._lut_cache[cache_key] = lut
        return lut

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
        """Save 16-bit TIFF using PIL.

        Args:
            arr: 16-bit numpy array
            path: Output path
            kwargs: Additional save arguments
        """
        if arr.ndim == 2:
            # Grayscale
            img = Image.fromarray(arr, mode="I;16")
        else:
            # RGB - need to handle specially
            # PIL doesn't support 16-bit RGB directly, so we'll save as 8-bit
            arr_8bit = (arr / 257).astype(np.uint8)
            img = Image.fromarray(arr_8bit, mode="RGB")

        img.save(path, format="TIFF", **kwargs)

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported image formats for import."""
        return [
            ".jpg", ".jpeg", ".png", ".tiff", ".tif",
            ".bmp", ".gif", ".webp", ".ppm", ".pgm",
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
