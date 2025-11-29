"""
Unit tests for image processor module.

Tests curve application, inversion, and export functionality.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging import (
    ImageProcessor,
    ImageFormat,
    ExportSettings,
    ProcessingResult,
)
from ptpd_calibration.imaging.processor import ColorMode


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_all_formats_defined(self):
        """All expected formats should be defined."""
        assert ImageFormat.TIFF
        assert ImageFormat.TIFF_16BIT
        assert ImageFormat.PNG
        assert ImageFormat.PNG_16BIT
        assert ImageFormat.JPEG
        assert ImageFormat.JPEG_HIGH
        assert ImageFormat.ORIGINAL

    def test_format_values(self):
        """Format values should be valid strings."""
        assert ImageFormat.TIFF.value == "tiff"
        assert ImageFormat.PNG.value == "png"
        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.ORIGINAL.value == "original"


class TestColorMode:
    """Tests for ColorMode enum."""

    def test_color_mode_values(self):
        """Color mode values should be valid."""
        assert ColorMode.GRAYSCALE.value == "grayscale"
        assert ColorMode.RGB.value == "rgb"
        assert ColorMode.PRESERVE.value == "preserve"


class TestExportSettings:
    """Tests for ExportSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = ExportSettings()
        assert settings.format == ImageFormat.ORIGINAL
        assert settings.jpeg_quality == 95
        assert settings.preserve_metadata is True
        assert settings.preserve_resolution is True

    def test_custom_settings(self):
        """Custom settings should be applied."""
        settings = ExportSettings(
            format=ImageFormat.JPEG_HIGH,
            jpeg_quality=100,
            preserve_metadata=False,
            target_dpi=300,
        )
        assert settings.format == ImageFormat.JPEG_HIGH
        assert settings.jpeg_quality == 100
        assert settings.preserve_metadata is False
        assert settings.target_dpi == 300


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample processing result."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        return ProcessingResult(
            image=img,
            original_size=(100, 100),
            original_mode="RGB",
            original_format="PNG",
            original_dpi=(300, 300),
            curve_applied=True,
            inverted=False,
            processing_notes=["Test note"],
        )

    def test_get_info(self, sample_result):
        """get_info should return valid dictionary."""
        info = sample_result.get_info()
        assert "size" in info
        assert "original_size" in info
        assert "mode" in info
        assert "curve_applied" in info
        assert "inverted" in info
        assert "notes" in info

    def test_info_values(self, sample_result):
        """Info values should match result."""
        info = sample_result.get_info()
        assert info["size"] == "100x100"
        assert info["original_mode"] == "RGB"
        assert info["curve_applied"] is True
        assert info["inverted"] is False


class TestImageProcessor:
    """Tests for ImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create image processor."""
        return ImageProcessor()

    @pytest.fixture
    def grayscale_image(self):
        """Create a grayscale test image."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        # Create gradient
        for i in range(100):
            arr[i, :] = int(i * 2.55)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture
    def rgb_image(self):
        """Create an RGB test image."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create color gradient
        for i in range(100):
            arr[i, :, 0] = int(i * 2.55)  # Red
            arr[i, :, 1] = int((100 - i) * 2.55)  # Green
            arr[i, :, 2] = 128  # Blue
        return Image.fromarray(arr, mode="RGB")

    @pytest.fixture
    def linear_curve(self):
        """Create a linear curve (no change)."""
        return CurveData(
            name="Linear",
            input_values=[i / 10 for i in range(11)],
            output_values=[i / 10 for i in range(11)],
        )

    @pytest.fixture
    def contrast_curve(self):
        """Create an S-curve for contrast."""
        inputs = [i / 10 for i in range(11)]
        # S-curve formula
        outputs = [0.5 + 0.5 * np.tanh(2 * (x - 0.5)) for x in inputs]
        outputs = [(o - min(outputs)) / (max(outputs) - min(outputs)) for o in outputs]
        return CurveData(
            name="Contrast",
            input_values=inputs,
            output_values=outputs,
        )

    def test_load_image_from_pil(self, processor, grayscale_image):
        """Load image from PIL Image."""
        result = processor.load_image(grayscale_image)
        assert result.image is not None
        assert result.original_size == (100, 100)
        assert result.original_mode == "L"
        assert result.curve_applied is False
        assert result.inverted is False

    def test_load_image_from_numpy(self, processor):
        """Load image from numpy array."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        result = processor.load_image(arr)
        assert result.image is not None
        assert result.original_size == (50, 50)
        assert result.original_mode == "L"

    def test_load_image_rgb_from_numpy(self, processor):
        """Load RGB image from numpy array."""
        arr = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = processor.load_image(arr)
        assert result.image is not None
        assert result.original_mode == "RGB"

    def test_load_image_from_bytes(self, processor, grayscale_image):
        """Load image from bytes."""
        buffer = io.BytesIO()
        grayscale_image.save(buffer, format="PNG")
        buffer.seek(0)

        result = processor.load_image(buffer.getvalue())
        assert result.image is not None
        assert result.original_size == (100, 100)

    def test_apply_linear_curve_no_change(self, processor, grayscale_image, linear_curve):
        """Linear curve should not change image significantly."""
        result = processor.load_image(grayscale_image)
        processed = processor.apply_curve(result, linear_curve)

        assert processed.curve_applied is True
        assert processed.image.size == grayscale_image.size

        # Values should be approximately the same
        orig_arr = np.array(grayscale_image)
        proc_arr = np.array(processed.image)
        assert np.allclose(orig_arr, proc_arr, atol=2)

    def test_apply_contrast_curve(self, processor, grayscale_image, contrast_curve):
        """Contrast curve should modify image."""
        result = processor.load_image(grayscale_image)
        processed = processor.apply_curve(result, contrast_curve)

        assert processed.curve_applied is True

        # Midtones should be preserved, but darks darker and lights lighter
        orig_arr = np.array(grayscale_image)
        proc_arr = np.array(processed.image)

        # Not exactly the same
        assert not np.allclose(orig_arr, proc_arr, atol=5)

    def test_apply_curve_rgb(self, processor, rgb_image, linear_curve):
        """Curve should apply to RGB image."""
        result = processor.load_image(rgb_image)
        processed = processor.apply_curve(result, linear_curve, ColorMode.RGB)

        assert processed.curve_applied is True
        assert processed.image.mode == "RGB"

    def test_apply_curve_grayscale_conversion(self, processor, rgb_image, linear_curve):
        """RGB image should convert to grayscale when requested."""
        result = processor.load_image(rgb_image)
        processed = processor.apply_curve(result, linear_curve, ColorMode.GRAYSCALE)

        assert processed.image.mode == "L"

    def test_invert_grayscale(self, processor, grayscale_image):
        """Inverting grayscale should flip values."""
        result = processor.load_image(grayscale_image)
        inverted = processor.invert(result)

        assert inverted.inverted is True

        orig_arr = np.array(grayscale_image)
        inv_arr = np.array(inverted.image)

        # Check inversion
        assert np.allclose(inv_arr, 255 - orig_arr)

    def test_invert_rgb(self, processor, rgb_image):
        """Inverting RGB should flip all channels."""
        result = processor.load_image(rgb_image)
        inverted = processor.invert(result)

        assert inverted.inverted is True

        orig_arr = np.array(rgb_image)
        inv_arr = np.array(inverted.image)

        assert np.allclose(inv_arr, 255 - orig_arr)

    def test_double_invert_restores_original(self, processor, grayscale_image):
        """Double inversion should restore original."""
        result = processor.load_image(grayscale_image)
        inverted1 = processor.invert(result)
        inverted2 = processor.invert(inverted1)

        # inverted twice = not inverted
        assert inverted2.inverted is False

        orig_arr = np.array(grayscale_image)
        double_arr = np.array(inverted2.image)
        assert np.allclose(orig_arr, double_arr)

    def test_create_digital_negative(self, processor, grayscale_image, linear_curve):
        """Create complete digital negative."""
        result = processor.create_digital_negative(
            grayscale_image,
            curve=linear_curve,
            invert=True,
            color_mode=ColorMode.GRAYSCALE,
        )

        assert result.curve_applied is True
        assert result.inverted is True
        assert result.image.mode == "L"

    def test_create_digital_negative_no_curve(self, processor, grayscale_image):
        """Digital negative without curve (invert only)."""
        result = processor.create_digital_negative(
            grayscale_image,
            curve=None,
            invert=True,
        )

        assert result.curve_applied is False
        assert result.inverted is True

    def test_create_digital_negative_no_invert(self, processor, grayscale_image, linear_curve):
        """Digital negative with curve but no inversion."""
        result = processor.create_digital_negative(
            grayscale_image,
            curve=linear_curve,
            invert=False,
        )

        assert result.curve_applied is True
        assert result.inverted is False

    def test_preview_curve_effect(self, processor, grayscale_image, linear_curve):
        """Preview should return both original and processed images."""
        original, processed = processor.preview_curve_effect(
            grayscale_image,
            linear_curve,
        )

        assert original is not None
        assert processed is not None
        assert original.size == processed.size

    def test_preview_with_thumbnail(self, processor, grayscale_image, linear_curve):
        """Preview with thumbnail size should resize."""
        original, processed = processor.preview_curve_effect(
            grayscale_image,
            linear_curve,
            thumbnail_size=(50, 50),
        )

        assert max(original.size) <= 50
        assert max(processed.size) <= 50

    def test_export_to_file_png(self, processor, grayscale_image):
        """Export to PNG file."""
        result = processor.load_image(grayscale_image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            settings = ExportSettings(format=ImageFormat.PNG)
            processor.export(result, output_path, settings)

            assert output_path.exists()
            # Verify it's a valid image
            loaded = Image.open(output_path)
            assert loaded.size == grayscale_image.size
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_to_file_jpeg(self, processor, rgb_image):
        """Export to JPEG file."""
        result = processor.load_image(rgb_image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = Path(f.name)

        try:
            settings = ExportSettings(format=ImageFormat.JPEG, jpeg_quality=90)
            processor.export(result, output_path, settings)

            assert output_path.exists()
            loaded = Image.open(output_path)
            assert loaded.size == rgb_image.size
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_to_file_tiff(self, processor, grayscale_image):
        """Export to TIFF file."""
        result = processor.load_image(grayscale_image)

        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            output_path = Path(f.name)

        try:
            settings = ExportSettings(format=ImageFormat.TIFF)
            processor.export(result, output_path, settings)

            assert output_path.exists()
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_to_bytes(self, processor, grayscale_image):
        """Export to bytes."""
        result = processor.load_image(grayscale_image)
        settings = ExportSettings(format=ImageFormat.PNG)

        data, ext = processor.export_to_bytes(result, settings)

        assert data is not None
        assert len(data) > 0
        assert ext == ".png"

        # Verify it's a valid image
        img = Image.open(io.BytesIO(data))
        assert img.size == grayscale_image.size

    def test_export_jpeg_quality(self, processor, rgb_image):
        """Higher JPEG quality should produce larger files."""
        result = processor.load_image(rgb_image)

        low_quality = ExportSettings(format=ImageFormat.JPEG, jpeg_quality=50)
        high_quality = ExportSettings(format=ImageFormat.JPEG, jpeg_quality=100)

        low_data, _ = processor.export_to_bytes(result, low_quality)
        high_data, _ = processor.export_to_bytes(result, high_quality)

        # High quality should be larger
        assert len(high_data) > len(low_data)

    def test_get_supported_formats(self):
        """Supported formats should include common types."""
        formats = ImageProcessor.get_supported_formats()
        assert ".jpg" in formats
        assert ".png" in formats
        assert ".tiff" in formats

    def test_get_export_formats(self):
        """Export formats should be available."""
        formats = ImageProcessor.get_export_formats()
        assert len(formats) > 0
        # Each format should be (value, description) tuple
        for value, desc in formats:
            assert isinstance(value, str)
            assert isinstance(desc, str)


class TestCurveLUT:
    """Tests for LUT creation and application."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_lut_creates_256_entries(self, processor):
        """LUT should have 256 entries."""
        curve = CurveData(
            name="Test",
            input_values=[0, 0.5, 1],
            output_values=[0, 0.5, 1],
        )
        lut = processor._create_lut(curve)
        assert len(lut) == 256

    def test_lut_interpolates(self, processor):
        """LUT should interpolate between curve points."""
        curve = CurveData(
            name="Test",
            input_values=[0, 1],
            output_values=[0, 1],
        )
        lut = processor._create_lut(curve)

        # Check midpoint interpolation
        assert lut[127] in range(125, 130)
        assert lut[0] == 0
        assert lut[255] == 255

    def test_lut_caches(self, processor):
        """LUT should be cached for repeated use."""
        curve = CurveData(
            name="CachedCurve",
            input_values=[0, 1],
            output_values=[0, 1],
        )

        lut1 = processor._create_lut(curve)
        lut2 = processor._create_lut(curve)

        # Should be the same object from cache
        assert lut1 is lut2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_load_unsupported_type_raises(self, processor):
        """Loading unsupported type should raise."""
        with pytest.raises(TypeError):
            processor.load_image(12345)

    def test_invert_rgba_preserves_alpha(self, processor):
        """Inverting RGBA should preserve alpha channel."""
        arr = np.ones((50, 50, 4), dtype=np.uint8) * 128
        arr[:, :, 3] = 200  # Set alpha
        img = Image.fromarray(arr, mode="RGBA")

        result = processor.load_image(img)
        inverted = processor.invert(result)

        inv_arr = np.array(inverted.image)
        # Alpha should be preserved
        assert np.all(inv_arr[:, :, 3] == 200)
        # RGB should be inverted
        assert np.all(inv_arr[:, :, 0] == 127)

    def test_curve_with_single_point_interpolates(self, processor):
        """Curve with few points should still work."""
        curve = CurveData(
            name="Sparse",
            input_values=[0, 1],
            output_values=[0.2, 0.8],
        )

        arr = np.ones((10, 10), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")

        result = processor.load_image(img)
        processed = processor.apply_curve(result, curve)

        assert processed.image is not None

    def test_empty_processing_notes(self, processor):
        """New result should have empty notes."""
        arr = np.ones((10, 10), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")

        result = processor.load_image(img)
        assert result.processing_notes == []

    def test_processing_notes_accumulate(self, processor):
        """Notes should accumulate through processing."""
        curve = CurveData(
            name="Test",
            input_values=[0, 1],
            output_values=[0, 1],
        )

        arr = np.ones((10, 10), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")

        result = processor.load_image(img)
        result = processor.apply_curve(result, curve)
        result = processor.invert(result)

        assert len(result.processing_notes) == 2
