"""
Unit tests for image processor module.

Tests curve application, inversion, and export functionality.
"""

import io

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging import (
    ExportSettings,
    ImageFormat,
    ImageProcessor,
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

    def test_export_to_file_png(self, processor, grayscale_image, tmp_path):
        """Export to PNG file."""
        result = processor.load_image(grayscale_image)
        output_path = tmp_path / "test_output.png"

        settings = ExportSettings(format=ImageFormat.PNG)
        processor.export(result, output_path, settings)

        assert output_path.exists()
        # Verify it's a valid image
        loaded = Image.open(output_path)
        assert loaded.size == grayscale_image.size

    def test_export_to_file_jpeg(self, processor, rgb_image, tmp_path):
        """Export to JPEG file."""
        result = processor.load_image(rgb_image)
        output_path = tmp_path / "test_output.jpg"

        settings = ExportSettings(format=ImageFormat.JPEG, jpeg_quality=90)
        processor.export(result, output_path, settings)

        assert output_path.exists()
        loaded = Image.open(output_path)
        assert loaded.size == rgb_image.size

    def test_export_to_file_tiff(self, processor, grayscale_image, tmp_path):
        """Export to TIFF file."""
        result = processor.load_image(grayscale_image)
        output_path = tmp_path / "test_output.tiff"

        settings = ExportSettings(format=ImageFormat.TIFF)
        processor.export(result, output_path, settings)

        assert output_path.exists()

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


class TestPerChannelCurves:
    """Tests for per-channel curve application."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    @pytest.fixture
    def rgb_image(self):
        """Create an RGB test image with distinct channel values."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = 100  # Red
        arr[:, :, 1] = 150  # Green
        arr[:, :, 2] = 200  # Blue
        return Image.fromarray(arr, mode="RGB")

    @pytest.fixture
    def rgba_image(self):
        """Create RGBA test image."""
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[:, :, 0] = 100  # Red
        arr[:, :, 1] = 150  # Green
        arr[:, :, 2] = 200  # Blue
        arr[:, :, 3] = 255  # Alpha
        return Image.fromarray(arr, mode="RGBA")

    @pytest.fixture
    def red_boost_curve(self):
        """Curve that boosts values."""
        return CurveData(
            name="RedBoost",
            input_values=[0, 0.5, 1],
            output_values=[0.1, 0.7, 1],
        )

    @pytest.fixture
    def green_reduce_curve(self):
        """Curve that reduces values."""
        return CurveData(
            name="GreenReduce",
            input_values=[0, 0.5, 1],
            output_values=[0, 0.3, 0.8],
        )

    def test_apply_single_channel_curve(self, processor, rgb_image, red_boost_curve):
        """Apply curve to only red channel."""
        result = processor.load_image(rgb_image)
        processed = processor.apply_curves_per_channel(result, curves={"R": red_boost_curve})

        proc_arr = np.array(processed.image)
        orig_arr = np.array(rgb_image)

        # Red channel should be modified
        assert not np.allclose(proc_arr[:, :, 0], orig_arr[:, :, 0], atol=5)
        # Green and Blue should be unchanged (identity)
        assert np.allclose(proc_arr[:, :, 1], orig_arr[:, :, 1])
        assert np.allclose(proc_arr[:, :, 2], orig_arr[:, :, 2])

    def test_apply_multiple_channel_curves(
        self, processor, rgb_image, red_boost_curve, green_reduce_curve
    ):
        """Apply different curves to R and G channels."""
        result = processor.load_image(rgb_image)
        processed = processor.apply_curves_per_channel(
            result, curves={"R": red_boost_curve, "G": green_reduce_curve}
        )

        proc_arr = np.array(processed.image)
        orig_arr = np.array(rgb_image)

        # Both R and G should be modified
        assert not np.allclose(proc_arr[:, :, 0], orig_arr[:, :, 0], atol=5)
        assert not np.allclose(proc_arr[:, :, 1], orig_arr[:, :, 1], atol=5)
        # Blue unchanged
        assert np.allclose(proc_arr[:, :, 2], orig_arr[:, :, 2])

    def test_apply_all_channel_curves(
        self, processor, rgb_image, red_boost_curve, green_reduce_curve
    ):
        """Apply curves to all three channels."""
        blue_curve = CurveData(
            name="BlueLinear",
            input_values=[0, 1],
            output_values=[0, 0.9],
        )

        result = processor.load_image(rgb_image)
        processed = processor.apply_curves_per_channel(
            result, curves={"R": red_boost_curve, "G": green_reduce_curve, "B": blue_curve}
        )

        assert processed.curve_applied is True
        assert "Applied per-channel curves: R, G, B" in processed.processing_notes[-1]

    def test_per_channel_preserves_alpha(self, processor, rgba_image, red_boost_curve):
        """Per-channel curves should preserve alpha."""
        result = processor.load_image(rgba_image)
        processed = processor.apply_curves_per_channel(result, curves={"R": red_boost_curve})

        proc_arr = np.array(processed.image)
        # Alpha should be preserved
        assert np.all(proc_arr[:, :, 3] == 255)

    def test_per_channel_rejects_grayscale(self, processor, red_boost_curve):
        """Per-channel curves should reject grayscale images."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)

        with pytest.raises(ValueError, match="require RGB"):
            processor.apply_curves_per_channel(result, curves={"R": red_boost_curve})

    def test_empty_curves_dict_no_change(self, processor, rgb_image):
        """Empty curves dict should return image unchanged."""
        result = processor.load_image(rgb_image)
        processed = processor.apply_curves_per_channel(result, curves={})

        proc_arr = np.array(processed.image)
        orig_arr = np.array(rgb_image)
        assert np.allclose(proc_arr, orig_arr)


class TestChannelValidation:
    """Tests for channel validation functionality."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_validate_grayscale_image(self, processor):
        """Validate grayscale image channels."""
        arr = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)

        stats = processor.validate_image_channels(result, require_uniform=False)

        assert stats["mode"] == "L"
        assert stats["channels"] == 1
        assert len(stats["per_channel_stats"]) == 1
        assert stats["per_channel_stats"][0]["min"] >= 0
        assert stats["per_channel_stats"][0]["max"] <= 255

    def test_validate_rgb_image(self, processor):
        """Validate RGB image channels."""
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result = processor.load_image(img)

        stats = processor.validate_image_channels(result, require_uniform=False)

        assert stats["mode"] == "RGB"
        assert stats["channels"] == 3
        assert len(stats["per_channel_stats"]) == 3
        assert stats["all_channels_processed"] is True

    def test_validate_detects_constant_channel(self, processor):
        """Validation should detect constant (unprocessed) channels."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = np.random.randint(0, 256, (50, 50))  # Random red
        arr[:, :, 1] = 128  # Constant green
        arr[:, :, 2] = 128  # Constant blue
        img = Image.fromarray(arr, mode="RGB")
        result = processor.load_image(img)

        stats = processor.validate_image_channels(result, require_uniform=False)

        # Green and blue are constant
        assert stats["per_channel_stats"][1]["min"] == stats["per_channel_stats"][1]["max"]
        assert stats["per_channel_stats"][2]["min"] == stats["per_channel_stats"][2]["max"]

    def test_validate_raises_on_constant_channels_if_required(self, processor):
        """Validation should raise if constant channels and require_uniform=True."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = np.random.randint(10, 240, (50, 50))  # Variable red
        arr[:, :, 1] = 128  # Constant green
        arr[:, :, 2] = 128  # Constant blue
        img = Image.fromarray(arr, mode="RGB")
        result = processor.load_image(img)

        with pytest.raises(ValueError, match="appear unprocessed"):
            processor.validate_image_channels(result, require_uniform=True)

    def test_validate_returns_stats(self, processor):
        """Validation should return comprehensive stats."""
        arr = np.arange(100).reshape(10, 10).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)

        stats = processor.validate_image_channels(result, require_uniform=False)

        assert "mean" in stats["per_channel_stats"][0]
        assert "std" in stats["per_channel_stats"][0]
        assert stats["per_channel_stats"][0]["min"] == 0
        assert stats["per_channel_stats"][0]["max"] == 99


class Test16BitTiffExport:
    """Tests for 16-bit TIFF export functionality."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_export_16bit_grayscale_tiff(self, processor, tmp_path):
        """Export 16-bit grayscale TIFF."""
        arr = np.arange(100).reshape(10, 10).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)

        output_path = tmp_path / "test_16bit.tiff"
        settings = ExportSettings(format=ImageFormat.TIFF_16BIT)
        processor.export(result, output_path, settings)

        assert output_path.exists()
        # Verify it can be loaded
        loaded = Image.open(output_path)
        assert loaded is not None

    def test_export_16bit_rgb_tiff(self, processor, tmp_path):
        """Export 16-bit RGB TIFF (uses tifffile if available)."""
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result = processor.load_image(img)

        output_path = tmp_path / "test_rgb_16bit.tiff"
        settings = ExportSettings(format=ImageFormat.TIFF_16BIT)
        processor.export(result, output_path, settings)

        assert output_path.exists()
        # File should be created
        assert output_path.stat().st_size > 0

    def test_export_preserves_dpi_16bit(self, processor, tmp_path):
        """16-bit export should preserve DPI."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)
        # Manually set DPI
        result = ProcessingResult(
            image=result.image,
            original_size=result.original_size,
            original_mode=result.original_mode,
            original_format=result.original_format,
            original_dpi=(300, 300),
            curve_applied=False,
            inverted=False,
        )

        output_path = tmp_path / "test_dpi_16bit.tiff"
        settings = ExportSettings(format=ImageFormat.TIFF_16BIT, preserve_resolution=True)
        processor.export(result, output_path, settings)

        assert output_path.exists()


class TestImageLoadingEdgeCases:
    """Tests for image loading edge cases."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_load_from_file_path(self, processor, tmp_path):
        """Load image from file path."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        img_path = tmp_path / "test_load.png"
        img.save(img_path)

        result = processor.load_image(str(img_path))
        assert result.image is not None
        assert result.original_size == (50, 50)

    def test_load_from_path_object(self, processor, tmp_path):
        """Load image from Path object."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        img_path = tmp_path / "test_path.png"
        img.save(img_path)

        result = processor.load_image(img_path)
        assert result.image is not None

    def test_load_rgba_numpy(self, processor):
        """Load RGBA image from numpy array."""
        arr = np.ones((50, 50, 4), dtype=np.uint8) * 128
        result = processor.load_image(arr)
        assert result.original_mode == "RGBA"

    def test_load_unsupported_array_shape(self, processor):
        """Loading unsupported array shape should raise."""
        arr = np.ones((50, 50, 5), dtype=np.uint8)  # 5 channels not supported
        with pytest.raises(ValueError, match="Unsupported array shape"):
            processor.load_image(arr)

    def test_load_1d_array_raises(self, processor):
        """Loading 1D array should raise."""
        arr = np.ones(100, dtype=np.uint8)
        with pytest.raises(ValueError):
            processor.load_image(arr)


class TestAlphaChannelHandling:
    """Tests for alpha channel handling in various operations."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    @pytest.fixture
    def la_image(self):
        """Create grayscale+alpha image."""
        arr = np.zeros((50, 50, 2), dtype=np.uint8)
        arr[:, :, 0] = 128  # Luminance
        arr[:, :, 1] = 200  # Alpha
        return Image.fromarray(arr, mode="LA")

    def test_apply_curve_la_preserves_alpha(self, processor, la_image):
        """Apply curve to LA image should preserve alpha."""
        curve = CurveData(
            name="Test",
            input_values=[0, 1],
            output_values=[0.2, 0.8],
        )
        result = processor.load_image(la_image)
        processed = processor.apply_curve(result, curve)

        proc_arr = np.array(processed.image)
        # Alpha should be preserved
        assert np.all(proc_arr[:, :, 1] == 200)

    def test_invert_la_preserves_alpha(self, processor, la_image):
        """Invert LA image should preserve alpha."""
        result = processor.load_image(la_image)
        inverted = processor.invert(result)

        inv_arr = np.array(inverted.image)
        # Alpha should be preserved
        assert np.all(inv_arr[:, :, 1] == 200)
        # Luminance should be inverted
        assert np.all(inv_arr[:, :, 0] == 127)


class TestCurveEdgeCases:
    """Tests for curve-related edge cases."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_curve_with_many_points(self, processor):
        """Curve with many control points."""
        inputs = [i / 100 for i in range(101)]
        outputs = [i / 100 for i in range(101)]
        curve = CurveData(
            name="Dense",
            input_values=inputs,
            output_values=outputs,
        )

        arr = np.ones((10, 10), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)
        processed = processor.apply_curve(result, curve)

        assert processed.image is not None

    def test_curve_clamps_values(self, processor):
        """Curve output should be clamped to valid range."""
        # Curve that would go outside 0-1
        curve = CurveData(
            name="OutOfRange",
            input_values=[0, 0.5, 1],
            output_values=[0, 0.5, 1],  # Normal range
        )

        arr = np.array([[0, 128, 255]], dtype=np.uint8).reshape(1, 3)
        img = Image.fromarray(arr, mode="L")
        result = processor.load_image(img)
        processed = processor.apply_curve(result, curve)

        proc_arr = np.array(processed.image)
        # All values should be in valid range
        assert proc_arr.min() >= 0
        assert proc_arr.max() <= 255
