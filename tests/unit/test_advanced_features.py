"""
Unit tests for advanced features module.

Tests alternative process simulation, negative blending, QR metadata generation,
style transfer, and print comparison functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.advanced.features import (
    AlternativeProcessSimulator,
    AlternativeProcessParams,
    BlendMode,
    HistoricStyle,
    NegativeBlender,
    PrintComparison,
    PrintMetadata,
    QRMetadataGenerator,
    StyleParameters,
    StyleTransfer,
    HAS_QRCODE,
)


# Fixtures for test images
@pytest.fixture
def gray_test_image():
    """Create a test grayscale image with gradient."""
    arr = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


@pytest.fixture
def rgb_test_image():
    """Create a test RGB image."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :, 0] = 255  # Red half
    arr[50:, :, 2] = 255  # Blue half
    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def numpy_gray_image():
    """Create numpy grayscale test image."""
    return np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)


@pytest.fixture
def numpy_gray_normalized():
    """Create normalized numpy grayscale test image."""
    return np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)


@pytest.fixture
def test_metadata():
    """Create test print metadata."""
    return PrintMetadata(
        title="Test Print",
        artist="Test Artist",
        date="2025-11-30",
        edition="1/10",
        paper="Arches Platine",
        chemistry="Pt/Pd 50/50",
        exposure_time="5 minutes",
        developer="Potassium oxalate",
        curve_name="Test Curve",
        dmax=1.85,
        dmin=0.08,
        notes="Test notes",
    )


class TestAlternativeProcessParams:
    """Tests for AlternativeProcessParams dataclass."""

    def test_default_params(self):
        """Default parameters should be neutral."""
        params = AlternativeProcessParams()
        assert params.gamma == 1.0
        assert params.contrast == 1.0
        assert params.shadow_color == (0, 0, 0)
        assert params.highlight_color == (255, 255, 255)
        assert params.dmax == 1.8
        assert params.dmin == 0.1
        assert params.stain_level == 0.0

    def test_custom_params(self):
        """Custom parameters should be applied."""
        params = AlternativeProcessParams(
            gamma=1.2,
            contrast=1.1,
            shadow_color=(10, 20, 30),
            highlight_color=(240, 250, 255),
            dmax=1.9,
            dmin=0.05,
            stain_level=0.1,
        )
        assert params.gamma == 1.2
        assert params.contrast == 1.1
        assert params.shadow_color == (10, 20, 30)
        assert params.highlight_color == (240, 250, 255)


class TestAlternativeProcessSimulator:
    """Tests for AlternativeProcessSimulator."""

    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return AlternativeProcessSimulator()

    def test_simulator_initialization(self, simulator):
        """Simulator should initialize with presets."""
        assert simulator._process_presets is not None
        assert 'cyanotype' in simulator._process_presets
        assert 'vandyke' in simulator._process_presets

    def test_cyanotype_simulation_pil(self, simulator, gray_test_image):
        """Cyanotype simulation should produce blue-toned image from PIL."""
        result = simulator.simulate_cyanotype(gray_test_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size == gray_test_image.size
        assert result.info.get('process') == 'Cyanotype'

        # Check for blue tones
        arr = np.array(result)
        # Blue channel should dominate
        assert arr[:, :, 2].mean() > arr[:, :, 0].mean()
        assert arr[:, :, 2].mean() > arr[:, :, 1].mean()

    def test_cyanotype_simulation_numpy(self, simulator, numpy_gray_image):
        """Cyanotype simulation should work with numpy array."""
        result = simulator.simulate_cyanotype(numpy_gray_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('process') == 'Cyanotype'

    def test_cyanotype_custom_params(self, simulator, gray_test_image):
        """Cyanotype with custom parameters should apply them."""
        custom_params = AlternativeProcessParams(
            gamma=1.5,
            contrast=1.3,
            shadow_color=(0, 0, 100),
            highlight_color=(200, 220, 255),
        )
        result = simulator.simulate_cyanotype(gray_test_image, params=custom_params)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_vandyke_simulation(self, simulator, gray_test_image):
        """Van Dyke simulation should produce brown-toned image."""
        result = simulator.simulate_vandyke(gray_test_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('process') == 'Van Dyke Brown'

        # Check for warm brown tones (red should dominate)
        arr = np.array(result)
        assert arr[:, :, 0].mean() > arr[:, :, 2].mean()  # More red than blue

    def test_kallitype_simulation(self, simulator, gray_test_image):
        """Kallitype simulation should produce neutral-warm tones."""
        result = simulator.simulate_kallitype(gray_test_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('process') == 'Kallitype'

    def test_gum_bichromate_simulation(self, simulator, gray_test_image):
        """Gum bichromate should accept custom pigment color."""
        pigment = (150, 100, 50)
        result = simulator.simulate_gum_bichromate(gray_test_image, pigment_color=pigment)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('process') == 'Gum Bichromate'

    def test_salt_print_simulation(self, simulator, gray_test_image):
        """Salt print simulation should produce delicate warm tones."""
        result = simulator.simulate_salt_print(gray_test_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('process') == 'Salt Print'

    def test_rgb_input_conversion(self, simulator, rgb_test_image):
        """RGB images should be converted to grayscale for processing."""
        result = simulator.simulate_cyanotype(rgb_test_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_normalized_numpy_input(self, simulator, numpy_gray_normalized):
        """Normalized numpy arrays should be handled correctly."""
        result = simulator.simulate_cyanotype(numpy_gray_normalized)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_stain_level_application(self, simulator, gray_test_image):
        """Paper staining should affect the result."""
        params = AlternativeProcessParams(
            stain_level=0.5,
            highlight_color=(250, 240, 230),
        )
        result = simulator.simulate_cyanotype(gray_test_image, params=params)

        assert isinstance(result, Image.Image)

    def test_extreme_gamma_values(self, simulator, gray_test_image):
        """Extreme gamma values should be handled correctly."""
        params_low = AlternativeProcessParams(gamma=0.5)
        result_low = simulator.simulate_cyanotype(gray_test_image, params=params_low)
        assert isinstance(result_low, Image.Image)

        params_high = AlternativeProcessParams(gamma=2.5)
        result_high = simulator.simulate_cyanotype(gray_test_image, params=params_high)
        assert isinstance(result_high, Image.Image)


class TestBlendMode:
    """Tests for BlendMode enum."""

    def test_all_modes_defined(self):
        """All blend modes should be defined."""
        assert BlendMode.NORMAL
        assert BlendMode.MULTIPLY
        assert BlendMode.SCREEN
        assert BlendMode.OVERLAY
        assert BlendMode.SOFT_LIGHT
        assert BlendMode.HARD_LIGHT
        assert BlendMode.LINEAR_DODGE
        assert BlendMode.LINEAR_BURN

    def test_mode_values(self):
        """Mode values should be valid strings."""
        assert BlendMode.MULTIPLY.value == "multiply"
        assert BlendMode.SCREEN.value == "screen"


class TestNegativeBlender:
    """Tests for NegativeBlender."""

    @pytest.fixture
    def blender(self):
        """Create blender instance."""
        return NegativeBlender()

    @pytest.fixture
    def test_negatives(self):
        """Create test negative images."""
        neg1 = Image.new('L', (100, 100), color=128)
        neg2 = Image.new('L', (100, 100), color=200)
        return [neg1, neg2]

    def test_blend_two_negatives_normal(self, blender, test_negatives):
        """Blending with NORMAL mode should replace."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.NORMAL]
        )

        assert isinstance(result, Image.Image)
        assert result.mode == 'L'
        assert result.size == (100, 100)

    def test_blend_multiply_mode(self, blender, test_negatives):
        """MULTIPLY mode should darken."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.MULTIPLY]
        )

        arr = np.array(result)
        # Multiply should produce darker values
        assert arr.mean() < 200

    def test_blend_screen_mode(self, blender, test_negatives):
        """SCREEN mode should lighten."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.SCREEN]
        )

        arr = np.array(result)
        # Screen should produce lighter values
        assert arr.mean() > 128

    def test_blend_overlay_mode(self, blender):
        """OVERLAY mode should apply correctly."""
        neg1 = Image.new('L', (100, 100), color=100)
        neg2 = Image.new('L', (100, 100), color=150)

        result = blender.blend_negatives(
            [neg1, neg2],
            blend_modes=[BlendMode.NORMAL, BlendMode.OVERLAY]
        )

        assert isinstance(result, Image.Image)

    def test_blend_soft_light_mode(self, blender, test_negatives):
        """SOFT_LIGHT mode should apply correctly."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.SOFT_LIGHT]
        )

        assert isinstance(result, Image.Image)

    def test_blend_hard_light_mode(self, blender, test_negatives):
        """HARD_LIGHT mode should apply correctly."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.HARD_LIGHT]
        )

        assert isinstance(result, Image.Image)

    def test_blend_linear_dodge_mode(self, blender, test_negatives):
        """LINEAR_DODGE (add) mode should lighten."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.LINEAR_DODGE]
        )

        arr = np.array(result)
        assert arr.mean() >= 128  # Should be lighter

    def test_blend_linear_burn_mode(self, blender, test_negatives):
        """LINEAR_BURN mode should darken."""
        result = blender.blend_negatives(
            test_negatives,
            blend_modes=[BlendMode.NORMAL, BlendMode.LINEAR_BURN]
        )

        assert isinstance(result, Image.Image)

    def test_blend_with_masks(self, blender):
        """Blending with masks should apply selective effects."""
        neg1 = Image.new('L', (100, 100), color=100)
        neg2 = Image.new('L', (100, 100), color=200)

        # Create gradient mask
        mask = Image.new('L', (100, 100))
        mask_arr = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode='L')

        result = blender.blend_negatives(
            [neg1, neg2],
            masks=[None, mask],
            blend_modes=[BlendMode.NORMAL, BlendMode.NORMAL]
        )

        assert isinstance(result, Image.Image)
        # Result should have gradient from neg1 to neg2

    def test_blend_numpy_arrays(self, blender):
        """Blending should work with numpy arrays."""
        neg1 = np.ones((50, 50), dtype=np.float32) * 0.5
        neg2 = np.ones((50, 50), dtype=np.float32) * 0.8

        result = blender.blend_negatives([neg1, neg2])

        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

    def test_blend_size_mismatch(self, blender):
        """Mismatched sizes should raise error."""
        neg1 = Image.new('L', (100, 100), color=128)
        neg2 = Image.new('L', (50, 50), color=200)

        with pytest.raises(ValueError, match="size.*doesn't match"):
            blender.blend_negatives([neg1, neg2])

    def test_blend_no_negatives(self, blender):
        """Empty negative list should raise error."""
        with pytest.raises(ValueError, match="At least one negative"):
            blender.blend_negatives([])

    def test_create_contrast_mask(self, blender, gray_test_image):
        """Contrast mask should highlight high-contrast areas."""
        mask = blender.create_contrast_mask(gray_test_image, threshold=0.5)

        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'
        assert mask.size == gray_test_image.size

    def test_create_contrast_mask_numpy(self, blender, numpy_gray_normalized):
        """Contrast mask should work with numpy input."""
        mask = blender.create_contrast_mask(numpy_gray_normalized, threshold=0.3)

        assert isinstance(mask, Image.Image)

    def test_create_highlight_mask(self, blender, gray_test_image):
        """Highlight mask should target bright areas."""
        mask = blender.create_highlight_mask(gray_test_image, threshold=0.7)

        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'
        assert mask.size == gray_test_image.size

    def test_create_shadow_mask(self, blender, gray_test_image):
        """Shadow mask should target dark areas."""
        mask = blender.create_shadow_mask(gray_test_image, threshold=0.3)

        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'
        assert mask.size == gray_test_image.size

    def test_apply_dodge_burn_both(self, blender):
        """Dodge and burn should be applied together."""
        img = Image.new('L', (100, 100), color=128)
        dodge_mask = Image.new('L', (100, 100), color=255)
        burn_mask = Image.new('L', (100, 100), color=0)

        result = blender.apply_dodge_burn(
            img,
            dodge_mask=dodge_mask,
            burn_mask=burn_mask,
            dodge_amount=0.5,
            burn_amount=0.5
        )

        assert isinstance(result, Image.Image)
        assert result.mode == 'L'

    def test_apply_dodge_only(self, blender):
        """Dodging should lighten image."""
        img = Image.new('L', (100, 100), color=100)
        dodge_mask = Image.new('L', (100, 100), color=255)

        result = blender.apply_dodge_burn(
            img,
            dodge_mask=dodge_mask,
            dodge_amount=0.5
        )

        arr = np.array(result)
        assert arr.mean() > 100  # Should be lighter

    def test_apply_burn_only(self, blender):
        """Burning should darken image."""
        img = Image.new('L', (100, 100), color=150)
        burn_mask = Image.new('L', (100, 100), color=255)

        result = blender.apply_dodge_burn(
            img,
            burn_mask=burn_mask,
            burn_amount=0.5
        )

        arr = np.array(result)
        assert arr.mean() < 150  # Should be darker

    def test_create_multi_layer_mask_add(self, blender):
        """Multi-layer mask with add mode should sum layers."""
        layer1 = Image.new('L', (50, 50), color=100)
        layer2 = Image.new('L', (50, 50), color=50)

        result = blender.create_multi_layer_mask(
            [layer1, layer2],
            blend_modes=['add', 'add']
        )

        assert isinstance(result, Image.Image)
        arr = np.array(result)
        assert arr.mean() > 100  # Should be sum (clamped)

    def test_create_multi_layer_mask_multiply(self, blender):
        """Multi-layer mask with multiply should darken."""
        layer1 = Image.new('L', (50, 50), color=200)
        layer2 = Image.new('L', (50, 50), color=200)

        result = blender.create_multi_layer_mask(
            [layer1, layer2],
            blend_modes=['multiply', 'multiply']
        )

        assert isinstance(result, Image.Image)
        arr = np.array(result)
        # Multiply normalized: 200/255 * 200/255 = ~0.61, * 255 = ~156
        assert arr.mean() < 200

    def test_create_multi_layer_mask_max(self, blender):
        """Multi-layer mask with max mode."""
        layer1 = Image.new('L', (50, 50), color=100)
        layer2 = Image.new('L', (50, 50), color=150)

        result = blender.create_multi_layer_mask(
            [layer1, layer2],
            blend_modes=['max', 'max']
        )

        arr = np.array(result)
        assert np.isclose(arr.mean(), 150, atol=2)

    def test_create_multi_layer_mask_min(self, blender):
        """Multi-layer mask with min mode."""
        layer1 = Image.new('L', (50, 50), color=100)
        layer2 = Image.new('L', (50, 50), color=150)

        result = blender.create_multi_layer_mask(
            [layer1, layer2],
            blend_modes=['min', 'min']
        )

        arr = np.array(result)
        assert np.isclose(arr.mean(), 100, atol=2)

    def test_create_multi_layer_mask_no_layers(self, blender):
        """Empty layer list should raise error."""
        with pytest.raises(ValueError, match="At least one layer"):
            blender.create_multi_layer_mask([])


class TestQRMetadataGenerator:
    """Tests for QRMetadataGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance, skip if qrcode not available."""
        if not HAS_QRCODE:
            pytest.skip("qrcode library not installed")
        return QRMetadataGenerator()

    def test_generator_requires_qrcode(self):
        """Generator should require qrcode library."""
        if not HAS_QRCODE:
            with pytest.raises(ImportError, match="qrcode library is required"):
                QRMetadataGenerator()

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_encode_recipe(self, generator, test_metadata):
        """Recipe encoding should create compact string."""
        encoded = generator.encode_recipe(test_metadata)

        assert isinstance(encoded, str)
        assert 'title:Test Print' in encoded
        assert 'artist:Test Artist' in encoded
        assert '|' in encoded  # Separator

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_encode_dict(self, generator):
        """Should encode dictionary directly."""
        data = {'title': 'Test', 'artist': 'Artist'}
        encoded = generator.encode_recipe(data)

        assert 'title:Test' in encoded
        assert 'artist:Artist' in encoded

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_encode_empty_fields(self, generator):
        """Empty fields should be omitted."""
        metadata = PrintMetadata(title="Test", artist="")
        encoded = generator.encode_recipe(metadata)

        assert 'title:Test' in encoded
        assert 'artist:' not in encoded

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_generate_print_qr(self, generator, test_metadata):
        """QR code generation should produce valid image."""
        qr_img = generator.generate_print_qr(test_metadata, size=200)

        assert isinstance(qr_img, Image.Image)
        assert qr_img.size == (200, 200)
        assert qr_img.mode in ['1', 'L', 'RGB']

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_generate_qr_different_sizes(self, generator, test_metadata):
        """QR code should support different sizes."""
        for size in [100, 200, 400]:
            qr_img = generator.generate_print_qr(test_metadata, size=size)
            assert qr_img.size == (size, size)

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_generate_qr_error_correction(self, generator, test_metadata):
        """Different error correction levels should work."""
        for level in ['L', 'M', 'Q', 'H']:
            qr_img = generator.generate_print_qr(
                test_metadata,
                error_correction=level
            )
            assert isinstance(qr_img, Image.Image)

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_create_archival_label(self, generator, test_metadata):
        """Archival label should include QR and text."""
        label = generator.create_archival_label(test_metadata)

        assert isinstance(label, Image.Image)
        assert label.mode == 'RGB'
        assert label.size == (600, 300)  # Default size

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_create_archival_label_custom_size(self, generator, test_metadata):
        """Archival label should support custom size."""
        label = generator.create_archival_label(
            test_metadata,
            label_size=(800, 400),
            qr_size=250
        )

        assert label.size == (800, 400)

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_parse_encoded_data(self, generator):
        """Encoded data should be parseable."""
        encoded = "title:Test|artist:Artist|date:2025"
        parsed = generator._parse_encoded_data(encoded)

        assert parsed['title'] == 'Test'
        assert parsed['artist'] == 'Artist'
        assert parsed['date'] == '2025'

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not installed")
    def test_parse_encoded_data_with_colons(self, generator):
        """Should handle colons in values."""
        encoded = "notes:Test: with colons"
        parsed = generator._parse_encoded_data(encoded)

        assert parsed['notes'] == 'Test: with colons'


class TestHistoricStyle:
    """Tests for HistoricStyle enum."""

    def test_all_styles_defined(self):
        """All historic styles should be defined."""
        assert HistoricStyle.PICTORIALIST_1890S
        assert HistoricStyle.EDWARD_WESTON
        assert HistoricStyle.IRVING_PENN
        assert HistoricStyle.SALLY_MANN
        assert HistoricStyle.FREDERICK_EVANS
        assert HistoricStyle.PAUL_STRAND


class TestStyleParameters:
    """Tests for StyleParameters dataclass."""

    def test_default_style_parameters(self):
        """Default style parameters should be neutral."""
        style = StyleParameters(
            name="Test Style",
            description="Test description"
        )

        assert style.name == "Test Style"
        assert style.gamma == 1.0
        assert style.contrast == 1.0

    def test_custom_style_parameters(self):
        """Custom style parameters should be applied."""
        style = StyleParameters(
            name="Custom",
            description="Custom style",
            gamma=1.5,
            contrast=1.2,
            shadow_tone=(10, 10, 10),
            highlight_tone=(250, 250, 250),
            dmax=2.0,
        )

        assert style.gamma == 1.5
        assert style.dmax == 2.0


class TestStyleTransfer:
    """Tests for StyleTransfer."""

    @pytest.fixture
    def style_transfer(self):
        """Create style transfer instance."""
        return StyleTransfer()

    def test_load_historic_styles(self, style_transfer):
        """All historic styles should be loaded."""
        styles = style_transfer.styles

        assert HistoricStyle.PICTORIALIST_1890S in styles
        assert HistoricStyle.EDWARD_WESTON in styles
        assert HistoricStyle.IRVING_PENN in styles
        assert HistoricStyle.SALLY_MANN in styles
        assert HistoricStyle.FREDERICK_EVANS in styles
        assert HistoricStyle.PAUL_STRAND in styles

    def test_apply_weston_style(self, style_transfer, gray_test_image):
        """Edward Weston style should apply high contrast."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.EDWARD_WESTON
        )

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.info.get('style') == 'Edward Weston'

    def test_apply_pictorialist_style(self, style_transfer, gray_test_image):
        """Pictorialist style should apply soft tones."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.PICTORIALIST_1890S
        )

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_apply_penn_style(self, style_transfer, gray_test_image):
        """Irving Penn style should apply clean tones."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.IRVING_PENN
        )

        assert isinstance(result, Image.Image)

    def test_apply_mann_style(self, style_transfer, gray_test_image):
        """Sally Mann style should apply atmospheric tones."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.SALLY_MANN
        )

        assert isinstance(result, Image.Image)

    def test_apply_evans_style(self, style_transfer, gray_test_image):
        """Frederick Evans style should apply delicate highlights."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.FREDERICK_EVANS
        )

        assert isinstance(result, Image.Image)

    def test_apply_strand_style(self, style_transfer, gray_test_image):
        """Paul Strand style should apply modernist clarity."""
        result = style_transfer.apply_style(
            gray_test_image,
            HistoricStyle.PAUL_STRAND
        )

        assert isinstance(result, Image.Image)

    def test_apply_style_by_string(self, style_transfer, gray_test_image):
        """Should accept style as string."""
        result = style_transfer.apply_style(
            gray_test_image,
            "edward_weston"
        )

        assert isinstance(result, Image.Image)

    def test_apply_unknown_style(self, style_transfer, gray_test_image):
        """Unknown style should raise error."""
        with pytest.raises(ValueError, match="Unknown style"):
            style_transfer.apply_style(gray_test_image, "nonexistent_style")

    def test_analyze_style_grayscale(self, style_transfer, gray_test_image):
        """Style analysis should extract characteristics."""
        params = style_transfer.analyze_style(gray_test_image)

        assert isinstance(params, StyleParameters)
        assert params.name == "Analyzed Style"
        assert 0.5 <= params.gamma <= 2.0
        assert 0.5 <= params.contrast <= 1.5

    def test_analyze_style_rgb(self, style_transfer, rgb_test_image):
        """Style analysis should work with RGB images."""
        params = style_transfer.analyze_style(rgb_test_image)

        assert isinstance(params, StyleParameters)
        assert isinstance(params.shadow_tone, tuple)
        assert isinstance(params.highlight_tone, tuple)

    def test_analyze_style_numpy(self, style_transfer, numpy_gray_normalized):
        """Style analysis should work with numpy arrays."""
        params = style_transfer.analyze_style(numpy_gray_normalized)

        assert isinstance(params, StyleParameters)

    def test_create_custom_style(self, style_transfer):
        """Should create custom style from parameters."""
        custom_params = {
            'description': 'My custom style',
            'gamma': 1.3,
            'contrast': 1.1,
            'shadow_tone': (15, 15, 15),
            'highlight_tone': (250, 250, 250),
        }

        style = style_transfer.create_custom_style("My Style", custom_params)

        assert style.name == "My Style"
        assert style.gamma == 1.3
        assert style.contrast == 1.1
        assert "My Style" in style_transfer.styles

    def test_create_custom_style_defaults(self, style_transfer):
        """Custom style should use defaults for missing parameters."""
        style = style_transfer.create_custom_style("Minimal", {})

        assert style.gamma == 1.0
        assert style.contrast == 1.0
        assert style.description == 'Custom style'

    def test_apply_custom_style(self, style_transfer, gray_test_image):
        """Should apply custom style after creation."""
        style_transfer.create_custom_style("Test", {'gamma': 1.4})

        result = style_transfer.apply_style(gray_test_image, "Test")

        assert isinstance(result, Image.Image)

    def test_style_with_texture(self, style_transfer, gray_test_image):
        """Style with texture strength should add noise."""
        params = {
            'texture_strength': 0.1,
        }
        style_transfer.create_custom_style("Textured", params)

        result = style_transfer.apply_style(gray_test_image, "Textured")

        assert isinstance(result, Image.Image)


class TestPrintComparison:
    """Tests for PrintComparison."""

    @pytest.fixture
    def comparison(self):
        """Create comparison instance."""
        return PrintComparison()

    @pytest.fixture
    def test_image_pair(self):
        """Create pair of test images with variation."""
        # Use gradient images to avoid division by zero in range calculations
        arr1 = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        arr2 = np.clip(arr1 + 5, 0, 255).astype(np.uint8)
        img1 = Image.fromarray(arr1, mode='L')
        img2 = Image.fromarray(arr2, mode='L')
        return img1, img2

    def test_compare_before_after_identical(self, comparison):
        """Identical images should have high similarity."""
        # Use gradient to avoid division by zero
        arr = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')

        metrics = comparison.compare_before_after(img, img)

        assert metrics['rmse'] < 0.01
        assert metrics['similarity_score'] > 0.99
        assert metrics['histogram_correlation'] > 0.99

    def test_compare_before_after_different(self, comparison, test_image_pair):
        """Different images should have lower similarity."""
        img1, img2 = test_image_pair

        metrics = comparison.compare_before_after(img1, img2)

        assert 'rmse' in metrics
        assert 'psnr' in metrics
        assert 'similarity_score' in metrics
        assert 'histogram_correlation' in metrics
        assert 'tonal_range_preservation' in metrics

    def test_compare_numpy_arrays(self, comparison):
        """Should work with numpy arrays."""
        arr1 = np.ones((50, 50), dtype=np.float32) * 0.5
        arr2 = np.ones((50, 50), dtype=np.float32) * 0.6

        metrics = comparison.compare_before_after(arr1, arr2)

        assert isinstance(metrics['rmse'], float)
        assert metrics['rmse'] > 0

    def test_compare_different_sizes(self, comparison):
        """Different sized images should be resized and compared."""
        img1 = Image.new('L', (100, 100), color=128)
        img2 = Image.new('L', (200, 200), color=128)

        metrics = comparison.compare_before_after(img1, img2)

        assert isinstance(metrics, dict)

    def test_generate_difference_map_colorized(self, comparison, test_image_pair):
        """Colorized difference map should be RGB."""
        img1, img2 = test_image_pair

        diff_map = comparison.generate_difference_map(img1, img2, colorize=True)

        assert isinstance(diff_map, Image.Image)
        assert diff_map.mode == 'RGB'
        assert diff_map.size == img1.size

    def test_generate_difference_map_grayscale(self, comparison, test_image_pair):
        """Grayscale difference map should be L mode."""
        img1, img2 = test_image_pair

        diff_map = comparison.generate_difference_map(img1, img2, colorize=False)

        assert isinstance(diff_map, Image.Image)
        assert diff_map.mode == 'L'

    def test_difference_map_identical_images(self, comparison):
        """Difference map of identical images should be neutral."""
        img = Image.new('L', (50, 50), color=100)

        diff_map = comparison.generate_difference_map(img, img, colorize=True)

        arr = np.array(diff_map)
        # Should be near gray (128, 128, 128) for no difference
        assert np.abs(arr.mean() - 128) < 10

    def test_calculate_similarity_mse(self, comparison):
        """MSE similarity should work correctly."""
        arr = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        img1 = Image.fromarray(arr, mode='L')
        img2 = Image.fromarray(arr.copy(), mode='L')

        score = comparison.calculate_similarity_score(img1, img2, method='mse')

        assert 0.0 <= score <= 1.0
        assert score > 0.99  # Identical images

    def test_calculate_similarity_correlation(self, comparison):
        """Correlation similarity should work correctly."""
        arr1 = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        arr2 = np.clip(arr1 + 10, 0, 255).astype(np.uint8)
        img1 = Image.fromarray(arr1, mode='L')
        img2 = Image.fromarray(arr2, mode='L')

        score = comparison.calculate_similarity_score(
            img1, img2, method='correlation'
        )

        assert 0.0 <= score <= 1.0

    def test_calculate_similarity_ssim(self, comparison):
        """SSIM similarity should work correctly."""
        arr = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        img1 = Image.fromarray(arr, mode='L')
        img2 = Image.fromarray(arr.copy(), mode='L')

        score = comparison.calculate_similarity_score(img1, img2, method='ssim')

        assert 0.0 <= score <= 1.0
        assert score > 0.9  # Very similar

    def test_calculate_similarity_unknown_method(self, comparison):
        """Unknown method should raise error."""
        img = Image.new('L', (50, 50))

        with pytest.raises(ValueError, match="Unknown similarity method"):
            comparison.calculate_similarity_score(img, img, method='unknown')

    def test_generate_comparison_report(self, comparison):
        """Comparison report should analyze multiple images."""
        arr1 = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        arr2 = np.clip(arr1 + 5, 0, 255).astype(np.uint8)
        arr3 = np.clip(arr1 - 5, 0, 255).astype(np.uint8)
        images = {
            'original': Image.fromarray(arr1, mode='L'),
            'print1': Image.fromarray(arr2, mode='L'),
            'print2': Image.fromarray(arr3, mode='L'),
        }

        report = comparison.generate_comparison_report(images)

        assert report['reference'] == 'original'
        assert report['num_images'] == 3
        assert 'print1' in report['comparisons']
        assert 'print2' in report['comparisons']
        assert 'summary' in report
        assert 'average_similarity' in report['summary']

    def test_generate_comparison_report_custom_reference(self, comparison):
        """Should accept custom reference key."""
        arr1 = np.linspace(0, 255, 2500).reshape(50, 50).astype(np.uint8)
        arr2 = np.clip(arr1 + 10, 0, 255).astype(np.uint8)
        images = {
            'img1': Image.fromarray(arr1, mode='L'),
            'img2': Image.fromarray(arr2, mode='L'),
        }

        report = comparison.generate_comparison_report(
            images,
            reference_key='img2'
        )

        assert report['reference'] == 'img2'

    def test_generate_comparison_report_invalid_reference(self, comparison):
        """Invalid reference key should raise error."""
        images = {'img1': Image.new('L', (50, 50))}

        with pytest.raises(ValueError, match="Reference key.*not in images"):
            comparison.generate_comparison_report(
                images,
                reference_key='nonexistent'
            )

    def test_generate_comparison_report_empty(self, comparison):
        """Empty image dict should raise error."""
        with pytest.raises(ValueError, match="At least one image"):
            comparison.generate_comparison_report({})

    def test_to_gray_array_pil_grayscale(self, comparison):
        """Should convert PIL grayscale correctly."""
        img = Image.new('L', (50, 50), color=128)
        arr = comparison._to_gray_array(img)

        assert arr.shape == (50, 50)
        assert np.isclose(arr.mean(), 128/255, atol=0.01)

    def test_to_gray_array_pil_rgb(self, comparison):
        """Should convert PIL RGB to grayscale."""
        img = Image.new('RGB', (50, 50), color=(128, 128, 128))
        arr = comparison._to_gray_array(img)

        assert arr.shape == (50, 50)
        assert arr.ndim == 2

    def test_to_gray_array_numpy_rgb(self, comparison):
        """Should convert numpy RGB to grayscale."""
        arr_rgb = np.ones((50, 50, 3), dtype=np.uint8) * 100
        arr = comparison._to_gray_array(arr_rgb)

        assert arr.shape == (50, 50)
        assert arr.ndim == 2

    def test_to_gray_array_normalized(self, comparison):
        """Should handle normalized arrays."""
        arr_norm = np.ones((50, 50), dtype=np.float32) * 0.5
        arr = comparison._to_gray_array(arr_norm)

        assert arr.max() <= 1.0
        assert np.isclose(arr.mean(), 0.5, atol=0.01)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_image(self):
        """Empty images should be handled."""
        simulator = AlternativeProcessSimulator()
        empty = Image.new('L', (1, 1), color=0)

        result = simulator.simulate_cyanotype(empty)
        assert isinstance(result, Image.Image)

    def test_extreme_values_image(self):
        """Images with extreme values should be handled."""
        simulator = AlternativeProcessSimulator()

        # All white
        white = Image.new('L', (10, 10), color=255)
        result = simulator.simulate_cyanotype(white)
        assert isinstance(result, Image.Image)

        # All black
        black = Image.new('L', (10, 10), color=0)
        result = simulator.simulate_cyanotype(black)
        assert isinstance(result, Image.Image)

    def test_large_image_processing(self):
        """Large images should be processed efficiently."""
        simulator = AlternativeProcessSimulator()
        large = Image.new('L', (1000, 1000), color=128)

        result = simulator.simulate_cyanotype(large)
        assert result.size == (1000, 1000)

    def test_single_pixel_operations(self):
        """Single pixel images should work."""
        blender = NegativeBlender()
        img = Image.new('L', (1, 1), color=128)

        result = blender.blend_negatives([img, img])
        assert result.size == (1, 1)
