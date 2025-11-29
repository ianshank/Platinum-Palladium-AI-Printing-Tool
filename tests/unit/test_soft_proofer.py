"""Tests for soft proofing simulation module."""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.proofing import (
    SoftProofer,
    ProofSettings,
    ProofResult,
    PaperSimulation,
    PAPER_PRESETS,
)


@pytest.fixture
def proofer():
    """Create default soft proofer."""
    return SoftProofer()


@pytest.fixture
def custom_proofer():
    """Create soft proofer with custom settings."""
    settings = ProofSettings(
        paper_white_rgb=(245, 240, 230),
        paper_dmax=1.7,
        paper_dmin=0.06,
        platinum_ratio=0.5,
    )
    return SoftProofer(settings)


@pytest.fixture
def gray_image():
    """Create a grayscale test image."""
    arr = np.linspace(0, 255, 100).reshape(10, 10).astype(np.uint8)
    arr = np.repeat(np.repeat(arr, 10, axis=0), 10, axis=1)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def rgb_image():
    """Create an RGB test image."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 150  # Red
    arr[:, :, 1] = 120  # Green
    arr[:, :, 2] = 100  # Blue
    return Image.fromarray(arr, mode="RGB")


class TestProofSettings:
    """Test proof settings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = ProofSettings()

        assert settings.paper_white_rgb == (250, 246, 238)
        assert settings.paper_dmax == 1.6
        assert settings.paper_dmin == 0.07
        assert settings.platinum_ratio == 0.0

    def test_from_paper_preset_arches(self):
        """Test settings from Arches Platine preset."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.ARCHES_PLATINE)

        assert settings.paper_dmax == 1.6

    def test_from_paper_preset_bergger(self):
        """Test settings from Bergger COT320 preset."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.BERGGER_COT320)

        assert settings.paper_dmax == 1.55

    def test_from_paper_preset_hahnemuhle(self):
        """Test settings from Hahnemuhle preset."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.HAHNEMUHLE_PLATINUM)

        assert settings.paper_dmax == 1.65

    def test_from_paper_preset_custom(self):
        """Test settings from custom preset."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.CUSTOM)

        # Should return default settings
        assert settings.paper_dmax == 1.6

    def test_custom_settings(self):
        """Test custom settings."""
        settings = ProofSettings(
            paper_white_rgb=(255, 255, 255),
            paper_dmax=1.8,
            paper_dmin=0.05,
            platinum_ratio=1.0,
            add_paper_texture=True,
            texture_strength=0.2,
        )

        assert settings.paper_dmax == 1.8
        assert settings.platinum_ratio == 1.0
        assert settings.add_paper_texture is True


class TestSoftProofer:
    """Test soft proofer functionality."""

    def test_proof_grayscale(self, proofer, gray_image):
        """Test proofing a grayscale image."""
        result = proofer.proof(gray_image)

        assert isinstance(result, ProofResult)
        assert result.image is not None
        assert result.image.mode == "RGB"

    def test_proof_rgb(self, proofer, rgb_image):
        """Test proofing an RGB image."""
        result = proofer.proof(rgb_image)

        assert result.image is not None
        # RGB should be converted to grayscale for Pt/Pd simulation
        assert "grayscale" in " ".join(result.notes).lower()

    def test_proof_preserves_size(self, proofer, gray_image):
        """Test that proofing preserves image size."""
        result = proofer.proof(gray_image)

        assert result.image.size == gray_image.size

    def test_proof_records_original_size(self, proofer, gray_image):
        """Test that original size is recorded."""
        result = proofer.proof(gray_image)

        assert result.original_size == gray_image.size

    def test_proof_warm_tones(self, gray_image):
        """Test warm tones (palladium-dominant)."""
        settings = ProofSettings(platinum_ratio=0.0)
        proofer = SoftProofer(settings)
        result = proofer.proof(gray_image)

        assert "warm" in " ".join(result.notes).lower()

    def test_proof_cool_tones(self, gray_image):
        """Test cool tones (platinum-dominant)."""
        settings = ProofSettings(platinum_ratio=1.0)
        proofer = SoftProofer(settings)
        result = proofer.proof(gray_image)

        assert "cool" in " ".join(result.notes).lower()

    def test_proof_with_texture(self, gray_image):
        """Test proofing with paper texture."""
        settings = ProofSettings(add_paper_texture=True, texture_strength=0.2)
        proofer = SoftProofer(settings)
        result = proofer.proof(gray_image)

        assert "texture" in " ".join(result.notes).lower()

    def test_proof_brightness_adjustment(self, gray_image):
        """Test viewing brightness adjustment."""
        settings_dim = ProofSettings(viewing_brightness=0.8)
        settings_bright = ProofSettings(viewing_brightness=1.2)

        proofer_dim = SoftProofer(settings_dim)
        proofer_bright = SoftProofer(settings_bright)

        result_dim = proofer_dim.proof(gray_image)
        result_bright = proofer_bright.proof(gray_image)

        # Bright should be brighter on average
        arr_dim = np.array(result_dim.image)
        arr_bright = np.array(result_bright.image)
        assert np.mean(arr_bright) > np.mean(arr_dim)

    def test_proof_color_temperature(self, gray_image):
        """Test ambient light temperature adjustment."""
        settings_warm = ProofSettings(ambient_light_temperature=4000)  # Warm
        settings_cool = ProofSettings(ambient_light_temperature=7000)  # Cool

        proofer_warm = SoftProofer(settings_warm)
        proofer_cool = SoftProofer(settings_cool)

        result_warm = proofer_warm.proof(gray_image)
        result_cool = proofer_cool.proof(gray_image)

        # Both should produce valid results
        assert result_warm.image is not None
        assert result_cool.image is not None

    def test_compare_multiple_settings(self, proofer, gray_image):
        """Test comparing proofs with different settings."""
        settings_list = [
            ProofSettings(platinum_ratio=0.0),
            ProofSettings(platinum_ratio=0.5),
            ProofSettings(platinum_ratio=1.0),
        ]

        results = proofer.compare(gray_image, settings_list)

        assert len(results) == 3
        for result in results:
            assert result.image is not None

    def test_proof_result_to_dict(self, proofer, gray_image):
        """Test proof result conversion to dictionary."""
        result = proofer.proof(gray_image)
        d = result.to_dict()

        assert "size" in d
        assert "paper_white" in d
        assert "dmax" in d
        assert "platinum_ratio" in d

    def test_paper_presets(self):
        """Test getting paper presets list."""
        presets = SoftProofer.get_paper_presets()

        assert len(presets) >= 5
        assert any("arches" in p[1].lower() for p in presets)

    def test_dmax_range(self):
        """Test getting typical Dmax range."""
        min_dmax, max_dmax = SoftProofer.get_dmax_range()

        assert 1.0 <= min_dmax < max_dmax <= 2.0


class TestPaperPresets:
    """Test paper presets."""

    def test_all_presets_exist(self):
        """Test all standard presets exist."""
        assert PaperSimulation.ARCHES_PLATINE in PAPER_PRESETS
        assert PaperSimulation.BERGGER_COT320 in PAPER_PRESETS
        assert PaperSimulation.HAHNEMUHLE_PLATINUM in PAPER_PRESETS
        assert PaperSimulation.REVERE_PLATINUM in PAPER_PRESETS
        assert PaperSimulation.STONEHENGE in PAPER_PRESETS

    def test_presets_have_required_fields(self):
        """Test presets have all required fields."""
        for preset, data in PAPER_PRESETS.items():
            assert "white_rgb" in data
            assert "dmax" in data
            assert "dmin" in data
            assert "tone" in data

    def test_preset_white_rgb_valid(self):
        """Test preset white RGB values are valid."""
        for preset, data in PAPER_PRESETS.items():
            rgb = data["white_rgb"]
            assert len(rgb) == 3
            for value in rgb:
                assert 200 <= value <= 255

    def test_preset_dmax_valid(self):
        """Test preset Dmax values are valid."""
        for preset, data in PAPER_PRESETS.items():
            assert 1.0 <= data["dmax"] <= 2.0

    def test_preset_dmin_valid(self):
        """Test preset Dmin values are valid."""
        for preset, data in PAPER_PRESETS.items():
            assert 0.0 <= data["dmin"] <= 0.2
