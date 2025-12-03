"""
Tests for soft proofing simulation module.

Tests print appearance simulation for alternative processes.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.proofing.simulation import (
    PaperSimulation,
    ProofSettings,
    ProofResult,
    SoftProofer,
    PAPER_PRESETS,
)


class TestPaperSimulation:
    """Tests for PaperSimulation enum."""

    def test_all_presets_exist(self):
        """All paper presets should exist."""
        assert PaperSimulation.ARCHES_PLATINE.value == "arches_platine"
        assert PaperSimulation.BERGGER_COT320.value == "bergger_cot320"
        assert PaperSimulation.HAHNEMUHLE_PLATINUM.value == "hahnemuhle_platinum"
        assert PaperSimulation.REVERE_PLATINUM.value == "revere_platinum"
        assert PaperSimulation.STONEHENGE.value == "stonehenge"
        assert PaperSimulation.CUSTOM.value == "custom"

    def test_presets_have_data(self):
        """All paper presets should have data."""
        for preset in PaperSimulation:
            if preset != PaperSimulation.CUSTOM:
                assert preset in PAPER_PRESETS


class TestProofSettings:
    """Tests for ProofSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = ProofSettings()
        assert len(settings.paper_white_rgb) == 3
        assert settings.paper_dmax > settings.paper_dmin
        assert 0 <= settings.platinum_ratio <= 1

    def test_custom_settings(self):
        """Custom settings should be applied."""
        settings = ProofSettings(
            paper_white_rgb=(240, 235, 220),
            paper_dmax=1.5,
            paper_dmin=0.1,
            platinum_ratio=0.7,
        )
        assert settings.paper_white_rgb == (240, 235, 220)
        assert settings.paper_dmax == 1.5
        assert settings.platinum_ratio == 0.7

    def test_from_paper_preset(self):
        """Should create settings from preset."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.BERGGER_COT320)

        assert settings.paper_white_rgb == PAPER_PRESETS[PaperSimulation.BERGGER_COT320]["white_rgb"]
        assert settings.paper_dmax == PAPER_PRESETS[PaperSimulation.BERGGER_COT320]["dmax"]

    def test_from_paper_preset_custom(self):
        """Custom preset should return defaults."""
        settings = ProofSettings.from_paper_preset(PaperSimulation.CUSTOM)
        default = ProofSettings()
        assert settings.paper_white_rgb == default.paper_white_rgb


class TestProofResult:
    """Tests for ProofResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample proof result."""
        img = Image.new("RGB", (100, 100), color=(200, 195, 185))
        settings = ProofSettings()
        return ProofResult(
            image=img,
            settings=settings,
            original_size=(100, 100),
            notes=["Test note"],
        )

    def test_to_dict(self, sample_result):
        """Should serialize to dictionary."""
        d = sample_result.to_dict()
        assert "size" in d
        assert "original_size" in d
        assert "paper_white" in d
        assert "dmax" in d
        assert "notes" in d
        assert d["notes"] == ["Test note"]


class TestSoftProofer:
    """Tests for SoftProofer class."""

    @pytest.fixture
    def proofer(self):
        """Create soft proofer with default settings."""
        return SoftProofer()

    @pytest.fixture
    def gray_image(self):
        """Create grayscale test image."""
        arr = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture
    def rgb_image(self):
        """Create RGB test image."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 100
        arr[:, :, 1] = 150
        arr[:, :, 2] = 200
        return Image.fromarray(arr, mode="RGB")

    def test_proofer_default_settings(self, proofer):
        """Proofer should use default settings."""
        assert proofer.settings is not None
        assert proofer.settings.paper_dmax == 1.6

    def test_proofer_custom_settings(self):
        """Proofer should accept custom settings."""
        settings = ProofSettings(paper_dmax=1.7)
        proofer = SoftProofer(settings=settings)
        assert proofer.settings.paper_dmax == 1.7

    def test_proof_grayscale(self, proofer, gray_image):
        """Should proof grayscale image."""
        result = proofer.proof(gray_image)

        assert result.image is not None
        assert result.image.mode == "RGB"
        assert result.image.size == gray_image.size
        assert result.original_size == gray_image.size

    def test_proof_rgb_converts(self, proofer, rgb_image):
        """Should convert RGB to grayscale for proofing."""
        result = proofer.proof(rgb_image)

        assert result.image.mode == "RGB"
        assert any("grayscale" in note.lower() for note in result.notes)

    def test_proof_with_override_settings(self, proofer, gray_image):
        """Should use override settings when provided."""
        override = ProofSettings(paper_dmax=1.8)
        result = proofer.proof(gray_image, settings=override)

        assert result.settings.paper_dmax == 1.8

    def test_proof_with_texture(self, gray_image):
        """Should apply paper texture."""
        settings = ProofSettings(add_paper_texture=True, texture_strength=0.2)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)

        assert any("texture" in note.lower() for note in result.notes)

    def test_proof_platinum_tone(self, gray_image):
        """Should note platinum tones."""
        settings = ProofSettings(platinum_ratio=0.8)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)

        assert any("platinum" in note.lower() or "cool" in note.lower() for note in result.notes)

    def test_proof_palladium_tone(self, gray_image):
        """Should note palladium tones."""
        settings = ProofSettings(platinum_ratio=0.2)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)

        assert any("palladium" in note.lower() or "warm" in note.lower() for note in result.notes)

    def test_compare_multiple_settings(self, proofer, gray_image):
        """Should compare multiple settings."""
        settings_list = [
            ProofSettings(paper_dmax=1.5),
            ProofSettings(paper_dmax=1.6),
            ProofSettings(paper_dmax=1.7),
        ]

        results = proofer.compare(gray_image, settings_list)

        assert len(results) == 3
        assert results[0].settings.paper_dmax == 1.5
        assert results[2].settings.paper_dmax == 1.7

    def test_proof_viewing_brightness(self, gray_image):
        """Should apply viewing brightness."""
        settings_bright = ProofSettings(viewing_brightness=1.3)
        settings_dim = ProofSettings(viewing_brightness=0.7)

        proofer_bright = SoftProofer(settings=settings_bright)
        proofer_dim = SoftProofer(settings=settings_dim)

        result_bright = proofer_bright.proof(gray_image)
        result_dim = proofer_dim.proof(gray_image)

        # Bright should be lighter on average
        bright_arr = np.array(result_bright.image)
        dim_arr = np.array(result_dim.image)
        assert bright_arr.mean() > dim_arr.mean()

    def test_proof_color_temperature_warm(self, gray_image):
        """Should adjust for warm lighting."""
        settings = ProofSettings(ambient_light_temperature=3500)  # Warm
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)

        # Should produce valid image
        assert result.image is not None

    def test_proof_color_temperature_cool(self, gray_image):
        """Should adjust for cool lighting."""
        settings = ProofSettings(ambient_light_temperature=7500)  # Cool
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)

        # Should produce valid image
        assert result.image is not None

    def test_get_paper_presets(self):
        """Should return paper preset list."""
        presets = SoftProofer.get_paper_presets()

        assert len(presets) > 0
        assert all(isinstance(p, tuple) for p in presets)
        assert all(len(p) == 2 for p in presets)

        # Check some known presets
        values = [p[0] for p in presets]
        assert "arches_platine" in values
        assert "bergger_cot320" in values

    def test_get_dmax_range(self):
        """Should return Dmax range."""
        dmin, dmax = SoftProofer.get_dmax_range()
        assert dmin < dmax
        assert 1.0 <= dmin <= 1.5
        assert 1.5 <= dmax <= 2.0


class TestSoftProoferEdgeCases:
    """Tests for edge cases in soft proofing."""

    def test_proof_tiny_image(self):
        """Should handle tiny images."""
        proofer = SoftProofer()
        img = Image.new("L", (2, 2), color=128)
        result = proofer.proof(img)
        assert result.image.size == (2, 2)

    def test_proof_large_image(self):
        """Should handle larger images."""
        proofer = SoftProofer()
        img = Image.new("L", (500, 500), color=128)
        result = proofer.proof(img)
        assert result.image.size == (500, 500)

    def test_proof_all_black(self):
        """Should handle all-black image."""
        proofer = SoftProofer()
        img = Image.new("L", (50, 50), color=0)
        result = proofer.proof(img)
        assert result.image is not None

    def test_proof_all_white(self):
        """Should handle all-white image."""
        proofer = SoftProofer()
        img = Image.new("L", (50, 50), color=255)
        result = proofer.proof(img)
        assert result.image is not None

    def test_proof_with_alpha_channel(self):
        """Should handle image with alpha channel."""
        proofer = SoftProofer()
        # LA mode = grayscale with alpha
        img = Image.new("LA", (50, 50), color=(128, 255))
        result = proofer.proof(img)
        assert result.image.mode == "RGB"

    def test_proof_all_presets(self):
        """All presets should produce valid proofs."""
        proofer = SoftProofer()
        img = Image.new("L", (50, 50), color=128)

        for preset in PaperSimulation:
            settings = ProofSettings.from_paper_preset(preset)
            result = proofer.proof(img, settings=settings)
            assert result.image is not None
            assert result.image.mode == "RGB"


class TestSoftProoferToning:
    """Tests for toning/color rendering."""

    def test_platinum_vs_palladium_tone(self):
        """Platinum and palladium should produce different tones."""
        img = Image.new("L", (50, 50), color=128)

        platinum_settings = ProofSettings(platinum_ratio=1.0)
        palladium_settings = ProofSettings(platinum_ratio=0.0)

        pt_proofer = SoftProofer(settings=platinum_settings)
        pd_proofer = SoftProofer(settings=palladium_settings)

        pt_result = pt_proofer.proof(img)
        pd_result = pd_proofer.proof(img)

        # They should produce different colors
        pt_arr = np.array(pt_result.image)
        pd_arr = np.array(pd_result.image)

        # Color difference should be non-zero
        assert not np.allclose(pt_arr, pd_arr)

    def test_paper_white_affects_highlights(self):
        """Paper white should affect highlight areas."""
        img = Image.new("L", (50, 50), color=250)  # Near white

        cool_settings = ProofSettings(paper_white_rgb=(255, 255, 255))
        warm_settings = ProofSettings(paper_white_rgb=(250, 240, 220))

        cool_proofer = SoftProofer(settings=cool_settings)
        warm_proofer = SoftProofer(settings=warm_settings)

        cool_result = cool_proofer.proof(img)
        warm_result = warm_proofer.proof(img)

        cool_arr = np.array(cool_result.image)
        warm_arr = np.array(warm_result.image)

        # Warm should have more red relative to blue
        cool_rb_ratio = cool_arr[:, :, 0].mean() / max(cool_arr[:, :, 2].mean(), 1)
        warm_rb_ratio = warm_arr[:, :, 0].mean() / max(warm_arr[:, :, 2].mean(), 1)

        assert warm_rb_ratio > cool_rb_ratio
