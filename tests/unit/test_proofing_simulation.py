"""Tests for the soft proofing simulation module.

Covers PaperSimulation, ProofSettings, ProofResult, and SoftProofer classes.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.proofing.simulation import (
    PAPER_PRESETS,
    PaperSimulation,
    ProofResult,
    ProofSettings,
    SoftProofer,
)

# ── PaperSimulation enum tests ──────────────────────────────────────

class TestPaperSimulation:
    def test_preset_count(self) -> None:
        assert len(PaperSimulation) == 6

    def test_custom_exists(self) -> None:
        assert PaperSimulation.CUSTOM.value == "custom"

    def test_all_presets_have_data(self) -> None:
        for preset in PaperSimulation:
            if preset != PaperSimulation.CUSTOM:
                assert preset in PAPER_PRESETS

    @pytest.mark.parametrize("preset", [p for p in PaperSimulation if p != PaperSimulation.CUSTOM])
    def test_preset_data_structure(self, preset: PaperSimulation) -> None:
        data = PAPER_PRESETS[preset]
        assert "white_rgb" in data
        assert "dmax" in data
        assert "dmin" in data
        assert "tone" in data
        assert len(data["white_rgb"]) == 3
        assert data["dmax"] > data["dmin"]


# ── ProofSettings tests ─────────────────────────────────────────────

class TestProofSettings:
    def test_defaults(self) -> None:
        settings = ProofSettings()
        assert settings.paper_white_rgb == (250, 246, 238)
        assert settings.paper_dmax == 1.6
        assert settings.paper_dmin == 0.07
        assert settings.platinum_ratio == 0.0
        assert settings.add_paper_texture is False

    def test_from_paper_preset(self) -> None:
        settings = ProofSettings.from_paper_preset(PaperSimulation.ARCHES_PLATINE)
        assert settings.paper_white_rgb == (250, 246, 238)
        assert settings.paper_dmax == 1.6

    def test_from_custom_preset(self) -> None:
        settings = ProofSettings.from_paper_preset(PaperSimulation.CUSTOM)
        # Should use defaults
        assert isinstance(settings.paper_dmax, float)

    @pytest.mark.parametrize("preset", [p for p in PaperSimulation if p != PaperSimulation.CUSTOM])
    def test_all_presets_create_settings(self, preset: PaperSimulation) -> None:
        settings = ProofSettings.from_paper_preset(preset)
        assert isinstance(settings, ProofSettings)
        assert settings.paper_dmax > 0


# ── ProofResult tests ────────────────────────────────────────────────

class TestProofResult:
    def test_to_dict(self) -> None:
        img = Image.new("RGB", (100, 100))
        settings = ProofSettings()
        result = ProofResult(image=img, settings=settings, original_size=(100, 100))
        d = result.to_dict()
        assert "size" in d
        assert "original_size" in d
        assert "dmax" in d
        assert "notes" in d


# ── SoftProofer tests ────────────────────────────────────────────────

class TestSoftProofer:
    @pytest.fixture()
    def proofer(self) -> SoftProofer:
        return SoftProofer()

    @pytest.fixture()
    def gray_image(self) -> Image.Image:
        arr = np.full((50, 50), 128, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture()
    def gradient_image(self) -> Image.Image:
        arr = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (50, 1))
        return Image.fromarray(arr, mode="L")

    @pytest.fixture()
    def rgb_image(self) -> Image.Image:
        arr = np.full((50, 50, 3), 128, dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_proof_returns_result(
        self, proofer: SoftProofer, gray_image: Image.Image
    ) -> None:
        result = proofer.proof(gray_image)
        assert isinstance(result, ProofResult)
        assert isinstance(result.image, Image.Image)
        assert result.image.mode == "RGB"

    def test_proof_preserves_size(
        self, proofer: SoftProofer, gray_image: Image.Image
    ) -> None:
        result = proofer.proof(gray_image)
        assert result.image.size == gray_image.size
        assert result.original_size == gray_image.size

    def test_proof_rgb_conversion(
        self, proofer: SoftProofer, rgb_image: Image.Image
    ) -> None:
        result = proofer.proof(rgb_image)
        assert isinstance(result, ProofResult)
        assert any("grayscale" in note.lower() for note in result.notes)

    def test_proof_with_texture(self, gray_image: Image.Image) -> None:
        settings = ProofSettings(add_paper_texture=True, texture_strength=0.5)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)
        assert any("texture" in note.lower() for note in result.notes)

    def test_proof_warm_tone(self, gray_image: Image.Image) -> None:
        settings = ProofSettings(platinum_ratio=0.0)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)
        assert any("warm" in note.lower() for note in result.notes)

    def test_proof_cool_tone(self, gray_image: Image.Image) -> None:
        settings = ProofSettings(platinum_ratio=0.8)
        proofer = SoftProofer(settings=settings)
        result = proofer.proof(gray_image)
        assert any("cool" in note.lower() for note in result.notes)

    def test_proof_settings_override(
        self, proofer: SoftProofer, gray_image: Image.Image
    ) -> None:
        override = ProofSettings(paper_dmax=2.0, paper_dmin=0.1)
        result = proofer.proof(gray_image, settings=override)
        assert "2.00 Dmax" in " ".join(result.notes)

    def test_compare_multiple_settings(
        self, proofer: SoftProofer, gray_image: Image.Image
    ) -> None:
        settings_list = [
            ProofSettings.from_paper_preset(PaperSimulation.ARCHES_PLATINE),
            ProofSettings.from_paper_preset(PaperSimulation.BERGGER_COT320),
        ]
        results = proofer.compare(gray_image, settings_list)
        assert len(results) == 2
        assert all(isinstance(r, ProofResult) for r in results)

    def test_viewing_brightness(self, gray_image: Image.Image) -> None:
        bright = SoftProofer(ProofSettings(viewing_brightness=1.5))
        dim = SoftProofer(ProofSettings(viewing_brightness=0.5))
        r_bright = bright.proof(gray_image)
        r_dim = dim.proof(gray_image)
        # Bright version should have higher mean pixel values
        mean_bright = np.array(r_bright.image).mean()
        mean_dim = np.array(r_dim.image).mean()
        assert mean_bright > mean_dim

    def test_color_temperature(self, gray_image: Image.Image) -> None:
        warm = SoftProofer(ProofSettings(ambient_light_temperature=3500))
        cool = SoftProofer(ProofSettings(ambient_light_temperature=7500))
        r_warm = warm.proof(gray_image)
        r_cool = cool.proof(gray_image)
        # Both should produce valid images
        assert r_warm.image.size == r_cool.image.size

    def test_get_paper_presets(self) -> None:
        presets = SoftProofer.get_paper_presets()
        assert len(presets) == 6
        assert all(isinstance(p, tuple) and len(p) == 2 for p in presets)
        values = [p[0] for p in presets]
        assert "arches_platine" in values
        assert "custom" in values

    def test_get_dmax_range(self) -> None:
        dmax_range = SoftProofer.get_dmax_range()
        assert len(dmax_range) == 2
        assert dmax_range[0] < dmax_range[1]
        assert 1.0 <= dmax_range[0] <= 2.0
        assert 1.0 <= dmax_range[1] <= 2.5

    def test_gradient_produces_tonal_range(
        self, proofer: SoftProofer, gradient_image: Image.Image
    ) -> None:
        result = proofer.proof(gradient_image)
        arr = np.array(result.image)
        # Should have significant tonal range
        assert arr.max() - arr.min() > 50
