"""Tests for the Zone System mapping module.

Covers Zone enum, ZoneMapping, ZoneAnalysis, and ZoneMapper classes.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.zones.mapping import (
    ZONE_DESCRIPTIONS,
    Zone,
    ZoneAnalysis,
    ZoneMapper,
    ZoneMapping,
)


# ── Zone enum tests ──────────────────────────────────────────────────

class TestZone:
    def test_zone_values_range(self) -> None:
        assert Zone.ZONE_0.value == 0
        assert Zone.ZONE_X.value == 10

    def test_zone_count(self) -> None:
        assert len(Zone) == 11

    def test_zone_ordering(self) -> None:
        zones = list(Zone)
        for i in range(len(zones) - 1):
            assert zones[i].value < zones[i + 1].value

    def test_zone_descriptions_cover_all_zones(self) -> None:
        for zone in Zone:
            assert zone in ZONE_DESCRIPTIONS
            assert isinstance(ZONE_DESCRIPTIONS[zone], str)
            assert len(ZONE_DESCRIPTIONS[zone]) > 0


# ── ZoneMapping tests ────────────────────────────────────────────────

class TestZoneMapping:
    def test_default_densities_computed(self) -> None:
        mapping = ZoneMapping()
        assert len(mapping.zone_densities) == 11

    def test_zone_0_is_dmax(self) -> None:
        mapping = ZoneMapping(paper_dmax=2.0, paper_dmin=0.05)
        assert mapping.zone_densities[Zone.ZONE_0] == 2.0

    def test_zone_x_is_dmin(self) -> None:
        mapping = ZoneMapping(paper_dmax=2.0, paper_dmin=0.05)
        assert mapping.zone_densities[Zone.ZONE_X] == pytest.approx(0.05, abs=0.01)

    def test_densities_monotonically_decrease(self) -> None:
        mapping = ZoneMapping()
        zones = list(Zone)
        for i in range(len(zones) - 1):
            assert mapping.zone_densities[zones[i]] >= mapping.zone_densities[zones[i + 1]]

    def test_get_density_returns_float(self) -> None:
        mapping = ZoneMapping()
        d = mapping.get_density(Zone.ZONE_V)
        assert isinstance(d, float)
        assert 0.0 <= d <= 3.0

    def test_get_density_missing_zone_returns_zero(self) -> None:
        mapping = ZoneMapping(zone_densities={Zone.ZONE_0: 1.5})
        assert mapping.get_density(Zone.ZONE_X) == 0.0

    def test_get_zone_for_density_exact_match(self) -> None:
        mapping = ZoneMapping()
        zone_v_density = mapping.zone_densities[Zone.ZONE_V]
        assert mapping.get_zone_for_density(zone_v_density) == Zone.ZONE_V

    def test_get_zone_for_density_closest(self) -> None:
        mapping = ZoneMapping(paper_dmax=2.0, paper_dmin=0.0)
        assert mapping.get_zone_for_density(2.0) == Zone.ZONE_0
        assert mapping.get_zone_for_density(0.0) == Zone.ZONE_X

    @pytest.mark.parametrize("dmax,dmin", [
        (1.6, 0.08),
        (2.0, 0.05),
        (2.5, 0.10),
    ])
    def test_custom_paper_characteristics(self, dmax: float, dmin: float) -> None:
        mapping = ZoneMapping(paper_dmax=dmax, paper_dmin=dmin)
        assert mapping.zone_densities[Zone.ZONE_0] == dmax
        assert mapping.zone_densities[Zone.ZONE_X] == pytest.approx(dmin, abs=0.01)

    def test_to_dict_structure(self) -> None:
        mapping = ZoneMapping()
        d = mapping.to_dict()
        assert "paper_dmax" in d
        assert "paper_dmin" in d
        assert "zones" in d
        assert len(d["zones"]) == 11
        for zone_name, info in d["zones"].items():
            assert "value" in info
            assert "density" in info
            assert "description" in info


# ── ZoneAnalysis tests ───────────────────────────────────────────────

class TestZoneAnalysis:
    def test_defaults(self) -> None:
        analysis = ZoneAnalysis()
        assert analysis.average_zone == 5.0
        assert analysis.development_adjustment == "N"
        assert analysis.exposure_adjustment_stops == 0.0

    def test_to_dict_keys(self) -> None:
        analysis = ZoneAnalysis()
        d = analysis.to_dict()
        expected_keys = {
            "zone_histogram", "shadow_zone", "highlight_zone",
            "average_zone", "zone_range", "exposure_adjustment_stops",
            "development_adjustment", "notes",
        }
        assert set(d.keys()) == expected_keys


# ── ZoneMapper tests ─────────────────────────────────────────────────

class TestZoneMapper:
    @pytest.fixture()
    def mapper(self) -> ZoneMapper:
        return ZoneMapper()

    @pytest.fixture()
    def gray_image(self) -> Image.Image:
        """Create a mid-gray test image."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture()
    def gradient_image(self) -> Image.Image:
        """Create a gradient image spanning full tonal range."""
        arr = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (50, 1))
        return Image.fromarray(arr, mode="L")

    def test_analyze_image_returns_analysis(
        self, mapper: ZoneMapper, gray_image: Image.Image
    ) -> None:
        result = mapper.analyze_image(gray_image)
        assert isinstance(result, ZoneAnalysis)

    def test_analyze_image_histogram_sums_to_one(
        self, mapper: ZoneMapper, gray_image: Image.Image
    ) -> None:
        result = mapper.analyze_image(gray_image)
        total = sum(result.zone_histogram.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_analyze_gradient_has_spread(
        self, mapper: ZoneMapper, gradient_image: Image.Image
    ) -> None:
        result = mapper.analyze_image(gradient_image)
        assert result.zone_range >= 5

    def test_analyze_image_with_placed_shadow(
        self, mapper: ZoneMapper, gradient_image: Image.Image
    ) -> None:
        result = mapper.analyze_image(gradient_image, placed_shadow=3)
        assert result.shadow_zone == Zone.ZONE_III

    def test_analyze_image_with_placed_highlight(
        self, mapper: ZoneMapper, gradient_image: Image.Image
    ) -> None:
        result = mapper.analyze_image(gradient_image, placed_highlight=8)
        assert result.highlight_zone == Zone.ZONE_VIII

    def test_analyze_high_contrast_recommends_n_minus(
        self, mapper: ZoneMapper
    ) -> None:
        """Very bright and dark values = high contrast = N- development."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:50, :] = 255  # Half white, half black
        image = Image.fromarray(arr, mode="L")
        result = mapper.analyze_image(image)
        assert result.development_adjustment.startswith("N-")

    def test_analyze_low_contrast_recommends_n_plus(
        self, mapper: ZoneMapper
    ) -> None:
        """Very narrow tonal range = low contrast = N+ development."""
        arr = np.random.default_rng(42).integers(100, 150, (100, 100), dtype=np.uint8)
        image = Image.fromarray(arr, mode="L")
        result = mapper.analyze_image(image)
        assert result.development_adjustment.startswith("N+") or result.development_adjustment == "N"

    def test_analyze_rgb_image(self, mapper: ZoneMapper) -> None:
        """RGB images should be converted to grayscale internally."""
        arr = np.full((50, 50, 3), 128, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = mapper.analyze_image(image)
        assert isinstance(result, ZoneAnalysis)

    def test_create_zone_scale(self, mapper: ZoneMapper) -> None:
        scale = mapper.create_zone_scale(width=500, height=50)
        assert isinstance(scale, Image.Image)
        assert scale.size == (500, 50)
        assert scale.mode == "L"

    def test_create_zone_scale_custom_size(self, mapper: ZoneMapper) -> None:
        scale = mapper.create_zone_scale(width=220, height=30)
        assert scale.size == (220, 30)

    def test_visualize_zones_posterized(
        self, mapper: ZoneMapper, gradient_image: Image.Image
    ) -> None:
        result = mapper.visualize_zones(gradient_image, posterize=True)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        # Posterized image should have fewer unique values
        unique_values = len(np.unique(np.array(result)))
        assert unique_values <= 11

    def test_visualize_zones_not_posterized(
        self, mapper: ZoneMapper, gradient_image: Image.Image
    ) -> None:
        result = mapper.visualize_zones(gradient_image, posterize=False)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_density_to_zone(self, mapper: ZoneMapper) -> None:
        zone = mapper.density_to_zone(1.6)  # High density = dark zone
        assert isinstance(zone, Zone)

    def test_zone_to_density(self, mapper: ZoneMapper) -> None:
        density = mapper.zone_to_density(Zone.ZONE_V)
        assert isinstance(density, float)
        assert 0.0 <= density <= 3.0

    def test_density_zone_roundtrip(self, mapper: ZoneMapper) -> None:
        """Converting zone->density->zone should return close to original."""
        for zone in Zone:
            density = mapper.zone_to_density(zone)
            recovered = mapper.density_to_zone(density)
            assert recovered == zone

    def test_get_exposure_scale(self, mapper: ZoneMapper) -> None:
        result = mapper.get_exposure_scale()
        assert isinstance(result, str)
        assert "Paper range:" in result
        assert "stops" in result

    def test_get_zone_descriptions_returns_copy(self) -> None:
        descs = ZoneMapper.get_zone_descriptions()
        assert descs == ZONE_DESCRIPTIONS
        # Modifying copy shouldn't affect original
        descs[Zone.ZONE_0] = "modified"
        assert ZONE_DESCRIPTIONS[Zone.ZONE_0] != "modified"

    def test_get_development_adjustments(self) -> None:
        adjustments = ZoneMapper.get_development_adjustments()
        assert "N" in adjustments
        assert "N-1" in adjustments
        assert "N+1" in adjustments
        assert "N-2" in adjustments
        assert "N+2" in adjustments

    def test_custom_mapping(self) -> None:
        mapping = ZoneMapping(paper_dmax=2.5, paper_dmin=0.1)
        mapper = ZoneMapper(mapping=mapping)
        assert mapper.zone_to_density(Zone.ZONE_0) == 2.5
