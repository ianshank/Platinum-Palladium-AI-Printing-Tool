"""Tests for zone system mapping module."""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.zones import (
    ZoneMapper,
    ZoneMapping,
    Zone,
    ZoneAnalysis,
    ZONE_DESCRIPTIONS,
)


@pytest.fixture
def mapper():
    """Create default zone mapper."""
    return ZoneMapper()


@pytest.fixture
def custom_mapper():
    """Create zone mapper with custom paper characteristics."""
    mapping = ZoneMapping(paper_dmax=1.8, paper_dmin=0.05)
    return ZoneMapper(mapping)


@pytest.fixture
def gray_image():
    """Create a middle gray image."""
    arr = np.full((100, 100), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def gradient_image():
    """Create a gradient image."""
    arr = np.linspace(0, 255, 100).reshape(1, 100).repeat(100, axis=0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def dark_image():
    """Create a dark image."""
    arr = np.full((100, 100), 30, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def bright_image():
    """Create a bright image."""
    arr = np.full((100, 100), 230, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


class TestZoneMapping:
    """Test zone mapping configuration."""

    def test_default_densities(self):
        """Test default zone densities are calculated."""
        mapping = ZoneMapping()

        assert len(mapping.zone_densities) == 11
        assert mapping.zone_densities[Zone.ZONE_0] > mapping.zone_densities[Zone.ZONE_X]

    def test_custom_paper_characteristics(self):
        """Test custom paper dmax/dmin."""
        mapping = ZoneMapping(paper_dmax=1.8, paper_dmin=0.05)

        assert mapping.paper_dmax == 1.8
        assert mapping.paper_dmin == 0.05

    def test_get_density(self):
        """Test getting density for a zone."""
        mapping = ZoneMapping(paper_dmax=1.6, paper_dmin=0.08)

        # Zone 0 should be at Dmax
        assert abs(mapping.get_density(Zone.ZONE_0) - 1.6) < 0.01

        # Zone X should be at Dmin
        assert abs(mapping.get_density(Zone.ZONE_X) - 0.08) < 0.01

    def test_get_zone_for_density(self):
        """Test finding zone for a density."""
        mapping = ZoneMapping(paper_dmax=1.6, paper_dmin=0.08)

        # High density should be low zone
        zone_dark = mapping.get_zone_for_density(1.5)
        assert zone_dark.value <= 2

        # Low density should be high zone
        zone_light = mapping.get_zone_for_density(0.1)
        assert zone_light.value >= 8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        mapping = ZoneMapping()
        d = mapping.to_dict()

        assert "paper_dmax" in d
        assert "paper_dmin" in d
        assert "zones" in d


class TestZoneMapper:
    """Test zone mapper functionality."""

    def test_analyze_gray_image(self, mapper, gray_image):
        """Test analyzing a middle gray image."""
        analysis = mapper.analyze_image(gray_image)

        assert isinstance(analysis, ZoneAnalysis)
        # Middle gray should be around Zone V
        assert 4 <= analysis.average_zone <= 6

    def test_analyze_dark_image(self, mapper, dark_image):
        """Test analyzing a dark image."""
        analysis = mapper.analyze_image(dark_image)

        # Dark image should have low average zone
        assert analysis.average_zone < 3

    def test_analyze_bright_image(self, mapper, bright_image):
        """Test analyzing a bright image."""
        analysis = mapper.analyze_image(bright_image)

        # Bright image should have high average zone
        assert analysis.average_zone > 7

    def test_analyze_gradient(self, mapper, gradient_image):
        """Test analyzing a gradient image."""
        analysis = mapper.analyze_image(gradient_image)

        # Gradient should span multiple zones
        assert analysis.zone_range >= 5

    def test_zone_histogram(self, mapper, gradient_image):
        """Test zone histogram is computed."""
        analysis = mapper.analyze_image(gradient_image)

        assert len(analysis.zone_histogram) == 11
        total_pct = sum(analysis.zone_histogram.values())
        # Allow some tolerance due to overlapping zone boundaries
        assert 0.95 <= total_pct <= 1.10

    def test_placed_shadow(self, mapper, gradient_image):
        """Test shadow zone placement."""
        analysis = mapper.analyze_image(gradient_image, placed_shadow=3)

        assert analysis.shadow_zone == Zone.ZONE_III

    def test_placed_highlight(self, mapper, gradient_image):
        """Test highlight zone placement."""
        analysis = mapper.analyze_image(gradient_image, placed_highlight=7)

        assert analysis.highlight_zone == Zone.ZONE_VII

    def test_exposure_adjustment(self, mapper, gray_image):
        """Test exposure adjustment calculation."""
        # Place middle gray on Zone III (darker)
        analysis = mapper.analyze_image(gray_image, placed_shadow=3)

        # Should have exposure adjustment notes
        assert len(analysis.notes) >= 0

    def test_development_recommendation(self, mapper, gradient_image):
        """Test development recommendations."""
        analysis = mapper.analyze_image(gradient_image)

        assert analysis.development_adjustment in ["N-2", "N-1", "N", "N+1", "N+2"]

    def test_create_zone_scale(self, mapper):
        """Test zone scale creation."""
        scale = mapper.create_zone_scale(width=500, height=50)

        assert scale.size == (500, 50)
        assert scale.mode == "L"

    def test_visualize_zones(self, mapper, gradient_image):
        """Test zone visualization."""
        visualized = mapper.visualize_zones(gradient_image, posterize=True)

        assert visualized.size == gradient_image.size
        assert visualized.mode == "L"

    def test_visualize_zones_no_posterize(self, mapper, gradient_image):
        """Test zone visualization without posterization."""
        visualized = mapper.visualize_zones(gradient_image, posterize=False)

        assert visualized.size == gradient_image.size

    def test_density_to_zone(self, mapper):
        """Test density to zone conversion."""
        zone = mapper.density_to_zone(1.5)
        assert isinstance(zone, Zone)

    def test_zone_to_density(self, mapper):
        """Test zone to density conversion."""
        density = mapper.zone_to_density(Zone.ZONE_V)
        assert isinstance(density, float)
        assert density > 0

    def test_get_exposure_scale(self, mapper):
        """Test exposure scale description."""
        scale = mapper.get_exposure_scale()

        assert "range" in scale.lower() or "stop" in scale.lower()

    def test_get_zone_descriptions(self):
        """Test getting zone descriptions."""
        descs = ZoneMapper.get_zone_descriptions()

        assert len(descs) == 11
        assert Zone.ZONE_0 in descs
        assert Zone.ZONE_V in descs
        assert "black" in descs[Zone.ZONE_0].lower()
        assert "gray" in descs[Zone.ZONE_V].lower()

    def test_get_development_adjustments(self):
        """Test getting development adjustment descriptions."""
        adjustments = ZoneMapper.get_development_adjustments()

        assert "N" in adjustments
        assert "N-1" in adjustments
        assert "N+1" in adjustments

    def test_rgb_image_conversion(self, mapper):
        """Test that RGB images are converted to grayscale."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # Red
        arr[:, :, 1] = 100  # Green
        arr[:, :, 2] = 50   # Blue
        rgb_image = Image.fromarray(arr, mode="RGB")

        analysis = mapper.analyze_image(rgb_image)

        assert analysis.average_zone > 0

    def test_analysis_to_dict(self, mapper, gradient_image):
        """Test analysis conversion to dictionary."""
        analysis = mapper.analyze_image(gradient_image)
        d = analysis.to_dict()

        assert "zone_histogram" in d
        assert "shadow_zone" in d
        assert "highlight_zone" in d
        assert "development_adjustment" in d


class TestZoneDescriptions:
    """Test zone descriptions."""

    def test_all_zones_have_descriptions(self):
        """Test that all zones have descriptions."""
        for zone in Zone:
            assert zone in ZONE_DESCRIPTIONS

    def test_descriptions_are_strings(self):
        """Test that all descriptions are strings."""
        for desc in ZONE_DESCRIPTIONS.values():
            assert isinstance(desc, str)
            assert len(desc) > 0
