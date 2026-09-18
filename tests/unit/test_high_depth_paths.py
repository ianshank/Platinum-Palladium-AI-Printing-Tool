"""Reading an image must not depend on the bit depth it was stored at.

``open_image_safely`` deliberately preserves ``I;16`` and ``I``, and Pillow's
``convert("L")`` clips those at 255 rather than scaling, so a 16-bit scan
arrived as very nearly solid white. Every module that took a grayscale array of
an image it had not decoded itself carried the same defect. These tests drive
each one at both depths and require the same answer, because the 8-bit
rendering of the same file is the ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.imaging.processor import as_eight_bit_gray

#: 0..65535 across 256 samples, so the 8-bit rendering is an even 0..255 ramp.
SIXTEEN_BIT_RAMP = np.linspace(0, 65535, 256, dtype=np.uint16).reshape(16, 16)
EIGHT_TO_SIXTEEN = 257


@pytest.fixture
def pair() -> tuple[Image.Image, Image.Image]:
    """The same picture as a 16-bit image and as its true 8-bit rendering."""
    wide = Image.fromarray(SIXTEEN_BIT_RAMP)
    narrow = Image.fromarray((SIXTEEN_BIT_RAMP // EIGHT_TO_SIXTEEN).astype(np.uint8))
    assert wide.mode == "I;16"
    assert narrow.mode == "L"
    return wide, narrow


class TestSharedHelper:
    """``as_eight_bit_gray`` is the one replacement for a bare ``convert("L")``."""

    def test_high_depth_is_scaled_not_clipped(self, pair) -> None:
        wide, narrow = pair

        assert np.array_equal(np.array(as_eight_bit_gray(wide)), np.array(narrow))

    def test_clipping_is_what_it_replaces(self, pair) -> None:
        """Pin the old behaviour so the difference stays visible."""
        wide, _ = pair

        assert np.array(wide.convert("L")).mean() == pytest.approx(254.0, abs=1.0)
        assert np.array(as_eight_bit_gray(wide)).mean() == pytest.approx(127.5, abs=1.0)

    def test_an_eight_bit_image_is_returned_unchanged(self, pair) -> None:
        """Cheap enough to call unconditionally."""
        _, narrow = pair

        assert as_eight_bit_gray(narrow) is narrow

    def test_colour_is_still_converted(self) -> None:
        rgb = Image.fromarray(np.full((4, 4, 3), 128, np.uint8))

        assert as_eight_bit_gray(rgb).mode == "L"


class TestDepthDoesNotChangeTheReading:
    """Each module that reads tone gives the same answer at either depth."""

    def test_negative_density_validator(self, pair) -> None:
        """A 16-bit negative reported nearly every pixel as paper white."""
        from ptpd_calibration.qa.quality_assurance import NegativeDensityValidator

        validator = NegativeDensityValidator()
        wide, narrow = (validator.validate_density_range(image) for image in pair)

        assert wide.mean_density == pytest.approx(narrow.mean_density)
        assert wide.zone_distribution == narrow.zone_distribution
        # The clipped reading put essentially the whole frame in Zone X.
        assert wide.zone_distribution[10] < 0.5

    def test_density_histogram(self, pair) -> None:
        from ptpd_calibration.qa.quality_assurance import NegativeDensityValidator

        validator = NegativeDensityValidator()
        wide, narrow = (validator.get_density_histogram(image) for image in pair)

        assert np.array_equal(wide[0], narrow[0])

    def test_zone_mapper(self, pair) -> None:
        """A clipped scan landed in Zone X and drew a development change from it."""
        from ptpd_calibration.zones.mapping import ZoneMapper

        mapper = ZoneMapper()
        wide, narrow = (mapper.analyze_image(image) for image in pair)

        assert wide.average_zone == pytest.approx(narrow.average_zone)
        assert wide.zone_range == narrow.zone_range
        assert wide.development_adjustment == narrow.development_adjustment

    def test_zone_visualisation_leaves_the_caller_image_alone(self, pair) -> None:
        from ptpd_calibration.zones.mapping import ZoneMapper

        _, narrow = pair
        before = np.array(narrow).copy()

        ZoneMapper().visualize_zones(narrow)

        assert np.array_equal(np.array(narrow), before)

    def test_soft_proofer(self, pair) -> None:
        """The proof came out at paper Dmin across the whole frame."""
        from ptpd_calibration.proofing.simulation import SoftProofer

        proofer = SoftProofer()
        wide, narrow = (proofer.proof(image) for image in pair)

        assert np.array(wide.image).mean() == pytest.approx(np.array(narrow.image).mean(), abs=1.0)

    def test_print_comparison(self, pair) -> None:
        """One helper behind three public comparison entry points."""
        from ptpd_calibration.advanced.features import PrintComparison

        _, narrow = pair
        wide_image, narrow_image = pair

        # Comparing the two depths of one picture must read as identical.
        score = PrintComparison().calculate_similarity_score(wide_image, narrow_image)

        assert score == pytest.approx(PrintComparison().calculate_similarity_score(narrow, narrow))

    def test_style_analysis(self, pair) -> None:
        """Clipping pinned the gamma estimate at its clamp."""
        from ptpd_calibration.advanced.features import StyleTransfer

        analyzer = StyleTransfer()
        wide, narrow = (analyzer.analyze_style(image) for image in pair)

        assert wide == narrow


class TestSixteenBitTiffFallback:
    """The no-``tifffile`` path declared a mode instead of inferring it."""

    def test_an_rgba_array_is_not_reinterpreted_as_rgb(self) -> None:
        """``mode="RGB"`` on a four-channel buffer shifts every pixel by a byte.

        ``Image.fromarray`` reinterprets rather than converts, so the declared
        mode silently folded alpha into the colour stream. Inferring the mode
        from the array's own shape is what the export path needs.
        """
        from ptpd_calibration.imaging.safe_image import image_from_array

        rgba = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)

        inferred = image_from_array(rgba)

        assert inferred.mode == "RGBA"
        assert np.array_equal(np.array(inferred), rgba)

        # The bug, kept visible: the declared form does not give the array back.
        # Pillow 13 restricts the parameter rather than removing it, so the call
        # is refused there instead of misaligning; either answer proves the
        # point, and accepting both keeps this test working across the upgrade.
        try:
            declared = np.array(Image.fromarray(rgba, mode="RGB"))
        except (ValueError, TypeError):
            return
        assert declared[0][1].tolist() != rgba[0][1][:3].tolist()
