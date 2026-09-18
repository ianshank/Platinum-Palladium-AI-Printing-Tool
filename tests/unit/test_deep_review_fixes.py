"""Regressions for defects a deep review found, each reproduced before fixing.

Every test here failed on the code as it stood. They are grouped by the
mechanism rather than the module, because the same mechanism appeared in
several places: an 8-bit assumption on a read path, an unguarded division, and
a Pillow mode argument that reinterprets a buffer instead of converting it.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.logging import SafeFormatter, sanitize_log_text
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.linearization import AutoLinearizer, LinearizationConfig
from ptpd_calibration.curves.modifier import CurveModifier
from ptpd_calibration.imaging.histogram import HistogramAnalyzer
from ptpd_calibration.imaging.safe_image import image_from_array, to_uint8_scale
from ptpd_calibration.imaging.split_grade import SplitGradeSettings, SplitGradeSimulator

EIGHT_BIT_MAX = 255
SIXTEEN_BIT_MAX = 65535


class TestNonEightBitInputIsScaled:
    """`astype(np.uint8)` truncates modulo 256; these paths assumed 8-bit input.

    A 16-bit scan's paper white and its near-black came out four code values
    apart, so density extraction and patch detection ran on noise.
    """

    @pytest.mark.parametrize(
        ("array", "expected"),
        [
            (np.array([[0, 30000, SIXTEEN_BIT_MAX]], np.uint16), [0, 117, 255]),
            (np.array([[0, 30000, SIXTEEN_BIT_MAX]], np.int32), [0, 117, 255]),
            (np.array([[0.0, 0.5, 1.0]], np.float32), [0, 128, 255]),
            (np.array([[0.0, 128.0, 255.0]], np.float64), [0, 128, 255]),
            (np.array([[0, 128, 255]], np.uint8), [0, 128, 255]),
        ],
        ids=["uint16", "int32", "float-unit", "float-code", "uint8"],
    )
    def test_scaling_preserves_tone_relationships(
        self, array: np.ndarray, expected: list[int]
    ) -> None:
        assert to_uint8_scale(array).ravel().tolist() == expected

    def test_truncation_would_have_scrambled_the_wedge(self) -> None:
        """Pin the old behaviour so the difference is unmistakable."""
        wedge = np.array([[0, 30000, SIXTEEN_BIT_MAX]], np.uint16)

        assert wedge.astype(np.uint8).ravel().tolist() == [0, 48, 255]
        assert to_uint8_scale(wedge).ravel().tolist() == [0, 117, 255]

    def test_histogram_reads_a_sixteen_bit_array_correctly(self) -> None:
        sixteen = np.array([[0, 256, 512, 30000, SIXTEEN_BIT_MAX]], dtype=np.uint16)

        result = HistogramAnalyzer().analyze(sixteen)

        # Truncation reported a mean near 60 because 256 and 512 wrapped to 0.
        assert result.stats.mean > 70

    def test_histogram_reads_a_unit_float_array_correctly(self) -> None:
        unit = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)

        result = HistogramAnalyzer().analyze(unit)

        # Truncation made every value 0, reporting a brightness near zero.
        assert result.stats.mean > 100

    #: A 16-bit ramp whose 8-bit rendering has a mean at the middle of the range.
    SIXTEEN_BIT_RAMP = np.linspace(0, SIXTEEN_BIT_MAX, 256, dtype=np.uint16).reshape(16, 16)
    EIGHT_BIT_MIDPOINT = 127.5

    def _write_sixteen_bit_png(self, tmp_path: Path) -> Path:
        path = tmp_path / "ramp16.png"
        Image.fromarray(self.SIXTEEN_BIT_RAMP).save(path)
        return path

    @pytest.mark.parametrize("as_path", [True, False], ids=["path-input", "pil-input"])
    def test_histogram_scales_a_sixteen_bit_file_instead_of_clipping(
        self, tmp_path: Path, as_path: bool
    ) -> None:
        """The array branch was scaled but the file branch still reached convert("L").

        Pillow clips ``I;16`` at 255 rather than scaling it, so every sample
        above 255 came out white: this ramp reported a mean near 254, and the
        statistics and printing recommendations drawn from it described a frame
        that had lost every tone above 255.
        """
        written = self._write_sixteen_bit_png(tmp_path)
        source = written if as_path else Image.open(written)

        result = HistogramAnalyzer().analyze(source)

        assert result.stats.mean == pytest.approx(self.EIGHT_BIT_MIDPOINT, abs=1.0)

    def test_every_input_shape_agrees_on_the_same_image(self, tmp_path: Path) -> None:
        """A path, a PIL image and an array of one file must read alike."""
        written = self._write_sixteen_bit_png(tmp_path)
        analyzer = HistogramAnalyzer()

        means = [
            float(analyzer.analyze(written).stats.mean),
            float(analyzer.analyze(Image.open(written)).stats.mean),
            float(analyzer.analyze(self.SIXTEEN_BIT_RAMP).stats.mean),
            float(analyzer.analyze((self.SIXTEEN_BIT_RAMP // 257).astype(np.uint8)).stats.mean),
        ]

        assert means == pytest.approx([means[0]] * len(means), abs=1.0)
        assert means[0] == pytest.approx(self.EIGHT_BIT_MIDPOINT, abs=1.0)

    def test_the_reported_mode_still_names_the_source(self, tmp_path: Path) -> None:
        """Scaling is for the analysis; the caller is still told what it supplied."""
        written = self._write_sixteen_bit_png(tmp_path)

        assert HistogramAnalyzer().analyze(written).image_mode == "I;16"


class TestUnguardedDivisions:
    """Legal inputs divided by zero and poisoned the result with NaN."""

    def test_a_zero_shadow_threshold_returns_an_empty_mask(self) -> None:
        simulator = SplitGradeSimulator(
            SplitGradeSettings(shadow_threshold=0.0, highlight_threshold=0.6)
        )

        mask = np.asarray(simulator.create_shadow_mask(np.array([[0.0, 0.5, 1.0]], np.float32)))

        assert not np.isnan(mask).any()

    def test_a_full_highlight_threshold_returns_an_empty_mask(self) -> None:
        simulator = SplitGradeSimulator(
            SplitGradeSettings(shadow_threshold=0.4, highlight_threshold=1.0)
        )

        mask = np.asarray(simulator.create_highlight_mask(np.array([[0.0, 0.5, 1.0]], np.float32)))

        assert not np.isnan(mask).any()

    @staticmethod
    def _linear_curve() -> CurveData:
        axis = list(np.linspace(0.0, 1.0, 5))
        return CurveData(name="linear", input_values=axis, output_values=axis)

    def test_zero_width_midtone_adjustment_keeps_the_curve_finite(self) -> None:
        adjusted = CurveModifier().adjust_midtones(self._linear_curve(), amount=0.5, width=0.0)

        assert np.all(np.isfinite(np.asarray(adjusted.output_values)))

    def test_zero_influence_point_adjustment_keeps_the_curve_finite(self) -> None:
        adjusted = CurveModifier().add_point_adjustment(
            self._linear_curve(), 0.5, 0.7, influence=0.0
        )

        assert np.all(np.isfinite(np.asarray(adjusted.output_values)))


class TestRefinementUnits:
    """`refine_curve` compared raw density against a normalised target."""

    def test_refining_a_linear_wedge_is_a_no_op(self) -> None:
        linearizer = AutoLinearizer(LinearizationConfig())
        densities = list(np.linspace(0.06, 1.75, 21))
        base = linearizer.linearize(densities)

        refined = linearizer.refine_curve(base.curve, densities)

        outputs = np.asarray(refined.curve.output_values)
        # The mismatch left the curve topping out at 0.625.
        assert outputs[-1] == pytest.approx(1.0, abs=1e-6)
        assert refined.residual_error == pytest.approx(0.0, abs=1e-6)

    def test_a_flat_reading_is_refused_rather_than_divided_by_zero(self) -> None:
        linearizer = AutoLinearizer(LinearizationConfig())
        densities = list(np.linspace(0.06, 1.75, 21))
        base = linearizer.linearize(densities)

        with pytest.raises(ValueError, match="measurable density range"):
            linearizer.refine_curve(base.curve, [0.5] * 21)

    def test_the_damping_factor_is_configurable(self) -> None:
        assert LinearizationConfig().refinement_damping == pytest.approx(0.5)
        assert LinearizationConfig(refinement_damping=0.25).refinement_damping == 0.25


class TestModeArgumentReinterpretsBuffers:
    """`Image.fromarray(mode=)` reinterprets raw bytes; it does not convert."""

    def test_declaring_rgb_for_rgba_misaligns_every_pixel(self) -> None:
        """The premise, pinned so the helper's reason stays visible."""
        rgba = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)

        declared = np.asarray(Image.fromarray(rgba, mode="RGB"))

        assert not np.array_equal(declared, rgba[..., :3])

    def test_the_helper_infers_and_keeps_every_channel(self) -> None:
        rgba = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)

        image = image_from_array(rgba)

        assert image.mode == "RGBA"
        assert np.array_equal(np.asarray(image), rgba)

    def test_an_unsupported_shape_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            image_from_array(np.zeros((2, 2, 5), dtype=np.uint8))

    def test_a_float_array_is_scaled_rather_than_reinterpreted(self) -> None:
        """Declaring "L" for float data produced a saveable image of noise."""
        floats = np.linspace(0.0, 1.0, 16).reshape(4, 4)

        image = image_from_array(to_uint8_scale(floats))

        assert image.mode == "L"
        assert int(np.asarray(image).max()) == EIGHT_BIT_MAX
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        assert buffer.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"


class TestLogInjection:
    """Untrusted text must not be able to forge a second log record."""

    @pytest.mark.parametrize("break_char", ["\n", "\r", "\x0b", "\x85", " ", " "], ids=repr)
    def test_every_line_break_is_escaped(self, break_char: str) -> None:
        escaped = sanitize_log_text(f"before{break_char}after")

        assert len(escaped.splitlines()) == 1
        assert "after" in escaped

    def test_a_forged_record_stays_on_one_line(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SafeFormatter("%(levelname)s %(message)s"))
        logger = logging.getLogger("tests.sanitize.forged")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        logger.debug("path=%s", "/api/\nCRITICAL forged: every calibration deleted")

        lines = stream.getvalue().splitlines()
        assert len(lines) == 1
        assert not any(line.startswith("CRITICAL forged") for line in lines)

    def test_a_traceback_keeps_its_line_breaks(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SafeFormatter("%(levelname)s %(message)s"))
        logger = logging.getLogger("tests.sanitize.traceback")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        try:
            raise ZeroDivisionError("division by zero")
        except ZeroDivisionError:
            logger.error("failed", exc_info=True)

        assert len(stream.getvalue().splitlines()) >= 3

    def test_long_values_are_truncated_with_a_count(self) -> None:
        escaped = sanitize_log_text("x" * 100, max_length=10)

        assert escaped.startswith("x" * 10)
        assert "90 more" in escaped


class TestDensityExtractionNormalisesDepth:
    """The density path divided by 255, so a 16-bit scan read as blank paper.

    ``load_image_array`` hands a 16-bit file over at its full depth on purpose,
    which made reflectance far greater than 1 and drove every density to 0.
    """

    @staticmethod
    def _density(rgb: np.ndarray) -> float:
        from ptpd_calibration.detection.extractor import DensityExtractor

        reflectance = DensityExtractor()._to_reflectance(rgb)
        return float(np.mean(-np.log10(np.clip(reflectance, 1e-6, 1.0))))

    def test_the_same_tone_reads_the_same_at_either_depth(self) -> None:
        eight = np.array([[[235, 235, 235]]], dtype=np.uint8)
        sixteen = np.array([[[235 * 257, 235 * 257, 235 * 257]]], dtype=np.uint16)

        assert self._density(eight) == pytest.approx(self._density(sixteen), abs=1e-3)

    def test_a_sixteen_bit_scan_no_longer_reads_as_blank_paper(self) -> None:
        dark = np.array([[[3000, 3000, 3000]]], dtype=np.uint16)

        assert self._density(dark) > 1.0

    def test_reflectance_stays_within_its_physical_range(self) -> None:
        from ptpd_calibration.detection.extractor import DensityExtractor

        wedge = np.array([[[0, 0, 0], [30000, 30000, 30000], [65535, 65535, 65535]]], np.uint16)

        reflectance = DensityExtractor()._to_reflectance(wedge)

        assert float(np.max(reflectance)) <= 1.0
        assert float(np.min(reflectance)) >= 0.0
