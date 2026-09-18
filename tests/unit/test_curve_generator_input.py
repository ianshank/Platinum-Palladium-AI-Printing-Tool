"""Input-domain tests for :class:`CurveGenerator`, plus the closure invariant.

The correction curve is defined by ``measured⁻¹ ∘ target``, so the property
that makes it a *linearisation* curve is closure: applying the curve and then
the measured response must reproduce the target. No test asserted that before,
which is why three silent failures survived a property-testing campaign:

* a step wedge read from the wrong end (descending densities),
* a single unreadable patch (NaN),
* a series that rises and falls.

Each produced a plausible-looking curve that would have ruined a print.
"""

from __future__ import annotations

import numpy as np
import pytest

from ptpd_calibration.config import CurveSettings
from ptpd_calibration.curves.generator import CurveGenerator

pytestmark = pytest.mark.unit

STEPS = 21

# Analytic response shapes a real process plausibly produces. Each maps a
# normalised input to a density, ascending and finite.
RESPONSES = {
    "gamma_1_2": lambda x: 0.10 + 1.30 * x**1.2,
    "gamma_1_6": lambda x: 0.08 + 1.37 * x**1.6,
    "gamma_0_7": lambda x: 0.12 + 1.25 * x**0.7,
    "cubic": lambda x: 0.10 + 1.50 * x**3,
    "s_curve": lambda x: 0.10 + 1.40 / (1.0 + np.exp(-8.0 * (x - 0.5))),
    "near_linear": lambda x: 0.05 + 1.45 * x,
}

# Closure tolerance in normalised output units. One 8-bit code value is
# 1/255 = 0.0039; the generator interpolates to 256 points and re-pins the
# endpoints, so a few code values is the honest bound for every shape.
CLOSURE_TOLERANCE = 0.02


def _densities(shape: str, steps: int = STEPS) -> np.ndarray:
    return RESPONSES[shape](np.linspace(0.0, 1.0, steps))


def _closure_error(densities: np.ndarray, outputs: list[float]) -> float:
    """Return ``max |m(c(x)) - t(x)|`` for a linear target.

    ``c`` is the generated curve sampled on its own abscissa, ``m`` the
    measured response normalised to [0, 1], and the target is the identity.
    """
    curve = np.asarray(outputs, dtype=float)
    axis = np.linspace(0.0, 1.0, curve.size)
    measured_axis = np.linspace(0.0, 1.0, densities.size)
    normalised = (densities - densities.min()) / (densities.max() - densities.min())
    reproduced = np.interp(curve, measured_axis, normalised)
    return float(np.abs(reproduced - axis).max())


class TestClosureInvariant:
    """Applying the curve then the process must reproduce the target."""

    @pytest.mark.parametrize("shape", sorted(RESPONSES))
    def test_curve_linearises_the_response(self, shape: str) -> None:
        densities = _densities(shape)
        curve = CurveGenerator().generate(list(densities))

        assert _closure_error(densities, curve.output_values) <= CLOSURE_TOLERANCE

    @pytest.mark.parametrize("steps", [11, 21, 31, 41])
    def test_closure_holds_at_every_supported_step_count(self, steps: int) -> None:
        densities = _densities("gamma_1_6", steps)
        curve = CurveGenerator().generate(list(densities))

        assert _closure_error(densities, curve.output_values) <= CLOSURE_TOLERANCE

    def test_a_reversed_scan_produces_the_same_curve(self) -> None:
        """The decisive case: the same print read from the other end."""
        densities = _densities("gamma_1_2")
        generator = CurveGenerator()

        forward = generator.generate(list(densities))
        backward = generator.generate(list(densities[::-1]))

        assert np.allclose(forward.output_values, backward.output_values)


class TestDensityValidation:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_patch_is_refused_and_named(self, bad: float) -> None:
        densities = list(_densities("gamma_1_2"))
        densities[7] = bad

        with pytest.raises(ValueError, match="finite") as excinfo:
            CurveGenerator().generate(densities)

        assert "patch 7" in str(excinfo.value)

    def test_rising_then_falling_series_is_refused(self) -> None:
        """A wedge that reverses mid-way has no single inverse, so it cannot be fixed."""
        densities = np.concatenate([np.linspace(0.1, 1.2, 11), np.linspace(1.15, 0.4, 10)])

        with pytest.raises(ValueError, match="monotonic"):
            CurveGenerator().generate(list(densities))

    def test_noise_within_tolerance_is_accepted(self) -> None:
        densities = _densities("gamma_1_2")
        densities[5] = densities[4] - 0.01  # a backwards step under the default tolerance

        curve = CurveGenerator().generate(list(densities))

        assert len(curve.output_values) > 0

    def test_tolerance_is_configurable(self) -> None:
        densities = _densities("gamma_1_2")
        densities[5] = densities[4] - 0.05

        with pytest.raises(ValueError, match="monotonic"):
            CurveGenerator().generate(list(densities))

        tolerant = CurveGenerator(settings=CurveSettings(density_monotonicity_tolerance=0.1))
        assert tolerant.generate(list(densities)).output_values

    def test_auto_orientation_can_be_disabled(self) -> None:
        """With reversal off, a descending wedge is refused rather than silently fixed."""
        densities = _densities("gamma_1_2")[::-1]
        strict = CurveGenerator(settings=CurveSettings(auto_orient_densities=False))

        with pytest.raises(ValueError, match="monotonic"):
            strict.generate(list(densities))

    def test_reversal_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        densities = _densities("gamma_1_2")[::-1]

        with caplog.at_level("INFO", logger="ptpd_calibration.curves.generator"):
            CurveGenerator().generate(list(densities))

        assert "descending" in caplog.text


class TestDensityScale:
    """Reported densities must match what a densitometer reads.

    The extractor used gamma-encoded scanner values directly as reflectance.
    Density is ``-log10(reflectance)`` and reflectance is linear, so skipping
    the sRGB transfer function compressed the scale by about half: an
    excellent print reading 1.95 was reported as 0.97, and the quality gates,
    set from published figures for this process, became unreachable. The tool
    then told the printer to increase exposure on a perfect print.
    """

    @staticmethod
    def _encode(linear: float) -> float:
        """Encode a linear reflectance the way a scanner file stores it."""
        if linear <= 0.0031308:
            return linear * 12.92 * 255.0
        return (1.055 * linear ** (1 / 2.4) - 0.055) * 255.0

    def _code_for_density(self, density: float) -> np.ndarray:
        return np.repeat(np.clip(self._encode(10.0**-density), 0.0, 255.0), 3)

    @pytest.mark.parametrize("paper,black", [(0.08, 1.95), (0.10, 1.45), (0.06, 1.20)])
    def test_reported_density_matches_the_true_density(self, paper: float, black: float) -> None:
        from ptpd_calibration.detection.extractor import DensityExtractor

        reference = tuple(float(v) for v in self._code_for_density(paper))
        reported = DensityExtractor()._rgb_to_density(self._code_for_density(black), reference)

        assert reported == pytest.approx(black - paper, abs=0.01)

    def test_an_excellent_print_clears_the_quality_gate(self) -> None:
        """The gate is 1.8, which is right for this process and was unreachable."""
        from ptpd_calibration.detection.extractor import DensityExtractor

        reference = tuple(float(v) for v in self._code_for_density(0.08))
        reported = DensityExtractor()._rgb_to_density(self._code_for_density(1.95), reference)

        assert reported >= 1.8

    def test_linearisation_can_be_disabled_for_already_linear_input(self) -> None:
        from ptpd_calibration.config import ExtractionSettings
        from ptpd_calibration.detection.extractor import DensityExtractor

        extractor = DensityExtractor(settings=ExtractionSettings(linearize_srgb=False))
        reference = tuple(float(v) for v in self._code_for_density(0.08))
        reported = extractor._rgb_to_density(self._code_for_density(1.95), reference)

        assert reported < 1.2  # the old, compressed scale

    def test_the_weights_setting_keeps_its_old_environment_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Renaming the field must not silently ignore existing configuration."""
        from ptpd_calibration.config import ExtractionSettings

        monkeypatch.setenv("PTPD_EXTRACTION_STATUS_A_WEIGHTS", "[0.3, 0.5, 0.2]")
        assert ExtractionSettings().visual_density_weights == (0.3, 0.5, 0.2)
