"""
Tests for curve adjuster module.

Tests calibration profiles, curve calibrator, and QTR curve parsing.
"""

import pytest
import tempfile
from pathlib import Path

from ptpd_calibration.calibration.curve_adjuster import (
    CalibrationProfile,
    CurveCalibrator,
    QuadCurveParser,
    CALIBRATION_PROFILES,
    adjust_curve_for_paper,
    refine_curve_from_print,
    get_available_calibration_profiles,
    get_calibration_profile,
)
from ptpd_calibration.papers.profiles import BUILTIN_PAPERS


# =============================================================================
# CalibrationProfile Tests
# =============================================================================


class TestCalibrationProfile:
    """Tests for CalibrationProfile dataclass."""

    def test_default_profile(self):
        """Default profile should have neutral values."""
        profile = CalibrationProfile(name="Test")
        assert profile.name == "Test"
        assert profile.absorption_factor == 1.0
        assert profile.highlight_boost == 0.0
        assert profile.midtone_boost == 0.0
        assert profile.shadow_boost == 0.0

    def test_custom_profile(self):
        """Custom profile should accept all parameters."""
        profile = CalibrationProfile(
            name="Custom Paper",
            absorption_factor=1.08,
            highlight_boost=0.02,
            midtone_boost=0.10,
            shadow_boost=0.07,
            dot_gain_curve=[1.02, 1.06, 1.08, 1.10, 1.10, 1.08, 1.07],
            notes="Test notes"
        )
        assert profile.name == "Custom Paper"
        assert profile.absorption_factor == 1.08
        assert profile.midtone_boost == 0.10
        assert len(profile.dot_gain_curve) == 7

    def test_to_dict(self):
        """Profile should serialize to dictionary."""
        profile = CalibrationProfile(
            name="Test",
            absorption_factor=1.05,
            midtone_boost=0.08
        )
        d = profile.to_dict()
        assert d['name'] == "Test"
        assert d['absorption_factor'] == 1.05
        assert d['midtone_boost'] == 0.08
        assert 'dot_gain_curve' in d

    def test_from_dict(self):
        """Profile should deserialize from dictionary."""
        data = {
            'name': 'Loaded Profile',
            'absorption_factor': 1.06,
            'highlight_boost': 0.02,
            'midtone_boost': 0.07,
            'shadow_boost': 0.05,
        }
        profile = CalibrationProfile.from_dict(data)
        assert profile.name == "Loaded Profile"
        assert profile.absorption_factor == 1.06
        assert profile.midtone_boost == 0.07

    def test_from_paper_profile(self):
        """Should create calibration profile from paper profile."""
        paper = BUILTIN_PAPERS.get('arches_platine')
        if paper is None:
            pytest.skip("Arches Platine not in BUILTIN_PAPERS")

        profile = CalibrationProfile.from_paper_profile(paper)
        assert profile.name == paper.name
        assert profile.absorption_factor > 1.0  # Papers are absorbent
        assert len(profile.dot_gain_curve) == 7


# =============================================================================
# CALIBRATION_PROFILES Tests
# =============================================================================


class TestCalibrationProfiles:
    """Tests for built-in calibration profiles."""

    def test_profiles_exist(self):
        """Should have built-in calibration profiles."""
        assert len(CALIBRATION_PROFILES) > 0

    def test_arches_platine_profile(self):
        """Arches Platine should have specific values."""
        profile = CALIBRATION_PROFILES.get('arches_platine')
        assert profile is not None
        assert profile.name == "Arches Platine"
        assert profile.midtone_boost == 0.10  # Key finding: 10% boost
        assert profile.shadow_boost == 0.07

    def test_all_profiles_have_required_fields(self):
        """All profiles should have required fields."""
        for key, profile in CALIBRATION_PROFILES.items():
            assert profile.name
            assert isinstance(profile.absorption_factor, (int, float))
            assert isinstance(profile.highlight_boost, (int, float))
            assert isinstance(profile.midtone_boost, (int, float))
            assert isinstance(profile.shadow_boost, (int, float))

    def test_custom_profile_is_neutral(self):
        """Custom profile should be neutral (no adjustment)."""
        profile = CALIBRATION_PROFILES.get('custom')
        assert profile is not None
        assert profile.absorption_factor == 1.0
        assert profile.highlight_boost == 0.0
        assert profile.midtone_boost == 0.0
        assert profile.shadow_boost == 0.0

    def test_get_available_calibration_profiles(self):
        """Should return list of profile names."""
        names = get_available_calibration_profiles()
        assert isinstance(names, list)
        assert 'arches_platine' in names
        assert 'custom' in names

    def test_get_calibration_profile(self):
        """Should retrieve profile by name."""
        profile = get_calibration_profile('arches_platine')
        assert profile is not None
        assert profile.name == "Arches Platine"

    def test_get_calibration_profile_not_found(self):
        """Should return None for non-existent profile."""
        profile = get_calibration_profile('nonexistent_paper_xyz')
        assert profile is None


# =============================================================================
# CurveCalibrator Tests
# =============================================================================


class TestCurveCalibrator:
    """Tests for CurveCalibrator class."""

    @pytest.fixture
    def arches_calibrator(self):
        """Create calibrator with Arches Platine profile."""
        return CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])

    @pytest.fixture
    def neutral_calibrator(self):
        """Create calibrator with neutral (custom) profile."""
        return CurveCalibrator(CALIBRATION_PROFILES['custom'])

    @pytest.fixture
    def sample_curve(self):
        """Create a sample 256-value curve."""
        # Linear curve from 0 to 65535
        return [int(65535 * i / 255) for i in range(256)]

    @pytest.fixture
    def sample_curves_dict(self, sample_curve):
        """Create sample curves dictionary."""
        return {
            'K': sample_curve.copy(),
            'C': sample_curve.copy(),
            'M': sample_curve.copy(),
            'Y': sample_curve.copy(),
        }

    def test_init_with_profile(self):
        """Should initialize with profile."""
        profile = CALIBRATION_PROFILES['arches_platine']
        calibrator = CurveCalibrator(profile)
        assert calibrator.profile.name == "Arches Platine"

    def test_init_without_profile(self):
        """Should initialize with neutral profile when none specified."""
        calibrator = CurveCalibrator()
        assert calibrator.profile.name == "Custom"
        assert calibrator.profile.absorption_factor == 1.0

    def test_set_profile(self):
        """Should set profile by name."""
        calibrator = CurveCalibrator()
        calibrator.set_profile('bergger_cot320')
        assert calibrator.profile.name == "Bergger COT320"

    def test_set_profile_invalid(self):
        """Should raise error for invalid profile name."""
        calibrator = CurveCalibrator()
        with pytest.raises(ValueError):
            calibrator.set_profile('nonexistent_paper')

    def test_set_profile_from_paper(self):
        """Should set profile from built-in paper."""
        calibrator = CurveCalibrator()
        calibrator.set_profile_from_paper('arches_platine')
        assert 'Arches' in calibrator.profile.name

    def test_adjust_curve_basic(self, arches_calibrator, sample_curve):
        """Should adjust curve values."""
        adjusted = arches_calibrator.adjust_curve(sample_curve, 'K')

        # Curve should be adjusted (values should change)
        # With Arches profile, values should generally increase
        assert len(adjusted) == 256
        assert adjusted[0] == 0  # Black point preserved

        # Midtones should be boosted
        mid_original = sample_curve[128]
        mid_adjusted = adjusted[128]
        assert mid_adjusted >= mid_original

    def test_adjust_curve_neutral(self, neutral_calibrator, sample_curve):
        """Neutral profile should not significantly change curve."""
        adjusted = neutral_calibrator.adjust_curve(sample_curve, 'K')

        # Values should be nearly identical with neutral profile
        for i in range(256):
            assert abs(adjusted[i] - sample_curve[i]) < 10

    def test_adjust_curve_preserves_zeros(self, arches_calibrator):
        """Should preserve zero values."""
        curve_with_zeros = [0] * 50 + [int(65535 * i / 205) for i in range(206)]
        adjusted = arches_calibrator.adjust_curve(curve_with_zeros, 'K')

        # All zeros should remain zero
        for i in range(50):
            assert adjusted[i] == 0

    def test_adjust_curve_clamps_to_max(self, arches_calibrator):
        """Should clamp values to 65535 maximum."""
        # Curve already near max
        high_curve = [65000] * 256
        adjusted = arches_calibrator.adjust_curve(high_curve, 'K')

        for val in adjusted:
            assert val <= 65535

    def test_adjust_curve_channel_scaling(self, arches_calibrator, sample_curve):
        """Different channels should have different scaling."""
        k_adjusted = arches_calibrator.adjust_curve(sample_curve, 'K')
        y_adjusted = arches_calibrator.adjust_curve(sample_curve, 'Y')

        # Y channel should have less aggressive adjustment than K
        k_diff = sum(k_adjusted[i] - sample_curve[i] for i in range(256))
        y_diff = sum(y_adjusted[i] - sample_curve[i] for i in range(256))

        assert k_diff > y_diff  # K gets more boost than Y

    def test_adjust_all_curves(self, arches_calibrator, sample_curves_dict):
        """Should adjust all channels."""
        adjusted = arches_calibrator.adjust_all_curves(sample_curves_dict)

        assert 'K' in adjusted
        assert 'C' in adjusted
        assert 'M' in adjusted
        assert 'Y' in adjusted

        # All channels should be adjusted
        for channel in adjusted:
            assert len(adjusted[channel]) == 256

    def test_adjust_from_feedback(self, neutral_calibrator, sample_curve):
        """Should apply feedback-based adjustments."""
        adjusted = neutral_calibrator.adjust_from_feedback(
            sample_curve,
            highlight_delta=0.05,
            midtone_delta=0.10,
            shadow_delta=0.07,
            channel='K'
        )

        assert len(adjusted) == 256
        assert adjusted[0] == 0  # Black preserved

        # Midtones should be boosted
        mid_original = sample_curve[128]
        mid_adjusted = adjusted[128]
        assert mid_adjusted > mid_original

    def test_adjust_from_feedback_negative(self, neutral_calibrator, sample_curve):
        """Should reduce values with negative adjustments."""
        adjusted = neutral_calibrator.adjust_from_feedback(
            sample_curve,
            highlight_delta=-0.05,
            midtone_delta=-0.10,
            shadow_delta=-0.07,
            channel='K'
        )

        # Midtones should be reduced
        mid_original = sample_curve[128]
        mid_adjusted = adjusted[128]
        assert mid_adjusted < mid_original

    def test_adjust_all_from_feedback(self, neutral_calibrator, sample_curves_dict):
        """Should apply feedback to all channels."""
        adjusted = neutral_calibrator.adjust_all_from_feedback(
            sample_curves_dict,
            highlight_delta=0.02,
            midtone_delta=0.08,
            shadow_delta=0.05
        )

        for channel in adjusted:
            assert len(adjusted[channel]) == 256


# =============================================================================
# QuadCurveParser Tests
# =============================================================================


class TestQuadCurveParser:
    """Tests for QuadCurveParser class."""

    @pytest.fixture
    def sample_quad_content(self):
        """Create sample .quad file content."""
        lines = [
            "# Platinum-Palladium Profile",
            "# Generated by PTPD Calibration",
            "# K Curve",
        ]
        # Add 256 values for K channel
        for i in range(256):
            lines.append(str(int(65535 * i / 255)))

        # Add empty C channel
        lines.append("# C Curve")
        for _ in range(256):
            lines.append("0")

        return "\n".join(lines)

    @pytest.fixture
    def quad_file(self, tmp_path, sample_quad_content):
        """Create a sample .quad file."""
        quad_path = tmp_path / "test_curve.quad"
        quad_path.write_text(sample_quad_content)
        return quad_path

    def test_parse_quad_file(self, quad_file):
        """Should parse .quad file correctly."""
        header, curves = QuadCurveParser.parse(str(quad_file))

        assert len(header) >= 2
        assert 'K' in curves
        assert 'C' in curves
        assert len(curves['K']) == 256
        assert len(curves['C']) == 256

    def test_parse_header_comments(self, quad_file):
        """Should extract header comments."""
        header, _ = QuadCurveParser.parse(str(quad_file))

        # Should have the header comments (comments before first Curve line)
        assert len(header) >= 1
        assert any('Platinum-Palladium' in h or 'Generated' in h for h in header)

    def test_parse_curve_values(self, quad_file):
        """Should parse curve values correctly."""
        _, curves = QuadCurveParser.parse(str(quad_file))

        k_curve = curves['K']
        assert k_curve[0] == 0  # First value
        assert k_curve[-1] == 65535  # Last value (approximately)

    def test_write_quad_file(self, tmp_path):
        """Should write .quad file correctly."""
        header = ["# Test Curve", "# Generated by test"]
        curves = {
            'K': [int(65535 * i / 255) for i in range(256)],
            'C': [0] * 256,
        }
        output_path = tmp_path / "output.quad"

        QuadCurveParser.write(str(output_path), header, curves)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# K Curve" in content
        assert "# C Curve" in content

    def test_write_with_extra_comments(self, tmp_path):
        """Should include extra comments in output."""
        header = ["# Original header"]
        curves = {'K': [0] * 256}
        extra_comments = ["Adjusted for test paper", "Midtones +10%"]
        output_path = tmp_path / "output.quad"

        QuadCurveParser.write(str(output_path), header, curves, extra_comments)

        content = output_path.read_text()
        assert "Adjusted for test paper" in content
        assert "Midtones +10%" in content

    def test_roundtrip_parse_write(self, tmp_path):
        """Parsing and writing should preserve data."""
        # Create original file
        original_curves = {
            'K': [int(65535 * i / 255) for i in range(256)],
            'C': [int(32767 * i / 255) for i in range(256)],
        }
        original_header = ["# Original curve"]
        original_path = tmp_path / "original.quad"
        QuadCurveParser.write(str(original_path), original_header, original_curves)

        # Parse and rewrite
        header, curves = QuadCurveParser.parse(str(original_path))
        output_path = tmp_path / "output.quad"
        QuadCurveParser.write(str(output_path), header, curves)

        # Parse again and compare
        _, reparsed_curves = QuadCurveParser.parse(str(output_path))

        assert reparsed_curves['K'] == original_curves['K']
        assert reparsed_curves['C'] == original_curves['C']


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def sample_quad_file(self, tmp_path):
        """Create a sample .quad file for testing."""
        curves = {
            'K': [int(65535 * i / 255) for i in range(256)],
            'C': [0] * 256,
            'M': [0] * 256,
            'Y': [0] * 256,
        }
        header = ["# Test Curve"]
        quad_path = tmp_path / "test.quad"
        QuadCurveParser.write(str(quad_path), header, curves)
        return quad_path

    def test_adjust_curve_for_paper(self, sample_quad_file, tmp_path):
        """Should adjust curve for specific paper."""
        output_path = tmp_path / "adjusted.quad"

        adjusted = adjust_curve_for_paper(
            str(sample_quad_file),
            str(output_path),
            'arches_platine'
        )

        assert output_path.exists()
        assert 'K' in adjusted
        assert len(adjusted['K']) == 256

        # Verify the output file has paper info in comments
        content = output_path.read_text()
        assert 'Arches Platine' in content

    def test_adjust_curve_for_paper_invalid_paper(self, sample_quad_file, tmp_path):
        """Should raise error for invalid paper."""
        output_path = tmp_path / "adjusted.quad"

        with pytest.raises(ValueError):
            adjust_curve_for_paper(
                str(sample_quad_file),
                str(output_path),
                'nonexistent_paper_xyz'
            )

    def test_refine_curve_from_print(self, sample_quad_file, tmp_path):
        """Should refine curve based on print feedback."""
        output_path = tmp_path / "refined.quad"

        refined = refine_curve_from_print(
            str(sample_quad_file),
            str(output_path),
            highlight_adj=0.02,
            midtone_adj=0.08,
            shadow_adj=0.05
        )

        assert output_path.exists()
        assert 'K' in refined

        # Verify the output file has adjustment info
        content = output_path.read_text()
        assert 'Refined from print feedback' in content


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_curve(self):
        """Should handle empty curve."""
        calibrator = CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])
        result = calibrator.adjust_curve([], 'K')
        assert result == []

    def test_single_value_curve(self):
        """Should handle single-value curve."""
        calibrator = CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])
        result = calibrator.adjust_curve([32768], 'K')
        assert len(result) == 1

    def test_all_zeros_curve(self):
        """Should handle all-zeros curve."""
        calibrator = CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])
        zero_curve = [0] * 256
        result = calibrator.adjust_curve(zero_curve, 'K')

        # All zeros should remain zero
        assert all(v == 0 for v in result)

    def test_all_max_curve(self):
        """Should handle all-max values curve."""
        calibrator = CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])
        max_curve = [65535] * 256
        result = calibrator.adjust_curve(max_curve, 'K')

        # Values should be clamped to max
        assert all(v <= 65535 for v in result)

    def test_unknown_channel(self):
        """Should handle unknown channel gracefully."""
        calibrator = CurveCalibrator(CALIBRATION_PROFILES['arches_platine'])
        curve = [int(65535 * i / 255) for i in range(256)]

        # Unknown channel should get neutral adjustment
        result = calibrator.adjust_curve(curve, 'UNKNOWN')
        assert len(result) == 256

    def test_extreme_adjustment_values(self):
        """Should handle extreme adjustment values."""
        calibrator = CurveCalibrator()
        curve = [int(65535 * i / 255) for i in range(256)]

        # Very large positive adjustment
        result = calibrator.adjust_from_feedback(
            curve,
            highlight_delta=0.15,
            midtone_delta=0.15,
            shadow_delta=0.15,
            channel='K'
        )
        assert all(v <= 65535 for v in result)
        assert all(v >= 0 for v in result)

        # Large negative adjustment
        result = calibrator.adjust_from_feedback(
            curve,
            highlight_delta=-0.15,
            midtone_delta=-0.15,
            shadow_delta=-0.15,
            channel='K'
        )
        assert all(v <= 65535 for v in result)
        assert all(v >= 0 for v in result)
