"""
Tests for the QTR .quad file parser.
"""

import pytest

from ptpd_calibration.curves.parser import (
    ChannelCurve,
    QuadFileParser,
    QuadProfile,
    load_quad_file,
    load_quad_string,
)


class TestChannelCurve:
    """Tests for ChannelCurve dataclass."""

    def test_channel_curve_creation(self):
        """Test creating a channel curve."""
        values = list(range(256))
        curve = ChannelCurve(name="K", values=values)

        assert curve.name == "K"
        assert len(curve.values) == 256
        assert curve.enabled is True

    def test_as_normalized(self):
        """Test normalized output conversion."""
        values = [0, 128, 255]
        curve = ChannelCurve(name="K", values=values)

        inputs, outputs = curve.as_normalized

        assert len(inputs) == 3
        assert len(outputs) == 3
        assert inputs[0] == 0.0
        assert inputs[-1] == pytest.approx(1.0, rel=0.01)
        assert outputs[0] == 0.0
        assert outputs[1] == pytest.approx(0.502, rel=0.01)
        assert outputs[2] == 1.0

    def test_to_curve_data(self):
        """Test conversion to CurveData model."""
        values = [0, 64, 128, 192, 255]
        curve = ChannelCurve(name="K", values=values)

        curve_data = curve.to_curve_data(" - Test Profile")

        assert "K - Test Profile" in curve_data.name
        assert len(curve_data.input_values) == 5
        assert len(curve_data.output_values) == 5


class TestQuadProfile:
    """Tests for QuadProfile dataclass."""

    def test_profile_defaults(self):
        """Test default profile values."""
        profile = QuadProfile()

        assert profile.profile_name == "Untitled"
        assert profile.resolution == 2880
        assert profile.ink_limit == 100.0
        assert len(profile.channels) == 0

    def test_primary_channel(self):
        """Test primary channel accessor."""
        profile = QuadProfile()
        profile.channels["K"] = ChannelCurve(name="K", values=[0] * 256)

        assert profile.primary_channel is not None
        assert profile.primary_channel.name == "K"

    def test_active_channels(self):
        """Test active channel detection."""
        profile = QuadProfile()
        profile.channels["K"] = ChannelCurve(name="K", values=[0, 10, 20, 30], enabled=True)
        profile.channels["C"] = ChannelCurve(name="C", values=[0, 0, 0, 0], enabled=True)
        profile.channels["M"] = ChannelCurve(name="M", values=[0, 5, 10, 15], enabled=False)

        active = profile.active_channels

        assert "K" in active
        assert "C" not in active  # All zeros
        assert "M" not in active  # Disabled

    def test_get_channel(self):
        """Test channel retrieval."""
        profile = QuadProfile()
        profile.channels["K"] = ChannelCurve(name="K", values=[0] * 256)

        assert profile.get_channel("K") is not None
        assert profile.get_channel("k") is not None  # Case insensitive
        assert profile.get_channel("X") is None

    def test_summary(self):
        """Test summary generation."""
        profile = QuadProfile(profile_name="Test Profile", resolution=1440)
        profile.channels["K"] = ChannelCurve(name="K", values=list(range(256)))

        summary = profile.summary()

        assert "Test Profile" in summary
        assert "1440" in summary


class TestQuadFileParser:
    """Tests for QuadFileParser."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = QuadFileParser()

        assert parser._current_section is None
        assert parser._profile is None

    def test_parse_simple_quad_content(self):
        """Test parsing simple .quad content."""
        content = """
[General]
ProfileName=Test Profile
Resolution=2880
InkLimit=95.0

[K]
0=0
1=5
2=10
255=255
"""
        parser = QuadFileParser()
        profile = parser.parse_string(content, "Test")

        assert profile.profile_name == "Test Profile"
        assert profile.resolution == 2880
        assert profile.ink_limit == 95.0
        assert "K" in profile.channels

    def test_parse_multiple_channels(self):
        """Test parsing multiple channels."""
        content = """
[General]
ProfileName=Multi-Channel

[K]
0=0
255=255

[C]
0=0
255=128

[M]
0=0
255=64
"""
        parser = QuadFileParser()
        profile = parser.parse_string(content, "Multi")

        assert "K" in profile.channels
        assert "C" in profile.channels
        assert "M" in profile.channels

    def test_parse_comments(self):
        """Test parsing comments."""
        content = """
# This is a comment
; This is also a comment

[General]
ProfileName=Commented Profile
"""
        parser = QuadFileParser()
        profile = parser.parse_string(content)

        assert len(profile.comments) == 2
        assert "This is a comment" in profile.comments[0]

    def test_parse_media_settings(self):
        """Test parsing media settings."""
        content = """
[General]
MediaType=Glossy Photo
MediaSetting=Premium
"""
        parser = QuadFileParser()
        profile = parser.parse_string(content)

        assert profile.media_type == "Glossy Photo"
        assert profile.media_setting == "Premium"

    def test_parse_curve_values_in_range(self):
        """Test that curve values are clamped to valid range."""
        content = """
[K]
0=0
1=300
2=-10
"""
        parser = QuadFileParser()
        profile = parser.parse_string(content)

        curve = profile.channels["K"]
        assert curve.values[0] == 0
        assert curve.values[1] == 255  # Clamped from 300
        assert curve.values[2] == 0  # Clamped from -10


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_quad_string(self):
        """Test load_quad_string function."""
        content = """
[General]
ProfileName=String Profile

[K]
0=0
255=255
"""
        profile = load_quad_string(content, "Named Profile")

        assert profile.profile_name == "String Profile"

    def test_load_quad_file_not_found(self, tmp_path):
        """Test load_quad_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_quad_file(tmp_path / "nonexistent.quad")

    def test_load_quad_file(self, tmp_path):
        """Test load_quad_file with actual file."""
        content = """
[General]
ProfileName=File Profile
Resolution=1440

[K]
0=0
128=100
255=255
"""
        file_path = tmp_path / "test.quad"
        file_path.write_text(content)

        profile = load_quad_file(file_path)

        assert profile.profile_name == "File Profile"
        assert profile.resolution == 1440
        assert profile.source_path == file_path

    def test_to_curve_data_conversion(self):
        """Test converting profile channel to CurveData."""
        content = """
[General]
ProfileName=Curve Test
MediaType=Arches Platine

[K]
0=0
64=48
128=96
192=168
255=255
"""
        profile = load_quad_string(content)
        curve_data = profile.to_curve_data("K")

        assert "K" in curve_data.name
        assert curve_data.paper_type == "Arches Platine"
        assert len(curve_data.input_values) == 256

    def test_to_curve_data_invalid_channel(self):
        """Test error when requesting invalid channel."""
        content = """
[General]
ProfileName=Test

[K]
0=0
"""
        profile = load_quad_string(content)

        with pytest.raises(ValueError, match="not found"):
            profile.to_curve_data("X")
