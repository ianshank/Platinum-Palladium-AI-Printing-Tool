"""Tests for the .quad parser size caps and overflow handling (SEC-14)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ptpd_calibration.curves.parser import (
    QuadFileParser,
    QuadParserLimits,
    load_quad_file,
    load_quad_string,
)

pytestmark = pytest.mark.unit

INI_PROFILE = """[General]
ProfileName=Test Profile
Resolution=2880
InkLimit=80
MediaType=Arches Platine

[K]
0=0
128=128
255=255
"""


class TestQuadParserLimits:
    def test_defaults(self) -> None:
        limits = QuadParserLimits()
        assert limits.max_bytes == 2 * 1024 * 1024
        assert limits.max_channels == 16
        assert limits.max_sections == 64
        assert limits.max_keys_per_section == 1024
        assert limits.max_name_length == 200

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_QUAD_MAX_CHANNELS", "2")
        monkeypatch.setenv("PTPD_QUAD_MAX_BYTES", "64")
        limits = QuadParserLimits()
        assert limits.max_channels == 2
        assert limits.max_bytes == 64

    def test_env_override_is_enforced_by_default_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PTPD_QUAD_MAX_BYTES", "16")
        with pytest.raises(ValueError, match="exceeds the limit"):
            load_quad_string(INI_PROFILE)


class TestSizeCaps:
    def test_oversize_string_rejected(self) -> None:
        parser = QuadFileParser(QuadParserLimits(max_bytes=100))
        with pytest.raises(ValueError, match="exceeds the limit"):
            parser.parse_string("#" * 101)

    def test_oversize_file_rejected_without_reading(self, tmp_path: Path) -> None:
        path = tmp_path / "huge.quad"
        path.write_bytes(b"#" * (2 * 1024 * 1024 + 1))
        started = time.perf_counter()
        with pytest.raises(ValueError, match="bytes"):
            load_quad_file(path)
        assert time.perf_counter() - started < 1.0

    def test_file_at_limit_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.quad"
        content = INI_PROFILE.encode()
        path.write_bytes(content)
        parser = QuadFileParser(QuadParserLimits(max_bytes=len(content)))
        assert parser.parse(path).profile_name == "Test Profile"

    def test_hundred_thousand_curve_headers_rejected_quickly(self) -> None:
        lines = ["## QuadToneRIP K"] + [f"# C{i} Curve" for i in range(100_000)]
        content = "\n".join(lines)
        parser = QuadFileParser()
        started = time.perf_counter()
        with pytest.raises(ValueError, match="more than 16 channels"):
            parser.parse_string(content)
        assert time.perf_counter() - started < 1.0
        assert parser._profile is not None
        assert len(parser._profile.channels) <= 16

    def test_repeated_known_channel_headers_do_not_count_twice(self) -> None:
        content = "## QuadToneRIP K\n" + "# K Curve\n0\n" * 50
        profile = load_quad_string(content)
        assert "K" in profile.channels

    def test_max_sections_enforced(self) -> None:
        parser = QuadFileParser(QuadParserLimits(max_sections=3))
        content = "\n".join(f"[S{i}]\nx=1" for i in range(4))
        with pytest.raises(ValueError, match="more than 3 sections"):
            parser.parse_string(content)

    def test_max_keys_per_section_enforced(self) -> None:
        parser = QuadFileParser(QuadParserLimits(max_keys_per_section=3))
        content = "[General]\n" + "\n".join(f"k{i}=v" for i in range(4))
        with pytest.raises(ValueError, match="more than 3 keys"):
            parser.parse_string(content)

    def test_duplicate_keys_do_not_count_against_cap(self) -> None:
        parser = QuadFileParser(QuadParserLimits(max_keys_per_section=2))
        content = "[General]\nProfileName=a\nProfileName=b\nProfileName=c\nResolution=1440"
        profile = parser.parse_string(content)
        assert profile.profile_name == "c"
        assert profile.resolution == 1440


class TestOverflowHandling:
    @pytest.mark.parametrize("value", ["inf", "1e400", "-inf", "nan"])
    def test_non_finite_channel_value_raises_value_error(self, value: str) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            load_quad_string(f"[K]\n0={value}\n")

    @pytest.mark.parametrize("value", ["inf", "1e400"])
    def test_non_finite_ink_limit_raises_value_error(self, value: str) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            load_quad_string(f"[General]\nInkLimit={value}\n")

    def test_overflow_error_never_escapes(self) -> None:
        try:
            load_quad_string("[K]\n0=1e400\n")
        except OverflowError:  # pragma: no cover - the regression this guards against
            pytest.fail("OverflowError leaked from the parser")
        except ValueError:
            pass

    def test_non_numeric_channel_value_still_ignored(self) -> None:
        profile = load_quad_string("[K]\n0=abc\n1=12\n")
        assert profile.channels["K"].values[0] == 0
        assert profile.channels["K"].values[1] == 12

    def test_huge_finite_value_clamped(self) -> None:
        profile = load_quad_string("[K]\n0=1e300\n")
        assert profile.channels["K"].values[0] == 255

    def test_simple_format_huge_integer_ignored(self) -> None:
        content = "## QuadToneRIP K\n# K Curve\n" + "9" * 400 + "\n65535\n"
        profile = load_quad_string(content)
        # The oversized line is skipped; the next value lands at index 0.
        assert profile.channels["K"].values[0] == 255


class TestNameTruncation:
    def test_long_profile_name_truncated_and_curve_data_valid(self) -> None:
        long_name = "N" * 300
        profile = load_quad_string(f"[General]\nProfileName={long_name}\n[K]\n0=0\n255=255\n")
        assert len(profile.profile_name) == 200
        curve = profile.to_curve_data("K")
        assert curve.name.startswith("K - " + "N" * 10)
        assert len(curve.name) <= 256

    def test_long_media_type_truncated(self) -> None:
        profile = load_quad_string("[General]\nMediaType=" + "M" * 250 + "\n[K]\n0=1\n")
        assert len(profile.media_type) == 200
        assert profile.to_curve_data("K").paper_type == "M" * 200

    def test_parse_string_name_argument_truncated(self) -> None:
        profile = load_quad_string("[K]\n0=1\n", name="P" * 500)
        assert len(profile.profile_name) == 200

    def test_custom_name_limit(self) -> None:
        parser = QuadFileParser(QuadParserLimits(max_name_length=5))
        profile = parser.parse_string("[General]\nProfileName=abcdefgh\n")
        assert profile.profile_name == "abcde"

    def test_long_channel_name_never_breaks_curve_data(self) -> None:
        content = "## QuadToneRIP K\n# " + "Z" * 400 + " Curve\n65535\n0\n"
        profile = load_quad_string(content, name="Q" * 300)
        channel = next(name for name in profile.channels if name.startswith("Z"))
        assert len(channel) == 200
        curve = profile.to_curve_data(channel)
        assert len(curve.name) <= 256


class TestEncodingAndBom:
    def test_utf8_bom_ini_string(self) -> None:
        profile = load_quad_string("﻿" + INI_PROFILE)
        assert profile.profile_name == "Test Profile"
        assert profile.channels["K"].values[255] == 255

    def test_utf8_bom_simple_format_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.quad"
        path.write_text("## QuadToneRIP K\n# K Curve\n0\n65535\n", encoding="utf-8-sig")
        profile = load_quad_file(path)
        assert profile.channels["K"].values[1] == 255

    def test_utf16_file_still_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "utf16.quad"
        path.write_text(INI_PROFILE, encoding="utf-16")
        profile = load_quad_file(path)
        assert profile.profile_name == "Test Profile"
        assert profile.channels["K"].values[128] == 128

    def test_latin1_file_still_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "latin1.quad"
        path.write_bytes(INI_PROFILE.replace("Test Profile", "Caf\xe9").encode("latin-1"))
        profile = load_quad_file(path)
        assert profile.profile_name == "Caf\xe9"

    def test_decode_happens_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "count.quad"
        path.write_text(INI_PROFILE)
        parser = QuadFileParser()
        calls: list[bytes] = []
        original = parser._decode

        def counting(raw: bytes, label: str) -> str:
            calls.append(raw)
            return original(raw, label)

        monkeypatch.setattr(parser, "_decode", counting)
        parser.parse(path)
        assert len(calls) == 1


class TestQuantisationUnchanged:
    """The 16-bit to 8-bit mapping the export round-trip test depends on."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0, 0), (65535, 255), (32768, 127), (257, 1), (70000, 255), (-5, 0)],
    )
    def test_simple_format_values(self, raw: int, expected: int) -> None:
        profile = load_quad_string(f"## QuadToneRIP K\n# K Curve\n{raw}\n")
        assert profile.channels["K"].values[0] == expected

    def test_ini_values_clamped_to_eight_bit(self) -> None:
        profile = load_quad_string("[K]\n0=300\n1=-4\n2=127.9\n")
        assert profile.channels["K"].values[:3] == [255, 0, 127]
