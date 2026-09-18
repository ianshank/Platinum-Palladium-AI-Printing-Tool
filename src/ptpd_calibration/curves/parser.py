"""
Comprehensive parser for QTR .quad files.

Parses QuadTone RIP profile files with full metadata extraction
and multi-channel curve support.

Input hardening (plan item SEC-14): a ``.quad`` file is untrusted input, so
the parser enforces the caps on :class:`QuadParserLimits` (file size, number
of channels, sections and keys, name lengths), decodes the file exactly once,
strips a UTF-8 BOM before header detection and rejects non-finite numeric
values with ``ValueError`` instead of leaking ``OverflowError``.
"""

from __future__ import annotations

import codecs
import logging
import math
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from annotated_types import MaxLen
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType

logger = logging.getLogger(__name__)


def _curve_name_limit() -> int:
    """Maximum ``CurveData.name`` length, read from the model so the two never drift."""
    for constraint in CurveData.model_fields["name"].metadata:
        if isinstance(constraint, MaxLen):
            return int(constraint.max_length)
    return 256  # pragma: no cover - CurveData.name always carries a max_length


_CURVE_NAME_MAX = _curve_name_limit()


class QuadParserLimits(BaseSettings):
    """Size caps applied while parsing ``.quad`` files.

    Every field can be overridden with a ``PTPD_QUAD_`` environment variable,
    e.g. ``PTPD_QUAD_MAX_BYTES=4194304``.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_QUAD_")

    max_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1,
        description="Maximum file size (or string length) accepted",
    )
    max_channels: int = Field(
        default=16, ge=1, description="Maximum number of ink channels a profile may declare"
    )
    max_sections: int = Field(
        default=64, ge=1, description="Maximum number of [Section] headers in INI-style files"
    )
    max_keys_per_section: int = Field(
        default=1024, ge=1, description="Maximum key=value pairs stored per section"
    )
    max_name_length: int = Field(
        default=200,
        ge=1,
        description="Names and metadata strings longer than this are truncated",
    )


@dataclass
class ChannelCurve:
    """A single channel curve from a .quad file."""

    name: str
    values: list[int]  # 0-255 values for each input step
    enabled: bool = True

    @property
    def as_normalized(self) -> tuple[list[float], list[float]]:
        """Get curve as normalized 0-1 input/output pairs."""
        inputs = [
            i / (len(self.values) - 1) if len(self.values) > 1 else 0.0
            for i in range(len(self.values))
        ]
        outputs = [v / 255.0 for v in self.values]
        return inputs, outputs

    def to_curve_data(self, name_suffix: str = "") -> CurveData:
        """Convert to CurveData model."""
        inputs, outputs = self.as_normalized
        name = f"{self.name}{name_suffix}"
        if len(name) > _CURVE_NAME_MAX:
            logger.debug("Truncating curve name from %d to %d chars", len(name), _CURVE_NAME_MAX)
            name = name[:_CURVE_NAME_MAX]
        return CurveData(
            name=name,
            input_values=inputs,
            output_values=outputs,
            curve_type=CurveType.CUSTOM,
        )


@dataclass
class QuadProfile:
    """
    Complete parsed .quad profile.

    Contains all metadata and channel curves from a QTR profile file.
    """

    # File info
    source_path: Path | None = None
    profile_name: str = "Untitled"

    # General settings
    resolution: int = 2880
    ink_limit: float = 100.0
    gray_ink_limit: float = 100.0
    linearization_type: str = "none"
    black_generation: str = "none"

    # Media settings
    media_type: str = ""
    media_setting: str = ""

    # Channel curves (K, C, M, Y, LC, LM, LK, LLK, etc.)
    channels: dict[str, ChannelCurve] = field(default_factory=dict)

    # Raw sections for preservation
    raw_sections: dict[str, dict[str, str]] = field(default_factory=dict)

    # Comments and metadata
    comments: list[str] = field(default_factory=list)

    @property
    def primary_channel(self) -> ChannelCurve | None:
        """Get the primary (K) channel curve."""
        return self.channels.get("K")

    @property
    def all_channel_names(self) -> list[str]:
        """Get list of all channel names."""
        return list(self.channels.keys())

    @property
    def active_channels(self) -> list[str]:
        """Get list of channels with non-zero curves."""
        active = []
        for name, curve in self.channels.items():
            if curve.enabled and any(v > 0 for v in curve.values):
                active.append(name)
        return active

    def get_channel(self, name: str) -> ChannelCurve | None:
        """Get a specific channel curve."""
        return self.channels.get(name.upper())

    def to_curve_data(self, channel: str = "K") -> CurveData:
        """
        Convert a channel to CurveData model.

        Args:
            channel: Channel name (K, C, M, etc.)

        Returns:
            CurveData for the specified channel.
        """
        ch = self.get_channel(channel)
        if ch is None:
            raise ValueError(f"Channel '{channel}' not found in profile")

        curve = ch.to_curve_data(f" - {self.profile_name}")
        curve.paper_type = self.media_type or None
        curve.notes = f"Imported from {self.source_path.name if self.source_path else 'unknown'}"

        return curve

    def summary(self) -> str:
        """Generate a summary string."""
        active = self.active_channels
        return (
            f"Profile: {self.profile_name}\n"
            f"Resolution: {self.resolution} DPI\n"
            f"Ink Limit: {self.ink_limit}%\n"
            f"Active Channels: {', '.join(active) if active else 'None'}\n"
            f"Media: {self.media_type or 'Not specified'}"
        )


class QuadFileParser:
    """
    Parser for QuadTone RIP .quad profile files.

    Handles various .quad file formats and extracts all
    metadata and curve information.
    """

    # Known section names
    GENERAL_SECTION = "General"
    CHANNEL_SECTIONS = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK", "PK", "MK"]

    def __init__(self, limits: QuadParserLimits | None = None) -> None:
        """Initialize the parser.

        Args:
            limits: Size caps to enforce; ``None`` reads :class:`QuadParserLimits`
                from the environment.
        """
        self.limits = limits or QuadParserLimits()
        self._current_section: str | None = None
        self._profile: QuadProfile | None = None

    def parse(self, path: Path) -> QuadProfile:
        """
        Parse a .quad file.

        Args:
            path: Path to the .quad file.

        Returns:
            QuadProfile with all parsed data.

        Raises:
            FileNotFoundError: The file does not exist.
            ValueError: The file exceeds a :class:`QuadParserLimits` cap or
                contains a non-finite numeric value.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size = path.stat().st_size
        if size > self.limits.max_bytes:
            logger.debug(
                "Rejected %s: %d bytes exceeds max_bytes=%d", path, size, self.limits.max_bytes
            )
            raise ValueError(
                f".quad file {path.name} is {size} bytes, which exceeds the limit of "
                f"{self.limits.max_bytes} bytes"
            )

        self._profile = QuadProfile(source_path=path)
        self._current_section = None

        content = self._decode(path.read_bytes(), str(path))

        # Parse the content
        self._parse_content(content)

        # Post-process
        self._post_process()

        return self._profile

    def parse_string(self, content: str, name: str = "Untitled") -> QuadProfile:
        """
        Parse .quad content from a string.

        Args:
            content: .quad file content as string.
            name: Name to use for the profile.

        Returns:
            QuadProfile with all parsed data.

        Raises:
            ValueError: The content exceeds a :class:`QuadParserLimits` cap or
                contains a non-finite numeric value.
        """
        if len(content) > self.limits.max_bytes:
            logger.debug(
                "Rejected string input: %d chars exceeds max_bytes=%d",
                len(content),
                self.limits.max_bytes,
            )
            raise ValueError(
                f".quad content is {len(content)} characters, which exceeds the limit of "
                f"{self.limits.max_bytes}"
            )

        self._profile = QuadProfile(profile_name=self._clip(name, "profile name"))
        self._current_section = None

        self._parse_content(content)
        self._post_process()

        return self._profile

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _decode(self, raw: bytes, label: str) -> str:
        """Decode file bytes exactly once.

        UTF-16 is recognised by its BOM; everything else is tried as UTF-8
        (a UTF-8 BOM is consumed) and falls back to Latin-1, which cannot fail
        and matches the behaviour of the previous multi-encoding probe for
        Windows/Mac authored profiles.
        """
        if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            logger.debug("Decoding %s as UTF-16 (BOM present)", label)
            return raw.decode("utf-16")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            logger.debug("Decoding %s as Latin-1 after UTF-8 failure at byte %d", label, exc.start)
            return raw.decode("latin-1")

    def _clip(self, value: str, what: str) -> str:
        """Truncate ``value`` to ``max_name_length`` so downstream models validate."""
        limit = self.limits.max_name_length
        if len(value) > limit:
            logger.debug("Truncating %s from %d to %d chars", what, len(value), limit)
            return value[:limit]
        return value

    def _ensure_channel(self, name: str) -> ChannelCurve:
        """Return the channel called ``name``, creating it under the channel cap."""
        assert self._profile is not None
        channel = self._profile.channels.get(name)
        if channel is None:
            if len(self._profile.channels) >= self.limits.max_channels:
                logger.debug(
                    "Rejected channel %r: max_channels=%d reached", name, self.limits.max_channels
                )
                raise ValueError(
                    f".quad profile declares more than {self.limits.max_channels} channels"
                )
            channel = ChannelCurve(name=name, values=[0] * 256)
            self._profile.channels[name] = channel
        return channel

    def _ensure_section(self, name: str) -> dict[str, str]:
        """Return raw storage for section ``name``, creating it under the section cap."""
        assert self._profile is not None
        section = self._profile.raw_sections.get(name)
        if section is None:
            if len(self._profile.raw_sections) >= self.limits.max_sections:
                logger.debug(
                    "Rejected section %r: max_sections=%d reached", name, self.limits.max_sections
                )
                raise ValueError(
                    f".quad profile declares more than {self.limits.max_sections} sections"
                )
            section = {}
            self._profile.raw_sections[name] = section
        return section

    @staticmethod
    def _finite_float(value: str, where: str) -> float | None:
        """Parse ``value`` as a float; ``None`` if non-numeric, ``ValueError`` if non-finite."""
        try:
            number = float(value)
        except ValueError:
            return None
        if not math.isfinite(number):
            logger.debug("Rejected non-finite value %r at %s", value, where)
            raise ValueError(f"Non-finite value {value!r} at {where} in .quad profile")
        return number

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_content(self, content: str) -> None:
        """Parse the file content."""
        assert self._profile is not None
        if content.startswith("﻿"):
            logger.debug("Stripping UTF-8 BOM before header detection")
            content = content[1:]
        lines = content.split("\n")

        # Check for simple format (starts with ## QuadToneRIP)
        if len(lines) > 0 and lines[0].startswith("## QuadToneRIP"):
            self._parse_simple_format(lines)
            return

        for _line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Handle comments
            if line.startswith("#") or line.startswith(";"):
                self._profile.comments.append(line[1:].strip())
                continue

            # Check for section header
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].strip()
                self._current_section = section_name

                # Initialize section storage
                self._ensure_section(section_name)

                # Initialize channel if it's a known channel section
                if section_name.upper() in self.CHANNEL_SECTIONS:
                    self._ensure_channel(section_name.upper())
                continue

            # Parse key=value pairs
            if "=" in line:
                self._parse_key_value(line)

    def _parse_simple_format(self, lines: list[str]) -> None:
        """Parse the simple list-based QuadToneRIP format."""
        assert self._profile is not None
        current_channel = None
        value_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Check for curve header (e.g. "# K Curve")
                if " Curve" in line:
                    parts = line.replace("#", "").strip().split()
                    # Usually "# K Curve" -> parts=["K", "Curve"]
                    if len(parts) >= 2 and parts[-1] == "Curve":
                        # Handle "K Curve", "LC Curve" etc.
                        channel_name = self._clip(parts[0].upper(), "channel name")
                        current_channel = channel_name
                        self._ensure_channel(current_channel)
                        value_index = 0
                elif line.startswith("## QuadToneRIP"):
                    pass  # Header
                else:
                    self._profile.comments.append(line[1:].strip())
                continue

            # Parse number
            if current_channel and (line[0].isdigit() or line.startswith("-")):
                try:
                    val = int(line)
                    # QTR simple format uses 16-bit values (0-65535)
                    # We normalize to 8-bit (0-255) for internal storage
                    norm_val = int((val / 65535.0) * 255.0)

                    if value_index < 256:
                        self._profile.channels[current_channel].values[value_index] = max(
                            0, min(255, norm_val)
                        )
                        value_index += 1
                except (ValueError, OverflowError):
                    pass

    def _parse_key_value(self, line: str) -> None:
        """Parse a key=value line."""
        assert self._profile is not None
        # Split on first = only
        parts = line.split("=", 1)
        if len(parts) != 2:
            return

        key = parts[0].strip()
        value = parts[1].strip()

        # Store in raw sections
        if self._current_section:
            section = self._ensure_section(self._current_section)
            if key not in section and len(section) >= self.limits.max_keys_per_section:
                logger.debug(
                    "Rejected key %r in [%s]: max_keys_per_section=%d reached",
                    key,
                    self._current_section,
                    self.limits.max_keys_per_section,
                )
                raise ValueError(
                    f"Section [{self._current_section}] has more than "
                    f"{self.limits.max_keys_per_section} keys"
                )
            section[key] = value

        # Process based on current section
        if self._current_section == self.GENERAL_SECTION:
            self._parse_general_setting(key, value)
        elif self._current_section and self._current_section.upper() in self.CHANNEL_SECTIONS:
            self._parse_channel_value(self._current_section.upper(), key, value)

    def _parse_general_setting(self, key: str, value: str) -> None:
        """Parse a general section setting."""
        assert self._profile is not None
        key_lower = key.lower()

        if key_lower == "profilename":
            self._profile.profile_name = self._clip(value, "profile name")
        elif key_lower == "resolution":
            with suppress(ValueError):
                self._profile.resolution = int(value)
        elif key_lower == "inklimit":
            number = self._finite_float(value, "InkLimit")
            if number is not None:
                self._profile.ink_limit = number
        elif key_lower == "grayinklimit":
            number = self._finite_float(value, "GrayInkLimit")
            if number is not None:
                self._profile.gray_ink_limit = number
        elif key_lower == "linearizationtype":
            self._profile.linearization_type = self._clip(value, "linearization type")
        elif key_lower == "blackgeneration":
            self._profile.black_generation = self._clip(value, "black generation")
        elif key_lower == "mediatype":
            self._profile.media_type = self._clip(value, "media type")
        elif key_lower == "mediasetting":
            self._profile.media_setting = self._clip(value, "media setting")

    def _parse_channel_value(self, channel: str, key: str, value: str) -> None:
        """Parse a channel curve value."""
        assert self._profile is not None
        # Check if key is a numeric index
        if key.isdigit():
            number = self._finite_float(value, f"[{channel}] {key}")
            if number is None:
                return
            try:
                index = int(key)
                output = int(number)

                if 0 <= index < 256:
                    self._profile.channels[channel].values[index] = max(0, min(255, output))
            except (ValueError, IndexError, OverflowError):
                pass

    def _post_process(self) -> None:
        """Post-process the parsed profile."""
        assert self._profile is not None
        # Ensure all standard channels exist (even if empty)
        for channel_name in ["K", "C", "M", "Y", "LC", "LM", "LK"]:
            if channel_name not in self._profile.channels:
                self._profile.channels[channel_name] = ChannelCurve(
                    name=channel_name,
                    values=[0] * 256,
                    enabled=False,
                )

        # Mark channels with data as enabled
        for channel in self._profile.channels.values():
            channel.enabled = any(v > 0 for v in channel.values)


def load_quad_file(path: Path) -> QuadProfile:
    """
    Convenience function to load a .quad file.

    Args:
        path: Path to the .quad file.

    Returns:
        QuadProfile with all parsed data.
    """
    parser = QuadFileParser()
    return parser.parse(path)


def load_quad_string(content: str, name: str = "Untitled") -> QuadProfile:
    """
    Convenience function to parse .quad content from string.

    Args:
        content: .quad file content.
        name: Profile name.

    Returns:
        QuadProfile with all parsed data.
    """
    parser = QuadFileParser()
    return parser.parse_string(content, name)
