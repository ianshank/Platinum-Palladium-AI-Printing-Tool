"""
Comprehensive parser for QTR .quad files.

Parses QuadTone RIP profile files with full metadata extraction
and multi-channel curve support.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType


@dataclass
class ChannelCurve:
    """A single channel curve from a .quad file."""

    name: str
    values: list[int]  # 0-255 values for each input step
    enabled: bool = True

    @property
    def as_normalized(self) -> tuple[list[float], list[float]]:
        """Get curve as normalized 0-1 input/output pairs."""
        inputs = [i / (len(self.values) - 1) if len(self.values) > 1 else 0.0 for i in range(len(self.values))]
        outputs = [v / 255.0 for v in self.values]
        return inputs, outputs

    def to_curve_data(self, name_suffix: str = "") -> CurveData:
        """Convert to CurveData model."""
        inputs, outputs = self.as_normalized
        return CurveData(
            name=f"{self.name}{name_suffix}",
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
    source_path: Optional[Path] = None
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
    def primary_channel(self) -> Optional[ChannelCurve]:
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

    def get_channel(self, name: str) -> Optional[ChannelCurve]:
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

    def __init__(self):
        """Initialize the parser."""
        self._current_section: Optional[str] = None
        self._profile: Optional[QuadProfile] = None

    def parse(self, path: Path) -> QuadProfile:
        """
        Parse a .quad file.

        Args:
            path: Path to the .quad file.

        Returns:
            QuadProfile with all parsed data.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self._profile = QuadProfile(source_path=path)
        self._current_section = None

        content = None
        # Try different encodings
        for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                    # Check if content looks reasonable (has sections)
                    if "[" in content and "]" in content:
                        break
            except UnicodeError:
                continue

        if content is None:
            # Fallback to replace
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

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
        """
        self._profile = QuadProfile(profile_name=name)
        self._current_section = None

        self._parse_content(content)
        self._post_process()

        return self._profile

    def _parse_content(self, content: str) -> None:
        """Parse the file content."""
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
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
                if section_name not in self._profile.raw_sections:
                    self._profile.raw_sections[section_name] = {}

                # Initialize channel if it's a known channel section
                if section_name.upper() in self.CHANNEL_SECTIONS:
                    if section_name.upper() not in self._profile.channels:
                        self._profile.channels[section_name.upper()] = ChannelCurve(
                            name=section_name.upper(),
                            values=[0] * 256,
                        )
                continue

            # Parse key=value pairs
            if "=" in line:
                self._parse_key_value(line)

    def _parse_key_value(self, line: str) -> None:
        """Parse a key=value line."""
        # Split on first = only
        parts = line.split("=", 1)
        if len(parts) != 2:
            return

        key = parts[0].strip()
        value = parts[1].strip()

        # Store in raw sections
        if self._current_section:
            self._profile.raw_sections[self._current_section][key] = value

        # Process based on current section
        if self._current_section == self.GENERAL_SECTION:
            self._parse_general_setting(key, value)
        elif self._current_section and self._current_section.upper() in self.CHANNEL_SECTIONS:
            self._parse_channel_value(self._current_section.upper(), key, value)

    def _parse_general_setting(self, key: str, value: str) -> None:
        """Parse a general section setting."""
        key_lower = key.lower()

        if key_lower == "profilename":
            self._profile.profile_name = value
        elif key_lower == "resolution":
            try:
                self._profile.resolution = int(value)
            except ValueError:
                pass
        elif key_lower == "inklimit":
            try:
                self._profile.ink_limit = float(value)
            except ValueError:
                pass
        elif key_lower == "grayinklimit":
            try:
                self._profile.gray_ink_limit = float(value)
            except ValueError:
                pass
        elif key_lower == "linearizationtype":
            self._profile.linearization_type = value
        elif key_lower == "blackgeneration":
            self._profile.black_generation = value
        elif key_lower == "mediatype":
            self._profile.media_type = value
        elif key_lower == "mediasetting":
            self._profile.media_setting = value

    def _parse_channel_value(self, channel: str, key: str, value: str) -> None:
        """Parse a channel curve value."""
        # Check if key is a numeric index
        if key.isdigit():
            try:
                index = int(key)
                output = int(float(value))

                if 0 <= index < 256:
                    self._profile.channels[channel].values[index] = max(0, min(255, output))
            except (ValueError, IndexError):
                pass

    def _post_process(self) -> None:
        """Post-process the parsed profile."""
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
