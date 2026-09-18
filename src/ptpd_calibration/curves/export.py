"""
Curve export to various RIP formats.

Supports QuadTone RIP, Piezography, CSV, and JSON formats.
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.models import CurveData

logger = logging.getLogger(__name__)

# Header that identifies the QuadToneRIP list format written by QTRExporter.
QTR_HEADER = "## QuadToneRIP"

# Metadata comment prefixes written by QTRExporter that are restored on load.
_QTR_NAME_KEYS = ("Name:", "Profile:")
_QTR_PAPER_KEY = "Paper:"
_QTR_CHEMISTRY_KEY = "Chemistry:"
_QTR_UNKNOWN = "Unknown"


class CurveExporter(ABC):
    """Abstract base class for curve exporters."""

    @abstractmethod
    def export(self, curve: CurveData, path: Path) -> None:
        """Export curve to file."""
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Get format name."""
        pass


class QTRExporter(CurveExporter):
    """
    Export curves to QuadTone RIP format.

    Supports both simple curve files and full .quad profiles.
    """

    def __init__(
        self,
        primary_channel: str = "K",
        ink_limit: float | None = None,
        resolution: int | None = None,
    ):
        """
        Initialize QTR exporter.

        Args:
            primary_channel: Primary ink channel (K, C, M, Y, LC, LM, LK).
            ink_limit: Maximum ink limit (0-100).
            resolution: Print resolution in DPI.
        """
        settings = get_settings().curves
        self.primary_channel = primary_channel
        self.ink_limit = ink_limit if ink_limit is not None else settings.qtr_ink_limit
        self.resolution = resolution if resolution is not None else settings.qtr_resolution

    def export(self, curve: CurveData, path: Path, format: str = "curve") -> None:
        """
        Export curve to QTR format.

        Args:
            curve: Curve data to export.
            path: Output file path.
            format: "curve" for simple curve, "quad" for full profile.
        """
        if format == "quad":
            self._export_quad_profile(curve, path)
        else:
            self._export_curve_file(curve, path)

    def _export_curve_file(self, curve: CurveData, path: Path) -> None:
        """Export as QTR curve file (single channel).

        Uses QuadTone RIP format with 16-bit values.
        """
        lines = [
            f"## QuadToneRIP {self.primary_channel}",
            "# QTR Curve File",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Name: {curve.name}",
            f"# Paper: {curve.paper_type or 'Unknown'}",
            f"# Chemistry: {curve.chemistry or 'Unknown'}",
            f"# Ink Limit: {self.ink_limit}%",
            f"# {self.primary_channel} Curve",
        ]

        # Interpolate to 256 points and convert to 16-bit values
        x_new = np.linspace(0, 1, 256)
        y_new = np.interp(x_new, curve.input_values, curve.output_values)

        for out in y_new:
            # Convert to 16-bit value (0-65535) with ink limit
            qtr_output = int(out * 65535 * self.ink_limit / 100)
            qtr_output = max(0, min(65535, qtr_output))
            lines.append(str(qtr_output))

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _export_quad_profile(self, curve: CurveData, path: Path) -> None:
        """Export as full .quad profile in QuadTone RIP format.

        QuadTone RIP format specification:
        - Header: ## QuadToneRIP K,C,M,Y,LC,LM,LK,LLK
        - Comments: lines starting with #
        - Channel headers: # K Curve, # C Curve, etc.
        - Values: 256 integers per channel (0-65535 range), one per line
        - NO section brackets like [K]
        - NO index= format like 0=value
        """
        # Standard channel order
        all_channels = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK"]

        lines = [
            f"## QuadToneRIP {','.join(all_channels)}",
            f"# Profile: {curve.name}",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Paper: {curve.paper_type or 'Unknown'}",
            f"# Chemistry: {curve.chemistry or 'Unknown'}",
            f"# Resolution: {self.resolution}",
            f"# Ink Limit: {self.ink_limit}%",
        ]

        # QTR uses 256-point curves with 16-bit values (0-65535)
        x_new = np.linspace(0, 1, 256)
        y_new = np.interp(x_new, curve.input_values, curve.output_values)

        # Primary channel
        lines.append(f"# {self.primary_channel} Curve")
        for out in y_new:
            # Convert to 16-bit value (0-65535) with ink limit
            qtr_output = int(out * 65535 * self.ink_limit / 100)
            qtr_output = max(0, min(65535, qtr_output))
            lines.append(str(qtr_output))

        # Add empty curves for other channels
        for channel in all_channels:
            if channel != self.primary_channel:
                lines.append(f"# {channel} Curve")
                for _ in range(256):
                    lines.append("0")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def get_format_name(self) -> str:
        return "QuadTone RIP"


class PiezographyExporter(CurveExporter):
    """
    Export curves to Piezography Print Tool format.

    Piezography uses percentage-based curves (0-100).
    """

    def __init__(
        self,
        ink_set: str = "K7",
        linearization_type: str = "single",
    ):
        """
        Initialize Piezography exporter.

        Args:
            ink_set: Ink set name (K7, K6, Pro, etc.).
            linearization_type: "single" or "multi" channel.
        """
        self.ink_set = ink_set
        self.linearization_type = linearization_type

    def export(self, curve: CurveData, path: Path, format: str = "ppt") -> None:
        """
        Export curve to Piezography format.

        Args:
            curve: Curve data to export.
            path: Output file path.
            format: "ppt" for PPT format, "qtr" for QTR-compatible.
        """
        if format == "qtr":
            self._export_qtr_compatible(curve, path)
        else:
            self._export_ppt_format(curve, path)

    def _export_ppt_format(self, curve: CurveData, path: Path) -> None:
        """Export as Piezography Print Tool format."""
        lines = [
            "# Piezography Linearization",
            f"# Name: {curve.name}",
            f"# Ink Set: {self.ink_set}",
            f"# Paper: {curve.paper_type or 'Unknown'}",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "[Linearization]",
            f"Type={self.linearization_type}",
            f"InkSet={self.ink_set}",
            "",
            "[Curve]",
        ]

        # Piezography uses 101 points (0-100%)
        x_new = np.linspace(0, 1, 101)
        y_new = np.interp(x_new, curve.input_values, curve.output_values)

        for _i, (inp, out) in enumerate(zip(x_new, y_new, strict=True)):
            pz_input = int(inp * 100)
            pz_output = out * 100
            lines.append(f"{pz_input}={pz_output:.2f}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _export_qtr_compatible(self, curve: CurveData, path: Path) -> None:
        """Export as QTR-compatible format for Piezography."""
        qtr_exporter = QTRExporter(primary_channel="K")
        qtr_exporter.export(curve, path, format="curve")

    def get_format_name(self) -> str:
        return "Piezography"


class CSVExporter(CurveExporter):
    """Export curves to CSV format."""

    def export(self, curve: CurveData, path: Path) -> None:
        """Export curve to CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "output"])
            for inp, out in zip(curve.input_values, curve.output_values, strict=True):
                writer.writerow([f"{inp:.6f}", f"{out:.6f}"])

    def get_format_name(self) -> str:
        return "CSV"


class JSONExporter(CurveExporter):
    """Export curves to JSON format."""

    def export(self, curve: CurveData, path: Path) -> None:
        """Export curve to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = curve.model_dump(mode="json")
        # Convert UUID to string
        data["id"] = str(curve.id)
        if curve.source_extraction_id:
            data["source_extraction_id"] = str(curve.source_extraction_id)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_format_name(self) -> str:
        return "JSON"


def save_curve(curve: CurveData, path: Path, format: str | None = None) -> None:
    """
    Save curve to file in specified format.

    Args:
        curve: Curve data to save.
        path: Output file path.
        format: Format name or None to infer from extension.
    """
    if format is None:
        # Infer from extension
        ext = path.suffix.lower()
        format_map = {
            ".txt": "qtr",
            ".quad": "qtr",
            ".ppt": "piezography",
            ".csv": "csv",
            ".json": "json",
        }
        format = format_map.get(ext, "json")

    format = format.lower()

    if format in ("qtr", "quadtone"):
        qtr_exporter = QTRExporter()
        export_format = "quad" if path.suffix.lower() == ".quad" else "curve"
        qtr_exporter.export(curve, path, format=export_format)
    elif format in ("piezography", "pz", "ppt"):
        exporter: CurveExporter = PiezographyExporter()
        exporter.export(curve, path)
    elif format == "csv":
        exporter = CSVExporter()
        exporter.export(curve, path)
    else:
        exporter = JSONExporter()
        exporter.export(curve, path)


def load_curve(path: Path) -> CurveData:
    """
    Load curve from file.

    Args:
        path: Path to curve file.

    Returns:
        CurveData loaded from file.

    Notes:
        ``.quad``/``.txt`` files written by :class:`QTRExporter` (QuadToneRIP
        list format, ``## QuadToneRIP`` header) are read back through
        :class:`~ptpd_calibration.curves.parser.QuadFileParser`. The parser
        stores channel values as 8-bit, so a ``save_curve``/``load_curve``
        round trip of a QTR file reproduces the curve only to within 2/255.
    """
    ext = path.suffix.lower()

    if ext == ".json":
        return _load_json_curve(path)
    elif ext == ".csv":
        return _load_csv_curve(path)
    elif ext in (".txt", ".quad", ".ppt"):
        return _load_text_curve(path)
    else:
        # Try JSON first, then text
        try:
            return _load_json_curve(path)
        except json.JSONDecodeError:
            return _load_text_curve(path)


def _load_json_curve(path: Path) -> CurveData:
    """Load curve from JSON file."""
    with open(path) as f:
        data = json.load(f)

    return CurveData(**data)


def _load_csv_curve(path: Path) -> CurveData:
    """Load curve from CSV file."""
    input_values = []
    output_values = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_values.append(float(row["input"]))
            output_values.append(float(row["output"]))

    return CurveData(
        name=path.stem,
        input_values=input_values,
        output_values=output_values,
    )


def _load_text_curve(path: Path) -> CurveData:
    """Load curve from text format (QTR/Piezography).

    Two layouts are understood:

    * ``key=value`` lines (Piezography ``.ppt``, bracketed ``[K]`` sections):
      parsed here directly as 0-255 index/value pairs.
    * QuadToneRIP list format (``## QuadToneRIP`` header, one integer per line
      under ``# <channel> Curve``), i.e. what :class:`QTRExporter` writes:
      delegated to :func:`_load_qtr_list_curve`.

    Before the delegation existed, ``save_curve(curve, "x.quad")`` followed by
    ``load_curve("x.quad")`` always raised ``ValueError("No curve data found")``
    because only the ``key=value`` layout was recognised (probe finding F3).
    That was a defect in the public round-trip contract and is fixed here.
    """
    with open(path) as f:
        header = f.readline()
    if header.lstrip().startswith(QTR_HEADER):
        return _load_qtr_list_curve(path)

    input_values = []
    output_values = []
    name = path.stem

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Check for curve data
                if key.isdigit():
                    inp = int(key) / 255.0  # Assume 0-255 range
                    out = float(value) / 255.0
                    input_values.append(inp)
                    output_values.append(out)
                elif key.lower() == "profilename":
                    name = value

    if not input_values:
        raise ValueError(f"No curve data found in {path}")

    return CurveData(
        name=name,
        input_values=input_values,
        output_values=output_values,
    )


def _load_qtr_list_curve(path: Path) -> CurveData:
    """Load a QuadToneRIP list-format file via :class:`QuadFileParser`.

    The first channel carrying non-zero data is used (``K`` when every channel
    is empty). Name, paper type and chemistry are restored from the header
    comments that :class:`QTRExporter` writes (read directly from the file:
    the parser treats any comment ending in " Curve" as a channel header, so
    its ``comments`` list is not reliable for this); the name falls back to
    the file stem. Values are quantised to 8-bit by the parser, so the loaded
    curve matches the exported one only to within 2/255 (16-bit -> 8-bit
    truncation plus ``int()`` truncation on export).
    """
    from ptpd_calibration.curves.parser import QuadFileParser

    profile = QuadFileParser().parse(path)
    active = profile.active_channels
    channel = active[0] if active else "K"

    header_name, paper_type, chemistry = _read_qtr_header_metadata(path)
    name = header_name or path.stem

    parsed = profile.to_curve_data(channel)
    logger.debug(
        "Loaded QTR list-format curve from %s (channel %s, %d points)",
        path,
        channel,
        len(parsed.output_values),
    )

    return CurveData(
        name=name,
        input_values=parsed.input_values,
        output_values=parsed.output_values,
        paper_type=paper_type,
        chemistry=chemistry,
        notes=parsed.notes,
    )


def _clean_qtr_value(comment: str, key: str) -> str | None:
    """Value after ``key`` in a header comment; ``None`` for empty or 'Unknown'."""
    value = comment[len(key) :].strip()
    return value if value and value != _QTR_UNKNOWN else None


def _read_qtr_header_metadata(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return ``(name, paper_type, chemistry)`` from the QTR header comments.

    Only the comment block preceding the first curve value is scanned.
    """
    name: str | None = None
    paper_type: str | None = None
    chemistry: str | None = None

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            comment = line.lstrip("#").strip()
            for key in _QTR_NAME_KEYS:
                if comment.startswith(key):
                    name = _clean_qtr_value(comment, key) or name
            if comment.startswith(_QTR_PAPER_KEY):
                paper_type = _clean_qtr_value(comment, _QTR_PAPER_KEY)
            elif comment.startswith(_QTR_CHEMISTRY_KEY):
                chemistry = _clean_qtr_value(comment, _QTR_CHEMISTRY_KEY)

    return name, paper_type, chemistry
