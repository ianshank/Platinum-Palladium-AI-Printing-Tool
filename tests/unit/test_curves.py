"""
Tests for curve generation and export.
"""

import json

import numpy as np
import pytest

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.analysis import CurveAnalyzer
from ptpd_calibration.curves.export import (
    CSVExporter,
    JSONExporter,
    PiezographyExporter,
    QTRExporter,
    load_curve,
    safe_comment_value,
    save_curve,
)
from ptpd_calibration.curves.generator import (
    CurveGenerator,
    TargetCurve,
    generate_linearization_curve,
)


class TestTargetCurve:
    """Tests for TargetCurve."""

    def test_linear_target(self):
        """Test linear target curve generation."""
        target = TargetCurve.linear(21)

        assert len(target.input_values) == 21
        assert len(target.output_values) == 21
        assert target.input_values[0] == pytest.approx(0.0)
        assert target.input_values[-1] == pytest.approx(1.0)
        assert target.output_values[0] == pytest.approx(0.0)
        assert target.output_values[-1] == pytest.approx(1.0)

    def test_paper_white_preserve_target(self):
        """Test paper white preserving target curve."""
        target = TargetCurve.paper_white_preserve(21, highlight_hold=0.1)

        assert len(target.input_values) == 21
        # First few values should be held at 0
        assert target.output_values[0] == pytest.approx(0.0)
        assert target.output_values[1] == pytest.approx(0.0)

    def test_aesthetic_target(self):
        """Test aesthetic S-curve target."""
        target = TargetCurve.aesthetic(21, shadow_boost=0.1)

        assert len(target.input_values) == 21
        # Should have S-curve characteristics
        mid_idx = 5
        # Midpoint should be slightly different from linear
        assert target.output_values[mid_idx] != pytest.approx(
            target.input_values[mid_idx], abs=0.01
        )


class TestCurveGenerator:
    """Tests for CurveGenerator."""

    @pytest.fixture
    def sample_densities(self):
        """Generate sample density measurements."""
        # Simulate typical Pt/Pd response (toe, linear, shoulder)
        steps = np.linspace(0, 1, 21)
        densities = 0.1 + 2.0 * (steps**0.8)  # Non-linear response
        return list(densities)

    def test_generate_linear_curve(self, sample_densities):
        """Test linear curve generation."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            curve_type=CurveType.LINEAR,
            name="Test Linear",
        )

        assert curve.name == "Test Linear"
        assert curve.curve_type == CurveType.LINEAR
        assert len(curve.input_values) > 0
        assert len(curve.output_values) > 0
        assert curve.input_values[0] == pytest.approx(0.0)
        assert curve.input_values[-1] == pytest.approx(1.0)

    def test_generate_paper_white_curve(self, sample_densities):
        """Test paper white preserving curve."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            curve_type=CurveType.PAPER_WHITE,
            name="Test Paper White",
        )

        assert curve.curve_type == CurveType.PAPER_WHITE
        # Output should be 0 at input 0
        assert curve.output_values[0] == pytest.approx(0.0, abs=0.01)

    def test_curve_monotonicity(self, sample_densities):
        """Test that generated curves are monotonic."""
        generator = CurveGenerator()
        curve = generator.generate(sample_densities)

        output = np.array(curve.output_values)
        diffs = np.diff(output)

        # All differences should be >= 0 (monotonically increasing)
        assert np.all(diffs >= -0.001)

    def test_generate_with_metadata(self, sample_densities):
        """Test curve generation with metadata."""
        generator = CurveGenerator()
        curve = generator.generate(
            sample_densities,
            name="Calibration",
            paper_type="Arches Platine",
            chemistry="50% Pt, 5 drops Na2",
        )

        assert curve.paper_type == "Arches Platine"
        assert curve.chemistry == "50% Pt, 5 drops Na2"

    def test_insufficient_measurements(self):
        """Test error with too few measurements."""
        generator = CurveGenerator()

        with pytest.raises(ValueError):
            generator.generate([0.1])  # Only 1 measurement

    def test_flat_measurements(self):
        """Test error with flat measurements (no range)."""
        generator = CurveGenerator()

        with pytest.raises(ValueError):
            generator.generate([1.0, 1.0, 1.0, 1.0, 1.0])


class TestCurveExporters:
    """Tests for curve exporters."""

    @pytest.fixture
    def sample_curve(self):
        """Create a sample curve for testing."""
        return CurveData(
            name="Test Export Curve",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.linspace(0, 1, 256) ** 0.9),
            paper_type="Test Paper",
            curve_type=CurveType.LINEAR,
        )

    def test_qtr_export(self, sample_curve, tmp_path):
        """Test QTR curve export."""
        exporter = QTRExporter()
        output_path = tmp_path / "test_curve.txt"

        exporter.export(sample_curve, output_path, format="curve")

        assert output_path.exists()
        content = output_path.read_text()
        # New QTR format uses QuadToneRIP header and raw values
        assert "## QuadToneRIP" in content
        assert "# K Curve" in content
        # Should have numeric values (256 points)
        lines = [line for line in content.split("\n") if line and not line.startswith("#")]
        assert len(lines) >= 256

    def test_qtr_quad_export(self, sample_curve, tmp_path):
        """Test QTR .quad profile export."""
        exporter = QTRExporter()
        output_path = tmp_path / "test_profile.quad"

        exporter.export(sample_curve, output_path, format="quad")

        assert output_path.exists()
        content = output_path.read_text()
        # New QTR format uses QuadToneRIP header with all channels
        assert "## QuadToneRIP" in content
        assert "K,C,M,Y" in content
        assert "# K Curve" in content

    def test_piezography_export(self, sample_curve, tmp_path):
        """Test Piezography export."""
        exporter = PiezographyExporter()
        output_path = tmp_path / "test_curve.ppt"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "[Linearization]" in content

    def test_csv_export(self, sample_curve, tmp_path):
        """Test CSV export."""
        exporter = CSVExporter()
        output_path = tmp_path / "test_curve.csv"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "input,output" in content
        lines = content.strip().split("\n")
        assert len(lines) > 1  # Header + data

    def test_json_export(self, sample_curve, tmp_path):
        """Test JSON export."""
        exporter = JSONExporter()
        output_path = tmp_path / "test_curve.json"

        exporter.export(sample_curve, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["name"] == "Test Export Curve"
        assert "input_values" in data
        assert "output_values" in data

    def test_save_and_load_json(self, sample_curve, tmp_path):
        """Test round-trip JSON save/load."""
        output_path = tmp_path / "roundtrip.json"

        save_curve(sample_curve, output_path, format="json")
        loaded = load_curve(output_path)

        assert loaded.name == sample_curve.name
        assert len(loaded.input_values) == len(sample_curve.input_values)
        assert loaded.input_values[0] == pytest.approx(sample_curve.input_values[0])

    def test_save_and_load_csv(self, sample_curve, tmp_path):
        """Test round-trip CSV save/load."""
        output_path = tmp_path / "roundtrip.csv"

        save_curve(sample_curve, output_path, format="csv")
        loaded = load_curve(output_path)

        assert len(loaded.input_values) == len(sample_curve.input_values)


class TestCurveAnalyzer:
    """Tests for CurveAnalyzer."""

    def test_analyze_linearity_perfect(self):
        """Test linearity analysis with perfect linear response."""
        densities = list(np.linspace(0.1, 2.1, 21))

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert analysis.is_monotonic
        assert analysis.max_error < 0.1
        assert analysis.rms_error < 0.05
        assert len(analysis.problem_regions) == 0

    def test_analyze_linearity_nonlinear(self):
        """Test linearity analysis with non-linear response."""
        # Create non-linear response (typical film/paper curve)
        steps = np.linspace(0, 1, 21)
        densities = list(0.1 + 2.0 * (steps**1.5))

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert analysis.is_monotonic
        assert analysis.max_error > 0.05  # Should have deviation

    def test_analyze_non_monotonic(self):
        """Test detection of non-monotonic response."""
        # Create response with reversal (solarization)
        densities = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0, 1.9, 1.85, 1.8, 1.75]

        analysis = CurveAnalyzer.analyze_linearity(densities)

        assert not analysis.is_monotonic

    def test_suggest_adjustments_low_range(self):
        """Test suggestions for low density range."""
        # Low density range
        densities = list(np.linspace(0.1, 1.0, 21))

        suggestions = CurveAnalyzer.suggest_adjustments(densities)

        assert len(suggestions) > 0
        assert any("range" in s.lower() for s in suggestions)

    def test_suggest_adjustments_high_dmin(self):
        """Test suggestions for high Dmin."""
        # High Dmin (fog or paper issue)
        densities = list(np.linspace(0.25, 2.25, 21))

        suggestions = CurveAnalyzer.suggest_adjustments(densities)

        assert any("dmin" in s.lower() or "clear" in s.lower() for s in suggestions)

    def test_analyze_curve(self):
        """Test comprehensive curve analysis."""
        curve = CurveData(
            name="Test",
            input_values=list(np.linspace(0, 1, 256)),
            output_values=list(np.linspace(0, 1, 256) ** 0.9),
        )

        analysis = CurveAnalyzer.analyze_curve(curve)

        assert "name" in analysis
        assert "num_points" in analysis
        assert "is_monotonic" in analysis
        assert analysis["is_monotonic"]
        assert analysis["shape"] in [
            "approximately linear",
            "convex (lifts midtones)",
            "concave (darkens midtones)",
        ]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_linearization_curve(self):
        """Test convenience function for linearization."""
        densities = list(np.linspace(0.1, 2.1, 21))

        curve = generate_linearization_curve(
            densities,
            name="Quick Cal",
            paper_type="Test Paper",
        )

        assert curve.name == "Quick Cal"
        assert curve.paper_type == "Test Paper"
        assert curve.curve_type == CurveType.LINEAR


class TestQTRRoundTrip:
    """Regression tests for probe finding F3: ``save_curve``/``load_curve`` for QTR files.

    ``QTRExporter`` writes 16-bit values while ``QuadFileParser`` stores 8-bit
    values, so the round trip is exact only to within 2/255 (one step of 16->8
    bit truncation plus ``int()`` truncation on export).
    """

    QUANTIZATION_TOLERANCE = 2 / 255
    QTR_POINTS = 256

    @pytest.fixture
    def qtr_curve(self):
        """Non-linear, monotone curve with metadata."""
        x = np.linspace(0, 1, 33)
        return CurveData(
            name="Round Trip Curve",
            input_values=list(x),
            output_values=list(x**0.9),
            paper_type="Arches Platine",
            chemistry="Pd/Pt 50:50",
            curve_type=CurveType.LINEAR,
        )

    @pytest.mark.parametrize("suffix", [".quad", ".txt"])
    def test_save_and_load_qtr_roundtrip(self, qtr_curve, tmp_path, suffix):
        """Both the full .quad profile and the single-channel curve file load back."""
        path = tmp_path / f"roundtrip{suffix}"
        save_curve(qtr_curve, path)

        loaded = load_curve(path)

        expected = np.interp(
            np.linspace(0, 1, self.QTR_POINTS), qtr_curve.input_values, qtr_curve.output_values
        )
        assert len(loaded.output_values) == self.QTR_POINTS
        assert np.max(np.abs(np.array(loaded.output_values) - expected)) <= (
            self.QUANTIZATION_TOLERANCE
        )
        assert np.all(np.diff(loaded.output_values) >= 0)
        assert loaded.name == qtr_curve.name
        assert loaded.paper_type == qtr_curve.paper_type
        assert loaded.chemistry == qtr_curve.chemistry

    def test_load_uses_channel_with_data(self, qtr_curve, tmp_path):
        """A profile exported on a non-K channel loads that channel, not the empty K."""
        path = tmp_path / "cyan.quad"
        QTRExporter(primary_channel="C").export(qtr_curve, path, format="quad")

        loaded = load_curve(path)

        assert max(loaded.output_values) > 0.5

    def test_missing_metadata_falls_back_to_stem(self, qtr_curve, tmp_path):
        """'Unknown' paper/chemistry map to None; a missing name uses the file stem."""
        path = tmp_path / "bare.quad"
        bare = CurveData(
            name="Bare", input_values=qtr_curve.input_values, output_values=qtr_curve.output_values
        )
        save_curve(bare, path)
        content = path.read_text().replace("# Profile: Bare\n", "")
        path.write_text(content)

        loaded = load_curve(path)

        assert loaded.name == "bare"
        assert loaded.paper_type is None
        assert loaded.chemistry is None

    def test_key_value_text_format_still_supported(self, tmp_path):
        """The legacy ``index=value`` layout keeps working."""
        path = tmp_path / "legacy.txt"
        path.write_text("ProfileName=Legacy\n0=0\n128=64\n255=255\n")

        loaded = load_curve(path)

        assert loaded.name == "Legacy"
        assert loaded.input_values == pytest.approx([0.0, 128 / 255, 1.0])
        assert loaded.output_values == pytest.approx([0.0, 64 / 255, 1.0])

    def test_text_without_curve_data_raises(self, tmp_path):
        """Files with neither layout still raise the documented ValueError."""
        path = tmp_path / "empty.txt"
        path.write_text("# just a comment\n")

        with pytest.raises(ValueError, match="No curve data found"):
            load_curve(path)


class TestExportCommentInjection:
    """A curve's metadata must not be able to forge lines in a text export.

    QTR and Piezography files are line-oriented, so a newline inside a name,
    paper type or chemistry used to write what looked like a further header
    comment -- or a bare curve value. One client could therefore decide what a
    different client saw after downloading and re-importing the same curve.
    """

    #: Break characters a line-oriented reader may honour, beyond the obvious two.
    LINE_BREAKS = ("\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", "\u2028", "\u2029")

    #: Lines to inspect when checking that no forged value reached the header.
    HEADER_WINDOW = 12

    @staticmethod
    def _curve(**metadata: str) -> CurveData:
        x = np.linspace(0, 1, 16)
        return CurveData(
            name=metadata.pop("name", "Honest Curve"),
            input_values=list(x),
            output_values=list(x),
            **metadata,
        )

    @staticmethod
    def _header_comments(path) -> list[str]:
        """Comment lines before the first value line, as the loader reads them."""
        comments = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            comments.append(line)
        return comments

    @pytest.mark.parametrize("field", ["name", "paper_type", "chemistry"])
    @pytest.mark.parametrize("export_format", ["quad", "curve"])
    def test_newline_in_metadata_forges_no_line(self, field, export_format, tmp_path):
        """Every metadata field stays on the single line the exporter allotted it."""
        payload = "Real\n# Profile: Spoofed\n65535"
        path = tmp_path / f"injected.{export_format}"
        QTRExporter().export(self._curve(**{field: payload}), path, format=export_format)

        head = path.read_text().splitlines()[: self.HEADER_WINDOW]
        comments = self._header_comments(path)

        assert all(line.startswith("#") for line in comments)
        assert not any(line.strip() == "65535" for line in head)
        assert sum(1 for line in comments if "Spoofed" in line) <= 1

    @pytest.mark.parametrize("break_char", LINE_BREAKS)
    def test_every_break_character_is_neutralised(self, break_char, tmp_path):
        """Not just LF and CR: a reader honouring U+2028 must not see two lines."""
        path = tmp_path / "break.quad"
        QTRExporter().export(
            self._curve(name=f"Real{break_char}# Paper: Forged"), path, format="quad"
        )

        head = path.read_text().splitlines()[: self.HEADER_WINDOW]

        assert not any(char in path.read_text() for char in self.LINE_BREAKS if char != "\n")
        assert sum(1 for line in head if line.startswith("# Paper:")) == 1

    def test_paper_type_cannot_rename_the_curve(self, tmp_path):
        """The reported regression: paper type dictated the re-imported name."""
        path = tmp_path / "poisoned.quad"
        curve = self._curve(name="Real Name", paper_type="Arches\n# Profile: Spoofed")
        save_curve(curve, path)

        loaded = load_curve(path)

        assert loaded.name == "Real Name"
        assert loaded.paper_type is not None
        assert "Spoofed" in loaded.paper_type  # inert inside the value, not a key

    def test_newline_in_name_no_longer_truncates_the_header(self, tmp_path):
        """A break in the name used to end the header and drop everything after it."""
        path = tmp_path / "truncated.quad"
        curve = self._curve(name="Innocent\n65535", paper_type="Arches", chemistry="Na2")
        save_curve(curve, path)

        loaded = load_curve(path)

        assert loaded.paper_type == "Arches"
        assert loaded.chemistry == "Na2"

    def test_piezography_header_is_sanitised_too(self, tmp_path):
        """The .ppt exporter writes the same comment shape and needs the same guard."""
        path = tmp_path / "pz.ppt"
        PiezographyExporter().export(self._curve(name="A\n# Paper: Forged"), path)

        header = path.read_text().split("[Linearization]")[0].splitlines()

        assert all(not line or line.startswith("#") for line in header)
        assert sum(1 for line in header if line.startswith("# Paper:")) == 1

    def test_externally_written_file_is_normalised_on_read(self, tmp_path):
        """A file this exporter did not write is untrusted too."""
        path = tmp_path / "foreign.quad"
        path.write_text(
            "## QuadToneRIP K\n"
            "# Profile: Foreign\x07\u202ename\n"
            "# Paper:\tTab\x00Separated\n"
            "# K Curve\n" + "\n".join(["0"] * 255) + "\n65535\n"
        )

        loaded = load_curve(path)

        assert loaded.name == "Foreign name"
        assert loaded.paper_type == "Tab Separated"


class TestSafeCommentValue:
    """Unit behaviour of the shared comment sanitiser."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Arches Platine", "Arches Platine"),
            ("  padded  ", "padded"),
            ("tab\tseparated", "tab separated"),
            ("collapse    runs", "collapse runs"),
            ("break\nhere", "break here"),
            ("bell\x07gone", "bell gone"),
            ("nb\u00a0space", "nb space"),
            ("hash # stays", "hash # stays"),
            ("key: looks like one", "key: looks like one"),
        ],
    )
    def test_single_line_results(self, raw, expected):
        assert safe_comment_value(raw) == expected

    @pytest.mark.parametrize("value", [None, "", "   ", "\n\r\t", "\u200b"])
    def test_empty_results_fall_back_to_the_default(self, value):
        assert safe_comment_value(value) == "Unknown"
        assert safe_comment_value(value, default="Anon") == "Anon"

    def test_truncation_uses_the_supplied_limit(self):
        assert safe_comment_value("a" * 100, max_length=16) == "a" * 16

    def test_truncation_limit_comes_from_settings(self, monkeypatch):
        """The bound is a settings field, not a literal in the exporter."""
        from ptpd_calibration import config

        monkeypatch.setenv("PTPD_CURVE_MAX_EXPORT_COMMENT_LENGTH", "12")
        monkeypatch.setattr(config, "_settings", None)

        assert safe_comment_value("b" * 40) == "b" * 12

    def test_non_string_values_are_rendered(self):
        assert safe_comment_value(42) == "42"
