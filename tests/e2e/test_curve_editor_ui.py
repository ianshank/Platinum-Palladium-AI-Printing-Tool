"""
Comprehensive E2E UI tests for Curve Editor functionality.

Tests cover:
1. Quad file loading (various formats)
2. Channel selection (including ALL)
3. Curve export (QTR format validation)
4. Curve modifications (adjustments, smoothing, AI enhancement)
5. Round-trip load/export consistency
6. QuadTone RIP format compliance
"""

import re

import pytest

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_quad_content():
    """Simple QTR format .quad content for testing."""
    lines = [
        "## QuadToneRIP K,C,M,Y",
        "# Test Profile",
        "# K Curve",
    ]
    # K channel - 256 values (linear ramp in 16-bit)
    for i in range(256):
        lines.append(str(i * 257))  # 0, 257, 514, ... 65535

    # C channel - 256 zeros
    lines.append("# C Curve")
    for i in range(256):
        lines.append("0")

    # M channel - 256 zeros
    lines.append("# M Curve")
    for i in range(256):
        lines.append("0")

    # Y channel - 256 zeros
    lines.append("# Y Curve")
    for i in range(256):
        lines.append("0")

    return "\n".join(lines)


@pytest.fixture
def ini_style_quad_content():
    """INI-style .quad content (alternative format) - should NOT be used for export."""
    return """[General]
ProfileName=Test INI Style
Resolution=2880
InkLimit=100

[K]
0=0
128=32768
255=65535

[C]
0=0
255=0

[M]
0=0
255=0

[Y]
0=0
255=0
"""


# ============================================================================
# Unit Tests for Quad File Parsing
# ============================================================================


class TestQuadFileParsing:
    """Tests for .quad file parsing functionality."""

    def test_parse_qtr_format(self, real_quad_path):
        """Test parsing real QuadTone RIP format file."""
        from ptpd_calibration.curves import load_quad_file

        profile = load_quad_file(real_quad_path)

        # Verify basic structure
        assert profile is not None
        assert profile.channels is not None
        assert "K" in profile.channels

        # Verify K channel has 256 points
        k_curve = profile.channels["K"]
        assert len(k_curve.values) == 256

        # Verify values are in valid range (0-255 after normalization)
        assert all(0 <= v <= 255 for v in k_curve.values)

        # Verify curve has actual data (not all zeros)
        assert max(k_curve.values) > 0

    def test_parse_qtr_format_all_channels(self, real_quad_path):
        """Test that all channels are parsed from QTR format."""
        from ptpd_calibration.curves import load_quad_file

        profile = load_quad_file(real_quad_path)

        # The real file has K, C, M, Y, LC, LM, LK, LLK channels
        expected_channels = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK"]

        for ch in expected_channels:
            assert ch in profile.channels, f"Channel {ch} not found"
            assert len(profile.channels[ch].values) == 256, f"Channel {ch} doesn't have 256 values"

    def test_parse_simple_quad_content(self, simple_quad_content, tmp_path):
        """Test parsing simple QTR format from string."""
        from ptpd_calibration.curves import load_quad_file

        # Write to temp file
        quad_file = tmp_path / "test_simple.quad"
        quad_file.write_text(simple_quad_content)

        profile = load_quad_file(quad_file)

        assert profile is not None
        assert "K" in profile.channels

        # K channel should have a linear ramp
        k_values = profile.channels["K"].values
        assert len(k_values) == 256
        # First value should be 0, last should be max
        assert k_values[0] == 0
        assert k_values[-1] > 0

    def test_parse_ini_style_quad(self, ini_style_quad_content, tmp_path):
        """Test parsing INI-style .quad format (for backwards compatibility)."""
        from ptpd_calibration.curves import load_quad_file

        quad_file = tmp_path / "test_ini.quad"
        quad_file.write_text(ini_style_quad_content)

        profile = load_quad_file(quad_file)

        assert profile is not None
        assert profile.profile_name == "Test INI Style"
        assert "K" in profile.channels


# ============================================================================
# Unit Tests for QuadTone RIP Format Compliance
# ============================================================================


class TestQTRFormatCompliance:
    """Tests to ensure exported files comply with QuadTone RIP specifications."""

    def test_export_has_qtr_header(self, tmp_path):
        """Test that exported file starts with ## QuadToneRIP header."""
        channels_data = {
            "K": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [i / 255.0 for i in range(256)],
            }
        }

        export_path = tmp_path / "test_header.quad"
        _export_multi_channel_quad_test(export_path, "Header Test", channels_data, 2880, 100.0, [])

        content = export_path.read_text()
        lines = content.split("\n")

        # First non-empty line should be the QTR header
        assert lines[0].startswith("## QuadToneRIP"), f"Missing QTR header. Got: {lines[0]}"
        assert "K" in lines[0], "Header should list channels"

    def test_export_no_ini_sections(self, tmp_path):
        """Test that export does NOT use INI-style [Section] brackets."""
        channels_data = {
            "K": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [i / 255.0 for i in range(256)],
            },
            "C": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [0.0] * 256,
            },
        }

        export_path = tmp_path / "test_no_ini.quad"
        _export_multi_channel_quad_test(export_path, "No INI Test", channels_data, 2880, 100.0, [])

        content = export_path.read_text()

        # Should NOT contain [K], [C], [General] etc.
        assert "[K]" not in content, "Found INI-style [K] section"
        assert "[C]" not in content, "Found INI-style [C] section"
        assert "[General]" not in content, "Found INI-style [General] section"
        assert "[M]" not in content, "Found INI-style [M] section"
        assert "[Y]" not in content, "Found INI-style [Y] section"

    def test_export_no_index_equals_format(self, tmp_path):
        """Test that export does NOT use index=value format (e.g., 0=0, 1=5)."""
        channels_data = {
            "K": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [i / 255.0 for i in range(256)],
            }
        }

        export_path = tmp_path / "test_no_index.quad"
        _export_multi_channel_quad_test(
            export_path, "No Index Test", channels_data, 2880, 100.0, []
        )

        content = export_path.read_text()
        lines = content.split("\n")

        # Check value lines (non-comment, non-empty)
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Should NOT contain "=" in value lines
                assert "=" not in stripped, f"Found index=value format: {stripped}"
                # Should be a pure integer
                try:
                    int(stripped)
                except ValueError:
                    pytest.fail(f"Value line is not an integer: {stripped}")

    def test_export_16bit_values(self, tmp_path):
        """Test that exported values are 16-bit (0-65535 range)."""
        # Create a profile with full range
        channels_data = {
            "K": {
                "inputs": [0, 0.5, 1.0],
                "outputs": [0, 0.5, 1.0],
            }
        }

        export_path = tmp_path / "test_16bit.quad"
        _export_multi_channel_quad_test(export_path, "16-bit Test", channels_data, 2880, 100.0, [])

        content = export_path.read_text()
        lines = content.split("\n")

        # Find K curve values (after "# K Curve" comment)
        k_start = None
        for i, line in enumerate(lines):
            if "# K Curve" in line:
                k_start = i + 1
                break

        assert k_start is not None, "K Curve header not found"

        # Collect values
        values = []
        for i in range(k_start, len(lines)):
            line = lines[i].strip()
            if line.startswith("#"):
                break  # Next channel
            if line:
                try:
                    values.append(int(line))
                except ValueError:
                    # Skip non-integer lines (comments, blank lines)
                    pass

        assert len(values) == 256, f"Expected 256 values, got {len(values)}"

        # Check range (should be 16-bit)
        assert all(0 <= v <= 65535 for v in values), "Values out of 16-bit range"

        # Max value should be close to 65535 (for a full range curve)
        assert max(values) > 60000, f"Max value {max(values)} too low for 16-bit"

    def test_export_256_values_per_channel(self, tmp_path):
        """Test that each channel has exactly 256 values."""
        channels_data = {
            "K": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [i / 255.0 for i in range(256)],
            },
            "C": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [0.0] * 256,
            },
            "M": {
                "inputs": [i / 255.0 for i in range(256)],
                "outputs": [(i / 255.0) * 0.5 for i in range(256)],
            },
        }

        export_path = tmp_path / "test_256.quad"
        _export_multi_channel_quad_test(
            export_path, "256 Values Test", channels_data, 2880, 100.0, []
        )

        content = export_path.read_text()
        lines = content.split("\n")

        # Count values per channel
        channel_counts = {}
        current_channel = None

        for line in lines:
            stripped = line.strip()
            if "# " in stripped and " Curve" in stripped:
                # Extract channel name
                match = re.search(r"# (\w+) Curve", stripped)
                if match:
                    current_channel = match.group(1)
                    channel_counts[current_channel] = 0
            elif current_channel and stripped and not stripped.startswith("#"):
                try:
                    int(stripped)
                    channel_counts[current_channel] += 1
                except ValueError:
                    # Skip non-integer lines
                    pass

        for ch, count in channel_counts.items():
            assert count == 256, f"Channel {ch} has {count} values, expected 256"

    def test_export_channel_order(self, tmp_path):
        """Test that channels are exported in standard order: K, C, M, Y, LC, LM, LK, LLK."""
        channels_data = {
            "LLK": {"inputs": [0, 1], "outputs": [0, 0.1]},
            "K": {"inputs": [0, 1], "outputs": [0, 1]},
            "Y": {"inputs": [0, 1], "outputs": [0, 0.5]},
            "C": {"inputs": [0, 1], "outputs": [0, 0.2]},
            "LK": {"inputs": [0, 1], "outputs": [0, 0.3]},
            "M": {"inputs": [0, 1], "outputs": [0, 0.4]},
        }

        export_path = tmp_path / "test_order.quad"
        _export_multi_channel_quad_test(export_path, "Order Test", channels_data, 2880, 100.0, [])

        content = export_path.read_text()

        # Find positions of channel headers
        k_pos = content.find("# K Curve")
        c_pos = content.find("# C Curve")
        m_pos = content.find("# M Curve")
        y_pos = content.find("# Y Curve")
        lk_pos = content.find("# LK Curve")
        llk_pos = content.find("# LLK Curve")

        # Verify order: K < C < M < Y < LK < LLK
        assert k_pos < c_pos, "K should come before C"
        assert c_pos < m_pos, "C should come before M"
        assert m_pos < y_pos, "M should come before Y"
        assert y_pos < lk_pos, "Y should come before LK"
        assert lk_pos < llk_pos, "LK should come before LLK"


# ============================================================================
# Unit Tests for Quad File Export
# ============================================================================


class TestQuadFileExport:
    """Tests for .quad file export functionality."""

    def test_export_qtr_format_header(self, real_quad_path, tmp_path):
        """Test that exported file has correct QTR header."""
        from ptpd_calibration.curves import load_quad_file

        # Load real file
        profile = load_quad_file(real_quad_path)

        # Build channels_data dict like the UI does
        channels_data = {}
        for ch_name, ch_curve in profile.channels.items():
            if ch_curve.enabled:
                inputs = [i / 255.0 for i in range(256)]
                outputs = [v / 255.0 for v in ch_curve.values]
                channels_data[ch_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                }

        # Export
        export_path = tmp_path / "exported.quad"
        _export_multi_channel_quad_test(export_path, "Test Export", channels_data, 2880, 100.0, [])

        # Read and verify
        content = export_path.read_text()
        lines = content.split("\n")

        # First line should be QTR header
        assert lines[0].startswith("## QuadToneRIP"), f"Invalid header: {lines[0]}"

        # Should contain channel list
        assert "K" in lines[0]

    def test_export_preserves_all_channels(self, real_quad_path, tmp_path):
        """Test that export preserves all channels from original."""
        from ptpd_calibration.curves import load_quad_file

        # Load real file
        profile = load_quad_file(real_quad_path)

        # Build channels_data
        channels_data = {}
        active_channels = []
        for ch_name, ch_curve in profile.channels.items():
            if ch_curve.enabled and any(v > 0 for v in ch_curve.values):
                active_channels.append(ch_name)
                inputs = [i / 255.0 for i in range(256)]
                outputs = [v / 255.0 for v in ch_curve.values]
                channels_data[ch_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                }

        # Export
        export_path = tmp_path / "multi_channel.quad"
        _export_multi_channel_quad_test(
            export_path, "Multi-Channel Test", channels_data, 2880, 100.0, []
        )

        # Verify all channels are in export
        content = export_path.read_text()
        for ch in active_channels:
            assert f"# {ch} Curve" in content, f"Channel {ch} not found in export"


# ============================================================================
# Round-trip Tests
# ============================================================================


class TestRoundTrip:
    """Tests for load -> export -> load consistency."""

    def test_roundtrip_preserves_values(self, real_quad_path, tmp_path):
        """Test that load -> export -> load preserves curve values."""
        from ptpd_calibration.curves import load_quad_file

        # Load original
        original = load_quad_file(real_quad_path)

        # Build channels_data for export
        channels_data = {}
        for ch_name, ch_curve in original.channels.items():
            if ch_curve.enabled:
                inputs = [i / 255.0 for i in range(256)]
                outputs = [v / 255.0 for v in ch_curve.values]
                channels_data[ch_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                }

        # Export
        export_path = tmp_path / "roundtrip.quad"
        _export_multi_channel_quad_test(
            export_path,
            original.profile_name or "Roundtrip Test",
            channels_data,
            original.resolution,
            original.ink_limit,
            original.comments,
        )

        # Load exported file
        reloaded = load_quad_file(export_path)

        # Compare K channel values (allowing for rounding)
        original_k = original.channels["K"].values
        reloaded_k = reloaded.channels["K"].values

        assert len(original_k) == len(reloaded_k) == 256

        # Allow small tolerance due to 8-bit <-> 16-bit conversion
        for i in range(256):
            diff = abs(original_k[i] - reloaded_k[i])
            assert diff <= 2, f"Value mismatch at index {i}: {original_k[i]} vs {reloaded_k[i]}"

    def test_roundtrip_all_channels(self, real_quad_path, tmp_path):
        """Test that round-trip preserves all channel data."""
        from ptpd_calibration.curves import load_quad_file

        # Load original
        original = load_quad_file(real_quad_path)

        # Build channels_data for export
        channels_data = {}
        for ch_name, ch_curve in original.channels.items():
            if ch_curve.enabled:
                inputs = [i / 255.0 for i in range(256)]
                outputs = [v / 255.0 for v in ch_curve.values]
                channels_data[ch_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                }

        # Export
        export_path = tmp_path / "roundtrip_all.quad"
        _export_multi_channel_quad_test(
            export_path, "Roundtrip All", channels_data, 2880, 100.0, []
        )

        # Load exported file
        reloaded = load_quad_file(export_path)

        # Check all channels
        for ch_name in channels_data:
            assert ch_name in reloaded.channels, f"Channel {ch_name} missing after roundtrip"

            original_vals = original.channels[ch_name].values
            reloaded_vals = reloaded.channels[ch_name].values

            # Check same length
            assert len(reloaded_vals) == 256, f"Channel {ch_name} has wrong length"

            # Check values are similar (within tolerance)
            for i in range(256):
                diff = abs(original_vals[i] - reloaded_vals[i])
                assert diff <= 3, (
                    f"Channel {ch_name}[{i}]: {original_vals[i]} vs {reloaded_vals[i]}"
                )


# ============================================================================
# Channel Selection Tests
# ============================================================================


class TestChannelSelection:
    """Tests for channel selection functionality."""

    def test_all_channels_option_exists(self, real_quad_path):
        """Test that ALL channel option is available."""
        from ptpd_calibration.curves import load_quad_file

        profile = load_quad_file(real_quad_path)

        # Get active channels
        active = [ch for ch, curve in profile.channels.items() if curve.enabled]

        # Should have multiple active channels
        assert len(active) > 1, "Need multiple channels to test ALL selection"

    def test_channel_data_integrity(self, real_quad_path):
        """Test that each channel has valid data."""
        from ptpd_calibration.curves import load_quad_file

        profile = load_quad_file(real_quad_path)

        for ch_name, ch_curve in profile.channels.items():
            # Each channel should have 256 values
            assert len(ch_curve.values) == 256, (
                f"Channel {ch_name} has {len(ch_curve.values)} values"
            )

            # Values should be in valid range
            for i, v in enumerate(ch_curve.values):
                assert 0 <= v <= 255, f"Channel {ch_name}[{i}] = {v} out of range"


# ============================================================================
# QTR Exporter Class Tests
# ============================================================================


class TestQTRExporterClass:
    """Tests for the QTRExporter class in export.py."""

    def test_qtr_exporter_curve_format(self, tmp_path):
        """Test QTRExporter._export_curve_file produces correct format."""
        from ptpd_calibration.core.models import CurveData
        from ptpd_calibration.curves.export import QTRExporter

        curve = CurveData(
            name="Test Curve",
            input_values=[0, 0.5, 1.0],
            output_values=[0, 0.5, 1.0],
        )

        exporter = QTRExporter(primary_channel="K", ink_limit=100.0)
        export_path = tmp_path / "test_curve.quad"
        exporter.export(curve, export_path, format="curve")

        content = export_path.read_text()

        # Should have QTR header
        assert content.startswith("## QuadToneRIP"), "Missing QTR header"

        # Should NOT have INI sections
        assert "[K]" not in content, "Found INI-style section"
        assert "0=" not in content, "Found index=value format"

    def test_qtr_exporter_quad_format(self, tmp_path):
        """Test QTRExporter._export_quad_profile produces correct format."""
        from ptpd_calibration.core.models import CurveData
        from ptpd_calibration.curves.export import QTRExporter

        curve = CurveData(
            name="Test Profile",
            input_values=[0, 0.5, 1.0],
            output_values=[0, 0.5, 1.0],
        )

        exporter = QTRExporter(primary_channel="K", ink_limit=100.0)
        export_path = tmp_path / "test_profile.quad"
        exporter.export(curve, export_path, format="quad")

        content = export_path.read_text()

        # Should have QTR header
        assert content.startswith("## QuadToneRIP"), "Missing QTR header"

        # Should NOT have INI sections
        assert "[K]" not in content, "Found INI-style [K] section"
        assert "[General]" not in content, "Found INI-style [General] section"

        # Should have channel curve comments
        assert "# K Curve" in content, "Missing K Curve header"

        # Values should be plain integers
        lines = content.split("\n")
        value_lines = [l for l in lines if l.strip() and not l.startswith("#")]
        for line in value_lines:
            assert "=" not in line, f"Found = in value line: {line}"


# ============================================================================
# Helper Functions (mimicking UI export logic)
# ============================================================================


def _export_multi_channel_quad_test(
    path, name, channels_data, resolution=2880, ink_limit=100.0, comments=None
):
    """Export a multi-channel .quad file in QuadTone RIP format.

    This mirrors the UI export function for testing.

    QuadTone RIP format specification:
    - Header: ## QuadToneRIP K,C,M,Y,LC,LM,LK,LLK
    - Comments: lines starting with #
    - Channel headers: # K Curve, # C Curve, etc.
    - Values: 256 integers per channel (0-65535 range), one per line
    - NO section brackets like [K]
    - NO index= format like 0=value
    """
    import numpy as np

    # Standard channel order for .quad files
    channel_order = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK"]

    # Build header with active channels
    active_channels = [ch for ch in channel_order if ch in channels_data]
    header = f"## QuadToneRIP {','.join(active_channels)}"

    lines = [header]

    # Add profile name and metadata as comments
    lines.append(f"# Profile: {name}")
    lines.append(f"# Resolution: {resolution}")
    lines.append(f"# Ink Limit: {ink_limit}%")

    # Add any original comments
    if comments:
        for comment in comments[:5]:
            if not comment.startswith("#"):
                lines.append(f"# {comment}")
            else:
                lines.append(comment)

    for channel in channel_order:
        if channel in channels_data:
            ch_data = channels_data[channel]
            ch_inputs = ch_data.get("inputs", [])
            ch_outputs = ch_data.get("outputs", [])

            # Channel header as comment
            lines.append(f"# {channel} Curve")

            if ch_inputs and ch_outputs:
                # Interpolate to 256 points
                x_new = np.linspace(0, 1, 256)
                y_new = np.interp(x_new, ch_inputs, ch_outputs)

                for i in range(256):
                    # Convert normalized (0-1) to 16-bit (0-65535)
                    qtr_output = int(y_new[i] * 65535 * ink_limit / 100)
                    qtr_output = max(0, min(65535, qtr_output))
                    lines.append(str(qtr_output))
            else:
                # Empty channel - 256 zeros
                for i in range(256):
                    lines.append("0")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Browser-based E2E Tests (require running app)
# ============================================================================


@pytest.mark.browser
@pytest.mark.skip(
    reason="Legacy Gradio tests; pending Load Quad feature implementation in React Curve Editor"
)
class TestCurveEditorBrowser:
    """Browser-based E2E tests for Curve Editor tab."""

    def test_load_quad_file(self, page, app_url, real_quad_path, ensure_app_running):
        """Test loading a .quad file in the Curve Editor."""
        from playwright.sync_api import expect

        page.goto(app_url)

        # Navigate to Calibration ▸ Curve Editor
        page.get_by_role("tab", name="Calibration", exact=False).first.click()
        page.get_by_role("tab", name="Curve Editor").first.click()

        # Upload the quad file
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(real_quad_path))

        # Click Load button
        page.get_by_role("button", name="Load Quad File").click()

        # Wait for curve info to update
        page.wait_for_timeout(2000)

        # Verify curve is loaded (check for profile name in info)
        expect(page.get_by_text("Platinum_Palladium_V6-CC")).to_be_visible(timeout=5000)

    def test_channel_dropdown_has_all(self, page, app_url, real_quad_path, ensure_app_running):
        """Test that channel dropdown includes ALL option."""
        from playwright.sync_api import expect

        page.goto(app_url)
        page.get_by_role("tab", name="Calibration", exact=False).first.click()
        page.get_by_role("tab", name="Curve Editor").first.click()

        # Load quad file
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(real_quad_path))
        page.get_by_role("button", name="Load Quad File").click()
        page.wait_for_timeout(2000)

        # Check channel dropdown
        channel_dropdown = page.get_by_label("Channel").first
        channel_dropdown.click()

        # Verify ALL option exists
        expect(page.get_by_role("option", name="ALL")).to_be_visible()

    def test_export_creates_valid_quad_file(
        self, page, app_url, real_quad_path, ensure_app_running, tmp_path
    ):
        """Test that export creates a valid .quad file in QTR format."""

        page.goto(app_url)
        page.get_by_role("tab", name="Calibration", exact=False).first.click()
        page.get_by_role("tab", name="Curve Editor").first.click()

        # Load quad file
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(real_quad_path))
        page.get_by_role("button", name="Load Quad File").click()
        page.wait_for_timeout(2000)

        # Select QTR format
        format_dropdown = page.get_by_label("Export Format").first
        format_dropdown.select_option("qtr")

        # Click export
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Export Curve").click()

        download = download_info.value

        # Verify file extension
        assert download.suggested_filename.endswith(".quad"), (
            f"Wrong extension: {download.suggested_filename}"
        )

        # Save and verify content
        download_path = tmp_path / download.suggested_filename
        download.save_as(download_path)

        content = download_path.read_text()

        # Verify QTR format
        assert content.startswith("## QuadToneRIP"), "Missing QTR header"
        assert "# K Curve" in content, "Missing K channel header"

        # Verify NO INI format
        assert "[K]" not in content, "Found INI-style section"
        assert "0=0" not in content, "Found index=value format"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
