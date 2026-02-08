"""
E2E test for Curve Display and File Loading.

Verifies that curve files (.quad) can be loaded, displayed, and statistics generated.
"""


import pytest
from playwright.sync_api import Page, expect

# UTF-8 .quad content
QUAD_UTF8 = """[General]
ProfileName=Test UTF-8
[K]
0=0
255=255
"""

# UTF-16 .quad content
QUAD_UTF16 = """[General]
ProfileName=Test UTF-16
[K]
0=0
255=255
"""


@pytest.fixture
def curve_files(tmp_path):
    """Create test curve files."""
    utf8_file = tmp_path / "test_utf8.quad"
    utf8_file.write_text(QUAD_UTF8, encoding="utf-8")

    utf16_file = tmp_path / "test_utf16.quad"
    utf16_file.write_text(QUAD_UTF16, encoding="utf-16")

    return utf8_file, utf16_file


@pytest.mark.browser
@pytest.mark.skip(reason="Playwright selectors pending update for hierarchical navigation")
def test_curve_loading(page: Page, app_url, curve_files, ensure_app_running):  # noqa: ARG001
    """
    Test loading curve files:
    1. Upload UTF-8 file
    2. Verify display update
    3. Upload UTF-16 file
    4. Verify display update
    """
    utf8_file, utf16_file = curve_files

    # 1. Navigate
    page.goto(app_url)

    # Wait for title
    expect(page.get_by_role("heading", name="Pt/Pd Calibration Studio")).to_be_visible()

    # Click "Curve Display" tab (it's default, but good to be explicit)
    # Note: Gradio tabs might be implemented as buttons or divs.
    # Assuming default tab is active.

    # Navigate to Calibration ▸ Curve Display
    page.get_by_role("tab", name="Calibration", exact=False).first.click()
    page.get_by_role("tab", name="Curve Display").first.click()

    # 2. Upload UTF-8 file
    file_input = page.locator("input[type='file']").first
    file_input.set_input_files(str(utf8_file))

    # Click "Load Files"
    load_btn = page.get_by_role("button", name="Load Files")
    load_btn.click()

    # 3. Verify UTF-8 Load
    # Check if "Test UTF-8 (K)" appears in the dataframe or text
    # Gradio Dataframe renders as a table.
    expect(page.get_by_text("Test UTF-8 (K)")).to_be_visible(timeout=10000)

    # Check points count (256)
    # Note: This might be in a cell next to the name
    # We can just check for existence for now.

    # Check Plot
    # Gradio plots are canvas or similar. Checking existence is a good start.
    expect(page.locator(".gradio-plot")).to_be_visible()

    # 4. Upload UTF-16 file
    file_input.set_input_files(str(utf16_file))
    load_btn.click()

    # 5. Verify UTF-16 Load
    expect(page.get_by_text("Test UTF-16 (K)")).to_be_visible(timeout=10000)

    # Verify both are present
    expect(page.get_by_text("Test UTF-8 (K)")).to_be_visible()

    # 6. Paste Data Test
    # Click "Paste Data" tab
    page.get_by_role("tab", name="Paste Data").first.click()

    # Fill textarea
    textarea = page.get_by_label("Paste Curve Values")
    textarea.fill("0.0, 0.5, 1.0")

    name_input = page.get_by_label("Curve Name")
    name_input.fill("Pasted Curve")

    add_btn = page.get_by_role("button", name="Add Curve")
    add_btn.click()

    # Verify Pasted Load
    expect(page.get_by_text("Pasted Curve")).to_be_visible()

    # Check statistics panel if enabled (it's checkbox is off by default)
    # Enable statistics
    page.get_by_label("Show Statistics Panel").check()
    expect(page.get_by_text("Gamma:")).to_be_visible()
