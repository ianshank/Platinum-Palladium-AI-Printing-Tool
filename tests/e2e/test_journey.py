"""
Browser-based E2E test using Playwright.

This test requires Playwright to be installed. If not installed, it will be skipped.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

# Skip this entire module if playwright is not installed
pytest.importorskip("playwright", reason="Playwright not installed - skipping browser tests")

from playwright.sync_api import Page, expect

@pytest.mark.skip(reason="Playwright selectors pending update for hierarchical navigation")
def test_full_calibration_flow(page: Page, app_url, tmp_path):
    """
    Test the full user journey:
    1. Upload step tablet scan
    2. Analyze scan
    3. Generate correction curve
    4. Verify download
    """
    # 1. Navigate to app
    page.goto(app_url)
    
    # Wait for title
    expect(page.get_by_role("heading", name="Pt/Pd Calibration Studio")).to_be_visible()

    # Prepare synthetic tablet image
    tablet_path = tmp_path / "tablet.png"
    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (32, 1))
    rgb = np.stack([gradient, gradient, gradient], axis=-1)
    Image.fromarray(rgb).save(tablet_path)

    # Switch to Calibration ▸ Step Wedge Analysis
    page.get_by_role("tab", name="Calibration", exact=False).first.click()
    page.get_by_role("tab", name="Step Wedge Analysis").first.click()

    # Upload tablet scan
    file_input = page.locator("input[type='file']").first
    file_input.set_input_files(str(tablet_path))
    page.wait_for_timeout(1000)

    # Analyze scan
    analyze_btn = page.get_by_role("button", name="Analyze Step Wedge")
    expect(analyze_btn).to_be_enabled()
    analyze_btn.click()

    # Wait for results
    # Look for "Analysis Results" or specific output
    # Wait for metrics to render
    expect(page.get_by_label("Quality Grade")).to_be_visible(timeout=15000)

    # Generate and export curve
    page.get_by_role("button", name="Generate Curve").click()
    expect(page.get_by_text("Calibration Curve")).to_be_visible(timeout=15000)

    with page.expect_download() as download_info:
        page.get_by_role("button", name="Export Curve").click()

    download = download_info.value
    assert download.suggested_filename.endswith(".quad")
