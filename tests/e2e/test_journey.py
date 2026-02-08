"""
Browser-based E2E test using Playwright.

This test requires Playwright to be installed. If not installed, it will be skipped.
"""

import numpy as np
import pytest
from PIL import Image

# Skip this entire module if playwright is not installed
pytest.importorskip("playwright", reason="Playwright not installed - skipping browser tests")

from playwright.sync_api import Page, expect


def test_full_calibration_flow(page: Page, app_url, tmp_path):
    """
    Test the full user journey:
    1. Navigate to Calibration Wizard
    2. Complete Setup and Print steps
    3. Upload step tablet scan
    4. Analyze and Generate curve
    5. Verify completion
    """
    # 1. Navigate to app
    page.goto(f"{app_url}/calibration")

    # Wait for title
    expect(page.get_by_role("heading", name="Calibration Wizard")).to_be_visible()

    # Prepare synthetic tablet image
    tablet_path = tmp_path / "tablet.png"
    # Create simple gray synthetic image
    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (32, 1))
    rgb = np.stack([gradient, gradient, gradient], axis=-1)
    Image.fromarray(rgb).save(tablet_path)

    # --- Step 1: Setup ---
    # Fill Paper Type and Exposure Time
    page.get_by_placeholder("e.g. Arches Platine").fill("Test Paper E2E")
    # Exposure time is input type number, maybe find by label
    page.locator("input[type='number']").fill("120")

    page.get_by_role("button", name="Next").click()

    # --- Step 2: Print ---
    expect(page.get_by_text("Print Target")).to_be_visible()
    # Click "I have printed the target"
    page.get_by_role("button", name="I have printed the target").click()

    # --- Step 3: Scan ---
    expect(page.get_by_text("Scan Target")).to_be_visible()

    # Upload tablet scan
    # Use the test-id we added
    file_input = page.get_by_test_id("scan-upload-input")
    file_input.set_input_files(str(tablet_path))

    # Click Upload button (it appears after file selection)
    page.get_by_role("button", name="Upload Scan").click()

    # --- Step 4: Analyze ---
    # Wait for transition to Analysis step (which happens after successful upload mock/real)
    # The ScanUpload component calls callback, Wizard moves to next step 'Analyze'
    expect(page.get_by_text("Analysis")).to_be_visible(timeout=10000)

    # Click Generate Curve
    page.get_by_role("button", name="Generate Curve").click()

    # --- Step 5: Finish ---
    # Should see "Complete" and "Curve Generated!" or similar
    expect(page.get_by_text("Curve Generated!", exact=False)).to_be_visible(timeout=10000)

    page.get_by_role("button", name="Next").click()

    expect(page.get_by_text("Complete")).to_be_visible()

    # Verify Editor is present
    expect(page.get_by_text("Curve Name")).to_be_visible()
