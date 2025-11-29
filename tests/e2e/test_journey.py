"""
Browser-based E2E test using Playwright.

This test requires Playwright to be installed. If not installed, it will be skipped.
"""

import pytest
from pathlib import Path

# Skip this entire module if playwright is not installed
pytest.importorskip("playwright", reason="Playwright not installed - skipping browser tests")

from playwright.sync_api import Page, expect

def test_full_calibration_flow(page: Page, app_url):
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

    # 2. Upload step tablet
    # Final attempt: target by data-testid
    try:
        # There might be multiple, we want the first one (usually input for the first tab)
        file_input = page.locator('[data-testid="file-upload"]').first
        file_input.set_input_files(dummy_image)
        print("Set files on data-testid='file-upload'")
        
        # Dispatch events
        file_input.evaluate("e => e.dispatchEvent(new Event('change', { bubbles: true }))")
        file_input.evaluate("e => e.dispatchEvent(new Event('blur', { bubbles: true }))")
    except Exception as e:
        print(f"Failed to set files on data-testid: {e}")

    # Wait for upload to process
    # The filename might not be visible, but the image should appear
    # Or we can just wait a bit
    page.wait_for_timeout(5000)
    
    # 3. Analyze
    # Click "Analyze" button
    analyze_btn = page.get_by_role("button", name="Analyze")
    expect(analyze_btn).to_be_enabled()
    analyze_btn.click()

    # Wait for results
    # Look for "Analysis Results" or specific output
    try:
        expect(page.get_by_text("Linearity Analysis")).to_be_visible(timeout=30000)
    except AssertionError:
        # Check for error messages
        if page.get_by_text("Error").is_visible():
            print("UI Error found")
        raise
    
    # Check if plot is visible
    expect(page.locator(".plot-container").first).to_be_visible()

    # 4. Generate Curve
    # Switch to "Generate Curve" tab
    page.get_by_role("tab", name="Generate Curve").click()
    
    # Select target (default should be Standard)
    # Click "Generate Curve" button
    page.get_by_role("button", name="Generate Curve").click()

    # 5. Verify Download
    # Check if download component appears
    expect(page.get_by_text("Download .quad file")).to_be_visible()
    
    # Optional: Verify download works
    with page.expect_download() as download_info:
        page.get_by_role("link", name="Download").click()
    
    download = download_info.value
    assert download.suggested_filename.endswith(".quad")
