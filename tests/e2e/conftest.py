"""E2E test fixtures.

This conftest supports both functional user journey tests and Playwright browser tests.
Playwright is optional - if not installed, browser tests will be skipped.
"""

import os
import subprocess
import time

import pytest

# Try to import playwright, but don't fail if not available
try:
    from playwright.sync_api import Page, expect

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None
    expect = None


@pytest.fixture(scope="session")
def app_url():
    """Return the URL of the running app."""
    return "http://127.0.0.1:7861"


@pytest.fixture(scope="session")
def ensure_app_running(app_url):
    """Ensure the app is running before tests start.

    This fixture is for browser-based tests that need a running server.
    Functional tests don't need this.
    """
    import urllib.request

    # Check if app is already running
    try:
        urllib.request.urlopen(app_url, timeout=5)
        print("App is already running.")
        yield
        return
    except Exception:
        print("App not running. Starting it...")

    # Start the app
    process = subprocess.Popen(
        ["python", "-m", "ptpd_calibration.ui.gradio_app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GRADIO_SERVER_NAME": "127.0.0.1",
            "GRADIO_SERVER_PORT": "7861",
        },
    )

    # Wait for app to start
    max_retries = 30
    for i in range(max_retries):
        try:
            urllib.request.urlopen(app_url, timeout=5)
            print("App started successfully.")
            break
        except Exception:
            time.sleep(1)
            if i == max_retries - 1:
                process.terminate()
                raise RuntimeError("Failed to start app.") from None

    yield

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


def pytest_configure(config):
    """Add markers for test categorization."""
    config.addinivalue_line(
        "markers", "browser: mark test as requiring a browser (requires playwright)"
    )
    config.addinivalue_line(
        "markers", "functional: mark test as a functional test (no browser required)"
    )


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip browser tests if playwright is not available."""
    if not PLAYWRIGHT_AVAILABLE:
        skip_browser = pytest.mark.skip(reason="Playwright not installed")
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(skip_browser)
