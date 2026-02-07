"""E2E test fixtures.

This conftest supports both functional user journey tests and Playwright browser tests.
Playwright is optional - if not installed, browser tests will be skipped.
"""


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
    return "http://localhost:3000"


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
        print("App is running.")
        yield
        return
    except Exception:
        pytest.fail(
            f"App not running at {app_url}. Please start the frontend (npm run dev) "
            "and backend (python -m ptpd_calibration.api.server) before running E2E tests."
        )


def pytest_configure(config):
    """Add markers for test categorization."""
    config.addinivalue_line(
        "markers", "browser: mark test as requiring a browser (requires playwright)"
    )
    config.addinivalue_line(
        "markers", "functional: mark test as a functional test (no browser required)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip browser tests if playwright is not available."""
    if not PLAYWRIGHT_AVAILABLE:
        skip_browser = pytest.mark.skip(reason="Playwright not installed")
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(skip_browser)

from pathlib import Path

@pytest.fixture
def real_quad_path():
    """Path to the real-world .quad fixture file."""
    return Path(__file__).parent.parent / "fixtures" / "Platinum_Palladium_V6-CC.quad"
