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
def real_quad_path(tmp_path_factory):
    """Path to a real-world style multi-channel .quad profile.

    ``*.quad`` files are git-ignored, so the original vendor profile is never
    present in a clean checkout. When it is absent a deterministic synthetic
    QuadToneRIP profile with the same shape (8 channels x 256 16-bit values,
    K as a full ramp, LK/LLK partial, the rest empty) is generated instead, so
    the tests exercise the real parser and exporter on a realistic file.
    """
    real = Path(__file__).parent.parent / "fixtures" / "Platinum_Palladium_V6-CC.quad"
    if real.exists():
        return real
    channels = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK"]
    max_value = 65535
    lines = [
        f"## QuadToneRIP {','.join(channels)}",
        "# Platinum-Palladium V6 CC (synthetic profile generated for tests)",
    ]
    for name in channels:
        lines.append(f"# {name} Curve")
        if name == "K":
            values = [round(i / 255 * max_value) for i in range(256)]
        elif name == "LK":
            values = [round(min(i / 255, 0.5) * max_value) for i in range(256)]
        elif name == "LLK":
            values = [round(min(i / 255, 0.25) * max_value) for i in range(256)]
        else:
            values = [0] * 256
        lines.extend(str(v) for v in values)
    path = tmp_path_factory.mktemp("quad") / "synthetic_platinum_palladium.quad"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(autouse=True)
def _legacy_ui_requires_gradio(request: pytest.FixtureRequest) -> None:
    """Skip tests marked ``legacy_ui`` when the retired Gradio UI is not installed.

    The marker documents that a journey exercises Gradio handlers (ADR-0004);
    gating here keeps the marker reusable instead of repeating importorskip.
    """
    if request.node.get_closest_marker("legacy_ui") is not None:
        pytest.importorskip(
            "gradio", reason="legacy Gradio UI journey needs the [ui] extra (ADR-0004)"
        )
