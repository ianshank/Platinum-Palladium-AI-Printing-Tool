"""
Selenium E2E test fixtures and configuration.

This module provides WebDriver setup, page objects base configuration,
screenshot capture on failure, and common test utilities.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

# Try to import selenium, skip tests if not available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    ChromeOptions = None
    FirefoxOptions = None
    EdgeOptions = None

# Try to import webdriver_manager for auto driver management
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager

    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False


# Test configuration
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
DEFAULT_TIMEOUT = 30
APP_URL = os.environ.get("PTPD_TEST_URL", "http://127.0.0.1:7861")
HEADLESS = os.environ.get("PTPD_TEST_HEADLESS", "true").lower() == "true"
BROWSER = os.environ.get("PTPD_TEST_BROWSER", "chrome").lower()


def pytest_configure(config):
    """Add Selenium markers."""
    config.addinivalue_line(
        "markers", "selenium: mark test as requiring Selenium WebDriver"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Selenium tests if Selenium is not available."""
    if not SELENIUM_AVAILABLE:
        skip_selenium = pytest.mark.skip(reason="Selenium not installed")
        for item in items:
            if "selenium" in item.keywords:
                item.add_marker(skip_selenium)


@pytest.fixture(scope="session")
def app_url() -> str:
    """Return the application URL for testing."""
    return APP_URL


@pytest.fixture(scope="session")
def screenshots_dir() -> Path:
    """Return and ensure screenshots directory exists."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOTS_DIR


@pytest.fixture(scope="session")
def app_process(app_url: str) -> Generator[subprocess.Popen | None, None, None]:
    """Start the application if not already running."""
    import urllib.request

    # Check if app is already running
    try:
        urllib.request.urlopen(app_url, timeout=5)
        print(f"App already running at {app_url}")
        yield None
        return
    except Exception:
        print(f"Starting app at {app_url}...")

    # Start the application
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
    max_retries = 60
    for i in range(max_retries):
        try:
            urllib.request.urlopen(app_url, timeout=5)
            print(f"App started successfully at {app_url}")
            break
        except Exception:
            time.sleep(1)
            if i == max_retries - 1:
                process.terminate()
                pytest.fail("Failed to start application within timeout")

    yield process

    # Cleanup
    if process:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def create_chrome_driver(headless: bool = True) -> "webdriver.Chrome":
    """Create a Chrome WebDriver instance."""
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")

    if WEBDRIVER_MANAGER_AVAILABLE:
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)
    return webdriver.Chrome(options=options)


def create_firefox_driver(headless: bool = True) -> "webdriver.Firefox":
    """Create a Firefox WebDriver instance."""
    options = FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--width=1920")
    options.add_argument("--height=1080")

    if WEBDRIVER_MANAGER_AVAILABLE:
        service = FirefoxService(GeckoDriverManager().install())
        return webdriver.Firefox(service=service, options=options)
    return webdriver.Firefox(options=options)


def create_edge_driver(headless: bool = True) -> "webdriver.Edge":
    """Create an Edge WebDriver instance."""
    options = EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    if WEBDRIVER_MANAGER_AVAILABLE:
        service = EdgeService(EdgeChromiumDriverManager().install())
        return webdriver.Edge(service=service, options=options)
    return webdriver.Edge(options=options)


@pytest.fixture(scope="function")
def driver(
    app_process, app_url: str, screenshots_dir: Path, request
) -> Generator["webdriver.Remote", None, None]:
    """
    Create and manage WebDriver instance.

    Yields a configured WebDriver and handles cleanup including
    screenshot capture on test failure.
    """
    if not SELENIUM_AVAILABLE:
        pytest.skip("Selenium not installed")

    # Create driver based on configured browser
    if BROWSER == "chrome":
        driver = create_chrome_driver(headless=HEADLESS)
    elif BROWSER == "firefox":
        driver = create_firefox_driver(headless=HEADLESS)
    elif BROWSER == "edge":
        driver = create_edge_driver(headless=HEADLESS)
    else:
        raise ValueError(f"Unsupported browser: {BROWSER}")

    driver.set_page_load_timeout(DEFAULT_TIMEOUT)
    driver.implicitly_wait(10)

    # Navigate to app
    driver.get(app_url)

    yield driver

    # Capture screenshot on failure
    if request.node.rep_call and request.node.rep_call.failed:
        screenshot_path = screenshots_dir / f"{request.node.name}_failure.png"
        driver.save_screenshot(str(screenshot_path))
        print(f"Screenshot saved to {screenshot_path}")

    driver.quit()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result for screenshot capture."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def wait(driver) -> WebDriverWait:
    """Create a WebDriverWait instance with default timeout."""
    return WebDriverWait(driver, DEFAULT_TIMEOUT)


@pytest.fixture
def wait_for_element(driver):
    """Factory fixture to wait for elements."""

    def _wait_for_element(locator, timeout=DEFAULT_TIMEOUT):
        wait = WebDriverWait(driver, timeout)
        return wait.until(EC.presence_of_element_located(locator))

    return _wait_for_element


@pytest.fixture
def wait_for_clickable(driver):
    """Factory fixture to wait for clickable elements."""

    def _wait_for_clickable(locator, timeout=DEFAULT_TIMEOUT):
        wait = WebDriverWait(driver, timeout)
        return wait.until(EC.element_to_be_clickable(locator))

    return _wait_for_clickable


@pytest.fixture
def gradio_wait(driver):
    """Wait for Gradio app to fully load."""

    def _gradio_wait(timeout=DEFAULT_TIMEOUT):
        wait = WebDriverWait(driver, timeout)
        # Wait for Gradio footer or main container to be present
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".gradio-container"))
        )
        # Wait for any loading indicators to disappear
        try:
            wait.until(
                EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, ".loading, .progress-bar:not([style*='display: none'])")
                )
            )
        except Exception:
            pass  # Loading indicator may not exist
        time.sleep(0.5)  # Small buffer for JS initialization

    return _gradio_wait
