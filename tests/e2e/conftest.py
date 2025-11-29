import pytest
import subprocess
import time
import os
import signal
from playwright.sync_api import Page, expect

@pytest.fixture(scope="session")
def app_url():
    """Return the URL of the running app."""
    return "http://127.0.0.1:7861"

@pytest.fixture(scope="session", autouse=True)
def ensure_app_running(app_url):
    """Ensure the app is running before tests start."""
    # Check if app is already running
    import urllib.request
    try:
        urllib.request.urlopen(app_url)
        print("App is already running.")
        yield
        return
    except Exception:
        print("App not running. Starting it...")

    # Start the app
    # We assume we are in the project root
    process = subprocess.Popen(
        ["python", "-m", "ptpd_calibration.ui.gradio_app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": "7861"}
    )
    
    # Wait for app to start
    max_retries = 30
    for i in range(max_retries):
        try:
            urllib.request.urlopen(app_url)
            print("App started successfully.")
            break
        except Exception:
            time.sleep(1)
            if i == max_retries - 1:
                process.terminate()
                raise RuntimeError("Failed to start app.")

    yield

    # Cleanup (optional, maybe we want to keep it running)
    # process.terminate()
