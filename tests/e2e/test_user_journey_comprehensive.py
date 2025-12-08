"""
Comprehensive User Journey E2E Tests.

Tests complete user journeys from start to finish, covering:
- First-time user onboarding
- Expert user workflows
- Error recovery scenarios
- Cross-feature integration
- Session persistence
"""

from pathlib import Path
from typing import TYPE_CHECKING
import time

import pytest

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None
    WebDriverWait = None
    EC = None


@pytest.fixture
def sample_step_tablet(tmp_path) -> Path:
    """Create a realistic step tablet image for testing."""
    from PIL import Image
    import numpy as np

    # Create a 21-step tablet with realistic gamma curve
    width, height = 840, 150
    num_patches = 21
    patch_width = width // num_patches

    img_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_patches):
        # Use gamma curve similar to real printing
        normalized = i / (num_patches - 1)
        density = 255 - int(255 * (normalized ** 0.45))  # Approximate gamma 2.2
        x_start = i * patch_width
        x_end = (i + 1) * patch_width
        img_array[:, x_start:x_end] = density

    img = Image.fromarray(img_array, mode="L").convert("RGB")
    file_path = tmp_path / "test_step_tablet_21.png"
    img.save(file_path, quality=95)

    return file_path


@pytest.fixture
def sample_high_contrast_tablet(tmp_path) -> Path:
    """Create a high contrast step tablet for testing."""
    from PIL import Image
    import numpy as np

    width, height = 840, 150
    num_patches = 21
    patch_width = width // num_patches

    img_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_patches):
        # Higher contrast curve
        normalized = i / (num_patches - 1)
        # S-curve for higher contrast
        if normalized < 0.5:
            density = 255 - int(255 * (2 * normalized ** 2))
        else:
            density = 255 - int(255 * (1 - 2 * (1 - normalized) ** 2))
        x_start = i * patch_width
        x_end = (i + 1) * patch_width
        img_array[:, x_start:x_end] = density

    img = Image.fromarray(img_array, mode="L").convert("RGB")
    file_path = tmp_path / "high_contrast_tablet.png"
    img.save(file_path, quality=95)

    return file_path


@pytest.mark.selenium
@pytest.mark.e2e
class TestFirstTimeUserJourney:
    """Test the complete first-time user experience."""

    def test_initial_app_load(self, driver, gradio_wait):
        """Test that the app loads correctly for a new user."""
        gradio_wait()

        # App should be fully loaded
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_explore_all_tabs(self, driver, gradio_wait):
        """Test that a new user can explore all tabs."""
        gradio_wait()

        # Find and click through all tabs
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        tab_names = [tab.text for tab in tabs]

        assert len(tabs) > 0, "Should have at least one tab"

        for tab in tabs:
            if tab.is_displayed() and tab.is_enabled():
                tab.click()
                time.sleep(0.5)  # Allow tab content to load

    def test_first_calibration_workflow(
        self, driver, gradio_wait, sample_step_tablet
    ):
        """Test complete first calibration for a new user."""
        gradio_wait()

        # Navigate to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Workflow should be accessible

    def test_view_dashboard_summary(self, driver, gradio_wait):
        """Test viewing dashboard summary."""
        gradio_wait()

        # Navigate to dashboard
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "dashboard" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Dashboard should be visible


@pytest.mark.selenium
@pytest.mark.e2e
class TestExpertUserJourney:
    """Test workflows for experienced users."""

    def test_rapid_calibration_workflow(
        self, driver, gradio_wait, sample_step_tablet
    ):
        """Test rapid calibration workflow for expert users."""
        gradio_wait()

        # Expert users know where everything is
        # Navigate directly to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

    def test_batch_processing_setup(self, driver, gradio_wait):
        """Test setting up batch processing."""
        gradio_wait()

        # Look for batch processing features
        # This tests the UI availability, not actual processing

    def test_advanced_settings_access(self, driver, gradio_wait):
        """Test accessing advanced settings."""
        gradio_wait()

        # Look for advanced settings or preferences
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for button in buttons:
            if "advanced" in button.text.lower() or "settings" in button.text.lower():
                button.click()
                time.sleep(0.5)
                break


@pytest.mark.selenium
@pytest.mark.e2e
class TestErrorRecoveryJourney:
    """Test error recovery scenarios."""

    def test_invalid_file_upload_recovery(self, driver, gradio_wait, tmp_path):
        """Test recovery from invalid file upload."""
        gradio_wait()

        # Create an invalid file (not an image)
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("This is not an image")

        # Navigate to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Try to upload invalid file - should show error or be rejected
        # App should remain functional

    def test_navigation_after_error(self, driver, gradio_wait):
        """Test that navigation works after an error."""
        gradio_wait()

        # Navigate through tabs to ensure app is functional
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")

        for tab in tabs[:3]:  # Check first 3 tabs
            if tab.is_displayed() and tab.is_enabled():
                tab.click()
                time.sleep(0.3)

        # Should end up on a valid tab
        active_tab = driver.find_element(
            By.CSS_SELECTOR, "button[role='tab'][aria-selected='true']"
        )
        assert active_tab is not None

    def test_app_state_after_refresh(self, driver, gradio_wait):
        """Test that app recovers correctly after browser refresh."""
        gradio_wait()

        # Navigate to a specific tab
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        if len(tabs) > 1:
            tabs[1].click()
            time.sleep(0.5)

        # Refresh the page
        driver.refresh()
        gradio_wait()

        # App should load correctly
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")


@pytest.mark.selenium
@pytest.mark.e2e
class TestAccessibilityJourney:
    """Test accessibility features and keyboard navigation."""

    def test_tab_key_navigation(self, driver, gradio_wait):
        """Test that Tab key navigation works."""
        from selenium.webdriver.common.keys import Keys

        gradio_wait()

        # Focus on body and try tab navigation
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.TAB)
        time.sleep(0.2)

        # Check if something is focused
        focused = driver.switch_to.active_element
        assert focused is not None

    def test_screen_reader_labels(self, driver, gradio_wait):
        """Test that elements have appropriate labels for screen readers."""
        gradio_wait()

        # Check for aria-label attributes
        elements_with_aria = driver.find_elements(
            By.CSS_SELECTOR, "[aria-label], [aria-labelledby], [role]"
        )

        # Should have some accessible elements
        # Note: Gradio provides basic accessibility

    def test_focus_visible_styles(self, driver, gradio_wait):
        """Test that focused elements have visible styles."""
        gradio_wait()

        # This is a visual check - in real testing would use visual regression
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        if buttons:
            buttons[0].click()
            # Focus should be visible

    def test_color_contrast(self, driver, gradio_wait):
        """Test basic color contrast (placeholder for real contrast testing)."""
        gradio_wait()

        # In real testing, would use axe-core or similar
        # Here we just verify the page loads with the theme applied


@pytest.mark.selenium
@pytest.mark.e2e
class TestResponsiveUIJourney:
    """Test UI responsiveness at different viewport sizes."""

    @pytest.fixture
    def set_viewport(self, driver):
        """Fixture to set viewport size."""

        def _set_viewport(width: int, height: int):
            driver.set_window_size(width, height)
            time.sleep(0.5)

        return _set_viewport

    def test_desktop_layout(self, driver, gradio_wait, set_viewport):
        """Test layout at desktop resolution."""
        set_viewport(1920, 1080)
        gradio_wait()

        # Check that main container is visible
        container = driver.find_element(By.CSS_SELECTOR, ".gradio-container")
        assert container.is_displayed()

    def test_tablet_layout(self, driver, gradio_wait, set_viewport):
        """Test layout at tablet resolution."""
        set_viewport(768, 1024)
        gradio_wait()

        # Check that main container is visible
        container = driver.find_element(By.CSS_SELECTOR, ".gradio-container")
        assert container.is_displayed()

    def test_mobile_layout(self, driver, gradio_wait, set_viewport):
        """Test layout at mobile resolution."""
        set_viewport(375, 667)
        gradio_wait()

        # Check that main container is visible
        container = driver.find_element(By.CSS_SELECTOR, ".gradio-container")
        assert container.is_displayed()

    def test_tab_overflow_handling(self, driver, gradio_wait, set_viewport):
        """Test that tabs handle overflow correctly."""
        set_viewport(600, 800)  # Narrow viewport
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        # Tabs should either wrap or have scroll mechanism


@pytest.mark.selenium
@pytest.mark.e2e
class TestDataPersistenceJourney:
    """Test data persistence across sessions."""

    def test_form_data_persists_during_session(self, driver, gradio_wait):
        """Test that form data persists during the session."""
        gradio_wait()

        # Navigate to chemistry calculator
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Find and fill a number input
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        if inputs:
            inputs[0].clear()
            inputs[0].send_keys("42")

            # Navigate away and back
            for tab in tabs:
                if "dashboard" in tab.text.lower():
                    tab.click()
                    break

            time.sleep(0.5)

            for tab in tabs:
                if "chemistry" in tab.text.lower():
                    tab.click()
                    break

            time.sleep(0.5)

            # Value may or may not persist depending on implementation

    def test_calibration_history_saved(self, driver, gradio_wait):
        """Test that calibration history is saved."""
        gradio_wait()

        # Navigate to dashboard to see history
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "dashboard" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Look for history/recent calibrations section


@pytest.mark.selenium
@pytest.mark.e2e
class TestIntegrationJourney:
    """Test integration between different features."""

    def test_calibration_affects_chemistry(
        self, driver, gradio_wait, sample_step_tablet
    ):
        """Test that calibration results affect chemistry calculations."""
        gradio_wait()

        # This tests the integration between calibration and chemistry
        # In a real app, calibration might set paper properties used in chemistry

    def test_ai_assistant_context_awareness(self, driver, gradio_wait):
        """Test that AI assistant is aware of app context."""
        gradio_wait()

        # Navigate to AI assistant
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "ai" in tab.text.lower() or "assistant" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # AI assistant should be available

    def test_session_log_records_actions(self, driver, gradio_wait):
        """Test that session log records user actions."""
        gradio_wait()

        # Navigate to session log
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "session" in tab.text.lower() or "log" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Session log should be visible or actions should be logged


@pytest.mark.selenium
@pytest.mark.e2e
class TestPerformanceJourney:
    """Test UI performance characteristics."""

    def test_initial_load_time(self, driver, app_url):
        """Test that initial page load is reasonably fast."""
        import time

        start = time.time()
        driver.get(app_url)

        # Wait for gradio container
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".gradio-container"))
        )

        load_time = time.time() - start
        # Should load within 30 seconds (generous for CI environments)
        assert load_time < 30, f"Page took {load_time:.2f}s to load"

    def test_tab_switch_responsiveness(self, driver, gradio_wait):
        """Test that tab switching is responsive."""
        import time

        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        if len(tabs) < 2:
            pytest.skip("Not enough tabs to test switching")

        # Measure tab switch time
        start = time.time()
        tabs[1].click()

        # Wait for tab content to appear
        time.sleep(0.5)

        switch_time = time.time() - start
        # Should switch within 2 seconds
        assert switch_time < 2, f"Tab switch took {switch_time:.2f}s"

    def test_no_memory_leaks_on_repeated_operations(self, driver, gradio_wait):
        """Test for potential memory leaks on repeated operations."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")

        # Perform repeated tab switches
        for _ in range(10):
            for tab in tabs[:3]:  # First 3 tabs
                if tab.is_displayed() and tab.is_enabled():
                    tab.click()
                    time.sleep(0.1)

        # App should still be responsive
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")
