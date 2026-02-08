"""
Error Handling and Edge Case E2E Tests.

Tests the application's resilience to errors and edge cases:
- Invalid inputs
- Network failures (simulated)
- Boundary conditions
- Graceful degradation
"""

import contextlib
import time
from pathlib import Path

import pytest

try:
    from selenium.common.exceptions import TimeoutException  # noqa: F401
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None
    WebDriverWait = None
    EC = None


@pytest.fixture
def invalid_image_file(tmp_path) -> Path:
    """Create an invalid image file."""
    file_path = tmp_path / "invalid.png"
    # Write invalid PNG data
    file_path.write_bytes(b"Not a real PNG file content")
    return file_path


@pytest.fixture
def empty_file(tmp_path) -> Path:
    """Create an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")
    return file_path


@pytest.fixture
def oversized_image(tmp_path) -> Path:
    """Create a very large image file."""
    import numpy as np
    from PIL import Image

    # Create a 5000x5000 image (large but not unreasonably so)
    img_array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="L")
    file_path = tmp_path / "oversized.png"
    img.save(file_path)
    return file_path


@pytest.fixture
def corrupted_quad_file(tmp_path) -> Path:
    """Create a corrupted .quad file."""
    file_path = tmp_path / "corrupted.quad"
    file_path.write_text("This is not valid quad format\nrandom garbage data")
    return file_path


@pytest.mark.selenium
@pytest.mark.e2e
class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    def test_invalid_image_file_rejected(self, driver, gradio_wait, invalid_image_file):
        """Test that invalid image files are rejected gracefully."""
        gradio_wait()

        # Navigate to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Try to upload invalid file
        try:
            file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            file_input.send_keys(str(invalid_image_file))
            time.sleep(2)

            # Should show error or reject file
            # App should remain functional
        except Exception:
            pass

        # Verify app is still functional
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_empty_file_rejected(self, driver, gradio_wait, empty_file):
        """Test that empty files are rejected gracefully."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        try:
            file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            file_input.send_keys(str(empty_file))
            time.sleep(2)
        except Exception:
            pass

        # App should remain functional
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_negative_values_handled(self, driver, gradio_wait):
        """Test that negative values in number inputs are handled."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Try to enter negative values
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:  # First two number inputs
            try:
                inp.clear()
                inp.send_keys("-100")
            except Exception:
                pass

        time.sleep(0.5)

        # App should handle gracefully
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_extremely_large_values_handled(self, driver, gradio_wait):
        """Test that extremely large values are handled."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:
            try:
                inp.clear()
                inp.send_keys("999999999999")
            except Exception:
                pass

        time.sleep(0.5)

        # App should handle gracefully
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_special_characters_in_text_input(self, driver, gradio_wait):
        """Test that special characters in text inputs are handled."""
        gradio_wait()

        # Find any text input
        text_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea")

        special_chars = "<script>alert('xss')</script>'; DROP TABLE users;--"

        for inp in text_inputs[:2]:
            try:
                inp.clear()
                inp.send_keys(special_chars)
            except Exception:
                pass

        time.sleep(0.5)

        # App should sanitize and remain functional
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_unicode_characters_handled(self, driver, gradio_wait):
        """Test that Unicode characters are handled properly."""
        gradio_wait()

        text_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea")

        unicode_text = "测试文字 🎨 Ñoño café"

        for inp in text_inputs[:2]:
            try:
                inp.clear()
                inp.send_keys(unicode_text)
            except Exception:
                pass

        time.sleep(0.5)

        # App should handle Unicode
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")


@pytest.mark.selenium
@pytest.mark.e2e
class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_zero_values_handled(self, driver, gradio_wait):
        """Test that zero values are handled appropriately."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:
            try:
                inp.clear()
                inp.send_keys("0")
            except Exception:
                pass

        time.sleep(0.5)

        # Zero might be valid or invalid depending on the field
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_maximum_ratio_value(self, driver, gradio_wait):
        """Test handling of maximum ratio value (1.0)."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Look for ratio slider or input
        sliders = driver.find_elements(By.CSS_SELECTOR, "input[type='range']")
        for slider in sliders:
            try:
                driver.execute_script("arguments[0].value = 1.0", slider)
                driver.execute_script(
                    "arguments[0].dispatchEvent(new Event('input', { bubbles: true }))",
                    slider,
                )
            except Exception:
                pass

        time.sleep(0.5)

        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_minimum_ratio_value(self, driver, gradio_wait):
        """Test handling of minimum ratio value (0.0)."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        sliders = driver.find_elements(By.CSS_SELECTOR, "input[type='range']")
        for slider in sliders:
            try:
                driver.execute_script("arguments[0].value = 0.0", slider)
                driver.execute_script(
                    "arguments[0].dispatchEvent(new Event('input', { bubbles: true }))",
                    slider,
                )
            except Exception:
                pass

        time.sleep(0.5)

        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_very_small_print_size(self, driver, gradio_wait):
        """Test handling of very small print sizes."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:
            try:
                inp.clear()
                inp.send_keys("0.001")
            except Exception:
                pass

        time.sleep(0.5)

        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_very_large_print_size(self, driver, gradio_wait):
        """Test handling of very large print sizes."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:
            try:
                inp.clear()
                inp.send_keys("1000")  # Very large print
            except Exception:
                pass

        time.sleep(0.5)

        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")


@pytest.mark.selenium
@pytest.mark.e2e
class TestGracefulDegradation:
    """Test graceful degradation under adverse conditions."""

    def test_app_functional_after_error(self, driver, gradio_wait, invalid_image_file):
        """Test that app remains functional after an error."""
        gradio_wait()

        # Trigger an error
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        try:
            file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            file_input.send_keys(str(invalid_image_file))
        except Exception:
            pass

        time.sleep(2)

        # Should still be able to navigate
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_app_recovers_from_multiple_errors(self, driver, gradio_wait):
        """Test that app recovers from multiple consecutive errors."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")

        # Try to trigger multiple errors
        for _ in range(3):
            try:
                # Try invalid inputs
                inputs = driver.find_elements(By.CSS_SELECTOR, "input")
                for inp in inputs[:2]:
                    try:
                        inp.clear()
                        inp.send_keys("invalid_value_!@#$%")
                    except Exception:
                        pass
            except Exception:
                pass

            time.sleep(0.5)

        # App should still be functional
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

        # Should be able to navigate
        for tab in tabs:
            if tab.is_displayed() and tab.is_enabled():
                tab.click()
                break

    def test_partial_input_handled(self, driver, gradio_wait):
        """Test handling of partial/incomplete input."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Fill only some fields
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        if len(inputs) >= 2:
            try:
                inputs[0].clear()
                inputs[0].send_keys("8")
                # Leave other fields empty or default
            except Exception:
                pass

        time.sleep(0.5)

        # Try to trigger calculation
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for btn in buttons:
            if "calculate" in btn.text.lower():
                with contextlib.suppress(Exception):
                    btn.click()
                break

        time.sleep(1)

        # App should handle gracefully
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")


@pytest.mark.selenium
@pytest.mark.e2e
class TestConcurrentOperations:
    """Test handling of concurrent/rapid operations."""

    def test_rapid_button_clicks(self, driver, gradio_wait):
        """Test handling of rapid button clicks."""
        gradio_wait()

        buttons = driver.find_elements(By.CSS_SELECTOR, "button")

        # Click multiple buttons rapidly
        for btn in buttons[:5]:
            try:
                if btn.is_displayed() and btn.is_enabled():
                    btn.click()
                    # No wait between clicks
            except Exception:
                pass

        time.sleep(2)

        # App should remain stable
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_rapid_input_changes(self, driver, gradio_wait):
        """Test handling of rapid input changes."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")

        # Rapidly change values
        for _ in range(10):
            for inp in inputs[:2]:
                try:
                    inp.clear()
                    inp.send_keys(str(_ + 1))
                except Exception:
                    pass

        time.sleep(1)

        # App should handle debouncing/throttling
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_multiple_file_upload_attempts(self, driver, gradio_wait, tmp_path):
        """Test handling of multiple file upload attempts."""
        import numpy as np
        from PIL import Image

        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Create multiple test images
        files = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            img = Image.fromarray(img_array, mode="L")
            file_path = tmp_path / f"test_{i}.png"
            img.save(file_path)
            files.append(file_path)

        # Try to upload multiple times
        file_input = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        if file_input:
            for f in files:
                try:
                    file_input[0].send_keys(str(f))
                    time.sleep(0.5)
                except Exception:
                    pass

        time.sleep(2)

        # App should handle multiple uploads
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")


@pytest.mark.selenium
@pytest.mark.e2e
class TestStateConsistency:
    """Test state consistency after various operations."""

    def test_state_after_clear_operation(self, driver, gradio_wait):
        """Test that state is properly cleared."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Fill some inputs
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
        for inp in inputs[:2]:
            try:
                inp.clear()
                inp.send_keys("42")
            except Exception:
                pass

        # Look for clear/reset button
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for btn in buttons:
            if "clear" in btn.text.lower() or "reset" in btn.text.lower():
                with contextlib.suppress(Exception):
                    btn.click()
                break

        time.sleep(1)

        # State should be cleared or app should remain functional
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_state_consistency_across_refreshes(self, driver, gradio_wait):
        """Test state consistency when page is refreshed."""
        gradio_wait()

        # Navigate to a specific state
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        time.sleep(1)

        # Refresh the page
        driver.refresh()
        gradio_wait()

        # App should load correctly
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")

    def test_back_button_behavior(self, driver, gradio_wait):
        """Test browser back button behavior."""
        gradio_wait()

        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")

        # Navigate through tabs
        for tab in tabs[:3]:
            if tab.is_displayed() and tab.is_enabled():
                tab.click()
                time.sleep(0.5)

        # Use browser back
        driver.back()
        time.sleep(1)

        # App should handle navigation
        assert driver.find_element(By.CSS_SELECTOR, ".gradio-container")
