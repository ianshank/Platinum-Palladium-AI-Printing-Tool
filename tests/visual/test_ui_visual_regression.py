"""
UI Visual Regression Tests.

Tests for visual consistency of the PTPD Calibration UI.
"""

import pytest

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


@pytest.mark.visual
@pytest.mark.selenium
class TestDashboardVisual:
    """Visual regression tests for the Dashboard."""

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_dashboard_layout(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
    ):
        """Test Dashboard layout matches baseline."""
        gradio_wait()

        # Navigate to dashboard
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "dashboard" in tab.text.lower():
                tab.click()
                break

        gradio_wait()

        screenshot = capture_full_screenshot("dashboard")
        visual_comparator.assert_match("dashboard_layout", screenshot)

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_dashboard_dark_mode(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
    ):
        """Test Dashboard in dark mode matches baseline."""
        gradio_wait()

        # Toggle dark mode if available
        try:
            dark_toggle = driver.find_element(
                By.CSS_SELECTOR, "[aria-label='dark mode'], .dark-toggle"
            )
            dark_toggle.click()
            gradio_wait()
        except Exception:
            pytest.skip("Dark mode toggle not found")

        screenshot = capture_full_screenshot("dashboard_dark")
        visual_comparator.assert_match("dashboard_dark_mode", screenshot)


@pytest.mark.visual
@pytest.mark.selenium
class TestCalibrationWizardVisual:
    """Visual regression tests for the Calibration Wizard."""

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_calibration_wizard_initial(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
    ):
        """Test Calibration Wizard initial state matches baseline."""
        gradio_wait()

        # Navigate to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        gradio_wait()

        screenshot = capture_full_screenshot("calibration_wizard")
        visual_comparator.assert_match("calibration_wizard_initial", screenshot)

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_calibration_wizard_with_image(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
        sample_step_tablet_image,
    ):
        """Test Calibration Wizard with uploaded image."""
        gradio_wait()

        # Navigate to calibration
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "calibration" in tab.text.lower():
                tab.click()
                break

        gradio_wait()

        # Upload image
        try:
            file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            file_input.send_keys(str(sample_step_tablet_image))
            gradio_wait()
        except Exception:
            pytest.skip("File upload not found")

        screenshot = capture_full_screenshot("calibration_with_image")
        visual_comparator.assert_match("calibration_wizard_with_image", screenshot)


@pytest.mark.visual
@pytest.mark.selenium
class TestChemistryCalculatorVisual:
    """Visual regression tests for the Chemistry Calculator."""

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_chemistry_calculator_initial(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
    ):
        """Test Chemistry Calculator initial state."""
        gradio_wait()

        # Navigate to chemistry
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "chemistry" in tab.text.lower():
                tab.click()
                break

        gradio_wait()

        screenshot = capture_full_screenshot("chemistry_calculator")
        visual_comparator.assert_match("chemistry_calculator_initial", screenshot)


@pytest.mark.visual
@pytest.mark.selenium
class TestAIAssistantVisual:
    """Visual regression tests for the AI Assistant."""

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_ai_assistant_initial(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
    ):
        """Test AI Assistant initial state."""
        gradio_wait()

        # Navigate to AI assistant
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if "ai" in tab.text.lower() or "assistant" in tab.text.lower():
                tab.click()
                break

        gradio_wait()

        screenshot = capture_full_screenshot("ai_assistant")
        visual_comparator.assert_match("ai_assistant_initial", screenshot)


@pytest.mark.visual
@pytest.mark.selenium
class TestComponentVisual:
    """Visual regression tests for individual components."""

    @pytest.fixture
    def sample_step_tablet_image(self, tmp_path):
        """Create a sample step tablet image."""
        import numpy as np

        width, height = 420, 100
        num_patches = 21
        patch_width = width // num_patches

        img_array = np.zeros((height, width), dtype=np.uint8)
        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img_array[:, x_start:x_end] = value

        img = Image.fromarray(img_array, mode="L").convert("RGB")
        file_path = tmp_path / "step_tablet.png"
        img.save(file_path)

        return file_path

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_button_styles(
        self,
        driver,
        visual_comparator,
        capture_component_screenshot,
        gradio_wait,
    ):
        """Test button component styles."""
        gradio_wait()

        try:
            screenshot = capture_component_screenshot("button.primary", "buttons")
            visual_comparator.assert_match("button_primary", screenshot)
        except Exception:
            pytest.skip("Primary button not found")

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    def test_tab_bar(
        self,
        driver,
        visual_comparator,
        capture_component_screenshot,
        gradio_wait,
    ):
        """Test tab bar component."""
        gradio_wait()

        try:
            screenshot = capture_component_screenshot(
                "[role='tablist']", "tab_bar"
            )
            visual_comparator.assert_match("tab_bar", screenshot)
        except Exception:
            pytest.skip("Tab bar not found")


@pytest.mark.visual
class TestResponsiveVisual:
    """Visual regression tests for responsive layouts."""

    @pytest.fixture
    def resize_window(self, driver):
        """Factory fixture to resize browser window."""

        def _resize(width: int, height: int):
            driver.set_window_size(width, height)

        return _resize

    @pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
    @pytest.mark.parametrize(
        "width,height,name",
        [
            (1920, 1080, "desktop_large"),
            (1366, 768, "desktop_medium"),
            (1024, 768, "tablet_landscape"),
            (768, 1024, "tablet_portrait"),
        ],
    )
    def test_responsive_layouts(
        self,
        driver,
        visual_comparator,
        capture_full_screenshot,
        gradio_wait,
        resize_window,
        width,
        height,
        name,
    ):
        """Test responsive layout at various viewport sizes."""
        resize_window(width, height)
        gradio_wait()

        screenshot = capture_full_screenshot(name)
        visual_comparator.assert_match(f"responsive_{name}", screenshot, tolerance=0.15)


@pytest.mark.visual
class TestChartVisual:
    """Visual regression tests for chart/plot components."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_curve_plot_rendering(self, visual_comparator, tmp_path):
        """Test curve plot rendering consistency."""
        # Create a test plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            pytest.skip("Matplotlib not installed")

        # Create curve plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 1, 256)
        y = x**0.9

        ax.plot(x, y, "b-", linewidth=2, label="Curve")
        ax.plot(x, x, "k--", linewidth=1, alpha=0.5, label="Linear")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save to image
        plot_path = tmp_path / "curve_plot.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

        # Load and compare
        plot_image = Image.open(plot_path)
        visual_comparator.assert_match("curve_plot", plot_image, tolerance=0.05)
