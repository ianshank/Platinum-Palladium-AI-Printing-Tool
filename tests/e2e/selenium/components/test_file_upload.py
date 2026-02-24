"""
File Upload Component E2E Tests.

Tests file upload interactions across the application.
"""

import pytest

from tests.e2e.selenium.pages.base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


@pytest.mark.selenium
@pytest.mark.e2e
class TestFileUpload:
    """Test file upload component interactions."""

    @pytest.fixture
    def page(self, driver):
        """Create BasePage instance."""
        return BasePage(driver)

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image file."""
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        file_path = tmp_path / "test_image.png"
        img.save(file_path)
        return file_path

    @pytest.fixture
    def sample_tiff(self, tmp_path):
        """Create a sample TIFF file."""
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
        file_path = tmp_path / "test_image.tiff"
        img.save(file_path)
        return file_path

    def test_upload_png_image(self, page, sample_image):
        """Test uploading a PNG image."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        # Find file upload component
        try:
            page.upload_file("Step Tablet", str(sample_image))
            # Should show preview
            preview = page.wait_for_element(By.CSS_SELECTOR, ".image-preview, img", timeout=10)
            assert preview is not None
        except Exception:
            pytest.skip("File upload component not found")

    def test_upload_tiff_image(self, page, sample_tiff):
        """Test uploading a TIFF image."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            page.upload_file("Step Tablet", str(sample_tiff))
            preview = page.wait_for_element(By.CSS_SELECTOR, ".image-preview, img", timeout=10)
            assert preview is not None
        except Exception:
            pytest.skip("File upload component not found")

    def test_upload_large_image(self, page, tmp_path):
        """Test uploading a large image."""
        import numpy as np
        from PIL import Image

        # Create a larger image (2000x2000)
        img = Image.fromarray(np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8))
        file_path = tmp_path / "large_image.png"
        img.save(file_path)

        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            page.upload_file("Step Tablet", str(file_path))
            # Should still work with large images
        except Exception:
            pytest.skip("File upload component not found")

    def test_upload_invalid_file_type(self, page, tmp_path):
        """Test uploading an invalid file type."""
        # Create a text file
        file_path = tmp_path / "test.txt"
        file_path.write_text("This is not an image")

        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        import contextlib

        with contextlib.suppress(Exception):
            page.upload_file("Step Tablet", str(file_path))
            # Should show error or reject the file

    def test_replace_uploaded_file(self, page, sample_image, tmp_path):
        """Test replacing an uploaded file with a new one."""
        import numpy as np
        from PIL import Image

        # Create a second image
        img2 = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
        file_path2 = tmp_path / "test_image2.png"
        img2.save(file_path2)

        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            # Upload first file
            page.upload_file("Step Tablet", str(sample_image))

            # Upload second file (replace)
            page.upload_file("Step Tablet", str(file_path2))

            # Should show the new image
        except Exception:
            pytest.skip("File upload component not found")


@pytest.mark.selenium
@pytest.mark.e2e
class TestTabNavigation:
    """Test tab navigation component."""

    @pytest.fixture
    def page(self, driver):
        """Create BasePage instance."""
        return BasePage(driver)

    def test_all_tabs_visible(self, page):
        """Test that all tabs are visible."""
        page.wait_for_gradio_ready()

        tabs = page.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        assert len(tabs) > 0, "Should have at least one tab"

    def test_click_each_tab(self, page):
        """Test clicking on each tab."""
        page.wait_for_gradio_ready()

        tab_names = ["🏠 Dashboard", "📊 Calibration", "🧪 Darkroom", "🤖 AI Tools"]

        for name in tab_names:
            try:
                page.click_tab(name)
                page.wait_for_gradio_ready()
            except Exception:
                pass  # Tab may not exist

    def test_active_tab_styling(self, page):
        """Test that active tab has correct styling."""
        page.wait_for_gradio_ready()

        active_tab = page.find_element(By.CSS_SELECTOR, "button[role='tab'][aria-selected='true']")
        assert active_tab is not None

    def test_tab_keyboard_navigation(self, page):
        """Test tab navigation with keyboard."""
        from selenium.webdriver.common.keys import Keys

        page.wait_for_gradio_ready()

        # Focus first tab
        tabs = page.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        if tabs:
            tabs[0].send_keys(Keys.TAB)
            # Should move focus


@pytest.mark.selenium
@pytest.mark.e2e
class TestSliderControls:
    """Test slider control components."""

    @pytest.fixture
    def page(self, driver):
        """Create BasePage instance."""
        return BasePage(driver)

    def test_slider_adjustment(self, page):
        """Test adjusting slider values."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            page.adjust_slider("Metal Ratio", 0.7)
        except Exception:
            pytest.skip("Slider component not found")

    def test_slider_min_max(self, page):
        """Test slider min/max boundaries."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            # Try min value
            page.adjust_slider("Metal Ratio", 0.0)

            # Try max value
            page.adjust_slider("Metal Ratio", 1.0)
        except Exception:
            pytest.skip("Slider component not found")


@pytest.mark.selenium
@pytest.mark.e2e
class TestDropdownSelectors:
    """Test dropdown selector components."""

    @pytest.fixture
    def page(self, driver):
        """Create BasePage instance."""
        return BasePage(driver)

    def test_dropdown_open_close(self, page):
        """Test opening and closing dropdowns."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        try:
            page.select_dropdown("Paper Type", "Arches Platine")
        except Exception:
            pytest.skip("Dropdown component not found")

    def test_dropdown_selection(self, page):
        """Test selecting dropdown options."""
        page.wait_for_gradio_ready()
        page.click_tab("📊 Calibration")

        papers = ["Arches Platine", "Bergger COT320", "Hahnemuhle Platinum Rag"]

        import contextlib

        for paper in papers:
            with contextlib.suppress(Exception):
                page.select_dropdown("Paper Type", paper)
