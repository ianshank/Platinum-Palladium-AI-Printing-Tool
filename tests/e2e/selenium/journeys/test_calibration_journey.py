"""
Calibration Journey E2E Tests.

Tests the complete calibration workflow from step tablet upload
through curve generation and export.
"""

import pytest

from tests.e2e.selenium.pages.calibration_wizard_page import CalibrationWizardPage
from tests.e2e.selenium.pages.dashboard_page import DashboardPage


@pytest.mark.selenium
@pytest.mark.e2e
class TestCalibrationJourney:
    """Test complete calibration workflow."""

    @pytest.fixture
    def wizard_page(self, driver):
        """Create CalibrationWizardPage instance."""
        return CalibrationWizardPage(driver)

    @pytest.fixture
    def dashboard_page(self, driver):
        """Create DashboardPage instance."""
        return DashboardPage(driver)

    @pytest.fixture
    def sample_step_tablet(self, tmp_path):
        """Create a sample step tablet image for testing."""
        import numpy as np
        from PIL import Image

        # Create a simple grayscale step tablet
        width, height = 420, 100
        num_patches = 21
        patch_width = width // num_patches

        img_array = np.zeros((height, width), dtype=np.uint8)
        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img_array[:, x_start:x_end] = value

        # Create RGB image
        img = Image.fromarray(img_array, mode="L").convert("RGB")
        file_path = tmp_path / "test_step_tablet.png"
        img.save(file_path)

        return file_path

    def test_navigate_to_calibration(self, wizard_page):
        """Test navigation to calibration wizard."""
        wizard_page.wait_for_gradio_ready()
        wizard_page.navigate_to_wizard()

        assert wizard_page.is_wizard_active()

    def test_upload_step_tablet(self, wizard_page, sample_step_tablet):
        """Test uploading a step tablet image."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)

        # Verify preview is displayed
        assert wizard_page.get_upload_preview()

    def test_configure_paper_settings(self, wizard_page, sample_step_tablet):
        """Test configuring paper and chemistry settings."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)

        # Configure settings
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.select_chemistry_type("Platinum/Palladium")
        wizard_page.set_metal_ratio(0.5)

        # Settings should persist (no error)

    def test_analyze_step_tablet(self, wizard_page, sample_step_tablet):
        """Test analyzing a step tablet."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        # Should detect patches
        patches = wizard_page.get_detected_patches()
        assert patches > 0, "Should detect at least one patch"

    def test_generate_calibration_curve(self, wizard_page, sample_step_tablet):
        """Test generating a calibration curve."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()

        assert wizard_page.is_curve_displayed()

    def test_export_quad_file(self, wizard_page, sample_step_tablet):
        """Test exporting curve as .quad file."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()
        wizard_page.click_export_quad()

        # Should have download link
        link = wizard_page.get_export_download_link()
        assert link is not None, "Should have export download link"

    def test_save_calibration_to_database(self, wizard_page, sample_step_tablet):
        """Test saving calibration to database."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Test Paper")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()
        wizard_page.click_save_calibration()

        confirmation = wizard_page.get_save_confirmation()
        assert "saved" in confirmation.lower() or "success" in confirmation.lower()

    def test_complete_calibration_workflow(self, wizard_page, sample_step_tablet):
        """Test the complete end-to-end calibration workflow."""
        success = wizard_page.complete_calibration_workflow(
            step_tablet_path=sample_step_tablet,
            paper_type="Arches Platine",
            chemistry_type="Platinum/Palladium",
            metal_ratio=0.5,
            exposure_time=180.0,
        )

        assert success, "Complete calibration workflow should succeed"

    def test_calibration_appears_on_dashboard(
        self, wizard_page, dashboard_page, sample_step_tablet
    ):
        """Test that saved calibration appears on dashboard."""
        # Complete calibration
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Dashboard Test Paper")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()
        wizard_page.click_save_calibration()

        # Navigate to dashboard
        dashboard_page.navigate_to_dashboard()

        # Check calibration appears in recent list
        calibrations = dashboard_page.get_recent_calibrations()
        papers = [c["paper"] for c in calibrations]
        assert any("Dashboard Test" in p for p in papers)

    def test_workflow_with_environment_settings(self, wizard_page, sample_step_tablet):
        """Test calibration workflow with environment settings."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)

        # Set environment conditions
        wizard_page.set_humidity(55.0)
        wizard_page.set_temperature(22.0)

        wizard_page.click_analyze()
        wizard_page.click_generate_curve()

        assert wizard_page.is_curve_displayed()

    def test_multiple_calibrations(self, wizard_page, sample_step_tablet):
        """Test performing multiple calibrations in sequence."""
        papers = ["Arches Platine", "Bergger COT320", "Hahnemuhle Platinum Rag"]

        for paper in papers:
            wizard_page.navigate_to_wizard()
            wizard_page.upload_step_tablet(sample_step_tablet)
            wizard_page.select_paper_type(paper)
            wizard_page.click_analyze()
            wizard_page.click_generate_curve()

            assert wizard_page.is_curve_displayed(), f"Curve should display for {paper}"
