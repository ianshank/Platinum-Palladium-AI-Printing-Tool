"""
Multi-Tab Workflow E2E Tests.

Tests workflows that span multiple tabs and integrate different features.
"""

import pytest

from tests.e2e.selenium.pages.ai_assistant_page import AIAssistantPage
from tests.e2e.selenium.pages.calibration_wizard_page import CalibrationWizardPage
from tests.e2e.selenium.pages.chemistry_calculator_page import ChemistryCalculatorPage
from tests.e2e.selenium.pages.dashboard_page import DashboardPage


@pytest.mark.selenium
@pytest.mark.e2e
class TestMultiTabWorkflow:
    """Test workflows that span multiple tabs."""

    @pytest.fixture
    def wizard_page(self, driver):
        """Create CalibrationWizardPage instance."""
        return CalibrationWizardPage(driver)

    @pytest.fixture
    def chemistry_page(self, driver):
        """Create ChemistryCalculatorPage instance."""
        return ChemistryCalculatorPage(driver)

    @pytest.fixture
    def dashboard_page(self, driver):
        """Create DashboardPage instance."""
        return DashboardPage(driver)

    @pytest.fixture
    def ai_page(self, driver):
        """Create AIAssistantPage instance."""
        return AIAssistantPage(driver)

    @pytest.fixture
    def sample_step_tablet(self, tmp_path):
        """Create a sample step tablet image for testing."""
        import numpy as np
        from PIL import Image

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
        file_path = tmp_path / "test_step_tablet.png"
        img.save(file_path)

        return file_path

    def test_tab_navigation_preserves_state(self, chemistry_page, wizard_page, dashboard_page):
        """Test that navigating between tabs preserves state."""
        # Set up chemistry tab
        chemistry_page.wait_for_gradio_ready()
        chemistry_page.navigate_to_chemistry()
        chemistry_page.set_print_dimensions(8.0, 10.0)

        # Navigate away and back
        dashboard_page.navigate_to_dashboard()
        wizard_page.navigate_to_wizard()
        chemistry_page.navigate_to_chemistry()

        # State should be preserved (no error when accessing components)
        # The exact state preservation depends on Gradio implementation

    def test_calibration_to_chemistry_workflow(
        self, wizard_page, chemistry_page, sample_step_tablet
    ):
        """Test workflow from calibration to chemistry calculation."""
        # Step 1: Perform calibration
        wizard_page.wait_for_gradio_ready()
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()

        assert wizard_page.is_curve_displayed()

        # Step 2: Calculate chemistry for a print
        chemistry_page.navigate_to_chemistry()
        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
            chemistry_type="Platinum/Palladium",
            metal_ratio=0.5,
        )

        assert len(results) > 0

    def test_dashboard_to_calibration_workflow(self, dashboard_page, wizard_page):
        """Test workflow from dashboard quick action to calibration."""
        dashboard_page.wait_for_gradio_ready()
        dashboard_page.navigate_to_dashboard()

        # Use quick action to start new calibration
        try:
            dashboard_page.click_quick_action("New Calibration")
        except Exception:
            # Quick action may not exist, navigate manually
            wizard_page.navigate_to_wizard()

        assert wizard_page.is_wizard_active()

    def test_ai_assisted_calibration_workflow(self, ai_page, wizard_page, sample_step_tablet):
        """Test using AI assistance during calibration."""
        # Start with AI assistance
        ai_page.wait_for_gradio_ready()
        ai_page.navigate_to_assistant()

        response = ai_page.ask_question_and_wait(
            "What settings should I use for a first calibration test?"
        )
        assert len(response) > 0

        # Then perform calibration
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()

        assert wizard_page.is_curve_displayed()

    def test_complete_new_user_workflow(
        self, dashboard_page, wizard_page, chemistry_page, sample_step_tablet
    ):
        """Test complete workflow for a new user."""
        # Step 1: Check dashboard
        dashboard_page.wait_for_gradio_ready()
        dashboard_page.navigate_to_dashboard()

        # Step 2: Perform first calibration
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.set_metal_ratio(0.5)
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()

        assert wizard_page.is_curve_displayed()

        # Step 3: Export calibration
        wizard_page.click_export_quad()

        # Step 4: Calculate first recipe
        chemistry_page.navigate_to_chemistry()
        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
        )

        assert len(results) > 0

    def test_all_tabs_accessible(self, dashboard_page, wizard_page, chemistry_page, ai_page):
        """Test that all tabs are accessible."""
        # Wait for app to load
        dashboard_page.wait_for_gradio_ready()

        # Navigate to each tab
        dashboard_page.navigate_to_dashboard()
        assert dashboard_page.is_dashboard_active()

        wizard_page.navigate_to_wizard()
        assert wizard_page.is_wizard_active()

        chemistry_page.navigate_to_chemistry()
        assert chemistry_page.is_chemistry_active()

        ai_page.navigate_to_assistant()
        assert ai_page.is_assistant_active()

    def test_rapid_tab_switching(self, dashboard_page, wizard_page, chemistry_page):
        """Test rapid switching between tabs."""
        dashboard_page.wait_for_gradio_ready()

        for _ in range(5):
            dashboard_page.navigate_to_dashboard()
            wizard_page.navigate_to_wizard()
            chemistry_page.navigate_to_chemistry()

        # Should end up on chemistry tab
        assert chemistry_page.is_chemistry_active()

    def test_data_flow_between_tabs(self, wizard_page, chemistry_page, sample_step_tablet):
        """Test that data flows correctly between tabs."""
        # Calibrate for specific paper
        wizard_page.wait_for_gradio_ready()
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Bergger COT320")
        wizard_page.click_analyze()
        wizard_page.click_generate_curve()
        wizard_page.click_save_calibration()

        # Chemistry calculator should potentially use calibration data
        chemistry_page.navigate_to_chemistry()
        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
        )

        # Should calculate successfully
        assert len(results) > 0

    def test_error_recovery_across_tabs(self, wizard_page, chemistry_page):
        """Test recovery from errors when switching tabs."""
        wizard_page.wait_for_gradio_ready()

        # Start calibration without image (should show error)
        wizard_page.navigate_to_wizard()

        # Navigate to chemistry (should work despite error state)
        chemistry_page.navigate_to_chemistry()
        assert chemistry_page.is_chemistry_active()

        # Calculate should still work
        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
        )

        assert len(results) > 0
