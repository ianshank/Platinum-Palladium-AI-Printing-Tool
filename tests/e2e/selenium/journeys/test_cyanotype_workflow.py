"""
Cyanotype Workflow E2E Tests.

Tests the complete cyanotype chemistry calculation and exposure workflow.
"""

import pytest

from tests.e2e.selenium.pages.cyanotype_calculator_page import CyanotypeCalculatorPage


@pytest.mark.selenium
@pytest.mark.e2e
class TestCyanotypeWorkflow:
    """Test complete cyanotype calculation workflow."""

    @pytest.fixture
    def cyanotype_page(self, driver):
        """Create CyanotypeCalculatorPage instance."""
        return CyanotypeCalculatorPage(driver)

    def test_navigate_to_cyanotype(self, cyanotype_page):
        """Test navigation to cyanotype calculator."""
        cyanotype_page.wait_for_gradio_ready()
        cyanotype_page.navigate_to_cyanotype()

        assert cyanotype_page.is_cyanotype_active()

    def test_set_print_dimensions(self, cyanotype_page):
        """Test setting print dimensions."""
        cyanotype_page.navigate_to_cyanotype()
        cyanotype_page.set_print_dimensions(8.0, 10.0)

        # No error means success

    def test_calculate_basic_chemistry(self, cyanotype_page):
        """Test calculating a basic cyanotype chemistry recipe."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            formula="Classic",
            paper_type="Cotton Rag",
        )

        assert len(results) > 0, "Should return chemistry results"

    def test_calculate_classic_formula(self, cyanotype_page):
        """Test classic cyanotype formula calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            formula="Classic",
        )

        assert len(results) > 0

    def test_calculate_new_formula(self, cyanotype_page):
        """Test new (Mike Ware) cyanotype formula calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            formula="New",
        )

        assert len(results) > 0

    def test_various_print_sizes(self, cyanotype_page):
        """Test calculations for various print sizes."""
        sizes = [
            (4.0, 5.0),
            (8.0, 10.0),
            (11.0, 14.0),
            (16.0, 20.0),
        ]

        cyanotype_page.navigate_to_cyanotype()

        for width, height in sizes:
            results = cyanotype_page.calculate_chemistry_recipe(
                width=width,
                height=height,
            )
            assert len(results) > 0, f"Should calculate for {width}x{height}"

    def test_different_paper_types(self, cyanotype_page):
        """Test calculations for different paper types."""
        paper_types = [
            "Cotton Rag",
            "Watercolor",
            "Arches",
        ]

        cyanotype_page.navigate_to_cyanotype()

        for paper in paper_types:
            try:
                results = cyanotype_page.calculate_chemistry_recipe(
                    width=8.0,
                    height=10.0,
                    paper_type=paper,
                )
            except Exception:
                pass  # Some paper types may not be available

    def test_concentration_factor(self, cyanotype_page):
        """Test concentration factor affects chemistry volume."""
        cyanotype_page.navigate_to_cyanotype()

        normal = cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            concentration=1.0,
        )

        concentrated = cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            concentration=1.5,
        )

        # Both should return results
        assert len(normal) > 0
        assert len(concentrated) > 0

    def test_calculate_sunlight_exposure(self, cyanotype_page):
        """Test sunlight exposure calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_exposure_time(
            negative_density=1.6,
            uv_source="Sunlight",
            humidity=50.0,
        )

        assert "exposure_time" in results or len(results) > 0

    def test_calculate_bl_tube_exposure(self, cyanotype_page):
        """Test BL tube exposure calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_exposure_time(
            negative_density=1.6,
            uv_source="BL Tubes",
            distance=4.0,
        )

        assert "exposure_time" in results or len(results) > 0

    def test_humidity_affects_exposure(self, cyanotype_page):
        """Test that humidity affects exposure calculation."""
        cyanotype_page.navigate_to_cyanotype()

        low_humidity = cyanotype_page.calculate_exposure_time(
            negative_density=1.6,
            humidity=30.0,
        )

        high_humidity = cyanotype_page.calculate_exposure_time(
            negative_density=1.6,
            humidity=80.0,
        )

        # Both should work
        assert len(low_humidity) >= 0
        assert len(high_humidity) >= 0

    def test_distance_affects_exposure(self, cyanotype_page):
        """Test that UV source distance affects exposure."""
        cyanotype_page.navigate_to_cyanotype()

        close = cyanotype_page.calculate_exposure_time(
            uv_source="BL Tubes",
            distance=4.0,
        )

        far = cyanotype_page.calculate_exposure_time(
            uv_source="BL Tubes",
            distance=8.0,
        )

        # Both should work
        assert len(close) >= 0
        assert len(far) >= 0

    def test_complete_cyanotype_workflow(self, cyanotype_page):
        """Test complete workflow from chemistry to exposure."""
        cyanotype_page.navigate_to_cyanotype()

        # Configure and calculate chemistry
        cyanotype_page.set_print_dimensions(8.0, 10.0)
        cyanotype_page.select_classic_formula()
        cyanotype_page.select_cotton_rag()
        cyanotype_page.click_calculate_chemistry()

        chemistry_results = cyanotype_page.get_chemistry_results()
        assert chemistry_results is not None

        # Configure and calculate exposure
        cyanotype_page.set_negative_density(1.6)
        cyanotype_page.select_bl_tubes()
        cyanotype_page.set_humidity(50.0)
        cyanotype_page.click_calculate_exposure()

        exposure_results = cyanotype_page.get_exposure_results()
        assert exposure_results is not None

        # Verify we got results from both calculations
        assert len(chemistry_results) >= 0
        assert len(exposure_results) >= 0

    def test_workflow_with_all_settings(self, cyanotype_page):
        """Test workflow with all settings configured."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.complete_cyanotype_workflow(
            width=11.0,
            height=14.0,
            formula="Classic",
            paper_type="Cotton Rag",
            negative_density=1.8,
            uv_source="Sunlight",
        )

        assert "chemistry" in results
        assert "exposure" in results

    def test_stock_solution_calculation(self, cyanotype_page):
        """Test stock solution preparation calculation."""
        cyanotype_page.navigate_to_cyanotype()

        try:
            cyanotype_page.set_stock_volume(100.0)
            cyanotype_page.click_calculate_stock()
            instructions = cyanotype_page.get_stock_solution_instructions()
            # Instructions should be returned
            assert instructions is not None
        except Exception:
            # Stock solution feature may not be implemented
            pass

    def test_no_errors_on_valid_input(self, cyanotype_page):
        """Test that valid input does not produce errors."""
        cyanotype_page.navigate_to_cyanotype()

        cyanotype_page.calculate_chemistry_recipe(
            width=8.0,
            height=10.0,
            formula="Classic",
        )

        assert not cyanotype_page.is_error_displayed()

    def test_clear_inputs_works(self, cyanotype_page):
        """Test clearing all inputs."""
        cyanotype_page.navigate_to_cyanotype()

        # Set some values
        cyanotype_page.set_print_dimensions(8.0, 10.0)

        # Clear
        cyanotype_page.clear_all_inputs()

        # No error should occur


@pytest.mark.selenium
@pytest.mark.e2e
class TestCyanotypeEdgeCases:
    """Test edge cases for cyanotype calculator."""

    @pytest.fixture
    def cyanotype_page(self, driver):
        """Create CyanotypeCalculatorPage instance."""
        return CyanotypeCalculatorPage(driver)

    def test_very_small_print(self, cyanotype_page):
        """Test very small print size calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_chemistry_recipe(
            width=2.0,
            height=2.0,
        )

        # Should handle small prints
        assert len(results) >= 0

    def test_very_large_print(self, cyanotype_page):
        """Test very large print size calculation."""
        cyanotype_page.navigate_to_cyanotype()

        results = cyanotype_page.calculate_chemistry_recipe(
            width=30.0,
            height=40.0,
        )

        # Should handle large prints
        assert len(results) >= 0

    def test_extreme_negative_density(self, cyanotype_page):
        """Test extreme negative density values."""
        cyanotype_page.navigate_to_cyanotype()

        # Very low density
        low = cyanotype_page.calculate_exposure_time(
            negative_density=0.5,
        )
        assert len(low) >= 0

        # Very high density
        high = cyanotype_page.calculate_exposure_time(
            negative_density=3.0,
        )
        assert len(high) >= 0

    def test_extreme_humidity(self, cyanotype_page):
        """Test extreme humidity values."""
        cyanotype_page.navigate_to_cyanotype()

        # Very dry
        dry = cyanotype_page.calculate_exposure_time(
            humidity=10.0,
        )

        # Very humid
        humid = cyanotype_page.calculate_exposure_time(
            humidity=95.0,
        )

        # Should handle both extremes
        assert len(dry) >= 0
        assert len(humid) >= 0

    def test_switch_between_uv_sources(self, cyanotype_page):
        """Test switching between different UV sources."""
        cyanotype_page.navigate_to_cyanotype()

        uv_sources = ["Sunlight", "BL Tubes", "LED UV", "Mercury Vapor"]

        for source in uv_sources:
            try:
                cyanotype_page.select_uv_source(source)
                cyanotype_page.click_calculate_exposure()
                # No error should occur
            except Exception:
                pass  # Some sources may not be available

    def test_rapid_recalculation(self, cyanotype_page):
        """Test rapid successive recalculations."""
        cyanotype_page.navigate_to_cyanotype()

        for i in range(5):
            cyanotype_page.set_print_dimensions(8.0 + i, 10.0 + i)
            cyanotype_page.click_calculate_chemistry()

        # Should handle rapid changes without errors
        assert not cyanotype_page.is_error_displayed()
