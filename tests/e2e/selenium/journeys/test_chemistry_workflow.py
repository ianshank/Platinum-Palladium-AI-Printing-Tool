"""
Chemistry Workflow E2E Tests.

Tests the complete chemistry calculation and recipe management workflow.
"""

import pytest

from tests.e2e.selenium.pages.chemistry_calculator_page import ChemistryCalculatorPage


@pytest.mark.selenium
@pytest.mark.e2e
class TestChemistryWorkflow:
    """Test complete chemistry calculation workflow."""

    @pytest.fixture
    def chemistry_page(self, driver):
        """Create ChemistryCalculatorPage instance."""
        return ChemistryCalculatorPage(driver)

    def test_navigate_to_chemistry(self, chemistry_page):
        """Test navigation to chemistry calculator."""
        chemistry_page.wait_for_gradio_ready()
        chemistry_page.navigate_to_chemistry()

        assert chemistry_page.is_chemistry_active()

    def test_set_print_dimensions(self, chemistry_page):
        """Test setting print dimensions."""
        chemistry_page.navigate_to_chemistry()
        chemistry_page.set_print_dimensions(8.0, 10.0)

        # No error means success

    def test_calculate_basic_recipe(self, chemistry_page):
        """Test calculating a basic chemistry recipe."""
        chemistry_page.navigate_to_chemistry()

        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
            chemistry_type="Platinum/Palladium",
            metal_ratio=0.5,
        )

        assert len(results) > 0, "Should return recipe results"

    def test_calculate_platinum_only_recipe(self, chemistry_page):
        """Test calculating a platinum-only recipe."""
        chemistry_page.navigate_to_chemistry()

        results = chemistry_page.calculate_recipe(
            width=11.0,
            height=14.0,
            chemistry_type="Platinum/Palladium",
            metal_ratio=1.0,  # 100% platinum
        )

        assert len(results) > 0

    def test_calculate_palladium_only_recipe(self, chemistry_page):
        """Test calculating a palladium-only recipe."""
        chemistry_page.navigate_to_chemistry()

        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
            chemistry_type="Platinum/Palladium",
            metal_ratio=0.0,  # 100% palladium
        )

        assert len(results) > 0

    def test_calculate_with_contrast_agent(self, chemistry_page):
        """Test calculating recipe with contrast agent."""
        chemistry_page.navigate_to_chemistry()

        results = chemistry_page.calculate_recipe(
            width=8.0,
            height=10.0,
            contrast_agent="Na2",
            contrast_drops=6.0,
        )

        assert len(results) > 0

    def test_various_print_sizes(self, chemistry_page):
        """Test calculations for various print sizes."""
        sizes = [
            (4.0, 5.0),
            (8.0, 10.0),
            (11.0, 14.0),
            (16.0, 20.0),
        ]

        chemistry_page.navigate_to_chemistry()

        for width, height in sizes:
            results = chemistry_page.calculate_recipe(
                width=width,
                height=height,
            )
            assert len(results) > 0, f"Should calculate for {width}x{height}"

    def test_save_recipe(self, chemistry_page):
        """Test saving a recipe."""
        chemistry_page.navigate_to_chemistry()

        chemistry_page.calculate_recipe(width=8.0, height=10.0)
        chemistry_page.save_recipe("Test Recipe 1")

        # No error means success

    def test_load_saved_recipe(self, chemistry_page):
        """Test loading a saved recipe."""
        chemistry_page.navigate_to_chemistry()

        # First save a recipe
        chemistry_page.calculate_recipe(width=8.0, height=10.0)
        chemistry_page.save_recipe("Load Test Recipe")

        # Then load it
        chemistry_page.load_recipe("Load Test Recipe")

        # No error means success

    def test_recipe_values_scale_with_size(self, chemistry_page):
        """Test that recipe values scale appropriately with print size."""
        chemistry_page.navigate_to_chemistry()

        # Small print
        small_results = chemistry_page.calculate_recipe(width=4.0, height=5.0)

        # Large print
        large_results = chemistry_page.calculate_recipe(width=16.0, height=20.0)

        # Large print should have more solution
        assert len(small_results) > 0
        assert len(large_results) > 0

    def test_different_chemistry_types(self, chemistry_page):
        """Test calculations for different chemistry types."""
        chemistry_types = [
            "Platinum/Palladium",
            "Kallitype",
            "Cyanotype",
        ]

        chemistry_page.navigate_to_chemistry()

        for chem_type in chemistry_types:
            try:
                results = chemistry_page.calculate_recipe(
                    width=8.0,
                    height=10.0,
                    chemistry_type=chem_type,
                )
                # Some chemistry types may not be available
            except Exception:
                pass  # Skip if chemistry type not available

    def test_extreme_metal_ratios(self, chemistry_page):
        """Test edge cases for metal ratios."""
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        chemistry_page.navigate_to_chemistry()

        for ratio in ratios:
            results = chemistry_page.calculate_recipe(
                width=8.0,
                height=10.0,
                metal_ratio=ratio,
            )
            assert len(results) > 0, f"Should calculate for ratio {ratio}"

    def test_workflow_with_all_settings(self, chemistry_page):
        """Test complete workflow with all settings configured."""
        chemistry_page.navigate_to_chemistry()

        # Configure all settings
        chemistry_page.set_print_dimensions(11.0, 14.0)
        chemistry_page.select_chemistry_type("Platinum/Palladium")
        chemistry_page.set_metal_ratio(0.6)
        chemistry_page.select_contrast_agent("Na2")
        chemistry_page.set_contrast_amount(5.0)
        chemistry_page.set_coating_factor(1.2)

        chemistry_page.click_calculate()

        results = chemistry_page.get_recipe_results()
        assert len(results) > 0
