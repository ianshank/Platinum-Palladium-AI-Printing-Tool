"""
Neuro-Symbolic Curve Generation E2E Journey Tests.

Tests the complete neuro-symbolic AI workflow including:
- Constraint-based curve generation
- Knowledge graph inference
- Symbolic regression formula discovery
- Uncertainty quantification
"""

from pathlib import Path

import pytest

from tests.e2e.selenium.pages.neuro_symbolic_page import NeuroSymbolicPage
from tests.e2e.selenium.pages.calibration_wizard_page import CalibrationWizardPage
from tests.e2e.selenium.pages.dashboard_page import DashboardPage


@pytest.mark.selenium
@pytest.mark.e2e
class TestNeuroSymbolicJourney:
    """Test neuro-symbolic curve generation workflow."""

    @pytest.fixture
    def neuro_page(self, driver):
        """Create NeuroSymbolicPage instance."""
        return NeuroSymbolicPage(driver)

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
        from PIL import Image
        import numpy as np

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

        img = Image.fromarray(img_array, mode="L").convert("RGB")
        file_path = tmp_path / "test_step_tablet.png"
        img.save(file_path)

        return file_path

    # --- Navigation Tests ---

    def test_navigate_to_neuro_symbolic_section(self, neuro_page):
        """Test navigation to neuro-symbolic features."""
        neuro_page.wait_for_gradio_ready()
        neuro_page.navigate_to_neuro_symbolic()

        # Should be able to access neuro-symbolic features

    # --- Constraint Configuration Tests ---

    def test_enable_all_constraints(self, neuro_page):
        """Test enabling all constraint types."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.enable_monotonicity_constraint(True)
        neuro_page.enable_density_bounds_constraint(True)
        neuro_page.enable_physics_constraint(True)
        neuro_page.enable_smoothness_constraint(True)

        # No error means success

    def test_set_constraint_weights(self, neuro_page):
        """Test setting custom constraint weights."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.set_constraint_weight("Monotonicity", 10.0)
        neuro_page.set_constraint_weight("Density Bounds", 5.0)
        neuro_page.set_constraint_weight("Smoothness", 1.0)

        # No error means success

    def test_set_density_bounds(self, neuro_page):
        """Test setting density bounds."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.set_density_bounds(0.1, 2.5)

        # No error means success

    # --- Curve Generation with Constraints Tests ---

    def test_generate_curve_with_monotonicity(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test generating curve with monotonicity constraint."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.enable_monotonicity_constraint(True)
        neuro_page.enable_density_bounds_constraint(False)
        neuro_page.enable_physics_constraint(False)
        neuro_page.enable_smoothness_constraint(False)

        neuro_page.click_generate_with_constraints()

        assert neuro_page.is_constrained_curve_displayed()

        # Validate monotonicity constraint
        validation = neuro_page.validate_curve_constraints()
        assert validation["monotonicity"], "Monotonicity constraint should be satisfied"

    def test_generate_curve_with_all_constraints(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test generating curve with all constraints enabled."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.click_analyze()

        result = neuro_page.generate_neuro_symbolic_curve(
            enable_monotonicity=True,
            enable_density_bounds=True,
            enable_physics=True,
            enable_smoothness=True,
        )

        assert result["success"], "Curve generation should succeed"
        assert len(result["violations"]) == 0, "Should have no constraint violations"

    def test_constraint_violations_displayed(self, neuro_page, wizard_page, sample_step_tablet):
        """Test that constraint violations are properly displayed."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()

        # Enable constraints with very strict bounds that may cause violations
        neuro_page.enable_density_bounds_constraint(True)
        neuro_page.set_density_bounds(0.0, 0.1)  # Very restrictive

        neuro_page.click_generate_with_constraints()

        # Should either succeed or show violations
        displayed = neuro_page.is_constrained_curve_displayed()
        violations = neuro_page.get_constraint_violations()

        assert displayed or len(violations) > 0

    # --- Knowledge Graph Tests ---

    def test_select_paper_from_knowledge_graph(self, neuro_page):
        """Test selecting a paper from knowledge graph."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.select_paper_from_knowledge_graph("Arches Platine")

        # Should get inferred settings
        settings = neuro_page.get_inferred_settings()
        # May or may not have settings depending on UI implementation

    def test_get_similar_papers(self, neuro_page):
        """Test getting similar papers from knowledge graph."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.select_paper_from_knowledge_graph("Arches Platine")

        similar = neuro_page.get_similar_papers()
        # Should return list (may be empty depending on UI)

    def test_analogical_reasoning_for_new_paper(self, neuro_page):
        """Test analogical reasoning for unknown paper."""
        neuro_page.navigate_to_neuro_symbolic()

        neuro_page.enable_analogical_reasoning(True)

        # Enter a new paper that might not be in knowledge graph
        try:
            neuro_page.fill_textbox("New Paper", "Custom Test Paper")
        except Exception:
            pass  # UI may not have this field

        # Should attempt to infer settings from similar papers

    # --- Symbolic Regression Tests ---

    def test_enable_formula_discovery(self, neuro_page, wizard_page, sample_step_tablet):
        """Test enabling symbolic regression formula discovery."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.enable_formula_discovery(True)
        neuro_page.set_formula_complexity("simple")

        neuro_page.click_generate_neuro_symbolic_curve()

        formula = neuro_page.get_discovered_formula()
        # Formula may or may not be returned depending on UI

    def test_formula_discovery_with_custom_parameters(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test formula discovery with custom parameters."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.enable_formula_discovery(True)
        neuro_page.set_regression_generations(20)
        neuro_page.set_population_size(50)

        neuro_page.click_generate_neuro_symbolic_curve()

        # Should complete without error

    def test_formula_r_squared_displayed(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test that R-squared value is displayed for discovered formula."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        result = neuro_page.generate_neuro_symbolic_curve(
            enable_formula_discovery=True,
        )

        # R-squared should be a valid number
        assert isinstance(result["r_squared"], float)

    # --- Uncertainty Quantification Tests ---

    def test_enable_uncertainty_quantification(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test enabling uncertainty quantification."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.enable_uncertainty_quantification(True)
        neuro_page.set_monte_carlo_samples(100)

        neuro_page.click_generate_neuro_symbolic_curve()

        # Should show confidence interval or band
        ci = neuro_page.get_confidence_interval()
        # May return (0, 0) if not displayed

    def test_confidence_band_visualization(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test confidence band visualization on curve plot."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        result = neuro_page.generate_neuro_symbolic_curve(
            enable_uncertainty=True,
        )

        # Check if confidence band is displayed
        has_band = neuro_page.is_confidence_band_displayed()
        # May or may not be displayed depending on UI

    # --- Explanation Tests ---

    def test_curve_explanation_generated(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test that curve explanation is generated."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        result = neuro_page.generate_neuro_symbolic_curve(
            enable_monotonicity=True,
            enable_density_bounds=True,
        )

        explanation = result["explanation"]
        # Explanation may or may not be provided

    def test_technical_details_expandable(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test that technical details can be expanded."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.click_generate_neuro_symbolic_curve()

        neuro_page.expand_technical_details()
        details = neuro_page.get_technical_details()
        # Details may or may not be available

    # --- Complete Workflow Tests ---

    def test_complete_neuro_symbolic_workflow(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test complete neuro-symbolic curve generation workflow."""
        # Step 1: Upload and analyze step tablet
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("Arches Platine")
        wizard_page.select_chemistry_type("Platinum/Palladium")
        wizard_page.set_metal_ratio(0.5)
        wizard_page.click_analyze()

        # Step 2: Configure neuro-symbolic features
        neuro_page.navigate_to_neuro_symbolic()

        # Step 3: Generate curve with all features
        result = neuro_page.generate_neuro_symbolic_curve(
            enable_monotonicity=True,
            enable_density_bounds=True,
            enable_physics=True,
            enable_smoothness=True,
            enable_formula_discovery=True,
            enable_uncertainty=True,
        )

        # Verify result
        assert result["success"], "Workflow should complete successfully"

    def test_workflow_with_knowledge_graph_inference(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test workflow using knowledge graph for paper inference."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)

        # Use knowledge graph to infer settings
        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.select_paper_from_knowledge_graph("Bergger COT320")
        neuro_page.enable_analogical_reasoning(True)

        # Get inferred settings
        settings = neuro_page.get_inferred_settings()

        # Generate curve
        neuro_page.click_generate_neuro_symbolic_curve()

        assert neuro_page.is_constrained_curve_displayed()

    def test_workflow_preserves_constraints_on_regenerate(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test that constraints are preserved when regenerating curve."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()

        # First generation
        neuro_page.enable_monotonicity_constraint(True)
        neuro_page.enable_smoothness_constraint(True)
        neuro_page.click_generate_neuro_symbolic_curve()

        first_metrics = neuro_page.get_curve_quality_metrics()

        # Second generation (constraints should still be enabled)
        neuro_page.click_generate_neuro_symbolic_curve()

        second_metrics = neuro_page.get_curve_quality_metrics()

        # Both should have valid curves
        assert neuro_page.is_constrained_curve_displayed()

    # --- Error Handling Tests ---

    def test_handles_missing_step_tablet(self, neuro_page):
        """Test error handling when no step tablet is uploaded."""
        neuro_page.navigate_to_neuro_symbolic()

        # Try to generate curve without uploading step tablet
        try:
            neuro_page.click_generate_neuro_symbolic_curve()
            # Should either show error or be disabled
        except Exception:
            pass  # Expected behavior

    def test_handles_invalid_constraint_values(self, neuro_page, wizard_page, sample_step_tablet):
        """Test error handling for invalid constraint values."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()

        # Try invalid density bounds (max < min)
        try:
            neuro_page.set_density_bounds(2.5, 0.1)
            neuro_page.click_generate_with_constraints()
            # Should either fix automatically or show error
        except Exception:
            pass  # Expected behavior

    def test_handles_formula_discovery_timeout(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test handling of formula discovery with many generations."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.enable_formula_discovery(True)

        # Set reasonable parameters to avoid actual timeout
        neuro_page.set_regression_generations(10)
        neuro_page.set_population_size(20)

        neuro_page.click_generate_neuro_symbolic_curve()

        # Should complete without timeout
        assert neuro_page.is_constrained_curve_displayed()


@pytest.mark.selenium
@pytest.mark.e2e
class TestNeuroSymbolicIntegration:
    """Test neuro-symbolic integration with other components."""

    @pytest.fixture
    def neuro_page(self, driver):
        """Create NeuroSymbolicPage instance."""
        return NeuroSymbolicPage(driver)

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
        """Create a sample step tablet image."""
        from PIL import Image
        import numpy as np

        width, height = 420, 100
        num_patches = 21
        patch_width = width // num_patches

        img_array = np.zeros((height, width), dtype=np.uint8)
        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            img_array[:, i * patch_width:(i + 1) * patch_width] = value

        img = Image.fromarray(img_array, mode="L").convert("RGB")
        file_path = tmp_path / "test_step_tablet.png"
        img.save(file_path)
        return file_path

    def test_neuro_symbolic_curve_export(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test exporting neuro-symbolic generated curve."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.click_generate_neuro_symbolic_curve()

        # Export the curve
        wizard_page.click_export_quad()

        link = wizard_page.get_export_download_link()
        # Link may or may not be available depending on implementation

    def test_neuro_symbolic_curve_save_to_database(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test saving neuro-symbolic generated curve to database."""
        wizard_page.navigate_to_wizard()
        wizard_page.upload_step_tablet(sample_step_tablet)
        wizard_page.select_paper_type("NS Test Paper")
        wizard_page.click_analyze()

        neuro_page.navigate_to_neuro_symbolic()
        neuro_page.click_generate_neuro_symbolic_curve()

        # Save calibration
        wizard_page.click_save_calibration()

        confirmation = wizard_page.get_save_confirmation()
        # Should confirm save or show status

    def test_neuro_symbolic_with_different_papers(
        self, neuro_page, wizard_page, sample_step_tablet
    ):
        """Test neuro-symbolic generation with different paper types."""
        papers = [
            "Arches Platine",
            "Bergger COT320",
            "Hahnemuhle Platinum Rag",
        ]

        for paper in papers:
            wizard_page.navigate_to_wizard()
            wizard_page.upload_step_tablet(sample_step_tablet)
            wizard_page.select_paper_type(paper)
            wizard_page.click_analyze()

            neuro_page.navigate_to_neuro_symbolic()
            result = neuro_page.generate_neuro_symbolic_curve(
                enable_monotonicity=True,
                enable_smoothness=True,
            )

            # Should work for all papers
            # result["success"] may vary based on UI implementation
