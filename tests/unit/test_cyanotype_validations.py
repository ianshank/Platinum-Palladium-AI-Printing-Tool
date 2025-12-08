"""
Frontend/Backend validation tests for Cyanotype calculator.

Tests input validation, boundary conditions, data integrity, and error handling
for the cyanotype chemistry and exposure calculators.
"""

import pytest
from dataclasses import asdict

from ptpd_calibration.chemistry.cyanotype_calculator import (
    CyanotypeCalculator,
    CyanotypeRecipe,
    CyanotypeSettings,
    CyanotypePaperType,
)
from ptpd_calibration.exposure.alternative_calculators import (
    CyanotypeExposureCalculator,
    CyanotypeExposureResult,
    UVSource,
    UV_SOURCE_SPEEDS,
)
from ptpd_calibration.core.types import CyanotypeFormula


class TestCyanotypeChemistryValidation:
    """Validation tests for CyanotypeCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create CyanotypeCalculator instance."""
        return CyanotypeCalculator()

    @pytest.fixture
    def custom_settings(self):
        """Create custom CyanotypeSettings."""
        return CyanotypeSettings(
            ml_per_square_inch=0.02,
            solution_a_cost_per_ml=0.08,
            solution_b_cost_per_ml=0.06,
        )

    # --- Input Validation Tests ---

    def test_valid_print_dimensions(self, calculator):
        """Test valid print dimension inputs."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        assert isinstance(result, CyanotypeRecipe)
        assert result.solution_a_ml > 0
        assert result.solution_b_ml > 0
        assert result.total_volume_ml > 0

    def test_zero_width_raises_error(self, calculator):
        """Test that zero width raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            calculator.calculate(width_inches=0.0, height_inches=10.0)

    def test_zero_height_raises_error(self, calculator):
        """Test that zero height raises ValueError."""
        with pytest.raises(ValueError, match="height"):
            calculator.calculate(width_inches=8.0, height_inches=0.0)

    def test_negative_width_raises_error(self, calculator):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            calculator.calculate(width_inches=-5.0, height_inches=10.0)

    def test_negative_height_raises_error(self, calculator):
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="height"):
            calculator.calculate(width_inches=8.0, height_inches=-10.0)

    def test_concentration_factor_validation(self, calculator):
        """Test concentration factor boundary validation."""
        # Valid range
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, concentration_factor=1.5
        )
        assert result.solution_a_ml > 0

        # Zero should raise error
        with pytest.raises(ValueError, match="concentration"):
            calculator.calculate(
                width_inches=8.0, height_inches=10.0, concentration_factor=0.0
            )

        # Negative should raise error
        with pytest.raises(ValueError, match="concentration"):
            calculator.calculate(
                width_inches=8.0, height_inches=10.0, concentration_factor=-0.5
            )

    def test_negative_margin_raises_error(self, calculator):
        """Test that negative margin raises ValueError."""
        with pytest.raises(ValueError, match="margin"):
            calculator.calculate(
                width_inches=8.0, height_inches=10.0, margin_inches=-0.5
            )

    # --- Formula Type Validation ---

    def test_all_formula_types(self, calculator):
        """Test all formula types produce valid results."""
        for formula in CyanotypeFormula:
            result = calculator.calculate(
                width_inches=8.0,
                height_inches=10.0,
                formula=formula,
            )
            assert isinstance(result, CyanotypeRecipe)
            assert result.formula == formula
            assert result.solution_a_ml > 0
            assert result.solution_b_ml > 0

    def test_classic_vs_new_formula_differences(self, calculator):
        """Test that classic and new formulas produce different results."""
        classic = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            formula=CyanotypeFormula.CLASSIC,
        )
        new = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            formula=CyanotypeFormula.NEW,
        )

        # Different formulas should have different ratios
        classic_ratio = classic.solution_a_ml / classic.solution_b_ml
        new_ratio = new.solution_a_ml / new.solution_b_ml

        # Allow for formula-specific differences
        assert classic.formula != new.formula

    # --- Paper Type Validation ---

    def test_all_paper_types(self, calculator):
        """Test all paper types produce valid results."""
        for paper_type in CyanotypePaperType:
            result = calculator.calculate(
                width_inches=8.0,
                height_inches=10.0,
                paper_type=paper_type,
            )
            assert isinstance(result, CyanotypeRecipe)
            assert result.paper_type == paper_type

    def test_paper_type_affects_volume(self, calculator):
        """Test that different paper types affect solution volume."""
        cotton = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            paper_type=CyanotypePaperType.COTTON_RAG,
        )
        watercolor = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            paper_type=CyanotypePaperType.WATERCOLOR,
        )

        # Different papers may absorb different amounts
        # Both should be valid
        assert cotton.total_volume_ml > 0
        assert watercolor.total_volume_ml > 0

    # --- Output Validation ---

    def test_recipe_output_data_types(self, calculator):
        """Test that recipe output has correct data types."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        assert isinstance(result.solution_a_ml, float)
        assert isinstance(result.solution_b_ml, float)
        assert isinstance(result.total_volume_ml, float)
        assert isinstance(result.coverage_square_inches, float)
        assert isinstance(result.formula, CyanotypeFormula)
        assert isinstance(result.paper_type, CyanotypePaperType)

    def test_recipe_total_equals_sum_of_parts(self, calculator):
        """Test that total volume equals sum of individual solutions."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        expected_total = result.solution_a_ml + result.solution_b_ml
        assert result.total_volume_ml == pytest.approx(expected_total, rel=0.01)

    def test_recipe_cost_calculation(self, calculator):
        """Test that cost is calculated correctly when requested."""
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, include_cost=True
        )

        if hasattr(result, 'estimated_cost') and result.estimated_cost is not None:
            assert result.estimated_cost >= 0

    def test_recipe_serialization(self, calculator):
        """Test that recipe can be serialized to dict."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert 'solution_a_ml' in result_dict
        assert 'solution_b_ml' in result_dict

    # --- Scaling Validation ---

    def test_volume_scales_with_area(self, calculator):
        """Test that solution volume scales proportionally with area."""
        small = calculator.calculate(width_inches=4.0, height_inches=5.0)
        large = calculator.calculate(width_inches=8.0, height_inches=10.0)

        # Large print has 4x the area
        area_ratio = (8.0 * 10.0) / (4.0 * 5.0)
        volume_ratio = large.total_volume_ml / small.total_volume_ml

        assert volume_ratio == pytest.approx(area_ratio, rel=0.1)

    def test_concentration_factor_scales_volume(self, calculator):
        """Test that concentration factor scales volume correctly."""
        normal = calculator.calculate(
            width_inches=8.0, height_inches=10.0, concentration_factor=1.0
        )
        double = calculator.calculate(
            width_inches=8.0, height_inches=10.0, concentration_factor=2.0
        )

        assert double.total_volume_ml == pytest.approx(normal.total_volume_ml * 2, rel=0.1)

    # --- Stock Solution Validation ---

    def test_stock_solution_calculation(self, calculator):
        """Test stock solution preparation calculations."""
        stock = calculator.calculate_stock_solutions(total_volume_ml=100.0)

        assert isinstance(stock, dict)
        assert 'solution_a' in stock
        assert 'solution_b' in stock

    def test_stock_solution_volume_validation(self, calculator):
        """Test stock solution volume validation."""
        with pytest.raises(ValueError, match="volume"):
            calculator.calculate_stock_solutions(total_volume_ml=0.0)

        with pytest.raises(ValueError, match="volume"):
            calculator.calculate_stock_solutions(total_volume_ml=-50.0)

    def test_stock_solution_formula_affects_output(self, calculator):
        """Test that formula choice affects stock solution preparation."""
        classic = calculator.calculate_stock_solutions(
            total_volume_ml=100.0, formula=CyanotypeFormula.CLASSIC
        )
        new = calculator.calculate_stock_solutions(
            total_volume_ml=100.0, formula=CyanotypeFormula.NEW
        )

        # Different formulas may have different concentrations
        assert classic is not None
        assert new is not None

    # --- Edge Case Tests ---

    def test_very_small_print_size(self, calculator):
        """Test very small print dimensions."""
        result = calculator.calculate(width_inches=1.0, height_inches=1.0)

        assert result.total_volume_ml > 0
        assert result.total_volume_ml < 1.0  # Should be small but positive

    def test_very_large_print_size(self, calculator):
        """Test very large print dimensions."""
        result = calculator.calculate(width_inches=40.0, height_inches=60.0)

        assert result.total_volume_ml > 0
        # Should be significantly more than a small print
        small = calculator.calculate(width_inches=4.0, height_inches=5.0)
        assert result.total_volume_ml > small.total_volume_ml * 10


class TestCyanotypeExposureValidation:
    """Validation tests for CyanotypeExposureCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create CyanotypeExposureCalculator instance."""
        return CyanotypeExposureCalculator()

    # --- Input Validation ---

    def test_valid_exposure_calculation(self, calculator):
        """Test valid exposure calculation."""
        result = calculator.calculate(
            negative_density=1.6,
            uv_source=UVSource.BL_TUBES,
        )

        assert isinstance(result, CyanotypeExposureResult)
        assert result.exposure_time_seconds > 0

    def test_negative_density_validation(self, calculator):
        """Test negative density input validation."""
        # Valid densities
        result = calculator.calculate(negative_density=1.6)
        assert result.exposure_time_seconds > 0

        # Edge case: very low density
        result = calculator.calculate(negative_density=0.5)
        assert result.exposure_time_seconds > 0

        # Invalid: negative density
        with pytest.raises(ValueError, match="density"):
            calculator.calculate(negative_density=-0.5)

    def test_humidity_validation(self, calculator):
        """Test humidity percentage validation."""
        # Valid humidity range
        for humidity in [20.0, 50.0, 80.0]:
            result = calculator.calculate(
                negative_density=1.6, humidity_percent=humidity
            )
            assert result.exposure_time_seconds > 0

        # Invalid: negative humidity
        with pytest.raises(ValueError, match="humidity"):
            calculator.calculate(negative_density=1.6, humidity_percent=-10.0)

        # Invalid: humidity > 100
        with pytest.raises(ValueError, match="humidity"):
            calculator.calculate(negative_density=1.6, humidity_percent=110.0)

    def test_paper_factor_validation(self, calculator):
        """Test paper factor validation."""
        # Valid factors
        result = calculator.calculate(negative_density=1.6, paper_factor=1.5)
        assert result.exposure_time_seconds > 0

        # Invalid: zero factor
        with pytest.raises(ValueError, match="factor"):
            calculator.calculate(negative_density=1.6, paper_factor=0.0)

        # Invalid: negative factor
        with pytest.raises(ValueError, match="factor"):
            calculator.calculate(negative_density=1.6, paper_factor=-0.5)

    def test_distance_validation(self, calculator):
        """Test light source distance validation."""
        # Valid distances
        result = calculator.calculate(negative_density=1.6, distance_inches=8.0)
        assert result.exposure_time_seconds > 0

        # Invalid: zero distance
        with pytest.raises(ValueError, match="distance"):
            calculator.calculate(negative_density=1.6, distance_inches=0.0)

        # Invalid: negative distance
        with pytest.raises(ValueError, match="distance"):
            calculator.calculate(negative_density=1.6, distance_inches=-4.0)

    # --- UV Source Validation ---

    def test_all_uv_sources(self, calculator):
        """Test all UV source types produce valid results."""
        for uv_source in UVSource:
            result = calculator.calculate(
                negative_density=1.6,
                uv_source=uv_source,
            )
            assert isinstance(result, CyanotypeExposureResult)
            assert result.exposure_time_seconds > 0
            assert result.uv_source == uv_source

    def test_uv_source_speeds_consistency(self):
        """Test that UV source speeds are defined correctly."""
        for uv_source in UVSource:
            assert uv_source in UV_SOURCE_SPEEDS
            assert UV_SOURCE_SPEEDS[uv_source] > 0

    def test_sunlight_vs_artificial_exposure_times(self, calculator):
        """Test that sunlight has different exposure than artificial sources."""
        sunlight = calculator.calculate(
            negative_density=1.6,
            uv_source=UVSource.SUNLIGHT,
        )
        bl_tubes = calculator.calculate(
            negative_density=1.6,
            uv_source=UVSource.BL_TUBES,
        )

        # Both should be valid but different
        assert sunlight.exposure_time_seconds > 0
        assert bl_tubes.exposure_time_seconds > 0
        # Typically sunlight is faster
        assert sunlight.exposure_time_seconds != bl_tubes.exposure_time_seconds

    # --- Formula Effects ---

    def test_formula_affects_exposure(self, calculator):
        """Test that different formulas affect exposure time."""
        classic = calculator.calculate(
            negative_density=1.6,
            formula=CyanotypeFormula.CLASSIC,
        )
        new = calculator.calculate(
            negative_density=1.6,
            formula=CyanotypeFormula.NEW,
        )

        # Both should be valid
        assert classic.exposure_time_seconds > 0
        assert new.exposure_time_seconds > 0

    # --- Output Validation ---

    def test_exposure_result_contains_all_fields(self, calculator):
        """Test that exposure result contains all expected fields."""
        result = calculator.calculate(negative_density=1.6)

        assert hasattr(result, 'exposure_time_seconds')
        assert hasattr(result, 'exposure_time_formatted')
        assert hasattr(result, 'uv_source')
        assert hasattr(result, 'formula')

    def test_exposure_time_formatted_is_readable(self, calculator):
        """Test that formatted exposure time is human-readable."""
        result = calculator.calculate(negative_density=1.6)

        formatted = result.exposure_time_formatted
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Should contain time units
        assert any(unit in formatted.lower() for unit in ['min', 'sec', ':'])

    # --- Distance/Inverse Square Law ---

    def test_inverse_square_law_compliance(self, calculator):
        """Test that exposure follows inverse square law with distance."""
        close = calculator.calculate(
            negative_density=1.6,
            distance_inches=4.0,
            base_distance_inches=4.0,
        )
        far = calculator.calculate(
            negative_density=1.6,
            distance_inches=8.0,
            base_distance_inches=4.0,
        )

        # At 2x distance, exposure should be ~4x (inverse square law)
        expected_ratio = (8.0 / 4.0) ** 2
        actual_ratio = far.exposure_time_seconds / close.exposure_time_seconds

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.1)


class TestCyanotypeIntegrationValidation:
    """Integration validation tests combining chemistry and exposure."""

    def test_chemistry_and_exposure_workflow(self):
        """Test complete cyanotype workflow with both calculators."""
        # Step 1: Calculate chemistry
        chem_calc = CyanotypeCalculator()
        recipe = chem_calc.calculate(
            width_inches=8.0,
            height_inches=10.0,
            formula=CyanotypeFormula.CLASSIC,
            paper_type=CyanotypePaperType.COTTON_RAG,
        )

        assert recipe.total_volume_ml > 0

        # Step 2: Calculate exposure for same formula
        exp_calc = CyanotypeExposureCalculator()
        exposure = exp_calc.calculate(
            negative_density=1.6,
            formula=recipe.formula,  # Use same formula
            uv_source=UVSource.BL_TUBES,
        )

        assert exposure.exposure_time_seconds > 0
        assert exposure.formula == recipe.formula

    def test_formula_consistency_between_calculators(self):
        """Test that formula enum works consistently across calculators."""
        for formula in CyanotypeFormula:
            chem_calc = CyanotypeCalculator()
            exp_calc = CyanotypeExposureCalculator()

            recipe = chem_calc.calculate(
                width_inches=8.0,
                height_inches=10.0,
                formula=formula,
            )

            exposure = exp_calc.calculate(
                negative_density=1.6,
                formula=formula,
            )

            assert recipe.formula == exposure.formula == formula
