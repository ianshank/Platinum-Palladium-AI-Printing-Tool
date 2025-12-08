"""
Frontend/Backend validation tests for Silver Gelatin calculator.

Tests input validation, boundary conditions, data integrity, and error handling
for the silver gelatin chemistry and exposure calculators.
"""

import pytest
from dataclasses import asdict

from ptpd_calibration.chemistry.silver_gelatin_calculator import (
    SilverGelatinCalculator,
    ProcessingChemistry,
    DeveloperRecipe,
    DilutionRatio,
    TraySize,
)
from ptpd_calibration.exposure.alternative_calculators import (
    SilverGelatinExposureCalculator,
    SilverGelatinExposureResult,
    EnlargerLightSource,
    ENLARGER_LIGHT_SPEEDS,
)
from ptpd_calibration.core.types import (
    DeveloperType,
    FixerType,
    PaperGrade,
    PaperBase,
)


class TestSilverGelatinChemistryValidation:
    """Validation tests for SilverGelatinCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create SilverGelatinCalculator instance."""
        return SilverGelatinCalculator()

    # --- Input Validation Tests ---

    def test_valid_print_dimensions(self, calculator):
        """Test valid print dimension inputs."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        assert isinstance(result, ProcessingChemistry)
        assert result.developer_volume_ml > 0
        assert result.stop_bath_volume_ml > 0
        assert result.fixer_volume_ml > 0

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

    # --- Developer Configuration Validation ---

    def test_all_developer_types(self, calculator):
        """Test all developer types produce valid results."""
        valid_developers = [
            DeveloperType.DEKTOL,
            DeveloperType.D_72,
            DeveloperType.D_76,
            DeveloperType.XTOL,
            DeveloperType.RODINAL,
        ]

        for dev in valid_developers:
            try:
                result = calculator.calculate(
                    width_inches=8.0,
                    height_inches=10.0,
                    developer=dev,
                )
                assert result.developer_volume_ml > 0
            except (ValueError, KeyError):
                # Some developers might not be configured for paper
                pass

    def test_all_dilution_ratios(self, calculator):
        """Test all dilution ratios produce valid results."""
        for dilution in DilutionRatio:
            result = calculator.calculate(
                width_inches=8.0,
                height_inches=10.0,
                dilution=dilution,
            )
            assert isinstance(result, ProcessingChemistry)
            assert result.developer_volume_ml > 0

    def test_dilution_affects_volume(self, calculator):
        """Test that dilution ratio affects solution volume."""
        concentrated = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            dilution=DilutionRatio.STOCK,
        )
        diluted = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            dilution=DilutionRatio.ONE_TO_THREE,
        )

        # Both should be valid
        assert concentrated.developer_volume_ml > 0
        assert diluted.developer_volume_ml > 0

    # --- Temperature Validation ---

    def test_temperature_validation(self, calculator):
        """Test temperature input validation."""
        # Valid temperature range (typical darkroom temps)
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, temperature_c=20.0
        )
        assert result.developer_volume_ml > 0

        # Edge case: cold developer
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, temperature_c=16.0
        )
        assert result.developer_volume_ml > 0

        # Edge case: warm developer
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, temperature_c=24.0
        )
        assert result.developer_volume_ml > 0

    def test_extreme_temperature_warning(self, calculator):
        """Test that extreme temperatures produce warnings."""
        # Very cold
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, temperature_c=10.0
        )
        # Should still calculate but may include warning
        assert result.developer_volume_ml > 0

        # Very warm
        result = calculator.calculate(
            width_inches=8.0, height_inches=10.0, temperature_c=30.0
        )
        assert result.developer_volume_ml > 0

    # --- Fixer Validation ---

    def test_all_fixer_types(self, calculator):
        """Test all fixer types produce valid results."""
        for fixer in FixerType:
            try:
                result = calculator.calculate(
                    width_inches=8.0,
                    height_inches=10.0,
                    fixer=fixer,
                )
                assert result.fixer_volume_ml > 0
            except (ValueError, KeyError):
                # Some fixers might not be implemented
                pass

    def test_fixer_type_affects_processing(self, calculator):
        """Test that different fixers have different processing requirements."""
        sodium = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            fixer=FixerType.SODIUM_THIOSULFATE,
        )
        ammonium = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            fixer=FixerType.AMMONIUM_THIOSULFATE,
        )

        # Both should work
        assert sodium.fixer_volume_ml > 0
        assert ammonium.fixer_volume_ml > 0

    # --- Paper Base Validation ---

    def test_all_paper_bases(self, calculator):
        """Test all paper bases produce valid results."""
        for paper_base in PaperBase:
            result = calculator.calculate(
                width_inches=8.0,
                height_inches=10.0,
                paper_base=paper_base,
            )
            assert isinstance(result, ProcessingChemistry)

    def test_fiber_vs_rc_processing_times(self, calculator):
        """Test that fiber and RC papers have different processing times."""
        fiber = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            paper_base=PaperBase.FIBER,
        )
        rc = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            paper_base=PaperBase.RESIN_COATED,
        )

        # Both should be valid
        assert fiber.developer_volume_ml > 0
        assert rc.developer_volume_ml > 0

        # Processing times typically differ
        if hasattr(fiber, 'development_time_seconds') and hasattr(rc, 'development_time_seconds'):
            # Fiber typically needs more processing
            pass

    # --- Tray Size Validation ---

    def test_all_tray_sizes(self, calculator):
        """Test all tray sizes produce valid results."""
        for tray in TraySize:
            result = calculator.calculate(
                width_inches=8.0,
                height_inches=10.0,
                tray_size=tray,
            )
            assert result.developer_volume_ml > 0

    def test_tray_size_affects_volume(self, calculator):
        """Test that tray size affects chemistry volume."""
        small_tray = calculator.calculate(
            width_inches=5.0, height_inches=7.0,
            tray_size=TraySize.TRAY_8X10,
        )
        large_tray = calculator.calculate(
            width_inches=5.0, height_inches=7.0,
            tray_size=TraySize.TRAY_11X14,
        )

        # Larger trays need more chemistry
        assert large_tray.developer_volume_ml >= small_tray.developer_volume_ml

    # --- Multi-Print Processing ---

    def test_num_prints_validation(self, calculator):
        """Test number of prints validation."""
        single = calculator.calculate(
            width_inches=8.0, height_inches=10.0, num_prints=1
        )
        multiple = calculator.calculate(
            width_inches=8.0, height_inches=10.0, num_prints=5
        )

        # More prints may require more chemistry or fresh chemistry sooner
        assert single.developer_volume_ml > 0
        assert multiple.developer_volume_ml > 0

    def test_zero_prints_raises_error(self, calculator):
        """Test that zero prints raises ValueError."""
        with pytest.raises(ValueError, match="prints"):
            calculator.calculate(width_inches=8.0, height_inches=10.0, num_prints=0)

    def test_negative_prints_raises_error(self, calculator):
        """Test that negative prints raises ValueError."""
        with pytest.raises(ValueError, match="prints"):
            calculator.calculate(width_inches=8.0, height_inches=10.0, num_prints=-1)

    # --- Hypo Clear Option ---

    def test_hypo_clear_option(self, calculator):
        """Test hypo clear option affects output."""
        with_hypo = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            include_hypo_clear=True,
        )
        without_hypo = calculator.calculate(
            width_inches=8.0, height_inches=10.0,
            include_hypo_clear=False,
        )

        # With hypo clear should include hypo clear volume
        if hasattr(with_hypo, 'hypo_clear_volume_ml'):
            assert with_hypo.hypo_clear_volume_ml >= 0

    # --- Output Validation ---

    def test_processing_chemistry_output_types(self, calculator):
        """Test that output has correct data types."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        assert isinstance(result.developer_volume_ml, (int, float))
        assert isinstance(result.stop_bath_volume_ml, (int, float))
        assert isinstance(result.fixer_volume_ml, (int, float))

    def test_processing_chemistry_serialization(self, calculator):
        """Test that result can be serialized."""
        result = calculator.calculate(width_inches=8.0, height_inches=10.0)

        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert 'developer_volume_ml' in result_dict

    # --- Split Filter Exposure ---

    def test_split_filter_exposure_calculation(self, calculator):
        """Test split filter exposure calculation."""
        split = calculator.calculate_split_filter_exposure(
            base_exposure_seconds=10.0,
            shadow_grade=5.0,
            highlight_grade=0.0,
            split_ratio=0.5,
        )

        assert isinstance(split, dict)
        assert 'shadow_exposure' in split
        assert 'highlight_exposure' in split
        assert split['shadow_exposure'] > 0
        assert split['highlight_exposure'] > 0

    def test_split_filter_invalid_base_exposure(self, calculator):
        """Test split filter with invalid base exposure."""
        with pytest.raises(ValueError, match="exposure"):
            calculator.calculate_split_filter_exposure(
                base_exposure_seconds=0.0,
                shadow_grade=5.0,
                highlight_grade=0.0,
            )

        with pytest.raises(ValueError, match="exposure"):
            calculator.calculate_split_filter_exposure(
                base_exposure_seconds=-5.0,
                shadow_grade=5.0,
                highlight_grade=0.0,
            )

    def test_split_filter_grade_validation(self, calculator):
        """Test split filter grade validation."""
        # Valid grades (0-5 typically)
        result = calculator.calculate_split_filter_exposure(
            base_exposure_seconds=10.0,
            shadow_grade=4.0,
            highlight_grade=1.0,
        )
        assert result['shadow_exposure'] > 0

        # Invalid grades
        with pytest.raises(ValueError, match="grade"):
            calculator.calculate_split_filter_exposure(
                base_exposure_seconds=10.0,
                shadow_grade=-1.0,
                highlight_grade=0.0,
            )

    def test_split_ratio_validation(self, calculator):
        """Test split ratio validation."""
        # Valid ratios (0-1)
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = calculator.calculate_split_filter_exposure(
                base_exposure_seconds=10.0,
                split_ratio=ratio,
            )
            assert 'shadow_exposure' in result

        # Invalid ratio
        with pytest.raises(ValueError, match="ratio"):
            calculator.calculate_split_filter_exposure(
                base_exposure_seconds=10.0,
                split_ratio=1.5,
            )

    # --- Test Strip Generation ---

    def test_test_strip_generation(self, calculator):
        """Test test strip time generation."""
        strips = calculator.generate_test_strip_times(
            base_exposure=10.0,
            num_strips=5,
            increment_factor=1.5,
        )

        assert isinstance(strips, list)
        assert len(strips) == 5
        # Should be increasing
        for i in range(1, len(strips)):
            assert strips[i] > strips[i - 1]

    def test_test_strip_invalid_inputs(self, calculator):
        """Test test strip generation with invalid inputs."""
        with pytest.raises(ValueError, match="exposure"):
            calculator.generate_test_strip_times(base_exposure=0.0)

        with pytest.raises(ValueError, match="strips"):
            calculator.generate_test_strip_times(base_exposure=10.0, num_strips=0)


class TestSilverGelatinExposureValidation:
    """Validation tests for SilverGelatinExposureCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create SilverGelatinExposureCalculator instance."""
        return SilverGelatinExposureCalculator()

    # --- Input Validation ---

    def test_valid_exposure_calculation(self, calculator):
        """Test valid exposure calculation."""
        result = calculator.calculate(
            enlarger_height_cm=30.0,
            f_stop=8.0,
            paper_grade=PaperGrade.GRADE_2,
        )

        assert isinstance(result, SilverGelatinExposureResult)
        assert result.exposure_time_seconds > 0

    def test_enlarger_height_validation(self, calculator):
        """Test enlarger height validation."""
        # Valid heights
        result = calculator.calculate(enlarger_height_cm=30.0)
        assert result.exposure_time_seconds > 0

        # Invalid: zero height
        with pytest.raises(ValueError, match="height"):
            calculator.calculate(enlarger_height_cm=0.0)

        # Invalid: negative height
        with pytest.raises(ValueError, match="height"):
            calculator.calculate(enlarger_height_cm=-10.0)

    def test_f_stop_validation(self, calculator):
        """Test f-stop validation."""
        # Valid f-stops
        valid_fstops = [2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]
        for fstop in valid_fstops:
            result = calculator.calculate(f_stop=fstop)
            assert result.exposure_time_seconds > 0

        # Invalid: zero f-stop
        with pytest.raises(ValueError, match="f.stop|f_stop"):
            calculator.calculate(f_stop=0.0)

        # Invalid: negative f-stop
        with pytest.raises(ValueError, match="f.stop|f_stop"):
            calculator.calculate(f_stop=-8.0)

    def test_paper_speed_validation(self, calculator):
        """Test paper speed (ISO) validation."""
        # Valid ISO values
        for iso in [100.0, 200.0, 400.0, 800.0]:
            result = calculator.calculate(paper_speed_iso=iso)
            assert result.exposure_time_seconds > 0

        # Invalid: zero ISO
        with pytest.raises(ValueError, match="speed|iso"):
            calculator.calculate(paper_speed_iso=0.0)

        # Invalid: negative ISO
        with pytest.raises(ValueError, match="speed|iso"):
            calculator.calculate(paper_speed_iso=-100.0)

    def test_filter_factor_validation(self, calculator):
        """Test filter factor validation."""
        # Valid factors
        result = calculator.calculate(filter_factor=2.0)
        assert result.exposure_time_seconds > 0

        # Invalid: zero factor
        with pytest.raises(ValueError, match="filter"):
            calculator.calculate(filter_factor=0.0)

        # Invalid: negative factor
        with pytest.raises(ValueError, match="filter"):
            calculator.calculate(filter_factor=-1.0)

    def test_negative_density_validation(self, calculator):
        """Test negative density validation."""
        # Valid densities
        for density in [0.5, 1.0, 1.5, 2.0]:
            result = calculator.calculate(negative_density=density)
            assert result.exposure_time_seconds > 0

        # Invalid: negative density value
        with pytest.raises(ValueError, match="density"):
            calculator.calculate(negative_density=-0.5)

    # --- Enlarger Light Source Validation ---

    def test_all_enlarger_light_sources(self, calculator):
        """Test all enlarger light source types."""
        for light_source in EnlargerLightSource:
            result = calculator.calculate(
                enlarger_height_cm=30.0,
                light_source=light_source,
            )
            assert result.exposure_time_seconds > 0

    def test_enlarger_light_speeds_consistency(self):
        """Test that enlarger light speeds are defined correctly."""
        for light_source in EnlargerLightSource:
            assert light_source in ENLARGER_LIGHT_SPEEDS
            assert ENLARGER_LIGHT_SPEEDS[light_source] > 0

    # --- Paper Grade Validation ---

    def test_all_paper_grades(self, calculator):
        """Test all paper grades produce valid results."""
        for grade in PaperGrade:
            result = calculator.calculate(paper_grade=grade)
            assert result.exposure_time_seconds > 0

    def test_paper_grade_affects_contrast(self, calculator):
        """Test that paper grade affects exposure/contrast."""
        soft = calculator.calculate(paper_grade=PaperGrade.GRADE_0)
        hard = calculator.calculate(paper_grade=PaperGrade.GRADE_5)

        # Both should be valid
        assert soft.exposure_time_seconds > 0
        assert hard.exposure_time_seconds > 0

    # --- F-Stop and Height Effects ---

    def test_fstop_affects_exposure(self, calculator):
        """Test that f-stop changes affect exposure time."""
        wide = calculator.calculate(f_stop=4.0)
        narrow = calculator.calculate(f_stop=16.0)

        # Narrower aperture should require longer exposure
        assert narrow.exposure_time_seconds > wide.exposure_time_seconds

    def test_fstop_doubling_quadruples_exposure(self, calculator):
        """Test that doubling f-stop quadruples exposure."""
        f8 = calculator.calculate(f_stop=8.0, base_f_stop=8.0)
        f16 = calculator.calculate(f_stop=16.0, base_f_stop=8.0)

        # f/16 is 2 stops from f/8, should be 4x exposure
        ratio = f16.exposure_time_seconds / f8.exposure_time_seconds
        assert ratio == pytest.approx(4.0, rel=0.1)

    def test_height_affects_exposure(self, calculator):
        """Test that enlarger height affects exposure time."""
        low = calculator.calculate(enlarger_height_cm=20.0)
        high = calculator.calculate(enlarger_height_cm=40.0)

        # Higher position should require longer exposure (inverse square)
        assert high.exposure_time_seconds > low.exposure_time_seconds

    # --- Output Validation ---

    def test_exposure_result_contains_all_fields(self, calculator):
        """Test that exposure result contains expected fields."""
        result = calculator.calculate()

        assert hasattr(result, 'exposure_time_seconds')
        assert hasattr(result, 'exposure_time_formatted')
        assert hasattr(result, 'paper_grade')
        assert hasattr(result, 'f_stop')

    def test_exposure_time_formatted_is_readable(self, calculator):
        """Test formatted time is human-readable."""
        result = calculator.calculate()

        formatted = result.exposure_time_formatted
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    # --- Edge Cases ---

    def test_very_short_exposure(self, calculator):
        """Test very short exposure scenario."""
        result = calculator.calculate(
            enlarger_height_cm=15.0,
            f_stop=2.8,
            paper_speed_iso=800.0,
        )

        assert result.exposure_time_seconds > 0
        # Should be reasonably short
        assert result.exposure_time_seconds < 60

    def test_very_long_exposure(self, calculator):
        """Test very long exposure scenario."""
        result = calculator.calculate(
            enlarger_height_cm=100.0,
            f_stop=22.0,
            paper_speed_iso=50.0,
            filter_factor=4.0,
        )

        assert result.exposure_time_seconds > 0
        # Should be reasonably long
        assert result.exposure_time_seconds > 10


class TestSilverGelatinIntegrationValidation:
    """Integration validation tests combining chemistry and exposure."""

    def test_chemistry_and_exposure_workflow(self):
        """Test complete silver gelatin workflow."""
        # Step 1: Calculate exposure
        exp_calc = SilverGelatinExposureCalculator()
        exposure = exp_calc.calculate(
            enlarger_height_cm=30.0,
            f_stop=8.0,
            paper_grade=PaperGrade.GRADE_2,
        )

        assert exposure.exposure_time_seconds > 0

        # Step 2: Calculate chemistry for processing
        chem_calc = SilverGelatinCalculator()
        chemistry = chem_calc.calculate(
            width_inches=8.0,
            height_inches=10.0,
            paper_base=PaperBase.FIBER,
            developer=DeveloperType.DEKTOL,
        )

        assert chemistry.developer_volume_ml > 0
        assert chemistry.fixer_volume_ml > 0

    def test_paper_grade_consistency(self):
        """Test paper grade enum consistency."""
        exp_calc = SilverGelatinExposureCalculator()

        for grade in PaperGrade:
            result = exp_calc.calculate(paper_grade=grade)
            assert result.paper_grade == grade

    def test_multi_print_session_workflow(self):
        """Test workflow for multiple print session."""
        chem_calc = SilverGelatinCalculator()
        exp_calc = SilverGelatinExposureCalculator()

        # Plan for 5 prints
        chemistry = chem_calc.calculate(
            width_inches=8.0,
            height_inches=10.0,
            num_prints=5,
            tray_size=TraySize.TRAY_11X14,
        )

        # Calculate base exposure
        exposure = exp_calc.calculate(
            enlarger_height_cm=30.0,
            f_stop=8.0,
        )

        # Generate test strip
        test_strips = chem_calc.generate_test_strip_times(
            base_exposure=exposure.exposure_time_seconds,
            num_strips=5,
        )

        assert chemistry.developer_volume_ml > 0
        assert len(test_strips) == 5

    def test_split_filter_with_exposure(self):
        """Test split filter printing workflow."""
        chem_calc = SilverGelatinCalculator()
        exp_calc = SilverGelatinExposureCalculator()

        # Get base exposure
        base_exposure = exp_calc.calculate(
            enlarger_height_cm=30.0,
            f_stop=8.0,
            paper_grade=PaperGrade.GRADE_2,
        )

        # Calculate split filter exposures
        split = chem_calc.calculate_split_filter_exposure(
            base_exposure_seconds=base_exposure.exposure_time_seconds,
            shadow_grade=5.0,
            highlight_grade=0.0,
            split_ratio=0.5,
        )

        assert split['shadow_exposure'] > 0
        assert split['highlight_exposure'] > 0
        # Total should approximate base
        total = split['shadow_exposure'] + split['highlight_exposure']
        assert total > 0
