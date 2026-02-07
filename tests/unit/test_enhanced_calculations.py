"""
Comprehensive tests for enhanced calculation modules.

Tests:
- UVExposureCalculator: exposure calculation with various factors
- CoatingVolumeCalculator: volume for different papers/methods
- CostCalculator: print cost, session cost, usage estimation
- DilutionCalculator: developer and clearing bath dilutions
- EnvironmentalCompensation: altitude, season, drying time adjustments
"""


import pytest

from ptpd_calibration.calculations.enhanced import (
    CoatingVolumeCalculator,
    CostCalculator,
    DilutionCalculator,
    EnvironmentalCompensation,
    UVExposureCalculator,
)

# ============================================================================
# UVExposureCalculator Tests
# ============================================================================


class TestUVExposureCalculator:
    """Test UV exposure calculations with environmental compensation."""

    @pytest.fixture
    def calculator(self):
        """Create UVExposureCalculator instance."""
        return UVExposureCalculator()

    def test_basic_exposure_calculation(self, calculator):
        """Test basic exposure calculation at optimal conditions."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=100.0,
        )

        # At optimal conditions, adjustment should be minimal
        assert result.adjusted_exposure_minutes == pytest.approx(10.0, rel=0.1)
        assert result.base_time_minutes == 10.0
        assert result.base_negative_density == 1.6

    def test_humidity_adjustment(self, calculator):
        """Test humidity factor affects exposure time."""
        # Low humidity (drier)
        result_low = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=30.0,
            temperature=68.0,
            uv_intensity=100.0,
        )

        # High humidity (more moisture)
        result_high = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=70.0,
            temperature=68.0,
            uv_intensity=100.0,
        )

        # High humidity should reduce exposure time (faster exposure)
        assert result_high.adjusted_exposure_minutes < result_low.adjusted_exposure_minutes
        assert result_low.humidity_factor > result_high.humidity_factor

    def test_temperature_adjustment(self, calculator):
        """Test temperature factor affects exposure time."""
        # Low temperature
        result_low = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=55.0,
            uv_intensity=100.0,
        )

        # High temperature
        result_high = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=80.0,
            uv_intensity=100.0,
        )

        # High temperature should reduce exposure time (faster reactions)
        assert result_high.adjusted_exposure_minutes < result_low.adjusted_exposure_minutes

    def test_density_adjustment(self, calculator):
        """Test negative density affects exposure time."""
        # Thin negative
        result_thin = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.0,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=100.0,
        )

        # Dense negative
        result_dense = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=2.2,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=100.0,
        )

        # Dense negative should require longer exposure
        assert result_dense.adjusted_exposure_minutes > result_thin.adjusted_exposure_minutes
        assert result_dense.density_factor > result_thin.density_factor

    def test_uv_intensity_adjustment(self, calculator):
        """Test UV intensity affects exposure time."""
        # Low intensity
        result_low = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=50.0,
        )

        # High intensity
        result_high = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=120.0,
        )

        # Low intensity should require longer exposure
        assert result_low.adjusted_exposure_minutes > result_high.adjusted_exposure_minutes
        assert result_low.intensity_factor > result_high.intensity_factor

    def test_paper_factor(self, calculator):
        """Test paper speed factor affects exposure time."""
        # Fast paper
        result_fast = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            paper_factor=0.8,
        )

        # Slow paper
        result_slow = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            paper_factor=1.2,
        )

        # Fast paper should require less exposure
        assert result_fast.adjusted_exposure_minutes < result_slow.adjusted_exposure_minutes

    def test_chemistry_factor(self, calculator):
        """Test chemistry speed factor affects exposure time."""
        # Fast chemistry
        result_fast = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            chemistry_factor=0.9,
        )

        # Slow chemistry
        result_slow = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            chemistry_factor=1.1,
        )

        # Fast chemistry should require less exposure
        assert result_fast.adjusted_exposure_minutes < result_slow.adjusted_exposure_minutes

    def test_confidence_interval(self, calculator):
        """Test confidence interval calculation."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            uncertainty_percent=15.0,
        )

        # Check confidence interval bounds
        assert result.confidence_lower_minutes < result.adjusted_exposure_minutes
        assert result.confidence_upper_minutes > result.adjusted_exposure_minutes

        # Check interval width is correct
        interval_width = (
            result.confidence_upper_minutes - result.confidence_lower_minutes
        )
        expected_width = result.adjusted_exposure_minutes * 0.3  # ±15% = 30% total
        assert interval_width == pytest.approx(expected_width, rel=0.01)

    def test_warnings_low_humidity(self, calculator):
        """Test warnings for low humidity."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=25.0,
            temperature=68.0,
        )

        assert any("humidity" in w.lower() for w in result.warnings)

    def test_warnings_high_temperature(self, calculator):
        """Test warnings for high temperature."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=85.0,
        )

        assert any("temperature" in w.lower() for w in result.warnings)

    def test_warnings_low_uv_intensity(self, calculator):
        """Test warnings for low UV intensity."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
            uv_intensity=60.0,
        )

        assert any("intensity" in w.lower() for w in result.warnings)

    def test_warnings_long_exposure(self, calculator):
        """Test warnings for very long exposure times."""
        result = calculator.calculate_uv_exposure(
            base_time=40.0,  # Long base time
            negative_density=1.6,
            humidity=50.0,
            temperature=68.0,
        )

        assert any("long exposure" in w.lower() for w in result.warnings)

    def test_combined_factors(self, calculator):
        """Test multiple factors working together."""
        result = calculator.calculate_uv_exposure(
            base_time=10.0,
            negative_density=2.0,  # Dense negative
            humidity=70.0,  # High humidity (faster)
            temperature=75.0,  # High temp (faster)
            uv_intensity=80.0,  # Low intensity (slower)
            paper_factor=1.1,  # Slow paper
            chemistry_factor=0.9,  # Fast chemistry
        )

        # All factors should multiply together
        assert result.adjusted_exposure_minutes > 0
        assert result.density_factor > 1.0  # Dense negative increases time


# ============================================================================
# CoatingVolumeCalculator Tests
# ============================================================================


class TestCoatingVolumeCalculator:
    """Test coating volume calculations for different papers and methods."""

    @pytest.fixture
    def calculator(self):
        """Create CoatingVolumeCalculator instance."""
        return CoatingVolumeCalculator()

    def test_basic_volume_calculation(self, calculator):
        """Test basic volume calculation."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,  # 8x10 inches
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        assert result.total_ml > 0
        assert result.paper_area_sq_inches == 80.0
        assert result.paper_type == "arches_platine"
        assert result.coating_method == "brush"

    def test_different_paper_types(self, calculator):
        """Test different paper absorbency rates."""
        # Hot press (low absorbency)
        result_hp = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        # Cold press (higher absorbency)
        result_cp = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="fabriano_artistico_cp",
            coating_method="brush",
            humidity=50.0,
        )

        # Cold press should require more volume
        assert result_cp.total_ml > result_hp.total_ml

    def test_coating_method_efficiency(self, calculator):
        """Test different coating methods use different volumes."""
        # Brush (least efficient)
        result_brush = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        # Coating rod (most efficient)
        result_rod = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="coating_rod",
            humidity=50.0,
        )

        # Coating rod should use less volume
        assert result_rod.total_ml < result_brush.total_ml

    def test_humidity_adjustment(self, calculator):
        """Test humidity affects coating volume."""
        # Low humidity
        result_low = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=30.0,
        )

        # High humidity
        result_high = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=70.0,
        )

        # High humidity should require more volume
        assert result_high.total_ml > result_low.total_ml

    def test_waste_factor(self, calculator):
        """Test waste factor increases total volume."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
            waste_factor=1.2,  # 20% waste
        )

        assert result.waste_volume_ml > 0
        assert result.total_ml > result.adjusted_volume_ml

    def test_drops_conversion(self, calculator):
        """Test conversion to drops."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        # Drops should be positive
        assert result.total_drops > 0
        assert result.recommended_drops > 0

    def test_recommended_rounding(self, calculator):
        """Test recommended values are rounded for practical use."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        # Recommended ml should be rounded to nearest 0.5
        assert result.recommended_ml % 0.5 == 0

    def test_unknown_paper_type(self, calculator):
        """Test unknown paper type uses default values."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="unknown_paper",
            coating_method="brush",
            humidity=50.0,
        )

        # Should still work with defaults
        assert result.total_ml > 0
        assert any("default" in n.lower() for n in result.notes)

    def test_unknown_coating_method(self, calculator):
        """Test unknown coating method uses default efficiency."""
        result = calculator.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="unknown_method",
            humidity=50.0,
        )

        # Should use default efficiency (1.0)
        assert result.method_efficiency_factor == 1.0
        assert any("default" in n.lower() for n in result.notes)

    def test_small_area_warning(self, calculator):
        """Test warning for small volumes."""
        result = calculator.determine_coating_volume(
            paper_area=10.0,  # Small area
            paper_type="arches_platine",
            coating_method="coating_rod",  # Efficient method
            humidity=50.0,
        )

        # Should have note about careful measurement
        if result.total_ml < 1.0:
            assert any("small volume" in n.lower() or "careful" in n.lower() for n in result.notes)

    def test_large_area_calculation(self, calculator):
        """Test calculation for large print area."""
        result = calculator.determine_coating_volume(
            paper_area=200.0,  # 16x20 inches
            paper_type="arches_platine",
            coating_method="brush",
            humidity=50.0,
        )

        # Should handle large areas
        assert result.total_ml > 0
        if result.total_ml > 10.0:
            assert any("large" in n.lower() or "batch" in n.lower() for n in result.notes)


# ============================================================================
# CostCalculator Tests
# ============================================================================


class TestCostCalculator:
    """Test cost calculations for prints and sessions."""

    @pytest.fixture
    def calculator(self):
        """Create CostCalculator instance."""
        return CostCalculator()

    def test_basic_print_cost(self, calculator):
        """Test basic print cost calculation."""
        chemistry = {
            "ferric_oxalate_ml": 2.0,
            "platinum_ml": 1.0,
            "palladium_ml": 1.0,
            "na2_ml": 0.2,
        }

        result = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
            paper_type="arches_platine",
        )

        assert result.total_cost_usd > 0
        assert result.paper_area_sq_inches == 80.0
        assert result.ferric_oxalate_cost > 0
        assert result.platinum_cost > 0
        assert result.palladium_cost > 0

    def test_platinum_only_cost(self, calculator):
        """Test cost for platinum-only print."""
        chemistry = {
            "ferric_oxalate_ml": 2.0,
            "platinum_ml": 2.0,
            "palladium_ml": 0.0,
            "na2_ml": 0.0,
        }

        result = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
            paper_type="arches_platine",
        )

        assert result.platinum_cost > 0
        assert result.palladium_cost == 0
        assert result.metal_ratio_platinum == 1.0

    def test_palladium_only_cost(self, calculator):
        """Test cost for palladium-only print."""
        chemistry = {
            "ferric_oxalate_ml": 2.0,
            "platinum_ml": 0.0,
            "palladium_ml": 2.0,
            "na2_ml": 0.0,
        }

        result = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
            paper_type="arches_platine",
        )

        assert result.platinum_cost == 0
        assert result.palladium_cost > 0
        assert result.metal_ratio_platinum == 0.0

    def test_processing_costs(self, calculator):
        """Test including processing costs."""
        chemistry = {"ferric_oxalate_ml": 2.0, "platinum_ml": 1.0, "palladium_ml": 1.0}

        result_with = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
            include_processing=True,
        )

        result_without = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
            include_processing=False,
        )

        # With processing should cost more
        assert result_with.total_cost_usd > result_without.total_cost_usd
        assert result_with.other_costs > 0
        assert result_without.other_costs == 0

    def test_different_paper_sizes(self, calculator):
        """Test costs scale with paper size."""
        chemistry = {"ferric_oxalate_ml": 2.0, "platinum_ml": 1.0, "palladium_ml": 1.0}

        result_small = calculator.calculate_print_cost(
            paper_size="5x7",
            chemistry=chemistry,
        )

        result_large = calculator.calculate_print_cost(
            paper_size="11x14",
            chemistry=chemistry,
        )

        # Larger prints should cost more (paper cost)
        assert result_large.paper_cost > result_small.paper_cost

    def test_invalid_paper_size_format(self, calculator):
        """Test invalid paper size format raises error."""
        chemistry = {"ferric_oxalate_ml": 2.0}

        with pytest.raises(ValueError, match="Invalid paper size"):
            calculator.calculate_print_cost(
                paper_size="invalid",
                chemistry=chemistry,
            )

    def test_session_cost_calculation(self, calculator):
        """Test session cost aggregation."""
        # Create multiple print costs
        prints = []
        for _i in range(3):
            chemistry = {
                "ferric_oxalate_ml": 2.0,
                "platinum_ml": 1.0,
                "palladium_ml": 1.0,
            }
            result = calculator.calculate_print_cost(
                paper_size="8x10",
                chemistry=chemistry,
            )
            prints.append(result)

        session = calculator.calculate_session_cost(prints)

        assert session.num_prints == 3
        assert session.total_session_cost_usd > 0
        assert session.average_cost_per_print > 0
        assert session.total_chemistry_cost > 0
        assert len(session.print_costs) == 3

    def test_empty_session(self, calculator):
        """Test session cost with no prints."""
        session = calculator.calculate_session_cost([])

        assert session.num_prints == 0
        assert session.total_session_cost_usd == 0
        assert session.average_cost_per_print == 0
        assert "No prints" in session.notes[0]

    def test_solution_usage_estimate(self, calculator):
        """Test estimating solution usage."""
        estimate = calculator.estimate_solution_usage(
            num_prints=10,
            avg_size="8x10",
            avg_platinum_ratio=0.5,
            coating_method="brush",
        )

        assert estimate.num_prints == 10
        assert estimate.ferric_oxalate_ml > 0
        assert estimate.platinum_ml > 0
        assert estimate.palladium_ml > 0
        assert estimate.developer_ml > 0
        assert estimate.clearing_bath_ml > 0

        # Recommended stock should be higher (safety margin)
        assert estimate.recommended_stock_ferric_oxalate_ml > estimate.ferric_oxalate_ml

    def test_cost_report_generation(self, calculator):
        """Test generating cost report."""
        chemistry = {"ferric_oxalate_ml": 2.0, "platinum_ml": 1.0, "palladium_ml": 1.0}
        print_cost = calculator.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
        )
        session = calculator.calculate_session_cost([print_cost])

        report = calculator.generate_cost_report(session, time_period="test")

        assert "COST ANALYSIS REPORT" in report
        assert "TEST" in report
        assert "Number of prints" in report
        assert "Total cost" in report


# ============================================================================
# DilutionCalculator Tests
# ============================================================================


class TestDilutionCalculator:
    """Test dilution calculations for developers and clearing baths."""

    @pytest.fixture
    def calculator(self):
        """Create DilutionCalculator instance."""
        return DilutionCalculator()

    def test_developer_dilution(self, calculator):
        """Test developer dilution calculation."""
        result = calculator.calculate_developer_dilution(
            concentrate_strength=20.0,  # 20% EDTA
            target_strength=2.0,  # 2% working solution
            volume=1000.0,  # 1000ml
        )

        assert result.concentrate_ml == pytest.approx(100.0)
        assert result.water_ml == pytest.approx(900.0)
        assert result.total_ml == 1000.0
        assert result.dilution_ratio == "1:9"

    def test_dilution_ratio_calculation(self, calculator):
        """Test various dilution ratios."""
        # 1:4 dilution
        result = calculator.calculate_developer_dilution(
            concentrate_strength=10.0,
            target_strength=2.0,
            volume=500.0,
        )

        assert "1:4" in result.dilution_ratio

    def test_invalid_dilution_strength(self, calculator):
        """Test error when concentrate weaker than target."""
        with pytest.raises(ValueError, match="must be greater than"):
            calculator.calculate_developer_dilution(
                concentrate_strength=2.0,
                target_strength=10.0,
                volume=1000.0,
            )

    def test_clearing_bath_1(self, calculator):
        """Test first clearing bath preparation."""
        result = calculator.calculate_clearing_bath(
            volume=1000.0,
            bath_number=1,
        )

        assert result.total_ml == 1000.0
        assert result.target_strength == 1.0
        assert "citric acid" in result.notes[0].lower()

    def test_clearing_bath_2(self, calculator):
        """Test second clearing bath preparation."""
        result = calculator.calculate_clearing_bath(
            volume=1000.0,
            bath_number=2,
        )

        assert result.total_ml == 1000.0
        assert result.target_strength == 0.5
        assert "0.5%" in result.notes[0]

    def test_clearing_bath_3(self, calculator):
        """Test third clearing bath (water rinse)."""
        result = calculator.calculate_clearing_bath(
            volume=1000.0,
            bath_number=3,
        )

        assert result.concentrate_ml == 0.0
        assert result.water_ml == 1000.0
        assert "water" in result.notes[0].lower()

    def test_replenishment_within_threshold(self, calculator):
        """Test replenishment when below exhaustion threshold."""
        result = calculator.suggest_replenishment(
            solution="developer",
            usage=200.0,
            current_volume=1000.0,
            exhaustion_threshold=0.30,
        )

        assert not result.should_replace
        assert result.replenish_ml == 200.0
        assert result.exhaustion_percent == 20.0

    def test_replenishment_above_threshold(self, calculator):
        """Test full replacement when above exhaustion threshold."""
        result = calculator.suggest_replenishment(
            solution="developer",
            usage=400.0,
            current_volume=1000.0,
            exhaustion_threshold=0.30,
        )

        assert result.should_replace
        assert result.replenish_ml == 1000.0
        assert result.exhaustion_percent == 40.0

    def test_replenishment_clearing_bath_notes(self, calculator):
        """Test clearing bath specific notes."""
        result = calculator.suggest_replenishment(
            solution="clearing_bath_1",
            usage=100.0,
            current_volume=1000.0,
        )

        assert any("clearing" in n.lower() for n in result.notes)


# ============================================================================
# EnvironmentalCompensation Tests
# ============================================================================


class TestEnvironmentalCompensation:
    """Test environmental compensation calculations."""

    @pytest.fixture
    def calculator(self):
        """Create EnvironmentalCompensation instance."""
        return EnvironmentalCompensation()

    def test_altitude_drying_adjustment(self, calculator):
        """Test altitude adjustment for drying time."""
        result = calculator.adjust_for_altitude(
            base_value=20.0,  # 20 minutes at sea level
            altitude=5000.0,  # 5000 feet
            value_type="drying_time",
        )

        # Higher altitude = faster drying
        assert result.adjusted_value < result.base_value
        assert result.adjustment_factor < 1.0
        assert result.altitude_feet == 5000.0

    def test_altitude_exposure_adjustment(self, calculator):
        """Test altitude adjustment for exposure time."""
        result = calculator.adjust_for_altitude(
            base_value=10.0,  # 10 minutes at sea level
            altitude=5000.0,
            value_type="exposure_time",
        )

        # Higher altitude = more UV = shorter exposure
        assert result.adjusted_value < result.base_value

    def test_sea_level_altitude(self, calculator):
        """Test no adjustment at sea level."""
        result = calculator.adjust_for_altitude(
            base_value=20.0,
            altitude=0.0,
            value_type="drying_time",
        )

        # No adjustment at sea level
        assert result.adjusted_value == result.base_value
        assert result.adjustment_factor == 1.0

    def test_high_altitude_warning(self, calculator):
        """Test warning for very high altitudes."""
        result = calculator.adjust_for_altitude(
            base_value=20.0,
            altitude=6000.0,
            value_type="drying_time",
        )

        assert any("high altitude" in n.lower() for n in result.notes)

    def test_seasonal_adjustment_summer(self, calculator):
        """Test seasonal adjustment for summer."""
        result = calculator.adjust_for_season(
            base_value=20.0,
            month=7,  # July
            value_type="drying_time",
        )

        # Summer = faster drying
        assert result.adjusted_value < result.base_value
        assert "Summer" in result.notes[0]

    def test_seasonal_adjustment_winter(self, calculator):
        """Test seasonal adjustment for winter."""
        result = calculator.adjust_for_season(
            base_value=20.0,
            month=1,  # January
            value_type="drying_time",
        )

        # Winter = slower drying
        assert result.adjusted_value > result.base_value
        assert "Winter" in result.notes[0]

    def test_seasonal_latitude_effect(self, calculator):
        """Test latitude affects seasonal variation magnitude."""
        # High latitude (more seasonal variation)
        result_high = calculator.adjust_for_season(
            base_value=20.0,
            month=7,
            latitude=60.0,
        )

        # Low latitude (less seasonal variation)
        result_low = calculator.adjust_for_season(
            base_value=20.0,
            month=7,
            latitude=10.0,
        )

        # High latitude should have larger adjustment
        high_diff = abs(result_high.adjusted_value - result_high.base_value)
        low_diff = abs(result_low.adjusted_value - result_low.base_value)
        assert high_diff > low_diff

    def test_get_optimal_conditions(self, calculator):
        """Test getting optimal working conditions."""
        conditions = calculator.get_optimal_conditions()

        assert conditions.temperature_f_min < conditions.temperature_f_ideal
        assert conditions.temperature_f_ideal < conditions.temperature_f_max
        assert conditions.humidity_percent_min < conditions.humidity_percent_ideal
        assert conditions.humidity_percent_ideal < conditions.humidity_percent_max
        assert conditions.altitude_feet_max > 0
        assert len(conditions.notes) > 0

    def test_drying_time_calculation(self, calculator):
        """Test drying time estimation."""
        result = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="arches_platine",
            forced_air=False,
        )

        assert result.drying_minutes > 0
        assert result.drying_hours > 0
        assert result.humidity_percent == 50.0
        assert result.temperature_fahrenheit == 68.0

    def test_drying_time_high_humidity(self, calculator):
        """Test drying time increases with humidity."""
        result_low = calculator.calculate_drying_time(
            humidity=30.0,
            temperature=68.0,
            paper="arches_platine",
        )

        result_high = calculator.calculate_drying_time(
            humidity=70.0,
            temperature=68.0,
            paper="arches_platine",
        )

        # Higher humidity = longer drying
        assert result_high.drying_minutes > result_low.drying_minutes

    def test_drying_time_temperature_effect(self, calculator):
        """Test drying time decreases with temperature."""
        result_low = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=60.0,
            paper="arches_platine",
        )

        result_high = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=75.0,
            paper="arches_platine",
        )

        # Higher temperature = faster drying
        assert result_high.drying_minutes < result_low.drying_minutes

    def test_drying_time_paper_absorbency(self, calculator):
        """Test different papers have different drying times."""
        result_hp = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="arches_platine",  # Hot press
        )

        result_cp = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="fabriano_artistico_cp",  # Cold press
        )

        # Cold press should take longer to dry
        assert result_cp.drying_minutes > result_hp.drying_minutes

    def test_drying_time_forced_air(self, calculator):
        """Test forced air reduces drying time."""
        result_natural = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="arches_platine",
            forced_air=False,
        )

        result_forced = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="arches_platine",
            forced_air=True,
        )

        # Forced air should be significantly faster
        assert result_forced.drying_minutes < result_natural.drying_minutes
        assert result_forced.drying_minutes == pytest.approx(
            result_natural.drying_minutes * 0.5, rel=0.1
        )

    def test_drying_time_range(self, calculator):
        """Test drying time includes estimated range."""
        result = calculator.calculate_drying_time(
            humidity=50.0,
            temperature=68.0,
            paper="arches_platine",
        )

        range_min, range_max = result.estimated_range_minutes

        assert range_min < result.drying_minutes
        assert range_max > result.drying_minutes

    def test_forced_air_recommendation(self, calculator):
        """Test forced air recommended in poor conditions."""
        result = calculator.calculate_drying_time(
            humidity=70.0,  # High humidity
            temperature=62.0,  # Low temperature
            paper="arches_platine",
        )

        assert result.forced_air_recommended

    def test_season_name_mapping(self):
        """Test season name mapping from months."""
        assert EnvironmentalCompensation._get_season_name(1) == "Winter"
        assert EnvironmentalCompensation._get_season_name(4) == "Spring"
        assert EnvironmentalCompensation._get_season_name(7) == "Summer"
        assert EnvironmentalCompensation._get_season_name(10) == "Fall"


# ============================================================================
# Integration Tests
# ============================================================================


class TestCalculatorIntegration:
    """Test calculators working together."""

    def test_complete_print_workflow(self):
        """Test complete print planning workflow."""
        # 1. Calculate exposure
        exposure_calc = UVExposureCalculator()
        exposure = exposure_calc.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.8,
            humidity=55.0,
            temperature=70.0,
        )

        # 2. Calculate coating volume
        coating_calc = CoatingVolumeCalculator()
        coating = coating_calc.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=55.0,
        )

        # 3. Calculate costs
        cost_calc = CostCalculator()
        chemistry = {
            "ferric_oxalate_ml": coating.total_ml / 2,
            "platinum_ml": coating.total_ml / 4,
            "palladium_ml": coating.total_ml / 4,
        }
        cost = cost_calc.calculate_print_cost(
            paper_size="8x10",
            chemistry=chemistry,
        )

        # 4. Calculate drying time
        env_calc = EnvironmentalCompensation()
        drying = env_calc.calculate_drying_time(
            humidity=55.0,
            temperature=70.0,
            paper="arches_platine",
        )

        # All calculations should complete successfully
        assert exposure.adjusted_exposure_minutes > 0
        assert coating.total_ml > 0
        assert cost.total_cost_usd > 0
        assert drying.drying_minutes > 0

    def test_environmental_consistency(self):
        """Test environmental factors are consistent across calculators."""
        humidity = 65.0
        temperature = 72.0

        # Exposure calculation
        exposure_calc = UVExposureCalculator()
        exposure = exposure_calc.calculate_uv_exposure(
            base_time=10.0,
            negative_density=1.6,
            humidity=humidity,
            temperature=temperature,
        )

        # Coating calculation
        coating_calc = CoatingVolumeCalculator()
        coating = coating_calc.determine_coating_volume(
            paper_area=80.0,
            paper_type="arches_platine",
            coating_method="brush",
            humidity=humidity,
        )

        # Drying calculation
        env_calc = EnvironmentalCompensation()
        drying = env_calc.calculate_drying_time(
            humidity=humidity,
            temperature=temperature,
            paper="arches_platine",
        )

        # All should use same environmental parameters
        assert exposure.humidity_percent == humidity
        assert exposure.temperature_fahrenheit == temperature
        assert coating.humidity_adjustment_factor != 1.0  # Should adjust
        assert drying.humidity_percent == humidity
        assert drying.temperature_fahrenheit == temperature
