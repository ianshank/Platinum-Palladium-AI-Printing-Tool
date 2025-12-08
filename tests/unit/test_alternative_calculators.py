"""
Unit tests for alternative process chemistry and exposure calculators.

Tests cyanotype and silver gelatin calculator modules.
"""

import pytest
from ptpd_calibration.chemistry.cyanotype_calculator import (
    CyanotypeCalculator,
    CyanotypeRecipe,
    CyanotypeSettings,
    CyanotypePaperType,
    CYANOTYPE_PAPER_FACTORS,
)
from ptpd_calibration.chemistry.silver_gelatin_calculator import (
    SilverGelatinCalculator,
    ProcessingChemistry,
    SilverGelatinSettings,
    DeveloperRecipe,
    DilutionRatio,
    TraySize,
    DILUTION_MULTIPLIERS,
    TRAY_VOLUMES_ML,
)
from ptpd_calibration.exposure.alternative_calculators import (
    CyanotypeExposureCalculator,
    CyanotypeExposureResult,
    SilverGelatinExposureCalculator,
    SilverGelatinExposureResult,
    VanDykeExposureCalculator,
    KallitypeExposureCalculator,
    UVSource,
    UV_SOURCE_SPEEDS,
    EnlargerLightSource,
    ENLARGER_LIGHT_SPEEDS,
)
from ptpd_calibration.core.types import (
    CyanotypeFormula,
    DeveloperType,
    FixerType,
    PaperBase,
    PaperGrade,
)


# =====================================================
# CYANOTYPE CHEMISTRY CALCULATOR TESTS
# =====================================================

class TestCyanotypePaperFactors:
    """Tests for cyanotype paper absorbency factors."""

    def test_cotton_rag_baseline(self):
        """Cotton rag should be baseline 1.0."""
        assert CYANOTYPE_PAPER_FACTORS[CyanotypePaperType.COTTON_RAG] == 1.0

    def test_fabric_higher_absorbency(self):
        """Fabric types should have higher absorbency."""
        assert CYANOTYPE_PAPER_FACTORS[CyanotypePaperType.FABRIC_COTTON] > 1.0
        assert CYANOTYPE_PAPER_FACTORS[CyanotypePaperType.FABRIC_LINEN] > 1.0

    def test_hot_press_lower_absorbency(self):
        """Hot press watercolor should have lower absorbency."""
        assert CYANOTYPE_PAPER_FACTORS[CyanotypePaperType.WATERCOLOR_HOT] < 1.0

    def test_all_paper_types_defined(self):
        """All paper types should have factors defined."""
        for paper_type in list(CyanotypePaperType):
            assert paper_type in CYANOTYPE_PAPER_FACTORS


class TestCyanotypeRecipe:
    """Tests for CyanotypeRecipe dataclass."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a sample recipe for testing."""
        return CyanotypeRecipe(
            print_width_inches=8.0,
            print_height_inches=10.0,
            coating_width_inches=7.0,
            coating_height_inches=9.0,
            coating_area_sq_inches=63.0,
            solution_a_ml=0.5,
            solution_b_ml=0.5,
            total_sensitizer_ml=1.0,
            solution_a_drops=10.0,
            solution_b_drops=10.0,
            total_drops=20.0,
            formula=CyanotypeFormula.CLASSIC,
            paper_type=CyanotypePaperType.COTTON_RAG,
            concentration_factor=1.0,
            development_method="Running water 5-10 minutes",
            estimated_exposure_minutes=15.0,
            estimated_cost_usd=0.10,
            notes=["Test note"],
        )

    def test_recipe_to_dict(self, sample_recipe):
        """Recipe should convert to dictionary correctly."""
        d = sample_recipe.to_dict()
        assert d["print_dimensions"]["width_inches"] == 8.0
        assert d["print_dimensions"]["height_inches"] == 10.0
        assert d["milliliters"]["solution_a_fac"] == 0.5
        assert d["milliliters"]["solution_b_potassium_ferricyanide"] == 0.5
        assert d["settings"]["formula"] == "classic"

    def test_recipe_format(self, sample_recipe):
        """Recipe should format as readable text."""
        text = sample_recipe.format_recipe()
        assert "CYANOTYPE" in text
        assert "8.0" in text  # Width
        assert "10.0" in text  # Height
        assert "Ferric Ammonium Citrate" in text
        assert "Potassium Ferricyanide" in text
        assert "drops" in text
        assert "ml" in text


class TestCyanotypeCalculator:
    """Tests for CyanotypeCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with default settings."""
        return CyanotypeCalculator()

    def test_calculate_8x10_default(self, calculator):
        """8x10 print with defaults should produce reasonable results."""
        recipe = calculator.calculate(8.0, 10.0)

        assert recipe.print_width_inches == 8.0
        assert recipe.print_height_inches == 10.0
        assert recipe.coating_area_sq_inches > 0
        assert recipe.solution_a_ml > 0
        assert recipe.solution_b_ml > 0
        assert recipe.total_sensitizer_ml > 0

    def test_calculate_equal_solutions(self, calculator):
        """Solution A and B should be equal for classic formula."""
        recipe = calculator.calculate(8.0, 10.0, formula=CyanotypeFormula.CLASSIC)

        assert abs(recipe.solution_a_ml - recipe.solution_b_ml) < 0.001

    def test_calculate_small_vs_large(self, calculator):
        """Larger prints should use more chemistry."""
        small = calculator.calculate(4.0, 5.0)
        large = calculator.calculate(16.0, 20.0)

        assert small.total_sensitizer_ml < large.total_sensitizer_ml
        assert small.total_drops < large.total_drops

    def test_calculate_new_formula_more_efficient(self, calculator):
        """New cyanotype formula should use less chemistry."""
        classic = calculator.calculate(8.0, 10.0, formula=CyanotypeFormula.CLASSIC)
        new = calculator.calculate(8.0, 10.0, formula=CyanotypeFormula.NEW)

        assert new.total_sensitizer_ml < classic.total_sensitizer_ml

    def test_calculate_fabric_more_chemistry(self, calculator):
        """Fabric should require more sensitizer than paper."""
        paper = calculator.calculate(8.0, 10.0, paper_type=CyanotypePaperType.COTTON_RAG)
        fabric = calculator.calculate(8.0, 10.0, paper_type=CyanotypePaperType.FABRIC_COTTON)

        assert fabric.total_sensitizer_ml > paper.total_sensitizer_ml

    def test_calculate_concentration_factor(self, calculator):
        """Higher concentration should increase amounts."""
        normal = calculator.calculate(8.0, 10.0, concentration_factor=1.0)
        high = calculator.calculate(8.0, 10.0, concentration_factor=1.5)

        assert high.total_sensitizer_ml > normal.total_sensitizer_ml

    def test_calculate_cost_estimate(self, calculator):
        """Cost estimate should be calculated when enabled."""
        with_cost = calculator.calculate(8.0, 10.0, include_cost=True)
        without_cost = calculator.calculate(8.0, 10.0, include_cost=False)

        assert with_cost.estimated_cost_usd is not None
        assert with_cost.estimated_cost_usd > 0
        assert without_cost.estimated_cost_usd is None

    def test_calculate_generates_notes(self, calculator):
        """Calculator should generate helpful notes."""
        recipe = calculator.calculate(8.0, 10.0)
        assert len(recipe.notes) > 0

    def test_calculate_invalid_dimensions_raises(self, calculator):
        """Invalid dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            calculator.calculate(0.0, 10.0)
        with pytest.raises(ValueError):
            calculator.calculate(8.0, -5.0)

    def test_calculate_invalid_concentration_raises(self, calculator):
        """Invalid concentration should raise ValueError."""
        with pytest.raises(ValueError):
            calculator.calculate(8.0, 10.0, concentration_factor=0.2)
        with pytest.raises(ValueError):
            calculator.calculate(8.0, 10.0, concentration_factor=3.0)

    def test_calculate_stock_solutions(self, calculator):
        """Stock solution calculation should work."""
        solutions = calculator.calculate_stock_solutions(100.0, CyanotypeFormula.CLASSIC)

        assert "solution_a" in solutions
        assert "solution_b" in solutions
        assert solutions["solution_a"]["chemical_grams"] == 25.0  # 25% FAC
        assert solutions["solution_b"]["chemical_grams"] == 10.0  # 10% KFC

    def test_get_standard_sizes(self):
        """Standard sizes should be available."""
        sizes = CyanotypeCalculator.get_standard_sizes()

        assert "4x5" in sizes
        assert "8x10" in sizes
        assert "16x20" in sizes
        assert sizes["8x10"] == (8, 10)

    def test_get_troubleshooting_guide(self):
        """Troubleshooting guide should have common issues."""
        guide = CyanotypeCalculator.get_troubleshooting_guide()

        assert "weak_blues" in guide
        assert "yellow_stain" in guide
        assert "bronzing" in guide


# =====================================================
# SILVER GELATIN CHEMISTRY CALCULATOR TESTS
# =====================================================

class TestDilutionMultipliers:
    """Tests for dilution ratio multipliers."""

    def test_stock_is_full_strength(self):
        """Stock dilution should be 1.0."""
        assert DILUTION_MULTIPLIERS[DilutionRatio.STOCK] == 1.0

    def test_one_to_one_is_half(self):
        """1:1 dilution should be 0.5."""
        assert DILUTION_MULTIPLIERS[DilutionRatio.ONE_TO_ONE] == 0.5

    def test_dilutions_decrease(self):
        """Higher dilutions should have lower multipliers."""
        assert DILUTION_MULTIPLIERS[DilutionRatio.ONE_TO_TWO] < DILUTION_MULTIPLIERS[DilutionRatio.ONE_TO_ONE]
        assert DILUTION_MULTIPLIERS[DilutionRatio.ONE_TO_THREE] < DILUTION_MULTIPLIERS[DilutionRatio.ONE_TO_TWO]


class TestTrayVolumes:
    """Tests for tray volume constants."""

    def test_larger_trays_more_volume(self):
        """Larger trays should hold more volume."""
        assert TRAY_VOLUMES_ML[TraySize.EIGHT_BY_TEN] > TRAY_VOLUMES_ML[TraySize.FIVE_BY_SEVEN]
        assert TRAY_VOLUMES_ML[TraySize.ELEVEN_BY_FOURTEEN] > TRAY_VOLUMES_ML[TraySize.EIGHT_BY_TEN]

    def test_all_sizes_defined(self):
        """All tray sizes should have volumes defined."""
        for size in list(TraySize):
            assert size in TRAY_VOLUMES_ML


class TestSilverGelatinCalculator:
    """Tests for SilverGelatinCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with default settings."""
        return SilverGelatinCalculator()

    def test_calculate_8x10_default(self, calculator):
        """8x10 print with defaults should produce reasonable results."""
        recipe = calculator.calculate(8.0, 10.0)

        assert recipe.print_size == (8.0, 10.0)
        assert recipe.developer.total_ml > 0
        assert recipe.stop_bath_ml > 0
        assert recipe.fixer_ml > 0

    def test_calculate_auto_tray_selection(self, calculator):
        """Tray should be auto-selected based on print size."""
        small = calculator.calculate(4.0, 5.0)
        large = calculator.calculate(11.0, 14.0)

        assert small.tray_size == TraySize.FIVE_BY_SEVEN
        assert large.tray_size == TraySize.ELEVEN_BY_FOURTEEN

    def test_calculate_fb_vs_rc_times(self, calculator):
        """FB paper should have longer fix and wash times."""
        fb = calculator.calculate(8.0, 10.0, paper_base=PaperBase.FIBER)
        rc = calculator.calculate(8.0, 10.0, paper_base=PaperBase.RESIN_COATED)

        assert fb.fixer_time_seconds > rc.fixer_time_seconds
        assert fb.wash_time_minutes > rc.wash_time_minutes

    def test_calculate_hypo_clear_for_fb(self, calculator):
        """FB paper should include hypo clear option."""
        fb = calculator.calculate(8.0, 10.0, paper_base=PaperBase.FIBER, include_hypo_clear=True)
        rc = calculator.calculate(8.0, 10.0, paper_base=PaperBase.RESIN_COATED, include_hypo_clear=True)

        assert fb.hypo_clear_ml is not None
        assert fb.hypo_clear_ml > 0
        assert rc.hypo_clear_ml is None  # RC doesn't need hypo clear

    def test_calculate_developer_dilution(self, calculator):
        """Developer dilution should affect stock amount."""
        one_to_one = calculator.calculate(8.0, 10.0, dilution=DilutionRatio.ONE_TO_ONE)
        one_to_two = calculator.calculate(8.0, 10.0, dilution=DilutionRatio.ONE_TO_TWO)

        assert one_to_one.developer.stock_ml > one_to_two.developer.stock_ml

    def test_calculate_temperature_affects_time(self, calculator):
        """Lower temperature should increase development time."""
        warm = calculator.calculate(8.0, 10.0, temperature_c=22.0)
        cool = calculator.calculate(8.0, 10.0, temperature_c=18.0)

        assert cool.developer.development_time_seconds > warm.developer.development_time_seconds

    def test_calculate_cost_estimate(self, calculator):
        """Cost estimate should be calculated when enabled."""
        with_cost = calculator.calculate(8.0, 10.0, include_cost=True)
        without_cost = calculator.calculate(8.0, 10.0, include_cost=False)

        assert with_cost.estimated_cost_usd is not None
        assert with_cost.estimated_cost_usd > 0
        assert without_cost.estimated_cost_usd is None

    def test_calculate_generates_notes(self, calculator):
        """Calculator should generate helpful notes."""
        recipe = calculator.calculate(8.0, 10.0)
        assert len(recipe.notes) > 0

    def test_calculate_test_strip_times(self, calculator):
        """Test strip times should be reasonable sequence."""
        times = calculator.calculate_test_strip_times(10.0, num_strips=5)

        assert len(times) == 5
        # Should be ascending
        for i in range(1, len(times)):
            assert times[i] > times[i-1]

    def test_calculate_split_filter(self, calculator):
        """Split filter calculation should work."""
        split = calculator.calculate_split_filter_exposure(10.0, shadow_grade=5.0, highlight_grade=0.0)

        assert "shadow_exposure" in split
        assert "highlight_exposure" in split
        assert split["shadow_exposure"]["time_seconds"] + split["highlight_exposure"]["time_seconds"] == 10.0

    def test_get_developer_info(self):
        """Developer info should have standard developers."""
        info = SilverGelatinCalculator.get_developer_info()

        assert DeveloperType.DEKTOL.value in info
        assert "tone" in info[DeveloperType.DEKTOL.value]

    def test_get_troubleshooting_guide(self):
        """Troubleshooting guide should have common issues."""
        guide = SilverGelatinCalculator.get_troubleshooting_guide()

        assert "flat_prints" in guide
        assert "muddy_shadows" in guide


# =====================================================
# CYANOTYPE EXPOSURE CALCULATOR TESTS
# =====================================================

class TestUVSourceSpeeds:
    """Tests for UV source speed constants."""

    def test_sunlight_fastest(self):
        """Direct sunlight should be one of the fastest."""
        assert UV_SOURCE_SPEEDS[UVSource.DIRECT_SUNLIGHT] < UV_SOURCE_SPEEDS[UVSource.BL_TUBES]

    def test_cloudy_slowest_natural(self):
        """Cloudy should be slowest natural light."""
        assert UV_SOURCE_SPEEDS[UVSource.SUNLIGHT_CLOUDY] > UV_SOURCE_SPEEDS[UVSource.SUNLIGHT_SHADE]

    def test_all_sources_defined(self):
        """All UV sources should have speeds defined."""
        for source in list(UVSource):
            assert source in UV_SOURCE_SPEEDS


class TestCyanotypeExposureCalculator:
    """Tests for CyanotypeExposureCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create an exposure calculator."""
        return CyanotypeExposureCalculator()

    def test_calculate_default(self, calculator):
        """Default calculation should produce reasonable results."""
        result = calculator.calculate()

        assert result.exposure_minutes > 0
        assert result.exposure_seconds == result.exposure_minutes * 60

    def test_calculate_denser_negative_longer_exposure(self, calculator):
        """Denser negative should require longer exposure."""
        thin = calculator.calculate(negative_density=1.4)
        dense = calculator.calculate(negative_density=2.0)

        assert dense.exposure_minutes > thin.exposure_minutes

    def test_calculate_sunlight_vs_bl_tubes(self, calculator):
        """Sunlight should be faster than BL tubes."""
        sunlight = calculator.calculate(uv_source=UVSource.DIRECT_SUNLIGHT)
        bl = calculator.calculate(uv_source=UVSource.BL_TUBES)

        assert sunlight.exposure_minutes < bl.exposure_minutes

    def test_calculate_new_formula_faster(self, calculator):
        """New cyanotype formula should expose faster."""
        classic = calculator.calculate(formula=CyanotypeFormula.CLASSIC)
        new = calculator.calculate(formula=CyanotypeFormula.NEW)

        assert new.exposure_minutes < classic.exposure_minutes

    def test_calculate_humidity_affects_exposure(self, calculator):
        """Humidity should affect exposure time."""
        low_humidity = calculator.calculate(humidity_percent=30.0)
        high_humidity = calculator.calculate(humidity_percent=70.0)

        # Different humidity should produce different times
        assert low_humidity.exposure_minutes != high_humidity.exposure_minutes

    def test_result_has_visual_indicators(self, calculator):
        """Result should include visual exposure indicators."""
        result = calculator.calculate()

        assert result.unexposed_color is not None
        assert result.properly_exposed_color is not None
        assert result.overexposed_color is not None

    def test_result_format_time(self, calculator):
        """Format time should produce readable string."""
        result = calculator.calculate()
        formatted = result.format_time()

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_calculate_test_strip(self, calculator):
        """Test strip times should be reasonable."""
        times = calculator.calculate_test_strip(15.0, strips=5)

        assert len(times) == 5
        assert 15.0 in times  # Center exposure should be included

    def test_get_uv_sources(self):
        """UV sources list should be available."""
        sources = CyanotypeExposureCalculator.get_uv_sources()

        assert len(sources) > 0
        assert any("Sunlight" in s[1] for s in sources)


# =====================================================
# SILVER GELATIN EXPOSURE CALCULATOR TESTS
# =====================================================

class TestEnlargerLightSpeeds:
    """Tests for enlarger light source speeds."""

    def test_tungsten_baseline(self):
        """Tungsten incandescent should be baseline."""
        assert ENLARGER_LIGHT_SPEEDS[EnlargerLightSource.TUNGSTEN_INCANDESCENT] == 1.0

    def test_led_faster(self):
        """LED should be faster than tungsten."""
        assert ENLARGER_LIGHT_SPEEDS[EnlargerLightSource.LED_ENLARGER] < ENLARGER_LIGHT_SPEEDS[EnlargerLightSource.TUNGSTEN_INCANDESCENT]

    def test_all_sources_defined(self):
        """All enlarger sources should have speeds defined."""
        for source in list(EnlargerLightSource):
            assert source in ENLARGER_LIGHT_SPEEDS


class TestSilverGelatinExposureCalculator:
    """Tests for SilverGelatinExposureCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create an exposure calculator."""
        return SilverGelatinExposureCalculator()

    def test_calculate_default(self, calculator):
        """Default calculation should produce reasonable results."""
        result = calculator.calculate()

        assert result.exposure_seconds > 0
        assert result.f_stop > 0

    def test_calculate_higher_magnification_longer_exposure(self, calculator):
        """Higher magnification should require longer exposure."""
        low = calculator.calculate(enlarger_height_cm=20.0)
        high = calculator.calculate(enlarger_height_cm=60.0)

        assert high.exposure_seconds > low.exposure_seconds

    def test_calculate_includes_test_strip_times(self, calculator):
        """Result should include test strip times."""
        result = calculator.calculate()

        assert len(result.test_strip_times) > 0

    def test_calculate_dodging_burning(self, calculator):
        """Dodging/burning calculation should work."""
        dodge_time, dodge_desc = calculator.calculate_dodging_burning(10.0, -1.0)
        burn_time, burn_desc = calculator.calculate_dodging_burning(10.0, 1.0)

        assert "dodge" in dodge_desc.lower()
        assert "burn" in burn_desc.lower()

    def test_calculate_split_grade(self, calculator):
        """Split grade calculation should produce valid results."""
        split = calculator.calculate_split_grade(10.0)

        assert "shadow_exposure" in split
        assert "highlight_exposure" in split
        assert split["total_time"] == 10.0

    def test_get_filter_factors(self):
        """Filter factors should be available."""
        factors = SilverGelatinExposureCalculator.get_filter_factors()

        assert "Grade 2" in factors
        assert factors["Grade 2"] == 1.0  # Baseline


# =====================================================
# VAN DYKE & KALLITYPE EXPOSURE CALCULATOR TESTS
# =====================================================

class TestVanDykeExposureCalculator:
    """Tests for Van Dyke exposure calculator."""

    @pytest.fixture
    def calculator(self):
        return VanDykeExposureCalculator()

    def test_calculate_produces_result(self, calculator):
        """Calculation should produce valid result."""
        result = calculator.calculate()

        assert result["exposure_minutes"] > 0
        assert result["exposure_seconds"] > 0
        assert "visual_indicators" in result

    def test_van_dyke_faster_than_base(self, calculator):
        """Van Dyke should be faster than its base exposure."""
        result = calculator.calculate()
        assert result["exposure_minutes"] < 20  # Should be reasonably fast


class TestKallitypeExposureCalculator:
    """Tests for Kallitype exposure calculator."""

    @pytest.fixture
    def calculator(self):
        return KallitypeExposureCalculator()

    def test_calculate_produces_result(self, calculator):
        """Calculation should produce valid result."""
        result = calculator.calculate()

        assert result["exposure_minutes"] > 0
        assert result["exposure_seconds"] > 0

    def test_developer_affects_tone(self, calculator):
        """Different developers should produce different expected tones."""
        oxalate = calculator.calculate(developer_type="potassium_oxalate")
        borax = calculator.calculate(developer_type="borax")

        assert oxalate["expected_tone"] != borax["expected_tone"]
