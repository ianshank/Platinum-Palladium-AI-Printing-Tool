"""
Unit tests for chemistry calculator module.

Tests coating chemistry calculations based on Bostick-Sullivan formulas.
"""

import pytest

from ptpd_calibration.chemistry import (
    METAL_MIX_RATIOS,
    ChemistryCalculator,
    ChemistryRecipe,
    CoatingMethod,
    MetalMix,
    PaperAbsorbency,
)
from ptpd_calibration.config import ChemistrySettings


class TestMetalMixRatios:
    """Tests for metal mix presets."""

    def test_pure_palladium_ratio(self):
        """Pure palladium should have 0% platinum."""
        assert METAL_MIX_RATIOS[MetalMix.PURE_PALLADIUM] == 0.0

    def test_pure_platinum_ratio(self):
        """Pure platinum should have 100% platinum."""
        assert METAL_MIX_RATIOS[MetalMix.PURE_PLATINUM] == 1.0

    def test_classic_mix_ratio(self):
        """Classic mix should be 50/50."""
        assert METAL_MIX_RATIOS[MetalMix.CLASSIC_MIX] == 0.5

    def test_warm_mix_ratio(self):
        """Warm mix should be 25% platinum."""
        assert METAL_MIX_RATIOS[MetalMix.WARM_MIX] == 0.25

    def test_cool_mix_ratio(self):
        """Cool mix should be 75% platinum."""
        assert METAL_MIX_RATIOS[MetalMix.COOL_MIX] == 0.75

    def test_all_presets_defined(self):
        """All MetalMix values should have ratios defined."""
        for mix in list(MetalMix):
            assert mix in METAL_MIX_RATIOS


class TestChemistryRecipe:
    """Tests for ChemistryRecipe dataclass."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a sample recipe for testing."""
        return ChemistryRecipe(
            print_width_inches=8.0,
            print_height_inches=10.0,
            coating_width_inches=7.0,
            coating_height_inches=9.0,
            coating_area_sq_inches=63.0,
            ferric_oxalate_drops=12.0,
            ferric_oxalate_contrast_drops=0.0,
            palladium_drops=12.0,
            platinum_drops=0.0,
            na2_drops=3.0,
            total_drops=27.0,
            ferric_oxalate_ml=0.6,
            ferric_oxalate_contrast_ml=0.0,
            palladium_ml=0.6,
            platinum_ml=0.0,
            na2_ml=0.15,
            total_ml=1.35,
            platinum_ratio=0.0,
            palladium_ratio=1.0,
            paper_absorbency=PaperAbsorbency.MEDIUM,
            coating_method=CoatingMethod.BRUSH,
            contrast_boost=0.0,
            estimated_cost_usd=1.50,
            notes=["Test note"],
        )

    def test_recipe_to_dict(self, sample_recipe):
        """Recipe should convert to dictionary correctly."""
        d = sample_recipe.to_dict()
        assert d["print_dimensions"]["width_inches"] == 8.0
        assert d["print_dimensions"]["height_inches"] == 10.0
        assert d["drops"]["ferric_oxalate_1"] == 12.0
        assert d["drops"]["palladium"] == 12.0
        assert d["metal_ratios"]["platinum_percent"] == 0.0
        assert d["metal_ratios"]["palladium_percent"] == 100.0
        assert d["settings"]["paper_absorbency"] == "medium"

    def test_recipe_format(self, sample_recipe):
        """Recipe should format as readable text."""
        text = sample_recipe.format_recipe()
        assert "8.0" in text  # Width
        assert "10.0" in text  # Height
        assert "Ferric Oxalate" in text
        assert "Palladium" in text
        assert "drops" in text
        assert "ml" in text

    def test_recipe_notes_included_in_format(self, sample_recipe):
        """Notes should appear in formatted recipe."""
        text = sample_recipe.format_recipe()
        assert "Test note" in text


class TestChemistryCalculator:
    """Tests for ChemistryCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with default settings."""
        return ChemistryCalculator()

    @pytest.fixture
    def custom_calculator(self):
        """Create a calculator with custom settings."""
        settings = ChemistrySettings(
            drops_per_square_inch=0.5,
            drops_per_ml=20.0,
            default_platinum_ratio=0.5,
            default_na2_drops_ratio=0.25,
        )
        return ChemistryCalculator(settings=settings)

    def test_calculate_8x10_default(self, calculator):
        """8x10 print with defaults should produce reasonable results."""
        recipe = calculator.calculate(8.0, 10.0)

        # Check dimensions
        assert recipe.print_width_inches == 8.0
        assert recipe.print_height_inches == 10.0
        assert recipe.coating_area_sq_inches > 0

        # Check drops are positive
        assert recipe.ferric_oxalate_drops > 0
        assert recipe.total_drops > 0

        # Check metal ratio
        assert recipe.palladium_ratio == 1.0  # Default is pure palladium
        assert recipe.platinum_ratio == 0.0

    def test_calculate_small_print(self, calculator):
        """Small print should use less chemistry."""
        small = calculator.calculate(4.0, 5.0)
        large = calculator.calculate(16.0, 20.0)

        assert small.total_drops < large.total_drops
        assert small.total_ml < large.total_ml

    def test_calculate_pure_platinum(self, calculator):
        """Pure platinum recipe should have no palladium."""
        recipe = calculator.calculate(8.0, 10.0, platinum_ratio=1.0)

        assert recipe.platinum_ratio == 1.0
        assert recipe.palladium_ratio == 0.0
        assert recipe.platinum_drops > 0
        assert recipe.palladium_drops == 0

    def test_calculate_mixed_metals(self, calculator):
        """Mixed metal recipe should split correctly."""
        recipe = calculator.calculate(8.0, 10.0, platinum_ratio=0.5)

        assert recipe.platinum_ratio == 0.5
        assert recipe.palladium_ratio == 0.5
        # Platinum and palladium drops should be equal
        assert abs(recipe.platinum_drops - recipe.palladium_drops) < 0.01

    def test_calculate_contrast_boost(self, calculator):
        """Contrast boost should use FO#2."""
        no_contrast = calculator.calculate(8.0, 10.0, contrast_boost=0.0)
        with_contrast = calculator.calculate(8.0, 10.0, contrast_boost=0.5)

        assert no_contrast.ferric_oxalate_contrast_drops == 0
        assert with_contrast.ferric_oxalate_contrast_drops > 0
        # Total FO should be split
        total_fo_with = (
            with_contrast.ferric_oxalate_drops + with_contrast.ferric_oxalate_contrast_drops
        )
        assert abs(total_fo_with - no_contrast.ferric_oxalate_drops) < 0.1

    def test_calculate_high_absorbency_paper(self, calculator):
        """High absorbency paper should use more chemistry."""
        medium = calculator.calculate(8.0, 10.0, paper_absorbency=PaperAbsorbency.MEDIUM)
        high = calculator.calculate(8.0, 10.0, paper_absorbency=PaperAbsorbency.HIGH)

        assert high.total_drops > medium.total_drops

    def test_calculate_low_absorbency_paper(self, calculator):
        """Low absorbency paper should use less chemistry."""
        medium = calculator.calculate(8.0, 10.0, paper_absorbency=PaperAbsorbency.MEDIUM)
        low = calculator.calculate(8.0, 10.0, paper_absorbency=PaperAbsorbency.LOW)

        assert low.total_drops < medium.total_drops

    def test_calculate_rod_coating(self, calculator):
        """Rod coating should use less chemistry than brush."""
        brush = calculator.calculate(8.0, 10.0, coating_method=CoatingMethod.BRUSH)
        rod = calculator.calculate(8.0, 10.0, coating_method=CoatingMethod.ROD)

        assert rod.total_drops < brush.total_drops

    def test_calculate_na2_ratio(self, calculator):
        """Custom Na2 ratio should be applied."""
        low_na2 = calculator.calculate(8.0, 10.0, na2_ratio=0.1)
        high_na2 = calculator.calculate(8.0, 10.0, na2_ratio=0.4)

        assert low_na2.na2_drops < high_na2.na2_drops

    def test_calculate_custom_margin(self, calculator):
        """Custom margin should affect coating area."""
        small_margin = calculator.calculate(8.0, 10.0, margin_inches=0.25)
        large_margin = calculator.calculate(8.0, 10.0, margin_inches=1.0)

        assert small_margin.coating_area_sq_inches > large_margin.coating_area_sq_inches

    def test_calculate_cost_estimate(self, calculator):
        """Cost estimate should be calculated when enabled."""
        with_cost = calculator.calculate(8.0, 10.0, include_cost=True)
        without_cost = calculator.calculate(8.0, 10.0, include_cost=False)

        assert with_cost.estimated_cost_usd is not None
        assert with_cost.estimated_cost_usd > 0
        assert without_cost.estimated_cost_usd is None

    def test_calculate_platinum_more_expensive(self, calculator):
        """Platinum prints should cost more than palladium."""
        palladium = calculator.calculate(8.0, 10.0, platinum_ratio=0.0)
        platinum = calculator.calculate(8.0, 10.0, platinum_ratio=1.0)

        assert platinum.estimated_cost_usd > palladium.estimated_cost_usd

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

    def test_calculate_invalid_platinum_ratio_raises(self, calculator):
        """Invalid platinum ratio should raise ValueError."""
        with pytest.raises(ValueError):
            calculator.calculate(8.0, 10.0, platinum_ratio=1.5)
        with pytest.raises(ValueError):
            calculator.calculate(8.0, 10.0, platinum_ratio=-0.1)

    def test_calculate_invalid_contrast_raises(self, calculator):
        """Invalid contrast boost should raise ValueError."""
        with pytest.raises(ValueError):
            calculator.calculate(8.0, 10.0, contrast_boost=1.5)

    def test_calculate_from_preset(self, calculator):
        """Calculate from preset should use correct ratio."""
        pure_pd = calculator.calculate_from_preset(8.0, 10.0, MetalMix.PURE_PALLADIUM)
        pure_pt = calculator.calculate_from_preset(8.0, 10.0, MetalMix.PURE_PLATINUM)

        assert pure_pd.platinum_ratio == 0.0
        assert pure_pt.platinum_ratio == 1.0

    def test_calculate_from_preset_with_options(self, calculator):
        """Preset calculation should accept other options."""
        recipe = calculator.calculate_from_preset(
            8.0,
            10.0,
            MetalMix.CLASSIC_MIX,
            paper_absorbency=PaperAbsorbency.HIGH,
            coating_method=CoatingMethod.ROD,
            contrast_boost=0.25,
        )

        assert recipe.platinum_ratio == 0.5
        assert recipe.paper_absorbency == PaperAbsorbency.HIGH
        assert recipe.coating_method == CoatingMethod.ROD
        assert recipe.contrast_boost == 0.25

    def test_scale_recipe(self, calculator):
        """Scaling recipe should multiply all amounts."""
        original = calculator.calculate(8.0, 10.0)
        scaled = calculator.scale_recipe(original, 2.0)

        assert abs(scaled.total_drops - original.total_drops * 2) < 0.01
        assert abs(scaled.ferric_oxalate_drops - original.ferric_oxalate_drops * 2) < 0.01
        assert abs(scaled.palladium_drops - original.palladium_drops * 2) < 0.01

    def test_scale_recipe_half(self, calculator):
        """Scaling by 0.5 should halve amounts."""
        original = calculator.calculate(8.0, 10.0)
        scaled = calculator.scale_recipe(original, 0.5)

        assert abs(scaled.total_drops - original.total_drops * 0.5) < 0.01

    def test_scale_recipe_invalid_factor_raises(self, calculator):
        """Invalid scale factor should raise ValueError."""
        original = calculator.calculate(8.0, 10.0)
        with pytest.raises(ValueError):
            calculator.scale_recipe(original, 0)
        with pytest.raises(ValueError):
            calculator.scale_recipe(original, -1)

    def test_scale_recipe_adds_note(self, calculator):
        """Scaled recipe should note the scaling."""
        original = calculator.calculate(8.0, 10.0)
        scaled = calculator.scale_recipe(original, 2.0)

        assert any("Scaled" in note for note in scaled.notes)

    def test_get_standard_sizes(self):
        """Standard sizes should be available."""
        sizes = ChemistryCalculator.get_standard_sizes()

        assert "4x5" in sizes
        assert "5x7" in sizes
        assert "8x10" in sizes
        assert "11x14" in sizes
        assert "16x20" in sizes

        # Check values
        assert sizes["8x10"] == (8, 10)
        assert sizes["4x5"] == (4, 5)

    def test_custom_settings(self, custom_calculator):
        """Calculator should use custom settings."""
        recipe = custom_calculator.calculate(8.0, 10.0)

        # Custom settings have 0.5 drops/sq inch vs default 0.465
        # So total should be slightly higher
        default_calc = ChemistryCalculator()
        default_recipe = default_calc.calculate(8.0, 10.0)

        # The custom calculator has higher drops per sq inch
        assert recipe.total_drops > default_recipe.total_drops


class TestChemistryEdgeCases:
    """Tests for edge cases in chemistry calculations."""

    @pytest.fixture
    def calculator(self):
        return ChemistryCalculator()

    def test_very_small_print(self, calculator):
        """Very small prints should still work."""
        recipe = calculator.calculate(2.0, 2.0)
        assert recipe.total_drops > 0
        assert recipe.coating_area_sq_inches > 0

    def test_very_large_print(self, calculator):
        """Very large prints should work."""
        recipe = calculator.calculate(30.0, 40.0)
        assert recipe.total_drops > 0
        # Should note that it's a large print
        assert any("Large" in note or "large" in note for note in recipe.notes)

    def test_zero_na2(self, calculator):
        """Zero Na2 should be allowed."""
        recipe = calculator.calculate(8.0, 10.0, na2_ratio=0.0)
        assert recipe.na2_drops == 0
        assert recipe.na2_ml == 0

    def test_full_contrast_boost(self, calculator):
        """Full contrast boost should use all FO#2."""
        recipe = calculator.calculate(8.0, 10.0, contrast_boost=1.0)
        assert recipe.ferric_oxalate_drops == 0
        assert recipe.ferric_oxalate_contrast_drops > 0

    def test_ml_conversion_accuracy(self, calculator):
        """ML conversion should be accurate based on drops/ml setting."""
        recipe = calculator.calculate(8.0, 10.0)
        settings = calculator.settings

        expected_ml = recipe.total_drops / settings.drops_per_ml
        assert abs(recipe.total_ml - expected_ml) < 0.001

    def test_metal_balance(self, calculator):
        """Metal drops should approximately equal FO drops."""
        recipe = calculator.calculate(8.0, 10.0, contrast_boost=0.0, na2_ratio=0.0)

        total_fo = recipe.ferric_oxalate_drops + recipe.ferric_oxalate_contrast_drops
        total_metal = recipe.palladium_drops + recipe.platinum_drops

        # Should be approximately equal
        assert abs(total_fo - total_metal) < 0.1


class TestPaperAbsorbencyEnum:
    """Tests for PaperAbsorbency enum."""

    def test_low_absorbency_value(self):
        assert PaperAbsorbency.LOW.value == "low"

    def test_medium_absorbency_value(self):
        assert PaperAbsorbency.MEDIUM.value == "medium"

    def test_high_absorbency_value(self):
        assert PaperAbsorbency.HIGH.value == "high"

    def test_from_string(self):
        assert PaperAbsorbency("low") == PaperAbsorbency.LOW
        assert PaperAbsorbency("medium") == PaperAbsorbency.MEDIUM
        assert PaperAbsorbency("high") == PaperAbsorbency.HIGH


class TestCoatingMethodEnum:
    """Tests for CoatingMethod enum."""

    def test_brush_value(self):
        assert CoatingMethod.BRUSH.value == "brush"

    def test_rod_value(self):
        assert CoatingMethod.ROD.value == "rod"

    def test_puddle_pusher_value(self):
        assert CoatingMethod.PUDDLE_PUSHER.value == "puddle_pusher"

    def test_from_string(self):
        assert CoatingMethod("brush") == CoatingMethod.BRUSH
        assert CoatingMethod("rod") == CoatingMethod.ROD


class TestChemistrySettings:
    """Tests for ChemistrySettings configuration."""

    def test_default_drops_per_square_inch(self):
        settings = ChemistrySettings()
        assert settings.drops_per_square_inch == 0.465

    def test_default_drops_per_ml(self):
        settings = ChemistrySettings()
        assert settings.drops_per_ml == 20.0

    def test_absorbency_multipliers(self):
        settings = ChemistrySettings()
        assert settings.low_absorbency_multiplier < settings.medium_absorbency_multiplier
        assert settings.medium_absorbency_multiplier < settings.high_absorbency_multiplier

    def test_coating_method_multipliers(self):
        settings = ChemistrySettings()
        assert settings.rod_coating_multiplier < settings.brush_coating_multiplier

    def test_cost_settings(self):
        settings = ChemistrySettings()
        # Platinum should be more expensive than palladium
        assert settings.platinum_cost_per_ml > settings.palladium_cost_per_ml
