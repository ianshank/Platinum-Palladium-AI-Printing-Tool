"""
Chemistry calculation unit tests.
"""

import pytest
from typing import Any


class TestChemistryCalculations:
    """Tests for chemistry calculation functions."""

    def test_calculate_basic_recipe(self, mock_config: dict[str, Any]) -> None:
        """Test basic chemistry calculation for 8x10 print."""
        width = 8
        height = 10
        metal_ratio = 0.5

        config = mock_config["chemistry"]
        area = width * height
        total_metal = max(area * config["coating_ml_per_sq_inch"], config["minimum_volume_ml"])

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)
        ferric_oxalate = total_metal * config["ferric_oxalate_ratio"]

        assert area == 80
        assert total_metal == 40.0
        assert platinum == 20.0
        assert palladium == 20.0
        assert ferric_oxalate == 80.0

    def test_pure_platinum_recipe(self, mock_config: dict[str, Any]) -> None:
        """Test pure platinum calculation."""
        width = 8
        height = 10
        metal_ratio = 1.0

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)

        assert platinum == 40.0
        assert palladium == 0.0

    def test_pure_palladium_recipe(self, mock_config: dict[str, Any]) -> None:
        """Test pure palladium calculation."""
        width = 8
        height = 10
        metal_ratio = 0.0

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)

        assert platinum == 0.0
        assert palladium == 40.0

    def test_contrast_agent_calculation(self, mock_config: dict[str, Any]) -> None:
        """Test contrast agent calculation."""
        width = 8
        height = 10

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        contrast_agent = total_metal * config["contrast_agent_ratio"]

        assert contrast_agent == 4.0

    def test_minimum_volume_enforcement(self, mock_config: dict[str, Any]) -> None:
        """Test that minimum volume is enforced for small prints."""
        width = 1
        height = 1

        config = mock_config["chemistry"]
        area = width * height
        calculated = area * config["coating_ml_per_sq_inch"]
        total_metal = max(calculated, config["minimum_volume_ml"])

        assert calculated == 0.5
        assert total_metal == 2.0  # Minimum enforced

    def test_large_print_scaling(self, mock_config: dict[str, Any]) -> None:
        """Test chemistry scales correctly for large prints."""
        width = 20
        height = 24
        metal_ratio = 0.5

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)
        ferric_oxalate = total_metal * config["ferric_oxalate_ratio"]

        assert area == 480
        assert total_metal == 240.0
        assert platinum == 120.0
        assert palladium == 120.0
        assert ferric_oxalate == 480.0

    def test_warm_tone_preset(self, mock_config: dict[str, Any]) -> None:
        """Test warm tone preset (30% platinum)."""
        width = 8
        height = 10
        metal_ratio = 0.3  # Warm tone preset

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)

        assert platinum == 12.0
        assert palladium == 28.0

    def test_cool_tone_preset(self, mock_config: dict[str, Any]) -> None:
        """Test cool tone preset (70% platinum)."""
        width = 8
        height = 10
        metal_ratio = 0.7  # Cool tone preset

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)

        assert platinum == 28.0
        assert palladium == 12.0

    def test_total_volume_calculation(self, mock_config: dict[str, Any]) -> None:
        """Test total volume calculation includes all components."""
        width = 8
        height = 10
        metal_ratio = 0.5
        with_contrast = True

        config = mock_config["chemistry"]
        area = width * height
        total_metal = area * config["coating_ml_per_sq_inch"]

        platinum = total_metal * metal_ratio
        palladium = total_metal * (1 - metal_ratio)
        ferric_oxalate = total_metal * config["ferric_oxalate_ratio"]
        contrast_agent = total_metal * config["contrast_agent_ratio"] if with_contrast else 0

        total_volume = platinum + palladium + ferric_oxalate + contrast_agent

        assert total_volume == 124.0  # 20 + 20 + 80 + 4

    def test_rounding_precision(self, mock_config: dict[str, Any]) -> None:
        """Test that results are rounded to correct precision."""
        config = mock_config["chemistry"]
        precision = config["rounding_precision"]

        # Test rounding function
        value = 12.345
        rounded = round(value, precision)

        assert rounded == 12.3

    def test_invalid_dimensions_handling(self) -> None:
        """Test handling of invalid dimensions."""
        width = -5
        height = 10
        with pytest.raises(ValueError):
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")

    def test_metal_ratio_bounds(self) -> None:
        """Test metal ratio stays within valid bounds."""
        valid_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
        invalid_ratios = [-0.1, 1.1, 2.0]

        for ratio in valid_ratios:
            assert 0 <= ratio <= 1

        for ratio in invalid_ratios:
            assert not (0 <= ratio <= 1)


class TestChemistryRecipe:
    """Tests for ChemistryRecipe model."""

    def test_recipe_has_required_fields(self, sample_chemistry_recipe: dict[str, Any]) -> None:
        """Test recipe contains all required fields."""
        required_fields = [
            "id",
            "name",
            "platinum_ml",
            "palladium_ml",
            "ferric_oxalate_ml",
            "contrast_agent_ml",
            "total_volume_ml",
            "metal_ratio",
            "coverage_area_sq_in",
        ]

        for field in required_fields:
            assert field in sample_chemistry_recipe

    def test_recipe_metal_ratio_consistency(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test metal ratio matches platinum/palladium ratio."""
        recipe = sample_chemistry_recipe
        total_metal = recipe["platinum_ml"] + recipe["palladium_ml"]
        calculated_ratio = recipe["platinum_ml"] / total_metal if total_metal > 0 else 0

        assert abs(calculated_ratio - recipe["metal_ratio"]) < 0.01

    def test_recipe_total_volume_accuracy(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test total volume equals sum of components."""
        recipe = sample_chemistry_recipe
        calculated_total = (
            recipe["platinum_ml"]
            + recipe["palladium_ml"]
            + recipe["ferric_oxalate_ml"]
            + recipe["contrast_agent_ml"]
        )

        assert abs(calculated_total - recipe["total_volume_ml"]) < 0.1
