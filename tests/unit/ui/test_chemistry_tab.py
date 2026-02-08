from unittest.mock import patch

from ptpd_calibration.ui.tabs.chemistry import build_chemistry_tab


def test_chemistry_tab_structure():
    with (
        patch("gradio.TabItem"),
        patch("gradio.Markdown"),
        patch("gradio.Row"),
        patch("gradio.Column"),
        patch("gradio.Button") as MockButton,
        patch("gradio.Number"),
        patch("gradio.Slider"),
        patch("gradio.HTML"),
        patch("gradio.Dropdown"),
        patch("gradio.Textbox"),
        patch("gradio.JSON"),
    ):
        build_chemistry_tab()

        # Verify key buttons exist
        # We expect size buttons (4x5, etc) + Calculate + Copy + Log
        assert MockButton.call_count >= 8


# We can test the internal logic by mocking the module functions if we refactored them out?
# The logic `calculate` IS nested.
# But `update_size` is simple lambda.
# `update_viz` handles HTML generation.


def test_chemistry_logic_coverage():
    # Since we can't easily reach nested functions, we might need to refactor them out in the future.
    # For now, we rely on `test_chemistry_calculator.py` for core logic coverage.
    # This test just ensures the UI code is valid Python and calls Gradio correctly.
    with (
        patch("gradio.TabItem"),
        patch("gradio.Markdown"),
        patch("gradio.Row"),
        patch("gradio.Column"),
        patch("gradio.Button"),
        patch("gradio.Number"),
        patch("gradio.Slider"),
        patch("gradio.HTML"),
        patch("gradio.Dropdown"),
        patch("gradio.Textbox"),
        patch("gradio.JSON"),
    ):
        build_chemistry_tab()


def test_calculate_recipe_ui_attribute_access():
    """Test that calculate_recipe_ui uses correct attributes on ChemistryRecipe."""
    from ptpd_calibration.ui.tabs.chemistry import calculate_recipe_ui

    # We use real calculator here to ensure integration works with real data class
    # This prevents regression of 'AttributeError: ferric_oxalate_1'

    html, text, data = calculate_recipe_ui(
        w=8, h=10, pt_ratio=50, absorbency="medium", method="brush", cont=0, na2_val=0
    )

    assert "Error:" not in html
    assert "Total Drops" in html
    assert "FO#1" in html
    assert isinstance(data, dict)
    assert "drops" in data
