"""
Unit tests for Calibration Wizard Step 3 - Linearization UI Refactor.

Tests cover:
- Linearization mode configuration
- Mode change handlers
- Validation logic
- Paper preset selection
- Curve generation for all modes
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from ptpd_calibration.ui.tabs.calibration_wizard import (
    # Mode configuration
    WizardLinearizationMode,
    LinearizationModeConfig,
    LINEARIZATION_MODES,
    get_linearization_mode_choices,
    get_mode_by_label,
    get_mode_value_by_label,
    # Strategy configuration
    get_strategy_choices,
    get_strategy_labels,
    get_strategy_value_by_label,
    # Target configuration
    get_target_choices,
    get_target_labels,
    get_target_value_by_label,
    # Paper configuration
    get_paper_preset_choices,
    get_paper_chemistry_notes,
    # Validation
    wizard_is_valid_config,
    # Handlers
    wizard_on_mode_change,
    wizard_on_paper_change,
    wizard_on_config_change,
)
from ptpd_calibration.curves.linearization import LinearizationMethod, TargetResponse


class TestLinearizationModeConfiguration:
    """Tests for linearization mode configuration."""

    def test_all_modes_defined(self):
        """Verify all wizard linearization modes are defined."""
        expected_modes = {
            "single_curve",
            "multi_curve",
            "use_existing",
            "no_linearization",
        }
        assert set(LINEARIZATION_MODES.keys()) == expected_modes

    def test_mode_enum_values(self):
        """Verify enum values match configuration keys."""
        assert WizardLinearizationMode.SINGLE_CURVE.value == "single_curve"
        assert WizardLinearizationMode.MULTI_CURVE.value == "multi_curve"
        assert WizardLinearizationMode.USE_EXISTING.value == "use_existing"
        assert WizardLinearizationMode.NO_LINEARIZATION.value == "no_linearization"

    def test_get_linearization_mode_choices(self):
        """Verify mode choices return labels."""
        choices = get_linearization_mode_choices()
        assert len(choices) == 4
        assert "Single-curve linearization (recommended)" in choices
        assert "Multi-curve / split-tone (advanced)" in choices
        assert "Use existing profile" in choices
        assert "No linearization (straight curve)" in choices

    def test_get_mode_by_label_single_curve(self):
        """Test getting single-curve mode by label."""
        mode = get_mode_by_label("Single-curve linearization (recommended)")
        assert mode is not None
        assert mode.value == "single_curve"
        assert mode.requires_target is True
        assert mode.requires_strategy is True
        assert mode.requires_paper_preset is True
        assert mode.requires_existing_profile is False
        assert mode.advanced is False

    def test_get_mode_by_label_multi_curve(self):
        """Test getting multi-curve mode by label."""
        mode = get_mode_by_label("Multi-curve / split-tone (advanced)")
        assert mode is not None
        assert mode.value == "multi_curve"
        assert mode.advanced is True

    def test_get_mode_by_label_use_existing(self):
        """Test getting use existing profile mode by label."""
        mode = get_mode_by_label("Use existing profile")
        assert mode is not None
        assert mode.value == "use_existing"
        assert mode.requires_target is False
        assert mode.requires_strategy is False
        assert mode.requires_existing_profile is True

    def test_get_mode_by_label_no_linearization(self):
        """Test getting no linearization mode by label."""
        mode = get_mode_by_label("No linearization (straight curve)")
        assert mode is not None
        assert mode.value == "no_linearization"
        assert mode.requires_target is False
        assert mode.requires_strategy is False

    def test_get_mode_by_label_invalid(self):
        """Test getting mode with invalid label returns None."""
        mode = get_mode_by_label("Invalid Mode Label")
        assert mode is None

    def test_get_mode_value_by_label(self):
        """Test getting mode value by label."""
        value = get_mode_value_by_label("Single-curve linearization (recommended)")
        assert value == "single_curve"

        value = get_mode_value_by_label("Invalid")
        assert value is None


class TestStrategyConfiguration:
    """Tests for curve strategy configuration."""

    def test_get_strategy_choices(self):
        """Verify strategy choices are correctly mapped."""
        choices = get_strategy_choices()
        assert len(choices) == 5

        # Verify each choice has label and value
        for label, value in choices:
            assert isinstance(label, str)
            assert isinstance(value, str)
            # Value should be valid LinearizationMethod
            assert value in [m.value for m in LinearizationMethod]

    def test_get_strategy_labels(self):
        """Verify strategy labels are returned."""
        labels = get_strategy_labels()
        assert "Smooth spline (recommended)" in labels
        assert "Polynomial fit" in labels
        assert "Iterative refinement" in labels

    def test_get_strategy_value_by_label(self):
        """Test getting strategy value by label."""
        assert get_strategy_value_by_label("Smooth spline (recommended)") == LinearizationMethod.SPLINE_FIT.value
        assert get_strategy_value_by_label("Polynomial fit") == LinearizationMethod.POLYNOMIAL_FIT.value
        assert get_strategy_value_by_label("Invalid") is None


class TestTargetConfiguration:
    """Tests for target response configuration."""

    def test_get_target_choices(self):
        """Verify target choices are correctly mapped."""
        choices = get_target_choices()
        assert len(choices) == 5

        # Verify each choice has label and value
        for label, value in choices:
            assert isinstance(label, str)
            assert isinstance(value, str)
            # Value should be valid TargetResponse
            assert value in [t.value for t in TargetResponse]

    def test_get_target_labels(self):
        """Verify target labels are returned."""
        labels = get_target_labels()
        assert "Even tonal steps (linear)" in labels
        assert "Match digital gamma 2.2 (sRGB)" in labels
        assert "Preserve paper white (highlights)" in labels

    def test_get_target_value_by_label(self):
        """Test getting target value by label."""
        assert get_target_value_by_label("Even tonal steps (linear)") == TargetResponse.LINEAR.value
        assert get_target_value_by_label("Match digital gamma 2.2 (sRGB)") == TargetResponse.GAMMA_22.value
        assert get_target_value_by_label("Invalid") is None


class TestPaperPresetConfiguration:
    """Tests for paper preset configuration."""

    def test_get_paper_preset_choices(self):
        """Verify paper preset choices include papers and custom option."""
        choices = get_paper_preset_choices()
        assert len(choices) > 0
        assert "Other / custom" in choices
        # Should include at least some built-in papers
        assert any("Arches" in c for c in choices)

    def test_get_paper_chemistry_notes_builtin(self):
        """Test getting chemistry notes for built-in paper."""
        notes = get_paper_chemistry_notes("Arches Platine")
        # Should return some chemistry information
        assert isinstance(notes, str)

    def test_get_paper_chemistry_notes_custom(self):
        """Test getting chemistry notes for custom option."""
        notes = get_paper_chemistry_notes("Other / custom")
        assert notes == ""

    def test_get_paper_chemistry_notes_invalid(self):
        """Test getting chemistry notes for invalid paper."""
        notes = get_paper_chemistry_notes("Invalid Paper Name")
        assert notes == ""


class TestValidation:
    """Tests for wizard configuration validation."""

    def test_valid_single_curve_config(self):
        """Test validation passes for valid single-curve config."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is True
        assert msg == ""

    def test_invalid_missing_mode(self):
        """Test validation fails without mode."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Invalid Mode",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "mode" in msg.lower()

    def test_invalid_missing_curve_name(self):
        """Test validation fails without curve name."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="",
        )
        assert is_valid is False
        assert "name" in msg.lower()

    def test_invalid_missing_target_for_single_curve(self):
        """Test validation fails without target for single-curve mode."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "target" in msg.lower()

    def test_invalid_missing_strategy_for_single_curve(self):
        """Test validation fails without strategy for single-curve mode."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "strategy" in msg.lower()

    def test_valid_use_existing_config(self):
        """Test validation passes for valid use-existing config."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Use existing profile",
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile="Existing Curve",
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is True

    def test_invalid_use_existing_no_profile(self):
        """Test validation fails for use-existing without profile."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Use existing profile",
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "profile" in msg.lower()

    def test_invalid_use_existing_no_curves_available(self):
        """Test validation fails for use-existing with 'No curves available'."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Use existing profile",
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile="No curves available",
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "profile" in msg.lower()

    def test_valid_no_linearization_config(self):
        """Test validation passes for no-linearization config."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="No linearization (straight curve)",
            target_label="",
            strategy_label="",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is True

    def test_invalid_custom_paper_without_chemistry(self):
        """Test validation fails for custom paper without chemistry notes."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Other / custom",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "chemistry" in msg.lower()

    def test_valid_custom_paper_with_chemistry(self):
        """Test validation passes for custom paper with chemistry notes."""
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Other / custom",
            existing_profile=None,
            custom_chemistry="50/50 Pt/Pd, 5 drops Na2",
            curve_name="Test Curve",
        )
        assert is_valid is True


class TestModeChangeHandler:
    """Tests for mode change handler."""

    def test_single_curve_mode_visibility(self):
        """Test visibility updates for single-curve mode."""
        results = wizard_on_mode_change("Single-curve linearization (recommended)")
        assert len(results) == 7

        # Target should be visible
        assert results[0]["visible"] is True
        # Strategy should be visible
        assert results[1]["visible"] is True
        # Paper preset should be visible
        assert results[2]["visible"] is True
        # Existing profile should be hidden
        assert results[3]["visible"] is False
        # Advanced options should be hidden
        assert results[4]["visible"] is False
        # Curve name should be visible
        assert results[5]["visible"] is True

    def test_use_existing_mode_visibility(self):
        """Test visibility updates for use-existing mode."""
        results = wizard_on_mode_change("Use existing profile")
        assert len(results) == 7

        # Target should be hidden
        assert results[0]["visible"] is False
        # Strategy should be hidden
        assert results[1]["visible"] is False
        # Paper preset should be hidden
        assert results[2]["visible"] is False
        # Existing profile should be visible
        assert results[3]["visible"] is True
        # Curve name should be hidden (using existing name)
        assert results[5]["visible"] is False

    def test_multi_curve_mode_visibility(self):
        """Test visibility updates for multi-curve mode."""
        results = wizard_on_mode_change("Multi-curve / split-tone (advanced)")
        assert len(results) == 7

        # Target should be visible
        assert results[0]["visible"] is True
        # Strategy should be visible
        assert results[1]["visible"] is True
        # Advanced options should be visible for advanced mode
        assert results[4]["visible"] is True

    def test_no_linearization_mode_visibility(self):
        """Test visibility updates for no-linearization mode."""
        results = wizard_on_mode_change("No linearization (straight curve)")
        assert len(results) == 7

        # Target should be hidden
        assert results[0]["visible"] is False
        # Strategy should be hidden
        assert results[1]["visible"] is False
        # Paper preset should be visible
        assert results[2]["visible"] is True

    def test_invalid_mode_defaults(self):
        """Test default visibility for invalid mode."""
        results = wizard_on_mode_change("Invalid Mode")
        assert len(results) == 7
        # Should default to showing most controls
        assert results[0]["visible"] is True


class TestPaperChangeHandler:
    """Tests for paper change handler."""

    def test_custom_paper_shows_chemistry_input(self):
        """Test custom paper selection shows chemistry input."""
        custom_visible, notes_update = wizard_on_paper_change("Other / custom")
        assert custom_visible["visible"] is True
        assert custom_visible["interactive"] is True
        assert notes_update["value"] == ""

    def test_builtin_paper_hides_chemistry_input(self):
        """Test built-in paper hides chemistry input."""
        custom_visible, notes_update = wizard_on_paper_change("Arches Platine")
        assert custom_visible["visible"] is False
        # Should have some chemistry notes
        assert isinstance(notes_update["value"], str)


class TestConfigChangeHandler:
    """Tests for config change handler."""

    def test_valid_config_enables_button(self):
        """Test valid config enables generate button."""
        button_update, validation_msg = wizard_on_config_change(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert button_update["interactive"] is True
        assert "valid" in validation_msg.lower()

    def test_invalid_config_disables_button(self):
        """Test invalid config disables generate button."""
        button_update, validation_msg = wizard_on_config_change(
            mode_label="Single-curve linearization (recommended)",
            target_label="",  # Missing target
            strategy_label="Smooth spline (recommended)",
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert button_update["interactive"] is False
        assert "target" in validation_msg.lower()


class TestBuildWizardTab:
    """Tests for building the wizard tab."""

    def test_wizard_tab_builds_without_error(self):
        """Test that the wizard tab can be built without errors."""
        with patch('gradio.TabItem'), \
             patch('gradio.Markdown'), \
             patch('gradio.State') as MockState, \
             patch('gradio.Group'), \
             patch('gradio.Row'), \
             patch('gradio.Column'), \
             patch('gradio.Image'), \
             patch('gradio.Dropdown') as MockDropdown, \
             patch('gradio.Slider'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Button') as MockButton, \
             patch('gradio.Plot'), \
             patch('gradio.Dataframe'), \
             patch('gradio.Textbox'), \
             patch('gradio.Number'), \
             patch('gradio.Radio'), \
             patch('gradio.File'), \
             patch('gradio.Accordion'):

            from ptpd_calibration.ui.tabs.calibration_wizard import build_calibration_wizard_tab
            build_calibration_wizard_tab()

            # Verify key components were created
            assert MockState.call_count >= 5  # Multiple state components
            assert MockDropdown.call_count >= 5  # Mode, target, strategy, paper, existing profile
            assert MockButton.call_count >= 5  # Various navigation buttons

    def test_linearization_mode_dropdown_created_with_correct_choices(self):
        """Test that linearization mode dropdown has correct choices."""
        choices = get_linearization_mode_choices()
        assert len(choices) == 4
        # All modes should have distinct labels
        assert len(set(choices)) == 4


class TestIntegrationWithLinearizer:
    """Integration tests with the auto-linearizer."""

    def test_strategy_values_match_linearization_methods(self):
        """Verify strategy values map to valid LinearizationMethod enums."""
        for label, value in get_strategy_choices():
            # Should not raise an exception
            method = LinearizationMethod(value)
            assert method is not None

    def test_target_values_match_target_response(self):
        """Verify target values map to valid TargetResponse enums."""
        for label, value in get_target_choices():
            # Should not raise an exception
            target = TargetResponse(value)
            assert target is not None
