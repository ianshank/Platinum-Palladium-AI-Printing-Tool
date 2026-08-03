"""
Tests for UI configuration module.
"""

import pytest

from ptpd_calibration.ui.config import (
    ChannelColors,
    create_darkroom_theme,
    get_custom_css,
    get_keyboard_js,
)


class TestChannelColors:
    """Test ChannelColors dataclass."""

    def test_channel_colors_default_values(self) -> None:
        """Test that ChannelColors has correct default values."""
        colors = ChannelColors()
        assert colors.K == "#1a1a1a"
        assert colors.C == "#00BFFF"
        assert colors.M == "#FF1493"
        assert colors.Y == "#FFD700"

    def test_channel_colors_to_dict(self) -> None:
        """Test conversion to dictionary."""
        colors = ChannelColors()
        color_dict = colors.to_dict()

        assert isinstance(color_dict, dict)
        assert len(color_dict) == 11
        assert color_dict["K"] == "#1a1a1a"
        assert "LC" in color_dict
        assert "MK" in color_dict

    def test_channel_colors_custom_values(self) -> None:
        """Test ChannelColors with custom values."""
        colors = ChannelColors(K="#000000", C="#FFFFFF")
        assert colors.K == "#000000"
        assert colors.C == "#FFFFFF"
        assert colors.M == "#FF1493"  # Default


class TestThemeCreation:
    """Test theme creation."""

    def test_create_darkroom_theme(self) -> None:
        """Test darkroom theme creation."""
        theme = create_darkroom_theme()
        assert theme is not None
        # Theme object should be Gradio theme

    def test_theme_has_styling(self) -> None:
        """Test that theme has styling applied."""
        theme = create_darkroom_theme()
        # Check that the theme object exists and is callable/valid
        assert hasattr(theme, "set")


class TestCSSGeneration:
    """Test CSS generation."""

    def test_get_custom_css(self) -> None:
        """Test custom CSS generation."""
        css = get_custom_css()
        assert isinstance(css, str)
        assert len(css) > 0
        assert "--ptpd-bg" in css
        assert "--ptpd-accent" in css

    def test_css_contains_themes(self) -> None:
        """Test that CSS contains theme definitions."""
        css = get_custom_css()
        assert "darkroom" in css or "data-ptpd-theme" in css
        assert "light" in css or "data-ptpd-theme" in css
        assert "print" in css or "data-ptpd-theme" in css

    def test_css_contains_media_queries(self) -> None:
        """Test that CSS contains responsive styles."""
        css = get_custom_css()
        assert "@media" in css
        assert "768px" in css


class TestKeyboardJSGeneration:
    """Test keyboard JavaScript generation."""

    def test_get_keyboard_js(self) -> None:
        """Test keyboard JS generation."""
        js = get_keyboard_js()
        assert isinstance(js, str)
        assert len(js) > 0

    def test_keyboard_js_has_event_listener(self) -> None:
        """Test that keyboard JS sets up event listener."""
        js = get_keyboard_js()
        assert "addEventListener" in js
        assert "keydown" in js

    def test_keyboard_js_has_shortcuts(self) -> None:
        """Test that keyboard JS includes shortcuts."""
        js = get_keyboard_js()
        assert "ctrlKey" in js
        # Ctrl+1-5 for tab switching
        assert "parseInt" in js or "event.key" in js
        # Ctrl+S for save
        assert "toLowerCase" in js or "event.key.toLowerCase()" in js

    def test_keyboard_js_syntax(self) -> None:
        """Test that generated JS has valid syntax."""
        js = get_keyboard_js()
        # Check for balanced braces and parentheses
        assert js.count("(") == js.count(")")
        assert js.count("{") == js.count("}")
