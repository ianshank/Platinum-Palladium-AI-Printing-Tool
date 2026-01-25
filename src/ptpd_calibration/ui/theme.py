from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

# =============================================================================
# MATPLOTLIB THEME COLORS
# =============================================================================
# Centralized color definitions for matplotlib plots used in UI components


class PlotColors:
    """Color scheme for matplotlib plots in Gradio UI."""

    # Background colors
    DARK_BG: Final[str] = "#1a1a2e"
    DARK_BORDER: Final[str] = "#333"

    # Text colors
    TEXT_WHITE: Final[str] = "white"
    TEXT_LIGHT: Final[str] = "#e0e0e0"

    # Chart colors
    PRIMARY: Final[str] = "#f59e0b"        # Orange/amber - main accent
    SECONDARY: Final[str] = "#3b82f6"      # Blue - secondary accent
    SUCCESS: Final[str] = "#4ade80"        # Green - positive values
    ERROR: Final[str] = "#f87171"          # Red - negative values
    WARNING: Final[str] = "#fbbf24"        # Yellow/gold - warnings

    # Specific use colors
    TRAINING_LOSS: Final[str] = "#f59e0b"  # Orange for training loss
    VALIDATION_LOSS: Final[str] = "#3b82f6"  # Blue for validation loss
    POSITIVE_CORRECTION: Final[str] = "#4ade80"  # Green for positive corrections
    NEGATIVE_CORRECTION: Final[str] = "#f87171"  # Red for negative corrections

    # Metal colors for Pt/Pd visualization
    PLATINUM: Final[str] = "#fbbf24"       # Gold for platinum
    PALLADIUM: Final[str] = "#94a3b8"      # Silver for palladium
    PLATINUM_GRADIENT_END: Final[str] = "#f59e0b"
    PALLADIUM_GRADIENT_END: Final[str] = "#64748b"


def apply_dark_theme_to_axes(ax: Any) -> None:
    """Apply dark theme styling to matplotlib axes.

    Args:
        ax: Matplotlib axes object to style.
    """
    ax.set_facecolor(PlotColors.DARK_BG)
    ax.tick_params(colors=PlotColors.TEXT_WHITE)
    ax.xaxis.label.set_color(PlotColors.TEXT_WHITE)
    ax.yaxis.label.set_color(PlotColors.TEXT_WHITE)
    ax.title.set_color(PlotColors.TEXT_WHITE)

    for spine in ax.spines.values():
        spine.set_color(PlotColors.DARK_BORDER)


def apply_dark_theme_to_figure(fig: Any) -> None:
    """Apply dark theme styling to matplotlib figure.

    Args:
        fig: Matplotlib figure object to style.
    """
    fig.patch.set_facecolor(PlotColors.DARK_BG)


def apply_dark_theme(
    fig: Any,
    axes: Sequence[Any] | None = None
) -> None:
    """Apply dark theme to figure and all axes.

    Args:
        fig: Matplotlib figure object.
        axes: Optional sequence of axes. If None, applies to all axes in figure.
    """
    apply_dark_theme_to_figure(fig)

    if axes is None:
        axes = fig.get_axes()

    for ax in axes:
        apply_dark_theme_to_axes(ax)

class ProLabTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.cyan,
        secondary_hue: colors.Color | str = colors.slate,
        neutral_hue: colors.Color | str = colors.zinc,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_sm,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="#121212",
            body_background_fill_dark="#0A0A0A",
            block_background_fill="#1E1E1E",
            block_background_fill_dark="#161616",
            block_border_width="1px",
            block_border_color="#333333",
            block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.16)",

            # Input fields
            input_background_fill="#262626",
            input_background_fill_dark="#202020",
            input_border_color="#404040",
            input_radius="4px",

            # Buttons
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_hover="*primary_500",
            button_primary_text_color="white",
            button_secondary_background_fill="#333333",
            button_secondary_background_fill_hover="#404040",
            button_secondary_text_color="*neutral_200",

            # Text
            block_label_text_color="*neutral_400",
            block_title_text_color="*neutral_200",
            body_text_color="*neutral_300",
            # prose_header_text_color="*neutral_100",  # Removed as it caused error

            # Borders
            border_color_primary="#333333",
        )
