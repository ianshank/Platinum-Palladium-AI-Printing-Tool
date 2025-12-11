from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

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
