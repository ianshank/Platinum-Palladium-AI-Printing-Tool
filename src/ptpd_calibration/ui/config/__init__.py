"""
UI Configuration Module.

Provides centralized configuration for Gradio UI components.
"""

from dataclasses import dataclass
from typing import Dict

import gradio as gr


@dataclass
class ChannelColors:
    """Color configuration for different channels."""

    K: str = "#1a1a1a"
    C: str = "#00BFFF"
    M: str = "#FF1493"
    Y: str = "#FFD700"
    LC: str = "#87CEEB"
    LM: str = "#FFB6C1"
    LK: str = "#696969"
    LLK: str = "#A9A9A9"
    PK: str = "#2F4F4F"
    MK: str = "#4A4A4A"
    V: str = "#8B8B8B"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "K": self.K, "C": self.C, "M": self.M, "Y": self.Y,
            "LC": self.LC, "LM": self.LM, "LK": self.LK, "LLK": self.LLK,
            "PK": self.PK, "MK": self.MK, "V": self.V,
        }


def create_darkroom_theme() -> gr.themes.Base:
    """Create the darkroom theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
    ).set(
        body_background_fill="#111111",
        body_background_fill_dark="#0b0b0b",
        block_background_fill="#1c1c1c",
        block_background_fill_dark="#0f0f0f",
        block_label_text_color="#f5f5f5",
        input_background_fill="#2a2a2a",
        input_background_fill_dark="#1f1f1f",
    )


def get_custom_css() -> str:
    """Get custom CSS for UI."""
    return """:root, [data-ptpd-theme="darkroom"] {
        --ptpd-bg: #0f0f0f; --ptpd-card: #1f1f1f; --ptpd-text: #f5f5f5;
        --ptpd-muted: #a3a3a3; --ptpd-accent: #fbbf24;
    }
    [data-ptpd-theme="light"] {
        --ptpd-bg: #f8f8f8; --ptpd-card: #ffffff; --ptpd-text: #1f1f1f;
        --ptpd-muted: #6b7280; --ptpd-accent: #d97706;
    }
    [data-ptpd-theme="print"] {
        --ptpd-bg: #ffffff; --ptpd-card: #fdfbf6; --ptpd-text: #111111;
        --ptpd-muted: #4b5563; --ptpd-accent: #b45309;
    }
    body { background: var(--ptpd-bg); color: var(--ptpd-text); }
    .ptpd-card {
        background: var(--ptpd-card) !important;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px !important; padding: 16px;
    }
    .top-bar { align-items: center; justify-content: space-between; gap: 1rem; }
    .main-tabs .tab-nav { flex-wrap: wrap; gap: 0.5rem; }
    @media (max-width: 768px) {
        .main-tabs .tab-nav > button { flex: 1 1 45%; font-size: 0.9rem; }
        .stack-on-mobile { flex-direction: column !important; }
    }
    @media (pointer: coarse) {
        button, input, select, textarea { min-height: 44px; font-size: 1rem; }
    }"""


def get_keyboard_js() -> str:
    """Get keyboard shortcuts JavaScript."""
    return """document.addEventListener('keydown', (event) => {
        if (!event.ctrlKey) return;
        const tabButtons = document.querySelectorAll('.main-tabs .tab-nav button');
        if (event.key >= '1' && event.key <= '5') {
            const idx = parseInt(event.key, 10) - 1;
            tabButtons[idx]?.click();
            event.preventDefault();
        }
        if (event.key.toLowerCase() === 's') {
            event.preventDefault();
            document.querySelector('#save-curve-btn')?.click();
        }
    });"""


__all__ = [
    "ChannelColors",
    "create_darkroom_theme",
    "get_custom_css",
    "get_keyboard_js",
]
