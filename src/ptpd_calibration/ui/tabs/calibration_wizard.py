import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from ptpd_calibration.analysis import StepWedgeAnalyzer, WedgeAnalysisConfig
from ptpd_calibration.config import TabletType
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves import save_curve
from ptpd_calibration.curves.linearization import LinearizationMethod, TargetResponse
from ptpd_calibration.papers import PaperDatabase

# ---------------------------------------------------------------------------
# Linearization mode configuration (Step 3 of calibration wizard)
# ---------------------------------------------------------------------------


class WizardLinearizationMode(str, Enum):
    """Linearization modes available in the calibration wizard."""

    SINGLE_CURVE = "single_curve"
    MULTI_CURVE = "multi_curve"
    USE_EXISTING = "use_existing"
    NO_LINEARIZATION = "no_linearization"


@dataclass
class LinearizationModeConfig:
    """Configuration for a single linearization mode."""

    value: str
    label: str
    requires_target: bool = True
    requires_strategy: bool = True
    requires_paper_preset: bool = True
    requires_existing_profile: bool = False
    advanced: bool = False


LINEARIZATION_MODES: dict[str, LinearizationModeConfig] = {
    "single_curve": LinearizationModeConfig(
        value="single_curve",
        label="Single-curve linearization (recommended)",
        requires_target=True,
        requires_strategy=True,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=False,
    ),
    "multi_curve": LinearizationModeConfig(
        value="multi_curve",
        label="Multi-curve / split-tone (advanced)",
        requires_target=True,
        requires_strategy=True,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=True,
    ),
    "use_existing": LinearizationModeConfig(
        value="use_existing",
        label="Use existing profile",
        requires_target=False,
        requires_strategy=False,
        requires_paper_preset=False,
        requires_existing_profile=True,
        advanced=False,
    ),
    "no_linearization": LinearizationModeConfig(
        value="no_linearization",
        label="No linearization (straight curve)",
        requires_target=False,
        requires_strategy=False,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=False,
    ),
}

_STRATEGY_CHOICES: list[tuple[str, str]] = [
    ("Smooth spline (recommended)", LinearizationMethod.SPLINE_FIT.value),
    ("Polynomial fit", LinearizationMethod.POLYNOMIAL_FIT.value),
    ("Iterative refinement", LinearizationMethod.ITERATIVE.value),
    ("Direct inversion (fast)", LinearizationMethod.DIRECT_INVERSION.value),
    ("Hybrid (best quality)", LinearizationMethod.HYBRID.value),
]

_TARGET_CHOICES: list[tuple[str, str]] = [
    ("Even tonal steps (linear)", TargetResponse.LINEAR.value),
    ("Match digital gamma 2.2 (sRGB)", TargetResponse.GAMMA_22.value),
    ("Preserve paper white (highlights)", TargetResponse.PAPER_WHITE.value),
    ("Perceptually uniform", TargetResponse.PERCEPTUAL.value),
    ("Match monitor gamma 1.8", TargetResponse.GAMMA_18.value),
]


def get_linearization_mode_choices() -> list[str]:
    """Return list of mode labels for the linearization mode dropdown."""
    return [mode.label for mode in LINEARIZATION_MODES.values()]


def get_mode_by_label(label: str) -> LinearizationModeConfig | None:
    """Return mode config matching *label*, or None if not found."""
    for mode in LINEARIZATION_MODES.values():
        if mode.label == label:
            return mode
    return None


def get_mode_value_by_label(label: str) -> str | None:
    """Return mode value string matching *label*, or None if not found."""
    mode = get_mode_by_label(label)
    return mode.value if mode else None


def get_strategy_choices() -> list[tuple[str, str]]:
    """Return list of (label, value) tuples for linearization strategy dropdown."""
    return list(_STRATEGY_CHOICES)


def get_strategy_labels() -> list[str]:
    """Return list of strategy labels."""
    return [label for label, _ in _STRATEGY_CHOICES]


def get_strategy_value_by_label(label: str) -> str | None:
    """Return strategy value string matching *label*, or None if not found."""
    for lbl, val in _STRATEGY_CHOICES:
        if lbl == label:
            return val
    return None


def get_target_choices() -> list[tuple[str, str]]:
    """Return list of (label, value) tuples for target response dropdown."""
    return list(_TARGET_CHOICES)


def get_target_labels() -> list[str]:
    """Return list of target labels."""
    return [label for label, _ in _TARGET_CHOICES]


def get_target_value_by_label(label: str) -> str | None:
    """Return target value string matching *label*, or None if not found."""
    for lbl, val in _TARGET_CHOICES:
        if lbl == label:
            return val
    return None


def get_paper_preset_choices() -> list[str]:
    """Return list of paper preset names from PaperDatabase plus 'Other / custom'."""
    try:
        db = PaperDatabase()
        names = [p.name for p in db.list_papers() if not getattr(p, "is_custom", False)]
    except Exception:
        names = []
    return names + ["Other / custom"]


def get_paper_chemistry_notes(paper_name: str) -> str:
    """Return chemistry notes for *paper_name*, empty string if not found."""
    if paper_name in ("Other / custom", ""):
        return ""
    try:
        db = PaperDatabase()
        for paper in db.list_papers():
            if paper.name == paper_name:
                return str(getattr(paper, "chemistry_notes", "") or "")
    except Exception:
        pass
    return ""


def wizard_is_valid_config(
    mode_label: str,
    target_label: str,
    strategy_label: str,
    paper_preset: str,
    existing_profile: str | None,
    custom_chemistry: str,
    curve_name: str,
) -> tuple[bool, str]:
    """Validate wizard configuration.

    Returns:
        (is_valid, error_message) — error_message is empty string when valid.
    """
    mode = get_mode_by_label(mode_label)
    if mode is None:
        return False, "Please select a valid linearization mode."
    if not curve_name.strip():
        return False, "Please enter a curve name."
    if mode.requires_target and not target_label.strip():
        return False, "Please select a target response."
    if mode.requires_strategy and not strategy_label.strip():
        return False, "Please select a strategy."
    if mode.requires_existing_profile:
        if not existing_profile or existing_profile == "No curves available":
            return False, "Please select an existing profile."
    if (
        mode.requires_paper_preset
        and paper_preset == "Other / custom"
        and not custom_chemistry.strip()
    ):
        return False, "Please enter chemistry notes for the custom paper."
    return True, ""


def wizard_on_mode_change(mode_label: str) -> tuple:
    """Return 7 UI update dicts based on selected mode.

    Order: [target, strategy, paper_preset, existing_profile,
            advanced_options, curve_name, status_message]
    """
    mode = get_mode_by_label(mode_label)
    if mode is None:
        return (
            {"visible": True},
            {"visible": True},
            {"visible": True},
            {"visible": False},
            {"visible": False},
            {"visible": True},
            {"visible": False},
        )
    return (
        {"visible": mode.requires_target},
        {"visible": mode.requires_strategy},
        {"visible": mode.requires_paper_preset},
        {"visible": mode.requires_existing_profile},
        {"visible": mode.advanced},
        {"visible": not mode.requires_existing_profile},
        {"visible": False},
    )


def wizard_on_paper_change(paper_name: str) -> tuple[dict, dict]:
    """Return (custom_chemistry_visibility, chemistry_notes_update) based on paper selection."""
    if paper_name == "Other / custom":
        return {"visible": True, "interactive": True}, {"value": ""}
    notes = get_paper_chemistry_notes(paper_name)
    return {"visible": False}, {"value": notes}


def wizard_on_config_change(
    mode_label: str,
    target_label: str,
    strategy_label: str,
    paper_preset: str,
    existing_profile: str | None,
    custom_chemistry: str,
    curve_name: str,
) -> tuple[dict, str]:
    """Return (button_update, validation_message) based on current config."""
    is_valid, error_msg = wizard_is_valid_config(
        mode_label,
        target_label,
        strategy_label,
        paper_preset,
        existing_profile,
        custom_chemistry,
        curve_name,
    )
    if is_valid:
        return {"interactive": True}, "Configuration is valid."
    return {"interactive": False}, error_msg


def build_calibration_wizard_tab() -> None:
    """Build the Calibration Wizard tab."""
    with gr.TabItem("Calibration Wizard"):
        gr.Markdown(
            """
            ### 🧙 Calibration Wizard

            Follow the guided five-step wizard to analyze a step tablet, choose a method,
            generate a curve, and export it for your printer driver.
            """
        )

        wizard_step_state = gr.State(1)
        wizard_analysis_state = gr.State(None)
        wizard_curve_state = gr.State(None)
        wizard_mode_state = gr.State(None)
        wizard_config_valid_state = gr.State(False)
        step_titles = [
            "Scan your step tablet",
            "Review detection results",
            "Choose linearization method",
            "Review generated curve",
            "Export curve file",
        ]

        def _wizard_visibility(target_step: int) -> tuple:
            updates = [gr.update(visible=index + 1 == target_step) for index in range(5)]
            return (
                target_step,
                f"**Step {target_step} of 5:** {step_titles[target_step - 1]}",
                *updates,
            )

        progress = gr.Markdown(f"**Step 1 of 5:** {step_titles[0]}")

        with gr.Group(visible=True) as wizard_step_one:
            gr.Markdown("#### Step 1: Upload scan and configure detection")
            with gr.Row():
                wizard_tablet_upload = gr.Image(
                    type="filepath",
                    label="Step Tablet Scan",
                )
                with gr.Column():
                    wizard_tablet_type = gr.Dropdown(
                        choices=[t.value for t in TabletType],
                        value=TabletType.STOUFFER_21.value,
                        label="Tablet Type",
                        info="Match the physical tablet you exposed in your contact print.",
                    )
                    wizard_density_range = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        step=0.1,
                        value=1.5,
                        label="Min Density Range",
                        info="Ensures the scan spans enough density for a usable curve.",
                    )
                    wizard_fix_reversals = gr.Checkbox(
                        label="Auto-fix density reversals",
                        value=True,
                    )
                    wizard_reject_outliers = gr.Checkbox(
                        label="Reject outlier patches",
                        value=True,
                    )
            wizard_analyze_btn = gr.Button("Analyze Tablet →", variant="primary")

        with gr.Group(visible=False) as wizard_step_two:
            gr.Markdown("#### Step 2: Review detection")
            with gr.Row():
                wizard_detection_plot = gr.Plot(label="Detected Patches")
                wizard_density_table = gr.Dataframe(
                    headers=["Patch", "Density", "Status"],
                    interactive=False,
                )
            with gr.Row():
                wizard_grade = gr.Textbox(label="Quality Grade", interactive=False)
                wizard_quality_score = gr.Number(label="Quality Score", interactive=False)
            wizard_warnings = gr.Textbox(
                label="Warnings",
                interactive=False,
                lines=4,
            )
            wizard_recommendations = gr.Textbox(
                label="Recommendations",
                interactive=False,
                lines=4,
            )
            with gr.Row():
                wizard_back_to_upload = gr.Button("← Back")
                wizard_continue_to_methods = gr.Button("Next: Choose Method →", variant="primary")

        with gr.Group(visible=False) as wizard_step_three:
            gr.Markdown("#### Step 3: Choose linearization method")
            wizard_linearization_mode = gr.Dropdown(
                choices=get_linearization_mode_choices(),
                value=get_linearization_mode_choices()[0],
                label="Linearization Mode",
            )
            wizard_target = gr.Dropdown(
                choices=get_target_labels(),
                value=get_target_labels()[0],
                label="Target Response",
            )
            wizard_strategy = gr.Dropdown(
                choices=get_strategy_labels(),
                value=get_strategy_labels()[0],
                label="Strategy",
            )
            wizard_paper_preset = gr.Dropdown(
                choices=get_paper_preset_choices(),
                value=get_paper_preset_choices()[0],
                label="Paper",
            )
            wizard_existing_profile = gr.Dropdown(
                choices=["No curves available"],
                value="No curves available",
                label="Existing Profile",
                visible=False,
            )
            wizard_curve_name = gr.Textbox(label="Curve Name", value="Wizard Curve")
            wizard_chemistry = gr.Textbox(
                label="Chemistry Notes",
                placeholder="e.g., 50/50 Pt/Pd, 5 drops Na2",
                visible=False,
            )
            wizard_generate_curve = gr.Button("Generate Curve →", variant="primary")

        with gr.Group(visible=False) as wizard_step_four:
            gr.Markdown("#### Step 4: Review curve")
            wizard_curve_plot = gr.Plot(label="Generated Curve")
            wizard_curve_summary = gr.Textbox(
                label="Summary",
                interactive=False,
                lines=3,
            )
            with gr.Row():
                wizard_back_to_methods = gr.Button("← Back")
                wizard_continue_to_export = gr.Button("Export →", variant="primary")

        with gr.Group(visible=False) as wizard_step_five:
            gr.Markdown("#### Step 5: Export curve")
            wizard_export_format = gr.Dropdown(
                choices=["qtr", "piezography", "csv", "json"],
                label="Format",
                value="qtr",
            )
            wizard_export_btn = gr.Button("Download Curve", elem_id="save-curve-btn")
            wizard_export_file = gr.File(label="Download")
            wizard_finish = gr.Button("Finish & Restart", variant="secondary")

        def wizard_analyze(
            image_path: str | None,
            tablet_type: str,
            density_range: float,
            fix_rev: bool,
            reject_outliers: bool,
        ) -> tuple:
            if image_path is None:
                vis = _wizard_visibility(1)
                return (
                    None,
                    "No image",
                    0,
                    None,
                    [],
                    "",
                    "",
                    *vis,
                )

            try:
                config = WedgeAnalysisConfig(
                    tablet_type=TabletType(tablet_type),
                    min_density_range=density_range,
                    auto_fix_reversals=fix_rev,
                    outlier_rejection=reject_outliers,
                )
                analyzer = StepWedgeAnalyzer(config)
                result = analyzer.analyze(image_path, generate_curve=False)

                fig, ax = plt.subplots(figsize=(8, 4))
                if result.densities:
                    x = np.linspace(0, 100, len(result.densities))
                    ax.plot(x, result.densities, "o-", color="#fbbf24", linewidth=2)
                ax.set_xlabel("Input %")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.2)
                ax.set_title("Detected Step Tablet")

                table_rows = []
                if result.densities:
                    for idx, density in enumerate(result.densities):
                        table_rows.append([idx + 1, round(density, 3), "✓"])

                grade = result.quality.grade.value.upper() if result.quality else "N/A"
                score = result.quality.score if result.quality else 0
                warnings = ""
                if result.quality and result.quality.warnings:
                    warnings = "\n".join(
                        f"[{w.level.value.upper()}] {w.message}" for w in result.quality.warnings
                    )
                recs = ""
                if result.quality and result.quality.recommendations:
                    recs = "\n".join(f"• {rec}" for rec in result.quality.recommendations)

                visibility = _wizard_visibility(2)
                return (
                    result,
                    grade,
                    score,
                    fig,
                    table_rows,
                    warnings,
                    recs,
                    visibility[0],
                    visibility[1],
                    *visibility[2:],
                )
            except Exception as exc:
                vis = _wizard_visibility(1)
                return (
                    None,
                    f"Error: {exc}",
                    0,
                    None,
                    [],
                    str(exc),
                    "",
                    *vis,
                )

        wizard_analyze_btn.click(
            wizard_analyze,
            inputs=[
                wizard_tablet_upload,
                wizard_tablet_type,
                wizard_density_range,
                wizard_fix_reversals,
                wizard_reject_outliers,
            ],
            outputs=[
                wizard_analysis_state,
                wizard_grade,
                wizard_quality_score,
                wizard_detection_plot,
                wizard_density_table,
                wizard_warnings,
                wizard_recommendations,
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        def go_to_step(step: int) -> tuple:
            return _wizard_visibility(step)

        wizard_back_to_upload.click(
            lambda: go_to_step(1),
            outputs=[
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        wizard_continue_to_methods.click(
            lambda: go_to_step(3),
            outputs=[
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        def wizard_generate(
            result: Any, name: str, paper: str, chemistry: str, method: str
        ) -> tuple:
            if result is None:
                return (
                    None,
                    None,
                    "Analyze the tablet first.",
                    *_wizard_visibility(2),
                )
            try:
                curve_type = CurveType.LINEAR
                analyzer = StepWedgeAnalyzer(WedgeAnalysisConfig(default_curve_type=curve_type))
                analysis = analyzer.analyze_from_densities(
                    result.densities,
                    curve_name=name or "Wizard Curve",
                    paper_type=paper or None,
                    chemistry=chemistry or None,
                    generate_curve=True,
                    curve_type=curve_type,
                )
                curve = analysis.curve

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(curve.input_values, curve.output_values, color="#f59e0b", linewidth=2)
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")
                ax.grid(True, alpha=0.2)
                ax.set_title(curve.name)

                visibility = _wizard_visibility(4)
                summary = f"Curve points: {len(curve.output_values)}\nMethod: {method}"
                return (
                    curve,
                    fig,
                    summary,
                    visibility[0],
                    visibility[1],
                    *visibility[2:],
                )
            except Exception as exc:
                return (
                    None,
                    None,
                    f"Error: {exc}",
                    *_wizard_visibility(3),
                )

        wizard_generate_curve.click(
            wizard_generate,
            inputs=[
                wizard_analysis_state,
                wizard_curve_name,
                wizard_paper_preset,
                wizard_chemistry,
                wizard_linearization_mode,
            ],
            outputs=[
                wizard_curve_state,
                wizard_curve_plot,
                wizard_curve_summary,
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        wizard_back_to_methods.click(
            lambda: go_to_step(3),
            outputs=[
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        wizard_continue_to_export.click(
            lambda: go_to_step(5),
            outputs=[
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )

        def wizard_export(curve: Any, fmt: str) -> str | None:
            if curve is None:
                return None
            try:
                ext_map = {"qtr": ".quad", "piezography": ".ppt", "csv": ".csv", "json": ".json"}
                ext = ext_map.get(fmt, ".quad")
                safe_name = "".join(c for c in curve.name if c.isalnum() or c in " -_")[:40]
                temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"
                save_curve(curve, temp_path, format=fmt)
                return str(temp_path)
            except Exception:
                return None

        wizard_export_btn.click(
            wizard_export,
            inputs=[wizard_curve_state, wizard_export_format],
            outputs=[wizard_export_file],
        )

        wizard_finish.click(
            lambda: go_to_step(1),
            outputs=[
                wizard_step_state,
                progress,
                wizard_step_one,
                wizard_step_two,
                wizard_step_three,
                wizard_step_four,
                wizard_step_five,
            ],
        )
