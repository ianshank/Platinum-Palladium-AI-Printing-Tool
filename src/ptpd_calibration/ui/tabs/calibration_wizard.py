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
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves import save_curve
from ptpd_calibration.curves.linearization import (
    AutoLinearizer,
    LinearizationConfig,
    LinearizationMethod,
    TargetResponse,
)
from ptpd_calibration.papers.profiles import PaperDatabase

# =============================================================================
# Linearization Mode Configuration
# =============================================================================


class WizardLinearizationMode(str, Enum):
    """High-level linearization modes for the wizard."""

    SINGLE_CURVE = "single_curve"
    MULTI_CURVE = "multi_curve"
    USE_EXISTING = "use_existing"
    NO_LINEARIZATION = "no_linearization"


@dataclass
class LinearizationModeConfig:
    """Configuration for a linearization mode."""

    value: str
    label: str
    description: str
    requires_target: bool = True
    requires_strategy: bool = True
    requires_paper_preset: bool = True
    requires_existing_profile: bool = False
    advanced: bool = False


# Define the linearization modes with their configurations
LINEARIZATION_MODES: dict[str, LinearizationModeConfig] = {
    WizardLinearizationMode.SINGLE_CURVE.value: LinearizationModeConfig(
        value=WizardLinearizationMode.SINGLE_CURVE.value,
        label="Single-curve linearization (recommended)",
        description="Standard one-curve correction from your analyzed step wedge. Best starting point for a new paper/chemistry setup.",
        requires_target=True,
        requires_strategy=True,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=False,
    ),
    WizardLinearizationMode.MULTI_CURVE.value: LinearizationModeConfig(
        value=WizardLinearizationMode.MULTI_CURVE.value,
        label="Multi-curve / split-tone (advanced)",
        description="For users intending different curves for different tonal ranges. Currently generates a single curve preview.",
        requires_target=True,
        requires_strategy=True,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=True,
    ),
    WizardLinearizationMode.USE_EXISTING.value: LinearizationModeConfig(
        value=WizardLinearizationMode.USE_EXISTING.value,
        label="Use existing profile",
        description="Reuse a curve already present in your curve library rather than generating a new one.",
        requires_target=False,
        requires_strategy=False,
        requires_paper_preset=False,
        requires_existing_profile=True,
        advanced=False,
    ),
    WizardLinearizationMode.NO_LINEARIZATION.value: LinearizationModeConfig(
        value=WizardLinearizationMode.NO_LINEARIZATION.value,
        label="No linearization (straight curve)",
        description="Use a near-straight curve as a baseline. Useful for comparison or when testing new papers.",
        requires_target=False,
        requires_strategy=False,
        requires_paper_preset=True,
        requires_existing_profile=False,
        advanced=False,
    ),
}


def get_linearization_mode_choices() -> list[str]:
    """Get list of linearization mode labels for dropdown."""
    return [config.label for config in LINEARIZATION_MODES.values()]


def get_mode_by_label(label: str) -> LinearizationModeConfig | None:
    """Get mode configuration by its display label."""
    for config in LINEARIZATION_MODES.values():
        if config.label == label:
            return config
    return None


def get_mode_value_by_label(label: str) -> str | None:
    """Get mode value by its display label."""
    config = get_mode_by_label(label)
    return config.value if config else None


# =============================================================================
# Strategy (Method) Configuration
# =============================================================================


def get_strategy_choices() -> list[tuple[str, str]]:
    """Get user-friendly strategy choices mapped to LinearizationMethod values."""
    return [
        ("Smooth spline (recommended)", LinearizationMethod.SPLINE_FIT.value),
        ("Polynomial fit", LinearizationMethod.POLYNOMIAL_FIT.value),
        ("Iterative refinement", LinearizationMethod.ITERATIVE.value),
        ("Hybrid (spline + iterative)", LinearizationMethod.HYBRID.value),
        ("Direct inversion (simple)", LinearizationMethod.DIRECT_INVERSION.value),
    ]


def get_strategy_labels() -> list[str]:
    """Get list of strategy labels for dropdown."""
    return [label for label, _ in get_strategy_choices()]


def get_strategy_value_by_label(label: str) -> str | None:
    """Get LinearizationMethod value by label."""
    for lbl, val in get_strategy_choices():
        if lbl == label:
            return val
    return None


# =============================================================================
# Target Response Configuration
# =============================================================================


def get_target_choices() -> list[tuple[str, str]]:
    """Get user-friendly target response choices mapped to TargetResponse values."""
    return [
        ("Even tonal steps (linear)", TargetResponse.LINEAR.value),
        ("Match digital gamma 2.2 (sRGB)", TargetResponse.GAMMA_22.value),
        ("Match gamma 1.8 (Mac display)", TargetResponse.GAMMA_18.value),
        ("Preserve paper white (highlights)", TargetResponse.PAPER_WHITE.value),
        ("Perceptually uniform (L* curve)", TargetResponse.PERCEPTUAL.value),
    ]


def get_target_labels() -> list[str]:
    """Get list of target labels for dropdown."""
    return [label for label, _ in get_target_choices()]


def get_target_value_by_label(label: str) -> str | None:
    """Get TargetResponse value by label."""
    for lbl, val in get_target_choices():
        if lbl == label:
            return val
    return None


# =============================================================================
# Paper Preset Configuration
# =============================================================================


def get_paper_preset_choices() -> list[str]:
    """Get paper preset choices including custom option."""
    db = PaperDatabase()
    papers = db.list_paper_names()
    return papers + ["Other / custom"]


def get_paper_chemistry_notes(paper_name: str) -> str:
    """Get chemistry notes for a paper preset."""
    if paper_name == "Other / custom":
        return ""
    db = PaperDatabase()
    paper = db.get_paper(paper_name)
    if paper:
        notes = []
        if paper.chemistry_notes:
            notes.append(paper.chemistry_notes)
        if paper.recommended_pt_ratio > 0:
            notes.append(f"Recommended Pt ratio: {paper.recommended_pt_ratio:.0%}")
        if paper.recommended_na2_ratio > 0:
            notes.append(f"Recommended Na2: {paper.recommended_na2_ratio:.0%}")
        return " | ".join(notes) if notes else "No specific chemistry notes for this paper."
    return ""


# =============================================================================
# Validation Helpers
# =============================================================================


def wizard_is_valid_config(
    mode_label: str,
    target_label: str,
    strategy_label: str,
    paper_preset: str,
    existing_profile: str | None,
    custom_chemistry: str,
    curve_name: str,
) -> tuple[bool, str]:
    """
    Validate wizard configuration.

    Returns:
        Tuple of (is_valid, error_message)
    """
    mode_config = get_mode_by_label(mode_label)
    if not mode_config:
        return False, "Please select a linearization mode."

    # Check curve name
    if not curve_name or not curve_name.strip():
        return False, "Please enter a curve name."

    # Mode-specific validation
    if mode_config.requires_target and not target_label:
        return False, "Please select a target response."

    if mode_config.requires_strategy and not strategy_label:
        return False, "Please select a curve strategy."

    if mode_config.requires_paper_preset and not paper_preset:
        return False, "Please select a paper preset."

    if mode_config.requires_existing_profile and (
        not existing_profile or existing_profile == "No curves available"
    ):
        return (
            False,
            "Please select an existing curve profile. Load curves in the Curve Display tab first.",
        )

    # Custom paper validation
    if paper_preset == "Other / custom" and not custom_chemistry.strip():
        return False, "Please enter chemistry notes for custom paper."

    return True, ""


# =============================================================================
# Mode Change Handler
# =============================================================================


def wizard_on_mode_change(mode_label: str) -> tuple[Any, ...]:
    """
    Handle linearization mode change.

    Returns gr.update objects to control visibility and interactive states.
    """
    mode_config = get_mode_by_label(mode_label)

    if not mode_config:
        # Default: show everything
        return (
            gr.update(visible=True, interactive=True),  # target dropdown
            gr.update(visible=True, interactive=True),  # strategy dropdown
            gr.update(visible=True, interactive=True),  # paper preset dropdown
            gr.update(visible=False),  # existing profile dropdown
            gr.update(visible=False),  # advanced options accordion
            gr.update(visible=True),  # curve name textbox
            gr.update(visible=False),  # custom chemistry textbox
        )

    # Determine visibility based on mode requirements
    show_target = mode_config.requires_target
    show_strategy = mode_config.requires_strategy
    show_paper = mode_config.requires_paper_preset
    show_existing = mode_config.requires_existing_profile
    show_advanced = mode_config.advanced

    return (
        gr.update(visible=show_target, interactive=show_target),
        gr.update(visible=show_strategy, interactive=show_strategy),
        gr.update(visible=show_paper, interactive=show_paper),
        gr.update(visible=show_existing, interactive=show_existing),
        gr.update(visible=show_advanced),
        gr.update(visible=not show_existing),  # Hide curve name when using existing
        gr.update(visible=False),  # Custom chemistry - updated by paper change
    )


def wizard_on_paper_change(paper_preset: str) -> tuple[Any, ...]:
    """
    Handle paper preset change.

    Returns gr.update objects for chemistry-related fields.
    """
    is_custom = paper_preset == "Other / custom"
    chemistry_notes = get_paper_chemistry_notes(paper_preset)

    return (
        gr.update(visible=is_custom, interactive=is_custom),  # custom chemistry textbox
        gr.update(value=chemistry_notes),  # chemistry notes display
    )


def wizard_on_config_change(
    mode_label: str,
    target_label: str,
    strategy_label: str,
    paper_preset: str,
    existing_profile: str | None,
    custom_chemistry: str,
    curve_name: str,
) -> tuple[Any, str]:
    """
    Handle any configuration change to validate and update button state.

    Returns:
        Tuple of (button update, validation message)
    """
    is_valid, message = wizard_is_valid_config(
        mode_label,
        target_label,
        strategy_label,
        paper_preset,
        existing_profile,
        custom_chemistry,
        curve_name,
    )

    return (
        gr.update(interactive=is_valid),
        message if not is_valid else "✓ Configuration valid",
    )


def build_calibration_wizard_tab():
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
        step_titles = [
            "Scan your step tablet",
            "Review detection results",
            "Configure curve generation",
            "Review generated curve",
            "Export curve file",
        ]

        def _wizard_visibility(target_step: int):
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

        # State for available curves (for "Use existing profile" mode)
        wizard_available_curves = gr.State([])
        _wizard_available_curve_names = gr.State([])  # Reserved for UI integration

        with gr.Group(visible=False) as wizard_step_three:
            gr.Markdown(
                """
                #### Step 3: Decide how the tool adjusts your Pt/Pd curve

                This step chooses how the wizard will shape your Pt/Pd output curve
                based on the measured step tablet. Select a mode and configure the options below.
                """
            )

            # Primary linearization mode dropdown
            wizard_linearization_mode = gr.Dropdown(
                choices=get_linearization_mode_choices(),
                value=get_linearization_mode_choices()[0],
                label="Linearization Mode",
                info="Choose how you want to generate your correction curve.",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Target response dropdown
                    wizard_target_response = gr.Dropdown(
                        choices=get_target_labels(),
                        value=get_target_labels()[0],
                        label="Target Response",
                        info="What tonal distribution should the corrected print have?",
                        visible=True,
                    )

                    # Curve strategy dropdown
                    wizard_curve_strategy = gr.Dropdown(
                        choices=get_strategy_labels(),
                        value=get_strategy_labels()[0],
                        label="Curve Strategy",
                        info="Mathematical method for computing the correction curve.",
                        visible=True,
                    )

                with gr.Column(scale=1):
                    # Paper preset dropdown
                    wizard_paper_preset = gr.Dropdown(
                        choices=get_paper_preset_choices(),
                        value=get_paper_preset_choices()[0] if get_paper_preset_choices() else None,
                        label="Paper Preset",
                        info="Select your paper to load recommended chemistry notes.",
                        visible=True,
                    )

                    # Existing profile dropdown (hidden by default)
                    wizard_existing_profile = gr.Dropdown(
                        choices=["No curves available"],
                        value=None,
                        label="Existing Profile",
                        info="Select a curve from your library to reuse.",
                        visible=False,
                    )

            # Chemistry notes display (read-only helper)
            wizard_chemistry_notes_display = gr.Textbox(
                label="Chemistry Notes (from preset)",
                value=get_paper_chemistry_notes(
                    get_paper_preset_choices()[0] if get_paper_preset_choices() else ""
                ),
                interactive=False,
                lines=1,
            )

            # Custom chemistry textbox (shown when "Other / custom" is selected)
            wizard_custom_chemistry = gr.Textbox(
                label="Custom Chemistry Notes",
                placeholder="e.g., 50/50 Pt/Pd, 5 drops Na2, ammonium citrate",
                visible=False,
                lines=2,
            )

            # Curve name
            wizard_curve_name = gr.Textbox(
                label="Curve Name",
                value="Wizard Curve",
                info="Name used in the saved curve file.",
                visible=True,
            )

            # Advanced options accordion (shown for advanced modes)
            with gr.Accordion(
                "Advanced Options", open=False, visible=False
            ) as wizard_advanced_options:
                wizard_smoothing = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.1,
                    label="Smoothing Factor",
                    info="Higher values = smoother curve transitions.",
                )
                wizard_iterations = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                    label="Refinement Iterations",
                    info="Number of iterative refinement passes (for iterative methods).",
                )
                wizard_output_points = gr.Slider(
                    minimum=32,
                    maximum=512,
                    step=32,
                    value=256,
                    label="Output Points",
                    info="Number of points in the output curve.",
                )

            # Validation message
            wizard_validation_msg = gr.Markdown(
                value="✓ Configuration valid",
                visible=True,
            )

            # Learn more accordion
            with gr.Accordion("Learn more about linearization methods", open=False):
                gr.Markdown(
                    """
                    **Linearization modes:**
                    - **Single-curve**: Standard approach - generates one correction curve from your step wedge.
                    - **Multi-curve / split-tone**: For advanced users working with different tonal ranges.
                    - **Use existing profile**: Skip generation and reuse a trusted curve from your library.
                    - **No linearization**: Use a straight (identity) curve as a baseline for comparison.

                    **Target responses:**
                    - **Linear**: Equal density steps from white to black.
                    - **Gamma 2.2**: Matches sRGB/digital preview appearance.
                    - **Paper white**: Preserves highlight detail by lifting blacks slightly.
                    - **Perceptual (L*)**: Follows human perception of lightness.

                    **Strategies:**
                    - **Smooth spline**: Best for most cases - produces smooth transitions.
                    - **Polynomial**: Mathematical fit - good for simple response curves.
                    - **Iterative**: Refines progressively - good for complex responses.
                    - **Hybrid**: Combines spline smoothness with iterative refinement.
                    """
                )

            wizard_generate_curve = gr.Button(
                "Generate Curve →", variant="primary", interactive=True
            )

            # Wire up mode change handler
            wizard_linearization_mode.change(
                wizard_on_mode_change,
                inputs=[wizard_linearization_mode],
                outputs=[
                    wizard_target_response,
                    wizard_curve_strategy,
                    wizard_paper_preset,
                    wizard_existing_profile,
                    wizard_advanced_options,
                    wizard_curve_name,
                    wizard_custom_chemistry,
                ],
            )

            # Wire up paper preset change handler
            wizard_paper_preset.change(
                wizard_on_paper_change,
                inputs=[wizard_paper_preset],
                outputs=[
                    wizard_custom_chemistry,
                    wizard_chemistry_notes_display,
                ],
            )

            # Wire up validation on any config change
            config_inputs = [
                wizard_linearization_mode,
                wizard_target_response,
                wizard_curve_strategy,
                wizard_paper_preset,
                wizard_existing_profile,
                wizard_custom_chemistry,
                wizard_curve_name,
            ]

            for component in config_inputs:
                component.change(
                    wizard_on_config_change,
                    inputs=config_inputs,
                    outputs=[wizard_generate_curve, wizard_validation_msg],
                )

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

        def wizard_analyze(image_path, tablet_type, density_range, fix_rev, reject_outliers):
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

        def go_to_step(step):
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
            result,
            mode_label,
            target_label,
            strategy_label,
            paper_preset,
            existing_profile,
            custom_chemistry,
            curve_name,
            available_curves,
            smoothing,
            iterations,
            output_points,
        ):
            """Generate curve based on selected mode and options."""
            # Validate configuration first
            is_valid, error_msg = wizard_is_valid_config(
                mode_label,
                target_label,
                strategy_label,
                paper_preset,
                existing_profile,
                custom_chemistry,
                curve_name,
            )
            if not is_valid:
                return (
                    None,
                    None,
                    f"Configuration error: {error_msg}",
                    *_wizard_visibility(3),
                )

            mode_value = get_mode_value_by_label(mode_label)

            # Handle "Use existing profile" mode
            if mode_value == WizardLinearizationMode.USE_EXISTING.value:
                if not available_curves or not existing_profile:
                    return (
                        None,
                        None,
                        "No existing profile selected. Load curves in the Curve Display tab first.",
                        *_wizard_visibility(3),
                    )
                # Find the selected curve
                curve = None
                for c in available_curves:
                    if c.name == existing_profile:
                        curve = c
                        break
                if curve is None:
                    return (
                        None,
                        None,
                        f"Could not find curve: {existing_profile}",
                        *_wizard_visibility(3),
                    )

                # Create plot for existing curve
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(curve.input_values, curve.output_values, color="#f59e0b", linewidth=2)
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")
                ax.grid(True, alpha=0.2)
                ax.set_title(curve.name)

                visibility = _wizard_visibility(4)
                summary = (
                    f"Mode: Use existing profile\n"
                    f"Curve: {curve.name}\n"
                    f"Points: {len(curve.output_values)} (no new linearization computed)"
                )
                return (
                    curve,
                    fig,
                    summary,
                    visibility[0],
                    visibility[1],
                    *visibility[2:],
                )

            # Handle "No linearization" mode - create identity curve
            if mode_value == WizardLinearizationMode.NO_LINEARIZATION.value:
                num_points = int(output_points)
                input_vals = [i / (num_points - 1) for i in range(num_points)]
                output_vals = input_vals.copy()

                # Determine paper/chemistry from preset or custom
                paper = paper_preset if paper_preset != "Other / custom" else "Custom"
                chemistry = (
                    custom_chemistry
                    if paper_preset == "Other / custom"
                    else get_paper_chemistry_notes(paper_preset)
                )

                curve = CurveData(
                    name=curve_name or "Identity Curve",
                    input_values=input_vals,
                    output_values=output_vals,
                    paper_type=paper,
                    chemistry=chemistry,
                    curve_type=CurveType.LINEAR,
                )

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(curve.input_values, curve.output_values, color="#f59e0b", linewidth=2)
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")
                ax.grid(True, alpha=0.2)
                ax.set_title(curve.name)

                visibility = _wizard_visibility(4)
                summary = (
                    f"Mode: No linearization (straight curve)\n"
                    f"Paper: {paper}\n"
                    f"Points: {len(curve.output_values)} (identity curve - no correction applied)"
                )
                return (
                    curve,
                    fig,
                    summary,
                    visibility[0],
                    visibility[1],
                    *visibility[2:],
                )

            # Handle auto-linearization modes (single-curve and multi-curve)
            if result is None:
                return (
                    None,
                    None,
                    "Analyze the tablet first before generating a curve.",
                    *_wizard_visibility(2),
                )

            try:
                # Get method and target values from labels
                method_value = get_strategy_value_by_label(strategy_label)
                target_value = get_target_value_by_label(target_label)

                # Convert to enum types
                method_enum = (
                    LinearizationMethod(method_value)
                    if method_value
                    else LinearizationMethod.SPLINE_FIT
                )
                target_enum = (
                    TargetResponse(target_value) if target_value else TargetResponse.LINEAR
                )

                # Determine paper/chemistry
                paper = paper_preset if paper_preset != "Other / custom" else "Custom"
                chemistry = (
                    custom_chemistry
                    if paper_preset == "Other / custom"
                    else get_paper_chemistry_notes(paper_preset)
                )

                # Create linearization config
                config = LinearizationConfig(
                    method=method_enum,
                    target=target_enum,
                    output_points=int(output_points),
                    smoothing=float(smoothing),
                    iterations=int(iterations),
                )

                # Create linearizer and generate curve
                linearizer = AutoLinearizer(config)
                linearization_result = linearizer.linearize(
                    measured_densities=result.densities,
                    curve_name=curve_name or "Wizard Curve",
                    target=target_enum,
                    method=method_enum,
                )

                curve = linearization_result.curve
                # Add paper/chemistry metadata
                curve = CurveData(
                    name=curve.name,
                    input_values=curve.input_values,
                    output_values=curve.output_values,
                    paper_type=paper,
                    chemistry=chemistry,
                    curve_type=curve.curve_type,
                )

                # Create plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(curve.input_values, curve.output_values, color="#f59e0b", linewidth=2)
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")
                ax.grid(True, alpha=0.2)
                ax.set_title(curve.name)

                # Determine mode label for summary
                mode_display = "Single-curve linearization"
                if mode_value == WizardLinearizationMode.MULTI_CURVE.value:
                    mode_display = "Multi-curve / split-tone (single curve preview)"

                visibility = _wizard_visibility(4)
                summary = (
                    f"Mode: {mode_display}\n"
                    f"Target: {target_label}\n"
                    f"Strategy: {strategy_label}\n"
                    f"Paper: {paper}\n"
                    f"Points: {len(curve.output_values)}\n"
                    f"Residual error: {linearization_result.residual_error:.4f}"
                )
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
                    f"Error generating curve: {exc}",
                    *_wizard_visibility(3),
                )

        wizard_generate_curve.click(
            wizard_generate,
            inputs=[
                wizard_analysis_state,
                wizard_linearization_mode,
                wizard_target_response,
                wizard_curve_strategy,
                wizard_paper_preset,
                wizard_existing_profile,
                wizard_custom_chemistry,
                wizard_curve_name,
                wizard_available_curves,
                wizard_smoothing,
                wizard_iterations,
                wizard_output_points,
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

        def wizard_export(curve, fmt):
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
