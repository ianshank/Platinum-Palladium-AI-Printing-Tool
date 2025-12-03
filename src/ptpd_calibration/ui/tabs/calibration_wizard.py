import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from ptpd_calibration.config import TabletType
from ptpd_calibration.analysis import WedgeAnalysisConfig, StepWedgeAnalyzer
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves import save_curve

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
            "Choose linearization method",
            "Review generated curve",
            "Export curve file",
        ]

        def _wizard_visibility(target_step: int):
            updates = [gr.update(visible=index + 1 == target_step) for index in range(5)]
            return (
                target_step,
                f"**Step {target_step} of 5:** {step_titles[target_step-1]}",
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
            wizard_method = gr.Radio(
                [
                    "Spline (Smooth transitions)",
                    "Polynomial",
                    "Iterative (Best fit)",
                ],
                value="Spline (Smooth transitions)",
                label="Method",
            )
            wizard_curve_name = gr.Textbox(label="Curve Name", value="Wizard Curve")
            wizard_paper = gr.Textbox(label="Paper", value="Arches Platine")
            wizard_chemistry = gr.Textbox(
                label="Chemistry Notes",
                placeholder="e.g., 50/50 Pt/Pd, 5 drops Na2",
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
                        f"[{w.level.value.upper()}] {w.message}"
                        for w in result.quality.warnings
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

        def wizard_generate(result, name, paper, chemistry, method):
            if result is None:
                return (
                    None,
                    None,
                    "Analyze the tablet first.",
                    *_wizard_visibility(2),
                )
            try:
                curve_type = CurveType.LINEAR
                analyzer = StepWedgeAnalyzer(
                    WedgeAnalysisConfig(default_curve_type=curve_type)
                )
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
                wizard_paper,
                wizard_chemistry,
                wizard_method,
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

