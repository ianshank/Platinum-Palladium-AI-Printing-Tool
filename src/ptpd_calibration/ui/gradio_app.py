"""
Gradio-based user interface for PTPD Calibration System.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def create_gradio_app(share: bool = False):
    """
    Create the Gradio interface.

    Args:
        share: Whether to create a public share link.

    Returns:
        Gradio Blocks interface.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for UI. Install with: pip install ptpd-calibration[ui]"
        )

    from ptpd_calibration.config import TabletType
    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.core.models import CurveData
    from ptpd_calibration.curves import (
        CurveAnalyzer,
        CurveGenerator,
        load_quad_file,
        load_quad_string,
        CurveModifier,
        SmoothingMethod,
        BlendMode,
        CurveAIEnhancer,
        EnhancementGoal,
        save_curve,
    )
    from ptpd_calibration.detection import StepTabletReader

    # Create the interface
    with gr.Blocks(
        title="Pt/Pd Calibration Studio",
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="stone",
        ),
    ) as app:
        gr.Markdown(
            """
            # Pt/Pd Calibration Studio

            AI-powered calibration system for platinum/palladium printing.
            """
        )

        with gr.Tabs():
            # Analyze Tab
            with gr.TabItem("Analyze Scan"):
                gr.Markdown("### Upload Step Tablet Scan")

                with gr.Row():
                    with gr.Column():
                        scan_input = gr.Image(
                            label="Step Tablet Scan",
                            type="filepath",
                        )
                        tablet_type = gr.Dropdown(
                            choices=["stouffer_21", "stouffer_31", "stouffer_41", "custom"],
                            value="stouffer_21",
                            label="Tablet Type",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column():
                        analysis_output = gr.JSON(label="Analysis Results")
                        density_plot = gr.Plot(label="Density Curve")

                def analyze_scan(image_path, tablet):
                    if image_path is None:
                        return {"error": "No image provided"}, None

                    try:
                        reader = StepTabletReader(tablet_type=TabletType(tablet))
                        result = reader.read(image_path)

                        densities = result.extraction.get_densities()

                        analysis = {
                            "num_patches": result.extraction.num_patches,
                            "dmin": result.extraction.dmin,
                            "dmax": result.extraction.dmax,
                            "density_range": result.extraction.density_range,
                            "quality": result.extraction.overall_quality,
                            "warnings": result.extraction.warnings,
                            "densities": [round(d, 3) for d in densities],
                        }

                        # Create plot
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(8, 5))
                        x = np.linspace(0, 100, len(densities))
                        ax.plot(x, densities, "o-", color="#B8860B", linewidth=2, markersize=6)
                        ax.set_xlabel("Input %")
                        ax.set_ylabel("Density")
                        ax.set_title("Step Tablet Response")
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor("#FAF8F5")
                        fig.patch.set_facecolor("#FAF8F5")

                        return analysis, fig

                    except Exception as e:
                        return {"error": str(e)}, None

                analyze_btn.click(
                    analyze_scan,
                    inputs=[scan_input, tablet_type],
                    outputs=[analysis_output, density_plot],
                )

            # Generate Curve Tab
            with gr.TabItem("Generate Curve"):
                gr.Markdown("### Generate Calibration Curve")

                with gr.Row():
                    with gr.Column():
                        density_input = gr.Textbox(
                            label="Densities (comma-separated)",
                            placeholder="0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0",
                            lines=3,
                        )
                        curve_name = gr.Textbox(
                            label="Curve Name",
                            value="My Calibration",
                        )
                        curve_type = gr.Dropdown(
                            choices=["linear", "paper_white", "aesthetic"],
                            value="linear",
                            label="Curve Type",
                        )
                        generate_btn = gr.Button("Generate Curve", variant="primary")

                    with gr.Column():
                        curve_output = gr.JSON(label="Curve Info")
                        curve_plot = gr.Plot(label="Correction Curve")

                def generate_curve(densities_str, name, ct):
                    try:
                        densities = [float(d.strip()) for d in densities_str.split(",")]

                        generator = CurveGenerator()
                        curve = generator.generate(
                            densities,
                            curve_type=CurveType(ct),
                            name=name,
                        )

                        info = {
                            "name": curve.name,
                            "curve_type": curve.curve_type.value,
                            "num_points": len(curve.input_values),
                        }

                        # Create plot
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(
                            curve.input_values,
                            curve.output_values,
                            "-",
                            color="#8B4513",
                            linewidth=2,
                        )
                        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.set_title(f"Correction Curve: {name}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor("#FAF8F5")
                        fig.patch.set_facecolor("#FAF8F5")

                        return info, fig

                    except Exception as e:
                        return {"error": str(e)}, None

                generate_btn.click(
                    generate_curve,
                    inputs=[density_input, curve_name, curve_type],
                    outputs=[curve_output, curve_plot],
                )

            # Curve Editor Tab
            with gr.TabItem("Curve Editor"):
                gr.Markdown(
                    """
                    ### Curve Editor

                    Upload an existing .quad file or enter curve data to modify, enhance, and export.
                    """
                )

                # State to hold current curve data
                current_curve_inputs = gr.State([])
                current_curve_outputs = gr.State([])
                current_curve_name = gr.State("Edited Curve")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Load Curve")

                        with gr.Tabs():
                            with gr.TabItem("Upload .quad"):
                                quad_upload = gr.File(
                                    label="Upload .quad File",
                                    file_types=[".quad", ".txt"],
                                )
                                channel_select = gr.Dropdown(
                                    choices=["K", "C", "M", "Y", "LC", "LM", "LK"],
                                    value="K",
                                    label="Channel",
                                )
                                upload_quad_btn = gr.Button("Load Quad File")

                            with gr.TabItem("Paste Data"):
                                curve_data_input = gr.Textbox(
                                    label="Curve Values (comma-separated 0-1 outputs)",
                                    placeholder="0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0",
                                    lines=3,
                                )
                                load_data_btn = gr.Button("Load Data")

                            with gr.TabItem("Paste .quad Content"):
                                quad_content_input = gr.Textbox(
                                    label="Paste .quad File Content",
                                    placeholder="[General]\nProfileName=MyProfile\n[K]\n0=0\n1=5\n...",
                                    lines=8,
                                )
                                parse_quad_btn = gr.Button("Parse Content")

                        gr.Markdown("---")
                        gr.Markdown("#### Modify Curve")

                        adjustment_type = gr.Dropdown(
                            choices=["brightness", "contrast", "gamma", "highlights", "shadows", "midtones"],
                            value="brightness",
                            label="Adjustment Type",
                        )
                        adjustment_amount = gr.Slider(
                            minimum=-1.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            label="Amount",
                        )
                        apply_adjust_btn = gr.Button("Apply Adjustment")

                        gr.Markdown("---")
                        gr.Markdown("#### Smooth Curve")

                        smooth_method = gr.Dropdown(
                            choices=["gaussian", "savgol", "moving_average", "spline"],
                            value="gaussian",
                            label="Smoothing Method",
                        )
                        smooth_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Strength",
                        )
                        apply_smooth_btn = gr.Button("Apply Smoothing")

                        gr.Markdown("---")
                        gr.Markdown("#### AI Enhancement")

                        enhance_goal = gr.Dropdown(
                            choices=[
                                "linearization",
                                "maximize_range",
                                "smooth_gradation",
                                "highlight_detail",
                                "shadow_detail",
                                "neutral_midtones",
                                "print_stability",
                            ],
                            value="linearization",
                            label="Enhancement Goal",
                        )
                        enhance_context = gr.Textbox(
                            label="Additional Context (optional)",
                            placeholder="Describe your printing setup or goals...",
                        )
                        apply_enhance_btn = gr.Button("AI Enhance", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("#### Curve Preview")
                        editor_plot = gr.Plot(label="Curve Visualization")

                        editor_info = gr.JSON(label="Curve Information")

                        with gr.Row():
                            export_format = gr.Dropdown(
                                choices=["qtr", "piezography", "csv", "json"],
                                value="qtr",
                                label="Export Format",
                            )
                            export_btn = gr.Button("Export Curve")

                        export_file = gr.File(label="Download Curve")

                        ai_analysis = gr.Textbox(
                            label="AI Analysis",
                            lines=5,
                            visible=True,
                        )

                def create_curve_plot(inputs, outputs, name="Curve"):
                    """Create a matplotlib plot for the curve."""
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 6))

                    if inputs and outputs:
                        ax.plot(inputs, outputs, "-", color="#8B4513", linewidth=2, label=name)

                    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear Reference")
                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.set_title(f"Curve: {name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    return fig

                def load_quad_uploaded(file, channel):
                    """Load curve from uploaded .quad file."""
                    if file is None:
                        return [], [], "No Curve", {"error": "No file uploaded"}, None

                    try:
                        profile = load_quad_file(Path(file.name))

                        if channel.upper() not in profile.channels:
                            return [], [], "No Curve", {"error": f"Channel {channel} not found"}, None

                        curve_data = profile.to_curve_data(channel.upper())
                        inputs = curve_data.input_values
                        outputs = curve_data.output_values
                        name = f"{profile.profile_name} - {channel}"

                        info = {
                            "profile_name": profile.profile_name,
                            "channel": channel,
                            "resolution": profile.resolution,
                            "ink_limit": profile.ink_limit,
                            "active_channels": profile.active_channels,
                            "num_points": len(inputs),
                        }

                        fig = create_curve_plot(inputs, outputs, name)

                        return inputs, outputs, name, info, fig
                    except Exception as e:
                        return [], [], "Error", {"error": str(e)}, None

                def load_data_from_text(data_str):
                    """Load curve from comma-separated values."""
                    try:
                        values = [float(v.strip()) for v in data_str.split(",")]
                        inputs = [i / (len(values) - 1) for i in range(len(values))]
                        outputs = values
                        name = "Custom Curve"

                        info = {
                            "name": name,
                            "num_points": len(values),
                            "min_output": min(values),
                            "max_output": max(values),
                        }

                        fig = create_curve_plot(inputs, outputs, name)

                        return inputs, outputs, name, info, fig
                    except Exception as e:
                        return [], [], "Error", {"error": str(e)}, None

                def parse_quad_content_fn(content, channel="K"):
                    """Parse .quad content from text."""
                    try:
                        profile = load_quad_string(content, "Pasted Profile")

                        if channel.upper() not in profile.channels:
                            return [], [], "No Curve", {"error": f"Channel {channel} not found"}, None

                        curve_data = profile.to_curve_data(channel.upper())
                        inputs = curve_data.input_values
                        outputs = curve_data.output_values
                        name = f"{profile.profile_name} - {channel}"

                        info = {
                            "profile_name": profile.profile_name,
                            "active_channels": profile.active_channels,
                            "num_points": len(inputs),
                        }

                        fig = create_curve_plot(inputs, outputs, name)

                        return inputs, outputs, name, info, fig
                    except Exception as e:
                        return [], [], "Error", {"error": str(e)}, None

                def apply_adjustment(inputs, outputs, name, adj_type, amount):
                    """Apply curve adjustment."""
                    if not inputs or not outputs:
                        return inputs, outputs, name, {"error": "No curve loaded"}, None

                    try:
                        curve = CurveData(
                            name=name,
                            input_values=inputs,
                            output_values=outputs,
                        )

                        modifier = CurveModifier()

                        if adj_type == "brightness":
                            modified = modifier.adjust_brightness(curve, amount)
                        elif adj_type == "contrast":
                            modified = modifier.adjust_contrast(curve, amount)
                        elif adj_type == "gamma":
                            gamma = max(0.1, 1.0 + amount)
                            modified = modifier.adjust_gamma(curve, gamma)
                        elif adj_type == "highlights":
                            modified = modifier.adjust_highlights(curve, amount)
                        elif adj_type == "shadows":
                            modified = modifier.adjust_shadows(curve, amount)
                        elif adj_type == "midtones":
                            modified = modifier.adjust_midtones(curve, amount)
                        else:
                            return inputs, outputs, name, {"error": "Unknown adjustment"}, None

                        new_name = f"{name} ({adj_type} {amount:+.2f})"

                        info = {
                            "name": new_name,
                            "adjustment": adj_type,
                            "amount": amount,
                            "num_points": len(modified.output_values),
                        }

                        fig = create_curve_plot(modified.input_values, modified.output_values, new_name)

                        return modified.input_values, modified.output_values, new_name, info, fig
                    except Exception as e:
                        return inputs, outputs, name, {"error": str(e)}, None

                def apply_smoothing(inputs, outputs, name, method, strength):
                    """Apply curve smoothing."""
                    if not inputs or not outputs:
                        return inputs, outputs, name, {"error": "No curve loaded"}, None

                    try:
                        curve = CurveData(
                            name=name,
                            input_values=inputs,
                            output_values=outputs,
                        )

                        modifier = CurveModifier()
                        smoothed = modifier.smooth(
                            curve,
                            method=SmoothingMethod(method),
                            strength=strength,
                        )

                        new_name = f"{name} (smoothed)"

                        info = {
                            "name": new_name,
                            "method": method,
                            "strength": strength,
                            "num_points": len(smoothed.output_values),
                        }

                        fig = create_curve_plot(smoothed.input_values, smoothed.output_values, new_name)

                        return smoothed.input_values, smoothed.output_values, new_name, info, fig
                    except Exception as e:
                        return inputs, outputs, name, {"error": str(e)}, None

                async def apply_ai_enhancement(inputs, outputs, name, goal, context):
                    """Apply AI enhancement to curve."""
                    if not inputs or not outputs:
                        return inputs, outputs, name, {"error": "No curve loaded"}, None, "No curve loaded"

                    try:
                        curve = CurveData(
                            name=name,
                            input_values=inputs,
                            output_values=outputs,
                        )

                        enhancer = CurveAIEnhancer()

                        try:
                            result = await enhancer.enhance_with_llm(
                                curve,
                                goal=EnhancementGoal(goal),
                                additional_context=context if context else None,
                            )
                        except Exception:
                            result = await enhancer.analyze_and_enhance(
                                curve,
                                goal=EnhancementGoal(goal),
                            )

                        enhanced = result.enhanced_curve
                        new_name = f"{name} (AI enhanced)"

                        info = {
                            "name": new_name,
                            "goal": goal,
                            "confidence": result.confidence,
                            "changes_made": result.changes_made,
                        }

                        fig = create_curve_plot(enhanced.input_values, enhanced.output_values, new_name)

                        analysis_text = result.analysis if result.analysis else "Enhancement applied successfully."

                        return enhanced.input_values, enhanced.output_values, new_name, info, fig, analysis_text
                    except Exception as e:
                        return inputs, outputs, name, {"error": str(e)}, None, f"Error: {str(e)}"

                def export_current_curve(inputs, outputs, name, format_type):
                    """Export current curve to file."""
                    if not inputs or not outputs:
                        return None

                    try:
                        import tempfile

                        curve = CurveData(
                            name=name,
                            input_values=inputs,
                            output_values=outputs,
                        )

                        ext_map = {
                            "qtr": ".txt",
                            "piezography": ".ppt",
                            "csv": ".csv",
                            "json": ".json",
                        }
                        ext = ext_map.get(format_type, ".txt")

                        # Create temp file
                        safe_name = "".join(c for c in name if c.isalnum() or c in " -_")[:50]
                        temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                        save_curve(curve, temp_path, format=format_type)

                        return str(temp_path)
                    except Exception as e:
                        return None

                # Connect event handlers
                upload_quad_btn.click(
                    load_quad_uploaded,
                    inputs=[quad_upload, channel_select],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                load_data_btn.click(
                    load_data_from_text,
                    inputs=[curve_data_input],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                parse_quad_btn.click(
                    lambda content: parse_quad_content_fn(content, "K"),
                    inputs=[quad_content_input],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                apply_adjust_btn.click(
                    apply_adjustment,
                    inputs=[current_curve_inputs, current_curve_outputs, current_curve_name, adjustment_type, adjustment_amount],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                apply_smooth_btn.click(
                    apply_smoothing,
                    inputs=[current_curve_inputs, current_curve_outputs, current_curve_name, smooth_method, smooth_strength],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                apply_enhance_btn.click(
                    apply_ai_enhancement,
                    inputs=[current_curve_inputs, current_curve_outputs, current_curve_name, enhance_goal, enhance_context],
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot, ai_analysis],
                )

                export_btn.click(
                    export_current_curve,
                    inputs=[current_curve_inputs, current_curve_outputs, current_curve_name, export_format],
                    outputs=[export_file],
                )

            # AI Assistant Tab
            with gr.TabItem("AI Assistant"):
                gr.Markdown(
                    """
                    ### Pt/Pd Printing Assistant

                    Ask questions about platinum/palladium printing, get recipe suggestions,
                    or troubleshoot problems.

                    **Note:** Requires PTPD_LLM_API_KEY environment variable.
                    """
                )

                chatbot = gr.Chatbot(label="Chat", height=400)
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask about Pt/Pd printing...",
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

                async def chat(message, history):
                    try:
                        from ptpd_calibration.llm import create_assistant

                        assistant = create_assistant()
                        response = await assistant.chat(message, include_history=False)
                        history = history or []
                        history.append((message, response))
                        return "", history
                    except Exception as e:
                        history = history or []
                        history.append((message, f"Error: {str(e)}"))
                        return "", history

                send_btn.click(
                    chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
                msg_input.submit(
                    chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
                clear_btn.click(lambda: (None, []), outputs=[msg_input, chatbot])

            # Quick Tools Tab
            with gr.TabItem("Quick Tools"):
                gr.Markdown("### Quick Calibration Tools")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Recipe Suggestion")
                        paper_input = gr.Textbox(
                            label="Paper Type",
                            placeholder="Arches Platine",
                        )
                        chars_input = gr.Textbox(
                            label="Desired Characteristics",
                            placeholder="Maximum density range, neutral tones",
                        )
                        recipe_btn = gr.Button("Get Recipe")
                        recipe_output = gr.Textbox(
                            label="Recipe",
                            lines=10,
                        )

                    with gr.Column():
                        gr.Markdown("#### Troubleshooting")
                        problem_input = gr.Textbox(
                            label="Problem Description",
                            placeholder="My highlights are blocking up...",
                            lines=3,
                        )
                        troubleshoot_btn = gr.Button("Troubleshoot")
                        troubleshoot_output = gr.Textbox(
                            label="Solution",
                            lines=10,
                        )

                async def get_recipe(paper, chars):
                    try:
                        from ptpd_calibration.llm import create_assistant

                        assistant = create_assistant()
                        return await assistant.suggest_recipe(paper, chars)
                    except Exception as e:
                        return f"Error: {str(e)}"

                async def troubleshoot(problem):
                    try:
                        from ptpd_calibration.llm import create_assistant

                        assistant = create_assistant()
                        return await assistant.troubleshoot(problem)
                    except Exception as e:
                        return f"Error: {str(e)}"

                recipe_btn.click(
                    get_recipe,
                    inputs=[paper_input, chars_input],
                    outputs=recipe_output,
                )
                troubleshoot_btn.click(
                    troubleshoot,
                    inputs=problem_input,
                    outputs=troubleshoot_output,
                )

            # About Tab
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## Pt/Pd Calibration Studio

                    An AI-powered calibration system for platinum/palladium printing.

                    ### Features

                    - **Step Tablet Analysis**: Upload scans and extract density measurements
                    - **Curve Generation**: Create linearization curves for digital negatives
                    - **Curve Editor**: Upload .quad files, modify curves (brightness, contrast, gamma, highlights, shadows, midtones), smooth curves, and apply AI-powered enhancements
                    - **AI Enhancement**: Intelligent curve optimization for linearization, maximum range, smooth gradation, and more
                    - **AI Assistant**: Get help from an AI expert in Pt/Pd printing
                    - **Recipe Suggestions**: Get starting parameters for new papers
                    - **Troubleshooting**: Diagnose and fix common problems

                    ### Supported Formats

                    - QuadTone RIP (.txt, .quad)
                    - Piezography (.ppt)
                    - CSV
                    - JSON

                    ### Links

                    - [GitHub Repository](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool)
                    - [Documentation](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool#readme)
                    """
                )

    return app


def launch_ui(share: bool = False, port: int = 7860):
    """
    Launch the Gradio UI.

    Args:
        share: Whether to create a public share link.
        port: Port to run on.
    """
    app = create_gradio_app(share=share)
    app.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch_ui()
