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
    from ptpd_calibration.curves import CurveAnalyzer, CurveGenerator
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
