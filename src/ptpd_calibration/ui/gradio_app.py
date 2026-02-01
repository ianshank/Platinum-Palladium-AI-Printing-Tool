"""
Gradio-based user interface for PTPD Calibration System.

Provides comprehensive curve display, step wedge analysis, and calibration tools.
"""

from pathlib import Path

import numpy as np

from ptpd_calibration.ui.tabs.ai_assistant import build_ai_assistant_tab as build_ai_new
from ptpd_calibration.ui.tabs.calibration_wizard import (
    build_calibration_wizard_tab as build_wizard_new,
)
from ptpd_calibration.ui.tabs.chemistry import build_chemistry_tab as build_chemistry_new

# Import new modular tabs
from ptpd_calibration.ui.tabs.dashboard import build_dashboard_tab as build_dashboard_new
from ptpd_calibration.ui.tabs.neural_curve import build_neural_curve_tab as build_neural_new
from ptpd_calibration.ui.tabs.session_log import build_session_log_tab as build_session_log_new


def _patch_gradio_client_utils():
    """
    Monkey-patch gradio_client.utils to handle additionalProperties: true.

    This fixes a bug where Pydantic models with Dict[str, Any] fields
    generate JSON schemas with additionalProperties: true (a boolean),
    but gradio_client expects it to be a dict.
    """
    try:
        import gradio_client.utils as client_utils

        original_get_type = client_utils.get_type

        def patched_get_type(schema):
            # Handle case where schema is a boolean (additionalProperties: true)
            if isinstance(schema, bool):
                return "Any"
            return original_get_type(schema)

        client_utils.get_type = patched_get_type

        # Also patch _json_schema_to_python_type to handle boolean schemas
        original_json_schema_to_python_type = client_utils._json_schema_to_python_type

        def patched_json_schema_to_python_type(schema, defs):
            # Handle case where schema is a boolean
            if isinstance(schema, bool):
                return "Any"
            return original_json_schema_to_python_type(schema, defs)

        client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    except Exception:
        pass  # Silently ignore if patching fails


# Apply the patch before importing gradio
_patch_gradio_client_utils()


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
    except ImportError as err:
        raise ImportError(
            "Gradio is required for UI. Install with: pip install ptpd-calibration[ui]"
        ) from err

    from ptpd_calibration.analysis import (
        StepWedgeAnalyzer,
        WedgeAnalysisConfig,
    )
    from ptpd_calibration.config import TabletType, get_settings
    from ptpd_calibration.core.models import CurveData
    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.curves import (
        ColorScheme,
        CurveAIEnhancer,
        CurveGenerator,
        CurveModifier,
        CurveVisualizer,
        EnhancementGoal,
        PlotStyle,
        SmoothingMethod,
        VisualizationConfig,
        load_quad_file,
        load_quad_string,
        save_curve,
    )
    from ptpd_calibration.detection import StepTabletReader
    from ptpd_calibration.imaging import (
        ExportSettings,
        ImageFormat,
        ImageProcessor,
    )
    from ptpd_calibration.imaging.processor import ColorMode
    from ptpd_calibration.session import (
        SessionLogger,
    )

    # Get settings for configuration-driven defaults
    settings = get_settings()

    # Channel colors for multi-channel view
    CHANNEL_COLORS = {
        "K": "#1a1a1a",
        "C": "#00BFFF",
        "M": "#FF1493",
        "Y": "#FFD700",
        "LC": "#87CEEB",
        "LM": "#FFB6C1",
        "LK": "#696969",
        "LLK": "#A9A9A9",
        "PK": "#2F4F4F",
        "MK": "#4A4A4A",
    }

    # Load custom theme and CSS
    from ptpd_calibration.ui.theme import ProLabTheme

    theme = ProLabTheme()

    css_path = Path(__file__).parent / "styles.css"
    custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    # Legacy inline CSS removed in favor of styles.css

    keyboard_js = """
    document.addEventListener('keydown', (event) => {
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
    });
    """

    session_logger = SessionLogger()

    def build_curve_display_tab():
        # ========================================
        # TAB 1: Curve Display
        # ========================================
        with gr.TabItem("Curve Display"):
            gr.Markdown(
                """
                ### Curve Visualization

                Upload and compare calibration curves with comprehensive statistics.
                """
            )

            # State for loaded curves
            loaded_curves = gr.State([])
            curve_names_list = gr.State([])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Load Curves")

                    with gr.Tabs():
                        with gr.TabItem("Upload File"):
                            curve_file_upload = gr.File(
                                label="Upload Curve File(s)",
                                file_types=[".quad", ".txt", ".csv", ".json"],
                                file_count="multiple",
                            )
                            load_files_btn = gr.Button("Load Files", variant="primary")

                        with gr.TabItem("Paste Data"):
                            paste_curve_data = gr.Textbox(
                                label="Paste Curve Values (comma-separated)",
                                placeholder="0.0, 0.1, 0.2, 0.3, ..., 1.0",
                                lines=3,
                            )
                            paste_curve_name = gr.Textbox(
                                label="Curve Name",
                                value="Custom Curve",
                            )
                            add_pasted_btn = gr.Button("Add Curve")

                    gr.Markdown("---")
                    gr.Markdown("#### Display Options")

                    plot_style = gr.Dropdown(
                        choices=["line", "line_markers", "scatter", "area", "step"],
                        value="line",
                        label="Plot Style",
                    )
                    color_scheme = gr.Dropdown(
                        choices=["platinum", "monochrome", "vibrant", "pastel", "accessible"],
                        value=settings.visualization.color_scheme,
                        label="Color Scheme",
                    )
                    show_reference = gr.Checkbox(
                        label="Show Linear Reference",
                        value=settings.visualization.show_reference_line,
                    )
                    show_statistics = gr.Checkbox(
                        label="Show Statistics Panel",
                        value=False,
                    )
                    show_difference = gr.Checkbox(
                        label="Show Difference Plot",
                        value=False,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Loaded Curves")
                    curves_list_display = gr.Dataframe(
                        headers=["Name", "Points", "Gamma"],
                        label="Loaded Curves",
                        interactive=False,
                    )
                    clear_curves_btn = gr.Button("Clear All Curves")

                with gr.Column(scale=2):
                    gr.Markdown("#### Curve Visualization")
                    curve_display_plot = gr.Plot(label="Curves")

                    with gr.Row():
                        stats_output = gr.JSON(label="Statistics")

                    with gr.Row():
                        gr.JSON(label="Comparison Metrics", visible=False)

            def load_curve_files(files, existing_curves, existing_names):
                """Load curves from uploaded files."""
                if not files:
                    return existing_curves, existing_names, None, None, gr.update()

                curves = list(existing_curves) if existing_curves else []
                names = list(existing_names) if existing_names else []

                for file in files:
                    try:
                        file_path = Path(file.name)
                        suffix = file_path.suffix.lower()

                        if suffix in [".quad", ".txt"]:
                            profile = load_quad_file(file_path)
                            # Get primary channel
                            if profile.primary_channel and profile.primary_channel.enabled:
                                curve = profile.to_curve_data("K")
                                curves.append(curve)
                                names.append(f"{profile.profile_name} (K)")
                            else:
                                # Try to find any active channel
                                active = profile.active_channels
                                if active:
                                    curve = profile.to_curve_data(active[0])
                                    curves.append(curve)
                                    names.append(f"{profile.profile_name} ({active[0]})")
                        elif suffix == ".json" or suffix == ".csv":
                            from ptpd_calibration.curves.export import load_curve

                            curve = load_curve(file_path)
                            curves.append(curve)
                            names.append(curve.name)
                    except Exception:
                        continue

                # Update display
                table_data, plot, stats = update_curve_display(
                    curves, names, "line", "platinum", True, False, False
                )

                return curves, names, table_data, plot, stats

            def add_pasted_curve(data_str, name, existing_curves, existing_names):
                """Add curve from pasted data."""
                if not data_str.strip():
                    return existing_curves, existing_names, None, None, None

                try:
                    values = [float(v.strip()) for v in data_str.split(",")]
                    inputs = [i / (len(values) - 1) for i in range(len(values))]

                    curve = CurveData(
                        name=name or "Custom Curve",
                        input_values=inputs,
                        output_values=values,
                    )

                    curves = list(existing_curves) if existing_curves else []
                    names = list(existing_names) if existing_names else []
                    curves.append(curve)
                    names.append(name or "Custom Curve")

                    table_data, plot, stats = update_curve_display(
                        curves, names, "line", "platinum", True, False, False
                    )

                    return curves, names, table_data, plot, stats
                except Exception:
                    return existing_curves, existing_names, None, None, None

            def update_curve_display(curves, names, style, scheme, show_ref, show_stats, show_diff):
                """Update the curve display with current settings."""
                if not curves:
                    return [], None, {}

                # Create visualizer with settings
                vis_config = VisualizationConfig(
                    figure_width=settings.visualization.figure_width,
                    figure_height=settings.visualization.figure_height,
                    dpi=settings.visualization.dpi,
                    background_color=settings.visualization.background_color,
                    grid_alpha=settings.visualization.grid_alpha,
                    line_width=settings.visualization.line_width,
                    marker_size=settings.visualization.marker_size,
                    color_scheme=ColorScheme(scheme),
                    show_reference_line=show_ref,
                    show_statistics=show_stats,
                    show_difference=show_diff,
                )
                visualizer = CurveVisualizer(vis_config)

                # Compute statistics
                stats_list = []
                table_data = []
                for curve in curves:
                    stats = visualizer.compute_statistics(curve)
                    stats_list.append(stats)
                    table_data.append([curve.name, stats.num_points, round(stats.gamma, 2)])

                # Create plot
                if show_stats and len(curves) > 0:
                    fig = visualizer.plot_with_statistics(curves, title="Curve Comparison")
                elif len(curves) == 1:
                    fig = visualizer.plot_single_curve(
                        curves[0],
                        style=PlotStyle(style),
                        show_stats=show_stats,
                    )
                else:
                    fig = visualizer.plot_multiple_curves(
                        curves,
                        title="Curve Comparison",
                        style=PlotStyle(style),
                        show_difference=show_diff,
                    )

                # Build stats output
                stats_output = {s.name: s.to_dict() for s in stats_list}

                return table_data, fig, stats_output

            def clear_all_curves():
                """Clear all loaded curves."""
                return [], [], [], None, {}

            def on_display_options_change(
                curves, names, style, scheme, show_ref, show_stats, show_diff
            ):
                """Handle display option changes."""
                if not curves:
                    return [], None, {}
                return update_curve_display(
                    curves, names, style, scheme, show_ref, show_stats, show_diff
                )

            # Connect event handlers
            load_files_btn.click(
                load_curve_files,
                inputs=[curve_file_upload, loaded_curves, curve_names_list],
                outputs=[
                    loaded_curves,
                    curve_names_list,
                    curves_list_display,
                    curve_display_plot,
                    stats_output,
                ],
            )

            add_pasted_btn.click(
                add_pasted_curve,
                inputs=[paste_curve_data, paste_curve_name, loaded_curves, curve_names_list],
                outputs=[
                    loaded_curves,
                    curve_names_list,
                    curves_list_display,
                    curve_display_plot,
                    stats_output,
                ],
            )

            clear_curves_btn.click(
                clear_all_curves,
                outputs=[
                    loaded_curves,
                    curve_names_list,
                    curves_list_display,
                    curve_display_plot,
                    stats_output,
                ],
            )

            # Display option change handlers
            for component in [
                plot_style,
                color_scheme,
                show_reference,
                show_statistics,
                show_difference,
            ]:
                component.change(
                    on_display_options_change,
                    inputs=[
                        loaded_curves,
                        curve_names_list,
                        plot_style,
                        color_scheme,
                        show_reference,
                        show_statistics,
                        show_difference,
                    ],
                    outputs=[curves_list_display, curve_display_plot, stats_output],
                )

    def build_step_wedge_tab():
        # ========================================
        with gr.TabItem("Step Wedge Analysis"):
            gr.Markdown(
                """
                ### Step Wedge Scan Analysis

                Upload a step wedge scan to automatically extract densities and generate calibration curves.
                """
            )

            # State for analysis results
            analysis_result_state = gr.State(None)
            generated_curve_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload & Configure")

                    wedge_image_upload = gr.Image(
                        label="Step Wedge Scan",
                        type="filepath",
                    )

                    with gr.Accordion("Analysis Settings", open=True):
                        tablet_type_select = gr.Dropdown(
                            choices=[t.value for t in TabletType],
                            value=settings.wedge_analysis.default_tablet_type,
                            label="Step Tablet Type",
                        )
                        min_density_range = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=settings.wedge_analysis.min_density_range,
                            step=0.1,
                            label="Min Density Range",
                        )
                        auto_fix_reversals = gr.Checkbox(
                            label="Auto-fix Density Reversals",
                            value=settings.wedge_analysis.auto_fix_reversals,
                        )
                        outlier_rejection = gr.Checkbox(
                            label="Outlier Rejection",
                            value=settings.wedge_analysis.outlier_rejection,
                        )

                    analyze_wedge_btn = gr.Button("Analyze Step Wedge", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("#### Curve Generation")

                    with gr.Accordion("Curve Settings", open=True):
                        curve_name_input = gr.Textbox(
                            label="Curve Name",
                            value="",
                            placeholder="Auto-generated if empty",
                        )
                        paper_type_input = gr.Textbox(
                            label="Paper Type",
                            placeholder="e.g., Arches Platine",
                        )
                        chemistry_input = gr.Textbox(
                            label="Chemistry",
                            placeholder="e.g., 50% Pt, 5 drops Na2",
                        )
                        curve_type_select = gr.Dropdown(
                            choices=["linear", "paper_white", "aesthetic"],
                            value=settings.wedge_analysis.default_curve_type,
                            label="Curve Type",
                        )

                    generate_curve_btn = gr.Button("Generate Curve", variant="secondary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Analysis Results")

                    with gr.Row():
                        quality_grade_display = gr.Textbox(
                            label="Quality Grade",
                            interactive=False,
                        )
                        quality_score_display = gr.Number(
                            label="Quality Score",
                            interactive=False,
                        )

                    with gr.Tabs():
                        with gr.TabItem("Density Curve"):
                            density_curve_plot = gr.Plot(label="Measured Densities")

                        with gr.TabItem("Generated Curve"):
                            generated_curve_plot = gr.Plot(label="Calibration Curve")

                        with gr.TabItem("Quality Metrics"):
                            quality_metrics_json = gr.JSON(label="Quality Assessment")

                    with gr.Accordion("Warnings & Recommendations", open=False):
                        warnings_output = gr.Textbox(
                            label="Analysis Warnings",
                            lines=4,
                            interactive=False,
                        )
                        recommendations_output = gr.Textbox(
                            label="Recommendations",
                            lines=3,
                            interactive=False,
                        )

                    with gr.Row():
                        export_format_select = gr.Dropdown(
                            choices=["qtr", "piezography", "csv", "json"],
                            value="qtr",
                            label="Export Format",
                        )
                        export_curve_btn = gr.Button("Export Curve")
                    export_file_output = gr.File(label="Download Curve")

            def analyze_step_wedge(
                image_path, tablet_type, min_range, fix_reversals, reject_outliers
            ):
                """Analyze uploaded step wedge scan."""
                if image_path is None:
                    return None, "No Image", 0, None, {}, "", "", gr.update()

                try:
                    config = WedgeAnalysisConfig(
                        tablet_type=TabletType(tablet_type),
                        min_density_range=min_range,
                        auto_fix_reversals=fix_reversals,
                        outlier_rejection=reject_outliers,
                    )

                    analyzer = StepWedgeAnalyzer(config)
                    result = analyzer.analyze(
                        image_path,
                        generate_curve=False,  # Generate separately
                    )

                    # Quality info
                    grade = result.quality.grade.value.upper() if result.quality else "N/A"
                    score = result.quality.score if result.quality else 0

                    # Create density plot
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 6))

                    if result.densities:
                        x = np.linspace(0, 100, len(result.densities))
                        ax.plot(
                            x,
                            result.densities,
                            "o-",
                            color="#B8860B",
                            linewidth=2,
                            markersize=6,
                            label="Measured",
                        )

                        if result.raw_densities and result.raw_densities != result.densities:
                            ax.plot(
                                x,
                                result.raw_densities,
                                "x--",
                                color="#808080",
                                alpha=0.5,
                                label="Raw",
                            )
                            ax.legend()

                    ax.set_xlabel("Input %")
                    ax.set_ylabel("Density")
                    ax.set_title(
                        f"Step Wedge Response (Dmin: {result.dmin:.3f}, Dmax: {result.dmax:.3f})"
                    )
                    ax.grid(True, alpha=0.3)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    # Quality metrics
                    quality_metrics = result.quality.to_dict() if result.quality else {}

                    # Warnings
                    warnings_text = ""
                    if result.quality and result.quality.warnings:
                        warnings_text = "\n".join(
                            [
                                f"[{w.level.value.upper()}] {w.message}"
                                for w in result.quality.warnings
                            ]
                        )

                    # Recommendations
                    recommendations_text = ""
                    if result.quality and result.quality.recommendations:
                        recommendations_text = "\n".join(
                            [f"• {r}" for r in result.quality.recommendations]
                        )

                    return (
                        result,
                        grade,
                        score,
                        fig,
                        quality_metrics,
                        warnings_text,
                        recommendations_text,
                        gr.update(visible=True),
                    )

                except Exception as e:
                    return None, f"Error: {str(e)}", 0, None, {}, str(e), "", gr.update()

            def generate_calibration_curve(result, curve_name, paper_type, chemistry, curve_type):
                """Generate calibration curve from analysis result."""
                if result is None or not result.densities:
                    return None, None, gr.update()

                try:
                    config = WedgeAnalysisConfig(
                        default_curve_type=CurveType(curve_type),
                    )
                    analyzer = StepWedgeAnalyzer(config)

                    # Generate curve from existing densities
                    analysis = analyzer.analyze_from_densities(
                        result.densities,
                        curve_name=curve_name if curve_name else None,
                        paper_type=paper_type if paper_type else None,
                        chemistry=chemistry if chemistry else None,
                        generate_curve=True,
                        curve_type=CurveType(curve_type),
                    )

                    curve = analysis.curve
                    if curve is None:
                        return None, None, gr.update()

                    # Create curve plot
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.plot(
                        curve.input_values,
                        curve.output_values,
                        "-",
                        color="#8B4513",
                        linewidth=2,
                        label=curve.name,
                    )
                    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear Reference")

                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.set_title(f"Calibration Curve: {curve.name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    return curve, fig, gr.update(visible=True)

                except Exception:
                    return None, None, gr.update()

            def export_generated_curve(curve, format_type):
                """Export the generated curve."""
                if curve is None:
                    return None

                try:
                    import tempfile

                    ext_map = {
                        "qtr": ".quad",
                        "piezography": ".ppt",
                        "csv": ".csv",
                        "json": ".json",
                    }
                    ext = ext_map.get(format_type, ".quad")

                    safe_name = "".join(c for c in curve.name if c.isalnum() or c in " -_")[:50]
                    temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                    save_curve(curve, temp_path, format=format_type)
                    return str(temp_path)
                except Exception:
                    return None

            # Connect handlers
            analyze_wedge_btn.click(
                analyze_step_wedge,
                inputs=[
                    wedge_image_upload,
                    tablet_type_select,
                    min_density_range,
                    auto_fix_reversals,
                    outlier_rejection,
                ],
                outputs=[
                    analysis_result_state,
                    quality_grade_display,
                    quality_score_display,
                    density_curve_plot,
                    quality_metrics_json,
                    warnings_output,
                    recommendations_output,
                    generate_curve_btn,
                ],
            )

            generate_curve_btn.click(
                generate_calibration_curve,
                inputs=[
                    analysis_result_state,
                    curve_name_input,
                    paper_type_input,
                    chemistry_input,
                    curve_type_select,
                ],
                outputs=[generated_curve_state, generated_curve_plot, export_curve_btn],
            )

            export_curve_btn.click(
                export_generated_curve,
                inputs=[generated_curve_state, export_format_select],
                outputs=[export_file_output],
            )

    def build_step_tablet_reader_tab():
        # ========================================
        with gr.TabItem("Step Tablet Reader"):
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

    def build_generate_curve_tab():
        # ========================================
        with gr.TabItem("Curve Generator"):
            gr.Markdown("### Generate Calibration Curve")

            with gr.Row():
                with gr.Column():
                    density_input = gr.Textbox(
                        label="Densities (comma-separated)",
                        placeholder="0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0",
                        lines=3,
                    )
                    gen_curve_name = gr.Textbox(
                        label="Curve Name",
                        value="My Calibration",
                    )
                    gen_curve_type = gr.Dropdown(
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
                inputs=[density_input, gen_curve_name, gen_curve_type],
                outputs=[curve_output, curve_plot],
            )

    def build_curve_editor_tab():
        # ========================================
        with gr.TabItem("Curve Editor"):
            gr.Markdown(
                """
                ### Curve Editor

                Upload an existing .quad file or enter curve data to modify, enhance, and export.
                """
            )

            # State to hold current curve data and profile
            current_curve_inputs = gr.State([])
            current_curve_outputs = gr.State([])
            current_curve_name = gr.State("Edited Curve")
            current_profile_data = gr.State(None)

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
                                choices=[
                                    "ALL",
                                    "K",
                                    "C",
                                    "M",
                                    "Y",
                                    "LC",
                                    "LM",
                                    "LK",
                                    "LLK",
                                    "V",
                                    "MK",
                                    "PK",
                                ],
                                value="K",
                                label="Channel",
                                interactive=True,
                            )
                            show_all_channels = gr.Checkbox(
                                label="Show all channels",
                                value=False,
                            )
                            upload_quad_btn = gr.Button("Load Quad File", variant="primary")

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
                        choices=[
                            "brightness",
                            "contrast",
                            "gamma",
                            "highlights",
                            "shadows",
                            "midtones",
                        ],
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
                        export_btn = gr.Button("Export Curve", elem_id="save-curve-btn")

                    export_file = gr.File(label="Download Curve")

                    ai_analysis = gr.Textbox(
                        label="AI Analysis",
                        lines=5,
                        visible=True,
                    )

            def create_curve_plot(
                inputs,
                outputs,
                name="Curve",
                profile_data=None,
                show_all=False,
                selected_channel="K",
            ):
                """Create a matplotlib plot for the curve with multi-channel support."""
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))

                if show_all and profile_data is not None:
                    channels_data = profile_data.get("channels", {})
                    for ch_name, ch_data in channels_data.items():
                        ch_inputs = ch_data.get("inputs", [])
                        ch_outputs = ch_data.get("outputs", [])
                        if ch_inputs and ch_outputs:
                            color = CHANNEL_COLORS.get(ch_name, "#8B4513")
                            linewidth = 2.5 if ch_name == selected_channel else 1.5
                            alpha = 1.0 if ch_name == selected_channel else 0.6
                            ax.plot(
                                ch_inputs,
                                ch_outputs,
                                "-",
                                color=color,
                                linewidth=linewidth,
                                alpha=alpha,
                                label=ch_name,
                            )
                elif inputs and outputs:
                    color = CHANNEL_COLORS.get(selected_channel, "#8B4513")
                    ax.plot(inputs, outputs, "-", color=color, linewidth=2, label=name)

                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear Reference")
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")

                if show_all and profile_data:
                    ax.set_title(f"All Channels - {profile_data.get('profile_name', 'Profile')}")
                else:
                    ax.set_title(f"Curve: {name}")

                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_facecolor("#FAF8F5")
                fig.patch.set_facecolor("#FAF8F5")

                return fig

            def load_quad_uploaded(file, channel, show_all):
                """Load curve from uploaded .quad file."""
                if file is None:
                    return (
                        [],
                        [],
                        "No Curve",
                        {"error": "No file uploaded"},
                        None,
                        None,
                        gr.update(),
                    )

                try:
                    profile = load_quad_file(Path(file.name))

                    profile_data = {
                        "profile_name": profile.profile_name,
                        "resolution": profile.resolution,
                        "ink_limit": profile.ink_limit,
                        "media_type": profile.media_type,
                        "active_channels": profile.active_channels,
                        "all_channels": profile.all_channel_names,
                        "channels": {},
                    }

                    for ch_name in profile.all_channel_names:
                        ch = profile.get_channel(ch_name)
                        if ch:
                            ch_inputs, ch_outputs = ch.as_normalized
                            profile_data["channels"][ch_name] = {
                                "inputs": ch_inputs,
                                "outputs": ch_outputs,
                            }

                    available_channels = (
                        profile.all_channel_names if profile.all_channel_names else ["K"]
                    )
                    selected_channel = (
                        channel.upper()
                        if channel.upper() in available_channels
                        else available_channels[0]
                    )

                    if selected_channel in profile.channels:
                        curve_data = profile.to_curve_data(selected_channel)
                        inputs = curve_data.input_values
                        outputs = curve_data.output_values
                    else:
                        inputs = []
                        outputs = []

                    name = f"{profile.profile_name} - {selected_channel}"

                    info = {
                        "profile_name": profile.profile_name,
                        "selected_channel": selected_channel,
                        "resolution": profile.resolution,
                        "ink_limit": profile.ink_limit,
                        "media_type": profile.media_type or "Not specified",
                        "active_channels": profile.active_channels,
                        "all_channels": available_channels,
                        "num_points": len(inputs),
                    }

                    fig = create_curve_plot(
                        inputs,
                        outputs,
                        name,
                        profile_data=profile_data,
                        show_all=show_all,
                        selected_channel=selected_channel,
                    )

                    # Add "ALL" option at the beginning of channel choices
                    dropdown_choices = ["ALL"] + available_channels
                    dropdown_update = gr.update(choices=dropdown_choices, value=selected_channel)

                    return inputs, outputs, name, info, fig, profile_data, dropdown_update
                except Exception as e:
                    import traceback

                    error_detail = f"{str(e)}\n{traceback.format_exc()}"
                    return (
                        [],
                        [],
                        "Error",
                        {"error": str(e), "detail": error_detail},
                        None,
                        None,
                        gr.update(),
                    )

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

            def parse_quad_content_fn(content, channel="K", show_all=False):
                """Parse .quad content from text."""
                try:
                    profile = load_quad_string(content, "Pasted Profile")

                    profile_data = {
                        "profile_name": profile.profile_name,
                        "active_channels": profile.active_channels,
                        "all_channels": profile.all_channel_names,
                        "channels": {},
                    }

                    for ch_name in profile.all_channel_names:
                        ch = profile.get_channel(ch_name)
                        if ch:
                            ch_inputs, ch_outputs = ch.as_normalized
                            profile_data["channels"][ch_name] = {
                                "inputs": ch_inputs,
                                "outputs": ch_outputs,
                            }

                    available_channels = (
                        profile.all_channel_names if profile.all_channel_names else ["K"]
                    )
                    selected_channel = (
                        channel.upper()
                        if channel.upper() in available_channels
                        else available_channels[0]
                    )

                    if selected_channel in profile.channels:
                        curve_data = profile.to_curve_data(selected_channel)
                        inputs = curve_data.input_values
                        outputs = curve_data.output_values
                    else:
                        inputs = []
                        outputs = []

                    name = f"{profile.profile_name} - {selected_channel}"

                    info = {
                        "profile_name": profile.profile_name,
                        "selected_channel": selected_channel,
                        "active_channels": profile.active_channels,
                        "all_channels": available_channels,
                        "num_points": len(inputs),
                    }

                    fig = create_curve_plot(
                        inputs,
                        outputs,
                        name,
                        profile_data=profile_data,
                        show_all=show_all,
                        selected_channel=selected_channel,
                    )

                    # Add "ALL" option at the beginning of channel choices
                    dropdown_choices = ["ALL"] + available_channels
                    dropdown_update = gr.update(choices=dropdown_choices, value=selected_channel)

                    return inputs, outputs, name, info, fig, profile_data, dropdown_update
                except Exception as e:
                    return [], [], "Error", {"error": str(e)}, None, None, gr.update()

            def on_channel_change(channel, profile_data, show_all):
                """Handle channel selection change."""
                if profile_data is None:
                    return [], [], "No Curve", {"error": "No profile loaded"}, None

                try:
                    channels_data = profile_data.get("channels", {})
                    selected_channel = channel.upper()

                    # Handle "ALL" channel selection - show all channels
                    if selected_channel == "ALL":
                        # Use K channel data as the primary, but show all
                        active_channels = profile_data.get("active_channels", [])
                        primary_channel = active_channels[0] if active_channels else "K"

                        if primary_channel in channels_data:
                            ch_data = channels_data[primary_channel]
                            inputs = ch_data.get("inputs", [])
                            outputs = ch_data.get("outputs", [])
                        else:
                            inputs = []
                            outputs = []

                        name = f"{profile_data.get('profile_name', 'Profile')} - All Channels"

                        info = {
                            "profile_name": profile_data.get("profile_name"),
                            "selected_channel": "ALL",
                            "active_channels": active_channels,
                            "all_channels": profile_data.get("all_channels", []),
                            "num_points": len(inputs),
                            "display_mode": "all_channels",
                        }

                        fig = create_curve_plot(
                            inputs,
                            outputs,
                            name,
                            profile_data=profile_data,
                            show_all=True,  # Force show_all when ALL is selected
                            selected_channel=primary_channel,
                        )

                        return inputs, outputs, name, info, fig

                    if selected_channel not in channels_data:
                        return [], [], "No Curve", {"error": f"Channel {channel} not found"}, None

                    ch_data = channels_data[selected_channel]
                    inputs = ch_data.get("inputs", [])
                    outputs = ch_data.get("outputs", [])
                    name = f"{profile_data.get('profile_name', 'Profile')} - {selected_channel}"

                    info = {
                        "profile_name": profile_data.get("profile_name"),
                        "selected_channel": selected_channel,
                        "active_channels": profile_data.get("active_channels", []),
                        "all_channels": profile_data.get("all_channels", []),
                        "num_points": len(inputs),
                    }

                    fig = create_curve_plot(
                        inputs,
                        outputs,
                        name,
                        profile_data=profile_data,
                        show_all=show_all,
                        selected_channel=selected_channel,
                    )

                    return inputs, outputs, name, info, fig
                except Exception as e:
                    return [], [], "Error", {"error": str(e)}, None

            def on_show_all_toggle(show_all, channel, profile_data, inputs, outputs, name):
                """Handle show all channels checkbox toggle."""
                if profile_data is None:
                    fig = create_curve_plot(inputs, outputs, name)
                    return fig

                selected_channel = channel.upper() if channel else "K"

                fig = create_curve_plot(
                    inputs,
                    outputs,
                    name,
                    profile_data=profile_data,
                    show_all=show_all,
                    selected_channel=selected_channel,
                )

                return fig

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
                    return (
                        inputs,
                        outputs,
                        name,
                        {"error": "No curve loaded"},
                        None,
                        "No curve loaded",
                    )

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
                        "adjustments_applied": result.adjustments_applied,
                    }

                    fig = create_curve_plot(enhanced.input_values, enhanced.output_values, new_name)

                    # Handle analysis which can be a dict or string
                    if isinstance(result.analysis, dict):
                        analysis_text = result.analysis.get(
                            "summary", "Enhancement applied successfully."
                        )
                    elif result.analysis:
                        analysis_text = str(result.analysis)
                    else:
                        analysis_text = "Enhancement applied successfully."

                    return (
                        enhanced.input_values,
                        enhanced.output_values,
                        new_name,
                        info,
                        fig,
                        analysis_text,
                    )
                except Exception as e:
                    return inputs, outputs, name, {"error": str(e)}, None, f"Error: {str(e)}"

            def export_current_curve(inputs, outputs, name, format_type, profile_data):
                """Export current curve to file, preserving all channels if available."""
                if not inputs or not outputs:
                    return None

                try:
                    import tempfile

                    ext_map = {
                        "qtr": ".quad",
                        "piezography": ".ppt",
                        "csv": ".csv",
                        "json": ".json",
                    }
                    ext = ext_map.get(format_type, ".quad")

                    safe_name = "".join(c for c in name if c.isalnum() or c in " -_")[:50]
                    temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                    # For QTR format with profile data, export all channels
                    if format_type == "qtr" and profile_data is not None:
                        channels_data = profile_data.get("channels", {})
                        if channels_data:
                            # Export full multi-channel .quad file in QTR format
                            _export_multi_channel_quad(
                                temp_path,
                                name,
                                channels_data,
                                profile_data.get("resolution", 2880),
                                profile_data.get("ink_limit", 100.0),
                                profile_data.get("comments", []),
                            )
                            return str(temp_path)

                    # Fallback to single curve export
                    curve = CurveData(
                        name=name,
                        input_values=inputs,
                        output_values=outputs,
                    )
                    save_curve(curve, temp_path, format=format_type)

                    return str(temp_path)
                except Exception:
                    import traceback

                    traceback.print_exc()
                    return None

            def _export_multi_channel_quad(
                path, name, channels_data, resolution=2880, ink_limit=100.0, comments=None
            ):
                """Export a multi-channel .quad file in QuadTone RIP format.

                QuadTone RIP format uses:
                - Header line: ## QuadToneRIP K,C,M,Y,LC,LM,LK,LLK
                - Comments starting with #
                - Channel headers as comments: # K Curve
                - 256 values per channel (one per line, no index=)
                - 16-bit values (0-65535 range)
                """
                import numpy as np

                # Standard channel order for .quad files
                channel_order = ["K", "C", "M", "Y", "LC", "LM", "LK", "LLK"]

                # Build header with active channels
                active_channels = [ch for ch in channel_order if ch in channels_data]
                header = f"## QuadToneRIP {','.join(active_channels)}"

                lines = [header]

                # Add profile name and metadata as comments
                lines.append(f"# Profile: {name}")
                lines.append(f"# Resolution: {resolution}")
                lines.append(f"# Ink Limit: {ink_limit}%")

                # Add any original comments
                if comments:
                    for comment in comments[:5]:  # Limit to first 5 comments
                        if not comment.startswith("#"):
                            lines.append(f"# {comment}")
                        else:
                            lines.append(comment)

                for channel in channel_order:
                    if channel in channels_data:
                        ch_data = channels_data[channel]
                        ch_inputs = ch_data.get("inputs", [])
                        ch_outputs = ch_data.get("outputs", [])

                        # Channel header as comment
                        lines.append(f"# {channel} Curve")

                        if ch_inputs and ch_outputs:
                            # Interpolate to 256 points
                            x_new = np.linspace(0, 1, 256)
                            y_new = np.interp(x_new, ch_inputs, ch_outputs)

                            for i in range(256):
                                # Convert normalized (0-1) to 16-bit (0-65535)
                                # Apply ink limit
                                qtr_output = int(y_new[i] * 65535 * ink_limit / 100)
                                qtr_output = max(0, min(65535, qtr_output))
                                lines.append(str(qtr_output))
                        else:
                            # Empty channel - 256 zeros
                            for _i in range(256):
                                lines.append("0")

                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write("\n".join(lines))

            # Connect event handlers
            upload_quad_btn.click(
                load_quad_uploaded,
                inputs=[quad_upload, channel_select, show_all_channels],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                    current_profile_data,
                    channel_select,
                ],
            )

            load_data_btn.click(
                load_data_from_text,
                inputs=[curve_data_input],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                ],
            )

            parse_quad_btn.click(
                lambda content, ch, show: parse_quad_content_fn(content, ch, show),
                inputs=[quad_content_input, channel_select, show_all_channels],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                    current_profile_data,
                    channel_select,
                ],
            )

            channel_select.change(
                on_channel_change,
                inputs=[channel_select, current_profile_data, show_all_channels],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                ],
            )

            show_all_channels.change(
                on_show_all_toggle,
                inputs=[
                    show_all_channels,
                    channel_select,
                    current_profile_data,
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                ],
                outputs=[editor_plot],
            )

            apply_adjust_btn.click(
                apply_adjustment,
                inputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    adjustment_type,
                    adjustment_amount,
                ],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                ],
            )

            apply_smooth_btn.click(
                apply_smoothing,
                inputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    smooth_method,
                    smooth_strength,
                ],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                ],
            )

            apply_enhance_btn.click(
                apply_ai_enhancement,
                inputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    enhance_goal,
                    enhance_context,
                ],
                outputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    editor_info,
                    editor_plot,
                    ai_analysis,
                ],
            )

            export_btn.click(
                export_current_curve,
                inputs=[
                    current_curve_inputs,
                    current_curve_outputs,
                    current_curve_name,
                    export_format,
                    current_profile_data,
                ],
                outputs=[export_file],
            )

    def build_quick_tools_tab():
        # ========================================
        with gr.TabItem("AI Curve Enhancement"):
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

    def build_image_preview_tab():
        # ========================================
        with gr.TabItem("Image Preview"):
            gr.Markdown(
                """
                ### Curve Preview on Image

                Upload an image and select a curve to preview how the curve will affect your image.
                """
            )

            # State for preview
            preview_curve_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload Image")
                    preview_image_upload = gr.Image(
                        label="Source Image",
                        type="filepath",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Select Curve")

                    with gr.Tabs():
                        with gr.TabItem("Upload Curve"):
                            preview_curve_file = gr.File(
                                label="Upload Curve File",
                                file_types=[".quad", ".txt", ".csv", ".json"],
                            )
                            load_preview_curve_btn = gr.Button("Load Curve")

                        with gr.TabItem("Enter Values"):
                            preview_curve_values = gr.Textbox(
                                label="Curve Values (comma-separated, 0-1)",
                                placeholder="0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0",
                                lines=2,
                            )
                            preview_curve_name_input = gr.Textbox(
                                label="Curve Name",
                                value="Custom Preview Curve",
                            )
                            load_custom_curve_btn = gr.Button("Load Values")

                    preview_curve_info = gr.JSON(label="Loaded Curve")

                    gr.Markdown("---")
                    gr.Markdown("#### Preview Options")

                    preview_color_mode = gr.Dropdown(
                        choices=[
                            ("Preserve Original", "preserve"),
                            ("Grayscale", "grayscale"),
                            ("RGB", "rgb"),
                        ],
                        value="preserve",
                        label="Color Mode",
                    )

                    generate_preview_btn = gr.Button("Generate Preview", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Before / After Comparison")

                    with gr.Row():
                        original_preview = gr.Image(
                            label="Original",
                            interactive=False,
                        )
                        processed_preview = gr.Image(
                            label="With Curve Applied",
                            interactive=False,
                        )

                    preview_info_display = gr.Textbox(
                        label="Preview Info",
                        interactive=False,
                        lines=3,
                    )

            def load_curve_for_preview(file):
                """Load curve from file for preview."""
                if file is None:
                    return None, {"error": "No file uploaded"}

                try:
                    file_path = Path(file.name)
                    suffix = file_path.suffix.lower()

                    if suffix in [".quad", ".txt"]:
                        profile = load_quad_file(file_path)
                        curve = profile.to_curve_data("K")
                    elif suffix == ".json" or suffix == ".csv":
                        from ptpd_calibration.curves.export import load_curve

                        curve = load_curve(file_path)
                    else:
                        return None, {"error": f"Unsupported format: {suffix}"}

                    return curve, {
                        "name": curve.name,
                        "points": len(curve.input_values),
                        "range": f"{min(curve.output_values):.3f} - {max(curve.output_values):.3f}",
                    }
                except Exception as e:
                    return None, {"error": str(e)}

            def load_custom_curve_values(values_str, name):
                """Load curve from custom values."""
                if not values_str.strip():
                    return None, {"error": "No values provided"}

                try:
                    values = [float(v.strip()) for v in values_str.split(",")]
                    inputs = [i / (len(values) - 1) for i in range(len(values))]

                    curve = CurveData(
                        name=name or "Custom Curve",
                        input_values=inputs,
                        output_values=values,
                    )

                    return curve, {
                        "name": curve.name,
                        "points": len(curve.input_values),
                        "range": f"{min(values):.3f} - {max(values):.3f}",
                    }
                except Exception as e:
                    return None, {"error": str(e)}

            def generate_image_preview(image_path, curve, color_mode_str):
                """Generate before/after preview."""
                if image_path is None:
                    return None, None, "No image uploaded"

                if curve is None:
                    return None, None, "No curve loaded"

                try:
                    processor = ImageProcessor()
                    color_mode = ColorMode(color_mode_str)

                    # Generate preview
                    original, processed = processor.preview_curve_effect(
                        image_path,
                        curve,
                        color_mode=color_mode,
                        thumbnail_size=(800, 800),
                    )

                    info = f"Curve: {curve.name}\nColor Mode: {color_mode_str}\nOriginal Size: {original.size}"

                    return original, processed, info
                except Exception as e:
                    return None, None, f"Error: {str(e)}"

            # Connect handlers
            load_preview_curve_btn.click(
                load_curve_for_preview,
                inputs=[preview_curve_file],
                outputs=[preview_curve_state, preview_curve_info],
            )

            load_custom_curve_btn.click(
                load_custom_curve_values,
                inputs=[preview_curve_values, preview_curve_name_input],
                outputs=[preview_curve_state, preview_curve_info],
            )

            generate_preview_btn.click(
                generate_image_preview,
                inputs=[preview_image_upload, preview_curve_state, preview_color_mode],
                outputs=[original_preview, processed_preview, preview_info_display],
            )

    def build_digital_negative_tab():
        # ========================================
        with gr.TabItem("Digital Negative"):
            gr.Markdown(
                """
                ### Digital Negative Creator

                Create inverted digital negatives with calibration curves applied.
                Export in the same format and resolution as your original, or choose a different format.
                """
            )

            # State
            dn_curve_state = gr.State(None)
            dn_result_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Source Image")
                    dn_image_upload = gr.Image(
                        label="Upload Image",
                        type="filepath",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Calibration Curve (Optional)")

                    with gr.Tabs():
                        with gr.TabItem("Upload Curve"):
                            dn_curve_file = gr.File(
                                label="Upload Curve File",
                                file_types=[".quad", ".txt", ".csv", ".json"],
                            )
                            load_dn_curve_btn = gr.Button("Load Curve")

                        with gr.TabItem("Enter Values"):
                            dn_curve_values = gr.Textbox(
                                label="Curve Values (comma-separated)",
                                lines=2,
                            )
                            dn_curve_name_input = gr.Textbox(
                                label="Curve Name",
                                value="Digital Negative Curve",
                            )
                            load_dn_custom_btn = gr.Button("Load Values")

                    dn_curve_info = gr.JSON(label="Loaded Curve")

                    gr.Markdown("---")
                    gr.Markdown("#### Processing Options")

                    dn_invert = gr.Checkbox(
                        label="Invert Image (Create Negative)",
                        value=True,
                    )

                    dn_color_mode = gr.Dropdown(
                        choices=[
                            ("Grayscale", "grayscale"),
                            ("Preserve Original", "preserve"),
                            ("RGB", "rgb"),
                        ],
                        value="grayscale",
                        label="Color Mode",
                    )

                    process_dn_btn = gr.Button("Create Digital Negative", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Result")

                    dn_result_image = gr.Image(
                        label="Digital Negative Preview",
                        interactive=False,
                    )

                    dn_info_display = gr.JSON(label="Processing Info")

                    gr.Markdown("---")
                    gr.Markdown("#### Export")

                    dn_export_format = gr.Dropdown(
                        choices=[
                            ("Same as Original", "original"),
                            ("TIFF (Lossless)", "tiff"),
                            ("TIFF 16-bit", "tiff_16bit"),
                            ("PNG (Lossless)", "png"),
                            ("PNG 16-bit", "png_16bit"),
                            ("JPEG (Standard)", "jpeg"),
                            ("JPEG (High Quality)", "jpeg_high"),
                        ],
                        value="original",
                        label="Export Format",
                    )

                    dn_jpeg_quality = gr.Slider(
                        minimum=50,
                        maximum=100,
                        value=95,
                        step=5,
                        label="JPEG Quality (if applicable)",
                    )

                    export_dn_btn = gr.Button("Export Digital Negative", variant="secondary")
                    dn_export_file = gr.File(label="Download")

            def load_dn_curve(file):
                """Load curve for digital negative."""
                if file is None:
                    return None, {"status": "No curve (will invert only)"}

                try:
                    file_path = Path(file.name)
                    suffix = file_path.suffix.lower()

                    if suffix in [".quad", ".txt"]:
                        profile = load_quad_file(file_path)
                        curve = profile.to_curve_data("K")
                    elif suffix == ".json" or suffix == ".csv":
                        from ptpd_calibration.curves.export import load_curve

                        curve = load_curve(file_path)
                    else:
                        return None, {"error": f"Unsupported: {suffix}"}

                    return curve, {
                        "name": curve.name,
                        "points": len(curve.input_values),
                    }
                except Exception as e:
                    return None, {"error": str(e)}

            def load_dn_custom_values(values_str, name):
                """Load custom curve values."""
                if not values_str.strip():
                    return None, {"status": "No curve (will invert only)"}

                try:
                    values = [float(v.strip()) for v in values_str.split(",")]
                    inputs = [i / (len(values) - 1) for i in range(len(values))]

                    curve = CurveData(
                        name=name or "Custom Curve",
                        input_values=inputs,
                        output_values=values,
                    )

                    return curve, {
                        "name": curve.name,
                        "points": len(curve.input_values),
                    }
                except Exception as e:
                    return None, {"error": str(e)}

            def create_digital_negative(image_path, curve, invert, color_mode_str):
                """Create digital negative from image."""
                if image_path is None:
                    return None, None, {"error": "No image uploaded"}

                try:
                    processor = ImageProcessor()
                    color_mode = ColorMode(color_mode_str)

                    result = processor.create_digital_negative(
                        image_path,
                        curve=curve,
                        invert=invert,
                        color_mode=color_mode,
                    )

                    return result, result.image, result.get_info()
                except Exception as e:
                    return None, None, {"error": str(e)}

            def export_digital_negative(result, export_format, jpeg_quality):
                """Export the digital negative."""
                if result is None:
                    return None

                try:
                    import tempfile

                    processor = ImageProcessor()

                    settings = ExportSettings(
                        format=ImageFormat(export_format),
                        jpeg_quality=int(jpeg_quality),
                    )

                    # Determine extension
                    ext_map = {
                        "original": result.original_format or "png",
                        "tiff": "tiff",
                        "tiff_16bit": "tiff",
                        "png": "png",
                        "png_16bit": "png",
                        "jpeg": "jpg",
                        "jpeg_high": "jpg",
                    }
                    ext = ext_map.get(export_format, "png")

                    temp_path = Path(tempfile.gettempdir()) / f"digital_negative.{ext}"
                    processor.export(result, temp_path, settings)

                    return str(temp_path)
                except Exception:
                    return None

            # Connect handlers
            load_dn_curve_btn.click(
                load_dn_curve,
                inputs=[dn_curve_file],
                outputs=[dn_curve_state, dn_curve_info],
            )

            load_dn_custom_btn.click(
                load_dn_custom_values,
                inputs=[dn_curve_values, dn_curve_name_input],
                outputs=[dn_curve_state, dn_curve_info],
            )

            process_dn_btn.click(
                create_digital_negative,
                inputs=[dn_image_upload, dn_curve_state, dn_invert, dn_color_mode],
                outputs=[dn_result_state, dn_result_image, dn_info_display],
            )

            export_dn_btn.click(
                export_digital_negative,
                inputs=[dn_result_state, dn_export_format, dn_jpeg_quality],
                outputs=[dn_export_file],
            )

    def build_interactive_editor_tab():
        # ========================================
        with gr.TabItem("Interactive Editor"):
            gr.Markdown(
                """
                ### Interactive Curve Editor

                Create and edit calibration curves by adjusting control points numerically
                or using preset adjustments. Changes update the curve visualization in real-time.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Control Points")
                    gr.Markdown("*Adjust input/output pairs (0-1 range)*")

                    # Create 9 editable control points
                    ie_point_inputs = []
                    ie_point_outputs = []

                    with gr.Accordion("Control Points (0-1)", open=True):
                        for i in range(9):
                            with gr.Row():
                                inp = gr.Number(
                                    label=f"In {i + 1}",
                                    value=i / 8.0,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    scale=1,
                                )
                                out = gr.Number(
                                    label=f"Out {i + 1}",
                                    value=i / 8.0,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    scale=1,
                                )
                                ie_point_inputs.append(inp)
                                ie_point_outputs.append(out)

                    update_ie_curve_btn = gr.Button("Update Curve", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("#### Presets")

                    ie_preset_select = gr.Dropdown(
                        choices=[
                            ("Linear (No Change)", "linear"),
                            ("S-Curve (Contrast)", "s_curve"),
                            ("Brighten Highlights", "brighten"),
                            ("Darken Shadows", "darken"),
                            ("High Contrast", "high_contrast"),
                            ("Low Contrast", "low_contrast"),
                            ("Gamma 1.8", "gamma_18"),
                            ("Gamma 2.2", "gamma_22"),
                        ],
                        label="Load Preset",
                    )
                    apply_preset_btn = gr.Button("Apply Preset")

                    gr.Markdown("---")
                    gr.Markdown("#### Quick Adjustments")

                    ie_gamma_slider = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Gamma",
                    )
                    apply_gamma_btn = gr.Button("Apply Gamma")

                    ie_curve_name = gr.Textbox(
                        label="Curve Name",
                        value="Custom Curve",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("#### Curve Visualization")
                    ie_curve_plot = gr.Plot(label="Interactive Curve")

                    ie_curve_info = gr.JSON(label="Curve Data")

                    gr.Markdown("---")
                    gr.Markdown("#### Export")

                    ie_export_format = gr.Dropdown(
                        choices=["qtr", "csv", "json"],
                        value="qtr",
                        label="Export Format",
                    )
                    export_ie_btn = gr.Button("Export Curve")
                    ie_export_file = gr.File(label="Download")

            def update_ie_curve_from_points(*args):
                """Update curve from control point values."""
                import matplotlib.pyplot as plt

                # First 9 args are inputs, next 9 are outputs
                inputs = list(args[:9])
                outputs = list(args[9:18])

                # Filter out None values and create valid points
                points = []
                for inp, out in zip(inputs, outputs):
                    if inp is not None and out is not None:
                        points.append((float(inp), float(out)))

                # Sort by input value
                points.sort(key=lambda x: x[0])

                if len(points) < 2:
                    return None, {"error": "Need at least 2 points"}

                # Create curve data
                curve = CurveData(
                    name="Interactive Curve",
                    input_values=[p[0] for p in points],
                    output_values=[p[1] for p in points],
                )

                # Create plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(
                    curve.input_values,
                    curve.output_values,
                    "o-",
                    color="#8B4513",
                    linewidth=2,
                    markersize=8,
                    label="Curve",
                )
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear")
                ax.set_xlabel("Input")
                ax.set_ylabel("Output")
                ax.set_title("Interactive Curve Editor")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_facecolor("#FAF8F5")
                fig.patch.set_facecolor("#FAF8F5")

                info = {
                    "points": len(points),
                    "input_range": f"{min(curve.input_values):.3f} - {max(curve.input_values):.3f}",
                    "output_range": f"{min(curve.output_values):.3f} - {max(curve.output_values):.3f}",
                }

                return fig, info

            def apply_ie_preset(preset):
                """Apply a preset to the curve."""
                presets = {
                    "linear": [(i / 8, i / 8) for i in range(9)],
                    "s_curve": [
                        (0, 0),
                        (0.125, 0.08),
                        (0.25, 0.18),
                        (0.375, 0.35),
                        (0.5, 0.5),
                        (0.625, 0.65),
                        (0.75, 0.82),
                        (0.875, 0.92),
                        (1, 1),
                    ],
                    "brighten": [(i / 8, min(1, (i / 8) * 1.2 + 0.05)) for i in range(9)],
                    "darken": [(i / 8, max(0, (i / 8) * 0.8)) for i in range(9)],
                    "high_contrast": [
                        (0, 0),
                        (0.125, 0.03),
                        (0.25, 0.1),
                        (0.375, 0.25),
                        (0.5, 0.5),
                        (0.625, 0.75),
                        (0.75, 0.9),
                        (0.875, 0.97),
                        (1, 1),
                    ],
                    "low_contrast": [
                        (0, 0.1),
                        (0.125, 0.175),
                        (0.25, 0.275),
                        (0.375, 0.375),
                        (0.5, 0.5),
                        (0.625, 0.625),
                        (0.75, 0.725),
                        (0.875, 0.825),
                        (1, 0.9),
                    ],
                    "gamma_18": [(i / 8, (i / 8) ** (1 / 1.8)) for i in range(9)],
                    "gamma_22": [(i / 8, (i / 8) ** (1 / 2.2)) for i in range(9)],
                }

                points = presets.get(preset, presets["linear"])

                # Return values for all 9 input/output pairs
                results = []
                for i in range(9):
                    if i < len(points):
                        results.append(points[i][0])  # input
                    else:
                        results.append(i / 8.0)

                for i in range(9):
                    if i < len(points):
                        results.append(points[i][1])  # output
                    else:
                        results.append(i / 8.0)

                return results

            def apply_gamma_to_curve(gamma, *current_values):
                """Apply gamma adjustment to current curve."""
                outputs = list(current_values[9:18])

                # Apply gamma to outputs
                new_outputs = []
                for i, out in enumerate(outputs):
                    if out is not None:
                        inp = i / 8.0
                        # Apply gamma: output = input^(1/gamma)
                        new_out = inp ** (1 / gamma) if gamma > 0 else inp
                        new_outputs.append(min(1.0, max(0.0, new_out)))
                    else:
                        new_outputs.append(i / 8.0)

                return new_outputs

            def export_ie_curve(export_format, name, *point_values):
                """Export the interactive curve."""
                try:
                    import tempfile

                    inputs = list(point_values[:9])
                    outputs = list(point_values[9:18])

                    points = []
                    for inp, out in zip(inputs, outputs):
                        if inp is not None and out is not None:
                            points.append((float(inp), float(out)))

                    points.sort(key=lambda x: x[0])

                    curve = CurveData(
                        name=name or "Interactive Curve",
                        input_values=[p[0] for p in points],
                        output_values=[p[1] for p in points],
                    )

                    ext_map = {"qtr": ".quad", "csv": ".csv", "json": ".json"}
                    ext = ext_map.get(export_format, ".quad")

                    safe_name = "".join(c for c in curve.name if c.isalnum() or c in " -_")[:30]
                    temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                    save_curve(curve, temp_path, format=export_format)

                    return str(temp_path)
                except Exception:
                    return None

            # Connect handlers
            all_point_components = ie_point_inputs + ie_point_outputs

            update_ie_curve_btn.click(
                update_ie_curve_from_points,
                inputs=all_point_components,
                outputs=[ie_curve_plot, ie_curve_info],
            )

            apply_preset_btn.click(
                apply_ie_preset,
                inputs=[ie_preset_select],
                outputs=ie_point_inputs + ie_point_outputs,
            ).then(
                update_ie_curve_from_points,
                inputs=all_point_components,
                outputs=[ie_curve_plot, ie_curve_info],
            )

            apply_gamma_btn.click(
                apply_gamma_to_curve,
                inputs=[ie_gamma_slider] + all_point_components,
                outputs=ie_point_outputs,
            ).then(
                update_ie_curve_from_points,
                inputs=all_point_components,
                outputs=[ie_curve_plot, ie_curve_info],
            )

            export_ie_btn.click(
                export_ie_curve,
                inputs=[ie_export_format, ie_curve_name] + all_point_components,
                outputs=[ie_export_file],
            )

    def build_settings_tab():
        # ========================================
        with gr.TabItem("Settings"):
            gr.Markdown(
                """
                ### Application Settings

                Configure your API keys and preferences for the Pt/Pd Calibration Studio.
                Settings entered here are stored for your current session only.
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### LLM API Configuration")
                    gr.Markdown(
                        """
                        Enter your API key to enable AI-powered features:
                        - AI Curve Enhancement
                        - AI Assistant Chat
                        - Recipe Suggestions
                        - Troubleshooting

                        Your API key is stored locally and never shared.
                        """
                    )

                    llm_provider_select = gr.Dropdown(
                        choices=[
                            ("Anthropic (Claude)", "anthropic"),
                            ("OpenAI (GPT)", "openai"),
                        ],
                        value=settings.llm.provider.value,
                        label="LLM Provider",
                    )

                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key (e.g., sk-ant-... or sk-...)",
                        type="password",
                        value="",
                    )

                    save_api_key_btn = gr.Button("Save API Key", variant="primary")

                    api_key_status = gr.Textbox(
                        label="Status",
                        value="No API key configured"
                        if not settings.llm.get_active_api_key()
                        else "API key configured (from environment)",
                        interactive=False,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Model Selection")

                    anthropic_model_input = gr.Dropdown(
                        choices=[
                            "claude-sonnet-4-20250514",
                            "claude-3-5-sonnet-20241022",
                            "claude-3-haiku-20240307",
                        ],
                        value=settings.llm.anthropic_model,
                        label="Anthropic Model",
                    )

                    openai_model_input = gr.Dropdown(
                        choices=[
                            "gpt-4o",
                            "gpt-4o-mini",
                            "gpt-4-turbo",
                        ],
                        value=settings.llm.openai_model,
                        label="OpenAI Model",
                    )

                with gr.Column():
                    gr.Markdown("#### Getting API Keys")
                    gr.Markdown(
                        """
                        **Anthropic (Claude)**
                        1. Go to [console.anthropic.com](https://console.anthropic.com)
                        2. Sign up or log in
                        3. Navigate to API Keys
                        4. Create a new key

                        **OpenAI (GPT)**
                        1. Go to [platform.openai.com](https://platform.openai.com)
                        2. Sign up or log in
                        3. Navigate to API Keys
                        4. Create a new key

                        ---

                        #### Environment Variables

                        You can also set API keys via environment variables:
                        ```
                        PTPD_LLM_API_KEY=your-key
                        PTPD_LLM_ANTHROPIC_API_KEY=your-anthropic-key
                        PTPD_LLM_OPENAI_API_KEY=your-openai-key
                        ```

                        ---

                        #### Current Configuration

                        These settings are derived from environment variables
                        and can be overridden in the Settings tab.
                        """
                    )

                    current_config_display = gr.JSON(
                        label="Current LLM Configuration",
                        value={
                            "provider": settings.llm.provider.value,
                            "anthropic_model": settings.llm.anthropic_model,
                            "openai_model": settings.llm.openai_model,
                            "api_key_configured": bool(settings.llm.get_active_api_key()),
                            "max_tokens": settings.llm.max_tokens,
                            "temperature": settings.llm.temperature,
                        },
                    )

            def save_api_key(provider, api_key, anthropic_model, openai_model):
                """Save API key to runtime settings."""
                try:
                    if not api_key or not api_key.strip():
                        return "No API key provided", gr.update()

                    # Update the global settings
                    from ptpd_calibration.config import LLMProvider, get_settings

                    current_settings = get_settings()
                    current_settings.llm.runtime_api_key = api_key.strip()
                    current_settings.llm.provider = LLMProvider(provider)
                    current_settings.llm.anthropic_model = anthropic_model
                    current_settings.llm.openai_model = openai_model

                    # Verify the key works (basic format check)
                    key = api_key.strip()
                    if provider == "anthropic" and not key.startswith("sk-ant-"):
                        return (
                            "Warning: Anthropic keys typically start with 'sk-ant-'. Key saved anyway.",
                            {
                                "provider": provider,
                                "anthropic_model": anthropic_model,
                                "openai_model": openai_model,
                                "api_key_configured": True,
                                "max_tokens": current_settings.llm.max_tokens,
                                "temperature": current_settings.llm.temperature,
                            },
                        )
                    elif provider == "openai" and not key.startswith("sk-"):
                        return (
                            "Warning: OpenAI keys typically start with 'sk-'. Key saved anyway.",
                            {
                                "provider": provider,
                                "anthropic_model": anthropic_model,
                                "openai_model": openai_model,
                                "api_key_configured": True,
                                "max_tokens": current_settings.llm.max_tokens,
                                "temperature": current_settings.llm.temperature,
                            },
                        )

                    return f"API key saved successfully for {provider.title()}", {
                        "provider": provider,
                        "anthropic_model": anthropic_model,
                        "openai_model": openai_model,
                        "api_key_configured": True,
                        "max_tokens": current_settings.llm.max_tokens,
                        "temperature": current_settings.llm.temperature,
                    }
                except Exception as e:
                    return f"Error saving API key: {str(e)}", gr.update()

            save_api_key_btn.click(
                save_api_key,
                inputs=[
                    llm_provider_select,
                    api_key_input,
                    anthropic_model_input,
                    openai_model_input,
                ],
                outputs=[api_key_status, current_config_display],
            )

    def build_batch_processing_tab():
        # ========================================
        with gr.TabItem("Batch Processing"):
            gr.Markdown(
                """
                ### Batch Image Processing

                Process multiple images with the same calibration curve.
                Create digital negatives in batch for efficient workflow.
                """
            )

            batch_curve_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Source Files")
                    batch_files_upload = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".tiff", ".tif"],
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Calibration Curve")

                    batch_curve_file = gr.File(
                        label="Upload Curve File (Optional)",
                        file_types=[".quad", ".txt", ".csv", ".json"],
                    )
                    load_batch_curve_btn = gr.Button("Load Curve")
                    batch_curve_info = gr.JSON(label="Loaded Curve")

                    gr.Markdown("---")
                    gr.Markdown("#### Processing Options")

                    batch_invert = gr.Checkbox(
                        label="Invert Images (Create Negatives)",
                        value=True,
                    )
                    batch_grayscale = gr.Checkbox(
                        label="Convert to Grayscale",
                        value=True,
                    )
                    batch_format = gr.Dropdown(
                        choices=[
                            ("TIFF", "tiff"),
                            ("PNG", "png"),
                            ("JPEG", "jpeg"),
                            ("Same as Original", "original"),
                        ],
                        value="tiff",
                        label="Output Format",
                    )

                    process_batch_btn = gr.Button("Process Batch", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Results")
                    batch_progress = gr.Textbox(
                        label="Progress",
                        value="Ready to process",
                        interactive=False,
                    )
                    batch_results = gr.JSON(label="Processing Results")
                    batch_download = gr.File(
                        label="Download Processed Files", file_count="multiple"
                    )

            def load_batch_curve(file):
                """Load curve for batch processing."""
                if file is None:
                    return None, {"status": "No curve loaded"}
                try:
                    file_path = Path(file.name)
                    suffix = file_path.suffix.lower()
                    if suffix in [".quad", ".txt"]:
                        profile = load_quad_file(file_path)
                        curve = profile.to_curve_data("K")
                    else:
                        from ptpd_calibration.curves.export import load_curve

                        curve = load_curve(file_path)
                    return curve, {"name": curve.name, "points": len(curve.input_values)}
                except Exception as e:
                    return None, {"error": str(e)}

            def process_batch(files, curve, invert, grayscale, output_format):
                """Process batch of images."""
                if not files:
                    return "No files to process", {}, None

                try:
                    import os
                    import tempfile

                    processor = ImageProcessor()
                    color_mode = ColorMode.GRAYSCALE if grayscale else ColorMode.PRESERVE

                    results = []
                    output_files = []
                    temp_dir = tempfile.mkdtemp()

                    for _i, file in enumerate(files):
                        try:
                            result = processor.create_digital_negative(
                                file.name,
                                curve=curve,
                                invert=invert,
                                color_mode=color_mode,
                            )

                            # Export
                            base_name = Path(file.name).stem
                            ext_map = {
                                "tiff": ".tiff",
                                "png": ".png",
                                "jpeg": ".jpg",
                                "original": Path(file.name).suffix,
                            }
                            ext = ext_map.get(output_format, ".tiff")
                            out_path = os.path.join(temp_dir, f"{base_name}_processed{ext}")

                            settings = ExportSettings(format=ImageFormat(output_format))
                            processor.export(result, out_path, settings)

                            output_files.append(out_path)
                            results.append({"file": base_name, "status": "success"})
                        except Exception as e:
                            results.append(
                                {"file": Path(file.name).stem, "status": f"error: {str(e)}"}
                            )

                    return (
                        f"Processed {len(output_files)} files",
                        {"results": results},
                        output_files,
                    )

                except Exception as e:
                    return f"Error: {str(e)}", {"error": str(e)}, None

            load_batch_curve_btn.click(
                load_batch_curve,
                inputs=[batch_curve_file],
                outputs=[batch_curve_state, batch_curve_info],
            )

            process_batch_btn.click(
                process_batch,
                inputs=[
                    batch_files_upload,
                    batch_curve_state,
                    batch_invert,
                    batch_grayscale,
                    batch_format,
                ],
                outputs=[batch_progress, batch_results, batch_download],
            )

    def build_histogram_tab():
        # ========================================
        with gr.TabItem("Histogram Analysis"):
            gr.Markdown(
                """
                ### Image Histogram Analysis

                Analyze tonal distribution with zone-based visualization.
                Evaluate dynamic range, contrast, and clipping.
                """
            )

            from ptpd_calibration.imaging import HistogramAnalyzer, HistogramScale

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload Image")
                    hist_image_upload = gr.Image(
                        label="Image to Analyze",
                        type="filepath",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Display Options")

                    hist_scale = gr.Dropdown(
                        choices=[
                            ("Linear", "linear"),
                            ("Logarithmic", "logarithmic"),
                        ],
                        value="linear",
                        label="Histogram Scale",
                    )
                    hist_show_zones = gr.Checkbox(
                        label="Show Zone Boundaries",
                        value=True,
                    )
                    hist_show_rgb = gr.Checkbox(
                        label="Show RGB Channels",
                        value=True,
                    )

                    analyze_hist_btn = gr.Button("Analyze Histogram", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Histogram Visualization")
                    hist_plot = gr.Plot(label="Histogram")

                    hist_stats = gr.JSON(label="Statistics")

                    with gr.Accordion("Zone Reference", open=False):
                        gr.Markdown(
                            """
                            **Ansel Adams Zone System:**
                            - Zone 0: Pure black (no detail)
                            - Zone I-II: Near black with texture
                            - Zone III-IV: Dark shadows
                            - Zone V: Middle gray (18%)
                            - Zone VI-VII: Light tones
                            - Zone VIII-IX: Bright highlights
                            - Zone X: Pure white (paper)
                            """
                        )

            def analyze_histogram(image_path, scale, show_zones, show_rgb):
                """Analyze image histogram."""
                if image_path is None:
                    return None, {"error": "No image uploaded"}

                try:
                    analyzer = HistogramAnalyzer()
                    result = analyzer.analyze(image_path, include_rgb=show_rgb)

                    fig = analyzer.create_histogram_plot(
                        result,
                        scale=HistogramScale(scale),
                        show_zones=show_zones,
                        show_rgb=show_rgb,
                    )

                    return fig, result.to_dict()
                except Exception as e:
                    return None, {"error": str(e)}

            analyze_hist_btn.click(
                analyze_histogram,
                inputs=[hist_image_upload, hist_scale, hist_show_zones, hist_show_rgb],
                outputs=[hist_plot, hist_stats],
            )

    def build_exposure_tab():
        # ========================================
        with gr.TabItem("Exposure Calculator"):
            gr.Markdown(
                """
                ### UV Exposure Calculator

                Calculate exposure times based on negative density, light source, and conditions.
                Uses industry-standard formulas for alternative printing processes.
                """
            )

            from ptpd_calibration.exposure import (
                ExposureCalculator,
                ExposureSettings,
                LightSource,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Reference Exposure")

                    base_exposure = gr.Number(
                        label="Base Exposure (minutes)",
                        value=10.0,
                        minimum=0.5,
                        maximum=60.0,
                    )
                    base_density = gr.Number(
                        label="Base Negative Density",
                        value=1.6,
                        minimum=0.5,
                        maximum=3.0,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Current Negative")

                    current_density = gr.Number(
                        label="Negative Density Range",
                        value=1.6,
                        minimum=0.5,
                        maximum=3.0,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Light Source")

                    light_source_select = gr.Dropdown(
                        choices=[
                            ("NuArc 26-1K Platemaker", "nuarc_26_1k"),
                            ("NuArc FT40", "nuarc_ft40"),
                            ("BL Fluorescent Tubes", "bl_fluorescent"),
                            ("BLB Blacklight Tubes", "blb_fluorescent"),
                            ("UV LED Array", "led_uv"),
                            ("Metal Halide", "metal_halide"),
                            ("Mercury Vapor", "mercury_vapor"),
                            ("Direct Sunlight", "sunlight"),
                        ],
                        value="bl_fluorescent",
                        label="Light Source Type",
                    )

                    distance_inches = gr.Number(
                        label="Distance from Light (inches)",
                        value=4.0,
                        minimum=1.0,
                        maximum=24.0,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Chemistry")

                    exp_platinum_ratio = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=5,
                        label="Platinum % (0 = all Pd)",
                    )

                    calculate_exposure_btn = gr.Button("Calculate Exposure", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Calculated Exposure")

                    exposure_result_display = gr.Textbox(
                        label="Recommended Exposure Time",
                        interactive=False,
                        lines=3,
                    )

                    exposure_details = gr.JSON(label="Exposure Calculation Details")

                    gr.Markdown("---")
                    gr.Markdown("#### Test Strip Generator")

                    test_center_exposure = gr.Number(
                        label="Center Exposure (minutes)",
                        value=10.0,
                    )
                    test_steps = gr.Slider(
                        minimum=3,
                        maximum=9,
                        value=5,
                        step=2,
                        label="Number of Steps",
                    )
                    test_increment = gr.Slider(
                        minimum=0.25,
                        maximum=1.0,
                        value=0.5,
                        step=0.25,
                        label="Increment (stops)",
                    )

                    generate_test_strip_btn = gr.Button("Generate Test Strip Times")
                    test_strip_result = gr.JSON(label="Test Strip Exposures")

            def calculate_exposure(base_exp, base_dens, curr_dens, light, dist, pt_ratio):
                """Calculate exposure time."""
                try:
                    settings = ExposureSettings(
                        base_exposure_minutes=float(base_exp),
                        base_negative_density=float(base_dens),
                        light_source=LightSource(light),
                        base_distance_inches=4.0,
                        platinum_ratio=pt_ratio / 100.0,
                    )
                    calc = ExposureCalculator(settings)
                    result = calc.calculate(
                        negative_density=float(curr_dens),
                        distance_inches=float(dist),
                    )

                    display = f"Exposure Time: {result.format_time()}\n\n"
                    display += f"Density adjustment: {result.density_adjustment:.2f}x\n"
                    display += f"Light source: {result.light_source_adjustment:.2f}x\n"
                    display += f"Distance: {result.distance_adjustment:.2f}x"

                    return display, result.to_dict()
                except Exception as e:
                    return f"Error: {str(e)}", {"error": str(e)}

            def generate_test_strip(center, steps, increment):
                """Generate test strip times."""
                try:
                    calc = ExposureCalculator()
                    times = calc.calculate_test_strip(
                        center_exposure=float(center),
                        steps=int(steps),
                        increment_stops=float(increment),
                    )

                    result = {
                        "center_exposure": center,
                        "increment_stops": increment,
                        "times_minutes": [round(t, 2) for t in times],
                        "times_formatted": [f"{int(t)}:{int((t % 1) * 60):02d}" for t in times],
                    }
                    return result
                except Exception as e:
                    return {"error": str(e)}

            calculate_exposure_btn.click(
                calculate_exposure,
                inputs=[
                    base_exposure,
                    base_density,
                    current_density,
                    light_source_select,
                    distance_inches,
                    exp_platinum_ratio,
                ],
                outputs=[exposure_result_display, exposure_details],
            )

            generate_test_strip_btn.click(
                generate_test_strip,
                inputs=[test_center_exposure, test_steps, test_increment],
                outputs=[test_strip_result],
            )

    def build_zone_system_tab():
        # ========================================
        with gr.TabItem("Zone System"):
            gr.Markdown(
                """
                ### Zone System Analysis

                Analyze images using Ansel Adams' Zone System methodology.
                Get exposure and development recommendations.
                """
            )

            from ptpd_calibration.zones import (
                ZONE_DESCRIPTIONS,
                Zone,
                ZoneMapper,
                ZoneMapping,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload Image")
                    zone_image_upload = gr.Image(
                        label="Image to Analyze",
                        type="filepath",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Paper Characteristics")

                    zone_dmax = gr.Number(
                        label="Paper Dmax",
                        value=1.6,
                        minimum=1.0,
                        maximum=2.5,
                    )
                    zone_dmin = gr.Number(
                        label="Paper Dmin",
                        value=0.08,
                        minimum=0.0,
                        maximum=0.3,
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Zone Placement (Optional)")

                    placed_shadow = gr.Dropdown(
                        choices=[("Auto", None)] + [(f"Zone {i}", i) for i in range(0, 5)],
                        value=None,
                        label="Place Shadows On",
                    )
                    placed_highlight = gr.Dropdown(
                        choices=[("Auto", None)] + [(f"Zone {i}", i) for i in range(6, 11)],
                        value=None,
                        label="Place Highlights On",
                    )

                    analyze_zones_btn = gr.Button("Analyze Zones", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Zone Analysis Results")

                    zone_plot = gr.Plot(label="Zone Visualization")
                    zone_analysis_result = gr.JSON(label="Analysis")

                    with gr.Accordion("Zone Reference", open=True):
                        zone_ref_md = "**Zone Descriptions:**\n\n"
                        for z, desc in ZONE_DESCRIPTIONS.items():
                            zone_ref_md += f"- **Zone {z.value}**: {desc}\n"
                        gr.Markdown(zone_ref_md)

                    gr.Markdown("---")
                    gr.Markdown("#### Development Recommendations")
                    dev_recommendations = gr.Textbox(
                        label="Development Adjustment",
                        interactive=False,
                        lines=4,
                    )

            def analyze_zones(image_path, dmax, dmin, shadow_zone, highlight_zone):
                """Analyze image zones."""
                if image_path is None:
                    return None, {"error": "No image uploaded"}, ""

                try:
                    import matplotlib.pyplot as plt
                    from PIL import Image

                    mapping = ZoneMapping(paper_dmax=float(dmax), paper_dmin=float(dmin))
                    mapper = ZoneMapper(mapping)

                    analysis = mapper.analyze_image(
                        Image.open(image_path),
                        placed_shadow=shadow_zone,
                        placed_highlight=highlight_zone,
                    )

                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Zone histogram
                    zones = list(range(11))
                    pcts = [analysis.zone_histogram.get(Zone(z), 0) * 100 for z in zones]

                    colors = [
                        f"#{int(25.5 * z):02x}{int(25.5 * z):02x}{int(25.5 * z):02x}" for z in zones
                    ]
                    ax1.bar(zones, pcts, color=colors, edgecolor="black")
                    ax1.set_xlabel("Zone")
                    ax1.set_ylabel("Percentage")
                    ax1.set_title("Zone Distribution")
                    ax1.set_xticks(zones)
                    ax1.grid(True, alpha=0.3, axis="y")

                    # Zone scale
                    zone_scale = mapper.create_zone_scale(width=330, height=30)
                    ax2.imshow(zone_scale, cmap="gray", aspect="auto")
                    ax2.set_title("Zone Scale Reference")
                    ax2.set_xticks(np.linspace(0, 330, 11))
                    ax2.set_xticklabels([str(i) for i in range(11)])
                    ax2.set_yticks([])

                    plt.tight_layout()

                    # Development recommendation
                    dev_text = f"Development: {analysis.development_adjustment}\n\n"
                    dev_text += "\n".join(analysis.notes)

                    return fig, analysis.to_dict(), dev_text
                except Exception as e:
                    return None, {"error": str(e)}, f"Error: {str(e)}"

            analyze_zones_btn.click(
                analyze_zones,
                inputs=[zone_image_upload, zone_dmax, zone_dmin, placed_shadow, placed_highlight],
                outputs=[zone_plot, zone_analysis_result, dev_recommendations],
            )

    def build_soft_proofing_tab():
        # ========================================
        with gr.TabItem("Soft Proofing"):
            gr.Markdown(
                """
                ### Print Simulation (Soft Proofing)

                Preview how your image will look when printed on different papers.
                Simulates paper white point, Dmax, and metal tones.
                """
            )

            from ptpd_calibration.proofing import (
                PAPER_PRESETS,
                PaperSimulation,
                ProofSettings,
                SoftProofer,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload Image")
                    proof_image_upload = gr.Image(
                        label="Image to Proof",
                        type="filepath",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Paper Selection")

                    paper_preset_select = gr.Dropdown(
                        choices=[
                            ("Arches Platine", "arches_platine"),
                            ("Bergger COT 320", "bergger_cot320"),
                            ("Hahnemuhle Platinum Rag", "hahnemuhle_platinum"),
                            ("Revere Platinum", "revere_platinum"),
                            ("Stonehenge", "stonehenge"),
                            ("Custom", "custom"),
                        ],
                        value="arches_platine",
                        label="Paper Preset",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Custom Settings")

                    proof_dmax = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.6,
                        step=0.05,
                        label="Paper Dmax",
                    )
                    proof_dmin = gr.Slider(
                        minimum=0.0,
                        maximum=0.2,
                        value=0.07,
                        step=0.01,
                        label="Paper Dmin",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Metal Mix")

                    proof_platinum = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=10,
                        label="Platinum % (0 = warm Pd, 100 = cool Pt)",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Simulation Options")

                    proof_texture = gr.Checkbox(
                        label="Simulate Paper Texture",
                        value=False,
                    )
                    proof_brightness = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Viewing Brightness",
                    )

                    generate_proof_btn = gr.Button("Generate Soft Proof", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Soft Proof Preview")

                    with gr.Row():
                        original_proof_image = gr.Image(
                            label="Original",
                            interactive=False,
                        )
                        proofed_image = gr.Image(
                            label="Simulated Print",
                            interactive=False,
                        )

                    proof_info = gr.JSON(label="Proof Settings")

            def on_paper_preset_change(preset):
                """Update settings when paper preset changes."""
                if preset == "custom":
                    return gr.update(), gr.update()

                try:
                    preset_enum = PaperSimulation(preset)
                    if preset_enum in PAPER_PRESETS:
                        data = PAPER_PRESETS[preset_enum]
                        return data["dmax"], data["dmin"]
                except (ValueError, KeyError):
                    # Ignore invalid preset values
                    pass
                return gr.update(), gr.update()

            def generate_soft_proof(image_path, preset, dmax, dmin, pt_ratio, texture, brightness):
                """Generate soft proof."""
                if image_path is None:
                    return None, None, {"error": "No image uploaded"}

                try:
                    from PIL import Image

                    # Load and display original
                    original = Image.open(image_path)
                    original.thumbnail((600, 600), Image.Resampling.LANCZOS)

                    # Create proof settings
                    if preset != "custom":
                        settings = ProofSettings.from_paper_preset(PaperSimulation(preset))
                    else:
                        settings = ProofSettings(
                            paper_dmax=float(dmax),
                            paper_dmin=float(dmin),
                        )

                    settings.platinum_ratio = pt_ratio / 100.0
                    settings.add_paper_texture = texture
                    settings.viewing_brightness = brightness

                    # Generate proof
                    proofer = SoftProofer(settings)
                    result = proofer.proof(Image.open(image_path))

                    return original, result.image, result.to_dict()
                except Exception as e:
                    return None, None, {"error": str(e)}

            paper_preset_select.change(
                on_paper_preset_change,
                inputs=[paper_preset_select],
                outputs=[proof_dmax, proof_dmin],
            )

            generate_proof_btn.click(
                generate_soft_proof,
                inputs=[
                    proof_image_upload,
                    paper_preset_select,
                    proof_dmax,
                    proof_dmin,
                    proof_platinum,
                    proof_texture,
                    proof_brightness,
                ],
                outputs=[original_proof_image, proofed_image, proof_info],
            )

    def build_paper_profiles_tab():
        # ========================================
        with gr.TabItem("Paper Profiles"):
            gr.Markdown(
                """
                ### Paper Profiles Database

                Browse and manage paper profiles with recommended settings.
                """
            )

            from ptpd_calibration.papers import PaperDatabase, PaperProfile

            paper_db = PaperDatabase()

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Browse Papers")

                    paper_list = gr.Dropdown(
                        choices=[(p.name, p.name) for p in paper_db.list_papers()],
                        label="Select Paper",
                    )

                    view_paper_btn = gr.Button("View Profile")

                    gr.Markdown("---")
                    gr.Markdown("#### Add Custom Paper")

                    new_paper_name = gr.Textbox(label="Paper Name")
                    new_paper_mfr = gr.Textbox(label="Manufacturer")
                    new_paper_weight = gr.Number(label="Weight (gsm)", value=300)
                    new_paper_sizing = gr.Dropdown(
                        choices=["internal", "external", "none", "unknown"],
                        value="internal",
                        label="Sizing Type",
                    )
                    new_paper_texture = gr.Dropdown(
                        choices=["smooth", "medium", "rough"],
                        value="medium",
                        label="Texture",
                    )

                    add_paper_btn = gr.Button("Add Paper", variant="secondary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Paper Profile")
                    paper_profile_display = gr.JSON(label="Profile Details")

                    gr.Markdown("---")
                    gr.Markdown("#### Recommended Papers for Pt/Pd")
                    gr.Markdown(
                        """
                        **Top Choices:**
                        - **Arches Platine** - Smooth, internally sized, excellent Dmax
                        - **Bergger COT 320** - Heavy cotton, warm base
                        - **Hahnemuhle Platinum Rag** - Bright white, very smooth
                        - **Stonehenge** - Affordable, good results

                        **Key Properties:**
                        - Internal sizing preferred for even coating
                        - Cotton or rag content for longevity
                        - 250-320 gsm weight for minimal cockling
                        """
                    )

            def view_paper_profile(paper_name):
                """View paper profile details."""
                if not paper_name:
                    return {"error": "No paper selected"}
                try:
                    profile = paper_db.get_by_name(paper_name)
                    if profile:
                        return profile.to_dict()
                    return {"error": "Paper not found"}
                except Exception as e:
                    return {"error": str(e)}

            def add_custom_paper(name, mfr, weight, sizing, texture):
                """Add custom paper to database."""
                if not name:
                    return {"error": "Paper name required"}
                try:
                    from ptpd_calibration.papers.profiles import (
                        PaperCharacteristics,
                        SizingType,
                        TextureType,
                    )

                    profile = PaperProfile(
                        name=name,
                        manufacturer=mfr or "Custom",
                        characteristics=PaperCharacteristics(
                            weight_gsm=int(weight) if weight else 300,
                            sizing=SizingType(sizing),
                            texture=TextureType(texture),
                        ),
                    )
                    paper_db.add_profile(profile)
                    return {"status": "success", "profile": profile.to_dict()}
                except Exception as e:
                    return {"error": str(e)}

            view_paper_btn.click(
                view_paper_profile,
                inputs=[paper_list],
                outputs=[paper_profile_display],
            )

            add_paper_btn.click(
                add_custom_paper,
                inputs=[
                    new_paper_name,
                    new_paper_mfr,
                    new_paper_weight,
                    new_paper_sizing,
                    new_paper_texture,
                ],
                outputs=[paper_profile_display],
            )

    def build_auto_linearization_tab():
        # ========================================
        with gr.TabItem("Auto-Linearization"):
            gr.Markdown(
                """
                ### Auto-Linearization

                Automatically generate linearization curves from step wedge measurements.
                """
            )

            from ptpd_calibration.curves import (
                AutoLinearizer,
                LinearizationMethod,
                TargetResponse,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Input Densities")

                    linearize_densities = gr.Textbox(
                        label="Measured Densities (comma-separated)",
                        placeholder="0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60",
                        lines=3,
                    )

                    linearize_name = gr.Textbox(
                        label="Curve Name",
                        value="Auto-Linearized",
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### Method & Target")

                    linearize_method = gr.Dropdown(
                        choices=[
                            ("Spline Fit (Recommended)", "spline_fit"),
                            ("Direct Inversion", "direct_inversion"),
                            ("Polynomial Fit", "polynomial_fit"),
                            ("Iterative Refinement", "iterative"),
                            ("Hybrid", "hybrid"),
                        ],
                        value="spline_fit",
                        label="Linearization Method",
                    )

                    linearize_target = gr.Dropdown(
                        choices=[
                            ("Linear (Gamma 1.0)", "linear"),
                            ("Gamma 1.8", "gamma_18"),
                            ("Gamma 2.2 (sRGB)", "gamma_22"),
                            ("Paper White Preserve", "paper_white"),
                            ("Perceptual (L*)", "perceptual"),
                        ],
                        value="linear",
                        label="Target Response",
                    )

                    linearize_smoothing = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                        label="Smoothing Factor",
                    )

                    run_linearization_btn = gr.Button("Generate Curve", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("#### Linearization Results")

                    linearize_plot = gr.Plot(label="Linearization Curve")
                    linearize_result = gr.JSON(label="Results")

                    gr.Markdown("---")
                    linearize_export_format = gr.Dropdown(
                        choices=["qtr", "csv", "json"],
                        value="qtr",
                        label="Export Format",
                    )
                    export_linearize_btn = gr.Button("Export Curve")
                    linearize_export_file = gr.File(label="Download")

            linearize_curve_state = gr.State(None)

            def run_linearization(densities_str, name, method, target, smoothing):
                """Run auto-linearization."""
                if not densities_str.strip():
                    return None, None, {"error": "No densities provided"}

                try:
                    import matplotlib.pyplot as plt

                    densities = [float(d.strip()) for d in densities_str.split(",")]

                    from ptpd_calibration.curves.linearization import LinearizationConfig

                    config = LinearizationConfig(
                        method=LinearizationMethod(method),
                        target=TargetResponse(target),
                        smoothing=smoothing,
                    )

                    linearizer = AutoLinearizer(config)
                    result = linearizer.linearize(
                        densities,
                        curve_name=name,
                        target=TargetResponse(target),
                        method=LinearizationMethod(method),
                    )

                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.plot(
                        result.curve.input_values,
                        result.curve.output_values,
                        "-",
                        color="#8B4513",
                        linewidth=2,
                        label="Linearization Curve",
                    )
                    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear Reference")

                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.set_title(f"Auto-Linearization: {name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    return result.curve, fig, result.to_dict()
                except Exception as e:
                    import traceback

                    return None, None, {"error": str(e), "traceback": traceback.format_exc()}

            def export_linearization_curve(curve, export_format):
                """Export linearization curve."""
                if curve is None:
                    return None

                try:
                    import tempfile

                    ext_map = {"qtr": ".quad", "csv": ".csv", "json": ".json"}
                    ext = ext_map.get(export_format, ".quad")

                    safe_name = "".join(c for c in curve.name if c.isalnum() or c in " -_")[:30]
                    temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                    save_curve(curve, temp_path, format=export_format)
                    return str(temp_path)
                except Exception:
                    return None

            run_linearization_btn.click(
                run_linearization,
                inputs=[
                    linearize_densities,
                    linearize_name,
                    linearize_method,
                    linearize_target,
                    linearize_smoothing,
                ],
                outputs=[linearize_curve_state, linearize_plot, linearize_result],
            )

            export_linearize_btn.click(
                export_linearization_curve,
                inputs=[linearize_curve_state, linearize_export_format],
                outputs=[linearize_export_file],
            )

    def build_scanner_calibration_tab():
        # ========================================
        with gr.TabItem("Scanner Calibration"):
            gr.Markdown(
                """
                ### Scanner Calibration

                Create and manage scanner profiles for accurate density measurements.
                Calibrate your flatbed scanner using white/black point samples or reference targets.
                """
            )

            from ptpd_calibration.detection.scanner import ScannerCalibration

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Simple Calibration")
                    gr.Markdown("*Calibrate using measured white and black point values*")

                    profile_name = gr.Textbox(
                        label="Profile Name",
                        value="My Scanner Profile",
                        placeholder="Enter a name for this profile",
                    )

                    scanner_model = gr.Textbox(
                        label="Scanner Model (optional)",
                        placeholder="e.g., Epson V850",
                    )

                    gr.Markdown("---")
                    gr.Markdown("##### White Point Sample (RGB)")

                    white_r = gr.Number(label="Red", value=245, minimum=0, maximum=255)
                    white_g = gr.Number(label="Green", value=243, minimum=0, maximum=255)
                    white_b = gr.Number(label="Blue", value=240, minimum=0, maximum=255)

                    gr.Markdown("##### Black Point Sample (RGB)")

                    black_r = gr.Number(label="Red", value=12, minimum=0, maximum=255)
                    black_g = gr.Number(label="Green", value=10, minimum=0, maximum=255)
                    black_b = gr.Number(label="Blue", value=8, minimum=0, maximum=255)

                    calibrate_simple_btn = gr.Button("Create Profile", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("#### Target-Based Calibration")
                    gr.Markdown("*Upload a scan of a calibration target for full profiling*")

                    target_image = gr.Image(
                        label="Calibration Target Scan",
                        type="numpy",
                    )

                    target_profile_name = gr.Textbox(
                        label="Profile Name",
                        value="Target Profile",
                    )

                    calibrate_target_btn = gr.Button("Analyze Target", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown("#### Load Existing Profile")

                    profile_file = gr.File(
                        label="Upload Profile (.json)",
                        file_types=[".json"],
                    )

                    load_profile_btn = gr.Button("Load Profile")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Current Profile")

                    profile_status = gr.Textbox(
                        label="Profile Status",
                        value="No profile loaded",
                        interactive=False,
                        lines=4,
                    )

                    profile_curves_plot = gr.Plot(label="Response Curves")

                with gr.Column():
                    gr.Markdown("#### Apply Correction")

                    test_image = gr.Image(
                        label="Test Image",
                        type="numpy",
                    )

                    apply_correction_btn = gr.Button("Apply Correction")

                    corrected_image = gr.Image(
                        label="Corrected Image",
                        type="numpy",
                    )

            with gr.Row():
                save_profile_btn = gr.Button("Save Profile")
                profile_download = gr.File(label="Download Profile")

            # State for current calibration
            scanner_cal_state = gr.State(None)

            def create_simple_profile(name, model, wr, wg, wb, br, bg, bb):
                """Create profile from white/black point samples."""
                try:
                    import matplotlib.pyplot as plt

                    cal = ScannerCalibration()
                    profile = cal.calibrate_simple(
                        white_sample=(float(wr), float(wg), float(wb)),
                        black_sample=(float(br), float(bg), float(bb)),
                        name=name,
                    )

                    if model:
                        profile.scanner_model = model

                    # Create response curves plot
                    fig, ax = plt.subplots(figsize=(8, 5))

                    x = list(range(256))
                    ax.plot(x, profile.red_curve.output_values, "r-", label="Red", linewidth=1.5)
                    ax.plot(
                        x, profile.green_curve.output_values, "g-", label="Green", linewidth=1.5
                    )
                    ax.plot(x, profile.blue_curve.output_values, "b-", label="Blue", linewidth=1.5)
                    ax.plot([0, 255], [0, 255], "k--", alpha=0.3, label="Linear")

                    ax.set_xlabel("Input Value")
                    ax.set_ylabel("Output Value")
                    ax.set_title(f"Scanner Response Curves: {name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 255)
                    ax.set_ylim(0, 255)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    status = f"Profile: {profile.name}\n"
                    status += f"Created: {profile.created_at}\n"
                    if profile.scanner_model:
                        status += f"Scanner: {profile.scanner_model}\n"
                    status += "Type: Simple (white/black point)"

                    return cal, status, fig
                except Exception as e:
                    return None, f"Error: {str(e)}", None

            def calibrate_from_target(image, name):
                """Create profile from calibration target scan."""
                if image is None:
                    return None, "No image provided", None

                try:
                    import matplotlib.pyplot as plt

                    cal = ScannerCalibration()
                    # For full implementation, reference_values would come from target type
                    profile = cal.calibrate_from_target(
                        target_scan=image,
                        reference_values={},  # Placeholder
                        name=name,
                    )

                    # Create uniformity visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    # Response curves
                    x = list(range(256))
                    axes[0].plot(x, profile.red_curve.output_values, "r-", label="Red")
                    axes[0].plot(x, profile.green_curve.output_values, "g-", label="Green")
                    axes[0].plot(x, profile.blue_curve.output_values, "b-", label="Blue")
                    axes[0].set_title("Response Curves")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    # Uniformity map
                    if profile.uniformity_map is not None:
                        im = axes[1].imshow(
                            profile.uniformity_map, cmap="RdYlGn", vmin=0.9, vmax=1.1
                        )
                        axes[1].set_title("Field Uniformity")
                        plt.colorbar(im, ax=axes[1])
                    else:
                        axes[1].text(0.5, 0.5, "No uniformity data", ha="center", va="center")

                    fig.tight_layout()

                    status = f"Profile: {profile.name}\n"
                    status += f"Created: {profile.created_at}\n"
                    status += "Type: Target-based calibration\n"
                    if profile.uniformity_map is not None:
                        status += f"Uniformity map size: {profile.uniformity_map_size}"

                    return cal, status, fig
                except Exception as e:
                    import traceback

                    return None, f"Error: {str(e)}\n{traceback.format_exc()}", None

            def load_profile(file):
                """Load profile from JSON file."""
                if file is None:
                    return None, "No file selected", None

                try:
                    from pathlib import Path

                    import matplotlib.pyplot as plt

                    cal = ScannerCalibration.load(Path(file.name))
                    profile = cal.profile

                    # Create plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = list(range(256))
                    ax.plot(x, profile.red_curve.output_values, "r-", label="Red")
                    ax.plot(x, profile.green_curve.output_values, "g-", label="Green")
                    ax.plot(x, profile.blue_curve.output_values, "b-", label="Blue")
                    ax.plot([0, 255], [0, 255], "k--", alpha=0.3, label="Linear")
                    ax.set_xlabel("Input Value")
                    ax.set_ylabel("Output Value")
                    ax.set_title(f"Loaded Profile: {profile.name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_facecolor("#FAF8F5")
                    fig.patch.set_facecolor("#FAF8F5")

                    status = f"Profile: {profile.name}\n"
                    status += f"Created: {profile.created_at}\n"
                    if profile.scanner_model:
                        status += f"Scanner: {profile.scanner_model}\n"
                    status += "Loaded from file"

                    return cal, status, fig
                except Exception as e:
                    return None, f"Error loading profile: {str(e)}", None

            def apply_correction(cal, image):
                """Apply scanner correction to image."""
                if cal is None:
                    return None
                if image is None:
                    return None

                try:
                    corrected = cal.apply_correction(image)
                    return corrected
                except Exception:
                    return None

            def save_profile(cal):
                """Save current profile to file."""
                if cal is None:
                    return None

                try:
                    import tempfile
                    from pathlib import Path

                    safe_name = "".join(c for c in cal.profile.name if c.isalnum() or c in " -_")[
                        :30
                    ]
                    temp_path = Path(tempfile.gettempdir()) / f"{safe_name}_scanner_profile.json"
                    cal.save(temp_path)
                    return str(temp_path)
                except Exception:
                    return None

            calibrate_simple_btn.click(
                create_simple_profile,
                inputs=[
                    profile_name,
                    scanner_model,
                    white_r,
                    white_g,
                    white_b,
                    black_r,
                    black_g,
                    black_b,
                ],
                outputs=[scanner_cal_state, profile_status, profile_curves_plot],
            )

            calibrate_target_btn.click(
                calibrate_from_target,
                inputs=[target_image, target_profile_name],
                outputs=[scanner_cal_state, profile_status, profile_curves_plot],
            )

            load_profile_btn.click(
                load_profile,
                inputs=[profile_file],
                outputs=[scanner_cal_state, profile_status, profile_curves_plot],
            )

            apply_correction_btn.click(
                apply_correction,
                inputs=[scanner_cal_state, test_image],
                outputs=[corrected_image],
            )

            save_profile_btn.click(
                save_profile,
                inputs=[scanner_cal_state],
                outputs=[profile_download],
            )

    def build_about_tab(tab_label: str = "About"):
        # ========================================
        with gr.TabItem(tab_label):
            gr.Markdown(
                """
                ## Pt/Pd Calibration Studio

                An AI-powered calibration system for platinum/palladium printing.

                ### Core Features

                - **Curve Display**: Upload and compare multiple curves with comprehensive statistics
                - **Step Wedge Analysis**: Automatic step wedge detection, density extraction, and quality assessment
                - **Step Tablet Analysis**: Upload scans and extract density measurements
                - **Curve Generation**: Create linearization curves for digital negatives
                - **Curve Editor**: Upload .quad files, modify curves, smooth curves, and apply AI-powered enhancements
                - **AI Assistant**: Get help from an AI expert in Pt/Pd printing

                ### Image Processing

                - **Image Preview**: Upload an image, select a curve, and preview the output
                - **Digital Negative**: Create inverted negatives with curves applied
                - **Interactive Editor**: Create curves with numeric control points and presets
                - **Batch Processing**: Process multiple images with the same curve
                - **Histogram Analysis**: Analyze tonal distribution with zone-based visualization

                ### Printing Tools

                - **Chemistry Calculator**: Calculate coating solution amounts for any print size
                - **Exposure Calculator**: Calculate UV exposure times based on density and light source
                - **Zone System**: Analyze images using Ansel Adams' Zone System methodology
                - **Soft Proofing**: Preview how prints will look on different papers
                - **Paper Profiles**: Browse and manage paper profiles database

                ### Advanced Features

                - **Auto-Linearization**: Automatically generate linearization curves from measurements
                - **Print Session Log**: Track prints and build process knowledge over time
                - **Recipe Suggestions**: Get starting parameters for new papers
                - **Troubleshooting**: Diagnose and fix common problems

                ### Supported Formats

                - QuadTone RIP (.txt, .quad)
                - Piezography (.ppt)
                - CSV
                - JSON

                ### Image Export Formats

                - TIFF (8-bit and 16-bit)
                - PNG (8-bit and 16-bit)
                - JPEG (standard and high quality)
                - Original format preservation

                ### Chemistry Reference

                Based on [Bostick-Sullivan Platinum/Palladium Kit Instructions](https://www.bostick-sullivan.com/wp-content/uploads/2022/03/platinum-and-palladium-kit-instructions.pdf):
                - Standard formula: FO + Metal drops (1:1 ratio)
                - ~46 drops covers 8x10" coating area
                - Na2 adds contrast (typically 25% of metal drops)

                ### Links

                - [GitHub Repository](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool)
                - [Documentation](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool#readme)
                - [Bostick & Sullivan](https://www.bostick-sullivan.com/)
                """
            )

    # Create the interface
    with gr.Blocks(
        title="Pt/Pd Calibration Studio",
        theme=theme,
        analytics_enabled=False,
        css=custom_css,
    ) as app:
        onboarding_state = gr.State(
            {
                "experience": "Some experience, new to digital negatives",
                "paper": "Arches Platine",
            }
        )
        gr.HTML(
            "<script>document.documentElement.setAttribute('data-ptpd-theme','darkroom');</script>",
            visible=False,
        )
        gr.HTML(f"<script>{keyboard_js}</script>", visible=False)

        with gr.Row(elem_classes=["top-bar"]):
            gr.Markdown("### Pt/Pd Calibration Studio")
            theme_toggle = gr.Radio(
                ["🌙 Darkroom", "☀️ Light", "🖨️ Print"],
                label="Theme",
                value="🌙 Darkroom",
            )
        theme_update = gr.HTML("", visible=False)

        def apply_theme(mode: str):
            mapping = {
                "🌙 Darkroom": "darkroom",
                "☀️ Light": "light",
                "🖨️ Print": "print",
            }
            key = mapping.get(mode, "darkroom")
            return f"<script>document.documentElement.setAttribute('data-ptpd-theme','{key}');</script>"

        theme_toggle.change(
            apply_theme,
            inputs=[theme_toggle],
            outputs=[theme_update],
        )

        with gr.Tabs(elem_id="main-tabs"):
            # 1. Dashboard (New)
            build_dashboard_new(onboarding_state, session_logger)

            # 2. Calibration
            with gr.TabItem("📊 Calibration"), gr.Tabs(elem_id="calibration-tabs"):
                build_wizard_new()
                build_step_tablet_reader_tab()
                build_step_wedge_tab()
                build_curve_display_tab()
                build_generate_curve_tab()
                build_curve_editor_tab()
                build_auto_linearization_tab()

            # 3. Image Prep
            with gr.TabItem("🎨 Image Prep"), gr.Tabs(elem_id="image-tabs"):
                build_image_preview_tab()
                build_digital_negative_tab()
                build_interactive_editor_tab()
                build_batch_processing_tab()
                build_histogram_tab()
                build_zone_system_tab()
                build_soft_proofing_tab()

            # 4. Darkroom
            with gr.TabItem("🧪 Darkroom"), gr.Tabs(elem_id="darkroom-tabs"):
                build_chemistry_new()
                build_exposure_tab()
                build_paper_profiles_tab()
                build_session_log_new(session_logger)
                build_settings_tab()

            # 5. AI Tools
            with gr.TabItem("🤖 AI Tools"), gr.Tabs(elem_id="ai-tabs"):
                build_ai_new()
                build_neural_new(session_logger)
                build_quick_tools_tab()

            build_about_tab(tab_label="ℹ️ About")

    return app


def launch_ui(share: bool = True, port: int = 7860, server_name: str = "0.0.0.0"):
    """
    Launch the Gradio UI.

    Args:
        share: Whether to create a public share link.
        port: Port to run on.
        server_name: Server name to bind to.
    """
    app = create_gradio_app(share=share)
    app.launch(share=share, server_port=port, server_name=server_name, show_api=False)


if __name__ == "__main__":
    import os

    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_ui(port=port)
