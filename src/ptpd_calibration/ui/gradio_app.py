"""
Gradio-based user interface for PTPD Calibration System.

Provides comprehensive curve display, step wedge analysis, and calibration tools.
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

    from ptpd_calibration.config import TabletType, get_settings
    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.core.models import CurveData
    from ptpd_calibration.curves import (
        CurveAnalyzer,
        CurveGenerator,
        load_quad_file,
        load_quad_string,
        CurveModifier,
        SmoothingMethod,
        CurveAIEnhancer,
        EnhancementGoal,
        save_curve,
        CurveVisualizer,
        VisualizationConfig,
        PlotStyle,
        ColorScheme,
    )
    from ptpd_calibration.detection import StepTabletReader
    from ptpd_calibration.analysis import (
        StepWedgeAnalyzer,
        WedgeAnalysisConfig,
        QualityGrade,
    )
    from ptpd_calibration.chemistry import (
        ChemistryCalculator,
        ChemistryRecipe,
        PaperAbsorbency,
        CoatingMethod,
        MetalMix,
        METAL_MIX_RATIOS,
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

    # Create the interface
    with gr.Blocks(
        title="Pt/Pd Calibration Studio",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Pt/Pd Calibration Studio

            AI-powered calibration system for platinum/palladium printing.
            """
        )

        with gr.Tabs():
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
                            comparison_output = gr.JSON(label="Comparison Metrics", visible=False)

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
                                if profile.primary_channel:
                                    curve = profile.to_curve_data("K")
                                    curves.append(curve)
                                    names.append(f"{profile.profile_name} (K)")
                            elif suffix == ".json":
                                from ptpd_calibration.curves.export import load_curve
                                curve = load_curve(file_path)
                                curves.append(curve)
                                names.append(curve.name)
                            elif suffix == ".csv":
                                from ptpd_calibration.curves.export import load_curve
                                curve = load_curve(file_path)
                                curves.append(curve)
                                names.append(curve.name)
                        except Exception as e:
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

                def on_display_options_change(curves, names, style, scheme, show_ref, show_stats, show_diff):
                    """Handle display option changes."""
                    if not curves:
                        return [], None, {}
                    return update_curve_display(curves, names, style, scheme, show_ref, show_stats, show_diff)

                # Connect event handlers
                load_files_btn.click(
                    load_curve_files,
                    inputs=[curve_file_upload, loaded_curves, curve_names_list],
                    outputs=[loaded_curves, curve_names_list, curves_list_display, curve_display_plot, stats_output],
                )

                add_pasted_btn.click(
                    add_pasted_curve,
                    inputs=[paste_curve_data, paste_curve_name, loaded_curves, curve_names_list],
                    outputs=[loaded_curves, curve_names_list, curves_list_display, curve_display_plot, stats_output],
                )

                clear_curves_btn.click(
                    clear_all_curves,
                    outputs=[loaded_curves, curve_names_list, curves_list_display, curve_display_plot, stats_output],
                )

                # Display option change handlers
                for component in [plot_style, color_scheme, show_reference, show_statistics, show_difference]:
                    component.change(
                        on_display_options_change,
                        inputs=[loaded_curves, curve_names_list, plot_style, color_scheme, show_reference, show_statistics, show_difference],
                        outputs=[curves_list_display, curve_display_plot, stats_output],
                    )

            # ========================================
            # TAB 2: Step Wedge Analysis
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

                def analyze_step_wedge(image_path, tablet_type, min_range, fix_reversals, reject_outliers):
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
                            ax.plot(x, result.densities, "o-", color="#B8860B", linewidth=2, markersize=6, label="Measured")

                            if result.raw_densities and result.raw_densities != result.densities:
                                ax.plot(x, result.raw_densities, "x--", color="#808080", alpha=0.5, label="Raw")
                                ax.legend()

                        ax.set_xlabel("Input %")
                        ax.set_ylabel("Density")
                        ax.set_title(f"Step Wedge Response (Dmin: {result.dmin:.3f}, Dmax: {result.dmax:.3f})")
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor("#FAF8F5")
                        fig.patch.set_facecolor("#FAF8F5")

                        # Quality metrics
                        quality_metrics = result.quality.to_dict() if result.quality else {}

                        # Warnings
                        warnings_text = ""
                        if result.quality and result.quality.warnings:
                            warnings_text = "\n".join([
                                f"[{w.level.value.upper()}] {w.message}"
                                for w in result.quality.warnings
                            ])

                        # Recommendations
                        recommendations_text = ""
                        if result.quality and result.quality.recommendations:
                            recommendations_text = "\n".join([
                                f"• {r}" for r in result.quality.recommendations
                            ])

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

                def generate_calibration_curve(
                    result, curve_name, paper_type, chemistry, curve_type
                ):
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

                    except Exception as e:
                        return None, None, gr.update()

                def export_generated_curve(curve, format_type):
                    """Export the generated curve."""
                    if curve is None:
                        return None

                    try:
                        import tempfile

                        ext_map = {
                            "qtr": ".txt",
                            "piezography": ".ppt",
                            "csv": ".csv",
                            "json": ".json",
                        }
                        ext = ext_map.get(format_type, ".txt")

                        safe_name = "".join(c for c in curve.name if c.isalnum() or c in " -_")[:50]
                        temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                        save_curve(curve, temp_path, format=format_type)
                        return str(temp_path)
                    except Exception:
                        return None

                # Connect handlers
                analyze_wedge_btn.click(
                    analyze_step_wedge,
                    inputs=[wedge_image_upload, tablet_type_select, min_density_range, auto_fix_reversals, outlier_rejection],
                    outputs=[analysis_result_state, quality_grade_display, quality_score_display, density_curve_plot, quality_metrics_json, warnings_output, recommendations_output, generate_curve_btn],
                )

                generate_curve_btn.click(
                    generate_calibration_curve,
                    inputs=[analysis_result_state, curve_name_input, paper_type_input, chemistry_input, curve_type_select],
                    outputs=[generated_curve_state, generated_curve_plot, export_curve_btn],
                )

                export_curve_btn.click(
                    export_generated_curve,
                    inputs=[generated_curve_state, export_format_select],
                    outputs=[export_file_output],
                )

            # ========================================
            # TAB 3: Analyze Scan (Original)
            # ========================================
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

            # ========================================
            # TAB 4: Generate Curve
            # ========================================
            with gr.TabItem("Generate Curve"):
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

            # ========================================
            # TAB 5: Curve Editor
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
                                    choices=["K", "C", "M", "Y", "LC", "LM", "LK"],
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

                def create_curve_plot(inputs, outputs, name="Curve", profile_data=None, show_all=False, selected_channel="K"):
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
                                ax.plot(ch_inputs, ch_outputs, "-", color=color,
                                       linewidth=linewidth, alpha=alpha, label=ch_name)
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
                        return [], [], "No Curve", {"error": "No file uploaded"}, None, None, gr.update()

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

                        available_channels = profile.all_channel_names if profile.all_channel_names else ["K"]
                        selected_channel = channel.upper() if channel.upper() in available_channels else available_channels[0]

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
                            inputs, outputs, name,
                            profile_data=profile_data,
                            show_all=show_all,
                            selected_channel=selected_channel
                        )

                        dropdown_update = gr.update(choices=available_channels, value=selected_channel)

                        return inputs, outputs, name, info, fig, profile_data, dropdown_update
                    except Exception as e:
                        import traceback
                        error_detail = f"{str(e)}\n{traceback.format_exc()}"
                        return [], [], "Error", {"error": str(e), "detail": error_detail}, None, None, gr.update()

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

                        available_channels = profile.all_channel_names if profile.all_channel_names else ["K"]
                        selected_channel = channel.upper() if channel.upper() in available_channels else available_channels[0]

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
                            inputs, outputs, name,
                            profile_data=profile_data,
                            show_all=show_all,
                            selected_channel=selected_channel
                        )

                        dropdown_update = gr.update(choices=available_channels, value=selected_channel)

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
                            inputs, outputs, name,
                            profile_data=profile_data,
                            show_all=show_all,
                            selected_channel=selected_channel
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
                        inputs, outputs, name,
                        profile_data=profile_data,
                        show_all=show_all,
                        selected_channel=selected_channel
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

                        safe_name = "".join(c for c in name if c.isalnum() or c in " -_")[:50]
                        temp_path = Path(tempfile.gettempdir()) / f"{safe_name}{ext}"

                        save_curve(curve, temp_path, format=format_type)

                        return str(temp_path)
                    except Exception:
                        return None

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
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
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
                    outputs=[current_curve_inputs, current_curve_outputs, current_curve_name, editor_info, editor_plot],
                )

                show_all_channels.change(
                    on_show_all_toggle,
                    inputs=[show_all_channels, channel_select, current_profile_data, current_curve_inputs, current_curve_outputs, current_curve_name],
                    outputs=[editor_plot],
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

            # ========================================
            # TAB 6: AI Assistant
            # ========================================
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

            # ========================================
            # TAB 7: Quick Tools
            # ========================================
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

            # ========================================
            # TAB 8: Chemistry Calculator
            # ========================================
            with gr.TabItem("Chemistry Calculator"):
                gr.Markdown(
                    """
                    ### Coating Chemistry Calculator

                    Calculate platinum/palladium coating solution amounts based on your print dimensions.
                    Based on [Bostick-Sullivan formulas](https://www.bostick-sullivan.com/product-category/platinum-palladium-printing-process/).
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Print Dimensions")

                        with gr.Row():
                            print_width = gr.Number(
                                label="Width (inches)",
                                value=8.0,
                                minimum=1.0,
                                maximum=40.0,
                            )
                            print_height = gr.Number(
                                label="Height (inches)",
                                value=10.0,
                                minimum=1.0,
                                maximum=40.0,
                            )

                        standard_size_select = gr.Dropdown(
                            choices=["Custom", "4x5", "5x7", "8x10", "11x14", "16x20", "20x24"],
                            value="8x10",
                            label="Or Select Standard Size",
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### Metal Mix")

                        metal_mix_select = gr.Dropdown(
                            choices=[
                                ("Pure Palladium (warm tones)", "pure_palladium"),
                                ("Pure Platinum (cool tones, max Dmax)", "pure_platinum"),
                                ("Classic Mix 50/50", "classic_mix"),
                                ("Warm Mix (25% Pt / 75% Pd)", "warm_mix"),
                                ("Cool Mix (75% Pt / 25% Pd)", "cool_mix"),
                                ("Custom", "custom"),
                            ],
                            value="pure_palladium",
                            label="Metal Mix Preset",
                        )

                        platinum_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=5,
                            label="Platinum % (for custom mix)",
                            visible=False,
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### Paper & Coating")

                        paper_absorbency_select = gr.Dropdown(
                            choices=[
                                ("Low (Hot Press, sized)", "low"),
                                ("Medium (Standard art paper)", "medium"),
                                ("High (Cold Press, unsized)", "high"),
                            ],
                            value="medium",
                            label="Paper Absorbency",
                        )

                        coating_method_select = gr.Dropdown(
                            choices=[
                                ("Brush (Hake)", "brush"),
                                ("Glass Rod", "rod"),
                                ("Puddle Pusher", "puddle_pusher"),
                            ],
                            value="brush",
                            label="Coating Method",
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### Contrast & Na2")

                        contrast_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=10,
                            label="Contrast Boost % (FO#2)",
                        )

                        na2_slider = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=25,
                            step=5,
                            label="Na2 % (of metal drops)",
                        )

                        calculate_btn = gr.Button("Calculate Recipe", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("#### Coating Recipe")

                        recipe_display = gr.Textbox(
                            label="Recipe",
                            lines=25,
                            interactive=False,
                        )

                        with gr.Row():
                            recipe_json = gr.JSON(label="Recipe Details", visible=False)

                        gr.Markdown(
                            """
                            ---
                            #### Formula Reference

                            **Standard Pt/Pd Formula:**
                            - Solution A: Ferric Oxalate #1 (base sensitizer)
                            - Solution B: Ferric Oxalate #2 (contrast booster, optional)
                            - Solution C: Metal (Platinum, Palladium, or mix)
                            - Na2: Sodium Platinum (contrast agent)

                            **Rule:** Drops of metal (C) = Drops of FO (A + B)

                            **Coverage:** ~46 drops per 8x10" (~0.46 drops/sq inch)
                            """
                        )

                def on_standard_size_change(size):
                    """Update dimensions when standard size is selected."""
                    sizes = ChemistryCalculator.get_standard_sizes()
                    if size in sizes:
                        w, h = sizes[size]
                        return w, h
                    return gr.update(), gr.update()

                def on_metal_mix_change(mix):
                    """Show/hide custom slider based on mix selection."""
                    if mix == "custom":
                        return gr.update(visible=True)
                    return gr.update(visible=False)

                def calculate_chemistry(
                    width, height, metal_mix, pt_custom, absorbency, method, contrast, na2
                ):
                    """Calculate and display chemistry recipe."""
                    try:
                        calculator = ChemistryCalculator()

                        # Determine platinum ratio
                        if metal_mix == "custom":
                            pt_ratio = pt_custom / 100.0
                        else:
                            pt_ratio = METAL_MIX_RATIOS[MetalMix(metal_mix)]

                        recipe = calculator.calculate(
                            width_inches=float(width),
                            height_inches=float(height),
                            platinum_ratio=pt_ratio,
                            paper_absorbency=PaperAbsorbency(absorbency),
                            coating_method=CoatingMethod(method),
                            contrast_boost=contrast / 100.0,
                            na2_ratio=na2 / 100.0,
                        )

                        return recipe.format_recipe(), recipe.to_dict()
                    except Exception as e:
                        return f"Error: {str(e)}", {"error": str(e)}

                # Connect handlers
                standard_size_select.change(
                    on_standard_size_change,
                    inputs=[standard_size_select],
                    outputs=[print_width, print_height],
                )

                metal_mix_select.change(
                    on_metal_mix_change,
                    inputs=[metal_mix_select],
                    outputs=[platinum_slider],
                )

                calculate_btn.click(
                    calculate_chemistry,
                    inputs=[
                        print_width, print_height, metal_mix_select, platinum_slider,
                        paper_absorbency_select, coating_method_select, contrast_slider, na2_slider
                    ],
                    outputs=[recipe_display, recipe_json],
                )

            # ========================================
            # TAB 9: Settings
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
                            value="No API key configured" if not settings.llm.get_active_api_key() else "API key configured (from environment)",
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
                        from ptpd_calibration.config import get_settings, LLMProvider
                        current_settings = get_settings()
                        current_settings.llm.runtime_api_key = api_key.strip()
                        current_settings.llm.provider = LLMProvider(provider)
                        current_settings.llm.anthropic_model = anthropic_model
                        current_settings.llm.openai_model = openai_model

                        # Verify the key works (basic format check)
                        key = api_key.strip()
                        if provider == "anthropic" and not key.startswith("sk-ant-"):
                            return "Warning: Anthropic keys typically start with 'sk-ant-'. Key saved anyway.", {
                                "provider": provider,
                                "anthropic_model": anthropic_model,
                                "openai_model": openai_model,
                                "api_key_configured": True,
                                "max_tokens": current_settings.llm.max_tokens,
                                "temperature": current_settings.llm.temperature,
                            }
                        elif provider == "openai" and not key.startswith("sk-"):
                            return "Warning: OpenAI keys typically start with 'sk-'. Key saved anyway.", {
                                "provider": provider,
                                "anthropic_model": anthropic_model,
                                "openai_model": openai_model,
                                "api_key_configured": True,
                                "max_tokens": current_settings.llm.max_tokens,
                                "temperature": current_settings.llm.temperature,
                            }

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
                    inputs=[llm_provider_select, api_key_input, anthropic_model_input, openai_model_input],
                    outputs=[api_key_status, current_config_display],
                )

            # ========================================
            # TAB 10: About
            # ========================================
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## Pt/Pd Calibration Studio

                    An AI-powered calibration system for platinum/palladium printing.

                    ### Features

                    - **Curve Display**: Upload and compare multiple curves with comprehensive statistics
                    - **Step Wedge Analysis**: Automatic step wedge detection, density extraction, and quality assessment
                    - **Step Tablet Analysis**: Upload scans and extract density measurements
                    - **Curve Generation**: Create linearization curves for digital negatives
                    - **Curve Editor**: Upload .quad files, modify curves, smooth curves, and apply AI-powered enhancements
                    - **AI Enhancement**: Intelligent curve optimization
                    - **AI Assistant**: Get help from an AI expert in Pt/Pd printing
                    - **Chemistry Calculator**: Calculate coating solution amounts for any print size
                    - **Recipe Suggestions**: Get starting parameters for new papers
                    - **Troubleshooting**: Diagnose and fix common problems

                    ### Supported Formats

                    - QuadTone RIP (.txt, .quad)
                    - Piezography (.ppt)
                    - CSV
                    - JSON

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

    return app


def launch_ui(share: bool = False, port: int = 7860, server_name: str = "127.0.0.1"):
    """
    Launch the Gradio UI.

    Args:
        share: Whether to create a public share link.
        port: Port to run on.
        server_name: Server name to bind to.
    """
    app = create_gradio_app(share=share)
    app.launch(share=share, server_port=port, server_name=server_name)


if __name__ == "__main__":
    import os
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_ui(port=port)
