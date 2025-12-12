"""
Print Analysis tab for analyzing scanned prints and refining curves.

Provides a UI for:
- Uploading scanned prints for analysis
- Viewing density measurements and recommendations
- Applying feedback-based curve refinements
- Tracking calibration sessions
"""

import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple
from datetime import datetime

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from ptpd_calibration.calibration import (
    PrintAnalyzer,
    PrintAnalysis,
    CurveCalibrator,
    QuadCurveParser,
    CalibrationDatabase,
    TargetDensities,
    CALIBRATION_PROFILES,
    get_available_calibration_profiles,
)
from ptpd_calibration.papers.profiles import PaperDatabase, BUILTIN_PAPERS


def get_paper_choices() -> list[str]:
    """Get paper choices including calibration profiles."""
    db = PaperDatabase()
    papers = db.list_paper_names()
    # Add calibration profile keys as alternative choices
    for key in get_available_calibration_profiles():
        profile_name = CALIBRATION_PROFILES[key].name
        if profile_name not in papers:
            papers.append(profile_name)
    return sorted(papers)


def get_process_choices() -> list[str]:
    """Get available process types."""
    return ["Platinum/Palladium", "Cyanotype", "Silver Gelatin"]


def get_target_densities_for_process(process: str) -> TargetDensities:
    """Get target densities based on process type."""
    if process == "Cyanotype":
        return TargetDensities.for_cyanotype()
    elif process == "Silver Gelatin":
        return TargetDensities.for_silver_gelatin()
    else:
        return TargetDensities.for_platinum_palladium()


def analyze_print_image(
    image: Optional[np.ndarray],
    paper_type: str,
    process: str,
    exclude_borders: bool
) -> Tuple[str, Optional[plt.Figure], float, float, float, str]:
    """
    Analyze a scanned print image.

    Returns:
        Tuple of (summary_markdown, histogram_figure,
                  highlight_adj, midtone_adj, shadow_adj, notes)
    """
    if image is None:
        return (
            "Please upload a scanned print image.",
            None,
            0.0, 0.0, 0.0,
            ""
        )

    try:
        # Get target densities based on process
        targets = get_target_densities_for_process(process)
        analyzer = PrintAnalyzer(targets=targets)

        # Analyze the image
        analysis = analyzer.analyze_print_scan(
            image,
            exclude_borders=exclude_borders
        )
        analysis.paper_type = paper_type

        # Create summary markdown
        summary = _create_analysis_summary(analysis, targets)

        # Create zone histogram figure
        fig = _create_zone_histogram(analysis)

        # Get adjustment notes
        notes = "\n".join(f"- {note}" for note in analysis.notes)

        return (
            summary,
            fig,
            analysis.recommended_highlight_adj,
            analysis.recommended_midtone_adj,
            analysis.recommended_shadow_adj,
            notes
        )

    except Exception as e:
        return (
            f"Error analyzing image: {str(e)}",
            None,
            0.0, 0.0, 0.0,
            str(e)
        )


def _create_analysis_summary(analysis: PrintAnalysis, targets: TargetDensities) -> str:
    """Create a markdown summary of the analysis."""
    def status_icon(measured: float, target: float, tolerance: float) -> str:
        diff = abs(measured - target)
        if diff < tolerance:
            return "OK"
        elif diff < tolerance * 2:
            return "Fair"
        else:
            return "Adjust"

    h_status = status_icon(analysis.highlight_density, targets.highlight, 0.06)
    m_status = status_icon(analysis.midtone_density, targets.midtone, 0.10)
    s_status = status_icon(analysis.shadow_density, targets.shadow, 0.12)

    summary = f"""
### Analysis Results

| Zone | Measured | Target | Status |
|------|----------|--------|--------|
| Highlights | {analysis.highlight_density:.2f} | {targets.highlight:.2f} | {h_status} |
| Midtones | {analysis.midtone_density:.2f} | {targets.midtone:.2f} | {m_status} |
| Shadows (Dmax) | {analysis.shadow_density:.2f} | {targets.shadow:.2f} | {s_status} |

**Tonal Range:** {analysis.tonal_range:.2f} (target: {targets.tonal_range:.2f})

**Midtone Separation:** {analysis.midtone_separation:.2f}
{' (LOW - may appear flat)' if analysis.midtone_separation < 0.15 else ''}

### Recommended Adjustments

| Zone | Adjustment |
|------|------------|
| Highlights | {analysis.recommended_highlight_adj:+.1%} |
| Midtones | {analysis.recommended_midtone_adj:+.1%} |
| Shadows | {analysis.recommended_shadow_adj:+.1%} |
"""
    return summary


def _create_zone_histogram(analysis: PrintAnalysis) -> Optional[plt.Figure]:
    """Create a zone distribution histogram."""
    if not analysis.zone_histogram:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    zones = list(analysis.zone_histogram.keys())
    values = list(analysis.zone_histogram.values())

    # Create bar colors - darker for shadows, lighter for highlights
    colors = plt.cm.gray(np.linspace(0.1, 0.9, len(zones)))

    bars = ax.bar(range(len(zones)), [v * 100 for v in values], color=colors, edgecolor='black')

    # Add zone labels
    zone_labels = [f"Zone {i}\n({z})" for i, z in enumerate(
        ['Black', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'White']
    )]
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels(zone_labels[:len(zones)], fontsize=8)

    ax.set_ylabel('Percentage of Image (%)')
    ax.set_title('Ansel Adams Zone Distribution')
    ax.grid(axis='y', alpha=0.3)

    # Highlight Zone V (middle gray) reference
    if len(zones) > 5:
        bars[5].set_edgecolor('#f59e0b')
        bars[5].set_linewidth(2)

    plt.tight_layout()
    return fig


def generate_refined_curve(
    base_curve_file: Optional[Any],
    highlight_adj: float,
    midtone_adj: float,
    shadow_adj: float,
    output_name: str
) -> Tuple[Optional[str], str]:
    """
    Generate a refined curve based on adjustments.

    Returns:
        Tuple of (output_file_path, status_message)
    """
    if base_curve_file is None:
        return None, "Please upload a base curve file (.quad)"

    try:
        # Parse the input curve
        input_path = base_curve_file.name if hasattr(base_curve_file, 'name') else str(base_curve_file)
        header, curves = QuadCurveParser.parse(input_path)

        # Apply feedback adjustments
        calibrator = CurveCalibrator()
        refined = calibrator.adjust_all_from_feedback(
            curves,
            highlight_delta=highlight_adj,
            midtone_delta=midtone_adj,
            shadow_delta=shadow_adj
        )

        # Generate output path
        safe_name = "".join(c for c in output_name if c.isalnum() or c in " -_")[:40]
        if not safe_name:
            safe_name = "refined_curve"
        output_path = Path(tempfile.gettempdir()) / f"{safe_name}.quad"

        # Write the refined curve
        extra_comments = [
            f"Refined from print feedback - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Adjustments: H={highlight_adj:+.0%}, M={midtone_adj:+.0%}, S={shadow_adj:+.0%}"
        ]
        QuadCurveParser.write(str(output_path), header, refined, extra_comments)

        return str(output_path), f"Curve saved: {output_path.name}"

    except Exception as e:
        return None, f"Error generating curve: {str(e)}"


def apply_paper_profile(
    base_curve_file: Optional[Any],
    paper_type: str,
    output_name: str
) -> Tuple[Optional[str], str]:
    """
    Apply paper-specific adjustments to a curve.

    Returns:
        Tuple of (output_file_path, status_message)
    """
    if base_curve_file is None:
        return None, "Please upload a base curve file (.quad)"

    try:
        input_path = base_curve_file.name if hasattr(base_curve_file, 'name') else str(base_curve_file)
        header, curves = QuadCurveParser.parse(input_path)

        # Get paper key
        paper_key = paper_type.lower().replace(" ", "_")

        # Check for calibration profile first, then paper profile
        if paper_key in CALIBRATION_PROFILES:
            profile = CALIBRATION_PROFILES[paper_key]
            calibrator = CurveCalibrator(profile)
        elif paper_key in BUILTIN_PAPERS:
            # Generate calibration profile from paper profile
            from ptpd_calibration.calibration.curve_adjuster import CalibrationProfile
            profile = CalibrationProfile.from_paper_profile(BUILTIN_PAPERS[paper_key])
            calibrator = CurveCalibrator(profile)
        else:
            return None, f"Unknown paper: {paper_type}"

        # Apply adjustments
        adjusted = calibrator.adjust_all_curves(curves)

        # Generate output path
        safe_name = "".join(c for c in output_name if c.isalnum() or c in " -_")[:40]
        if not safe_name:
            safe_name = f"{paper_key}_curve"
        output_path = Path(tempfile.gettempdir()) / f"{safe_name}.quad"

        # Write the adjusted curve
        extra_comments = [
            f"Adjusted for {profile.name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Highlights +{profile.highlight_boost*100:.0f}%, "
            f"Midtones +{profile.midtone_boost*100:.0f}%, "
            f"Shadows +{profile.shadow_boost*100:.0f}%"
        ]
        QuadCurveParser.write(str(output_path), header, adjusted, extra_comments)

        return str(output_path), f"Curve saved: {output_path.name}"

    except Exception as e:
        return None, f"Error applying profile: {str(e)}"


def build_print_analysis_tab():
    """Build the Print Analysis & Curve Refinement tab."""
    with gr.TabItem("Print Analysis"):
        gr.Markdown(
            """
            ### Print Analysis & Curve Refinement

            Analyze scanned prints to measure tonal characteristics and generate
            curve adjustment recommendations. Upload a scan of your test print,
            review the analysis, and apply refinements to your curve.
            """
        )

        with gr.Tabs():
            # Tab 1: Analyze Print
            with gr.TabItem("Analyze Print"):
                with gr.Row():
                    with gr.Column(scale=1):
                        print_scan = gr.Image(
                            label="Upload Print Scan",
                            type="numpy",
                            sources=["upload"],
                        )

                        paper_dropdown = gr.Dropdown(
                            choices=get_paper_choices(),
                            value="Arches Platine",
                            label="Paper Type",
                            info="Select paper for context (optional)"
                        )

                        process_dropdown = gr.Dropdown(
                            choices=get_process_choices(),
                            value="Platinum/Palladium",
                            label="Process Type",
                            info="Determines target density values"
                        )

                        exclude_borders = gr.Checkbox(
                            label="Auto-exclude brushed borders",
                            value=True,
                            info="Automatically detect and exclude irregular borders"
                        )

                        analyze_btn = gr.Button(
                            "Analyze Print",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(
                            label="Analysis Results",
                            value="Upload a print scan to analyze."
                        )

                        zone_histogram = gr.Plot(
                            label="Zone Distribution"
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Recommended Adjustments")

                        highlight_slider = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Highlight Adjustment",
                            info="Positive = more ink, Negative = less ink"
                        )
                        midtone_slider = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Midtone Adjustment",
                            info="Key zone for pt/pd - typically needs boost"
                        )
                        shadow_slider = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Shadow Adjustment",
                            info="Affects Dmax"
                        )

                    with gr.Column():
                        gr.Markdown("### Observations")
                        notes_output = gr.Textbox(
                            label="Analysis Notes",
                            lines=6,
                            interactive=False
                        )

                # Wire up analysis
                analyze_btn.click(
                    analyze_print_image,
                    inputs=[print_scan, paper_dropdown, process_dropdown, exclude_borders],
                    outputs=[
                        analysis_output,
                        zone_histogram,
                        highlight_slider,
                        midtone_slider,
                        shadow_slider,
                        notes_output
                    ]
                )

            # Tab 2: Refine Curve
            with gr.TabItem("Refine Curve"):
                gr.Markdown(
                    """
                    ### Apply Adjustments to Curve

                    Use the recommended adjustments from the analysis (or your own values)
                    to generate a refined curve file.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        base_curve_upload = gr.File(
                            label="Base Curve File (.quad)",
                            file_types=[".quad"]
                        )

                        gr.Markdown("#### Adjustments to Apply")

                        refine_highlight = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Highlight Adjustment"
                        )
                        refine_midtone = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Midtone Adjustment"
                        )
                        refine_shadow = gr.Slider(
                            -0.15, 0.15, value=0, step=0.01,
                            label="Shadow Adjustment"
                        )

                        output_curve_name = gr.Textbox(
                            label="Output Curve Name",
                            value="Refined_Curve",
                            info="Name for the output file"
                        )

                        generate_btn = gr.Button(
                            "Generate Refined Curve",
                            variant="primary"
                        )

                    with gr.Column():
                        refined_curve_file = gr.File(
                            label="Download Refined Curve"
                        )

                        refine_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

                        gr.Markdown(
                            """
                            #### Tips

                            - Start with small adjustments (2-5%)
                            - The midtone zone is most critical for pt/pd
                            - Arches Platine typically needs +8-10% midtone boost
                            - Re-analyze after each print to converge on optimal curve
                            """
                        )

                # Copy adjustments from analysis tab
                def copy_adjustments(h, m, s):
                    return h, m, s

                # Button to copy from analysis (hidden, triggered by tab switch)
                highlight_slider.change(
                    copy_adjustments,
                    inputs=[highlight_slider, midtone_slider, shadow_slider],
                    outputs=[refine_highlight, refine_midtone, refine_shadow]
                )
                midtone_slider.change(
                    copy_adjustments,
                    inputs=[highlight_slider, midtone_slider, shadow_slider],
                    outputs=[refine_highlight, refine_midtone, refine_shadow]
                )
                shadow_slider.change(
                    copy_adjustments,
                    inputs=[highlight_slider, midtone_slider, shadow_slider],
                    outputs=[refine_highlight, refine_midtone, refine_shadow]
                )

                generate_btn.click(
                    generate_refined_curve,
                    inputs=[
                        base_curve_upload,
                        refine_highlight,
                        refine_midtone,
                        refine_shadow,
                        output_curve_name
                    ],
                    outputs=[refined_curve_file, refine_status]
                )

            # Tab 3: Paper Profiles
            with gr.TabItem("Paper Profiles"):
                gr.Markdown(
                    """
                    ### Apply Paper-Specific Adjustments

                    Apply pre-configured calibration profiles based on empirically tested
                    paper characteristics. These profiles compensate for absorption,
                    dot gain, and tonal response of specific papers.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        paper_base_curve = gr.File(
                            label="Base Curve File (.quad)",
                            file_types=[".quad"]
                        )

                        paper_profile_dropdown = gr.Dropdown(
                            choices=get_paper_choices(),
                            value="Arches Platine",
                            label="Paper Profile",
                            info="Select paper to apply its calibration profile"
                        )

                        paper_output_name = gr.Textbox(
                            label="Output Curve Name",
                            value="Paper_Adjusted_Curve",
                            info="Name for the output file"
                        )

                        apply_profile_btn = gr.Button(
                            "Apply Paper Profile",
                            variant="primary"
                        )

                    with gr.Column():
                        paper_curve_file = gr.File(
                            label="Download Adjusted Curve"
                        )

                        paper_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

                        # Show profile details
                        profile_info = gr.Markdown(
                            """
                            #### Available Calibration Profiles

                            | Paper | Midtone Boost | Shadow Boost | Notes |
                            |-------|--------------|--------------|-------|
                            | Arches Platine | +10% | +7% | High absorbency, needs aggressive midtone boost |
                            | Bergger COT320 | +4% | +5% | Less absorbent, holds highlights well |
                            | Hahnemuhle Platinum | +6% | +6% | Consistent results, slightly cool base |
                            | Revere Platinum | +5% | +6% | Good all-around, moderate absorption |
                            | Stonehenge | +5% | +5% | Affordable, warm base |
                            | Weston Diploma | +8% | +7% | Heavy cotton, needs midtone compensation |
                            """
                        )

                apply_profile_btn.click(
                    apply_paper_profile,
                    inputs=[
                        paper_base_curve,
                        paper_profile_dropdown,
                        paper_output_name
                    ],
                    outputs=[paper_curve_file, paper_status]
                )

        # Help accordion at the bottom
        with gr.Accordion("About Print Analysis", open=False):
            gr.Markdown(
                """
                ### How It Works

                1. **Upload a scan** of your test print (step tablet or actual image)
                2. The analyzer **measures density** in highlight, midtone, and shadow zones
                3. Densities are **compared to targets** for your process type
                4. **Recommendations** are generated for curve adjustments
                5. Apply adjustments to generate a **refined curve**
                6. **Repeat** until your prints match your vision

                ### Key Insights

                - **Midtones are critical** in pt/pd - they often appear muted without compensation
                - **Arches Platine** typically needs 8-12% midtone boost
                - The **zone histogram** shows distribution across Ansel Adams zones
                - **Zone V** (middle gray) should have good separation
                - Target **Dmax** for pt/pd is typically 1.5-1.7

                ### Target Densities by Process

                | Process | Highlight | Midtone | Shadow (Dmax) |
                |---------|-----------|---------|---------------|
                | Pt/Pd | 0.12 | 0.65 | 1.55 |
                | Cyanotype | 0.15 | 0.80 | 1.90 |
                | Silver Gelatin | 0.08 | 0.55 | 2.10 |
                """
            )
