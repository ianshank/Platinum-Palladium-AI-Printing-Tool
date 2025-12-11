import gradio as gr
from ptpd_calibration.chemistry import (
    ChemistryCalculator,
    PaperAbsorbency,
    CoatingMethod,
    MetalMix,
    METAL_MIX_RATIOS,
)


def calculate_recipe_ui(w, h, pt_ratio, absorbency, method, cont, na2_val):
    """Calculate recipe from UI inputs and return HTML, text, and dict outputs."""
    try:
        calculator = ChemistryCalculator()
        # pt_ratio is 0-100, convert to 0.0-1.0
        recipe = calculator.calculate(
            width_inches=float(w),
            height_inches=float(h),
            platinum_ratio=pt_ratio / 100.0,
            paper_absorbency=PaperAbsorbency(absorbency),
            coating_method=CoatingMethod(method),
            contrast_boost=cont / 100.0,
            na2_ratio=na2_val / 100.0,
        )

        # Generate visual HTML
        # Simple representation: Drops as circles
        # Total drops
        total_drops = recipe.total_drops

        html = f"""
        <div style="padding: 10px; background: var(--ptpd-card); border-radius: 8px;">
            <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">{total_drops} Total Drops</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        """

        # Helper to add drops
        def add_drops(name, count, color):
            return f"""
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="font-size: 20px; color: {color};">●</div>
                <div style="font-weight: bold; font-size: 18px;">{count}</div>
                <div style="font-size: 12px; opacity: 0.8;">{name}</div>
            </div>
            """

        if recipe.ferric_oxalate_drops > 0:
            html += add_drops("FO#1", recipe.ferric_oxalate_drops, "#fbbf24")  # Amber
        if recipe.ferric_oxalate_contrast_drops > 0:
            html += add_drops("FO#2", recipe.ferric_oxalate_contrast_drops, "#d97706")  # Darker Amber
        if recipe.platinum_drops > 0:
            html += add_drops("Pt", recipe.platinum_drops, "#c0c0c0")  # Silver
        if recipe.palladium_drops > 0:
            html += add_drops("Pd", recipe.palladium_drops, "#d4a574")  # Gold/Bronze
        if recipe.na2_drops > 0:
            html += add_drops("Na2", recipe.na2_drops, "#ef4444")  # Red

        html += "</div></div>"

        return html, recipe.format_recipe(), recipe.to_dict()
    except Exception as e:
        return f"Error: {str(e)}", str(e), {}


def build_chemistry_tab():
    """Build the Chemistry Calculator tab."""
    with gr.TabItem("🧪 Chemistry Calculator"):
        gr.Markdown(
            """
            ### Coating Solution Calculator

            Calculate platinum/palladium coating solution amounts based on your print dimensions.
            """
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Paper Size")

                with gr.Row():
                    btn_4x5 = gr.Button("4×5", size="sm")
                    btn_5x7 = gr.Button("5×7", size="sm")
                    btn_8x10 = gr.Button("8×10", size="sm")
                with gr.Row():
                    btn_11x14 = gr.Button("11×14", size="sm")
                    btn_16x20 = gr.Button("16×20", size="sm")
                    btn_custom = gr.Button("Custom", size="sm")

                with gr.Row():
                    width = gr.Number(label="Width (inches)", value=8)
                    height = gr.Number(label="Height (inches)", value=10)

                gr.Markdown("### Metal Ratio")
                ratio_slider = gr.Slider(
                    minimum=0, maximum=100, value=50,
                    label="← More Palladium | More Platinum →"
                )

                # Visual indicator
                ratio_viz = gr.HTML("""
                    <div style="display:flex; height:20px; border-radius:4px; overflow:hidden; margin-bottom: 5px;">
                        <div style="width:50%; background: #d4a574;"></div>
                        <div style="width:50%; background: #c0c0c0;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span>Warmer tones</span>
                        <span>Cooler tones</span>
                    </div>
                """)

                gr.Markdown("### Coating & Contrast")
                paper_absorbency = gr.Dropdown(
                    choices=[("Low (Hot Press)", "low"), ("Medium", "medium"), ("High (Cold Press)", "high")],
                    value="medium",
                    label="Paper Absorbency"
                )
                coating_method = gr.Dropdown(
                    choices=[("Brush", "brush"), ("Glass Rod", "rod"), ("Puddle Pusher", "puddle_pusher")],
                    value="brush",
                    label="Coating Method"
                )
                contrast = gr.Slider(0, 100, 0, label="Contrast Boost (FO#2) %")
                na2 = gr.Slider(0, 50, 25, label="Na2 % (of metal)")

                calculate_btn = gr.Button("Calculate Recipe", variant="primary")

            # Right Column: Results
            with gr.Column(scale=1):
                gr.Markdown("### Recipe")

                # Visual dropper representation
                recipe_html = gr.HTML(label="Visual Recipe")

                # Detailed text
                recipe_text = gr.Textbox(label="Details", lines=10, interactive=False)

                # Hidden JSON for data
                recipe_json = gr.JSON(visible=False)

                with gr.Row():
                    copy_btn = gr.Button("📋 Copy Recipe")
                    log_btn = gr.Button("📝 Log to Session")

        # Logic
        btn_4x5.click(lambda: (4, 5), outputs=[width, height])
        btn_5x7.click(lambda: (5, 7), outputs=[width, height])
        btn_8x10.click(lambda: (8, 10), outputs=[width, height])
        btn_11x14.click(lambda: (11, 14), outputs=[width, height])
        btn_16x20.click(lambda: (16, 20), outputs=[width, height])
        # Custom button just focuses inputs, essentially no-op here or could clear them

        def update_viz(value):
            # Update HTML gradient based on slider
            pd_pct = 100 - value
            pt_pct = value
            return f"""
            <div style="display:flex; height:20px; border-radius:4px; overflow:hidden; margin-bottom: 5px; background: linear-gradient(90deg, #d4a574 {pd_pct}%, #c0c0c0 {pd_pct}%);">
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px;">
                <span>{pd_pct}% Pd</span>
                <span>{pt_pct}% Pt</span>
            </div>
            """

        ratio_slider.change(update_viz, inputs=[ratio_slider], outputs=[ratio_viz])

        calculate_btn.click(
            calculate_recipe_ui,
            inputs=[width, height, ratio_slider, paper_absorbency, coating_method, contrast, na2],
            outputs=[recipe_html, recipe_text, recipe_json]
        )

        # Copy button (simulated with Javascript)
        copy_btn.click(None, [recipe_text], None, js="(text) => navigator.clipboard.writeText(text)")

        # Log to session (placeholder)
        log_btn.click(lambda x: gr.Info("Recipe logged to session!"), inputs=[recipe_json], outputs=[])
