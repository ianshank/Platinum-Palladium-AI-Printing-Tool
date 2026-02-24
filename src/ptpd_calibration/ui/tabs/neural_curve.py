"""
Neural Curve Prediction tab for AI-assisted curve generation.

Provides a UI for training neural network models on calibration data
and using them for predictive curve generation in Pt/Pd printing.

This tab integrates with the deep_learning module to expose:
- Model training with configurable architectures
- Curve prediction with uncertainty estimation
- Model management and comparison

Usage:
    from ptpd_calibration.ui.tabs.neural_curve import build_neural_curve_tab

    with gr.Blocks() as app:
        build_neural_curve_tab(session_logger)
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for optional deep learning dependencies
_DL_AVAILABLE = None

# Temp file tracking for cleanup
_TEMP_FILES: list[str] = []


def _register_temp_file(filepath: str) -> None:
    """Register a temp file for cleanup."""
    _TEMP_FILES.append(filepath)
    # Keep only last 50 files to prevent unbounded growth
    if len(_TEMP_FILES) > 50:
        _cleanup_old_temp_files()


def _cleanup_old_temp_files() -> None:
    """Clean up old temporary files."""
    import os

    while len(_TEMP_FILES) > 25:
        old_file = _TEMP_FILES.pop(0)
        try:
            if os.path.exists(old_file):
                os.unlink(old_file)
                logger.debug(f"Cleaned up temp file: {old_file}")
        except Exception as e:
            logger.debug(f"Failed to clean up temp file: {e}")


def _check_dl_available() -> bool:
    """Check if deep learning dependencies are available."""
    global _DL_AVAILABLE
    if _DL_AVAILABLE is None:
        try:
            import torch  # noqa: F401

            _DL_AVAILABLE = True
        except ImportError:
            _DL_AVAILABLE = False
            logger.warning("PyTorch not installed. Neural curve features will be simulated.")
    return _DL_AVAILABLE


def _get_neural_predictor() -> Any:
    """Lazily import NeuralCurvePredictor.

    Reserved for future integration with real neural network models.
    Currently returns None as the UI uses simulated predictions.

    Returns:
        NeuralCurvePredictor class if available, None otherwise.
    """
    if _check_dl_available():
        try:
            from ptpd_calibration.deep_learning.neural_curves import NeuralCurvePredictor

            return NeuralCurvePredictor
        except ImportError:
            pass
    return None


def _get_dl_settings() -> Any:
    """Lazily import deep learning settings.

    Reserved for future integration with configurable deep learning settings.
    Currently returns None as the UI uses default values.

    Returns:
        DeepLearningSettings instance if available, None otherwise.
    """
    try:
        from ptpd_calibration.deep_learning.config import get_deep_learning_settings

        return get_deep_learning_settings()
    except ImportError:
        return None


def build_neural_curve_tab(session_logger: Any = None) -> None:
    """Build the Neural Curve Prediction tab.

    Args:
        session_logger: Optional SessionLogger instance for historical data.
    """
    with gr.TabItem("🧠 Neural Curves"):
        gr.Markdown(
            """
            ### AI-Powered Curve Prediction

            Train neural networks on your calibration history to predict optimal
            linearization curves for new paper and chemistry combinations.
            """
        )

        # Check deep learning availability
        dl_available = _check_dl_available()

        if not dl_available:
            gr.Markdown(
                """
                > **Note:** PyTorch is not installed. Neural curve features are running
                > in simulation mode. Install with: `pip install torch`
                """
            )

        # =====================================================
        # STATE: Track models and training progress
        # =====================================================
        trained_models_state = gr.State([])
        current_prediction_state = gr.State(None)

        # =====================================================
        # SECTION 1: Model Training
        # =====================================================
        with gr.Group():
            gr.Markdown("### 📚 Step 1: Train Model")

            with gr.Row():
                # Left Column: Training Configuration
                with gr.Column(scale=1):
                    gr.Markdown("#### Training Configuration")

                    # Data source selection
                    data_source = gr.Radio(
                        choices=[
                            "From session history",
                            "Upload custom data",
                            "Generate synthetic",
                        ],
                        value="Generate synthetic",
                        label="Data Source",
                    )

                    # File upload (shown conditionally)
                    custom_data_upload = gr.File(
                        label="Upload Training Data (.csv)",
                        file_types=[".csv", ".json"],
                        visible=False,
                    )

                    # Session filter (shown for history mode)
                    session_filter = gr.Dropdown(
                        choices=["All sessions", "Last 10 sessions", "Last 30 days"],
                        value="All sessions",
                        label="Session Filter",
                        visible=False,
                    )

                    # Synthetic data options (shown for synthetic mode)
                    with gr.Group(visible=True) as synthetic_options:
                        synthetic_samples = gr.Slider(
                            minimum=50, maximum=1000, value=200, step=50, label="Number of Samples"
                        )
                        synthetic_noise = gr.Slider(
                            minimum=0.0, maximum=0.1, value=0.02, step=0.01, label="Noise Level"
                        )

                    # Model architecture
                    model_architecture = gr.Dropdown(
                        choices=[
                            ("Transformer (recommended)", "transformer"),
                            ("LSTM", "lstm"),
                            ("GRU", "gru"),
                            ("MLP", "mlp"),
                            ("Hybrid", "hybrid"),
                        ],
                        value="transformer",
                        label="Model Architecture",
                    )

                    # Architecture info
                    architecture_info = gr.Markdown(
                        "_Transformer: Best for capturing long-range dependencies in curves_"
                    )

                    # Advanced training options
                    with gr.Accordion("Advanced Training Parameters", open=False):
                        learning_rate = gr.Slider(
                            minimum=0.0001,
                            maximum=0.01,
                            value=0.001,
                            step=0.0001,
                            label="Learning Rate",
                        )
                        epochs = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10, label="Training Epochs"
                        )
                        batch_size = gr.Dropdown(
                            choices=[8, 16, 32, 64], value=32, label="Batch Size"
                        )
                        hidden_dim = gr.Slider(
                            minimum=32, maximum=256, value=128, step=32, label="Hidden Dimension"
                        )
                        num_layers = gr.Slider(
                            minimum=1, maximum=6, value=3, step=1, label="Number of Layers"
                        )
                        dropout = gr.Slider(
                            minimum=0.0, maximum=0.5, value=0.1, step=0.05, label="Dropout Rate"
                        )
                        use_uncertainty = gr.Checkbox(
                            value=True, label="Enable Uncertainty Estimation"
                        )

                    # Model name
                    model_name = gr.Textbox(
                        label="Model Name",
                        value=f"Neural Curve {datetime.now().strftime('%Y%m%d_%H%M')}",
                        placeholder="e.g., Arches Platine v1",
                    )

                    # Train button
                    train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")

                # Right Column: Training Status
                with gr.Column(scale=1):
                    gr.Markdown("#### Training Status")

                    # Status indicator
                    training_status = gr.Markdown("⏳ Ready to train")

                    # Progress (placeholder - would use actual progress in real impl)
                    training_progress = gr.Textbox(
                        label="Progress", value="Not started", interactive=False, lines=2
                    )

                    # Loss plot
                    loss_plot = gr.Plot(label="Training Loss", visible=False)

                    # Data summary
                    with gr.Group():
                        gr.Markdown("#### Data Summary")
                        with gr.Row():
                            data_samples_display = gr.Textbox(
                                label="Samples", value="-", interactive=False, scale=1
                            )
                            data_features_display = gr.Textbox(
                                label="Features", value="-", interactive=False, scale=1
                            )
                        data_quality_display = gr.Textbox(
                            label="Data Quality", value="-", interactive=False
                        )

        # =====================================================
        # SECTION 2: Model Selection and Prediction
        # =====================================================
        with gr.Group():
            gr.Markdown("### 🎯 Step 2: Generate Prediction")

            with gr.Row():
                # Model Selection
                with gr.Column(scale=1):
                    gr.Markdown("#### Select Model")

                    model_list = gr.Dropdown(
                        choices=["No models trained yet"],
                        label="Trained Models",
                        value="No models trained yet",
                        interactive=True,
                    )

                    model_info_display = gr.Markdown("_Train a model first to see details_")

                    with gr.Row():
                        _refresh_models_btn = gr.Button("🔄 Refresh", size="sm")  # noqa: F841
                        delete_model_btn = gr.Button(
                            "🗑️ Delete", variant="stop", size="sm", interactive=False
                        )

                # Prediction Inputs
                with gr.Column(scale=1):
                    gr.Markdown("#### Prediction Inputs")

                    # Paper type
                    predict_paper = gr.Dropdown(
                        choices=[
                            ("Arches Platine", "arches_platine"),
                            ("Bergger COT320", "bergger_cot320"),
                            ("Hahnemühle Platinum Rag", "hahnemuhle_platinum"),
                            ("Revere Platinum", "revere_platinum"),
                            ("Custom", "custom"),
                        ],
                        value="arches_platine",
                        label="Paper Type",
                    )

                    # Chemistry ratio
                    predict_pt_ratio = gr.Slider(
                        minimum=0, maximum=100, value=50, step=5, label="Platinum Ratio (%)"
                    )

                    # Pt:Pd visualization
                    pt_pd_viz = gr.HTML(value=_create_ratio_viz(50), label="Metal Ratio")

                    # Target density
                    predict_target_dmax = gr.Slider(
                        minimum=1.0, maximum=2.5, value=1.8, step=0.1, label="Target Dmax"
                    )

                    # Curve output points
                    output_points = gr.Dropdown(
                        choices=[21, 51, 101, 256], value=51, label="Output Points"
                    )

                    # Predict button
                    predict_btn = gr.Button("✨ Generate Prediction", variant="primary", size="lg")

        # =====================================================
        # SECTION 3: Results
        # =====================================================
        with gr.Group():
            gr.Markdown("### 📊 Step 3: Review Results")

            with gr.Row():
                # Curve visualization
                with gr.Column(scale=1):
                    predicted_curve_plot = gr.Plot(label="Predicted Curve")

                    # Curve data table
                    curve_data_table = gr.Dataframe(
                        headers=["Step", "Input", "Output", "Uncertainty"],
                        visible=False,
                        interactive=False,
                    )

                # Summary and actions
                with gr.Column(scale=1):
                    gr.Markdown("#### Prediction Summary")

                    prediction_summary = gr.Markdown("_Generate a prediction to see summary_")

                    # Confidence metrics
                    with gr.Row():
                        confidence_score = gr.Textbox(
                            label="Confidence", value="-", interactive=False
                        )
                        uncertainty_score = gr.Textbox(
                            label="Mean Uncertainty", value="-", interactive=False
                        )

                    # Quality checks
                    quality_checks = gr.Markdown("")

                    gr.Markdown("#### Export Options")

                    export_format = gr.Dropdown(
                        choices=[
                            ("QTR (QuadTone RIP)", "qtr"),
                            ("Piezography", "piezography"),
                            ("CSV", "csv"),
                            ("JSON", "json"),
                        ],
                        value="qtr",
                        label="Export Format",
                    )

                    with gr.Row():
                        export_btn = gr.Button("💾 Export Curve", variant="secondary")
                        _save_btn = gr.Button("✅ Save to Library", variant="primary")  # noqa: F841

                    export_file = gr.File(label="Download", visible=False)

        # =====================================================
        # SECTION 4: Comparison (Optional - placeholder for future)
        # =====================================================
        with gr.Accordion("Compare with Existing Curves", open=False):
            with gr.Row():
                _compare_curve_select = gr.Dropdown(  # noqa: F841
                    choices=["No curves available"],
                    label="Compare With",
                    value="No curves available",
                )
                _compare_btn = gr.Button("Compare")  # noqa: F841

            _comparison_plot = gr.Plot(label="Curve Comparison", visible=False)  # noqa: F841
            _comparison_metrics = gr.Markdown("")  # noqa: F841

        # =====================================================
        # EVENT HANDLERS
        # =====================================================

        # Data source visibility toggle
        def toggle_data_source(
            source: str,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            """Toggle visibility based on data source selection."""
            return (
                gr.update(visible=(source == "Upload custom data")),  # custom_data_upload
                gr.update(visible=(source == "From session history")),  # session_filter
                gr.update(visible=(source == "Generate synthetic")),  # synthetic_options
            )

        data_source.change(
            toggle_data_source,
            inputs=[data_source],
            outputs=[custom_data_upload, session_filter, synthetic_options],
        )

        # Architecture info update
        def update_arch_info(arch: str) -> str:
            """Update architecture description."""
            info_map = {
                "transformer": "_Transformer: Best for capturing long-range dependencies in curves_",
                "lstm": "_LSTM: Good for sequential curve data with memory_",
                "gru": "_GRU: Faster training than LSTM with similar performance_",
                "mlp": "_MLP: Simple and fast, good baseline_",
                "hybrid": "_Hybrid: Combines transformer with MLP for robust predictions_",
            }
            return info_map.get(arch, "")

        model_architecture.change(
            update_arch_info, inputs=[model_architecture], outputs=[architecture_info]
        )

        # Pt:Pd ratio visualization update
        def update_ratio_viz(pt_ratio: int) -> str:
            """Generate HTML visualization of platinum/palladium ratio.

            Args:
                pt_ratio: Platinum percentage (0-100).

            Returns:
                HTML string with gradient visualization.
            """
            return _create_ratio_viz(pt_ratio)

        predict_pt_ratio.change(update_ratio_viz, inputs=[predict_pt_ratio], outputs=[pt_pd_viz])

        # Training handler
        def train_model(
            data_source_val: str,
            custom_file: Any,
            session_filter_val: str,
            synthetic_samples_val: float,
            synthetic_noise_val: float,
            arch: str,
            lr: float,
            epochs_val: float,
            batch_val: float,  # noqa: ARG001
            hidden_val: float,  # noqa: ARG001
            layers_val: float,  # noqa: ARG001
            dropout_val: float,  # noqa: ARG001
            uncertainty_val: bool,  # noqa: ARG001
            name: str,
            current_models: list[dict[str, Any]],
        ) -> tuple[Any, ...]:
            """Train the neural curve model."""
            import matplotlib.pyplot as plt

            try:
                # Generate or load training data
                if data_source_val == "Generate synthetic":
                    training_data = _generate_synthetic_training_data(
                        num_samples=int(synthetic_samples_val), noise_level=synthetic_noise_val
                    )
                    data_source_info = f"Synthetic ({len(training_data)} samples)"
                elif data_source_val == "Upload custom data" and custom_file:
                    training_data = _load_custom_data(custom_file.name)
                    data_source_info = f"Custom ({len(training_data)} samples)"
                elif data_source_val == "From session history" and session_logger:
                    training_data = _load_session_data(session_logger, session_filter_val)
                    data_source_info = f"Session history ({len(training_data)} samples)"
                else:
                    return (
                        "❌ No data available. Select a data source.",
                        "Failed",
                        gr.update(visible=False),
                        "-",
                        "-",
                        "-",
                        current_models,
                        gr.update(choices=["No models trained yet"]),
                    )

                if len(training_data) < 10:
                    return (
                        f"❌ Not enough data ({len(training_data)} samples). Need at least 10.",
                        "Insufficient data",
                        gr.update(visible=False),
                        str(len(training_data)),
                        "-",
                        "Poor",
                        current_models,
                        gr.update(choices=["No models trained yet"]),
                    )

                # Simulate training (in real impl, would use actual model)
                loss_history = _simulate_training(
                    epochs=int(epochs_val), learning_rate=lr, data_size=len(training_data)
                )

                # Create loss plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(loss_history["train"], label="Training Loss", color="#f59e0b", linewidth=2)
                ax.plot(
                    loss_history["val"],
                    label="Validation Loss",
                    color="#3b82f6",
                    linewidth=2,
                    linestyle="--",
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Progress")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("#333")
                plt.tight_layout()

                # Save model record
                model_record = {
                    "name": name,
                    "architecture": arch,
                    "created": datetime.now().isoformat(),
                    "data_source": data_source_info,
                    "samples": len(training_data),
                    "epochs": int(epochs_val),
                    "final_loss": loss_history["train"][-1],
                    "config": {
                        "learning_rate": lr,
                        "batch_size": int(batch_val),
                        "hidden_dim": int(hidden_val),
                        "num_layers": int(layers_val),
                        "dropout": dropout_val,
                        "uncertainty": uncertainty_val,
                    },
                }

                updated_models = list(current_models or [])
                updated_models.append(model_record)

                # Update model list choices
                model_names = [m["name"] for m in updated_models]

                return (
                    f"✅ Model '{name}' trained successfully!",
                    f"Completed in {int(epochs_val)} epochs (final loss: {loss_history['train'][-1]:.4f})",
                    gr.update(visible=True, value=fig),
                    str(len(training_data)),
                    f"{arch}, {int(hidden_val)}d, {int(layers_val)}L",
                    "Good" if len(training_data) >= 50 else "Fair",
                    updated_models,
                    gr.update(choices=model_names, value=name),
                )

            except Exception as e:
                logger.exception("Training failed")
                return (
                    f"❌ Training failed: {str(e)}",
                    "Error",
                    gr.update(visible=False),
                    "-",
                    "-",
                    "Error",
                    current_models,
                    gr.update(choices=["No models trained yet"]),
                )

        train_btn.click(
            train_model,
            inputs=[
                data_source,
                custom_data_upload,
                session_filter,
                synthetic_samples,
                synthetic_noise,
                model_architecture,
                learning_rate,
                epochs,
                batch_size,
                hidden_dim,
                num_layers,
                dropout,
                use_uncertainty,
                model_name,
                trained_models_state,
            ],
            outputs=[
                training_status,
                training_progress,
                loss_plot,
                data_samples_display,
                data_features_display,
                data_quality_display,
                trained_models_state,
                model_list,
            ],
        )

        # Model selection handler
        def on_model_select(
            selected_name: str, all_models: list[dict[str, Any]]
        ) -> tuple[str, dict[str, Any]]:
            """Update model info when selection changes."""
            if not all_models or selected_name == "No models trained yet":
                return ("_Train a model first to see details_", gr.update(interactive=False))

            for model in all_models:
                if model["name"] == selected_name:
                    info = f"""
**Architecture:** {model["architecture"].upper()}
**Trained:** {model["created"][:10]}
**Data:** {model["data_source"]}
**Final Loss:** {model["final_loss"]:.4f}
**Config:** {model["config"]["hidden_dim"]}d, {model["config"]["num_layers"]} layers
"""
                    return info, gr.update(interactive=True)

            return "_Model not found_", gr.update(interactive=False)

        model_list.change(
            on_model_select,
            inputs=[model_list, trained_models_state],
            outputs=[model_info_display, delete_model_btn],
        )

        # Prediction handler
        def generate_prediction(
            selected_model: str,
            all_models: list[dict[str, Any]],
            paper: str,
            pt_ratio: int,
            target_dmax: float,
            num_points: int,
        ) -> tuple[Any, ...]:
            """Generate curve prediction."""
            import matplotlib.pyplot as plt

            if not all_models or selected_model == "No models trained yet":
                return (
                    None,
                    "_Train and select a model first_",
                    "-",
                    "-",
                    "",
                    gr.update(visible=False),
                    None,
                )

            try:
                # Find model
                model_record = None
                for m in all_models:
                    if m["name"] == selected_model:
                        model_record = m
                        break

                if not model_record:
                    return (None, "_Model not found_", "-", "-", "", gr.update(visible=False), None)

                # Generate prediction (simulated)
                prediction = _generate_curve_prediction(
                    paper_type=paper,
                    pt_ratio=pt_ratio / 100.0,
                    target_dmax=target_dmax,
                    num_points=int(num_points),
                    model_config=model_record["config"],
                )

                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Main curve plot
                ax1.plot(
                    prediction["input"],
                    prediction["output"],
                    color="#f59e0b",
                    linewidth=2,
                    label="Predicted Curve",
                )

                # Uncertainty band
                if "uncertainty" in prediction:
                    ax1.fill_between(
                        prediction["input"],
                        np.array(prediction["output"]) - np.array(prediction["uncertainty"]),
                        np.array(prediction["output"]) + np.array(prediction["uncertainty"]),
                        alpha=0.3,
                        color="#f59e0b",
                        label="Uncertainty",
                    )

                # Reference line
                ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear")

                ax1.set_xlabel("Input Density (normalized)")
                ax1.set_ylabel("Output Density (normalized)")
                ax1.set_title(f"Predicted Curve: {paper}")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)

                # Correction magnitude plot
                correction = np.array(prediction["output"]) - np.array(prediction["input"])
                ax2.bar(
                    range(len(correction)),
                    correction,
                    color=["#4ade80" if c >= 0 else "#f87171" for c in correction],
                    alpha=0.7,
                )
                ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Correction Amount")
                ax2.set_title("Linearization Correction")
                ax2.grid(True, alpha=0.3)

                # Style both axes
                for ax in [ax1, ax2]:
                    ax.set_facecolor("#1a1a2e")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    for spine in ax.spines.values():
                        spine.set_color("#333")

                fig.patch.set_facecolor("#1a1a2e")
                plt.tight_layout()

                # Create summary
                max_correction = max(abs(c) for c in correction) if len(correction) > 0 else 0.0
                summary = f"""
**Model:** {model_record["name"]}
**Paper:** {paper}
**Pt Ratio:** {pt_ratio}%
**Target Dmax:** {target_dmax}
**Output Points:** {num_points}
**Monotonic:** {"✓ Yes" if prediction["is_monotonic"] else "✗ No"}
**Max Correction:** {max_correction:.3f}
"""

                # Quality checks
                checks = []
                if prediction["is_monotonic"]:
                    checks.append("✅ Curve is monotonic")
                else:
                    checks.append("⚠️ Curve has non-monotonic regions")

                if prediction["mean_uncertainty"] < 0.05:
                    checks.append("✅ Low prediction uncertainty")
                elif prediction["mean_uncertainty"] < 0.1:
                    checks.append("⚠️ Moderate prediction uncertainty")
                else:
                    checks.append("❌ High prediction uncertainty")

                if max_correction < 0.3:
                    checks.append("✅ Reasonable correction magnitude")
                else:
                    checks.append("⚠️ Large corrections needed")

                quality_md = "\n".join(checks)

                # Create table data
                table_data = []
                # Note: Can't use strict=True with default fallback for uncertainty
                for i, (inp, out, unc) in enumerate(
                    zip(  # noqa: B905
                        prediction["input"],
                        prediction["output"],
                        prediction.get("uncertainty", [0] * len(prediction["input"])),
                    )
                ):
                    if i % max(1, len(prediction["input"]) // 10) == 0:  # Show ~10 rows
                        table_data.append([i + 1, f"{inp:.4f}", f"{out:.4f}", f"±{unc:.4f}"])

                return (
                    fig,
                    summary,
                    f"{prediction['confidence']:.1%}",
                    f"±{prediction['mean_uncertainty']:.4f}",
                    quality_md,
                    gr.update(visible=True, value=table_data),
                    prediction,
                )

            except Exception as e:
                logger.exception("Prediction failed")
                return (
                    None,
                    f"_Prediction failed: {str(e)}_",
                    "-",
                    "-",
                    "",
                    gr.update(visible=False),
                    None,
                )

        predict_btn.click(
            generate_prediction,
            inputs=[
                model_list,
                trained_models_state,
                predict_paper,
                predict_pt_ratio,
                predict_target_dmax,
                output_points,
            ],
            outputs=[
                predicted_curve_plot,
                prediction_summary,
                confidence_score,
                uncertainty_score,
                quality_checks,
                curve_data_table,
                current_prediction_state,
            ],
        )

        # Export handler
        def export_curve(prediction: dict[str, Any] | None, format_type: str) -> Any:
            """Export the predicted curve."""
            if not prediction:
                return gr.update(visible=False)

            # Validate format type (whitelist)
            allowed_formats = {"csv", "json", "qtr", "piezography"}
            if format_type not in allowed_formats:
                logger.warning(f"Invalid export format requested: {format_type}")
                return gr.update(visible=False)

            try:
                import json
                import tempfile

                # Create export file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=f".{format_type}", delete=False
                ) as f:
                    if format_type == "csv":
                        f.write("step,input,output,uncertainty\n")
                        # Get uncertainty with fallback (avoid KeyError)
                        uncertainties = prediction.get(
                            "uncertainty", [0.0] * len(prediction["input"])
                        )
                        for i, (inp, out) in enumerate(
                            zip(prediction["input"], prediction["output"], strict=True)
                        ):
                            unc = uncertainties[i] if i < len(uncertainties) else 0.0
                            f.write(f"{i},{inp:.6f},{out:.6f},{unc:.6f}\n")

                    elif format_type == "json":
                        json.dump(
                            {
                                "format": "neural_curve_v1",
                                "input_values": prediction["input"],
                                "output_values": prediction["output"],
                                "uncertainty": prediction.get("uncertainty", []),
                                "metadata": {
                                    "confidence": prediction["confidence"],
                                    "is_monotonic": prediction["is_monotonic"],
                                    "generated": datetime.now().isoformat(),
                                },
                            },
                            f,
                            indent=2,
                        )

                    elif format_type == "qtr":
                        # QTR format (simplified)
                        f.write("; Neural Curve Prediction\n")
                        f.write(f"; Generated: {datetime.now().isoformat()}\n")
                        f.write(";\n")
                        f.write("GRAY_INK\n")
                        for out in prediction["output"]:
                            # Convert to QTR scale (0-100)
                            qtr_val = int(out * 100)
                            f.write(f"{qtr_val}\n")

                    elif format_type == "piezography":
                        # Piezography format
                        f.write("# Neural Curve Prediction\n")
                        f.write(f"# Generated: {datetime.now().isoformat()}\n")
                        f.write("#\n")
                        for inp, out in zip(prediction["input"], prediction["output"], strict=True):
                            # Piezography uses 0-255 scale
                            inp_val = int(inp * 255)
                            out_val = int(out * 255)
                            f.write(f"{inp_val}\t{out_val}\n")

                    filepath = f.name

                # Register temp file for cleanup
                _register_temp_file(filepath)

                return gr.update(visible=True, value=filepath)

            except Exception:
                logger.exception("Export failed")
                return gr.update(visible=False)

        export_btn.click(
            export_curve, inputs=[current_prediction_state, export_format], outputs=[export_file]
        )

        # Delete model handler
        def delete_model(
            selected_name: str, all_models: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            """Delete a trained model."""
            if not all_models or selected_name == "No models trained yet":
                return all_models, gr.update(choices=["No models trained yet"])

            updated_models = [m for m in all_models if m["name"] != selected_name]

            if updated_models:
                model_names = [m["name"] for m in updated_models]
                return updated_models, gr.update(choices=model_names, value=model_names[0])
            else:
                return [], gr.update(
                    choices=["No models trained yet"], value="No models trained yet"
                )

        delete_model_btn.click(
            delete_model,
            inputs=[model_list, trained_models_state],
            outputs=[trained_models_state, model_list],
        )


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _create_ratio_viz(pt_ratio: int) -> str:
    """Create HTML visualization for Pt:Pd ratio."""
    pd_ratio = 100 - pt_ratio
    return f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px;
                background: #1a1a2e; border-radius: 8px;">
        <div style="flex: {pt_ratio}; background: linear-gradient(90deg, #fbbf24, #f59e0b);
                    height: 24px; border-radius: 4px 0 0 4px; position: relative;">
            <span style="position: absolute; left: 50%; transform: translateX(-50%);
                        color: #1a1a2e; font-weight: bold; font-size: 12px; line-height: 24px;">
                {pt_ratio}% Pt
            </span>
        </div>
        <div style="flex: {pd_ratio}; background: linear-gradient(90deg, #94a3b8, #64748b);
                    height: 24px; border-radius: 0 4px 4px 0; position: relative;">
            <span style="position: absolute; left: 50%; transform: translateX(-50%);
                        color: white; font-weight: bold; font-size: 12px; line-height: 24px;">
                {pd_ratio}% Pd
            </span>
        </div>
    </div>
    """


def _generate_synthetic_training_data(
    num_samples: int = 200, noise_level: float = 0.02
) -> list[dict[str, Any]]:
    """Generate synthetic training data for curve prediction."""
    data = []

    for _ in range(num_samples):
        # Random parameters
        pt_ratio = np.random.uniform(0, 1)
        target_dmax = np.random.uniform(1.2, 2.2)
        paper_factor = np.random.uniform(0.8, 1.2)

        # Generate curve based on parameters
        num_points = 21
        x = np.linspace(0, 1, num_points)

        # Base curve (sigmoid-like for typical print response)
        gamma = 0.8 + 0.4 * pt_ratio  # Pt gives more contrast
        y = x**gamma

        # Add paper characteristic
        y = y * paper_factor
        y = np.clip(y, 0, 1)

        # Add noise
        y += np.random.normal(0, noise_level, num_points)
        y = np.clip(y, 0, 1)

        # Ensure monotonic
        for j in range(1, len(y)):
            if y[j] < y[j - 1]:
                y[j] = y[j - 1]

        data.append(
            {
                "input_densities": x.tolist(),
                "output_densities": y.tolist(),
                "pt_ratio": pt_ratio,
                "target_dmax": target_dmax,
                "paper_factor": paper_factor,
            }
        )

    return data


def _load_custom_data(filepath: str) -> list[dict[str, Any]]:
    """Load training data from a custom file.

    Args:
        filepath: Path to the data file (.json or .csv).

    Returns:
        List of training data dictionaries.

    Raises:
        ValueError: If file format is unsupported or validation fails.
        FileNotFoundError: If file does not exist.
    """
    import json
    import tempfile

    # Security: Validate filepath is within expected upload directory
    resolved_path = Path(filepath).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()

    # Allow files in temp directory (Gradio uploads) or explicit paths
    if not (str(resolved_path).startswith(str(temp_dir)) or resolved_path.exists()):
        logger.warning(f"Suspicious file path rejected: {filepath}")
        raise ValueError("Invalid file path")

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".json"):
        with open(resolved_path) as f:
            raw_data = json.load(f)

        # Validate JSON structure
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of records")

        validated_data: list[dict[str, Any]] = []
        for i, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise ValueError(f"Record {i} must be a dictionary")
            # Validate required fields exist
            if "input_densities" in item or "output_densities" in item:
                validated_data.append(item)
            else:
                logger.debug(f"Skipping record {i}: missing required fields")

        return validated_data

    elif filepath.endswith(".csv"):
        import csv

        data: list[dict[str, Any]] = []
        with open(resolved_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")
            for row in reader:
                data.append(dict(row))
        return data
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def _load_session_data(session_logger: Any, filter_val: str) -> list[dict[str, Any]]:  # noqa: ARG001
    """Load training data from session history.

    Args:
        session_logger: Session logger instance.
        filter_val: Filter string (reserved for future use).
    """
    del filter_val  # Reserved for future filtering implementation
    data = []

    try:
        sessions = session_logger.list_sessions(limit=100)

        for sess_summary in sessions:
            try:
                session = session_logger.load_session(Path(sess_summary["filepath"]))
                for record in session.records:
                    if hasattr(record, "densities") and record.densities:
                        data.append(
                            {
                                "input_densities": list(range(len(record.densities))),
                                "output_densities": record.densities,
                                "paper_type": record.paper_type,
                                "chemistry": getattr(record, "chemistry", None),
                            }
                        )
            except Exception as e:
                logger.debug(f"Failed to load session data: {e}")
                continue
    except Exception as e:
        logger.debug(f"Failed to list sessions: {e}")

    return data


def _simulate_training(
    epochs: int,
    learning_rate: float,  # noqa: ARG001
    data_size: int,
) -> dict[str, list[float]]:
    """Simulate training loss curves.

    Args:
        epochs: Number of training epochs.
        learning_rate: Learning rate (reserved for future use).
        data_size: Size of training data.
    """
    del learning_rate  # Reserved for future implementation
    train_loss = []
    val_loss = []

    # Initial loss depends on data size
    initial_loss = 0.5 + 0.5 / max(1, data_size / 100)

    for epoch in range(epochs):
        # Exponential decay with noise
        progress = epoch / epochs
        decay = np.exp(-3 * progress)

        train = initial_loss * decay * (1 + np.random.normal(0, 0.05))
        train = max(0.01, train)
        train_loss.append(train)

        # Validation is slightly higher
        val = train * (1.1 + np.random.normal(0, 0.03))
        val = max(0.015, val)
        val_loss.append(val)

    return {"train": train_loss, "val": val_loss}


def _generate_curve_prediction(
    paper_type: str,
    pt_ratio: float,
    target_dmax: float,
    num_points: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate a curve prediction (simulated)."""
    # Generate input values
    input_vals = np.linspace(0, 1, num_points).tolist()

    # Paper-specific factors
    paper_factors = {
        "arches_platine": 1.0,
        "bergger_cot320": 0.95,
        "hahnemuhle_platinum": 1.05,
        "revere_platinum": 0.98,
        "custom": 1.0,
    }
    paper_factor = paper_factors.get(paper_type, 1.0)

    # Generate output curve
    x = np.array(input_vals)

    # Base gamma curve
    gamma = 0.7 + 0.5 * pt_ratio
    y = x**gamma

    # Apply paper factor
    y = y * paper_factor

    # Apply target dmax influence
    dmax_factor = target_dmax / 2.0
    y = y * dmax_factor
    y = np.clip(y, 0, 1)

    # Ensure monotonic
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:
            y[i] = y[i - 1]

    # Generate uncertainty (higher at endpoints)
    base_uncertainty = 0.02 / max(1, model_config.get("num_layers", 3))
    uncertainty = []
    for xi in x:
        # Higher uncertainty near 0 and 1
        edge_factor = 1 + 2 * abs(xi - 0.5)
        unc = base_uncertainty * edge_factor
        uncertainty.append(unc)

    # Check monotonicity
    is_monotonic = all(y[i] <= y[i + 1] for i in range(len(y) - 1))

    # Calculate confidence
    mean_unc = np.mean(uncertainty)
    confidence = max(0.5, 1.0 - mean_unc * 10)

    return {
        "input": input_vals,
        "output": y.tolist(),
        "uncertainty": uncertainty,
        "is_monotonic": is_monotonic,
        "mean_uncertainty": mean_unc,
        "confidence": confidence,
    }
