#!/usr/bin/env python3
"""
Deep Learning Usage Examples for Platinum-Palladium AI Printing Tool

This script demonstrates how to use the new deep learning features:
1. Diffusion Model Enhancement (DiffusionEnhancer)
2. Neural Curve Prediction (NeuralCurvePredictor)

Requirements:
    pip install torch torchvision numpy pillow diffusers>=0.25.0

Note: All examples use configuration-driven parameters with no hardcoded values.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


# =============================================================================
# Example 1: Diffusion Model Enhancement
# =============================================================================


def example_diffusion_tonal_enhancement():
    """Example: Enhance tonal range of a platinum-palladium print."""
    print("=" * 70)
    print("EXAMPLE 1: Diffusion Tonal Enhancement")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import DiffusionSettings
    from ptpd_calibration.deep_learning.diffusion_enhance import DiffusionEnhancer

    # Create settings (can be loaded from environment or .env file)
    settings = DiffusionSettings(
        model_type="sdxl",
        scheduler="euler",
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.5,
        use_controlnet=True,
        controlnet_type="canny",
        enable_vae_slicing=True,
        device="auto",  # Will auto-detect CUDA/MPS/CPU
    )

    # Initialize enhancer
    print("\nInitializing DiffusionEnhancer...")
    enhancer = DiffusionEnhancer(settings)

    # Create a sample image (replace with actual image loading)
    print("Creating sample image...")
    sample_image = np.random.rand(512, 512) * 0.8 + 0.1  # Grayscale image

    # Enhance tonal range
    print("\nEnhancing tonal range...")
    result = enhancer.enhance(
        image=sample_image,
        prompt="platinum palladium print, enhanced tonal range, rich shadows, delicate highlights",
        negative_prompt="blurry, noisy, harsh contrast",
        strength=0.5,
        use_controlnet=True,
    )

    print(f"\n✓ Enhancement complete!")
    print(f"  - Inference time: {result.inference_time_ms:.2f} ms")
    print(f"  - Device used: {result.device_used}")
    print(f"  - Original size: {result.original_size}")
    print(f"  - Output size: {result.output_size}")
    print(f"  - Structure preservation: {result.structure_preservation:.2%}")
    print(f"  - Tone fidelity: {result.tone_fidelity:.2%}")

    # Clean up to free memory
    enhancer.cleanup()

    print("\n" + "=" * 70 + "\n")


def example_diffusion_inpainting():
    """Example: Remove defects using inpainting."""
    print("=" * 70)
    print("EXAMPLE 2: Diffusion Inpainting (Defect Removal)")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import DiffusionSettings
    from ptpd_calibration.deep_learning.diffusion_enhance import DiffusionEnhancer

    # Settings for inpainting
    settings = DiffusionSettings(
        num_inference_steps=50,  # More steps for better inpainting
        guidance_scale=7.5,
        inpaint_mask_blur=4,
        inpaint_mask_padding=32,
        device="auto",
    )

    print("\nInitializing DiffusionEnhancer...")
    enhancer = DiffusionEnhancer(settings)

    # Create sample image and defect mask
    print("Creating sample image with defects...")
    sample_image = np.random.rand(512, 512) * 0.8 + 0.1

    # Create a defect mask (white = inpaint, black = keep)
    defect_mask = np.zeros((512, 512), dtype=np.uint8)
    defect_mask[100:150, 200:250] = 255  # Defect region

    # Inpaint defects
    print("\nInpainting defects...")
    result = enhancer.inpaint(
        image=sample_image,
        mask=defect_mask,
        prompt="seamless platinum palladium print, perfect coating, archival quality",
        negative_prompt="defects, spots, stains, damage",
    )

    print(f"\n✓ Inpainting complete!")
    print(f"  - Inference time: {result.inference_time_ms:.2f} ms")
    print(f"  - Regions enhanced: {len(result.regions_enhanced)}")
    print(f"  - Structure preservation: {result.structure_preservation:.2%}")

    enhancer.cleanup()

    print("\n" + "=" * 70 + "\n")


def example_diffusion_style_transfer():
    """Example: Apply master printer aesthetic."""
    print("=" * 70)
    print("EXAMPLE 3: Diffusion Style Transfer")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import DiffusionSettings
    from ptpd_calibration.deep_learning.diffusion_enhance import DiffusionEnhancer

    # Settings with custom LoRA for Pt/Pd aesthetic
    settings = DiffusionSettings(
        num_inference_steps=40,
        guidance_scale=8.0,
        strength=0.6,
        style_prompt_template="platinum palladium print, {style}, high quality, detailed tones",
        use_custom_lora=False,  # Set to True if you have custom LoRA weights
        # lora_weights_path="/path/to/ptpd_lora.safetensors",
        device="auto",
    )

    print("\nInitializing DiffusionEnhancer...")
    enhancer = DiffusionEnhancer(settings)

    # Sample image
    print("Creating sample image...")
    sample_image = np.random.rand(512, 512) * 0.8 + 0.1

    # Apply master printer style
    print("\nApplying Edward Weston aesthetic...")
    result = enhancer.style_transfer(
        image=sample_image,
        style="Edward Weston platinum palladium aesthetic, deep blacks, luminous highlights",
        strength=0.6,
    )

    print(f"\n✓ Style transfer complete!")
    print(f"  - Inference time: {result.inference_time_ms:.2f} ms")
    print(f"  - Prompt used: {result.prompt_used}")
    print(f"  - Structure preservation: {result.structure_preservation:.2%}")
    print(f"  - Tone fidelity: {result.tone_fidelity:.2%}")

    enhancer.cleanup()

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# Example 2: Neural Curve Prediction
# =============================================================================


def example_neural_curve_prediction():
    """Example: Predict tonal curve with uncertainty quantification."""
    print("=" * 70)
    print("EXAMPLE 4: Neural Curve Prediction")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import NeuralCurveSettings
    from ptpd_calibration.deep_learning.neural_curve import NeuralCurvePredictor

    # Create settings
    settings = NeuralCurveSettings(
        architecture="transformer",
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        output_points=256,
        uncertainty_method="ensemble",
        ensemble_size=5,
        include_conditioning=True,
        device="auto",
    )

    print("\nInitializing NeuralCurvePredictor...")
    predictor = NeuralCurvePredictor(settings)

    # Define input curve points
    print("Defining input curve points...")
    input_values = np.linspace(0, 1, 256)

    # Define process conditioning
    conditioning = {
        "paper_type": "Arches Platine",
        "metal_ratio": 0.5,  # 50% platinum, 50% palladium
        "exposure_time": 300,  # seconds
        "humidity": 0.45,  # 45% RH
        "temperature": 20,  # 20°C
    }

    # Predict curve
    print("\nPredicting tonal curve...")
    print(f"  Conditioning: {conditioning}")

    result = predictor.predict(
        input_values=input_values,
        conditioning=conditioning,
        return_uncertainty=True,
    )

    print(f"\n✓ Prediction complete!")
    print(f"  - Inference time: {result.inference_time_ms:.2f} ms")
    print(f"  - Device used: {result.device_used}")
    print(f"  - Number of points: {result.num_points}")
    print(f"  - Is monotonic: {result.is_monotonic}")
    print(f"  - Mean uncertainty: {result.mean_uncertainty:.4f}")
    print(f"  - Confidence: {result.confidence:.2%}")
    print(f"  - Slope range: [{result.min_slope:.3f}, {result.max_slope:.3f}]")

    # Display curve samples
    print("\n  Sample curve points:")
    for i in [0, 64, 128, 192, 255]:
        uncertainty_str = f" ± {result.uncertainty[i]:.4f}" if result.uncertainty else ""
        print(f"    Input {result.input_values[i]:.3f} → Output {result.output_values[i]:.3f}{uncertainty_str}")

    predictor.cleanup()

    print("\n" + "=" * 70 + "\n")


def example_neural_curve_training():
    """Example: Train a neural curve predictor."""
    print("=" * 70)
    print("EXAMPLE 5: Neural Curve Training")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import NeuralCurveSettings
    from ptpd_calibration.deep_learning.neural_curve import NeuralCurvePredictor

    # Create settings
    settings = NeuralCurveSettings(
        architecture="transformer",
        d_model=128,
        n_heads=4,
        n_layers=4,
        learning_rate=1e-4,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        loss_function="monotonic_mse",
        monotonicity_weight=0.1,
        smoothness_weight=0.05,
        device="auto",
    )

    print("\nInitializing NeuralCurvePredictor...")
    predictor = NeuralCurvePredictor(settings)

    # Generate synthetic training data
    print("Generating synthetic training data...")
    num_samples = 1000
    num_points = 64
    num_features = settings.input_features

    # Training features: (num_samples, num_points, num_features)
    X_train = np.random.rand(num_samples, num_points, num_features).astype(np.float32)

    # Training targets: (num_samples, num_points)
    # Generate monotonic curves
    y_train = np.sort(np.random.rand(num_samples, num_points).astype(np.float32), axis=1)

    # Validation data
    X_val = np.random.rand(200, num_points, num_features).astype(np.float32)
    y_val = np.sort(np.random.rand(200, num_points).astype(np.float32), axis=1)

    # Train model
    print("\nTraining model...")
    print(f"  Training samples: {num_samples}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Epochs: {settings.epochs}")
    print(f"  Batch size: {settings.batch_size}")

    history = predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_best=False,  # Set to True to save checkpoints
    )

    print(f"\n✓ Training complete!")
    print(f"  - Final training loss: {history['loss'][-1]:.4f}")
    print(f"  - Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"  - Total epochs: {len(history['loss'])}")

    predictor.cleanup()

    print("\n" + "=" * 70 + "\n")


def example_neural_curve_ensemble():
    """Example: Train ensemble for uncertainty quantification."""
    print("=" * 70)
    print("EXAMPLE 6: Neural Curve Ensemble Training")
    print("=" * 70)

    from ptpd_calibration.deep_learning.config import NeuralCurveSettings
    from ptpd_calibration.deep_learning.neural_curve import NeuralCurvePredictor

    # Settings for ensemble
    settings = NeuralCurveSettings(
        architecture="transformer",
        d_model=128,
        n_heads=4,
        n_layers=3,
        uncertainty_method="ensemble",
        ensemble_size=3,  # Small ensemble for example
        epochs=50,
        batch_size=32,
        device="auto",
    )

    print("\nInitializing NeuralCurvePredictor...")
    predictor = NeuralCurvePredictor(settings)

    # Generate training data
    print("Generating training data...")
    num_samples = 500
    num_points = 64

    X_train = np.random.rand(num_samples, num_points, settings.input_features).astype(np.float32)
    y_train = np.sort(np.random.rand(num_samples, num_points).astype(np.float32), axis=1)

    # Train ensemble
    print(f"\nTraining ensemble of {settings.ensemble_size} models...")
    histories = predictor.train_ensemble(
        X_train=X_train,
        y_train=y_train,
    )

    print(f"\n✓ Ensemble training complete!")
    for i, history in enumerate(histories):
        print(f"  Model {i + 1}: Final loss = {history['loss'][-1]:.4f}")

    # Predict with ensemble
    print("\nPredicting with ensemble...")
    input_values = np.linspace(0, 1, 64)
    result = predictor.predict(
        input_values=input_values,
        return_uncertainty=True,
    )

    print(f"\n✓ Ensemble prediction complete!")
    print(f"  - Mean uncertainty: {result.mean_uncertainty:.4f}")
    print(f"  - Confidence: {result.confidence:.2%}")

    predictor.cleanup()

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DEEP LEARNING USAGE EXAMPLES")
    print("Platinum-Palladium AI Printing Tool")
    print("=" * 70 + "\n")

    print("Note: These examples demonstrate the API without requiring")
    print("heavy dependencies. To run actual inference, install:")
    print("  pip install torch torchvision diffusers pillow\n")

    try:
        # Diffusion examples
        print("\n" + "▶" * 35)
        print("DIFFUSION MODEL ENHANCEMENT EXAMPLES")
        print("▶" * 35 + "\n")

        example_diffusion_tonal_enhancement()
        example_diffusion_inpainting()
        example_diffusion_style_transfer()

        # Neural curve examples
        print("\n" + "▶" * 35)
        print("NEURAL CURVE PREDICTION EXAMPLES")
        print("▶" * 35 + "\n")

        example_neural_curve_prediction()
        example_neural_curve_training()
        example_neural_curve_ensemble()

    except ImportError as e:
        print(f"\n⚠ Import Error: {e}")
        print("\nTo run these examples with actual inference, install:")
        print("  pip install torch>=2.0.0 diffusers>=0.25.0 pillow\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
