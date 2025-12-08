"""
Deep learning API endpoints for curve prediction.

Provides REST API endpoints for:
- Training deep learning models
- Predicting curves from process parameters
- Managing model lifecycle
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)


def determine_chemistry_type(metal_ratio: float):
    """
    Determine chemistry type from metal ratio.

    Args:
        metal_ratio: Platinum ratio (0.0 to 1.0).

    Returns:
        ChemistryType enum value.
    """
    from ptpd_calibration.core.types import ChemistryType

    if metal_ratio > 0.95:
        return ChemistryType.PURE_PLATINUM
    elif metal_ratio < 0.05:
        return ChemistryType.PURE_PALLADIUM
    else:
        return ChemistryType.PLATINUM_PALLADIUM


def create_deep_learning_router(database, model_storage: dict):
    """
    Create the deep learning API router.

    Args:
        database: CalibrationDatabase instance.
        model_storage: Dictionary to store trained models.

    Returns:
        FastAPI APIRouter with deep learning endpoints.
    """
    try:
        from fastapi import APIRouter, HTTPException, BackgroundTasks
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install ptpd-calibration[api]"
        )

    router = APIRouter(prefix="/api/deep", tags=["deep-learning"])

    # Check PyTorch availability
    try:
        import torch

        TORCH_AVAILABLE = True
        TORCH_VERSION = torch.__version__
    except ImportError:
        TORCH_AVAILABLE = False
        TORCH_VERSION = None

    # Pydantic models for request/response
    class TrainRequest(BaseModel):
        """Request to train a deep learning model."""

        model_name: str = Field(default="default", description="Name for the trained model")
        num_epochs: int = Field(default=50, ge=1, le=1000, description="Number of training epochs")
        batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")
        learning_rate: float = Field(default=1e-3, ge=1e-6, le=1.0, description="Learning rate")
        use_synthetic_data: bool = Field(
            default=False, description="Generate synthetic training data"
        )
        num_synthetic_samples: int = Field(
            default=200, ge=50, le=5000, description="Number of synthetic samples"
        )
        validation_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Validation split")
        # Advanced settings (configurable per Gemini feedback)
        use_ensemble: bool = Field(default=False, description="Use ensemble of models")
        device: str = Field(default="cpu", description="Device to train on (cpu/cuda)")
        num_control_points: int = Field(default=16, ge=4, le=64, description="Number of control points")
        hidden_dims: list[int] = Field(
            default=[128, 256, 128], description="Hidden layer dimensions"
        )
        early_stopping_patience: int = Field(
            default=10, ge=1, le=100, description="Early stopping patience"
        )

    class PredictRequest(BaseModel):
        """Request to predict a curve from process parameters."""

        model_name: str = Field(default="default", description="Name of the model to use")
        paper_type: str = Field(..., description="Paper type")
        metal_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Platinum ratio")
        exposure_time: float = Field(default=180.0, ge=1.0, description="Exposure time (seconds)")
        contrast_agent: str = Field(default="na2", description="Contrast agent type")
        contrast_amount: float = Field(default=5.0, ge=0.0, description="Contrast agent amount")
        humidity: Optional[float] = Field(default=50.0, ge=0.0, le=100.0, description="Humidity %")
        temperature: Optional[float] = Field(
            default=21.0, ge=-20.0, le=50.0, description="Temperature °C"
        )
        return_uncertainty: bool = Field(default=True, description="Return uncertainty estimate")

    class SuggestAdjustmentsRequest(BaseModel):
        """Request to get adjustment suggestions."""

        model_name: str = Field(default="default", description="Model to use")
        paper_type: str = Field(..., description="Paper type")
        metal_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
        exposure_time: float = Field(default=180.0, ge=1.0)
        target_curve: list[float] = Field(..., description="Target curve values")

    class TrainingSummary(BaseModel):
        """Summary of training progress."""

        model_name: str
        status: str
        epochs_completed: int = 0
        best_val_loss: Optional[float] = None
        training_time: Optional[float] = None
        error: Optional[str] = None

    # Training state storage
    training_status: dict[str, TrainingSummary] = {}

    @router.get("/status")
    async def get_deep_learning_status():
        """Get the status of deep learning capabilities."""
        return {
            "torch_available": TORCH_AVAILABLE,
            "torch_version": TORCH_VERSION,
            "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "models_loaded": list(model_storage.keys()),
            "database_records": len(database) if database else 0,
            "training_in_progress": [
                name for name, status in training_status.items() if status.status == "training"
            ],
        }

    @router.post("/train")
    async def train_model(
        request: TrainRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Train a deep learning model for curve prediction.

        Can use existing calibration data or generate synthetic data.
        Training runs in the background.
        """
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyTorch is not available. Install with: pip install ptpd-calibration[deep]",
            )

        # Check if already training
        if request.model_name in training_status:
            if training_status[request.model_name].status == "training":
                raise HTTPException(
                    status_code=409,
                    detail=f"Model '{request.model_name}' is already being trained",
                )

        # Initialize training status
        training_status[request.model_name] = TrainingSummary(
            model_name=request.model_name,
            status="starting",
        )

        # Start training in background
        background_tasks.add_task(
            _train_model_task,
            request,
            database,
            model_storage,
            training_status,
        )

        return {
            "success": True,
            "message": f"Training started for model '{request.model_name}'",
            "model_name": request.model_name,
            "status": "starting",
        }

    @router.get("/train/{model_name}/status")
    async def get_training_status(model_name: str):
        """Get the training status for a model."""
        if model_name not in training_status:
            raise HTTPException(status_code=404, detail=f"No training found for '{model_name}'")

        status = training_status[model_name]
        return {
            "model_name": status.model_name,
            "status": status.status,
            "epochs_completed": status.epochs_completed,
            "best_val_loss": status.best_val_loss,
            "training_time": status.training_time,
            "error": status.error,
        }

    @router.post("/predict")
    async def predict_curve(request: PredictRequest):
        """
        Predict a tone curve from process parameters.

        Requires a trained model.
        """
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyTorch is not available. Install with: pip install ptpd-calibration[deep]",
            )

        if request.model_name not in model_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found. Train a model first.",
            )

        try:
            from ptpd_calibration.core.models import CalibrationRecord
            from ptpd_calibration.core.types import ContrastAgent, DeveloperType

            predictor = model_storage[request.model_name]

            # Create calibration record from request
            contrast_agent = ContrastAgent.NA2 if request.contrast_agent.lower() == "na2" else ContrastAgent.NONE
            chemistry_type = determine_chemistry_type(request.metal_ratio)

            record = CalibrationRecord(
                paper_type=request.paper_type,
                metal_ratio=request.metal_ratio,
                exposure_time=request.exposure_time,
                contrast_agent=contrast_agent,
                contrast_amount=request.contrast_amount,
                chemistry_type=chemistry_type,
                developer=DeveloperType.POTASSIUM_OXALATE,
                humidity=request.humidity,
                temperature=request.temperature,
            )

            # Predict
            result = predictor.predict(record, return_uncertainty=request.return_uncertainty)

            return {
                "success": True,
                "curve": result.curve.tolist(),
                "control_points": result.control_points.tolist() if result.control_points is not None else None,
                "uncertainty": result.uncertainty,
                "confidence": result.confidence,
                "lut_size": len(result.curve),
            }

        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check server logs for details.",
            )

    @router.post("/suggest-adjustments")
    async def suggest_adjustments(request: SuggestAdjustmentsRequest):
        """
        Get suggestions for parameter adjustments to achieve a target curve.
        """
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyTorch is not available.",
            )

        if request.model_name not in model_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found.",
            )

        try:
            import numpy as np

            from ptpd_calibration.core.models import CalibrationRecord
            from ptpd_calibration.core.types import ContrastAgent, DeveloperType

            predictor = model_storage[request.model_name]

            # Use shared helper function to determine chemistry type (addresses Gemini feedback)
            chemistry_type = determine_chemistry_type(request.metal_ratio)

            record = CalibrationRecord(
                paper_type=request.paper_type,
                metal_ratio=request.metal_ratio,
                exposure_time=request.exposure_time,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                chemistry_type=chemistry_type,
                developer=DeveloperType.POTASSIUM_OXALATE,
            )

            target = np.array(request.target_curve)
            suggestions = predictor.suggest_adjustments(record, target)

            return {
                "success": True,
                "suggestions": suggestions["suggestions"],
                "adjustments": suggestions["adjustments"],
                "confidence": suggestions.get("confidence"),
            }

        except Exception as e:
            logger.exception("Suggest adjustments failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate suggestions. Check server logs for details.",
            )

    @router.get("/models")
    async def list_models():
        """List all available trained models."""
        models = []
        for name, predictor in model_storage.items():
            models.append({
                "name": name,
                "is_trained": predictor.is_trained,
                "num_features": predictor.encoder.num_features if predictor.encoder else None,
                "lut_size": predictor.settings.lut_size,
            })

        return {"models": models}

    @router.delete("/models/{model_name}")
    async def delete_model(model_name: str):
        """Delete a trained model."""
        if model_name not in model_storage:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        del model_storage[model_name]

        # Also remove from training status
        if model_name in training_status:
            del training_status[model_name]

        return {"success": True, "message": f"Model '{model_name}' deleted"}

    @router.post("/generate-synthetic")
    async def generate_synthetic_data(
        num_samples: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Generate synthetic calibration data and add to database.

        Useful for testing without real calibration data.
        """
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch is not available.")

        try:
            from ptpd_calibration.ml.deep.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig

            config = SyntheticDataConfig(num_records=num_samples, seed=seed)
            generator = SyntheticDataGenerator(config)

            records_added = 0
            for _ in range(num_samples):
                record = generator.generate_record()
                database.add_record(record)
                records_added += 1

            return {
                "success": True,
                "records_added": records_added,
                "total_records": len(database),
            }

        except Exception as e:
            logger.exception("Synthetic data generation failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate synthetic data. Check server logs for details.",
            )

    return router


async def _train_model_task(
    request,
    database,
    model_storage: dict,
    training_status: dict,
):
    """Background task to train a model."""
    import time

    start_time = time.time()
    model_name = request.model_name

    try:
        from ptpd_calibration.config import DeepLearningSettings
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        training_status[model_name].status = "training"

        # Prepare training data
        if request.use_synthetic_data or len(database) < 10:
            train_db = generate_training_data(
                num_records=request.num_synthetic_samples,
                seed=42,
            )
            # Also add existing records if any
            for record in database.get_all_records():
                train_db.add_record(record)
        else:
            train_db = database

        # Configure settings from request (addresses Gemini feedback on hardcoded values)
        settings = DeepLearningSettings(
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            use_ensemble=request.use_ensemble,
            device=request.device,
            num_control_points=request.num_control_points,
            lut_size=256,
            hidden_dims=request.hidden_dims,
            early_stopping_patience=request.early_stopping_patience,
        )

        # Create and train predictor
        predictor = DeepCurvePredictor(settings=settings)

        def update_callback(metrics):
            training_status[model_name].epochs_completed = metrics.epoch
            training_status[model_name].best_val_loss = metrics.val_loss

        stats = predictor.train(
            database=train_db,
            val_ratio=request.validation_split,
            num_epochs=request.num_epochs,
            callbacks=[update_callback],
        )

        # Store the trained model
        model_storage[model_name] = predictor

        training_status[model_name].status = "completed"
        training_status[model_name].training_time = time.time() - start_time
        training_status[model_name].best_val_loss = stats.get("best_val_loss")

    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")
        training_status[model_name].status = "failed"
        training_status[model_name].error = str(e)
