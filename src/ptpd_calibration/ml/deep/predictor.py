"""
High-level interface for deep learning curve prediction.

Provides a user-friendly API for training and using deep learning
models to predict tone curves from process parameters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ptpd_calibration.ml.deep.exceptions import (
    CheckpointError,
    ModelNotTrainedError,
)

if TYPE_CHECKING:
    from ptpd_calibration.config import DeepLearningSettings
    from ptpd_calibration.core.models import CalibrationRecord, CurveData
    from ptpd_calibration.ml.database import CalibrationDatabase
    from ptpd_calibration.ml.deep.dataset import FeatureEncoder

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep curve prediction. "
            "Install with: pip install ptpd-calibration[deep]"
        )


@dataclass
class PredictionResult:
    """Result from curve prediction."""

    curve: np.ndarray  # Predicted curve values (LUT)
    control_points: np.ndarray | None = None  # Control points if available
    uncertainty: float | None = None  # Prediction uncertainty
    confidence: float | None = None  # Confidence score (0-1)
    metadata: dict | None = None  # Additional metadata


class DeepCurvePredictor:
    """
    High-level predictor using deep learning models.

    Provides a simple API for:
    - Training models on calibration data
    - Predicting curves from process parameters
    - Ensemble predictions with uncertainty
    - Model persistence
    """

    def __init__(
        self,
        settings: DeepLearningSettings | None = None,
        device: str | None = None,
    ):
        """
        Initialize DeepCurvePredictor.

        Args:
            settings: Deep learning settings.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps').
        """
        _check_torch()

        from ptpd_calibration.config import get_settings
        from ptpd_calibration.ml.deep.training import get_device

        self.settings = settings or get_settings().deep_learning
        self.device = get_device(device or self.settings.device)

        # Model and encoder storage
        self.model: nn.Module | None = None
        self.encoder: FeatureEncoder | None = None
        self.is_trained = False

        # Ensemble models for uncertainty estimation
        self.ensemble_models: list[nn.Module] = []

        # Training history
        self.training_history: dict | None = None

    def train(
        self,
        database: CalibrationDatabase,
        val_ratio: float = 0.2,
        num_epochs: int | None = None,
        callbacks: list | None = None,
    ) -> dict:
        """
        Train the model on calibration data.

        Args:
            database: CalibrationDatabase with training records.
            val_ratio: Fraction of data for validation.
            num_epochs: Number of training epochs.
            callbacks: Optional training callbacks.

        Returns:
            Training statistics.
        """
        from ptpd_calibration.ml.deep.dataset import (
            DataAugmentation,
            create_dataloaders,
        )
        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import CurveTrainer

        logger.info("Starting deep learning training...")

        # Create dataloaders
        augmentation = DataAugmentation(
            enabled=self.settings.augmentation_enabled,
            noise_std=self.settings.noise_std,
            exposure_jitter=self.settings.exposure_jitter,
        )

        train_loader, val_loader, encoder = create_dataloaders(
            database=database,
            batch_size=self.settings.batch_size,
            val_ratio=val_ratio,
            target_length=self.settings.lut_size,
            augmentation=augmentation,
        )

        self.encoder = encoder
        logger.info(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
        logger.info(f"Features: {encoder.num_features}")

        # Create model
        self.model = CurveMLP.from_settings(
            num_features=encoder.num_features,
            settings=self.settings,
        )
        logger.info(f"Model: {type(self.model).__name__}")

        # Create trainer
        trainer = CurveTrainer(
            model=self.model,
            settings=self.settings,
            device=str(self.device),
        )

        # Add callbacks
        if callbacks:
            for cb in callbacks:
                trainer.add_callback(cb)

        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=self.settings.checkpoint_dir,
        )

        self.training_history = history.to_dict()
        self.is_trained = True

        # Evaluate final model
        eval_metrics = trainer.evaluate(val_loader)
        logger.info(f"Final validation - MAE: {eval_metrics['mae']:.4f}")

        # Train ensemble if enabled
        if self.settings.use_ensemble:
            self._train_ensemble(database, val_ratio, num_epochs)

        return {
            "num_samples": len(train_loader.dataset) + len(val_loader.dataset),
            "num_features": encoder.num_features,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "total_time": history.total_time,
            "final_metrics": eval_metrics,
        }

    def _train_ensemble(
        self,
        database: CalibrationDatabase,
        val_ratio: float,
        num_epochs: int | None,
    ) -> None:
        """Train ensemble of models for uncertainty estimation."""
        from ptpd_calibration.ml.deep.dataset import DataAugmentation, create_dataloaders
        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import CurveTrainer

        logger.info(f"Training ensemble of {self.settings.num_ensemble_models} models...")

        augmentation = DataAugmentation(
            enabled=self.settings.augmentation_enabled,
            noise_std=self.settings.noise_std,
            exposure_jitter=self.settings.exposure_jitter,
        )

        self.ensemble_models = []

        for i in range(self.settings.num_ensemble_models):
            # Create new dataloaders with different random split
            train_loader, val_loader, _ = create_dataloaders(
                database=database,
                batch_size=self.settings.batch_size,
                val_ratio=val_ratio,
                target_length=self.settings.lut_size,
                augmentation=augmentation,
                seed=i * 42,  # Different seed for each model
            )

            # Create model
            model = CurveMLP.from_settings(
                num_features=self.encoder.num_features,
                settings=self.settings,
            )

            # Train
            trainer = CurveTrainer(
                model=model,
                settings=self.settings,
                device=str(self.device),
            )

            # Use fewer epochs for ensemble members
            ensemble_epochs = (num_epochs or self.settings.num_epochs) // 2
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=ensemble_epochs,
            )

            self.ensemble_models.append(model)
            logger.info(f"Ensemble model {i + 1}/{self.settings.num_ensemble_models} trained")

    def predict(
        self,
        record: CalibrationRecord,
        return_uncertainty: bool = True,
    ) -> PredictionResult:
        """
        Predict tone curve for a calibration setup.

        Args:
            record: CalibrationRecord with process parameters.
            return_uncertainty: Whether to compute uncertainty.

        Returns:
            PredictionResult with predicted curve.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before prediction")

        # Encode features
        features = self.encoder.encode(record)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        # Predict with main model
        self.model.eval()
        with torch.no_grad():
            lut, control_points = self.model(features_tensor, return_control_points=True)

        curve = lut[0].cpu().numpy()
        cp = control_points[0].cpu().numpy() if control_points is not None else None

        # Compute uncertainty from ensemble
        uncertainty = None
        confidence = None
        if return_uncertainty and self.ensemble_models:
            uncertainty, confidence = self._compute_uncertainty(features_tensor)

        return PredictionResult(
            curve=curve,
            control_points=cp,
            uncertainty=uncertainty,
            confidence=confidence,
            metadata={
                "model_type": type(self.model).__name__,
                "num_features": self.encoder.num_features,
                "lut_size": len(curve),
            },
        )

    def _compute_uncertainty(
        self,
        features: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute uncertainty from ensemble predictions."""
        predictions = []

        for model in self.ensemble_models:
            model.eval()
            with torch.no_grad():
                pred, _ = model(features)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Uncertainty is standard deviation across ensemble
        uncertainty = float(np.mean(np.std(predictions, axis=0)))

        # Confidence is inverse of uncertainty (normalized)
        max_uncertainty = 0.2  # Typical max uncertainty
        confidence = float(max(0.0, 1.0 - uncertainty / max_uncertainty))

        return uncertainty, confidence

    def predict_with_mc_dropout(
        self,
        record: CalibrationRecord,
        num_samples: int | None = None,
    ) -> PredictionResult:
        """
        Predict with Monte Carlo dropout for uncertainty.

        Args:
            record: CalibrationRecord with process parameters.
            num_samples: Number of MC samples.

        Returns:
            PredictionResult with uncertainty from MC dropout.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before prediction")

        num_samples = num_samples or self.settings.mc_dropout_samples

        # Encode features
        features = self.encoder.encode(record)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        # Enable dropout for MC sampling
        self.model.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                lut, _ = self.model(features_tensor)
                predictions.append(lut[0].cpu().numpy())

        self.model.eval()  # Disable dropout

        predictions = np.array(predictions)
        mean_curve = np.mean(predictions, axis=0)
        uncertainty = float(np.mean(np.std(predictions, axis=0)))

        max_uncertainty = 0.2
        confidence = float(max(0.0, 1.0 - uncertainty / max_uncertainty))

        return PredictionResult(
            curve=mean_curve,
            uncertainty=uncertainty,
            confidence=confidence,
            metadata={
                "model_type": type(self.model).__name__,
                "mc_samples": num_samples,
            },
        )

    def to_curve_data(
        self,
        prediction: PredictionResult,
        name: str = "DL Predicted Curve",
        paper_type: str | None = None,
        chemistry: str | None = None,
    ) -> CurveData:
        """
        Convert prediction to CurveData model.

        Args:
            prediction: Prediction result.
            name: Curve name.
            paper_type: Paper type metadata.
            chemistry: Chemistry type metadata.

        Returns:
            CurveData instance.
        """
        from ptpd_calibration.core.models import CurveData
        from ptpd_calibration.core.types import CurveType

        input_values = list(np.linspace(0, 1, len(prediction.curve)))
        output_values = list(prediction.curve)

        return CurveData(
            name=name,
            curve_type=CurveType.CUSTOM,
            input_values=input_values,
            output_values=output_values,
            paper_type=paper_type,
            chemistry=chemistry,
            notes=f"Deep learning prediction (confidence: {prediction.confidence:.2%})"
            if prediction.confidence
            else "Deep learning prediction",
        )

    def save(self, path: Path) -> None:
        """
        Save the predictor to disk.

        Args:
            path: Directory to save to.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Save main model
            torch.save(self.model.state_dict(), path / "model.pt")

            # Save encoder
            encoder_data = self.encoder.to_dict()
            with open(path / "encoder.json", "w") as f:
                json.dump(encoder_data, f, indent=2)

            # Save ensemble models
            if self.ensemble_models:
                ensemble_dir = path / "ensemble"
                ensemble_dir.mkdir(exist_ok=True)
                for i, model in enumerate(self.ensemble_models):
                    torch.save(model.state_dict(), ensemble_dir / f"model_{i}.pt")

            # Save metadata
            metadata = {
                "model_type": self.settings.model_type,
                "num_features": self.encoder.num_features,
                "lut_size": self.settings.lut_size,
                "num_control_points": self.settings.num_control_points,
                "hidden_dims": self.settings.hidden_dims,
                "num_ensemble_models": len(self.ensemble_models),
                "training_history": self.training_history,
            }
            with open(path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved predictor to {path}")

        except Exception as e:
            raise CheckpointError(f"Failed to save predictor: {e}") from e

    @classmethod
    def load(
        cls,
        path: Path,
        device: str | None = None,
    ) -> DeepCurvePredictor:
        """
        Load a predictor from disk.

        Args:
            path: Directory to load from.
            device: Device to load to.

        Returns:
            Loaded DeepCurvePredictor.
        """
        _check_torch()

        from ptpd_calibration.config import get_settings
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder
        from ptpd_calibration.ml.deep.models import CurveMLP
        from ptpd_calibration.ml.deep.training import get_device

        path = Path(path)

        if not path.exists():
            raise CheckpointError(f"Checkpoint directory not found: {path}")

        try:
            # Load metadata
            with open(path / "metadata.json") as f:
                metadata = json.load(f)

            # Load encoder
            with open(path / "encoder.json") as f:
                encoder_data = json.load(f)
            encoder = FeatureEncoder.from_dict(encoder_data)

            # Create predictor
            settings = get_settings().deep_learning
            predictor = cls(settings=settings, device=device)
            predictor.encoder = encoder
            predictor.device = get_device(device or settings.device)

            # Create and load model
            predictor.model = CurveMLP(
                num_features=metadata["num_features"],
                num_control_points=metadata["num_control_points"],
                lut_size=metadata["lut_size"],
                hidden_dims=metadata["hidden_dims"],
            )
            predictor.model.load_state_dict(
                torch.load(path / "model.pt", map_location=predictor.device)
            )
            predictor.model.to(predictor.device)
            predictor.model.eval()

            # Load ensemble models
            ensemble_dir = path / "ensemble"
            if ensemble_dir.exists():
                predictor.ensemble_models = []
                for i in range(metadata["num_ensemble_models"]):
                    model = CurveMLP(
                        num_features=metadata["num_features"],
                        num_control_points=metadata["num_control_points"],
                        lut_size=metadata["lut_size"],
                        hidden_dims=metadata["hidden_dims"],
                    )
                    model.load_state_dict(
                        torch.load(ensemble_dir / f"model_{i}.pt", map_location=predictor.device)
                    )
                    model.to(predictor.device)
                    model.eval()
                    predictor.ensemble_models.append(model)

            predictor.training_history = metadata.get("training_history")
            predictor.is_trained = True

            logger.info(f"Loaded predictor from {path}")
            return predictor

        except Exception as e:
            raise CheckpointError(f"Failed to load predictor: {e}") from e

    def suggest_adjustments(
        self,
        record: CalibrationRecord,
        target_curve: np.ndarray,
    ) -> dict:
        """
        Suggest parameter adjustments to achieve target curve.

        Args:
            record: Current calibration setup.
            target_curve: Desired curve shape.

        Returns:
            Dictionary with adjustment suggestions.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before making suggestions")

        # Get current prediction
        current = self.predict(record, return_uncertainty=False)

        # Compare curves
        diff = target_curve - current.curve

        suggestions = []
        adjustments = {}

        # Analyze overall brightness
        mean_diff = np.mean(diff)
        if abs(mean_diff) > 0.05:
            if mean_diff > 0:
                suggestions.append("Increase exposure time for brighter output")
                adjustments["exposure_factor"] = 1.1 + abs(mean_diff)
            else:
                suggestions.append("Decrease exposure time for darker output")
                adjustments["exposure_factor"] = 0.9 - abs(mean_diff)

        # Analyze contrast
        target_range = np.max(target_curve) - np.min(target_curve)
        current_range = np.max(current.curve) - np.min(current.curve)
        range_diff = target_range - current_range

        if abs(range_diff) > 0.1:
            if range_diff > 0:
                suggestions.append("Increase platinum ratio for higher contrast")
                adjustments["metal_ratio_delta"] = min(0.2, range_diff)
            else:
                suggestions.append("Decrease platinum ratio for lower contrast")
                adjustments["metal_ratio_delta"] = max(-0.2, range_diff)

        # Analyze shadow detail
        shadow_diff = np.mean(diff[: len(diff) // 4])
        if abs(shadow_diff) > 0.05:
            if shadow_diff > 0:
                suggestions.append("Add contrast agent for more shadow detail")
                adjustments["contrast_increase"] = True
            else:
                suggestions.append("Reduce contrast agent for less shadow separation")
                adjustments["contrast_decrease"] = True

        return {
            "current_curve": current.curve.tolist(),
            "target_curve": target_curve.tolist(),
            "difference": diff.tolist(),
            "suggestions": suggestions,
            "adjustments": adjustments,
            "confidence": current.confidence,
        }
