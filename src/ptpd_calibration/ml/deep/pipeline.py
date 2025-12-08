"""
Training pipeline for deep learning curve prediction.

Provides high-level utilities for:
- Running complete training workflows
- Hyperparameter tuning
- Model evaluation and comparison
- Experiment tracking
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from ptpd_calibration.config import DeepLearningSettings, get_settings
from ptpd_calibration.ml.database import CalibrationDatabase
from ptpd_calibration.ml.deep.exceptions import TrainingError

if TYPE_CHECKING:
    from ptpd_calibration.core.models import CalibrationRecord

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""

    name: str
    description: str = ""

    # Data settings
    train_val_split: float = 0.2
    target_lut_size: int = 256

    # Model settings
    model_type: str = "curve_mlp"
    num_control_points: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 128])
    dropout_rate: float = 0.1

    # Training settings
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # Loss weights
    mse_weight: float = 1.0
    monotonicity_weight: float = 0.1
    smoothness_weight: float = 0.05

    # Augmentation
    augmentation_enabled: bool = True
    noise_std: float = 0.02

    # Ensemble
    use_ensemble: bool = False
    num_ensemble_models: int = 5

    # Output
    output_dir: Optional[Path] = None
    save_checkpoints: bool = True

    def to_dl_settings(self) -> DeepLearningSettings:
        """Convert to DeepLearningSettings."""
        return DeepLearningSettings(
            model_type=self.model_type,
            num_control_points=self.num_control_points,
            lut_size=self.target_lut_size,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            early_stopping_patience=self.early_stopping_patience,
            mse_weight=self.mse_weight,
            monotonicity_weight=self.monotonicity_weight,
            smoothness_weight=self.smoothness_weight,
            augmentation_enabled=self.augmentation_enabled,
            noise_std=self.noise_std,
            use_ensemble=self.use_ensemble,
            num_ensemble_models=self.num_ensemble_models,
            checkpoint_dir=self.output_dir,
            save_best_only=self.save_checkpoints,
        )


@dataclass
class ExperimentResult:
    """Results from a training experiment."""

    config: ExperimentConfig
    training_stats: dict
    evaluation_metrics: dict
    model_path: Optional[Path] = None
    training_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "model_type": self.config.model_type,
                "num_control_points": self.config.num_control_points,
                "hidden_dims": self.config.hidden_dims,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
            },
            "training_stats": self.training_stats,
            "evaluation_metrics": self.evaluation_metrics,
            "model_path": str(self.model_path) if self.model_path else None,
            "training_time": self.training_time,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingPipeline:
    """
    High-level training pipeline for deep learning models.

    Provides a complete workflow for:
    - Loading and preparing data
    - Training models
    - Evaluating performance
    - Saving results
    """

    def __init__(
        self,
        database: CalibrationDatabase,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            database: Calibration database with training data.
            output_dir: Directory for outputs.
        """
        self.database = database
        self.output_dir = Path(output_dir) if output_dir else Path("./experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for PyTorch
        try:
            import torch  # noqa: F401

            self.torch_available = True
        except ImportError:
            self.torch_available = False

    def run_experiment(
        self,
        config: ExperimentConfig,
        callbacks: Optional[list[Callable]] = None,
    ) -> ExperimentResult:
        """
        Run a complete training experiment.

        Args:
            config: Experiment configuration.
            callbacks: Optional training callbacks.

        Returns:
            Experiment results.
        """
        if not self.torch_available:
            raise TrainingError(
                "PyTorch is required for training. "
                "Install with: pip install ptpd-calibration[deep]"
            )

        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        logger.info(f"Starting experiment: {config.name}")
        start_time = time.time()

        # Setup output directory
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir = exp_dir

        try:
            # Create predictor with experiment settings
            settings = config.to_dl_settings()
            predictor = DeepCurvePredictor(settings=settings)

            # Train model
            training_stats = predictor.train(
                database=self.database,
                val_ratio=config.train_val_split,
                num_epochs=config.num_epochs,
                callbacks=callbacks,
            )

            # Save model
            model_path = exp_dir / "model"
            predictor.save(model_path)

            # Evaluate on full dataset
            eval_metrics = self._evaluate_model(predictor)

            training_time = time.time() - start_time

            result = ExperimentResult(
                config=config,
                training_stats=training_stats,
                evaluation_metrics=eval_metrics,
                model_path=model_path,
                training_time=training_time,
            )

            # Save result
            result.save(exp_dir / "result.json")

            logger.info(
                f"Experiment '{config.name}' completed in {training_time:.1f}s. "
                f"MAE: {eval_metrics['mae']:.4f}"
            )

            return result

        except Exception as e:
            raise TrainingError(f"Experiment '{config.name}' failed: {e}") from e

    def _evaluate_model(
        self,
        predictor: "DeepCurvePredictor",
    ) -> dict:
        """Evaluate model on the full database."""
        from ptpd_calibration.ml.deep.dataset import CalibrationDataset, DataAugmentation

        import torch
        from torch.utils.data import DataLoader

        # Create dataset without augmentation
        dataset = CalibrationDataset(
            database=self.database,
            target_length=predictor.settings.lut_size,
            encoder=predictor.encoder,
            augmentation=DataAugmentation(enabled=False),
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Evaluate
        predictor.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets in loader:
                features = features.to(predictor.device)
                predictions, _ = predictor.model(features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        mse = float(np.mean((predictions - targets) ** 2))
        mae = float(np.mean(np.abs(predictions - targets)))
        max_error = float(np.max(np.abs(predictions - targets)))

        # Monotonicity check
        diffs = np.diff(predictions, axis=1)
        monotonicity_rate = float(np.mean(diffs >= -1e-6))

        # Per-sample correlation
        correlations = []
        for p, t in zip(predictions, targets):
            corr = np.corrcoef(p, t)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        mean_correlation = float(np.mean(correlations)) if correlations else 0.0

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "monotonicity_rate": monotonicity_rate,
            "mean_correlation": mean_correlation,
            "num_samples": len(predictions),
        }

    def run_hyperparameter_search(
        self,
        base_config: ExperimentConfig,
        param_grid: dict[str, list[Any]],
        max_experiments: int = 20,
    ) -> list[ExperimentResult]:
        """
        Run hyperparameter search.

        Args:
            base_config: Base configuration to modify.
            param_grid: Dictionary of parameter name to list of values.
            max_experiments: Maximum number of experiments.

        Returns:
            List of experiment results.
        """
        import itertools

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        # Limit number of experiments
        if len(combinations) > max_experiments:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combinations), max_experiments, replace=False)
            combinations = [combinations[i] for i in indices]

        logger.info(f"Running {len(combinations)} hyperparameter experiments")

        results = []
        for i, combo in enumerate(combinations):
            # Create config for this combination
            config_dict = {
                "name": f"{base_config.name}_hp_{i}",
                "description": f"Hyperparameter search {i}",
            }

            # Copy base config values
            for field_name in [
                "train_val_split",
                "target_lut_size",
                "model_type",
                "num_control_points",
                "hidden_dims",
                "dropout_rate",
                "learning_rate",
                "batch_size",
                "num_epochs",
                "early_stopping_patience",
                "mse_weight",
                "monotonicity_weight",
                "smoothness_weight",
                "augmentation_enabled",
                "noise_std",
            ]:
                config_dict[field_name] = getattr(base_config, field_name)

            # Apply hyperparameters
            for name, value in zip(param_names, combo):
                config_dict[name] = value

            config = ExperimentConfig(**config_dict)

            try:
                result = self.run_experiment(config)
                results.append(result)
            except Exception as e:
                logger.warning(f"Experiment {i} failed: {e}")

        # Sort by MAE
        results.sort(key=lambda r: r.evaluation_metrics.get("mae", float("inf")))

        return results

    def compare_models(
        self,
        model_paths: list[Path],
        test_records: Optional[list["CalibrationRecord"]] = None,
    ) -> dict:
        """
        Compare multiple trained models.

        Args:
            model_paths: Paths to model directories.
            test_records: Optional test records (uses database if not provided).

        Returns:
            Comparison results.
        """
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        if test_records is None:
            test_records = self.database.get_all_records()[:50]

        results = {}

        for path in model_paths:
            try:
                predictor = DeepCurvePredictor.load(path)

                # Predict on test records
                predictions = []
                actuals = []

                for record in test_records:
                    if record.measured_densities:
                        result = predictor.predict(record, return_uncertainty=False)
                        predictions.append(result.curve)

                        # Interpolate actual to same length
                        actual = np.array(record.measured_densities)
                        if len(actual) != len(result.curve):
                            x_old = np.linspace(0, 1, len(actual))
                            x_new = np.linspace(0, 1, len(result.curve))
                            actual = np.interp(x_new, x_old, actual)
                        actuals.append(actual)

                predictions = np.array(predictions)
                actuals = np.array(actuals)

                mae = float(np.mean(np.abs(predictions - actuals)))
                mse = float(np.mean((predictions - actuals) ** 2))

                results[str(path)] = {
                    "mae": mae,
                    "mse": mse,
                    "num_samples": len(predictions),
                }

            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                results[str(path)] = {"error": str(e)}

        return results


def run_training_pipeline(
    database: CalibrationDatabase,
    output_dir: Path,
    experiment_name: str = "default",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> ExperimentResult:
    """
    Convenience function to run a training pipeline.

    Args:
        database: Training database.
        output_dir: Output directory.
        experiment_name: Name for the experiment.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.

    Returns:
        Experiment results.
    """
    config = ExperimentConfig(
        name=experiment_name,
        description="Default training run",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    pipeline = TrainingPipeline(database, output_dir)
    return pipeline.run_experiment(config)


def quick_train(
    num_synthetic_samples: int = 200,
    num_epochs: int = 50,
    output_dir: Optional[Path] = None,
) -> "DeepCurvePredictor":
    """
    Quick training for testing and demos.

    Args:
        num_synthetic_samples: Number of synthetic samples.
        num_epochs: Training epochs.
        output_dir: Output directory.

    Returns:
        Trained DeepCurvePredictor.
    """
    from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
    from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

    # Generate synthetic data
    database = generate_training_data(num_synthetic_samples, seed=42)

    # Configure for quick training
    settings = DeepLearningSettings(
        num_control_points=12,
        lut_size=128,
        hidden_dims=[64, 128, 64],
        num_epochs=num_epochs,
        batch_size=16,
        early_stopping_patience=5,
        use_ensemble=False,
        device="cpu",
    )

    # Train
    predictor = DeepCurvePredictor(settings=settings)
    predictor.train(database, num_epochs=num_epochs)

    # Save if output_dir provided
    if output_dir:
        predictor.save(Path(output_dir))

    return predictor
