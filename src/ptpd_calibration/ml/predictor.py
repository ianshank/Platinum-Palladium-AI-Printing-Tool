"""
ML-based curve prediction from calibration parameters.
"""

import pickle
from pathlib import Path

import numpy as np

from ptpd_calibration.config import MLSettings, get_settings
from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.ml.database import CalibrationDatabase


class CurvePredictor:
    """
    Machine learning predictor for density response curves.

    Uses gradient boosting or random forest to predict density
    responses based on process parameters.
    """

    def __init__(
        self,
        model_type: str | None = None,
        settings: MLSettings | None = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_type: Model type ("gradient_boosting", "random_forest").
            settings: ML settings.
        """
        self.settings = settings or get_settings().ml
        self.model_type = model_type or self.settings.default_model_type
        self.model = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self._paper_encoder: dict[str, int] = {}
        self._chemistry_encoder: dict[str, int] = {}

    def train(
        self,
        database: CalibrationDatabase,
        validation_split: float | None = None,
    ) -> dict:
        """
        Train the predictor on calibration database.

        Args:
            database: CalibrationDatabase with training records.
            validation_split: Fraction of data for validation.

        Returns:
            Dictionary with training statistics.
        """
        records = database.get_all_records()

        if len(records) < self.settings.min_training_samples:
            raise ValueError(
                f"Need at least {self.settings.min_training_samples} records, "
                f"got {len(records)}"
            )

        # Filter records with density measurements
        valid_records = [r for r in records if r.measured_densities]
        if not valid_records:
            raise ValueError("No records with density measurements")

        # Build feature encoders
        self._build_encoders(valid_records)

        # Prepare training data
        X, y = self._prepare_data(valid_records)

        # Split data
        split = validation_split or self.settings.validation_split
        split_idx = int(len(X) * (1 - split))

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate statistics
        train_pred = self.model.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - y_train))

        val_pred = self.model.predict(X_val) if len(X_val) > 0 else []
        val_mae = np.mean(np.abs(val_pred - y_val)) if len(val_pred) > 0 else 0.0

        return {
            "num_samples": len(valid_records),
            "num_features": X.shape[1],
            "training_mae": float(train_mae),
            "validation_mae": float(val_mae),
            "model_type": self.model_type,
        }

    def predict(
        self,
        record: CalibrationRecord,
        return_uncertainty: bool = False,
    ) -> tuple[list[float], float | None]:
        """
        Predict density response for a calibration setup.

        Args:
            record: CalibrationRecord with process parameters.
            return_uncertainty: Whether to return uncertainty estimate.

        Returns:
            Tuple of (predicted densities, uncertainty).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        # Prepare features
        X = self._record_to_features(record)

        # Predict
        prediction = self.model.predict([X])[0]

        uncertainty = None
        if return_uncertainty and hasattr(self.model, "estimators_"):
            # For ensemble models, use variance across estimators
            predictions = np.array([
                est.predict([X])[0] for est in self.model.estimators_
            ])
            uncertainty = float(np.std(predictions))

        return list(prediction), uncertainty

    def predict_curve_adjustment(
        self,
        record: CalibrationRecord,
        target_densities: list[float],
    ) -> dict:
        """
        Predict parameter adjustments to achieve target densities.

        Args:
            record: Current calibration setup.
            target_densities: Desired density values.

        Returns:
            Dictionary with adjustment suggestions.
        """
        current_pred, _ = self.predict(record)

        suggestions = []
        adjustments = {}

        # Compare predicted to target
        current_dmax = max(current_pred) if current_pred else 0
        target_dmax = max(target_densities) if target_densities else 2.0

        # Exposure adjustment
        if current_dmax > 0:
            exposure_factor = (target_dmax / current_dmax) ** 0.7
            if abs(exposure_factor - 1.0) > 0.1:
                new_exposure = record.exposure_time * exposure_factor
                adjustments["exposure_time"] = new_exposure
                suggestions.append(
                    f"Adjust exposure from {record.exposure_time:.0f}s to {new_exposure:.0f}s"
                )

        # Metal ratio adjustment
        current_range = max(current_pred) - min(current_pred) if current_pred else 0
        target_range = max(target_densities) - min(target_densities) if target_densities else 2.0

        if current_range < target_range * 0.85:
            new_ratio = min(1.0, record.metal_ratio + 0.1)
            if new_ratio != record.metal_ratio:
                adjustments["metal_ratio"] = new_ratio
                suggestions.append(
                    f"Increase platinum ratio to {new_ratio:.1%} for higher Dmax"
                )

        return {
            "current_prediction": current_pred,
            "target": target_densities,
            "adjustments": adjustments,
            "suggestions": suggestions,
        }

    def save(self, path: Path) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "paper_encoder": self._paper_encoder,
            "chemistry_encoder": self._chemistry_encoder,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "CurvePredictor":
        """Load trained model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        predictor = cls(model_type=data["model_type"])
        predictor.model = data["model"]
        predictor.feature_names = data["feature_names"]
        predictor._paper_encoder = data["paper_encoder"]
        predictor._chemistry_encoder = data["chemistry_encoder"]
        predictor.is_trained = True

        return predictor

    def _create_model(self):
        """Create the ML model based on settings."""
        try:
            if self.model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.multioutput import MultiOutputRegressor

                base = GradientBoostingRegressor(
                    n_estimators=self.settings.n_estimators,
                    max_depth=self.settings.max_depth,
                    random_state=42,
                )
                return MultiOutputRegressor(base)

            elif self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    n_estimators=self.settings.n_estimators,
                    max_depth=self.settings.max_depth,
                    random_state=42,
                )

            else:
                # Fallback to gradient boosting
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.multioutput import MultiOutputRegressor

                base = GradientBoostingRegressor(
                    n_estimators=self.settings.n_estimators,
                    max_depth=self.settings.max_depth,
                    random_state=42,
                )
                return MultiOutputRegressor(base)

        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for ML features. "
                "Install with: pip install ptpd-calibration[ml]"
            ) from e

    def _build_encoders(self, records: list[CalibrationRecord]) -> None:
        """Build categorical encoders from records."""
        papers = sorted({r.paper_type for r in records})
        self._paper_encoder = {p: i for i, p in enumerate(papers)}

        chemistries = sorted({r.chemistry_type.value for r in records})
        self._chemistry_encoder = {c: i for i, c in enumerate(chemistries)}

    def _prepare_data(
        self, records: list[CalibrationRecord]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from records."""
        X = []
        y = []

        for record in records:
            features = self._record_to_features(record)
            X.append(features)

            # Normalize densities to fixed length
            densities = record.measured_densities
            if len(densities) < 21:
                # Interpolate to 21 points
                x_old = np.linspace(0, 1, len(densities))
                x_new = np.linspace(0, 1, 21)
                densities = list(np.interp(x_new, x_old, densities))
            elif len(densities) > 21:
                # Downsample
                indices = np.linspace(0, len(densities) - 1, 21).astype(int)
                densities = [densities[i] for i in indices]

            y.append(densities)

        return np.array(X), np.array(y)

    def _record_to_features(self, record: CalibrationRecord) -> list[float]:
        """Convert record to feature vector."""
        features = []

        # Paper type (one-hot or ordinal)
        paper_idx = self._paper_encoder.get(record.paper_type, -1)
        features.append(float(paper_idx))

        # Chemistry type
        chem_idx = self._chemistry_encoder.get(record.chemistry_type.value, -1)
        features.append(float(chem_idx))

        # Numerical features
        features.append(record.metal_ratio)
        features.append(float(record.contrast_agent.value != "none"))
        features.append(record.contrast_amount)
        features.append(np.log(record.exposure_time + 1))  # Log scale
        features.append(record.humidity or 50.0)
        features.append(record.temperature or 21.0)

        self.feature_names = [
            "paper_type",
            "chemistry_type",
            "metal_ratio",
            "has_contrast_agent",
            "contrast_amount",
            "log_exposure",
            "humidity",
            "temperature",
        ]

        return features
