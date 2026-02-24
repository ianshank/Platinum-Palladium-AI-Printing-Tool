"""
PyTorch Dataset for calibration data.

Provides dataset classes for training deep learning models on
calibration records from the database.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ptpd_calibration.ml.deep.exceptions import DatasetError

if TYPE_CHECKING:
    from ptpd_calibration.core.models import CalibrationRecord
    from ptpd_calibration.ml.database import CalibrationDatabase

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    Dataset = object  # type: ignore


def _check_torch() -> None:
    """Raise error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning datasets. "
            "Install with: pip install ptpd-calibration[deep]"
        )


@dataclass
class FeatureEncoder:
    """
    Encodes calibration record features for neural network input.

    Handles categorical encoding, normalization, and feature ordering.
    """

    paper_to_idx: dict[str, int] = field(default_factory=dict)
    chemistry_to_idx: dict[str, int] = field(default_factory=dict)
    developer_to_idx: dict[str, int] = field(default_factory=dict)
    contrast_agent_to_idx: dict[str, int] = field(default_factory=dict)

    # Normalization statistics (log_exposure uses log-transformed values for proper scaling)
    log_exposure_mean: float = 5.2  # log(180+1) ≈ 5.2
    log_exposure_std: float = 0.3  # Typical std for log-transformed exposure times
    humidity_mean: float = 50.0
    humidity_std: float = 15.0
    temperature_mean: float = 21.0
    temperature_std: float = 5.0
    metal_ratio_range: tuple[float, float] = (0.0, 1.0)
    contrast_amount_max: float = 10.0

    # Feature names for interpretability
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def from_database(cls, database: CalibrationDatabase) -> FeatureEncoder:
        """
        Create encoder from a calibration database.

        Args:
            database: CalibrationDatabase with training records.

        Returns:
            Configured FeatureEncoder instance.
        """
        records = database.get_all_records()
        if not records:
            raise DatasetError("Cannot create encoder from empty database")

        # Build categorical mappings
        papers = sorted({r.paper_type for r in records})
        chemistries = sorted({r.chemistry_type.value for r in records})
        developers = sorted({r.developer.value for r in records})
        contrast_agents = sorted({r.contrast_agent.value for r in records})

        paper_to_idx = {p: i for i, p in enumerate(papers)}
        chemistry_to_idx = {c: i for i, c in enumerate(chemistries)}
        developer_to_idx = {d: i for i, d in enumerate(developers)}
        contrast_agent_to_idx = {a: i for i, a in enumerate(contrast_agents)}

        # Compute normalization statistics
        exposures = [r.exposure_time for r in records]
        humidities = [r.humidity for r in records if r.humidity is not None]
        temperatures = [r.temperature for r in records if r.temperature is not None]
        contrast_amounts = [r.contrast_amount for r in records]

        # For exposure times, compute mean/std on log-transformed values (standard approach)
        log_exposures = [np.log(e + 1) for e in exposures] if exposures else [np.log(181)]
        log_exposure_mean = np.mean(log_exposures)
        log_exposure_std = max(
            float(np.std(log_exposures)), 0.1
        )  # Minimum std to avoid division issues

        humidity_mean = np.mean(humidities) if humidities else 50.0
        humidity_std = max(float(np.std(humidities)), 1.0) if humidities else 15.0
        temperature_mean = np.mean(temperatures) if temperatures else 21.0
        temperature_std = max(float(np.std(temperatures)), 1.0) if temperatures else 5.0
        contrast_amount_max = max(contrast_amounts) if contrast_amounts else 10.0

        # Build feature names
        feature_names = [
            *[f"paper_{p}" for p in papers],
            *[f"chemistry_{c}" for c in chemistries],
            *[f"developer_{d}" for d in developers],
            *[f"contrast_agent_{a}" for a in contrast_agents],
            "metal_ratio",
            "contrast_amount",
            "log_exposure",
            "humidity",
            "temperature",
        ]

        return cls(
            paper_to_idx=paper_to_idx,
            chemistry_to_idx=chemistry_to_idx,
            developer_to_idx=developer_to_idx,
            contrast_agent_to_idx=contrast_agent_to_idx,
            log_exposure_mean=float(log_exposure_mean),
            log_exposure_std=float(log_exposure_std),
            humidity_mean=float(humidity_mean),
            humidity_std=float(humidity_std),
            temperature_mean=float(temperature_mean),
            temperature_std=float(temperature_std),
            contrast_amount_max=float(contrast_amount_max),
            feature_names=feature_names,
        )

    @property
    def num_features(self) -> int:
        """Get total number of features."""
        return (
            len(self.paper_to_idx)
            + len(self.chemistry_to_idx)
            + len(self.developer_to_idx)
            + len(self.contrast_agent_to_idx)
            + 5  # metal_ratio, contrast_amount, log_exposure, humidity, temperature
        )

    def encode(self, record: CalibrationRecord) -> np.ndarray:
        """
        Encode a calibration record to feature vector.

        Args:
            record: CalibrationRecord to encode.

        Returns:
            Feature vector as numpy array.
        """
        features: list[float] = []

        # One-hot encode paper type
        paper_onehot = np.zeros(len(self.paper_to_idx))
        idx = self.paper_to_idx.get(record.paper_type, -1)
        if idx >= 0:
            paper_onehot[idx] = 1.0
        else:
            # Unknown paper - use uniform distribution
            paper_onehot[:] = 1.0 / len(self.paper_to_idx)
        features.extend(paper_onehot)

        # One-hot encode chemistry type
        chem_onehot = np.zeros(len(self.chemistry_to_idx))
        idx = self.chemistry_to_idx.get(record.chemistry_type.value, -1)
        if idx >= 0:
            chem_onehot[idx] = 1.0
        else:
            chem_onehot[:] = 1.0 / len(self.chemistry_to_idx)
        features.extend(chem_onehot)

        # One-hot encode developer
        dev_onehot = np.zeros(len(self.developer_to_idx))
        idx = self.developer_to_idx.get(record.developer.value, -1)
        if idx >= 0:
            dev_onehot[idx] = 1.0
        else:
            dev_onehot[:] = 1.0 / len(self.developer_to_idx)
        features.extend(dev_onehot)

        # One-hot encode contrast agent
        agent_onehot = np.zeros(len(self.contrast_agent_to_idx))
        idx = self.contrast_agent_to_idx.get(record.contrast_agent.value, -1)
        if idx >= 0:
            agent_onehot[idx] = 1.0
        else:
            agent_onehot[:] = 1.0 / len(self.contrast_agent_to_idx)
        features.extend(agent_onehot)

        # Numerical features (normalized)
        features.append(record.metal_ratio)  # Already 0-1
        features.append(
            record.contrast_amount / self.contrast_amount_max
            if self.contrast_amount_max > 0
            else 0.0
        )
        # Standard z-score normalization on log-transformed exposure time
        log_exposure = np.log(record.exposure_time + 1)
        features.append((log_exposure - self.log_exposure_mean) / self.log_exposure_std)
        features.append(
            ((record.humidity or self.humidity_mean) - self.humidity_mean) / self.humidity_std
        )
        features.append(
            ((record.temperature or self.temperature_mean) - self.temperature_mean)
            / self.temperature_std
        )

        return np.array(features, dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize encoder to dictionary for saving."""
        return {
            "paper_to_idx": self.paper_to_idx,
            "chemistry_to_idx": self.chemistry_to_idx,
            "developer_to_idx": self.developer_to_idx,
            "contrast_agent_to_idx": self.contrast_agent_to_idx,
            "log_exposure_mean": self.log_exposure_mean,
            "log_exposure_std": self.log_exposure_std,
            "humidity_mean": self.humidity_mean,
            "humidity_std": self.humidity_std,
            "temperature_mean": self.temperature_mean,
            "temperature_std": self.temperature_std,
            "contrast_amount_max": self.contrast_amount_max,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureEncoder:
        """Load encoder from dictionary."""
        return cls(**data)


@dataclass
class DataAugmentation:
    """
    Data augmentation settings for training.

    Provides various augmentation strategies to improve model robustness.
    """

    enabled: bool = True
    noise_std: float = 0.02
    exposure_jitter: float = 0.1
    density_scale_range: tuple[float, float] = (0.95, 1.05)
    density_shift_range: tuple[float, float] = (-0.02, 0.02)

    def augment_features(
        self, features: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Apply augmentation to feature vector.

        Args:
            features: Feature vector.
            rng: Random number generator.

        Returns:
            Augmented feature vector.
        """
        if not self.enabled:
            return features

        if rng is None:
            rng = np.random.default_rng()

        features = features.copy()

        # Add noise to numerical features (last 5 features)
        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, 5)
            features[-5:] += noise

        return features

    def augment_densities(
        self, densities: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Apply augmentation to density values.

        Args:
            densities: Density curve.
            rng: Random number generator.

        Returns:
            Augmented densities.
        """
        if not self.enabled:
            return densities

        if rng is None:
            rng = np.random.default_rng()

        densities = densities.copy()

        # Scale
        scale = rng.uniform(*self.density_scale_range)
        densities *= scale

        # Shift
        shift = rng.uniform(*self.density_shift_range)
        densities += shift

        # Clamp to valid range
        densities = np.clip(densities, 0.0, 4.0)

        return densities


class CalibrationDataset(Dataset):
    """
    PyTorch Dataset for calibration records.

    Loads calibration records from a database and provides
    (features, densities) pairs for training.
    """

    def __init__(
        self,
        database: CalibrationDatabase,
        target_length: int = 256,
        encoder: FeatureEncoder | None = None,
        augmentation: DataAugmentation | None = None,
        transform: Callable | None = None,
    ):
        """
        Initialize CalibrationDataset.

        Args:
            database: CalibrationDatabase with training records.
            target_length: Length to interpolate density curves to.
            encoder: Feature encoder (created from database if not provided).
            augmentation: Data augmentation settings.
            transform: Optional transform to apply to samples.
        """
        _check_torch()
        super().__init__()

        self.target_length = target_length
        self.transform = transform
        self.augmentation = augmentation or DataAugmentation(enabled=False)
        self._rng = np.random.default_rng()

        # Filter records with valid density measurements
        self.records = [r for r in database.get_all_records() if r.measured_densities]

        if not self.records:
            raise DatasetError("No records with density measurements found")

        # Create or use provided encoder
        if encoder is None:
            self.encoder = FeatureEncoder.from_database(database)
        else:
            self.encoder = encoder

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, densities) tensors.
        """
        record = self.records[idx]

        # Encode features
        features = self.encoder.encode(record)

        # Interpolate densities to target length
        densities = np.array(record.measured_densities, dtype=np.float32)
        if len(densities) != self.target_length:
            x_old = np.linspace(0, 1, len(densities))
            x_new = np.linspace(0, 1, self.target_length)
            densities = np.interp(x_new, x_old, densities).astype(np.float32)

        # Apply augmentation
        if self.augmentation.enabled:
            features = self.augmentation.augment_features(features, self._rng)
            densities = self.augmentation.augment_densities(densities, self._rng)

        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        densities_tensor = torch.from_numpy(densities)

        # Apply transform if provided
        if self.transform is not None:
            features_tensor, densities_tensor = self.transform(features_tensor, densities_tensor)

        return features_tensor, densities_tensor

    @property
    def num_features(self) -> int:
        """Get number of input features."""
        return self.encoder.num_features

    def split(
        self,
        val_ratio: float = 0.2,
        seed: int | None = None,
    ) -> tuple[SubsetDataset, SubsetDataset]:
        """
        Split dataset into train and validation sets.

        Args:
            val_ratio: Fraction of data for validation.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.records))

        split_idx = int(len(indices) * (1 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Create subset datasets
        train_dataset = SubsetDataset(self, train_indices)
        val_dataset = SubsetDataset(self, val_indices)

        # Disable augmentation for validation
        val_dataset.augmentation = DataAugmentation(enabled=False)

        return train_dataset, val_dataset


class SubsetDataset(Dataset):
    """Subset of a CalibrationDataset."""

    def __init__(
        self,
        parent: CalibrationDataset,
        indices: np.ndarray,
    ):
        """
        Initialize SubsetDataset.

        Args:
            parent: Parent dataset.
            indices: Indices to include.
        """
        _check_torch()
        super().__init__()
        self.parent = parent
        self.indices = indices
        self.augmentation = parent.augmentation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        parent_idx = self.indices[idx]
        return self.parent[parent_idx]

    @property
    def num_features(self) -> int:
        return self.parent.num_features

    @property
    def encoder(self) -> FeatureEncoder:
        return self.parent.encoder

    @property
    def target_length(self) -> int:
        return self.parent.target_length


def create_dataloaders(
    database: CalibrationDatabase,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    target_length: int = 256,
    augmentation: DataAugmentation | None = None,
    num_workers: int = 0,
    seed: int | None = None,
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        database: CalibrationDatabase with training records.
        batch_size: Batch size for training.
        val_ratio: Fraction of data for validation.
        target_length: Length to interpolate density curves to.
        augmentation: Data augmentation settings.
        num_workers: Number of data loading workers.
        seed: Random seed.

    Returns:
        Tuple of (train_loader, val_loader, encoder).
    """
    _check_torch()
    from torch.utils.data import DataLoader

    # Create full dataset
    full_dataset = CalibrationDataset(
        database=database,
        target_length=target_length,
        augmentation=augmentation,
    )

    # Split into train/val
    train_dataset, val_dataset = full_dataset.split(val_ratio=val_ratio, seed=seed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, full_dataset.encoder
