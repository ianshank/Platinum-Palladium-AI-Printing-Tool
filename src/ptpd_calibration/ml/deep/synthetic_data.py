"""
Synthetic data generation for deep learning training.

Generates realistic calibration records based on physical models
of the Pt/Pd printing process. Useful for:

- Initial model training before real data is available
- Data augmentation to improve model robustness
- Testing and validation of the training pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    DeveloperType,
    PaperSizing,
)
from ptpd_calibration.ml.database import CalibrationDatabase

logger = logging.getLogger(__name__)


class PaperCharacteristic(str, Enum):
    """Paper characteristic types affecting response."""

    HOT_PRESS = "hot_press"  # Smooth, lower absorption
    COLD_PRESS = "cold_press"  # Textured, higher absorption
    ROUGH = "rough"  # Very textured


@dataclass
class PaperProfile:
    """Physical properties of a paper type."""

    name: str
    characteristic: PaperCharacteristic
    base_dmin: float  # Minimum density (paper base)
    base_dmax: float  # Maximum achievable density
    gamma_modifier: float  # Affects curve shape
    exposure_factor: float  # Relative exposure needed
    contrast_response: float  # How much contrast agent affects curve
    humidity_sensitivity: float  # How much humidity affects result


@dataclass
class ProcessModel:
    """Physical model of the Pt/Pd printing process."""

    # Base process parameters
    base_gamma: float = 1.8
    base_dmin: float = 0.08
    base_dmax: float = 2.0
    shoulder_strength: float = 0.15
    toe_strength: float = 0.1

    # Environmental coefficients
    humidity_coefficient: float = 0.008
    temperature_coefficient: float = 0.015
    uv_variability: float = 0.05

    # Chemistry effects
    platinum_gamma_boost: float = 0.15
    platinum_dmax_boost: float = 0.2
    na2_contrast_factor: float = 0.12
    developer_temp_factor: float = 0.02

    def compute_density_curve(
        self,
        paper: PaperProfile,
        metal_ratio: float,
        contrast_agent: ContrastAgent,
        contrast_amount: float,
        exposure_time: float,
        humidity: float,
        temperature: float,
        num_steps: int = 21,
        noise_std: float = 0.015,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Compute a realistic density curve based on physical parameters.

        Args:
            paper: Paper profile.
            metal_ratio: Platinum ratio (0=Pd, 1=Pt).
            contrast_agent: Type of contrast agent.
            contrast_amount: Amount of contrast agent (drops).
            exposure_time: Exposure time in seconds.
            humidity: Relative humidity (%).
            temperature: Temperature (°C).
            num_steps: Number of density steps.
            noise_std: Standard deviation of measurement noise.
            rng: Random number generator.

        Returns:
            Array of density values.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Base parameters modified by paper
        gamma = self.base_gamma * paper.gamma_modifier
        dmin = self.base_dmin + paper.base_dmin
        dmax = self.base_dmax * (paper.base_dmax / 2.0)

        # Platinum effect: higher gamma and dmax
        gamma += metal_ratio * self.platinum_gamma_boost
        dmax += metal_ratio * self.platinum_dmax_boost

        # Contrast agent effect
        if contrast_agent == ContrastAgent.NA2:
            gamma += contrast_amount * self.na2_contrast_factor * paper.contrast_response
            # NA2 can slightly reduce dmax at high amounts
            dmax -= contrast_amount * 0.01

        # Environmental effects
        humidity_effect = (humidity - 50.0) * self.humidity_coefficient * paper.humidity_sensitivity
        temp_effect = (temperature - 21.0) * self.temperature_coefficient

        gamma += humidity_effect
        dmax += temp_effect * 0.1

        # Exposure effect on density range
        optimal_exposure = 180.0 * paper.exposure_factor
        exposure_ratio = exposure_time / optimal_exposure

        # Under/over exposure affects dmax
        if exposure_ratio < 0.8:
            dmax *= 0.7 + 0.3 * exposure_ratio / 0.8
        elif exposure_ratio > 1.3:
            # Over-exposure causes some highlight blocking
            dmin += 0.05 * (exposure_ratio - 1.3)

        # Generate the characteristic curve
        steps = np.linspace(0, 1, num_steps)

        # Apply gamma curve with toe and shoulder
        curve = np.zeros(num_steps)
        for i, x in enumerate(steps):
            # Toe region (shadows)
            if x < 0.15:
                toe_factor = 1.0 - self.toe_strength * (1.0 - x / 0.15)
                y = (x**gamma) * toe_factor
            # Shoulder region (highlights)
            elif x > 0.85:
                shoulder_factor = 1.0 - self.shoulder_strength * ((x - 0.85) / 0.15) ** 2
                y = (x**gamma) * shoulder_factor
            else:
                y = x**gamma

            curve[i] = dmin + (dmax - dmin) * y

        # Add realistic measurement noise
        noise = rng.normal(0, noise_std, num_steps)
        curve += noise

        # Ensure monotonicity (with small tolerance for noise)
        for i in range(1, num_steps):
            if curve[i] < curve[i - 1] - 0.01:
                curve[i] = curve[i - 1] + rng.uniform(0.001, 0.01)

        # Clamp to valid range
        curve = np.clip(curve, 0.0, 3.5)

        return curve


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    # Number of records
    num_records: int = 500

    # Paper types to generate
    papers: list[PaperProfile] = field(default_factory=list)

    # Parameter ranges
    metal_ratio_range: tuple[float, float] = (0.0, 1.0)
    exposure_time_range: tuple[float, float] = (90.0, 360.0)
    contrast_amount_range: tuple[float, float] = (0.0, 10.0)
    humidity_range: tuple[float, float] = (35.0, 65.0)
    temperature_range: tuple[float, float] = (18.0, 25.0)

    # Noise and variability
    measurement_noise_std: float = 0.015
    batch_variability: float = 0.05

    # Random seed for reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize default papers if not provided."""
        if not self.papers:
            self.papers = get_default_paper_profiles()


def get_default_paper_profiles() -> list[PaperProfile]:
    """Get default paper profiles for synthetic data generation."""
    return [
        PaperProfile(
            name="Arches Platine",
            characteristic=PaperCharacteristic.HOT_PRESS,
            base_dmin=0.08,
            base_dmax=2.1,
            gamma_modifier=1.0,
            exposure_factor=1.0,
            contrast_response=1.0,
            humidity_sensitivity=1.0,
        ),
        PaperProfile(
            name="Bergger COT320",
            characteristic=PaperCharacteristic.HOT_PRESS,
            base_dmin=0.07,
            base_dmax=2.2,
            gamma_modifier=0.95,
            exposure_factor=0.9,
            contrast_response=1.1,
            humidity_sensitivity=0.9,
        ),
        PaperProfile(
            name="Hahnemühle Platinum Rag",
            characteristic=PaperCharacteristic.HOT_PRESS,
            base_dmin=0.09,
            base_dmax=2.0,
            gamma_modifier=1.05,
            exposure_factor=1.1,
            contrast_response=0.95,
            humidity_sensitivity=1.1,
        ),
        PaperProfile(
            name="Revere Platinum",
            characteristic=PaperCharacteristic.COLD_PRESS,
            base_dmin=0.10,
            base_dmax=1.9,
            gamma_modifier=1.1,
            exposure_factor=1.2,
            contrast_response=0.9,
            humidity_sensitivity=1.2,
        ),
        PaperProfile(
            name="Crane's Platinotype",
            characteristic=PaperCharacteristic.HOT_PRESS,
            base_dmin=0.06,
            base_dmax=2.3,
            gamma_modifier=0.9,
            exposure_factor=0.85,
            contrast_response=1.15,
            humidity_sensitivity=0.85,
        ),
        PaperProfile(
            name="Stonehenge",
            characteristic=PaperCharacteristic.COLD_PRESS,
            base_dmin=0.11,
            base_dmax=1.85,
            gamma_modifier=1.15,
            exposure_factor=1.25,
            contrast_response=0.85,
            humidity_sensitivity=1.3,
        ),
    ]


class SyntheticDataGenerator:
    """
    Generator for synthetic calibration data.

    Creates realistic calibration records based on physical models
    of the Pt/Pd printing process.
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize the generator.

        Args:
            config: Generation configuration.
        """
        self.config = config or SyntheticDataConfig()
        self.process_model = ProcessModel()
        self.rng = np.random.default_rng(self.config.seed)

    def generate_record(
        self,
        paper: Optional[PaperProfile] = None,
        metal_ratio: Optional[float] = None,
        exposure_time: Optional[float] = None,
        contrast_agent: Optional[ContrastAgent] = None,
        contrast_amount: Optional[float] = None,
        humidity: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> CalibrationRecord:
        """
        Generate a single calibration record.

        Args:
            paper: Paper profile (random if not specified).
            metal_ratio: Metal ratio (random if not specified).
            exposure_time: Exposure time (random if not specified).
            contrast_agent: Contrast agent type.
            contrast_amount: Contrast amount.
            humidity: Humidity level.
            temperature: Temperature.

        Returns:
            Generated CalibrationRecord.
        """
        # Select random values for unspecified parameters
        if paper is None:
            paper = self.rng.choice(self.config.papers)

        if metal_ratio is None:
            metal_ratio = self.rng.uniform(*self.config.metal_ratio_range)

        if exposure_time is None:
            exposure_time = self.rng.uniform(*self.config.exposure_time_range)

        if contrast_agent is None:
            # 70% chance of using NA2
            contrast_agent = (
                ContrastAgent.NA2 if self.rng.random() < 0.7 else ContrastAgent.NONE
            )

        if contrast_amount is None:
            if contrast_agent != ContrastAgent.NONE:
                contrast_amount = self.rng.uniform(*self.config.contrast_amount_range)
            else:
                contrast_amount = 0.0

        if humidity is None:
            humidity = self.rng.uniform(*self.config.humidity_range)

        if temperature is None:
            temperature = self.rng.uniform(*self.config.temperature_range)

        # Add batch-to-batch variability
        batch_factor = 1.0 + self.rng.normal(0, self.config.batch_variability)

        # Generate density curve
        densities = self.process_model.compute_density_curve(
            paper=paper,
            metal_ratio=metal_ratio,
            contrast_agent=contrast_agent,
            contrast_amount=contrast_amount,
            exposure_time=exposure_time * batch_factor,
            humidity=humidity,
            temperature=temperature,
            noise_std=self.config.measurement_noise_std,
            rng=self.rng,
        )

        # Determine chemistry type based on metal ratio
        if metal_ratio > 0.95:
            chemistry_type = ChemistryType.PURE_PLATINUM
        elif metal_ratio < 0.05:
            chemistry_type = ChemistryType.PURE_PALLADIUM
        else:
            chemistry_type = ChemistryType.PLATINUM_PALLADIUM

        # Create record
        return CalibrationRecord(
            paper_type=paper.name,
            paper_weight=300,  # Standard weight
            paper_sizing=PaperSizing.INTERNAL,
            chemistry_type=chemistry_type,
            metal_ratio=metal_ratio,
            contrast_agent=contrast_agent,
            contrast_amount=contrast_amount,
            developer=DeveloperType.POTASSIUM_OXALATE,
            exposure_time=exposure_time,
            humidity=humidity,
            temperature=temperature,
            measured_densities=list(densities),
        )

    def generate_database(
        self,
        num_records: Optional[int] = None,
    ) -> CalibrationDatabase:
        """
        Generate a full database of calibration records.

        Args:
            num_records: Number of records to generate.

        Returns:
            CalibrationDatabase with generated records.
        """
        num_records = num_records or self.config.num_records
        db = CalibrationDatabase()

        logger.info(f"Generating {num_records} synthetic calibration records...")

        for i in range(num_records):
            record = self.generate_record()
            db.add_record(record)

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{num_records} records")

        logger.info(f"Generated database with {len(db)} records")
        return db

    def generate_test_suite(self) -> dict[str, CalibrationDatabase]:
        """
        Generate a suite of test databases for different scenarios.

        Returns:
            Dictionary of scenario name to database.
        """
        suites = {}

        # Standard training set
        suites["training"] = self.generate_database(500)

        # Small dataset for quick testing
        small_config = SyntheticDataConfig(
            num_records=50,
            seed=42,
        )
        small_gen = SyntheticDataGenerator(small_config)
        suites["small"] = small_gen.generate_database()

        # Single paper type (for transfer learning tests)
        single_paper_db = CalibrationDatabase()
        paper = self.config.papers[0]
        for _ in range(100):
            record = self.generate_record(paper=paper)
            single_paper_db.add_record(record)
        suites["single_paper"] = single_paper_db

        # High contrast scenario
        high_contrast_db = CalibrationDatabase()
        for _ in range(100):
            record = self.generate_record(
                metal_ratio=self.rng.uniform(0.7, 1.0),
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=self.rng.uniform(6.0, 10.0),
            )
            high_contrast_db.add_record(record)
        suites["high_contrast"] = high_contrast_db

        # Low contrast scenario
        low_contrast_db = CalibrationDatabase()
        for _ in range(100):
            record = self.generate_record(
                metal_ratio=self.rng.uniform(0.0, 0.3),
                contrast_agent=ContrastAgent.NONE,
                contrast_amount=0.0,
            )
            low_contrast_db.add_record(record)
        suites["low_contrast"] = low_contrast_db

        return suites

    def generate_exposure_series(
        self,
        paper: Optional[PaperProfile] = None,
        metal_ratio: float = 0.5,
        num_exposures: int = 7,
    ) -> list[CalibrationRecord]:
        """
        Generate an exposure bracket series.

        Useful for testing exposure prediction.

        Args:
            paper: Paper profile.
            metal_ratio: Metal ratio to use.
            num_exposures: Number of exposures in series.

        Returns:
            List of calibration records at different exposures.
        """
        if paper is None:
            paper = self.config.papers[0]

        base_exposure = 180.0 * paper.exposure_factor
        exposures = np.logspace(
            np.log10(base_exposure * 0.5),
            np.log10(base_exposure * 2.0),
            num_exposures,
        )

        records = []
        for exposure in exposures:
            record = self.generate_record(
                paper=paper,
                metal_ratio=metal_ratio,
                exposure_time=exposure,
            )
            records.append(record)

        return records

    def generate_metal_ratio_series(
        self,
        paper: Optional[PaperProfile] = None,
        num_ratios: int = 5,
    ) -> list[CalibrationRecord]:
        """
        Generate a metal ratio series.

        Useful for testing metal ratio effects.

        Args:
            paper: Paper profile.
            num_ratios: Number of ratios in series.

        Returns:
            List of calibration records at different metal ratios.
        """
        if paper is None:
            paper = self.config.papers[0]

        ratios = np.linspace(0.0, 1.0, num_ratios)

        records = []
        for ratio in ratios:
            record = self.generate_record(
                paper=paper,
                metal_ratio=ratio,
            )
            records.append(record)

        return records


def generate_training_data(
    num_records: int = 500,
    seed: Optional[int] = None,
) -> CalibrationDatabase:
    """
    Convenience function to generate training data.

    Args:
        num_records: Number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        CalibrationDatabase with synthetic records.
    """
    config = SyntheticDataConfig(
        num_records=num_records,
        seed=seed,
    )
    generator = SyntheticDataGenerator(config)
    return generator.generate_database()
