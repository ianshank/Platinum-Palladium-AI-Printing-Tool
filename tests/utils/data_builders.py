"""
Test data builders for PTPD Calibration tests.

Provides builder classes and factories for creating test data.
"""

import random
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np


class DensityBuilder:
    """Builder for density measurement data."""

    def __init__(self):
        self._num_steps = 21
        self._dmin = 0.1
        self._dmax = 2.0
        self._gamma = 0.85
        self._noise = 0.0
        self._monotonic = True

    def with_steps(self, num_steps: int) -> "DensityBuilder":
        """Set the number of density steps."""
        self._num_steps = num_steps
        return self

    def with_range(self, dmin: float, dmax: float) -> "DensityBuilder":
        """Set the density range."""
        self._dmin = dmin
        self._dmax = dmax
        return self

    def with_gamma(self, gamma: float) -> "DensityBuilder":
        """Set the gamma curve."""
        self._gamma = gamma
        return self

    def with_noise(self, noise: float) -> "DensityBuilder":
        """Add random noise to measurements."""
        self._noise = noise
        return self

    def non_monotonic(self) -> "DensityBuilder":
        """Allow non-monotonic values."""
        self._monotonic = False
        return self

    def build(self) -> list[float]:
        """Build the density list."""
        steps = np.linspace(0, 1, self._num_steps)
        densities = self._dmin + (self._dmax - self._dmin) * (steps ** self._gamma)

        if self._noise > 0:
            noise = np.random.normal(0, self._noise, self._num_steps)
            densities = densities + noise

            if self._monotonic:
                # Ensure monotonicity
                for i in range(1, len(densities)):
                    if densities[i] < densities[i - 1]:
                        densities[i] = densities[i - 1] + 0.001

        return list(densities)


class CurveBuilder:
    """Builder for curve data."""

    def __init__(self):
        self._num_points = 256
        self._gamma = 0.9
        self._brightness = 0.0
        self._contrast = 0.0
        self._name = "Test Curve"

    def with_points(self, num_points: int) -> "CurveBuilder":
        """Set the number of curve points."""
        self._num_points = num_points
        return self

    def with_gamma(self, gamma: float) -> "CurveBuilder":
        """Set the gamma value."""
        self._gamma = gamma
        return self

    def with_brightness(self, brightness: float) -> "CurveBuilder":
        """Set brightness offset (-1 to 1)."""
        self._brightness = brightness
        return self

    def with_contrast(self, contrast: float) -> "CurveBuilder":
        """Set contrast adjustment (-1 to 1)."""
        self._contrast = contrast
        return self

    def with_name(self, name: str) -> "CurveBuilder":
        """Set the curve name."""
        self._name = name
        return self

    def build(self) -> dict:
        """Build the curve data."""
        input_values = list(np.linspace(0, 1, self._num_points))

        # Apply gamma
        output_values = [x ** self._gamma for x in input_values]

        # Apply brightness
        output_values = [min(1, max(0, y + self._brightness)) for y in output_values]

        # Apply contrast
        if self._contrast != 0:
            pivot = 0.5
            factor = (1 + self._contrast) if self._contrast > 0 else (1 / (1 - self._contrast))
            output_values = [
                min(1, max(0, (y - pivot) * factor + pivot))
                for y in output_values
            ]

        return {
            "name": self._name,
            "input_values": input_values,
            "output_values": output_values,
        }

    def build_model(self):
        """Build a CurveData model instance."""
        from ptpd_calibration.core.models import CurveData

        data = self.build()
        return CurveData(
            name=data["name"],
            input_values=data["input_values"],
            output_values=data["output_values"],
        )


class CalibrationRecordBuilder:
    """Builder for CalibrationRecord instances."""

    def __init__(self):
        from ptpd_calibration.core.types import (
            ChemistryType,
            ContrastAgent,
            DeveloperType,
        )

        self._paper_type = "Test Paper"
        self._paper_weight = 300
        self._exposure_time = 180.0
        self._metal_ratio = 0.5
        self._chemistry_type = ChemistryType.PLATINUM_PALLADIUM
        self._contrast_agent = ContrastAgent.NA2
        self._contrast_amount = 5.0
        self._developer = DeveloperType.POTASSIUM_OXALATE
        self._humidity = 50.0
        self._temperature = 21.0
        self._densities = None
        self._notes = None

    def with_paper(self, paper_type: str, weight: int = 300) -> "CalibrationRecordBuilder":
        """Set paper type and weight."""
        self._paper_type = paper_type
        self._paper_weight = weight
        return self

    def with_exposure(self, exposure_time: float) -> "CalibrationRecordBuilder":
        """Set exposure time."""
        self._exposure_time = exposure_time
        return self

    def with_metal_ratio(self, ratio: float) -> "CalibrationRecordBuilder":
        """Set metal ratio (0-1)."""
        self._metal_ratio = ratio
        return self

    def with_contrast(self, agent: str, amount: float) -> "CalibrationRecordBuilder":
        """Set contrast agent and amount."""
        from ptpd_calibration.core.types import ContrastAgent

        self._contrast_agent = ContrastAgent(agent)
        self._contrast_amount = amount
        return self

    def with_environment(
        self, humidity: float, temperature: float
    ) -> "CalibrationRecordBuilder":
        """Set environment conditions."""
        self._humidity = humidity
        self._temperature = temperature
        return self

    def with_densities(self, densities: list[float]) -> "CalibrationRecordBuilder":
        """Set measured densities."""
        self._densities = densities
        return self

    def with_notes(self, notes: str) -> "CalibrationRecordBuilder":
        """Set notes."""
        self._notes = notes
        return self

    def build(self):
        """Build a CalibrationRecord instance."""
        from ptpd_calibration.core.models import CalibrationRecord

        densities = self._densities or DensityBuilder().build()

        return CalibrationRecord(
            paper_type=self._paper_type,
            paper_weight=self._paper_weight,
            exposure_time=self._exposure_time,
            metal_ratio=self._metal_ratio,
            chemistry_type=self._chemistry_type,
            contrast_agent=self._contrast_agent,
            contrast_amount=self._contrast_amount,
            developer=self._developer,
            humidity=self._humidity,
            temperature=self._temperature,
            measured_densities=densities,
            notes=self._notes,
        )


class ImageBuilder:
    """Builder for test images."""

    def __init__(self):
        self._width = 100
        self._height = 100
        self._mode = "L"
        self._fill = "gradient"

    def with_size(self, width: int, height: int) -> "ImageBuilder":
        """Set image dimensions."""
        self._width = width
        self._height = height
        return self

    def with_mode(self, mode: str) -> "ImageBuilder":
        """Set color mode (L, RGB, RGBA)."""
        self._mode = mode
        return self

    def with_gradient(self) -> "ImageBuilder":
        """Fill with a gradient."""
        self._fill = "gradient"
        return self

    def with_random(self) -> "ImageBuilder":
        """Fill with random noise."""
        self._fill = "random"
        return self

    def with_solid(self, value: int = 128) -> "ImageBuilder":
        """Fill with solid color."""
        self._fill = ("solid", value)
        return self

    def with_step_tablet(self, num_patches: int = 21) -> "ImageBuilder":
        """Create a step tablet pattern."""
        self._fill = ("step_tablet", num_patches)
        return self

    def build(self) -> "Image":
        """Build the PIL Image."""
        from PIL import Image

        if self._fill == "gradient":
            if self._mode == "L":
                arr = np.tile(
                    np.linspace(0, 255, self._width).astype(np.uint8),
                    (self._height, 1),
                )
            else:
                gray = np.tile(
                    np.linspace(0, 255, self._width).astype(np.uint8),
                    (self._height, 1),
                )
                arr = np.stack([gray, gray, gray], axis=-1)

        elif self._fill == "random":
            if self._mode == "L":
                arr = np.random.randint(0, 255, (self._height, self._width), dtype=np.uint8)
            else:
                arr = np.random.randint(
                    0, 255, (self._height, self._width, 3), dtype=np.uint8
                )

        elif isinstance(self._fill, tuple) and self._fill[0] == "solid":
            value = self._fill[1]
            if self._mode == "L":
                arr = np.full((self._height, self._width), value, dtype=np.uint8)
            else:
                arr = np.full((self._height, self._width, 3), value, dtype=np.uint8)

        elif isinstance(self._fill, tuple) and self._fill[0] == "step_tablet":
            num_patches = self._fill[1]
            patch_width = self._width // num_patches
            arr = np.zeros((self._height, self._width), dtype=np.uint8)
            for i in range(num_patches):
                value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
                x_start = i * patch_width
                x_end = (i + 1) * patch_width
                arr[:, x_start:x_end] = value

        else:
            raise ValueError(f"Unknown fill type: {self._fill}")

        return Image.fromarray(arr, mode=self._mode)

    def build_path(self, tmp_path, filename: str = "test_image.png") -> "Path":
        """Build and save image, returning the path."""
        from pathlib import Path

        img = self.build()
        path = tmp_path / filename
        img.save(path)
        return path


def random_paper_type() -> str:
    """Generate a random paper type name."""
    papers = [
        "Arches Platine",
        "Bergger COT320",
        "Hahnemuhle Platinum Rag",
        "Revere Platinum",
        "Crane's Platinotype",
    ]
    return random.choice(papers)


def random_chemistry_type() -> str:
    """Generate a random chemistry type."""
    types = ["platinum_palladium", "palladium", "platinum", "kallitype"]
    return random.choice(types)


def random_densities(num_steps: int = 21) -> list[float]:
    """Generate random but realistic density measurements."""
    return DensityBuilder().with_steps(num_steps).with_noise(0.02).build()
