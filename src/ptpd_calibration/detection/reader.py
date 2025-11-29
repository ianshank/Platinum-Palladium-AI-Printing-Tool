"""
High-level step tablet reader combining detection and extraction.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ptpd_calibration.config import TabletType, get_settings
from ptpd_calibration.core.models import DensityMeasurement, ExtractionResult, StepTabletResult
from ptpd_calibration.core.types import MeasurementUnit
from ptpd_calibration.detection.detector import StepTabletDetector
from ptpd_calibration.detection.extractor import DensityExtractor
from ptpd_calibration.detection.scanner import ScannerCalibration


# Step tablet specifications
TABLET_SPECS: dict[TabletType, dict] = {
    TabletType.STOUFFER_21: {
        "num_steps": 21,
        "step_increment": 0.15,  # density increment per step
        "base_density": 0.05,
    },
    TabletType.STOUFFER_31: {
        "num_steps": 31,
        "step_increment": 0.10,
        "base_density": 0.05,
    },
    TabletType.STOUFFER_41: {
        "num_steps": 41,
        "step_increment": 0.075,
        "base_density": 0.05,
    },
    TabletType.CUSTOM: {
        "num_steps": None,  # Auto-detect
        "step_increment": None,
        "base_density": None,
    },
}


class StepTabletReader:
    """
    Complete step tablet reader for Pt/Pd calibration.

    Combines detection, extraction, and optional scanner calibration
    for accurate density measurements from scanned step tablets.
    """

    def __init__(
        self,
        tablet_type: TabletType = TabletType.STOUFFER_21,
        scanner_profile: Optional[ScannerCalibration] = None,
    ):
        """
        Initialize the reader.

        Args:
            tablet_type: Type of step tablet being used.
            scanner_profile: Optional scanner calibration for correction.
        """
        self.tablet_type = tablet_type
        self.tablet_spec = TABLET_SPECS[tablet_type]
        self.scanner_profile = scanner_profile

        settings = get_settings()
        self.detector = StepTabletDetector(settings.detection)
        self.extractor = DensityExtractor(settings.extraction)

    def read(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        reference_white: Optional[tuple[float, float, float]] = None,
    ) -> StepTabletResult:
        """
        Read a step tablet scan.

        Args:
            image: Input scan as array, PIL Image, or path.
            reference_white: Optional white reference RGB values.

        Returns:
            StepTabletResult with extraction and measurements.
        """
        # Load image
        img_array = self._load_image(image)

        # Apply scanner correction if available
        if self.scanner_profile is not None:
            img_array = self.scanner_profile.apply_correction(img_array)

        # Detect step tablet
        num_patches = self.tablet_spec.get("num_steps")
        detection = self.detector.detect(img_array, num_patches)

        # Extract densities
        extraction = self.extractor.extract(img_array, detection, reference_white)

        # Create measurements
        measurements = self._create_measurements(extraction)

        return StepTabletResult(
            extraction=extraction,
            measurements=measurements,
        )

    def read_from_path(self, path: Union[Path, str]) -> StepTabletResult:
        """Convenience method to read from file path."""
        return self.read(Path(path))

    def _load_image(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        """Load image from various sources."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            return np.array(image)
        if isinstance(image, (Path, str)):
            pil_img = Image.open(image)
            return np.array(pil_img)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _create_measurements(self, extraction: ExtractionResult) -> list[DensityMeasurement]:
        """Create density measurements from extraction."""
        measurements = []
        num_patches = len(extraction.patches)

        for i, patch in enumerate(extraction.patches):
            # Calculate input value (0 = paper white, 1 = max black)
            input_value = i / (num_patches - 1) if num_patches > 1 else 0.0

            # Get density (with fallback)
            density = patch.density if patch.density is not None else 0.0

            measurement = DensityMeasurement(
                step=i,
                input_value=input_value,
                density=density,
                lab=patch.lab_mean,
                unit=MeasurementUnit.VISUAL_DENSITY,
            )
            measurements.append(measurement)

        return measurements

    @classmethod
    def get_supported_tablets(cls) -> list[TabletType]:
        """Get list of supported tablet types."""
        return list(TABLET_SPECS.keys())

    @classmethod
    def get_tablet_info(cls, tablet_type: TabletType) -> dict:
        """Get specification for a tablet type."""
        return TABLET_SPECS.get(tablet_type, TABLET_SPECS[TabletType.CUSTOM])
