"""
Spectrophotometer integration for density and color measurements.

Provides abstract interface and concrete implementations for spectrophotometer devices.
Currently includes simulated X-Rite device support for testing and development.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MeasurementMode(str, Enum):
    """Measurement modes for spectrophotometer."""

    REFLECTION = "reflection"
    TRANSMISSION = "transmission"
    DENSITY = "density"


class ApertureSize(str, Enum):
    """Aperture sizes for measurement."""

    SMALL = "small"  # 2mm
    MEDIUM = "medium"  # 4mm
    LARGE = "large"  # 8mm


class ExportFormat(str, Enum):
    """Export formats for measurement data."""

    CGATS = "cgats"
    CSV = "csv"
    JSON = "json"
    XML = "xml"


@dataclass
class LABValue:
    """CIE L*a*b* color value."""

    L: float  # Lightness (0-100)
    a: float  # Green-Red axis
    b: float  # Blue-Yellow axis

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {"L": self.L, "a": self.a, "b": self.b}

    def delta_e(self, other: 'LABValue') -> float:
        """Calculate Delta E (CIE76) color difference."""
        return np.sqrt(
            (self.L - other.L) ** 2 +
            (self.a - other.a) ** 2 +
            (self.b - other.b) ** 2
        )


@dataclass
class SpectralData:
    """Spectral reflectance/transmittance data."""

    wavelengths: list[float]  # Wavelength values (nm)
    values: list[float]  # Reflectance/transmittance values (0-1)

    def to_dict(self) -> dict[str, list[float]]:
        """Convert to dictionary."""
        return {"wavelengths": self.wavelengths, "values": self.values}


class PatchMeasurement(BaseModel):
    """Single patch measurement result."""

    patch_id: str = Field(description="Patch identifier")
    density: float = Field(description="Optical density")
    lab: LABValue = Field(description="L*a*b* color values")
    rgb: tuple[int, int, int] | None = Field(default=None, description="RGB values (0-255)")
    spectral: SpectralData | None = Field(default=None, description="Spectral data")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class CalibrationResult(BaseModel):
    """Spectrophotometer calibration result."""

    success: bool = Field(description="Calibration success status")
    white_reference: LABValue | None = Field(default=None, description="White reference L*a*b*")
    black_reference: LABValue | None = Field(default=None, description="Black reference L*a*b*")
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = Field(default="", description="Status message")

    class Config:
        arbitrary_types_allowed = True


class SpectrophotometerInterface(ABC):
    """
    Abstract base class for spectrophotometer integrations.

    Defines the standard interface that all spectrophotometer implementations
    must follow. Concrete implementations should handle device-specific
    communication and data parsing.
    """

    def __init__(
        self,
        device_id: str | None = None,
        mode: MeasurementMode = MeasurementMode.REFLECTION,
        aperture: ApertureSize = ApertureSize.MEDIUM,
    ):
        """
        Initialize spectrophotometer interface.

        Args:
            device_id: Device identifier (serial number, USB path, etc.)
            mode: Measurement mode
            aperture: Aperture size for measurements
        """
        self.device_id = device_id
        self.mode = mode
        self.aperture = aperture
        self.is_connected = False
        self.is_calibrated = False
        self.last_calibration: CalibrationResult | None = None

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the spectrophotometer device.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device."""
        pass

    @abstractmethod
    def calibrate_device(
        self,
        white_tile: bool = True,
        black_trap: bool = True,
    ) -> CalibrationResult:
        """
        Calibrate the spectrophotometer.

        Args:
            white_tile: Calibrate using white reference tile
            black_trap: Calibrate using black trap

        Returns:
            CalibrationResult with calibration status and data
        """
        pass

    @abstractmethod
    def read_density(self, patch_id: str = "patch") -> float:
        """
        Read optical density from current position.

        Args:
            patch_id: Identifier for the patch being measured

        Returns:
            Optical density value
        """
        pass

    @abstractmethod
    def get_lab_values(self, patch_id: str = "patch") -> LABValue:
        """
        Get L*a*b* color values from current position.

        Args:
            patch_id: Identifier for the patch being measured

        Returns:
            LABValue with L*, a*, b* values
        """
        pass

    @abstractmethod
    def read_patch(self, patch_id: str = "patch") -> PatchMeasurement:
        """
        Read complete measurement (density, L*a*b*, spectral) from current position.

        Args:
            patch_id: Identifier for the patch being measured

        Returns:
            PatchMeasurement with all measurement data
        """
        pass

    @abstractmethod
    def read_strip(
        self,
        num_patches: int,
        patch_prefix: str = "patch",
        delay_seconds: float = 1.0,
    ) -> list[PatchMeasurement]:
        """
        Read a strip of patches.

        Args:
            num_patches: Number of patches in the strip
            patch_prefix: Prefix for patch IDs (will append index)
            delay_seconds: Delay between patch measurements

        Returns:
            List of PatchMeasurement objects
        """
        pass

    @abstractmethod
    def export_measurements(
        self,
        measurements: list[PatchMeasurement],
        output_path: Path,
        format: ExportFormat = ExportFormat.CGATS,
    ) -> Path:
        """
        Export measurements to file.

        Args:
            measurements: List of measurements to export
            output_path: Path for output file
            format: Export format

        Returns:
            Path to created file
        """
        pass


class XRiteIntegration(SpectrophotometerInterface):
    """
    X-Rite spectrophotometer integration (simulated).

    Simulates X-Rite i1Pro, i1iO, or similar devices for development and testing.
    In production, this would communicate with actual hardware via USB or SDK.
    """

    def __init__(
        self,
        device_id: str | None = None,
        mode: MeasurementMode = MeasurementMode.REFLECTION,
        aperture: ApertureSize = ApertureSize.MEDIUM,
        simulate: bool = True,
    ):
        """
        Initialize X-Rite spectrophotometer.

        Args:
            device_id: Device serial number
            mode: Measurement mode
            aperture: Aperture size
            simulate: If True, simulate device (for testing)
        """
        super().__init__(device_id, mode, aperture)
        self.simulate = simulate
        self.device_model = "i1Pro 3 Plus (Simulated)" if simulate else "i1Pro 3 Plus"

        # Simulated state
        self._white_reference = LABValue(L=95.0, a=0.0, b=0.0)
        self._black_reference = LABValue(L=5.0, a=0.0, b=0.0)

    def connect(self) -> bool:
        """Connect to X-Rite device."""
        logger.info(f"Connecting to {self.device_model}...")

        if self.simulate:
            # Simulate connection delay
            time.sleep(0.5)
            self.is_connected = True
            logger.info("Connected successfully (simulated)")
            return True

        # Real device connection would go here
        # e.g., using x-rite SDK or libusb
        logger.error("Real device connection not implemented")
        return False

    def disconnect(self) -> None:
        """Disconnect from device."""
        if self.is_connected:
            logger.info("Disconnecting from device...")
            self.is_connected = False
            self.is_calibrated = False

    def calibrate_device(
        self,
        white_tile: bool = True,
        black_trap: bool = True,
    ) -> CalibrationResult:
        """
        Calibrate the X-Rite spectrophotometer.

        Args:
            white_tile: Calibrate using white reference tile
            black_trap: Calibrate using black trap

        Returns:
            CalibrationResult with status
        """
        if not self.is_connected:
            return CalibrationResult(
                success=False,
                message="Device not connected"
            )

        logger.info("Calibrating device...")

        if self.simulate:
            # Simulate calibration process
            time.sleep(1.0)

            result = CalibrationResult(
                success=True,
                white_reference=self._white_reference if white_tile else None,
                black_reference=self._black_reference if black_trap else None,
                message="Calibration successful (simulated)"
            )

            self.is_calibrated = True
            self.last_calibration = result
            logger.info("Calibration complete")
            return result

        # Real calibration would go here
        return CalibrationResult(
            success=False,
            message="Real calibration not implemented"
        )

    def read_density(self, patch_id: str = "patch") -> float:
        """
        Read optical density.

        Args:
            patch_id: Patch identifier

        Returns:
            Optical density value
        """
        if not self.is_connected:
            raise ConnectionError("Device not connected")

        if not self.is_calibrated:
            logger.warning("Device not calibrated, results may be inaccurate")

        if self.simulate:
            # Simulate density reading with some noise
            # Base density varies by patch, add small random variation
            base_density = hash(patch_id) % 100 / 50.0  # 0.0 to 2.0 range
            noise = np.random.normal(0, 0.02)
            density = max(0.0, min(3.0, base_density + noise))

            logger.debug(f"Read density for {patch_id}: {density:.3f}")
            return density

        # Real device read would go here
        raise NotImplementedError("Real device reading not implemented")

    def get_lab_values(self, patch_id: str = "patch") -> LABValue:
        """
        Get L*a*b* color values.

        Args:
            patch_id: Patch identifier

        Returns:
            LABValue object
        """
        if not self.is_connected:
            raise ConnectionError("Device not connected")

        if not self.is_calibrated:
            logger.warning("Device not calibrated, results may be inaccurate")

        if self.simulate:
            # Simulate L*a*b* values based on patch_id hash
            seed = hash(patch_id)
            np.random.seed(seed % 2**32)

            # L* varies more (0-100)
            L = np.random.uniform(20, 90)
            # a* and b* centered around 0 with smaller range
            a = np.random.uniform(-20, 20)
            b = np.random.uniform(-20, 20)

            lab = LABValue(L=L, a=a, b=b)
            logger.debug(f"Read L*a*b* for {patch_id}: L={L:.1f}, a={a:.1f}, b={b:.1f}")
            return lab

        # Real device read would go here
        raise NotImplementedError("Real device reading not implemented")

    def _simulate_spectral_data(self, patch_id: str) -> SpectralData:
        """Simulate spectral reflectance data."""
        # Wavelengths from 400nm to 700nm in 10nm increments
        wavelengths = list(range(400, 710, 10))

        # Generate pseudo-random but consistent spectral curve
        seed = hash(patch_id)
        np.random.seed(seed % 2**32)

        # Create a smooth spectral curve
        base_curve = np.random.uniform(0.1, 0.9, len(wavelengths))
        # Smooth it
        kernel = np.array([0.25, 0.5, 0.25])
        values = np.convolve(base_curve, kernel, mode='same')
        values = np.clip(values, 0.0, 1.0)

        return SpectralData(
            wavelengths=wavelengths,
            values=values.tolist()
        )

    def read_patch(self, patch_id: str = "patch") -> PatchMeasurement:
        """
        Read complete patch measurement.

        Args:
            patch_id: Patch identifier

        Returns:
            PatchMeasurement with all data
        """
        density = self.read_density(patch_id)
        lab = self.get_lab_values(patch_id)

        # Convert L*a*b* to approximate RGB (simplified)
        # In production, use proper color conversion library
        rgb = self._lab_to_rgb_approximate(lab)

        spectral = None
        if self.simulate:
            spectral = self._simulate_spectral_data(patch_id)

        return PatchMeasurement(
            patch_id=patch_id,
            density=density,
            lab=lab,
            rgb=rgb,
            spectral=spectral
        )

    def _lab_to_rgb_approximate(self, lab: LABValue) -> tuple[int, int, int]:
        """Approximate L*a*b* to RGB conversion (simplified)."""
        # This is a very rough approximation for display purposes
        # Real conversion requires proper XYZ intermediate and white point

        # Map L* to grayscale as base
        gray = int(lab.L * 2.55)

        # Add chromatic components
        r = max(0, min(255, gray + int(lab.a * 2)))
        g = max(0, min(255, gray - int(lab.a * 1.5)))
        b = max(0, min(255, gray - int(lab.b * 2)))

        return (r, g, b)

    def read_strip(
        self,
        num_patches: int,
        patch_prefix: str = "patch",
        delay_seconds: float = 1.0,
    ) -> list[PatchMeasurement]:
        """
        Read a strip of patches.

        Args:
            num_patches: Number of patches
            patch_prefix: Prefix for patch IDs
            delay_seconds: Delay between measurements

        Returns:
            List of PatchMeasurement objects
        """
        if not self.is_connected:
            raise ConnectionError("Device not connected")

        logger.info(f"Reading strip of {num_patches} patches...")
        measurements = []

        for i in range(num_patches):
            patch_id = f"{patch_prefix}_{i+1:02d}"

            if self.simulate:
                # Simulate measurement time
                time.sleep(delay_seconds if i > 0 else 0)

            measurement = self.read_patch(patch_id)
            measurements.append(measurement)

            logger.debug(f"Measured {patch_id}: D={measurement.density:.3f}")

        logger.info(f"Strip reading complete: {len(measurements)} patches")
        return measurements

    def export_measurements(
        self,
        measurements: list[PatchMeasurement],
        output_path: Path,
        format: ExportFormat = ExportFormat.CGATS,
    ) -> Path:
        """
        Export measurements to file.

        Args:
            measurements: List of measurements
            output_path: Output file path
            format: Export format

        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ExportFormat.CGATS:
            return self._export_cgats(measurements, output_path)
        elif format == ExportFormat.CSV:
            return self._export_csv(measurements, output_path)
        elif format == ExportFormat.JSON:
            return self._export_json(measurements, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_cgats(self, measurements: list[PatchMeasurement], path: Path) -> Path:
        """Export to CGATS format."""

        with open(path, 'w', newline='') as f:
            # Write CGATS header
            f.write("CGATS.17\n")
            f.write(f"CREATED {datetime.now().isoformat()}\n")
            f.write(f"ORIGINATOR \"{self.device_model}\"\n")
            f.write("NUMBER_OF_FIELDS 7\n")
            f.write("BEGIN_DATA_FORMAT\n")
            f.write("SAMPLE_ID DENSITY LAB_L LAB_A LAB_B RGB_R RGB_G RGB_B\n")
            f.write("END_DATA_FORMAT\n")
            f.write(f"NUMBER_OF_SETS {len(measurements)}\n")
            f.write("BEGIN_DATA\n")

            # Write data
            for m in measurements:
                rgb = m.rgb or (0, 0, 0)
                f.write(f"{m.patch_id} {m.density:.4f} {m.lab.L:.2f} {m.lab.a:.2f} {m.lab.b:.2f} {rgb[0]} {rgb[1]} {rgb[2]}\n")

            f.write("END_DATA\n")

        logger.info(f"Exported {len(measurements)} measurements to CGATS: {path}")
        return path

    def _export_csv(self, measurements: list[PatchMeasurement], path: Path) -> Path:
        """Export to CSV format."""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['patch_id', 'density', 'L', 'a', 'b', 'R', 'G', 'B'])

            for m in measurements:
                rgb = m.rgb or (0, 0, 0)
                writer.writerow([
                    m.patch_id,
                    f"{m.density:.4f}",
                    f"{m.lab.L:.2f}",
                    f"{m.lab.a:.2f}",
                    f"{m.lab.b:.2f}",
                    rgb[0],
                    rgb[1],
                    rgb[2]
                ])

        logger.info(f"Exported {len(measurements)} measurements to CSV: {path}")
        return path

    def _export_json(self, measurements: list[PatchMeasurement], path: Path) -> Path:
        """Export to JSON format."""
        import json

        # Serialize calibration data if present
        calibration_data = None
        if self.last_calibration:
            cal_dict = self.last_calibration.dict()
            # Convert datetime to ISO format string
            if 'timestamp' in cal_dict and isinstance(cal_dict['timestamp'], datetime):
                cal_dict['timestamp'] = cal_dict['timestamp'].isoformat()
            calibration_data = cal_dict

        data = {
            "device": self.device_model,
            "mode": self.mode.value,
            "aperture": self.aperture.value,
            "calibration": calibration_data,
            "timestamp": datetime.now().isoformat(),
            "measurements": [
                {
                    "patch_id": m.patch_id,
                    "density": m.density,
                    "lab": m.lab.to_dict(),
                    "rgb": m.rgb,
                    "spectral": m.spectral.to_dict() if m.spectral else None,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in measurements
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(measurements)} measurements to JSON: {path}")
        return path
