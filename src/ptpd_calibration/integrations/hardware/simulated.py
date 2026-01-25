"""
Simulated hardware devices for testing and development.

Provides mock implementations of hardware protocols that generate
realistic but synthetic data. Useful for:
- Development without physical hardware
- Automated testing
- Demonstrations and tutorials

All simulated devices follow the protocols defined in protocols.py.
"""

import random
import time
from datetime import datetime, timezone
from typing import Any

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.constants import (
    DEFAULT_SPECTRAL_END_NM,
    DEFAULT_SPECTRAL_INTERVAL_NM,
    # Spectral constants
    DEFAULT_SPECTRAL_START_NM,
    INK_LEVEL_LOW_THRESHOLD,
    SIMULATED_CALIBRATE_BLACK_DELAY_SEC,
    SIMULATED_CALIBRATE_WHITE_DELAY_SEC,
    # Simulation timing constants
    SIMULATED_CONNECT_DELAY_SEC,
    SIMULATED_DENSITY_GAMMA,
    # Ink level ranges
    SIMULATED_INK_LEVEL_RANGES,
    SIMULATED_LAB_A_STD_DEV,
    SIMULATED_LAB_B_MEAN,
    SIMULATED_LAB_B_STD_DEV,
    SIMULATED_LAB_L_BASE,
    SIMULATED_LAB_L_SLOPE,
    SIMULATED_MAX_DENSITY_CLAMP,
    SIMULATED_MEASURE_DELAY_SEC,
    SIMULATED_PRINT_BASE_TIME_SEC,
    SIMULATED_PRINT_CONNECT_DELAY_SEC,
    SIMULATED_PRINT_MAX_TIME_SEC,
    SIMULATED_PRINT_RESOLUTION_REF_DPI,
    SIMULATED_PRINTER_FIRMWARE,
    SIMULATED_PRINTER_MODEL,
    SIMULATED_PRINTER_SERIAL,
    SIMULATED_PRINTER_VENDOR,
    SIMULATED_REFLECTANCE_BASE,
    SIMULATED_REFLECTANCE_RANGE,
    SIMULATED_REFLECTANCE_SLOPE,
    SIMULATED_REFLECTANCE_WAVELENGTH_REF,
    SIMULATED_SPECTRAL_DELAY_SEC,
    SIMULATED_SPECTRO_FIRMWARE,
    SIMULATED_SPECTRO_MODEL,
    SIMULATED_SPECTRO_SERIAL,
    # Device identifiers
    SIMULATED_SPECTRO_VENDOR,
    # Simulation measurement constants
    SIMULATED_STEP_TABLET_STEPS,
    # Paper and resolution constants
    STANDARD_PAPER_SIZES,
    STANDARD_RESOLUTIONS,
)
from ptpd_calibration.integrations.protocols import (
    DensityMeasurement,
    DeviceInfo,
    DeviceStatus,
    PrintJob,
    PrintResult,
    SpectralData,
)

logger = get_logger(__name__)


class SimulatedSpectrophotometer:
    """Simulated spectrophotometer for testing.

    Generates realistic density and spectral measurements with
    configurable noise and response characteristics.

    Attributes:
        simulate_delay: Add realistic delays to operations.
        noise_level: Standard deviation of measurement noise.
        base_density: Base density for paper white.
        max_density: Maximum measurable density.
    """

    def __init__(
        self,
        simulate_delay: bool = True,
        noise_level: float = 0.02,
        base_density: float = 0.08,
        max_density: float = 2.5,
    ):
        """Initialize simulated spectrophotometer.

        Args:
            simulate_delay: Add delays to simulate real device timing.
            noise_level: Measurement noise standard deviation.
            base_density: Paper base density (Dmin).
            max_density: Maximum density (Dmax).
        """
        self._simulate_delay = simulate_delay
        self._noise_level = noise_level
        self._base_density = base_density
        self._max_density = max_density

        self._status = DeviceStatus.DISCONNECTED
        self._device_info: DeviceInfo | None = None
        self._calibrated = False
        self._measurement_count = 0

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information."""
        return self._device_info

    def connect(
        self,
        port: str | None = None,  # noqa: ARG002
        timeout: float = 5.0,  # noqa: ARG002
    ) -> bool:
        """Simulate device connection.

        Args:
            port: Ignored for simulation.
            timeout: Ignored for simulation.

        Returns:
            Always True for simulation.
        """
        del port, timeout  # Unused in simulation
        logger.info("Simulated spectrophotometer connecting...")
        self._status = DeviceStatus.CONNECTING

        if self._simulate_delay:
            time.sleep(SIMULATED_CONNECT_DELAY_SEC)

        self._device_info = DeviceInfo(
            vendor=SIMULATED_SPECTRO_VENDOR,
            model=SIMULATED_SPECTRO_MODEL,
            serial_number=SIMULATED_SPECTRO_SERIAL,
            firmware_version=SIMULATED_SPECTRO_FIRMWARE,
            capabilities=[
                "density",
                "lab",
                "spectral",
                "reflection",
                "transmission",
            ],
        )
        self._status = DeviceStatus.CONNECTED
        logger.info(f"Connected to {self._device_info}")
        return True

    def disconnect(self) -> None:
        """Disconnect from simulated device."""
        logger.info("Simulated spectrophotometer disconnecting...")
        self._status = DeviceStatus.DISCONNECTED
        self._device_info = None
        self._calibrated = False

    def calibrate_white(self) -> bool:
        """Simulate white reference calibration.

        Returns:
            Always True for simulation.

        Raises:
            RuntimeError: If not connected.
        """
        if self._status != DeviceStatus.CONNECTED:
            logger.error("Cannot calibrate: not connected")
            raise RuntimeError("Cannot calibrate: device not connected")

        logger.info("Simulated white calibration...")
        self._status = DeviceStatus.CALIBRATING

        if self._simulate_delay:
            time.sleep(SIMULATED_CALIBRATE_WHITE_DELAY_SEC)

        self._calibrated = True
        self._status = DeviceStatus.CONNECTED
        logger.info("White calibration complete")
        return True

    def calibrate_black(self) -> bool:
        """Simulate black reference calibration.

        Returns:
            Always True for simulation.
        """
        if self._status != DeviceStatus.CONNECTED:
            logger.warning("Cannot calibrate: not connected")
            return False

        logger.info("Simulated black calibration...")
        self._status = DeviceStatus.CALIBRATING

        if self._simulate_delay:
            time.sleep(SIMULATED_CALIBRATE_BLACK_DELAY_SEC)

        self._status = DeviceStatus.CONNECTED
        logger.info("Black calibration complete")
        return True

    def read_density(self) -> DensityMeasurement:
        """Read simulated density measurement.

        Generates a realistic density value with noise. Cycles through
        a range of densities to simulate step tablet measurements.

        Returns:
            Simulated density measurement.

        Raises:
            RuntimeError: If not connected.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise RuntimeError("Device not connected")

        self._status = DeviceStatus.MEASURING

        if self._simulate_delay:
            time.sleep(SIMULATED_MEASURE_DELAY_SEC)

        # Generate density based on measurement count (simulates step tablet)
        step = self._measurement_count % SIMULATED_STEP_TABLET_STEPS
        t = step / (SIMULATED_STEP_TABLET_STEPS - 1)  # 0 to 1

        # Simulate photographic response curve
        base_density = self._base_density + (
            (self._max_density - self._base_density) * (t ** SIMULATED_DENSITY_GAMMA)
        )

        # Add noise
        noise = random.gauss(0, self._noise_level)
        density = max(0, min(SIMULATED_MAX_DENSITY_CLAMP, base_density + noise))

        # Generate corresponding Lab values
        # L* decreases with density (roughly 100 - density * 40)
        lab_l = max(0, min(100, SIMULATED_LAB_L_BASE - density * SIMULATED_LAB_L_SLOPE + random.gauss(0, 1)))
        lab_a = random.gauss(0, SIMULATED_LAB_A_STD_DEV)  # Neutral gray should be near 0
        lab_b = random.gauss(SIMULATED_LAB_B_MEAN, SIMULATED_LAB_B_STD_DEV)  # Slight warm bias typical of Pt/Pd

        self._measurement_count += 1
        self._status = DeviceStatus.CONNECTED

        logger.debug(f"Simulated measurement: D={density:.3f}, L*={lab_l:.1f}")

        return DensityMeasurement(
            density=round(density, 4),
            lab_l=round(lab_l, 2),
            lab_a=round(lab_a, 2),
            lab_b=round(lab_b, 2),
            status_a_density=round(density * 0.95, 4),  # Slight difference
            timestamp=datetime.now(timezone.utc),
            aperture_size="medium",
            measurement_mode="reflection",
        )

    def read_spectral(self) -> SpectralData:
        """Read simulated spectral data.

        Generates a realistic reflectance spectrum.

        Returns:
            Simulated spectral data.

        Raises:
            RuntimeError: If not connected.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise RuntimeError("Device not connected")

        self._status = DeviceStatus.MEASURING

        if self._simulate_delay:
            time.sleep(SIMULATED_SPECTRAL_DELAY_SEC)

        # Generate wavelengths from start to end in interval steps
        start_nm = DEFAULT_SPECTRAL_START_NM
        end_nm = DEFAULT_SPECTRAL_END_NM
        interval_nm = DEFAULT_SPECTRAL_INTERVAL_NM

        wavelengths: list[float] = []
        values: list[float] = []

        nm = start_nm
        while nm <= end_nm:
            wavelengths.append(nm)

            # Simulate neutral gray reflectance spectrum
            # Real Pt/Pd prints have slight warm tone (higher in red)
            base_reflectance = (
                SIMULATED_REFLECTANCE_BASE +
                SIMULATED_REFLECTANCE_SLOPE * (
                    (nm - SIMULATED_REFLECTANCE_WAVELENGTH_REF) / SIMULATED_REFLECTANCE_RANGE
                )
            )
            noise = random.gauss(0, self._noise_level)
            reflectance = max(0, min(1, base_reflectance + noise))
            values.append(round(reflectance, 4))

            nm += interval_nm

        self._status = DeviceStatus.CONNECTED

        return SpectralData(
            wavelengths=wavelengths,
            values=values,
            start_nm=start_nm,
            end_nm=end_nm,
            interval_nm=interval_nm,
        )

    def reset_measurement_count(self) -> None:
        """Reset the measurement counter for a new step tablet."""
        self._measurement_count = 0


class SimulatedPrinter:
    """Simulated printer for testing.

    Simulates print operations without actually printing.

    Attributes:
        simulate_delay: Add realistic delays to operations.
        failure_rate: Probability of simulated print failure (0-1).
    """

    def __init__(
        self,
        simulate_delay: bool = True,
        failure_rate: float = 0.0,
    ):
        """Initialize simulated printer.

        Args:
            simulate_delay: Add delays to simulate real printing.
            failure_rate: Probability of print failure (for testing error handling).
        """
        self._simulate_delay = simulate_delay
        self._failure_rate = failure_rate

        self._status = DeviceStatus.DISCONNECTED
        self._device_info: DeviceInfo | None = None
        self._job_counter = 0

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information."""
        return self._device_info

    def connect(self, printer_name: str | None = None) -> bool:  # noqa: ARG002
        """Simulate printer connection.

        Args:
            printer_name: Ignored for simulation.

        Returns:
            Always True for simulation.
        """
        del printer_name  # Unused in simulation
        logger.info("Simulated printer connecting...")

        if self._simulate_delay:
            time.sleep(SIMULATED_PRINT_CONNECT_DELAY_SEC)

        self._device_info = DeviceInfo(
            vendor=SIMULATED_PRINTER_VENDOR,
            model=SIMULATED_PRINTER_MODEL,
            serial_number=SIMULATED_PRINTER_SERIAL,
            firmware_version=SIMULATED_PRINTER_FIRMWARE,
            capabilities=[
                "color",
                "grayscale",
                "high_resolution",
                "roll_paper",
                "sheet_paper",
            ],
        )
        self._status = DeviceStatus.CONNECTED
        logger.info(f"Connected to {self._device_info}")
        return True

    def disconnect(self) -> None:
        """Disconnect from simulated printer."""
        logger.info("Simulated printer disconnecting...")
        self._status = DeviceStatus.DISCONNECTED
        self._device_info = None

    def print_image(self, job: PrintJob) -> PrintResult:
        """Simulate printing an image.

        Args:
            job: Print job specification.

        Returns:
            Simulated print result.
        """
        if self._status != DeviceStatus.CONNECTED:
            return PrintResult(
                success=False,
                error="Printer not connected",
            )

        self._status = DeviceStatus.BUSY
        self._job_counter += 1
        job_id = f"sim-job-{self._job_counter:04d}"

        logger.info(f"Simulated print job {job_id}: {job.name}")

        start_time = time.time()

        if self._simulate_delay:
            # Simulate print time based on resolution
            resolution_factor = job.resolution_dpi / SIMULATED_PRINT_RESOLUTION_REF_DPI
            simulated_print_time = SIMULATED_PRINT_BASE_TIME_SEC * resolution_factor
            time.sleep(min(simulated_print_time, SIMULATED_PRINT_MAX_TIME_SEC))

        # Simulate occasional failures if configured
        if self._failure_rate > 0 and random.random() < self._failure_rate:
            self._status = DeviceStatus.CONNECTED
            return PrintResult(
                success=False,
                job_id=job_id,
                error="Simulated print failure",
                duration_seconds=time.time() - start_time,
            )

        self._status = DeviceStatus.CONNECTED

        return PrintResult(
            success=True,
            job_id=job_id,
            pages_printed=job.copies,
            duration_seconds=time.time() - start_time,
        )

    def get_paper_sizes(self) -> list[str]:
        """Get simulated paper sizes.

        Returns:
            List of supported paper size names.
        """
        return list(STANDARD_PAPER_SIZES)

    def get_resolutions(self) -> list[int]:
        """Get simulated resolutions.

        Returns:
            List of supported DPI values.
        """
        return list(STANDARD_RESOLUTIONS)

    def get_ink_levels(self) -> dict[str, Any]:
        """Get simulated ink levels.

        Returns:
            Dictionary of ink colors and levels.
        """
        if self._status != DeviceStatus.CONNECTED:
            return {}

        levels: dict[str, Any] = {}
        for color, (min_level, max_level) in SIMULATED_INK_LEVEL_RANGES.items():
            level = random.randint(min_level, max_level)
            levels[color] = {
                "level": level,
                "status": "ok" if level > INK_LEVEL_LOW_THRESHOLD else "low",
            }
        return levels
