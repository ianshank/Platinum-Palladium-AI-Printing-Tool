"""
Hardware device protocols for PTPD Calibration System.

Defines abstract protocols for hardware integration including
spectrophotometers and printers. Implementations can be real
hardware drivers or simulations for testing.

Usage:
    from ptpd_calibration.integrations.protocols import (
        SpectrophotometerProtocol,
        PrinterProtocol,
        DeviceStatus,
    )

    class XRiteI1Pro(SpectrophotometerProtocol):
        # Implement all required methods
        ...

    # Use in your code
    def measure_density(spectro: SpectrophotometerProtocol):
        if spectro.status != DeviceStatus.CONNECTED:
            spectro.connect()
        return spectro.read_density()
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field


class DeviceStatus(str, Enum):
    """Device connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CALIBRATING = "calibrating"
    MEASURING = "measuring"
    ERROR = "error"
    BUSY = "busy"


class DeviceInfo(BaseModel):
    """Information about a connected device.

    Attributes:
        vendor: Device manufacturer.
        model: Device model name.
        serial_number: Device serial number (if available).
        firmware_version: Device firmware version (if available).
        capabilities: List of device capabilities.
    """

    vendor: str
    model: str
    serial_number: str | None = None
    firmware_version: str | None = None
    capabilities: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.vendor} {self.model}"


class DensityMeasurement(BaseModel):
    """Density measurement result from spectrophotometer.

    Attributes:
        density: Visual density value.
        lab_l: L* value in CIE Lab color space.
        lab_a: a* value in CIE Lab color space.
        lab_b: b* value in CIE Lab color space.
        status_a_density: Status A density (for photographic measurements).
        timestamp: When the measurement was taken.
        aperture_size: Aperture size used (if applicable).
        measurement_mode: Mode used (reflection, transmission, etc.).
    """

    density: float = Field(ge=0.0, le=5.0)
    lab_l: float = Field(ge=0.0, le=100.0)
    lab_a: float = Field(ge=-128.0, le=128.0)
    lab_b: float = Field(ge=-128.0, le=128.0)
    status_a_density: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    aperture_size: str | None = None
    measurement_mode: str = "reflection"

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Convert Lab to XYZ color space."""
        # CIE Lab to XYZ conversion
        fy = (self.lab_l + 16) / 116
        fx = self.lab_a / 500 + fy
        fz = fy - self.lab_b / 200

        def f_inv(t: float) -> float:
            delta = 6 / 29
            if t > delta:
                return t**3
            else:
                return 3 * delta**2 * (t - 4 / 29)

        # D50 illuminant reference
        Xn, Yn, Zn = 96.422, 100.0, 82.521

        X = Xn * f_inv(fx)
        Y = Yn * f_inv(fy)
        Z = Zn * f_inv(fz)

        return X, Y, Z


class SpectralData(BaseModel):
    """Full spectral measurement data.

    Attributes:
        wavelengths: Array of wavelength values (nm).
        values: Reflectance or transmittance values at each wavelength.
        start_nm: Starting wavelength.
        end_nm: Ending wavelength.
        interval_nm: Wavelength interval.
    """

    wavelengths: list[float]
    values: list[float]
    start_nm: float = 380.0
    end_nm: float = 730.0
    interval_nm: float = 10.0

    def __len__(self) -> int:
        return len(self.values)


class PrintJob(BaseModel):
    """Print job specification.

    Attributes:
        name: Job name for identification.
        image_path: Path to the image file to print.
        paper_size: Paper size (e.g., "8x10", "letter", "a4").
        resolution_dpi: Print resolution.
        copies: Number of copies.
        color_profile: ICC profile to use (if any).
        paper_type: Type of paper for profile selection.
    """

    name: str
    image_path: str
    paper_size: str = "8x10"
    resolution_dpi: int = Field(default=2880, ge=360, le=5760)
    copies: int = Field(default=1, ge=1)
    color_profile: str | None = None
    paper_type: str | None = None


class PrintResult(BaseModel):
    """Result of a print operation.

    Attributes:
        success: Whether printing succeeded.
        job_id: Unique identifier for the print job.
        pages_printed: Number of pages actually printed.
        error: Error message if printing failed.
        duration_seconds: Time taken to print.
    """

    success: bool
    job_id: str | None = None
    pages_printed: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class SpectrophotometerProtocol(Protocol):
    """Protocol for spectrophotometer device integration.

    Implementations must provide all methods. The protocol supports
    both simulation and real hardware implementations.

    Example implementation:
        class SimulatedSpectro:
            @property
            def status(self) -> DeviceStatus:
                return self._status

            @property
            def device_info(self) -> DeviceInfo | None:
                return self._device_info if self._connected else None

            def connect(self, port: str | None = None, timeout: float = 5.0) -> bool:
                self._connected = True
                return True

            def disconnect(self) -> None:
                self._connected = False

            def calibrate_white(self) -> bool:
                return True

            def calibrate_black(self) -> bool:
                return True

            def read_density(self) -> DensityMeasurement:
                return DensityMeasurement(density=1.0, lab_l=50, lab_a=0, lab_b=0)

            def read_spectral(self) -> SpectralData:
                return SpectralData(
                    wavelengths=list(range(380, 740, 10)),
                    values=[0.5] * 36
                )
    """

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        ...

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information (None if not connected)."""
        ...

    def connect(
        self,
        port: str | None = None,
        timeout: float = 5.0,
    ) -> bool:
        """Connect to the device.

        Args:
            port: Serial port or USB path. If None, auto-detect.
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the device."""
        ...

    def calibrate_white(self) -> bool:
        """Calibrate on white reference.

        Returns:
            True if calibration successful.
        """
        ...

    def calibrate_black(self) -> bool:
        """Calibrate on black reference (if supported).

        Returns:
            True if calibration successful.
        """
        ...

    def read_density(self) -> DensityMeasurement:
        """Read a single density measurement.

        Returns:
            DensityMeasurement with density and color data.

        Raises:
            RuntimeError: If device not connected or measurement failed.
        """
        ...

    def read_spectral(self) -> SpectralData:
        """Read full spectral data.

        Returns:
            SpectralData with wavelengths and values.

        Raises:
            RuntimeError: If device not connected or not supported.
        """
        ...


@runtime_checkable
class PrinterProtocol(Protocol):
    """Protocol for printer device integration.

    Implementations must provide all methods for printing
    digital negatives and test targets.
    """

    @property
    def status(self) -> DeviceStatus:
        """Get current printer status."""
        ...

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get printer information."""
        ...

    def connect(self, printer_name: str | None = None) -> bool:
        """Connect to printer.

        Args:
            printer_name: System printer name. If None, use default.

        Returns:
            True if connection successful.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from printer."""
        ...

    def print_image(
        self,
        job: PrintJob,
    ) -> PrintResult:
        """Print an image.

        Args:
            job: Print job specification.

        Returns:
            PrintResult indicating success/failure.
        """
        ...

    def get_paper_sizes(self) -> list[str]:
        """Get supported paper sizes.

        Returns:
            List of supported paper size names.
        """
        ...

    def get_resolutions(self) -> list[int]:
        """Get supported print resolutions.

        Returns:
            List of supported DPI values.
        """
        ...


class DeviceManager(Generic[T]):
    """Generic device manager for hardware devices.

    Manages connection lifecycle, auto-reconnection, and
    provides a consistent interface for device operations.

    Example:
        manager = DeviceManager(SimulatedSpectro())
        with manager:
            measurement = manager.device.read_density()
    """

    def __init__(
        self,
        device: T,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ):
        """Initialize device manager.

        Args:
            device: Device instance implementing appropriate protocol.
            auto_reconnect: Automatically reconnect on connection loss.
            max_reconnect_attempts: Max reconnection attempts.
        """
        self.device = device
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self._connected = False

    def __enter__(self) -> "DeviceManager[T]":
        """Enter context - connect to device."""
        if hasattr(self.device, "connect"):
            self._connected = self.device.connect()  # type: ignore
        return self

    def __exit__(self, *args) -> None:
        """Exit context - disconnect from device."""
        if hasattr(self.device, "disconnect"):
            self.device.disconnect()  # type: ignore
        self._connected = False

    def ensure_connected(self) -> bool:
        """Ensure device is connected, reconnecting if needed.

        Returns:
            True if device is connected.
        """
        if hasattr(self.device, "status"):
            status = self.device.status  # type: ignore
            if status == DeviceStatus.CONNECTED:
                return True

        if self.auto_reconnect and hasattr(self.device, "connect"):
            for _attempt in range(self.max_reconnect_attempts):
                if self.device.connect():  # type: ignore
                    return True

        return False
