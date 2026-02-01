"""
CUPS printer driver for PTPD Calibration.

Provides printing functionality via CUPS (Common Unix Printing System)
for Linux and macOS systems.

Requires:
    - pycups: pip install pycups (Linux/macOS only)

Usage:
    from ptpd_calibration.integrations.hardware import CUPSPrinterDriver

    driver = CUPSPrinterDriver()
    if driver.connect():
        result = driver.print_image(PrintJob(
            name="Test Print",
            image_path="/path/to/negative.tiff",
            paper_size="8x10",
            resolution_dpi=2880,
        ))
        if result.success:
            print(f"Print job submitted: {result.job_id}")
"""

import platform
import time
from pathlib import Path
from typing import Any

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.exceptions import (
    PrinterError,
    PrinterNotFoundError,
    PrintJobError,
)
from ptpd_calibration.integrations.protocols import (
    DeviceInfo,
    DeviceStatus,
    PrintJob,
    PrintResult,
)

logger = get_logger(__name__)


def _import_cups() -> Any:
    """Lazy import for pycups."""
    if platform.system() == "Windows":
        raise ImportError("CUPS is not available on Windows. Use Win32PrinterDriver instead.")

    try:
        import cups

        return cups
    except ImportError as e:
        raise ImportError(
            "pycups is required for CUPS printing. Install with: pip install pycups"
        ) from e


class CUPSPrinterDriver:
    """CUPS-based printer driver for Linux and macOS.

    Communicates with the CUPS print server to manage print jobs
    for digital negative output.

    Attributes:
        status: Current device status.
        device_info: Printer information (None if not connected).
    """

    # Ink level threshold (percentage below which ink is considered "low")
    INK_LEVEL_LOW_THRESHOLD = 25

    # Paper size mappings (common names to CUPS names)
    PAPER_SIZE_MAP = {
        "4x5": "Custom.4x5in",
        "5x7": "5x7",
        "8x10": "Custom.8x10in",
        "11x14": "Custom.11x14in",
        "16x20": "Custom.16x20in",
        "letter": "Letter",
        "a4": "A4",
        "a3": "A3",
        "roll_13in": "Custom.13x999in",
        "roll_17in": "Custom.17x999in",
        "roll_24in": "Custom.24x999in",
    }

    def __init__(
        self,
        printer_name: str | None = None,
    ):
        """Initialize CUPS printer driver.

        Args:
            printer_name: Printer name. Uses system default if None.
        """
        settings = get_settings()
        integrations = settings.integrations

        self._printer_name = printer_name or integrations.default_printer_name
        self._cups_conn = None
        self._status = DeviceStatus.DISCONNECTED
        self._device_info: DeviceInfo | None = None
        self._ppd = None

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get printer information (None if not connected)."""
        return self._device_info

    def connect(self, printer_name: str | None = None) -> bool:
        """Connect to CUPS and select printer.

        Args:
            printer_name: Printer name. Uses system default if None.

        Returns:
            True if connection successful.

        Raises:
            PrinterNotFoundError: If specified printer not found.
            PrinterError: If CUPS connection fails.
        """
        cups = _import_cups()
        printer_name = printer_name or self._printer_name

        logger.info(f"Connecting to CUPS printer (name={printer_name})")

        try:
            self._cups_conn = cups.Connection()
        except Exception as e:
            raise PrinterError(
                f"Failed to connect to CUPS server: {e}",
                operation="connect",
            ) from e

        # Get available printers
        # Type guard: _cups_conn is guaranteed non-None after Connection() above
        if self._cups_conn is None:
            raise PrinterError(
                "CUPS connection not established",
                operation="connect",
            )
        printers = self._cups_conn.getPrinters()

        if not printers:
            raise PrinterError(
                "No printers available in CUPS",
                operation="connect",
            )

        # Select printer
        if printer_name:
            if printer_name not in printers:
                raise PrinterNotFoundError(
                    f"Printer '{printer_name}' not found",
                    printer_name=printer_name,
                    available_printers=list(printers.keys()),
                )
            self._printer_name = printer_name
        else:
            # Use system default
            default = self._cups_conn.getDefault()
            if default:
                self._printer_name = default
            else:
                # Use first available printer
                self._printer_name = list(printers.keys())[0]

        # Get printer info
        printer_info = printers[self._printer_name]
        self._device_info = self._extract_device_info(printer_info)
        self._status = DeviceStatus.CONNECTED

        logger.info(f"Connected to printer: {self._device_info}")
        return True

    def disconnect(self) -> None:
        """Disconnect from CUPS."""
        logger.info("Disconnecting from CUPS")
        self._cups_conn = None
        self._status = DeviceStatus.DISCONNECTED
        self._device_info = None
        self._ppd = None

    def print_image(self, job: PrintJob) -> PrintResult:
        """Print an image via CUPS.

        Args:
            job: Print job specification.

        Returns:
            PrintResult indicating success/failure.

        Raises:
            PrintJobError: If print submission fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            return PrintResult(
                success=False,
                error="Printer not connected",
            )

        if not self._cups_conn:
            return PrintResult(
                success=False,
                error="CUPS connection lost",
            )

        # Verify image exists
        image_path = Path(job.image_path)
        if not image_path.exists():
            return PrintResult(
                success=False,
                error=f"Image file not found: {job.image_path}",
            )

        logger.info(f"Submitting print job: {job.name}")
        self._status = DeviceStatus.BUSY

        start_time = time.time()

        try:
            # Build print options
            options = self._build_print_options(job)
            logger.debug(f"Print options: {options}")

            # Submit print job
            job_id = self._cups_conn.printFile(
                self._printer_name,
                str(image_path),
                job.name,
                options,
            )

            self._status = DeviceStatus.CONNECTED

            logger.info(f"Print job submitted: {job_id}")

            return PrintResult(
                success=True,
                job_id=str(job_id),
                pages_printed=job.copies,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            self._status = DeviceStatus.CONNECTED
            logger.error(f"Print job failed: {e}")

            raise PrintJobError(
                f"Failed to submit print job: {e}",
                job_name=job.name,
            ) from e

    def get_paper_sizes(self) -> list[str]:
        """Get supported paper sizes from printer PPD.

        Returns:
            List of supported paper size names.
        """
        if not self._cups_conn or not self._printer_name:
            return list(self.PAPER_SIZE_MAP.keys())

        try:
            cups = _import_cups()
            ppd_path = self._cups_conn.getPPD(self._printer_name)
            if ppd_path:
                ppd = cups.PPD(ppd_path)
                page_size_option = ppd.findOption("PageSize")
                if page_size_option:
                    return [opt["choice"] for opt in page_size_option]
        except Exception as e:
            logger.warning(f"Failed to get paper sizes from PPD: {e}")

        return list(self.PAPER_SIZE_MAP.keys())

    def get_resolutions(self) -> list[int]:
        """Get supported print resolutions.

        Returns:
            List of supported DPI values.
        """
        # Common Epson photo printer resolutions
        # Could be extended to query from PPD
        return [360, 720, 1440, 2880, 5760]

    def get_printer_status(self) -> dict[str, Any]:
        """Get detailed printer status.

        Returns:
            Dictionary with printer status information.
        """
        if not self._cups_conn or not self._printer_name:
            return {"status": "disconnected"}

        try:
            attrs = self._cups_conn.getPrinterAttributes(self._printer_name)
            return {
                "status": attrs.get("printer-state", "unknown"),
                "state_message": attrs.get("printer-state-message", ""),
                "accepting_jobs": attrs.get("printer-is-accepting-jobs", False),
                "queued_jobs": attrs.get("queued-job-count", 0),
            }
        except Exception:
            # Log full details for debugging, return generic message
            logger.exception("Failed to get printer status")
            return {"status": "unknown", "error": "Unable to retrieve printer status"}

    def get_ink_levels(self) -> dict[str, Any]:
        """Get ink/supply levels via IPP.

        Returns:
            Dictionary of ink colors and levels (if available).
        """
        if not self._cups_conn or not self._printer_name:
            return {}

        try:
            attrs = self._cups_conn.getPrinterAttributes(self._printer_name)
            supplies = attrs.get("printer-supply", [])

            if not supplies:
                return {"note": "Ink levels not available for this printer"}

            levels = {}
            for supply in supplies:
                parsed = self._parse_supply_string(supply)
                if parsed:
                    levels[parsed["name"]] = {
                        "level": parsed["level"],
                        "status": "ok" if parsed["level"] > self.INK_LEVEL_LOW_THRESHOLD else "low",
                    }

            return levels

        except Exception:
            # Log full details for debugging, return generic message
            logger.exception("Failed to get ink levels")
            return {"error": "Unable to retrieve ink levels"}

    def _extract_device_info(self, printer_info: dict) -> DeviceInfo:
        """Extract DeviceInfo from CUPS printer info.

        Args:
            printer_info: CUPS printer info dictionary.

        Returns:
            DeviceInfo instance.
        """
        # Extract make and model from printer info
        make_model = printer_info.get("printer-make-and-model", "Unknown Printer")

        # Try to split make and model
        parts = make_model.split(" ", 1)
        vendor = parts[0] if parts else "Unknown"
        model = parts[1] if len(parts) > 1 else make_model

        return DeviceInfo(
            vendor=vendor,
            model=model,
            serial_number=None,  # Not available via CUPS
            firmware_version=None,
            capabilities=self._determine_capabilities(printer_info),
        )

    def _determine_capabilities(self, printer_info: dict) -> list[str]:
        """Determine printer capabilities from CUPS info.

        Args:
            printer_info: CUPS printer info dictionary.

        Returns:
            List of capability strings.
        """
        capabilities = []

        # Check for color support
        color_supported = printer_info.get("color-supported", False)
        if color_supported:
            capabilities.append("color")
        capabilities.append("grayscale")

        # Check for duplex
        if "sides-supported" in printer_info:
            sides = printer_info["sides-supported"]
            if "two-sided" in str(sides):
                capabilities.append("duplex")

        # Assume high resolution for photo printers
        capabilities.append("high_resolution")

        return capabilities

    def _build_print_options(self, job: PrintJob) -> dict[str, str]:
        """Build CUPS print options dictionary.

        Args:
            job: Print job specification.

        Returns:
            Dictionary of CUPS options.

        Raises:
            PrintJobError: If paper size is invalid.
        """
        options: dict[str, str] = {}

        # Paper size - validate against whitelist for security
        if job.paper_size in self.PAPER_SIZE_MAP:
            paper_size = self.PAPER_SIZE_MAP[job.paper_size]
        else:
            # Only allow alphanumeric, dots, and 'x' for custom sizes
            import re

            if not re.match(r"^[a-zA-Z0-9.x]+$", job.paper_size):
                raise PrintJobError(
                    f"Invalid paper size format: {job.paper_size}",
                    job_name=job.name,
                )
            paper_size = job.paper_size
            logger.warning(f"Using custom paper size: {paper_size}")

        options["media"] = paper_size

        # Resolution
        options["Resolution"] = f"{job.resolution_dpi}dpi"

        # Color mode (usually grayscale for digital negatives)
        options["ColorModel"] = "Gray"

        # Quality
        options["Quality"] = "high"

        # Number of copies
        if job.copies > 1:
            options["copies"] = str(job.copies)

        return options

    def _parse_supply_string(self, supply: str) -> dict[str, Any] | None:
        """Parse CUPS printer-supply string.

        Args:
            supply: Supply string from CUPS.

        Returns:
            Parsed dictionary or None if parsing fails.
        """
        # Format varies by manufacturer
        # Common format: "type=ink;name=black;level=75"
        result: dict[str, Any] = {}

        try:
            for part in supply.split(";"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key.strip()] = value.strip()

            if "name" in result and "level" in result:
                try:
                    result["level"] = int(result["level"])
                except ValueError:
                    result["level"] = 0
                return result
        except Exception as e:
            logger.debug(f"Failed to parse supply string '{supply}': {e}")

        return None
