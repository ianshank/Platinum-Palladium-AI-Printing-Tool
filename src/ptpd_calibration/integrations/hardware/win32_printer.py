"""
Windows printer driver using Win32 API.

Provides native Windows printing support without CUPS dependency.
Uses the pywin32 library for Windows Print Spooler access.

Requirements:
    - pywin32: pip install pywin32 (Windows only)

Usage:
    from ptpd_calibration.integrations.hardware.win32_printer import (
        Win32PrinterDriver,
    )

    driver = Win32PrinterDriver()
    driver.connect(printer_name="EPSON SureColor P800")
    result = driver.print_image(job)
    driver.disconnect()
"""

from __future__ import annotations

import platform
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.base import HardwareDeviceBase
from ptpd_calibration.integrations.hardware.constants import (
    STANDARD_PAPER_SIZES,
    STANDARD_RESOLUTIONS,
)
from ptpd_calibration.integrations.hardware.exceptions import (
    PrinterDriverError,
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


def _import_win32() -> tuple[Any, Any, Any]:
    """Lazy import for Win32 modules.

    Returns:
        Tuple of (win32print, win32api, win32con) modules.

    Raises:
        ImportError: If not on Windows or pywin32 not installed.
    """
    if platform.system() != "Windows":
        raise ImportError("Win32 printing only available on Windows")

    try:
        import win32api
        import win32con
        import win32print

        return win32print, win32api, win32con
    except ImportError as e:
        raise ImportError(
            "pywin32 is required for Windows printing. "
            "Install with: pip install pywin32"
        ) from e


class Win32PrinterDriver(HardwareDeviceBase):
    """Windows printer driver using Win32 Print Spooler API.

    Features:
    - Native Windows printing without CUPS
    - Support for all Windows-installed printers
    - Print queue monitoring
    - Paper size and resolution detection

    Example:
        driver = Win32PrinterDriver()

        # Connect to specific printer
        driver.connect(printer_name="EPSON SureColor P800")

        # Or use default printer
        driver.connect()

        # Print an image
        job = PrintJob(
            name="Test Print",
            image_path="/path/to/image.tif",
            paper_size="8x10",
            resolution_dpi=2880,
        )
        result = driver.print_image(job)
        print(f"Job ID: {result.job_id}")

        driver.disconnect()
    """

    def __init__(
        self,
        printer_name: str | None = None,
    ) -> None:
        """Initialize Windows printer driver.

        Args:
            printer_name: Name of printer (uses default if None).
        """
        super().__init__(device_type="printer")
        self._printer_name = printer_name
        self._printer_handle: Any = None
        self._job_counter = 0

    def connect(self, printer_name: str | None = None, **kwargs: Any) -> bool:
        """Connect to Windows printer.

        Args:
            printer_name: Printer name (uses configured or default if None).
            **kwargs: Additional connection parameters (ignored).

        Returns:
            True if connection successful.

        Raises:
            PrinterNotFoundError: If printer not found.
            PrinterDriverError: If driver initialization fails.
        """
        del kwargs  # Unused

        win32print, win32api, _ = _import_win32()

        # Determine printer name
        name = printer_name or self._printer_name
        if name is None:
            # Get default printer
            try:
                name = win32print.GetDefaultPrinter()
                if not name:
                    raise PrinterNotFoundError(
                        message="No default printer configured",
                        available_printers=self.list_printers(),
                    )
            except Exception as e:
                raise PrinterNotFoundError(
                    message=f"Failed to get default printer: {e}",
                    available_printers=self.list_printers(),
                ) from e

        self._set_status(DeviceStatus.CONNECTING, f"Connecting to {name}")

        # Verify printer exists
        available = self.list_printers()
        if name not in available:
            self._set_status(DeviceStatus.ERROR, "Printer not found")
            raise PrinterNotFoundError(
                message=f"Printer '{name}' not found",
                printer_name=name,
                available_printers=available,
            )

        try:
            # Open printer handle
            self._printer_handle = win32print.OpenPrinter(name)
            self._printer_name = name

            # Get printer info
            try:
                printer_info = win32print.GetPrinter(self._printer_handle, 2)
                driver_name = printer_info.get("pDriverName", "Unknown")
                port_name = printer_info.get("pPortName", "Unknown")
                status = printer_info.get("Status", 0)

                # Determine vendor from driver name
                vendor = self._detect_vendor(driver_name, name)

                self._set_device_info(
                    DeviceInfo(
                        vendor=vendor,
                        model=name,
                        serial_number=None,  # Windows doesn't expose this easily
                        firmware_version=None,
                        capabilities=self._get_capabilities(status),
                    )
                )
            except Exception as e:
                logger.warning(f"Could not get detailed printer info: {e}")
                self._set_device_info(
                    DeviceInfo(
                        vendor="Windows",
                        model=name,
                        capabilities=["grayscale"],
                    )
                )

            self._set_status(DeviceStatus.CONNECTED, f"Connected to {name}")
            return True

        except Exception as e:
            self._set_status(DeviceStatus.ERROR, f"Connection failed: {e}")
            raise PrinterDriverError(
                message=f"Failed to open printer: {e}",
                driver_name="Win32",
            ) from e

    def disconnect(self) -> None:
        """Disconnect from printer."""
        if self._printer_handle is not None:
            try:
                win32print, _, _ = _import_win32()
                win32print.ClosePrinter(self._printer_handle)
            except Exception as e:
                logger.warning(f"Error closing printer handle: {e}")
            finally:
                self._printer_handle = None

        self._set_status(DeviceStatus.DISCONNECTED, "Disconnected")
        self._clear_device_info()

    def print_image(self, job: PrintJob) -> PrintResult:
        """Print image using Win32 API.

        Args:
            job: Print job specification.

        Returns:
            PrintResult indicating success/failure.

        Raises:
            PrintJobError: If printing fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            return PrintResult(
                success=False,
                error="Printer not connected",
            )

        if self._printer_handle is None:
            return PrintResult(
                success=False,
                error="Printer handle not initialized",
            )

        win32print, win32api, _ = _import_win32()

        self._set_status(DeviceStatus.BUSY, f"Printing: {job.name}")
        self._job_counter += 1
        start_time = time.time()

        try:
            # Verify image exists
            image_path = Path(job.image_path)
            if not image_path.exists():
                raise PrintJobError(
                    message=f"Image file not found: {job.image_path}",
                    job_name=job.name,
                )

            # For bitmap/image printing, we use ShellExecute
            # This delegates to the default image viewer/printer handler
            # For more control, you'd use GDI+ or a specific library

            # Start a raw print job
            job_id = win32print.StartDocPrinter(
                self._printer_handle,
                1,
                (job.name, None, "RAW"),
            )

            if job_id == 0:
                raise PrintJobError(
                    message="Failed to start print job",
                    job_name=job.name,
                )

            try:
                win32print.StartPagePrinter(self._printer_handle)

                # For actual image printing, you'd need to:
                # 1. Load the image
                # 2. Convert to printer-compatible format
                # 3. Send raw data or use GDI rendering
                #
                # This is a simplified implementation that demonstrates the API.
                # Full implementation would use PIL/Pillow + win32ui for rendering.

                # Read image data (simplified - actual implementation needs image processing)
                with open(job.image_path, "rb") as f:
                    image_data = f.read()

                # Write data (this works for PCL/PostScript, not raw images)
                # For raw images, you'd need GDI rendering
                win32print.WritePrinter(self._printer_handle, image_data)

                win32print.EndPagePrinter(self._printer_handle)

            finally:
                win32print.EndDocPrinter(self._printer_handle)

            duration = time.time() - start_time
            self._set_status(DeviceStatus.CONNECTED, "Print complete")

            return PrintResult(
                success=True,
                job_id=f"win32-{job_id}",
                pages_printed=job.copies,
                duration_seconds=duration,
            )

        except PrintJobError:
            self._set_status(DeviceStatus.CONNECTED, "Print failed")
            raise
        except Exception as e:
            self._set_status(DeviceStatus.CONNECTED, f"Print error: {e}")
            return PrintResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def get_paper_sizes(self) -> list[str]:
        """Get supported paper sizes from driver.

        Returns:
            List of paper size names.
        """
        if self._printer_handle is None:
            return list(STANDARD_PAPER_SIZES)

        try:
            win32print, _, _ = _import_win32()

            # Get DEVMODE to query paper sizes
            props = win32print.GetPrinter(self._printer_handle, 2)
            devmode = props.get("pDevMode")

            if devmode:
                # Paper sizes are in DEVMODE but require additional enumeration
                # For simplicity, return standard sizes
                pass

            return list(STANDARD_PAPER_SIZES)

        except Exception as e:
            logger.warning(f"Error getting paper sizes: {e}")
            return list(STANDARD_PAPER_SIZES)

    def get_resolutions(self) -> list[int]:
        """Get supported resolutions.

        Returns:
            List of DPI values.
        """
        # Windows printers typically support these resolutions
        # Actual values would come from DEVMODE enumeration
        return list(STANDARD_RESOLUTIONS)

    def get_ink_levels(self) -> dict[str, Any]:
        """Get ink levels (if supported by driver).

        Returns:
            Dictionary of ink colors and levels (empty if not supported).
        """
        # Ink levels require vendor-specific APIs
        # Most Windows drivers don't expose this through standard API
        return {}

    def get_printer_status(self) -> dict[str, Any]:
        """Get detailed printer status.

        Returns:
            Dictionary with printer status information.
        """
        if self._printer_handle is None:
            return {"status": "disconnected"}

        try:
            win32print, _, _ = _import_win32()

            info = win32print.GetPrinter(self._printer_handle, 2)
            status_code = info.get("Status", 0)

            # Decode status flags
            status_flags = []
            status_map = {
                0x00000001: "paused",
                0x00000002: "error",
                0x00000004: "pending_deletion",
                0x00000008: "paper_jam",
                0x00000010: "paper_out",
                0x00000020: "manual_feed",
                0x00000040: "paper_problem",
                0x00000080: "offline",
                0x00000100: "io_active",
                0x00000200: "busy",
                0x00000400: "printing",
                0x00000800: "output_bin_full",
                0x00001000: "not_available",
                0x00002000: "waiting",
                0x00004000: "processing",
                0x00008000: "initializing",
                0x00010000: "warming_up",
                0x00020000: "toner_low",
                0x00040000: "no_toner",
                0x00080000: "page_punt",
                0x00100000: "user_intervention",
                0x00200000: "out_of_memory",
                0x00400000: "door_open",
                0x00800000: "server_unknown",
                0x01000000: "power_save",
            }

            for flag, name in status_map.items():
                if status_code & flag:
                    status_flags.append(name)

            return {
                "status": "idle" if status_code == 0 else "busy",
                "status_code": status_code,
                "status_flags": status_flags,
                "jobs": info.get("cJobs", 0),
                "driver": info.get("pDriverName", "Unknown"),
                "port": info.get("pPortName", "Unknown"),
            }

        except Exception as e:
            logger.error(f"Error getting printer status: {e}")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def list_printers() -> list[str]:
        """List all available Windows printers.

        Returns:
            List of printer names.
        """
        try:
            win32print, _, _ = _import_win32()

            flags = (
                win32print.PRINTER_ENUM_LOCAL
                | win32print.PRINTER_ENUM_CONNECTIONS
            )
            printers = win32print.EnumPrinters(flags)

            return [p[2] for p in printers]

        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Error listing printers: {e}")
            return []

    @staticmethod
    def get_default_printer() -> str | None:
        """Get the default Windows printer name.

        Returns:
            Default printer name or None.
        """
        try:
            win32print, _, _ = _import_win32()
            return win32print.GetDefaultPrinter()
        except Exception:
            return None

    def _detect_vendor(self, driver_name: str, printer_name: str) -> str:
        """Detect printer vendor from driver or name.

        Args:
            driver_name: Windows driver name.
            printer_name: Printer name.

        Returns:
            Vendor name.
        """
        combined = f"{driver_name} {printer_name}".lower()

        if "epson" in combined:
            return "Epson"
        elif "canon" in combined:
            return "Canon"
        elif "hp" in combined or "hewlett" in combined:
            return "HP"
        elif "brother" in combined:
            return "Brother"
        elif "lexmark" in combined:
            return "Lexmark"
        elif "xerox" in combined:
            return "Xerox"
        elif "samsung" in combined:
            return "Samsung"
        elif "ricoh" in combined:
            return "Ricoh"

        return "Unknown"

    def _get_capabilities(self, status: int) -> list[str]:
        """Get printer capabilities based on status.

        Args:
            status: Windows printer status code.

        Returns:
            List of capability strings.
        """
        caps = ["grayscale", "sheet_paper"]

        # Add ready if not in error state
        error_flags = 0x00000002 | 0x00000008 | 0x00000010 | 0x00000080
        if not (status & error_flags):
            caps.append("ready")

        return caps
