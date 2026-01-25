"""
X-Rite i1 Pro spectrophotometer driver for PTPD Calibration.

Provides real hardware communication with X-Rite i1 Pro, i1Pro2, and i1Pro3
spectrophotometers via USB serial interface.

Requires:
    - pyserial: pip install pyserial

Usage:
    from ptpd_calibration.integrations.hardware import XRiteI1ProDriver

    driver = XRiteI1ProDriver()
    if driver.connect():
        driver.calibrate_white()
        measurement = driver.read_density()
        print(f"Density: {measurement.density}")
        driver.disconnect()
"""

import re
from datetime import datetime, timezone

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import get_logger
from ptpd_calibration.integrations.hardware.exceptions import (
    CalibrationError,
    DeviceCommunicationError,
    DeviceConnectionError,
    DeviceNotFoundError,
    MeasurementError,
)
from ptpd_calibration.integrations.protocols import (
    DensityMeasurement,
    DeviceInfo,
    DeviceStatus,
    SpectralData,
)

logger = get_logger(__name__)

# X-Rite USB Vendor and Product IDs
XRITE_VENDOR_ID = 0x0765
XRITE_PRODUCT_IDS = {
    0x5001: "i1Pro",
    0x5010: "i1Pro2",
    0x5020: "i1Pro3",
}


def _import_serial():
    """Lazy import for pyserial."""
    try:
        import serial
        import serial.tools.list_ports
        return serial
    except ImportError as e:
        raise ImportError(
            "pyserial is required for X-Rite hardware. "
            "Install with: pip install pyserial"
        ) from e


class XRiteI1ProDriver:
    """Driver for X-Rite i1 Pro series spectrophotometers.

    Communicates with the device via USB serial protocol. Supports
    density measurements, Lab color, and full spectral data.

    Attributes:
        status: Current device status.
        device_info: Device information (None if not connected).
    """

    # Serial communication parameters
    DEFAULT_BAUD_RATE = 9600
    DEFAULT_TIMEOUT = 5.0
    COMMAND_TERMINATOR = "\r\n"
    RESPONSE_TERMINATOR = b"\r\n"

    def __init__(
        self,
        port: str | None = None,
        baud_rate: int | None = None,
        timeout: float | None = None,
    ):
        """Initialize X-Rite driver.

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3'). Auto-detect if None.
            baud_rate: Serial baud rate. Uses config default if None.
            timeout: Communication timeout in seconds. Uses config default if None.
        """
        settings = get_settings()
        integrations = settings.integrations

        self._port = port or integrations.spectrophotometer_port
        self._baud_rate = baud_rate or integrations.spectro_baud_rate
        self._timeout = timeout or integrations.spectro_timeout_seconds

        self._serial = None
        self._status = DeviceStatus.DISCONNECTED
        self._device_info: DeviceInfo | None = None
        self._calibrated = False

    @property
    def status(self) -> DeviceStatus:
        """Get current device status."""
        return self._status

    @property
    def device_info(self) -> DeviceInfo | None:
        """Get device information (None if not connected)."""
        return self._device_info

    def connect(
        self,
        port: str | None = None,
        timeout: float = 5.0,
    ) -> bool:
        """Connect to the spectrophotometer.

        Args:
            port: Serial port. Uses auto-detection if None.
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful.

        Raises:
            DeviceNotFoundError: If no X-Rite device found.
            DeviceConnectionError: If connection fails.
        """
        serial = _import_serial()
        port = port or self._port

        logger.info(f"Attempting to connect to X-Rite device (port={port})")
        self._status = DeviceStatus.CONNECTING

        # Auto-detect port if not specified
        if port is None:
            port = self._auto_detect_port()
            if port is None:
                self._status = DeviceStatus.DISCONNECTED
                raise DeviceNotFoundError(
                    "No X-Rite spectrophotometer found. "
                    "Ensure device is connected and powered on.",
                    device_type="spectrophotometer",
                )

        # Open serial connection
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self._baud_rate,
                timeout=timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            logger.debug(f"Serial port opened: {port}")
        except serial.SerialException as e:
            self._status = DeviceStatus.DISCONNECTED
            raise DeviceConnectionError(
                f"Failed to open serial port {port}: {e}",
                device_type="spectrophotometer",
                port=port,
            ) from e

        # Verify device communication
        try:
            self._device_info = self._identify_device()
            self._status = DeviceStatus.CONNECTED
            logger.info(f"Connected to {self._device_info}")
            return True
        except Exception as e:
            self._close_serial()
            self._status = DeviceStatus.DISCONNECTED
            raise DeviceConnectionError(
                f"Device identification failed: {e}",
                device_type="spectrophotometer",
                port=port,
            ) from e

    def disconnect(self) -> None:
        """Disconnect from the device."""
        logger.info("Disconnecting from X-Rite device")
        self._close_serial()
        self._status = DeviceStatus.DISCONNECTED
        self._device_info = None
        self._calibrated = False

    def calibrate_white(self) -> bool:
        """Calibrate on white reference tile.

        The device should be placed on the white calibration tile
        before calling this method.

        Returns:
            True if calibration successful.

        Raises:
            CalibrationError: If calibration fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise CalibrationError(
                "Cannot calibrate: not connected",
                device_type="spectrophotometer",
                calibration_type="white",
            )

        logger.info("Starting white reference calibration")
        self._status = DeviceStatus.CALIBRATING

        try:
            response = self._send_command("*CAL:W")
            if "OK" in response or "PASS" in response:
                self._calibrated = True
                self._status = DeviceStatus.CONNECTED
                logger.info("White calibration successful")
                return True
            else:
                self._status = DeviceStatus.CONNECTED
                raise CalibrationError(
                    f"White calibration failed: {response}",
                    device_type="spectrophotometer",
                    calibration_type="white",
                )
        except DeviceCommunicationError:
            self._status = DeviceStatus.CONNECTED
            raise

    def calibrate_black(self) -> bool:
        """Calibrate on black reference (if supported).

        Returns:
            True if calibration successful.

        Raises:
            CalibrationError: If calibration fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise CalibrationError(
                "Cannot calibrate: not connected",
                device_type="spectrophotometer",
                calibration_type="black",
            )

        logger.info("Starting black reference calibration")
        self._status = DeviceStatus.CALIBRATING

        try:
            response = self._send_command("*CAL:B")
            self._status = DeviceStatus.CONNECTED
            if "OK" in response or "PASS" in response:
                logger.info("Black calibration successful")
                return True
            else:
                # Black calibration may not be supported on all models
                logger.warning(f"Black calibration response: {response}")
                return False
        except DeviceCommunicationError:
            self._status = DeviceStatus.CONNECTED
            raise

    def read_density(self) -> DensityMeasurement:
        """Read a density measurement.

        Returns:
            Density measurement with Lab values.

        Raises:
            MeasurementError: If measurement fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise MeasurementError(
                "Cannot measure: not connected",
                device_type="spectrophotometer",
                measurement_type="density",
            )

        if not self._calibrated:
            logger.warning("Device not calibrated - measurements may be inaccurate")

        logger.debug("Reading density measurement")
        self._status = DeviceStatus.MEASURING

        try:
            # Request density and Lab measurement
            response = self._send_command("*MSR:DENSITY,LAB")
            self._status = DeviceStatus.CONNECTED

            # Parse response (format varies by model)
            measurement = self._parse_density_response(response)
            logger.debug(f"Density: {measurement.density:.3f}")
            return measurement

        except DeviceCommunicationError:
            self._status = DeviceStatus.CONNECTED
            raise
        except Exception as e:
            self._status = DeviceStatus.CONNECTED
            raise MeasurementError(
                f"Failed to read density: {e}",
                device_type="spectrophotometer",
                measurement_type="density",
            ) from e

    def read_spectral(self) -> SpectralData:
        """Read full spectral data.

        Returns:
            Spectral reflectance data.

        Raises:
            MeasurementError: If measurement fails.
        """
        if self._status != DeviceStatus.CONNECTED:
            raise MeasurementError(
                "Cannot measure: not connected",
                device_type="spectrophotometer",
                measurement_type="spectral",
            )

        logger.debug("Reading spectral measurement")
        self._status = DeviceStatus.MEASURING

        try:
            response = self._send_command("*MSR:SPECTRAL")
            self._status = DeviceStatus.CONNECTED

            spectral = self._parse_spectral_response(response)
            logger.debug(f"Spectral: {len(spectral.values)} values")
            return spectral

        except DeviceCommunicationError:
            self._status = DeviceStatus.CONNECTED
            raise
        except Exception as e:
            self._status = DeviceStatus.CONNECTED
            raise MeasurementError(
                f"Failed to read spectral: {e}",
                device_type="spectrophotometer",
                measurement_type="spectral",
            ) from e

    def _auto_detect_port(self) -> str | None:
        """Auto-detect X-Rite device serial port.

        Returns:
            Port path if device found, None otherwise.
        """
        serial = _import_serial()
        logger.debug("Auto-detecting X-Rite device...")

        for port_info in serial.tools.list_ports.comports():
            if port_info.vid == XRITE_VENDOR_ID:
                model = XRITE_PRODUCT_IDS.get(port_info.pid, "Unknown")
                logger.info(f"Found X-Rite {model} on {port_info.device}")
                return port_info.device

        logger.debug("No X-Rite device detected")
        return None

    def _identify_device(self) -> DeviceInfo:
        """Query device identification.

        Returns:
            DeviceInfo with vendor, model, serial, firmware.
        """
        response = self._send_command("*VER")

        # Parse version response
        # Expected format: "i1Pro2 V1.23.456 SN:AB123456"
        model = "i1Pro"
        firmware = "Unknown"
        serial_num = None

        if "i1Pro3" in response:
            model = "i1Pro3"
        elif "i1Pro2" in response:
            model = "i1Pro2"

        # Extract version
        ver_match = re.search(r"V?(\d+\.\d+(?:\.\d+)?)", response)
        if ver_match:
            firmware = ver_match.group(1)

        # Extract serial number
        sn_match = re.search(r"SN:?([A-Z0-9]+)", response, re.IGNORECASE)
        if sn_match:
            serial_num = sn_match.group(1)

        return DeviceInfo(
            vendor="X-Rite",
            model=model,
            serial_number=serial_num,
            firmware_version=firmware,
            capabilities=["density", "lab", "spectral", "reflection"],
        )

    def _send_command(self, command: str) -> str:
        """Send command and read response.

        Args:
            command: Command string to send.

        Returns:
            Response string from device.

        Raises:
            DeviceCommunicationError: If communication fails.
        """
        if self._serial is None:
            raise DeviceCommunicationError(
                "Serial port not open",
                device_type="spectrophotometer",
                command=command,
            )

        # Send command
        full_command = command + self.COMMAND_TERMINATOR
        logger.debug(f"Sending: {command}")

        try:
            self._serial.write(full_command.encode("ascii"))
            self._serial.flush()
        except Exception as e:
            raise DeviceCommunicationError(
                f"Failed to send command: {e}",
                device_type="spectrophotometer",
                command=command,
            ) from e

        # Read response
        try:
            response = self._serial.readline()
            if not response:
                raise DeviceCommunicationError(
                    "No response from device (timeout)",
                    device_type="spectrophotometer",
                    command=command,
                )

            response_str = response.decode("ascii").strip()
            logger.debug(f"Response: {response_str}")

            # Check for error response
            if response_str.startswith("ERR") or response_str.startswith("ERROR"):
                raise DeviceCommunicationError(
                    f"Device error: {response_str}",
                    device_type="spectrophotometer",
                    command=command,
                    response=response_str,
                )

            return response_str

        except UnicodeDecodeError as e:
            raise DeviceCommunicationError(
                f"Invalid response encoding: {e}",
                device_type="spectrophotometer",
                command=command,
            ) from e

    def _parse_density_response(self, response: str) -> DensityMeasurement:
        """Parse density measurement response.

        Args:
            response: Raw response string.

        Returns:
            Parsed DensityMeasurement.
        """
        # Expected formats:
        # "D=1.234,L=50.12,a=2.34,b=-1.23"
        # "OK:D=1.234,L=50.12,a=2.34,b=-1.23"

        density = 0.0
        lab_l = 50.0
        lab_a = 0.0
        lab_b = 0.0

        # Remove OK prefix if present
        response = response.replace("OK:", "").strip()

        # Parse key=value pairs
        for part in response.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().upper()
                try:
                    val = float(value.strip())
                    if key == "D":
                        density = val
                    elif key == "L":
                        lab_l = val
                    elif key == "A":
                        lab_a = val
                    elif key == "B":
                        lab_b = val
                except ValueError:
                    pass

        return DensityMeasurement(
            density=density,
            lab_l=lab_l,
            lab_a=lab_a,
            lab_b=lab_b,
            timestamp=datetime.now(timezone.utc),
            measurement_mode="reflection",
        )

    def _parse_spectral_response(self, response: str) -> SpectralData:
        """Parse spectral measurement response.

        Args:
            response: Raw response string.

        Returns:
            Parsed SpectralData.
        """
        # Expected format:
        # "380=0.123,390=0.134,400=0.145,..."

        wavelengths = []
        values = []

        response = response.replace("OK:", "").strip()

        for part in response.split(","):
            if "=" in part:
                wl, val = part.split("=", 1)
                try:
                    wavelengths.append(float(wl.strip()))
                    values.append(float(val.strip()))
                except ValueError:
                    pass

        # Default to standard range if parsing fails
        if not wavelengths:
            wavelengths = list(range(380, 740, 10))
            values = [0.0] * len(wavelengths)

        return SpectralData(
            wavelengths=wavelengths,
            values=values,
            start_nm=wavelengths[0] if wavelengths else 380.0,
            end_nm=wavelengths[-1] if wavelengths else 730.0,
            interval_nm=10.0,
        )

    def _close_serial(self) -> None:
        """Close serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning(f"Error closing serial port: {e}")
            self._serial = None
